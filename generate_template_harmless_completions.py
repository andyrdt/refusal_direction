#!/usr/bin/env python3
"""
Generate completions from template-formatted harmless instructions and evaluate with DeepSeek ASR.
Based on generate_template_completions.py but focused on harmless instructions.
"""

import os
# Suppress transformers warnings about invalid generation flags
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import json
import os
import argparse
import random
from typing import List, Dict, Tuple
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.evaluate_jailbreak import deepseekv31_judge_fn, evaluate_jailbreak
import importlib
import re


def extract_length_from_filename(template_name: str) -> str:
    """
    Extract length suffix from template filename.
    
    Args:
        template_name: Template filename (e.g., 'template_3k', 'template_11k')
        
    Returns:
        Length suffix (e.g., '3k', '11k') or 'unknown'
    """
    match = re.search(r'(\d+k)', template_name)
    return match.group(1) if match else "unknown"


def load_template_module(template_file: str):
    """
    Dynamically load template module and return format_thinking_template.
    
    Args:
        template_file: Template file name without .py extension
        
    Returns:
        format_thinking_template from the specified module
    """
    try:
        module = importlib.import_module(template_file)
        return module.format_thinking_template
    except ImportError as e:
        raise ImportError(f"Failed to import template module '{template_file}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Module '{template_file}' does not have 'format_thinking_template': {e}")


def load_harmless_instructions(model_path: str, cfg: Config, n_samples: int = 0) -> Tuple[object, List[str]]:
    """
    Setup model and load harmless instructions.
    
    Args:
        model_path: Path to the model
        cfg: Configuration object
        n_samples: Number of samples to process (0 for all)
        
    Returns:
        Tuple of (model_base, harmless_instructions)
    """
    print(f"Loading model from {model_path}")
    model_base = construct_model_base(model_path)
    
    # Use same sampling logic as original scripts
    random.seed(42)
    
    # Load harmless instructions from test split (same as run_pipeline.py step 4a)
    print("Loading harmless test dataset...")
    harmless_test = load_dataset_split(harmtype='harmless', split='test', instructions_only=True)
    harmless_instructions = random.sample(harmless_test, min(cfg.n_test, len(harmless_test)))
    
    # Limit samples if specified
    if n_samples > 0:
        harmless_instructions = harmless_instructions[:n_samples]
    
    print(f"Total loaded: {len(harmless_instructions)} harmless instructions")
    
    return model_base, harmless_instructions


def format_instructions_with_template(instructions: List[str], format_thinking_template: str) -> List[str]:
    """
    Format instructions using the specified format_thinking_template.
    
    Args:
        instructions: List of raw instructions
        format_thinking_template: Template string to use for formatting
        
    Returns:
        List of formatted instructions
    """
    print("Formatting instructions with thinking template...")
    formatted_instructions = []
    
    for instruction in tqdm(instructions, desc="Formatting"):
        formatted = format_thinking_template.format(instruction=instruction)
        formatted_instructions.append(formatted)
    
    return formatted_instructions


def generate_completions(model_base, formatted_instructions: List[str], 
                        max_new_tokens: int = 512, batch_size: int = 4) -> List[str]:
    """
    Generate completions from formatted instructions.
    
    Args:
        model_base: Model instance
        formatted_instructions: List of template-formatted instructions
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for generation
        
    Returns:
        List of generated completions
    """
    print(f"Generating completions with max_new_tokens={max_new_tokens}...")
    completions = []
    
    for i in tqdm(range(0, len(formatted_instructions), batch_size), desc="Generating"):
        batch_instructions = formatted_instructions[i:i+batch_size]
        
        # Tokenize the formatted instructions
        tokenized = model_base.tokenizer(
            batch_instructions,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Template already includes special tokens
        )
        
        input_ids = tokenized.input_ids.to(model_base.model.device)
        attention_mask = tokenized.attention_mask.to(model_base.model.device)
        
        # Generate completions
        with torch.no_grad():
            generation_outputs = model_base.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=model_base.tokenizer.pad_token_id,
                eos_token_id=model_base.tokenizer.eos_token_id,
            )
        
        # Extract only the newly generated tokens
        new_tokens = generation_outputs[:, input_ids.shape[-1]:]
        
        # Decode completions
        for j, new_token_ids in enumerate(new_tokens):
            completion = model_base.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            completions.append(completion)
    
    return completions


def create_completion_dicts(original_instructions: List[str], 
                           formatted_instructions: List[str], 
                           completions: List[str]) -> List[Dict]:
    """
    Create completion dictionaries in the format expected by evaluate_jailbreak.
    
    Args:
        original_instructions: Original harmless instructions  
        formatted_instructions: Template-formatted instructions
        completions: Generated completions
        
    Returns:
        List of completion dictionaries
    """
    completion_dicts = []
    
    for i, (original, formatted, completion) in enumerate(zip(
        original_instructions, formatted_instructions, completions
    )):
        completion_dict = {
            "instruction": original,
            "formatted_instruction": formatted,
            "completion": completion,
            "category": "template_generated_harmless",  # Mark as harmless
        }
        completion_dicts.append(completion_dict)
    
    return completion_dicts


def evaluate_with_deepseek(completion_dicts: List[Dict]) -> Dict:
    """
    Evaluate completions using DeepSeek ASR methodology.
    Note: For harmless instructions, we expect low ASR (most should be refused).
    
    Args:
        completion_dicts: List of completion dictionaries
        
    Returns:
        Evaluation results dictionary
    """
    print("Evaluating completions with DeepSeek ASR...")
    print("Note: For harmless instructions, we expect low ASR (most should be normal responses)")
    
    # Check if DeepSeek API key is set
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY not set. DeepSeek evaluation will be skipped.")
        return {"deepseek_asr": 0.0, "error": "DEEPSEEK_API_KEY not set"}
    
    try:
        # Prepare data for evaluation
        prompts = [comp["instruction"] for comp in completion_dicts]
        responses = [comp["completion"] for comp in completion_dicts]
        
        # Add category information for proper evaluation format
        for comp in completion_dicts:
            comp["prompt"] = comp["instruction"]
            comp["response"] = comp["completion"]
        
        # Use evaluate_jailbreak with deepseek methodology
        evaluation = evaluate_jailbreak(
            completions=completion_dicts,
            methodologies=["deepseekv31"]
        )
        
        return evaluation
        
    except Exception as e:
        print(f"Error during DeepSeek evaluation: {e}")
        return {"deepseek_asr": 0.0, "error": str(e)}


def save_results(completion_dicts: List[Dict], evaluation: Dict, output_dir: str, 
                 template_file: str = "template_3k", length_suffix: str = "unknown"):
    """
    Save completion results and evaluation to files with length suffix.
    
    Args:
        completion_dicts: List of completion dictionaries
        evaluation: Evaluation results
        output_dir: Output directory path
        template_file: Template file used
        length_suffix: Template length suffix (e.g., '3k', '11k')
    """
    print(f"Saving results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save completions with evaluation results
    results = {
        "completions": completion_dicts,
        "evaluation": evaluation,
        "metadata": {
            "n_samples": len(completion_dicts),
            "template_used": f"format_thinking_template from {template_file}",
            "template_file": template_file,
            "template_length": length_suffix,
            "evaluation_methodology": "deepseekv31",
            "dataset_type": "harmless"
        }
    }
    
    # Save complete results as JSON with length suffix
    json_path = os.path.join(output_dir, f'template_harmless_completions_with_asr_{length_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")
    
    # Save evaluation summary with length suffix
    summary_path = os.path.join(output_dir, f'harmless_asr_summary_{length_suffix}.json')
    summary = {
        "deepseek_asr": evaluation.get("deepseekv31_success_rate", 0.0),
        "total_samples": len(completion_dicts),
        "successful_jailbreaks": sum(1 for comp in completion_dicts 
                                   if comp.get("is_jailbreak_deepseekv31", 0) == 1),
        "normal_responses": sum(1 for comp in completion_dicts 
                              if comp.get("is_jailbreak_deepseekv31", 0) == 0),
        "metadata": results["metadata"]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved harmless ASR summary to {summary_path}")


def main():
    """Main function to execute the complete pipeline."""
    parser = argparse.ArgumentParser(description="Generate template completions for harmless instructions and evaluate ASR with DeepSeek")
    parser.add_argument('--model_path', type=str, 
                       default='Qwen/Qwen3-14B', 
                       help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='./template_generation_results', 
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=1000,
                       help='Maximum new tokens to generate')
    parser.add_argument('--n_samples', type=int, default=10, 
                       help='Number of samples (0 for all)')
    parser.add_argument('--template_file', type=str, default='template_3k', 
                       help='Template file to use (without .py extension, e.g., template_3k, template_11k)')
    
    args = parser.parse_args()
    
    print("Starting template-based harmless completion generation and ASR evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Template file: {args.template_file}")
    
    # Load template module
    try:
        format_thinking_template = load_template_module(args.template_file)
        length_suffix = extract_length_from_filename(args.template_file)
        print(f"Loaded template with length: {length_suffix}")
    except Exception as e:
        print(f"Error loading template: {e}")
        return
    
    # Create config object
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    
    # Load model and harmless instructions
    model_base, harmless_instructions = load_harmless_instructions(
        args.model_path, cfg, args.n_samples
    )
    
    print(f"Processing {len(harmless_instructions)} harmless instructions")
    
    # Format instructions with template
    formatted_instructions = format_instructions_with_template(harmless_instructions, format_thinking_template)
    
    # Generate completions
    print("\n=== Generating Completions for Harmless Instructions ===")
    completions = generate_completions(
        model_base, formatted_instructions, 
        args.max_new_tokens, args.batch_size
    )
    
    # Create completion dictionaries
    completion_dicts = create_completion_dicts(
        harmless_instructions, formatted_instructions, completions
    )
    
    # Evaluate with DeepSeek ASR
    print("\n=== Evaluating with DeepSeek ASR ===")
    evaluation = evaluate_with_deepseek(completion_dicts)
    
    # Print results
    if "deepseekv31_success_rate" in evaluation:
        asr = evaluation["deepseekv31_success_rate"]
        print(f"\nDeepSeek ASR for Harmless Instructions: {asr:.3f} ({asr*100:.1f}%)")
        print("(Note: Low ASR is expected for harmless instructions)")
    else:
        print(f"\nEvaluation failed: {evaluation.get('error', 'Unknown error')}")
    
    # Save results
    save_results(completion_dicts, evaluation, args.output_dir, args.template_file, length_suffix)
    
    print("\n=== Generation and Evaluation Complete ===")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()