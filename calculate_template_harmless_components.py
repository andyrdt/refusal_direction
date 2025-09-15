#!/usr/bin/env python3
"""
Calculate parallel components of activations with respect to refusal direction
using template-formatted instructions. This script processes only harmless instructions
and uses format_thinking_template from template.py to format them.
"""

import os
# Suppress transformers warnings about invalid generation flags
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import json
import os
import argparse
import random
import re
import importlib
from typing import List, Dict, Tuple
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks


def load_refusal_direction(direction_path: str, metadata_path: str) -> Tuple[torch.Tensor, Dict]:
    """
    Load refusal direction and metadata.
    
    Args:
        direction_path: Path to the direction.pt file
        metadata_path: Path to the direction_metadata.json file
        
    Returns:
        Tuple of (direction tensor, metadata dict)
    """
    print(f"Loading refusal direction from {direction_path}")
    direction = torch.load(direction_path, map_location='cpu')
    
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    print(f"Direction shape: {direction.shape}")
    print(f"Metadata: {metadata}")
    
    return direction, metadata


def setup_model_and_harmless_data(model_path: str, cfg: Config) -> Tuple[object, List[str]]:
    """
    Setup model and load only harmless datasets.
    
    Args:
        model_path: Path to the model
        cfg: Configuration object
        
    Returns:
        Tuple of (model_base, harmless_instructions)
    """
    print(f"Loading model from {model_path}")
    model_base = construct_model_base(model_path)
    
    # Use same sampling logic as run_pipeline.py
    random.seed(42)
    
    # Load harmless instructions from test split (same as run_pipeline.py step 4a)
    print("Loading harmless test dataset...")
    harmless_test = load_dataset_split(harmtype='harmless', split='test', instructions_only=True)
    harmless_instructions = random.sample(harmless_test, min(cfg.n_test, len(harmless_test)))
    
    print(f"Total loaded: {len(harmless_instructions)} harmless instructions")
    
    return model_base, harmless_instructions


def extract_length_from_filename(template_name: str) -> str:
    """
    Extract length suffix from template filename.
    
    Args:
        template_name: Template module name (e.g., 'template_3k')
        
    Returns:
        Length suffix (e.g., '3k')
    """
    match = re.search(r'(\d+k)', template_name)
    return match.group(1) if match else "unknown"


def load_template_module(template_file: str):
    """
    Dynamically import template module and get format_thinking_template function.
    
    Args:
        template_file: Template module name (e.g., 'template_3k')
        
    Returns:
        format_thinking_template function from the module
    """
    try:
        module = importlib.import_module(template_file)
        return module.format_thinking_template
    except ImportError as e:
        print(f"Error importing {template_file}: {e}")
        print("Available template files should be: template_3k, template_11k, template_21k, template_31k, template_47k")
        raise
    except AttributeError as e:
        print(f"Error: {template_file} does not have format_thinking_template function: {e}")
        raise


def format_instructions_with_template(instructions: List[str], template_file: str = "template_3k") -> List[str]:
    """
    Format instructions using the specified template.
    
    Args:
        instructions: List of raw instructions
        template_file: Template module name
        
    Returns:
        List of formatted instructions
    """
    print(f"Formatting instructions with template: {template_file}")
    format_thinking_template = load_template_module(template_file)
    formatted_instructions = []
    
    for instruction in tqdm(instructions, desc="Formatting"):
        formatted = format_thinking_template.format(instruction=instruction)
        formatted_instructions.append(formatted)
    
    return formatted_instructions


def get_parallel_component_hook(direction: torch.Tensor, results_cache: Dict, sample_idx: int, layer_idx: int):
    """
    Create a hook to collect parallel components.
    
    Args:
        direction: Refusal direction tensor
        results_cache: Dictionary to store results
        sample_idx: Index of current sample
        layer_idx: Index of current layer
        
    Returns:
        Hook function
    """
    def hook_fn(module, input_data):
        if isinstance(input_data, tuple):
            activation = input_data[0]
        else:
            activation = input_data
            
        # activation shape: [batch, seq, d_model]
        # Take the last token position
        last_token_activation = activation[:, -1, :]  # [batch, d_model]
        
        # Normalize direction and convert to activation dtype and device
        direction_norm = direction / (direction.norm() + 1e-8)
        direction_norm = direction_norm.to(device=activation.device, dtype=activation.dtype)
        
        # Calculate parallel component (keep sign for positive/negative analysis)
        parallel_component = last_token_activation @ direction_norm  # [batch]
        
        # Store results (move to CPU to save GPU memory)
        if sample_idx not in results_cache:
            results_cache[sample_idx] = {}
        results_cache[sample_idx][layer_idx] = parallel_component.cpu().numpy()
        
    return hook_fn


def collect_activations_with_hooks(model_base, formatted_instructions: List[str], 
                                   direction: torch.Tensor, batch_size: int = 8) -> Dict:
    """
    Single forward pass to collect activations and calculate parallel components.
    
    Args:
        model_base: Model instance
        formatted_instructions: List of template-formatted instructions
        direction: Refusal direction tensor
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with hooks...")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers
    
    for i in tqdm(range(0, len(formatted_instructions), batch_size), desc="Processing batches"):
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
        
        # Create hooks for all layers
        fwd_pre_hooks = []
        for layer_idx in range(n_layers):
            for batch_idx in range(len(batch_instructions)):
                global_sample_idx = i + batch_idx
                hook = get_parallel_component_hook(direction, results_cache, global_sample_idx, layer_idx)
                fwd_pre_hooks.append((model_base.model_block_modules[layer_idx], hook))
        
        # Forward pass with hooks
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Clear GPU cache after each batch to prevent memory explosion
        torch.cuda.empty_cache()
    
    return results_cache


def organize_results(original_instructions: List[str], formatted_instructions: List[str], 
                     component_values: Dict, n_layers: int, direction_metadata: Dict, 
                     template_file: str = "template_3k") -> Dict:
    """
    Organize results into structured format.
    
    Args:
        original_instructions: Original harmless instructions
        formatted_instructions: Template-formatted instructions
        component_values: Parallel component values
        n_layers: Number of layers in the model
        direction_metadata: Metadata about the refusal direction
        
    Returns:
        Structured results dictionary
    """
    print("Organizing results...")
    
    # Convert components to tensor
    def components_to_tensor(components_dict, n_samples, n_layers):
        tensor = torch.zeros(n_samples, n_layers)
        for sample_idx in range(n_samples):
            if sample_idx in components_dict:
                for layer_idx in range(n_layers):
                    if layer_idx in components_dict[sample_idx]:
                        # Take first element if batch dimension exists
                        values = components_dict[sample_idx][layer_idx]
                        if isinstance(values, (list, tuple)) or (hasattr(values, 'shape') and len(values.shape) > 0):
                            val = values[0] if hasattr(values, '__len__') and len(values) > 0 else values
                            tensor[sample_idx, layer_idx] = float(val)
                        else:
                            tensor[sample_idx, layer_idx] = float(values)
        return tensor
    
    component_tensor = components_to_tensor(component_values, len(original_instructions), n_layers)
    
    results = {
        "harmless": {
            "original_instructions": original_instructions,
            "formatted_instructions": formatted_instructions,
            "parallel_components": component_tensor.tolist(),
        },
        "metadata": {
            "n_layers": n_layers,
            "n_samples": len(original_instructions),
            "template_used": template_file,
            "template_length": extract_length_from_filename(template_file),
            "direction_metadata": direction_metadata
        }
    }
    
    return results


def save_results(results: Dict, output_dir: str, template_file: str = "template_3k"):
    """
    Save results to JSON and PyTorch tensor formats.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
    """
    print(f"Saving results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract length suffix for filename
    length_suffix = extract_length_from_filename(template_file)
    
    # Save complete results as JSON
    json_path = os.path.join(output_dir, f'template_harmless_components_{length_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Save tensors as PyTorch format for easier loading
    tensor_data = {
        'parallel_components': torch.tensor(results['harmless']['parallel_components']),
        'metadata': results['metadata']
    }
    pt_path = os.path.join(output_dir, f'template_harmless_components_{length_suffix}.pt')
    torch.save(tensor_data, pt_path)
    print(f"Saved PyTorch tensors to {pt_path}")


def main():
    """Main function to execute the complete pipeline."""
    parser = argparse.ArgumentParser(description="Calculate refusal direction parallel components using template for harmless instructions")
    parser.add_argument('--model_path', type=str, 
                       default='Qwen/Qwen3-14B', 
                       help='Path to the model')
    parser.add_argument('--direction_path', type=str, 
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/pipeline/runs/Qwen3-14B/direction.pt',
                       help='Path to direction.pt')
    parser.add_argument('--metadata_path', type=str, 
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/pipeline/runs/Qwen3-14B/direction_metadata.json',
                       help='Path to direction_metadata.json')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of samples (0 for all)')
    parser.add_argument('--template_file', type=str, default='template_1k', 
                       help='Template file to use (template_1k， template_3k, template_11k, template_21k, template_31k, template_47k)')
    
    args = parser.parse_args()
    
    print("Starting template-based harmless refusal component analysis...")
    print(f"Model path: {args.model_path}")
    print(f"Direction path: {args.direction_path}")
    print(f"Template file: {args.template_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Create config object to match run_pipeline.py behavior
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    
    # Load refusal direction and metadata
    direction, metadata = load_refusal_direction(args.direction_path, args.metadata_path)
    
    # Setup model and data (only harmless)
    model_base, harmless_instructions = setup_model_and_harmless_data(args.model_path, cfg)
    
    # Limit samples if specified
    if args.n_samples > 0:
        harmless_instructions = harmless_instructions[:args.n_samples]
    
    print(f"Processing {len(harmless_instructions)} harmless samples")
    
    # Format instructions with template
    formatted_instructions = format_instructions_with_template(harmless_instructions, args.template_file)
    
    # Collect activations and calculate parallel components
    print("\n=== Collecting Activations for Template-Formatted Harmless Instructions ===")
    component_values = collect_activations_with_hooks(model_base, formatted_instructions, direction, args.batch_size)
    
    # Clear GPU cache after processing
    torch.cuda.empty_cache()
    
    # Organize and save results
    n_layers = model_base.model.config.num_hidden_layers
    results = organize_results(harmless_instructions, formatted_instructions, 
                              component_values, n_layers, metadata, args.template_file)
    
    save_results(results, args.output_dir, args.template_file)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {args.output_dir}")
    print("Template-based harmless refusal component calculation finished")


if __name__ == "__main__":
    main()