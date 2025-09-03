#!/usr/bin/env python3
"""
Calculate parallel components of activations with respect to refusal direction.
This script implements the analysis described in refusal_component_analysis_plan.md.
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


def setup_model_and_data(model_path: str, cfg: Config) -> Tuple[object, List[str], List[str]]:
    """
    Setup model and load datasets.
    
    Args:
        model_path: Path to the model
        cfg: Configuration object
        
    Returns:
        Tuple of (model_base, harmful_instructions, harmless_instructions)
    """
    print(f"Loading model from {model_path}")
    model_base = construct_model_base(model_path)
    
    # Use same sampling logic as run_pipeline.py
    random.seed(42)
    
    # Load harmful instructions from evaluation datasets (same as run_pipeline.py step 3a)
    print(f"Loading harmful instructions from evaluation datasets: {cfg.evaluation_datasets}")
    harmful_instructions = []
    for dataset_name in cfg.evaluation_datasets:
        dataset = load_dataset(dataset_name)
        instructions = [d['instruction'] for d in dataset]
        harmful_instructions.extend(instructions)
        print(f"  - {dataset_name}: {len(instructions)} instructions")
    
    # Load harmless instructions from test split (same as run_pipeline.py step 4a)
    print("Loading harmless test dataset...")
    harmless_test = load_dataset_split(harmtype='harmless', split='test', instructions_only=True)
    harmless_instructions = random.sample(harmless_test, min(cfg.n_test, len(harmless_test)))
    
    print(f"Total loaded: {len(harmful_instructions)} harmful and {len(harmless_instructions)} harmless instructions")
    
    return model_base, harmful_instructions, harmless_instructions


def generate_first_token(model_base, instructions: List[str], batch_size: int = 8) -> List[Dict]:
    """
    First forward pass to generate new tokens (expected to be <think>).
    
    Args:
        model_base: Model instance
        instructions: List of instructions
        batch_size: Batch size for processing
        
    Returns:
        List of dicts with instruction, generated_token, token_id
    """
    print("Generating first tokens using thinking mode...")
    results = []
    
    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating tokens"):
        batch_instructions = instructions[i:i+batch_size]
        
        # Use thinking mode tokenization (first generation)
        tokenized = model_base.tokenize_instructions_thinking_fn(instructions=batch_instructions)
        
        # Generate one token
        with torch.no_grad():
            generation_toks = model_base.model.generate(
                input_ids=tokenized.input_ids.to(model_base.model.device),
                attention_mask=tokenized.attention_mask.to(model_base.model.device),
                max_new_tokens=1,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=model_base.tokenizer.pad_token_id,
            )
        
        # Extract generated tokens
        new_tokens = generation_toks[:, tokenized.input_ids.shape[-1]:]
        
        for j, (instruction, new_token_ids) in enumerate(zip(batch_instructions, new_tokens)):
            token_id = new_token_ids[0].item()
            token_str = model_base.tokenizer.decode([token_id], skip_special_tokens=False)
            
            results.append({
                'instruction': instruction,
                'generated_token': token_str,
                'token_id': token_id,
                'input_ids': tokenized.input_ids[j].tolist(),
                'new_token_position': len(tokenized.input_ids[j])
            })
    
    # Verify that generated tokens are as expected
    generated_tokens = [r['generated_token'] for r in results]
    unique_tokens = list(set(generated_tokens))
    print(f"Generated tokens distribution: {dict(zip(unique_tokens, [generated_tokens.count(t) for t in unique_tokens]))}")
    
    return results


def get_parallel_component_hook(direction: torch.Tensor, results_cache: Dict, batch_start_idx: int, layer_idx: int):
    """
    Create a hook to collect parallel components.
    
    Args:
        direction: Refusal direction tensor
        results_cache: Dictionary to store results
        batch_start_idx: Starting index of the current batch
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
        # Take the last token position (newly generated token)
        last_token_activation = activation[:, -1, :]  # [batch, d_model]
        
        # Normalize direction and convert to activation dtype and device
        direction_norm = direction / (direction.norm() + 1e-8)
        direction_norm = direction_norm.to(device=activation.device, dtype=activation.dtype)
        
        # Calculate parallel component (keep sign for positive/negative analysis)
        parallel_component = last_token_activation @ direction_norm  # [batch]
        
        # Store results for each sample in the batch (move to CPU to save GPU memory)
        parallel_component_cpu = parallel_component.cpu().numpy()
        for batch_idx in range(len(parallel_component_cpu)):
            global_sample_idx = batch_start_idx + batch_idx
            if global_sample_idx not in results_cache:
                results_cache[global_sample_idx] = {}
            results_cache[global_sample_idx][layer_idx] = parallel_component_cpu[batch_idx]
        
    return hook_fn


def collect_activations_with_hooks(model_base, generation_results: List[Dict], direction: torch.Tensor, batch_size: int = 8) -> Dict:
    """
    Second forward pass to collect activations and calculate parallel components.
    
    Args:
        model_base: Model instance
        generation_results: Results from first generation
        direction: Refusal direction tensor
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with hooks...")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers
    
    for i in tqdm(range(0, len(generation_results), batch_size), desc="Processing batches"):
        batch_results = generation_results[i:i+batch_size]
        
        # Prepare input sequences with generated tokens
        batch_input_ids = []
        batch_attention_masks = []
        
        for result in batch_results:
            # Reconstruct full input sequence with generated token
            full_input_ids = result['input_ids'] + [result['token_id']]
            batch_input_ids.append(full_input_ids)
        
        # Pad sequences
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids in batch_input_ids:
            pad_length = max_len - len(ids)
            padded_ids = [model_base.tokenizer.pad_token_id] * pad_length + ids
            attention_mask = [0] * pad_length + [1] * len(ids)
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(attention_mask)
        
        input_ids = torch.tensor(padded_input_ids).to(model_base.model.device)
        attention_mask = torch.tensor(padded_attention_masks).to(model_base.model.device)
        
        # Create hooks for all layers (one hook per layer, not per sample)
        fwd_pre_hooks = []
        for layer_idx in range(n_layers):
            hook = get_parallel_component_hook(direction, results_cache, i, layer_idx)
            fwd_pre_hooks.append((model_base.model_block_modules[layer_idx], hook))
        
        # Forward pass with hooks
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
    
    return results_cache


def organize_results(harmful_gen_results: List[Dict], harmless_gen_results: List[Dict], 
                    harmful_components: Dict, harmless_components: Dict, n_layers: int) -> Dict:
    """
    Organize results into structured format.
    
    Args:
        harmful_gen_results: Generation results for harmful instructions
        harmless_gen_results: Generation results for harmless instructions
        harmful_components: Parallel components for harmful instructions
        harmless_components: Parallel components for harmless instructions
        n_layers: Number of layers in the model
        
    Returns:
        Structured results dictionary
    """
    print("Organizing results...")
    
    # Convert components to tensors
    def components_to_tensor(components_dict, n_samples, n_layers):
        tensor = torch.zeros(n_samples, n_layers)
        for sample_idx in range(n_samples):
            if sample_idx in components_dict:
                for layer_idx in range(n_layers):
                    if layer_idx in components_dict[sample_idx]:
                        # Now values should be scalar (fixed the hook issue)
                        values = components_dict[sample_idx][layer_idx]
                        tensor[sample_idx, layer_idx] = float(values)
        return tensor
    
    harmful_tensor = components_to_tensor(harmful_components, len(harmful_gen_results), n_layers)
    harmless_tensor = components_to_tensor(harmless_components, len(harmless_gen_results), n_layers)
    
    results = {
        "harmful": {
            "instructions": [r['instruction'] for r in harmful_gen_results],
            "generated_tokens": [r['generated_token'] for r in harmful_gen_results],
            "token_ids": [r['token_id'] for r in harmful_gen_results],
            "parallel_components": harmful_tensor.tolist(),
        },
        "harmless": {
            "instructions": [r['instruction'] for r in harmless_gen_results],
            "generated_tokens": [r['generated_token'] for r in harmless_gen_results],
            "token_ids": [r['token_id'] for r in harmless_gen_results],
            "parallel_components": harmless_tensor.tolist(),
        },
        "metadata": {
            "n_layers": n_layers,
            "n_harmful_samples": len(harmful_gen_results),
            "n_harmless_samples": len(harmless_gen_results),
        }
    }
    
    return results


def save_results(results: Dict, output_dir: str):
    """
    Save results to JSON and PyTorch tensor formats.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
    """
    print(f"Saving results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results as JSON
    json_path = os.path.join(output_dir, 'refusal_components.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Save tensors as PyTorch format for easier loading
    tensor_data = {
        'harmful_components': torch.tensor(results['harmful']['parallel_components']),
        'harmless_components': torch.tensor(results['harmless']['parallel_components']),
        'metadata': results['metadata']
    }
    pt_path = os.path.join(output_dir, 'refusal_components.pt')
    torch.save(tensor_data, pt_path)
    print(f"Saved PyTorch tensors to {pt_path}")


def main():
    """Main function to execute the complete pipeline."""
    parser = argparse.ArgumentParser(description="Calculate refusal direction parallel components")
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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples per dataset (0 for all)')
    
    args = parser.parse_args()
    
    print("Starting refusal component analysis...")
    print(f"Model path: {args.model_path}")
    print(f"Direction path: {args.direction_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create config object to match run_pipeline.py behavior
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    print(f"Using evaluation datasets: {cfg.evaluation_datasets}")
    
    # Load refusal direction and metadata
    direction, metadata = load_refusal_direction(args.direction_path, args.metadata_path)
    
    # Setup model and data
    model_base, harmful_instructions, harmless_instructions = setup_model_and_data(args.model_path, cfg)
    
    # Limit samples if specified
    if args.n_samples > 0:
        harmful_instructions = harmful_instructions[:args.n_samples]
        harmless_instructions = harmless_instructions[:args.n_samples]
    
    print(f"Processing {len(harmful_instructions)} harmful and {len(harmless_instructions)} harmless samples")
    
    # Generate first tokens
    print("\n=== Processing Harmful Instructions ===")
    harmful_gen_results = generate_first_token(model_base, harmful_instructions, args.batch_size)
    
    print("\n=== Processing Harmless Instructions ===")
    harmless_gen_results = generate_first_token(model_base, harmless_instructions, args.batch_size)
    
    # Collect activations and calculate parallel components
    print("\n=== Collecting Activations for Harmful Instructions ===")
    harmful_components = collect_activations_with_hooks(model_base, harmful_gen_results, direction, args.batch_size)
    
    print("\n=== Collecting Activations for Harmless Instructions ===")
    harmless_components = collect_activations_with_hooks(model_base, harmless_gen_results, direction, args.batch_size)
    
    # Organize and save results
    n_layers = model_base.model.config.num_hidden_layers
    results = organize_results(harmful_gen_results, harmless_gen_results, 
                              harmful_components, harmless_components, n_layers)
    
    save_results(results, args.output_dir)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {args.output_dir}")
    print("Run visualize_refusal_components.py to generate plots")


if __name__ == "__main__":
    main()