#!/usr/bin/env python3
"""
Calculate parallel components of activations with respect to refusal direction
for jailbreak analysis. This script processes template_harmful_completions_with_asr.json
and groups samples by is_jailbreak_deepseekv31 field (0/1) to analyze complete
'formatted_instruction+completion' sequences.
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


def load_jailbreak_data(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load jailbreak data and split by is_jailbreak_deepseekv31.
    
    Args:
        json_path: Path to template_harmful_completions_with_asr.json
        
    Returns:
        Tuple of (successful_jailbreak_samples, failed_jailbreak_samples)
    """
    print(f"Loading jailbreak data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    completions = data['completions']
    
    successful_jailbreaks = []
    failed_jailbreaks = []
    
    for completion in completions:
        if completion['is_jailbreak_deepseekv31'] == 1:
            successful_jailbreaks.append(completion)
        else:
            failed_jailbreaks.append(completion)
    
    print(f"Loaded {len(successful_jailbreaks)} successful jailbreak samples")
    print(f"Loaded {len(failed_jailbreaks)} failed jailbreak samples")
    
    return successful_jailbreaks, failed_jailbreaks


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


def setup_model(model_path: str) -> object:
    """
    Setup model.
    
    Args:
        model_path: Path to the model
        
    Returns:
        model_base object
    """
    print(f"Loading model from {model_path}")
    model_base = construct_model_base(model_path)
    return model_base


def prepare_complete_sequences(samples: List[Dict]) -> List[str]:
    """
    Prepare complete sequences by concatenating formatted_instruction and completion.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        List of complete sequences
    """
    print("Preparing complete sequences...")
    complete_sequences = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        formatted_instruction = sample['formatted_instruction']
        completion = sample['completion']
        complete_sequence = formatted_instruction + completion
        complete_sequences.append(complete_sequence)
    
    return complete_sequences


def get_parallel_component_hook(direction: torch.Tensor, results_cache: Dict, sample_idx: int, layer_idx: int):
    """
    Create a hook to collect parallel components for complete sequences.
    
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
        # Take the last token position (end of complete sequence)
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


def collect_activations_with_hooks(model_base, complete_sequences: List[str], 
                                   direction: torch.Tensor, batch_size: int = 8) -> Dict:
    """
    Single forward pass to collect activations and calculate parallel components.
    
    Args:
        model_base: Model instance
        complete_sequences: List of complete sequences (formatted_instruction + completion)
        direction: Refusal direction tensor
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with hooks for complete sequences...")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers
    
    for i in tqdm(range(0, len(complete_sequences), batch_size), desc="Processing batches"):
        batch_sequences = complete_sequences[i:i+batch_size]
        
        # Tokenize the complete sequences
        tokenized = model_base.tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # Sequences already include special tokens
        )
        
        input_ids = tokenized.input_ids.to(model_base.model.device)
        attention_mask = tokenized.attention_mask.to(model_base.model.device)
        
        # Create hooks for all layers
        fwd_pre_hooks = []
        for layer_idx in range(n_layers):
            for batch_idx in range(len(batch_sequences)):
                global_sample_idx = i + batch_idx
                hook = get_parallel_component_hook(direction, results_cache, global_sample_idx, layer_idx)
                fwd_pre_hooks.append((model_base.model_block_modules[layer_idx], hook))
        
        # Forward pass with hooks
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Clear GPU cache after each batch to prevent memory accumulation
        torch.cuda.empty_cache()
    
    return results_cache


def organize_results(samples: List[Dict], complete_sequences: List[str], 
                     component_values: Dict, n_layers: int, direction_metadata: Dict, 
                     group_name: str) -> Dict:
    """
    Organize results into structured format.
    
    Args:
        samples: Original sample dictionaries
        complete_sequences: Complete sequences
        component_values: Parallel component values
        n_layers: Number of layers in the model
        direction_metadata: Metadata about the refusal direction
        group_name: Name of the group (e.g., 'successful_jailbreaks', 'failed_jailbreaks')
        
    Returns:
        Structured results dictionary
    """
    print(f"Organizing results for {group_name}...")
    
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
    
    component_tensor = components_to_tensor(component_values, len(samples), n_layers)
    
    results = {
        group_name: {
            "original_samples": samples,
            "complete_sequences": complete_sequences,
            "parallel_components": component_tensor.tolist(),
        },
        "metadata": {
            "n_layers": n_layers,
            "n_samples": len(samples),
            "group_name": group_name,
            "direction_metadata": direction_metadata
        }
    }
    
    return results


def save_results(results: Dict, output_dir: str, group_name: str):
    """
    Save results to JSON and PyTorch tensor formats.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
        group_name: Group name for filename
    """
    print(f"Saving {group_name} results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results as JSON
    json_path = os.path.join(output_dir, f'{group_name}_components.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Save tensors as PyTorch format for easier loading
    tensor_data = {
        'parallel_components': torch.tensor(results[group_name]['parallel_components']),
        'metadata': results['metadata']
    }
    pt_path = os.path.join(output_dir, f'{group_name}_components.pt')
    torch.save(tensor_data, pt_path)
    print(f"Saved PyTorch tensors to {pt_path}")


def main():
    """Main function to execute the complete pipeline."""
    parser = argparse.ArgumentParser(description="Calculate refusal direction parallel components for jailbreak analysis")
    parser.add_argument('--model_path', type=str, 
                       default='Qwen/Qwen3-14B', 
                       help='Path to the model')
    parser.add_argument('--jailbreak_json', type=str, 
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/template_generation_results/template_harmful_completions_with_asr.json',
                       help='Path to template_harmful_completions_with_asr.json')
    parser.add_argument('--direction_path', type=str, 
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/pipeline/runs/Qwen3-14B/direction.pt',
                       help='Path to direction.pt')
    parser.add_argument('--metadata_path', type=str, 
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/pipeline/runs/Qwen3-14B/direction_metadata.json',
                       help='Path to direction_metadata.json')
    parser.add_argument('--output_dir', type=str, default='./jailbreak_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--n_samples', type=int, default=0, help='Number of samples per group (0 for all)')
    
    args = parser.parse_args()
    
    print("Starting jailbreak component analysis...")
    print(f"Model path: {args.model_path}")
    print(f"Jailbreak JSON: {args.jailbreak_json}")
    print(f"Direction path: {args.direction_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Load refusal direction and metadata
    direction, metadata = load_refusal_direction(args.direction_path, args.metadata_path)
    
    # Setup model
    model_base = setup_model(args.model_path)
    
    # Load and split jailbreak data
    successful_jailbreaks, failed_jailbreaks = load_jailbreak_data(args.jailbreak_json)
    
    # Limit samples if specified
    if args.n_samples > 0:
        successful_jailbreaks = successful_jailbreaks[:args.n_samples]
        failed_jailbreaks = failed_jailbreaks[:args.n_samples]
    
    print(f"Processing {len(successful_jailbreaks)} successful jailbreak samples")
    print(f"Processing {len(failed_jailbreaks)} failed jailbreak samples")
    
    # Process successful jailbreaks
    if len(successful_jailbreaks) > 0:
        print("\n=== Processing Successful Jailbreaks ===")
        successful_sequences = prepare_complete_sequences(successful_jailbreaks)
        successful_components = collect_activations_with_hooks(
            model_base, successful_sequences, direction, args.batch_size
        )
        
        n_layers = model_base.model.config.num_hidden_layers
        successful_results = organize_results(
            successful_jailbreaks, successful_sequences, successful_components, 
            n_layers, metadata, "successful_jailbreaks"
        )
        save_results(successful_results, args.output_dir, "successful_jailbreaks")
    
    # Process failed jailbreaks
    if len(failed_jailbreaks) > 0:
        print("\n=== Processing Failed Jailbreaks ===")
        failed_sequences = prepare_complete_sequences(failed_jailbreaks)
        failed_components = collect_activations_with_hooks(
            model_base, failed_sequences, direction, args.batch_size
        )
        
        n_layers = model_base.model.config.num_hidden_layers
        failed_results = organize_results(
            failed_jailbreaks, failed_sequences, failed_components, 
            n_layers, metadata, "failed_jailbreaks"
        )
        save_results(failed_results, args.output_dir, "failed_jailbreaks")
    
    print("\n=== Jailbreak Analysis Complete ===")
    print(f"Results saved to {args.output_dir}")
    print("Jailbreak component calculation finished")


if __name__ == "__main__":
    main()