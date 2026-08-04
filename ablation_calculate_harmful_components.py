#!/usr/bin/env python3
"""
Calculate parallel components of activations with respect to refusal direction
using template-formatted instructions. This script processes only harmful instructions
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
from typing import List, Dict, Tuple
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks
import importlib
import re

# Attention Head Ablation Configuration for Qwen3-14B
# Format: {layer_idx: [head_indices_to_ablate]}
# Layers: 0-39, Heads per layer: 0-39
HEAD_ABLATION_CONFIG = {
    15: [23],
    18: [4],
    20: [37],
    23: [11],
    27: [34],
    31: [13],
}


# HEAD_ABLATION_CONFIG2 = {
#     15: [6, 15, 23],
#     16: [27, 37, 39],
#     17: [30],
#     18: [1, 4, 17, 19, 33],
#     19: [12, 14, 20, 22, 35],
#     20: [29, 35, 37],
#     21: [1, 11, 14, 18, 19, 30],
#     22: [17, 24, 26, 29, 35],
#     23: [4, 11, 13, 33],
#     24: [1, 3, 15, 22, 28, 29, 35, 38],
#     25: [36, 39],
#     26: [18, 21, 23, 36],
#     27: [3, 17, 23, 32, 34],
#     28: [4, 21, 30, 32, 35],
#     29: [3, 10, 20, 21, 30, 31],
#     30: [8, 13],
#     31: [4, 9, 13, 16, 19, 23, 24, 25, 39],
#     32: [0, 6, 9, 13, 27, 28, 37],
#     33: [5, 8, 17, 22, 25, 29],
#     34: [8, 10, 16, 36],
#     35: [2, 6, 9, 12, 19, 31, 34]
# }


def generate_random_head_ablation_config(exclude_config: Dict[int, List[int]],
                                        target_layers: range = range(15, 36),
                                        total_heads_to_ablate: int = 6,
                                        total_heads_per_layer: int = 40) -> Dict[int, List[int]]:
    """
    Generate random head ablation configuration excluding already configured heads.

    Args:
        exclude_config: Existing head ablation config to exclude
        target_layers: Range of layers to consider (default: 15-35)
        total_heads_to_ablate: Total number of heads to randomly select (default: 60)
        total_heads_per_layer: Total heads per layer (default: 40 for Qwen3-14B)

    Returns:
        Dictionary mapping layer indices to lists of head indices to ablate
    """
    print(f"Generating random head ablation config...")
    print(f"Target layers: {list(target_layers)}")
    print(f"Total heads to ablate: {total_heads_to_ablate}")

    # Set random seed for reproducibility
    random.seed(37)

    # Collect all available (layer, head) pairs in target layers
    available_heads = []
    excluded_heads = set()

    for layer_idx in target_layers:
        # Add existing ablated heads to exclusion set
        if layer_idx in exclude_config:
            for head_idx in exclude_config[layer_idx]:
                excluded_heads.add((layer_idx, head_idx))

        # Add all heads in this layer to available pool
        for head_idx in range(total_heads_per_layer):
            if (layer_idx, head_idx) not in excluded_heads:
                available_heads.append((layer_idx, head_idx))

    print(f"Available heads for random selection: {len(available_heads)}")
    print(f"Excluded heads from existing config: {len(excluded_heads)}")

    # Check if we have enough heads to select
    if len(available_heads) < total_heads_to_ablate:
        raise ValueError(f"Not enough available heads! "
                        f"Need {total_heads_to_ablate}, but only {len(available_heads)} available")

    # Randomly select heads
    selected_heads = random.sample(available_heads, total_heads_to_ablate)

    # Organize selected heads by layer
    random_config = {}
    for layer_idx, head_idx in selected_heads:
        if layer_idx not in random_config:
            random_config[layer_idx] = []
        random_config[layer_idx].append(head_idx)

    # Sort heads within each layer for consistency
    for layer_idx in random_config:
        random_config[layer_idx].sort()

    # Print summary
    print(f"Random ablation config generated:")
    total_selected = sum(len(heads) for heads in random_config.values())
    print(f"  - Layers affected: {len(random_config)}")
    print(f"  - Total heads selected: {total_selected}")
    for layer_idx in sorted(random_config.keys()):
        heads = random_config[layer_idx]
        print(f"  - Layer {layer_idx}: {len(heads)} heads {heads}")

    return random_config


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


def setup_model_and_harmful_data(model_path: str, cfg: Config) -> Tuple[object, List[str]]:
    """
    Setup model and load only harmful datasets.
    
    Args:
        model_path: Path to the model
        cfg: Configuration object
        
    Returns:
        Tuple of (model_base, harmful_instructions)
    """
    print(f"Loading model from {model_path}")
    model_base = construct_model_base(model_path)
    
    # Use same sampling logic as run_pipeline.py
    random.seed(37)
    
    # Load harmful instructions from evaluation datasets
    print(f"Loading harmful instructions from evaluation datasets: {cfg.evaluation_datasets}")
    harmful_instructions = []
    for dataset_name in cfg.evaluation_datasets:
        dataset = load_dataset(dataset_name)
        instructions = [d['instruction'] for d in dataset]
        harmful_instructions.extend(instructions)
        print(f"  - {dataset_name}: {len(instructions)} instructions")
    
    print(f"Total loaded: {len(harmful_instructions)} harmful instructions")
    
    return model_base, harmful_instructions


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


def get_parallel_component_hook(direction: torch.Tensor, results_cache: Dict, batch_start_idx: int, layer_idx: int):
    """
    Create a hook to collect parallel components for a batch.

    Args:
        direction: Refusal direction tensor
        results_cache: Dictionary to store results
        batch_start_idx: Starting index of current batch
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

        # Store results for each sample in the batch (move to CPU to save GPU memory)
        batch_results = parallel_component.cpu().numpy()
        for batch_idx, component_value in enumerate(batch_results):
            global_sample_idx = batch_start_idx + batch_idx
            if global_sample_idx not in results_cache:
                results_cache[global_sample_idx] = {}
            results_cache[global_sample_idx][layer_idx] = component_value

    return hook_fn


def get_qwen3_attention_head_ablation_hook(layer_idx: int, head_indices_to_ablate: List[int]):
    """
    Create correct attention head ablation hook for Qwen3-14B model.

    This hook operates on the Qwen3Attention module's forward output BEFORE o_proj.
    It correctly ablates individual attention heads in the multi-head format
    [batch, seq, heads, head_dim] before concatenation and linear projection.

    Qwen3-14B specifications:
    - 40 layers (0-39)
    - 40 attention heads per layer (0-39)
    - hidden_size = 5120
    - head_dim = 128

    Args:
        layer_idx: Layer index (0-39)
        head_indices_to_ablate: List of head indices to ablate (0-39)

    Returns:
        Hook function for attention head ablation
    """
    def hook_fn(module, input, output):
        # Qwen3Attention forward returns (attn_output, attn_weights)
        # We need to intercept BEFORE the final o_proj(attn_output) operation
        if isinstance(output, tuple):
            attn_output = output[0]  # [batch, seq, hidden_size] - this is AFTER o_proj
            attn_weights = output[1]  # attention weights
            other_outputs = output[2:] if len(output) > 2 else ()
        else:
            attn_output = output
            attn_weights = None
            other_outputs = ()

        # Problem: We're getting output AFTER o_proj, which is too late!
        # Current implementation is incorrect - we need to hook earlier in the process

        # For now, keep the old implementation but add warning
        print(f"WARNING: Current hook implementation is mathematically incorrect!")
        print(f"Hook is applied AFTER o_proj, which breaks the multi-head structure.")
        print(f"This should be fixed to hook BEFORE o_proj for correct ablation.")

        batch_size, seq_len, hidden_size = attn_output.shape

        # Qwen3-14B: 40 heads × 128 head_dim = 5120 hidden_size
        num_heads = 40
        head_dim = hidden_size // num_heads  # 128

        # This reshape is mathematically incorrect but kept for compatibility
        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)

        # Ablate specified heads (this is not correct ablation!)
        for head_idx in head_indices_to_ablate:
            if 0 <= head_idx < num_heads:
                reshaped_output[:, :, head_idx, :] = 0.0

        # Reshape back to original format: [batch, seq, hidden_size]
        ablated_output = reshaped_output.view(batch_size, seq_len, hidden_size)

        # Return modified output
        if attn_weights is not None:
            return (ablated_output, attn_weights, *other_outputs)
        else:
            return ablated_output

    return hook_fn


def get_correct_qwen3_attention_head_ablation_hook(layer_idx: int, head_indices_to_ablate: List[int]):
    """
    Create CORRECT attention head ablation hook for Qwen3-14B model.

    This implementation requires hooking into the attention computation BEFORE o_proj.
    It uses a pre-hook on the attention module to modify the internal computation.

    Args:
        layer_idx: Layer index (0-39)
        head_indices_to_ablate: List of head indices to ablate (0-39)

    Returns:
        Pre-hook function for correct attention head ablation
    """
    def pre_hook_fn(module, input):
        # Store the head indices to ablate in the module for access during forward
        module._head_indices_to_ablate = head_indices_to_ablate
        return input

    def patch_attention_forward(original_forward):
        """Patch the attention forward method to perform correct head ablation"""
        def patched_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
            # Get original shapes and parameters
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # Compute Q, K, V as normal
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # Apply RoPE
            cos, sin = position_embeddings
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Handle past key values if needed
            if kwargs.get('past_key_value') is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get('cache_position')}
                key_states, value_states = kwargs['past_key_value'].update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # Perform attention computation
            from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
            attn_output, attn_weights = eager_attention_forward(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs
            )

            # NOW perform correct head ablation on attn_output [batch, heads, seq, head_dim]
            if hasattr(self, '_head_indices_to_ablate'):
                for head_idx in self._head_indices_to_ablate:
                    if 0 <= head_idx < attn_output.shape[1]:  # heads dimension
                        attn_output[:, head_idx, :, :] = 0.0

            # Continue with normal processing
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            return attn_output, attn_weights

        return patched_forward

    return pre_hook_fn, patch_attention_forward


def create_head_ablation_hooks(model_base, head_ablation_config: Dict[int, List[int]], use_correct_implementation: bool = False):
    """
    Create all attention head ablation hooks based on configuration.

    Args:
        model_base: Qwen3 model instance
        head_ablation_config: {layer_idx: [head_indices_to_ablate]}
        use_correct_implementation: If True, use mathematically correct ablation

    Returns:
        List[Tuple[module, hook_fn]]: Hook list for add_hooks
    """
    ablation_hooks = []

    if use_correct_implementation:
        print(f"Creating CORRECT ablation hooks for {len(head_ablation_config)} layers...")
        # Store original forward methods for restoration
        original_forwards = {}

        for layer_idx, head_indices in head_ablation_config.items():
            if 0 <= layer_idx < len(model_base.model_block_modules):
                attn_module = model_base.model_block_modules[layer_idx].self_attn

                # Store original forward method
                original_forwards[layer_idx] = attn_module.forward

                # Get the patching functions
                pre_hook_fn, patch_forward_fn = get_correct_qwen3_attention_head_ablation_hook(layer_idx, head_indices)

                # Set ablation indices directly instead of using pre-hook
                attn_module._head_indices_to_ablate = head_indices

                # Patch the forward method
                attn_module.forward = patch_forward_fn(attn_module.forward).__get__(attn_module, type(attn_module))

                # No need for hooks since we're patching the forward method directly
                # ablation_hooks remains empty for correct implementation

                print(f"  Layer {layer_idx}: patched forward method to ablate heads {head_indices}")
            else:
                print(f"  Warning: Layer {layer_idx} out of range, skipping")

        # Store original forwards for potential restoration
        model_base._original_attention_forwards = original_forwards
    else:
        print(f"Creating LEGACY (incorrect) ablation hooks for {len(head_ablation_config)} layers...")
        print("WARNING: Using mathematically incorrect implementation!")

        for layer_idx, head_indices in head_ablation_config.items():
            if 0 <= layer_idx < len(model_base.model_block_modules):
                # Get self_attn module for specified layer
                attn_module = model_base.model_block_modules[layer_idx].self_attn

                # Create ablation hook for this layer
                ablation_hook = get_qwen3_attention_head_ablation_hook(layer_idx, head_indices)

                # Add to hooks list
                ablation_hooks.append((attn_module, ablation_hook))

                print(f"  Layer {layer_idx}: ablating heads {head_indices}")
            else:
                print(f"  Warning: Layer {layer_idx} out of range, skipping")

    return ablation_hooks


def restore_original_attention_forwards(model_base):
    """
    Restore original attention forward methods after ablation experiment.

    Args:
        model_base: Model instance with potentially patched attention forwards
    """
    if hasattr(model_base, '_original_attention_forwards'):
        print("Restoring original attention forward methods...")
        for layer_idx, original_forward in model_base._original_attention_forwards.items():
            if 0 <= layer_idx < len(model_base.model_block_modules):
                attn_module = model_base.model_block_modules[layer_idx].self_attn
                attn_module.forward = original_forward
                # Clean up the ablation indices
                if hasattr(attn_module, '_head_indices_to_ablate'):
                    delattr(attn_module, '_head_indices_to_ablate')

        # Clean up the stored forwards
        delattr(model_base, '_original_attention_forwards')
        print("Original attention forwards restored.")
    else:
        print("No original attention forwards found to restore.")


def collect_activations_with_head_ablation(model_base, formatted_instructions: List[str],
                                          direction: torch.Tensor, head_ablation_config: Dict,
                                          batch_size: int = 8, use_correct_implementation: bool = False) -> Dict:
    """
    Collect activations with attention head ablation applied.

    Args:
        model_base: Model instance
        formatted_instructions: List of template-formatted instructions
        direction: Refusal direction tensor
        head_ablation_config: Dictionary of layer->heads to ablate
        batch_size: Batch size for processing
        use_correct_implementation: If True, use mathematically correct ablation

    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with attention head ablation...")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers

    # Create ablation hooks
    ablation_hooks = create_head_ablation_hooks(model_base, head_ablation_config, use_correct_implementation)

    for i in tqdm(range(0, len(formatted_instructions), batch_size), desc="Processing batches with ablation"):
        batch_instructions = formatted_instructions[i:i+batch_size]

        # Tokenize
        tokenized = model_base.tokenizer(
            batch_instructions,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False
        )

        input_ids = tokenized.input_ids.to(model_base.model.device)
        attention_mask = tokenized.attention_mask.to(model_base.model.device)

        # Create parallel component collection hooks (one per layer for the batch)
        component_hooks = []
        for layer_idx in range(n_layers):
            hook = get_parallel_component_hook(direction, results_cache, i, layer_idx)
            component_hooks.append((model_base.model_block_modules[layer_idx], hook))

        # Forward pass with component collection hooks
        # Note: For correct implementation, ablation is done via patched forward methods
        if use_correct_implementation:
            hooks_to_use = []  # No additional hooks needed, ablation is in patched forward
        else:
            hooks_to_use = ablation_hooks  # Use legacy hook-based ablation

        with add_hooks(module_forward_pre_hooks=component_hooks,
                      module_forward_hooks=hooks_to_use):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)

        # Clear GPU memory
        torch.cuda.empty_cache()

    return results_cache


def save_ablation_metadata(head_ablation_config: Dict, output_dir: str, length_suffix: str):
    """Save ablation configuration metadata."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "ablation_type": "attention_head_ablation",
        "model": "Qwen3-14B",
        "total_layers": 40,
        "total_heads_per_layer": 40,
        "ablation_config": head_ablation_config,
        "ablated_layers": list(head_ablation_config.keys()),
        "total_ablated_heads": sum(len(heads) for heads in head_ablation_config.values()),
        "length_suffix": length_suffix
    }

    metadata_path = os.path.join(output_dir, "ablation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved ablation metadata to {metadata_path}")


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
        
        # Create hooks for all layers (one per layer for the batch)
        fwd_pre_hooks = []
        for layer_idx in range(n_layers):
            hook = get_parallel_component_hook(direction, results_cache, i, layer_idx)
            fwd_pre_hooks.append((model_base.model_block_modules[layer_idx], hook))
        
        # Forward pass with hooks
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Clear GPU cache after each batch to prevent memory accumulation
        torch.cuda.empty_cache()
    
    return results_cache


def organize_results(original_instructions: List[str], formatted_instructions: List[str], 
                     component_values: Dict, n_layers: int, direction_metadata: Dict) -> Dict:
    """
    Organize results into structured format.
    
    Args:
        original_instructions: Original harmful instructions
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
        "harmful": {
            "original_instructions": original_instructions,
            "formatted_instructions": formatted_instructions,
            "parallel_components": component_tensor.tolist(),
        },
        "metadata": {
            "n_layers": n_layers,
            "n_samples": len(original_instructions),
            "template_used": "format_thinking_template",
            "direction_metadata": direction_metadata
        }
    }
    
    return results


def save_results(results: Dict, output_dir: str, length_suffix: str = "unknown"):
    """
    Save results to JSON and PyTorch tensor formats with length suffix.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
        length_suffix: Template length suffix (e.g., '3k', '11k')
    """
    print(f"Saving results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results as JSON with length suffix
    json_path = os.path.join(output_dir, f'template_harmful_components_{length_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Save tensors as PyTorch format for easier loading with length suffix
    tensor_data = {
        'parallel_components': torch.tensor(results['harmful']['parallel_components']),
        'metadata': results['metadata']
    }
    pt_path = os.path.join(output_dir, f'template_harmful_components_{length_suffix}.pt')
    torch.save(tensor_data, pt_path)
    print(f"Saved PyTorch tensors to {pt_path}")


def main():
    """Main function to execute the complete pipeline."""
    parser = argparse.ArgumentParser(description="Calculate refusal direction parallel components using template for harmful instructions")
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
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples (0 for all)')
    parser.add_argument('--template_file', type=str, default='template_1k',
                       help='Template file to use (without .py extension, e.g., template_1k, template_11k)')

    # Attention Head Ablation parameters
    parser.add_argument('--enable_head_ablation', action='store_true',
                       help='Enable attention head ablation during forward pass')
    parser.add_argument('--ablation_output_dir', type=str, default='./results/head_ablation_results',
                       help='Output directory for ablation results')
    parser.add_argument('--use_correct_ablation', action='store_true', default=True,
                       help='Use mathematically correct attention head ablation implementation')
    parser.add_argument('--enable_random_head_ablation', action='store_true',
                       help='Enable random head ablation in layers 15-35 (60 heads, excluding configured heads)')

    args = parser.parse_args()

    # Check for conflicting ablation options
    if args.enable_head_ablation and args.enable_random_head_ablation:
        print("❌ Error: Cannot use both --enable_head_ablation and --enable_random_head_ablation simultaneously")
        print("Please choose only one ablation type:")
        print("  --enable_head_ablation: Use predefined HEAD_ABLATION_CONFIG")
        print("  --enable_random_head_ablation: Use randomly selected heads")
        return

    print("Starting template-based harmful component analysis...")
    if args.enable_head_ablation:
        print("🎯 ATTENTION HEAD ABLATION ENABLED")
        print(f"Will ablate {len(HEAD_ABLATION_CONFIG)} layers with {sum(len(heads) for heads in HEAD_ABLATION_CONFIG.values())} total heads")
    elif args.enable_random_head_ablation:
        print("🎯 RANDOM HEAD ABLATION ENABLED")
        print("Will randomly select 60 heads from layers 15-35 (excluding configured heads)")
    print(f"Model path: {args.model_path}")
    print(f"Direction path: {args.direction_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Template file: {args.template_file}")
    
    # Load template module
    try:
        format_thinking_template = load_template_module(args.template_file)
        length_suffix = extract_length_from_filename(args.template_file)
        print(f"Loaded template with length: {length_suffix}")
    except Exception as e:
        print(f"Error loading template: {e}")
        return
    
    # Create config object to match run_pipeline.py behavior
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    print(f"Using evaluation datasets: {cfg.evaluation_datasets}")
    
    # Load refusal direction and metadata
    direction, metadata = load_refusal_direction(args.direction_path, args.metadata_path)
    
    # Setup model and data (only harmful)
    model_base, harmful_instructions = setup_model_and_harmful_data(args.model_path, cfg)
    
    # Limit samples if specified
    if args.n_samples > 0:
        harmful_instructions = harmful_instructions[:args.n_samples]
    
    print(f"Processing {len(harmful_instructions)} harmful samples")
    
    # Format instructions with template
    formatted_instructions = format_instructions_with_template(harmful_instructions, format_thinking_template)

    # 🔥 Critical branch: Choose ablation type
    if args.enable_head_ablation:
        ablation_type = "CORRECT" if args.use_correct_ablation else "LEGACY"
        print(f"\n=== 🎯 Running with {ablation_type} Attention Head Ablation ===")
        component_values = collect_activations_with_head_ablation(
            model_base, formatted_instructions, direction, HEAD_ABLATION_CONFIG,
            args.batch_size, args.use_correct_ablation
        )

        # Use dedicated output directory and filename
        output_dir = args.ablation_output_dir
        ablation_suffix = "head_ablated"
        final_length_suffix = f"{length_suffix}_{ablation_suffix}"
        ablation_config_used = HEAD_ABLATION_CONFIG

        # Save ablation metadata
        save_ablation_metadata(HEAD_ABLATION_CONFIG, output_dir, final_length_suffix)

        # Restore original attention forwards if using correct implementation
        if args.use_correct_ablation:
            restore_original_attention_forwards(model_base)

    elif args.enable_random_head_ablation:
        print(f"\n=== 🎯 Running with Random Head Ablation ===")

        # Generate random head ablation config
        random_ablation_config = generate_random_head_ablation_config(HEAD_ABLATION_CONFIG)

        ablation_type = "CORRECT" if args.use_correct_ablation else "LEGACY"
        print(f"Using {ablation_type} ablation implementation")

        component_values = collect_activations_with_head_ablation(
            model_base, formatted_instructions, direction, random_ablation_config,
            args.batch_size, args.use_correct_ablation
        )

        # Use dedicated output directory and filename
        output_dir = args.ablation_output_dir
        ablation_suffix = "random_head_ablated"
        final_length_suffix = f"{length_suffix}_{ablation_suffix}"
        ablation_config_used = random_ablation_config

        # Save ablation metadata
        save_ablation_metadata(random_ablation_config, output_dir, final_length_suffix)

        # Restore original attention forwards if using correct implementation
        if args.use_correct_ablation:
            restore_original_attention_forwards(model_base)

    else:
        print("\n=== Running without Ablation ===")
        component_values = collect_activations_with_hooks(
            model_base, formatted_instructions, direction, args.batch_size
        )
        output_dir = args.output_dir
        final_length_suffix = length_suffix
        ablation_config_used = None

    # Organize and save results
    n_layers = model_base.model.config.num_hidden_layers
    if args.enable_head_ablation or args.enable_random_head_ablation:
        ablation_type_str = "predefined" if args.enable_head_ablation else "random"
        metadata.update({
            "ablation_applied": True,
            "ablation_type": ablation_type_str,
            "ablation_config": ablation_config_used,
            "template_file": args.template_file,
            "template_length": length_suffix
        })
    else:
        metadata.update({
            "ablation_applied": False,
            "template_file": args.template_file,
            "template_length": length_suffix
        })

    results = organize_results(harmful_instructions, formatted_instructions,
                              component_values, n_layers, metadata)

    save_results(results, output_dir, final_length_suffix)
    
    print(f"\n=== ✅ Analysis Complete ===")
    print(f"Results saved to {output_dir}")
    if args.enable_head_ablation:
        print(f"🎯 Predefined ablation applied to {len(HEAD_ABLATION_CONFIG)} layers")
    elif args.enable_random_head_ablation:
        print(f"🎯 Random ablation applied to {len(ablation_config_used)} layers with {sum(len(heads) for heads in ablation_config_used.values())} total heads")
    print("Template-based harmful component calculation finished")


if __name__ == "__main__":
    main()