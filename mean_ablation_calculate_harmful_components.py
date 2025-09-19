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
# HEAD_ABLATION_CONFIG = {
#     15: [23],
#     18: [1, 4, 19],
#     19: [12, 14],
#     20: [29, 35, 37],
#     21: [1, 11, 18, 19],
#     22: [17, 24, 26, 29, 35],
#     23: [11],
#     24: [1, 22, 35, 38],
#     25: [36],
#     26: [18, 21, 23],
#     27: [3, 23, 32, 34],
#     28: [4, 21, 30, 32, 35],
#     29: [3, 10, 20, 21],
#     30: [8, 13],
#     31: [9, 13, 16, 19, 24],
#     32: [0, 6, 13, 27, 28],
#     33: [8, 17, 22, 25],
#     34: [8, 36],
#     35: [6, 12]
# }


HEAD_ABLATION_CONFIG = {
    15: [6, 15, 23],
    16: [27, 37, 39],
    17: [30],
    18: [1, 4, 17, 19, 33],
    19: [12, 14, 20, 22, 35],
    20: [29, 35, 37],
    21: [1, 11, 14, 18, 19, 30],
    22: [17, 24, 26, 29, 35],
    23: [4, 11, 13, 33],
    24: [1, 3, 15, 22, 28, 29, 35, 38],
    25: [36, 39],
    26: [18, 21, 23, 36],
    27: [3, 17, 23, 32, 34],
    28: [4, 21, 30, 32, 35],
    29: [3, 10, 20, 21, 30, 31],
    30: [8, 13],
    31: [4, 9, 13, 16, 19, 23, 24, 25, 39],
    32: [0, 6, 9, 13, 27, 28, 37],
    33: [5, 8, 17, 22, 25, 29],
    34: [8, 10, 16, 36],
    35: [2, 6, 9, 12, 19, 31, 34]
}

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
    random.seed(42)
    
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


def get_qwen3_true_undifferentiated_attention_hooks(layer_idx: int, head_indices_to_modify: List[int]):
    """
    Create TRUE causal undifferentiated attention hooks for Qwen3-14B model.

    This implementation captures the input hidden states from transformer block level
    and recomputes attention output using causal uniform attention weights.

    Args:
        layer_idx: Layer index (0-39)
        head_indices_to_modify: List of head indices to make undifferentiated (0-39)

    Returns:
        Tuple of (block_pre_hook, attention_forward_hook) for capturing inputs and recomputing attention
    """
    # Cache to store hidden states between block pre-hook and attention forward-hook
    # Use a unique cache per function instance to avoid conflicts
    hidden_states_cache = {}

    def block_pre_hook(module, input_tuple):
        """Block pre-hook: Capture input hidden states from transformer block"""
        try:
            if len(input_tuple) == 0:
                return input_tuple

            # Transformer block forward signature: forward(hidden_states, ...)
            hidden_states = input_tuple[0]  # First argument is hidden_states [batch, seq, hidden_size]

            # Store in cache with layer-specific key
            cache_key = f'layer_{layer_idx}_hidden_states'
            hidden_states_cache[cache_key] = hidden_states.clone().detach()

        except Exception as e:
            print(f"Error in block pre-hook layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
        return input_tuple

    def forward_hook(module, input_args, output):
        """Forward-hook: Recompute attention with uniform weights for specified heads"""
        try:
            if isinstance(output, tuple):
                attn_output = output[0]  # [batch, seq, hidden_size=5120]
                other_outputs = output[1:]
            else:
                attn_output = output
                other_outputs = ()

            # Retrieve cached hidden states
            cache_key = f'layer_{layer_idx}_hidden_states'
            if cache_key not in hidden_states_cache:
                print(f"Warning: No cached hidden states for layer {layer_idx}")
                return output

            hidden_states = hidden_states_cache[cache_key]
            batch_size, seq_len, hidden_size = hidden_states.shape

            # Qwen3-14B model parameters
            num_heads = 40
            head_dim = hidden_size // num_heads  # 128
            num_key_value_heads = 8  # GQA: 8 key-value head groups
            num_key_value_groups = num_heads // num_key_value_heads  # 5 heads per group

            # Compute value states from original hidden states
            value_states = module.v_proj(hidden_states)  # [batch, seq, 1024]
            value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)

            # Reshape attention output to multi-head format for modification
            reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)

            # Apply TRUE uniform attention for specified heads
            for head_idx in head_indices_to_modify:
                if 0 <= head_idx < num_heads:
                    # Get corresponding key-value group for this query head (GQA)
                    kv_group_idx = head_idx // num_key_value_groups
                    head_values = value_states[:, :, kv_group_idx, :]  # [batch, seq, head_dim]

                    # APPROACH: Use model's natural causal mask, but make attention uniform within allowed positions
                    # We need to work with the model's existing causal attention logic

                    # Since we can't easily extract the exact attention weights that were computed,
                    # we'll create a causal uniform attention pattern that respects the standard causal mask

                    # Create standard causal mask (lower triangular)
                    causal_mask = torch.tril(torch.ones(
                        seq_len, seq_len,
                        device=attn_output.device,
                        dtype=attn_output.dtype
                    ))

                    # Make attention uniform within the causal constraints
                    # Each position attends uniformly to all allowed previous positions
                    row_sums = causal_mask.sum(dim=-1, keepdim=True)  # [seq_len, 1]
                    uniform_weights = causal_mask / row_sums  # [seq_len, seq_len]

                    # Expand to batch dimension
                    uniform_weights = uniform_weights.unsqueeze(0).expand(batch_size, -1, -1)

                    # Compute uniform attention output: uniform_weights @ values
                    true_uniform_output = torch.bmm(uniform_weights, head_values)

                    # Replace the head output with true uniform attention result
                    reshaped_output[:, :, head_idx, :] = true_uniform_output

            # Reshape back to original format
            modified_output = reshaped_output.view(batch_size, seq_len, hidden_size)

            # Clean up cache to prevent memory leaks
            if cache_key in hidden_states_cache:
                del hidden_states_cache[cache_key]

            # Return modified output
            if other_outputs:
                return (modified_output, *other_outputs)
            else:
                return modified_output

        except Exception as e:
            print(f"Error in true undifferentiated attention hook for layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up cache and return original output
            cache_key = f'layer_{layer_idx}_hidden_states'
            if cache_key in hidden_states_cache:
                del hidden_states_cache[cache_key]
            return output

    return block_pre_hook, forward_hook


def get_qwen3_attention_head_ablation_hook(layer_idx: int, head_indices_to_ablate: List[int]):
    """
    Create attention head ablation hook for Qwen3-14B model.

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
        # Qwen3 self_attn output: (attention_output, attention_weights, past_key_value)
        # We only need to modify attention_output
        if isinstance(output, tuple):
            attn_output = output[0]  # [batch, seq, hidden_size=5120]
            other_outputs = output[1:]
        else:
            attn_output = output
            other_outputs = ()

        batch_size, seq_len, hidden_size = attn_output.shape

        # Qwen3-14B: 40 heads × 128 head_dim = 5120 hidden_size
        num_heads = 40
        head_dim = hidden_size // num_heads  # 128

        # Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)

        # Ablate specified heads (set to zero)
        for head_idx in head_indices_to_ablate:
            if 0 <= head_idx < num_heads:
                reshaped_output[:, :, head_idx, :] = 0.0

        # Reshape back to original format: [batch, seq, hidden_size]
        ablated_output = reshaped_output.view(batch_size, seq_len, hidden_size)

        # Return modified output
        if other_outputs:
            return (ablated_output, *other_outputs)
        else:
            return ablated_output

    return hook_fn


def create_head_undifferentiated_attention_hooks(model_base, head_config: Dict[int, List[int]]):
    """
    Create TRUE undifferentiated attention hooks that replace specified heads with uniform attention.

    Args:
        model_base: Qwen3 model instance
        head_config: {layer_idx: [head_indices_to_make_undifferentiated]}

    Returns:
        Tuple of (pre_hooks, forward_hooks) for add_hooks
    """
    pre_hooks = []
    forward_hooks = []

    print(f"Creating TRUE undifferentiated attention hooks for {len(head_config)} layers...")

    for layer_idx, head_indices in head_config.items():
        if 0 <= layer_idx < len(model_base.model_block_modules):
            # Get transformer block and attention module for specified layer
            block_module = model_base.model_block_modules[layer_idx]
            attn_module = block_module.self_attn

            # Create true undifferentiated attention hooks for this layer
            block_pre_hook, attention_forward_hook = get_qwen3_true_undifferentiated_attention_hooks(layer_idx, head_indices)

            # Add block pre-hook to capture hidden states
            pre_hooks.append((block_module, block_pre_hook))
            # Add attention forward-hook to modify attention output
            forward_hooks.append((attn_module, attention_forward_hook))

            print(f"  Layer {layer_idx}: making heads {head_indices} TRULY undifferentiated")
        else:
            print(f"  Warning: Layer {layer_idx} out of range, skipping")

    return pre_hooks, forward_hooks


def create_head_ablation_hooks(model_base, head_ablation_config: Dict[int, List[int]]):
    """
    Create all attention head ablation hooks based on configuration.

    Args:
        model_base: Qwen3 model instance
        head_ablation_config: {layer_idx: [head_indices_to_ablate]}

    Returns:
        List[Tuple[module, hook_fn]]: Hook list for add_hooks
    """
    ablation_hooks = []

    print(f"Creating ablation hooks for {len(head_ablation_config)} layers...")

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


def collect_activations_with_head_ablation(model_base, formatted_instructions: List[str],
                                          direction: torch.Tensor, head_ablation_config: Dict,
                                          batch_size: int = 8) -> Dict:
    """
    Collect activations with attention head ablation applied.

    Args:
        model_base: Model instance
        formatted_instructions: List of template-formatted instructions
        direction: Refusal direction tensor
        head_ablation_config: Dictionary of layer->heads to ablate
        batch_size: Batch size for processing

    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with attention head ablation...")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers

    # Create ablation hooks
    ablation_hooks = create_head_ablation_hooks(model_base, head_ablation_config)

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

        # Forward pass with both ablation hooks and component collection hooks
        with add_hooks(module_forward_pre_hooks=component_hooks,
                      module_forward_hooks=ablation_hooks):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)

        # Clear GPU memory
        torch.cuda.empty_cache()

    return results_cache


def collect_activations_with_undifferentiated_attention(model_base, formatted_instructions: List[str],
                                                       direction: torch.Tensor, head_config: Dict,
                                                       batch_size: int = 8) -> Dict:
    """
    Collect activations with TRUE undifferentiated attention applied to specified heads.

    Args:
        model_base: Model instance
        formatted_instructions: List of template-formatted instructions
        direction: Refusal direction tensor
        head_config: Dictionary of layer->heads to make undifferentiated
        batch_size: Batch size for processing

    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with TRUE undifferentiated attention...")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers

    # Create TRUE undifferentiated attention hooks (pre-hook + forward-hook combination)
    undifferentiated_pre_hooks, undifferentiated_forward_hooks = create_head_undifferentiated_attention_hooks(model_base, head_config)

    for i in tqdm(range(0, len(formatted_instructions), batch_size), desc="Processing batches with TRUE undifferentiated attention"):
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

        # Combine all hooks: component collection (pre) + undifferentiated attention (pre + forward)
        all_pre_hooks = component_hooks + undifferentiated_pre_hooks
        all_forward_hooks = undifferentiated_forward_hooks

        # Forward pass with both undifferentiated attention hooks and component collection hooks
        with add_hooks(module_forward_pre_hooks=all_pre_hooks,
                      module_forward_hooks=all_forward_hooks):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)

        # Clear GPU memory
        torch.cuda.empty_cache()

    return results_cache


def save_undifferentiated_attention_metadata(head_config: Dict, output_dir: str, length_suffix: str):
    """Save undifferentiated attention configuration metadata."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "intervention_type": "undifferentiated_attention",
        "model": "Qwen3-14B",
        "total_layers": 40,
        "total_heads_per_layer": 40,
        "undifferentiated_config": head_config,
        "modified_layers": list(head_config.keys()),
        "total_modified_heads": sum(len(heads) for heads in head_config.values()),
        "length_suffix": length_suffix,
        "description": "Causal uniform attention weights applied to specified heads. Each position attends uniformly to all previous positions (including itself), respecting causality."
    }

    metadata_path = os.path.join(output_dir, "undifferentiated_attention_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved undifferentiated attention metadata to {metadata_path}")


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
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples (0 for all)')
    parser.add_argument('--template_file', type=str, default='template_1k',
                       help='Template file to use (without .py extension, e.g., template_1k, template_11k)')

    # Attention Head Intervention parameters
    parser.add_argument('--enable_head_ablation', action='store_true',
                       help='Enable attention head ablation during forward pass (set heads to zero)')
    parser.add_argument('--enable_undifferentiated_attention', action='store_true',
                       help='Enable undifferentiated attention (uniform attention weights for specified heads)')
    parser.add_argument('--ablation_output_dir', type=str, default='./results/head_ablation_results',
                       help='Output directory for ablation results')
    parser.add_argument('--undifferentiated_output_dir', type=str, default='./results/undifferentiated_attention_results',
                       help='Output directory for undifferentiated attention results')

    args = parser.parse_args()
    
    print("Starting template-based harmful component analysis...")

    # Validate intervention arguments
    if args.enable_head_ablation and args.enable_undifferentiated_attention:
        print("❌ ERROR: Cannot enable both head ablation and undifferentiated attention simultaneously")
        return

    if args.enable_head_ablation:
        print("🎯 ATTENTION HEAD ABLATION ENABLED")
        print(f"Will ablate {len(HEAD_ABLATION_CONFIG)} layers with {sum(len(heads) for heads in HEAD_ABLATION_CONFIG.values())} total heads")
    elif args.enable_undifferentiated_attention:
        print("🔄 UNDIFFERENTIATED ATTENTION ENABLED")
        print(f"Will apply uniform attention to {len(HEAD_ABLATION_CONFIG)} layers with {sum(len(heads) for heads in HEAD_ABLATION_CONFIG.values())} total heads")

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

    # 🔥 Critical branch: Choose intervention type
    if args.enable_head_ablation:
        print("\n=== 🎯 Running with Attention Head Ablation ===")
        component_values = collect_activations_with_head_ablation(
            model_base, formatted_instructions, direction, HEAD_ABLATION_CONFIG, args.batch_size
        )

        # Use dedicated output directory and filename
        output_dir = args.ablation_output_dir
        intervention_suffix = "head_ablated"
        final_length_suffix = f"{length_suffix}_{intervention_suffix}"

        # Save ablation metadata
        save_ablation_metadata(HEAD_ABLATION_CONFIG, output_dir, final_length_suffix)

        # Update metadata for saving
        metadata.update({
            "intervention_applied": "head_ablation",
            "intervention_config": HEAD_ABLATION_CONFIG,
            "template_file": args.template_file,
            "template_length": length_suffix
        })

    elif args.enable_undifferentiated_attention:
        print("\n=== 🔄 Running with Undifferentiated Attention ===")
        component_values = collect_activations_with_undifferentiated_attention(
            model_base, formatted_instructions, direction, HEAD_ABLATION_CONFIG, args.batch_size
        )

        # Use dedicated output directory and filename
        output_dir = args.undifferentiated_output_dir
        intervention_suffix = "undifferentiated"
        final_length_suffix = f"{length_suffix}_{intervention_suffix}"

        # Save undifferentiated attention metadata
        save_undifferentiated_attention_metadata(HEAD_ABLATION_CONFIG, output_dir, final_length_suffix)

        # Update metadata for saving
        metadata.update({
            "intervention_applied": "undifferentiated_attention",
            "intervention_config": HEAD_ABLATION_CONFIG,
            "template_file": args.template_file,
            "template_length": length_suffix
        })

    else:
        print("\n=== Running without Intervention ===")
        component_values = collect_activations_with_hooks(
            model_base, formatted_instructions, direction, args.batch_size
        )
        output_dir = args.output_dir
        final_length_suffix = length_suffix

        # Update metadata for saving
        metadata.update({
            "intervention_applied": False,
            "template_file": args.template_file,
            "template_length": length_suffix
        })

    # Organize and save results
    n_layers = model_base.model.config.num_hidden_layers

    results = organize_results(harmful_instructions, formatted_instructions,
                              component_values, n_layers, metadata)

    save_results(results, output_dir, final_length_suffix)
    
    print(f"\n=== ✅ Analysis Complete ===")
    print(f"Results saved to {output_dir}")

    if args.enable_head_ablation:
        print(f"🎯 Head ablation applied to {len(HEAD_ABLATION_CONFIG)} layers")
    elif args.enable_undifferentiated_attention:
        print(f"🔄 Undifferentiated attention applied to {len(HEAD_ABLATION_CONFIG)} layers")
        print("   Specified heads now use causal uniform attention weights (respecting causality)")

    print("Template-based harmful component calculation finished")


if __name__ == "__main__":
    main()