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

# Attention Head Patching Configuration for Qwen3-14B
# Format: {layer_idx: [head_indices_to_patch]}
# Layers: 0-39, Heads per layer: 0-39


# 60 heads
HEAD_PATCHING_CONFIG = {
    15: [23],
    18: [1, 4, 19],
    19: [12, 14],
    20: [29, 35, 37],
    21: [1, 11, 18, 19],
    22: [17, 24, 26, 29, 35],
    23: [11],
    24: [1, 22, 35, 38],
    25: [36],
    26: [18, 21, 23],
    27: [3, 23, 32, 34],
    28: [4, 21, 30, 32, 35],
    29: [3, 10, 20, 21],
    30: [8, 13],
    31: [9, 13, 16, 19, 24],
    32: [0, 6, 13, 27, 28],
    33: [8, 17, 22, 25],
    34: [8, 36],
    35: [6, 12]
}


# 100 heads
# HEAD_PATCHING_CONFIG = {
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


def setup_model_and_data(model_path: str, cfg: Config) -> Tuple[object, List[str], List[str]]:
    """
    Setup model and load both harmful and harmless datasets.

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

    # Load harmful instructions from evaluation datasets
    print(f"Loading harmful instructions from evaluation datasets: {cfg.evaluation_datasets}")
    harmful_instructions = []
    for dataset_name in cfg.evaluation_datasets:
        dataset = load_dataset(dataset_name)
        instructions = [d['instruction'] for d in dataset]
        harmful_instructions.extend(instructions)
        print(f"  - {dataset_name}: {len(instructions)} instructions")

    print(f"Total loaded: {len(harmful_instructions)} harmful instructions")

    # Load harmless instructions from test split (same as harmless components script)
    print("Loading harmless test dataset...")
    harmless_test = load_dataset_split(harmtype='harmless', split='test', instructions_only=True)
    harmless_instructions = random.sample(harmless_test, min(cfg.n_test, len(harmless_test)))
    print(f"Total loaded: {len(harmless_instructions)} harmless instructions")

    return model_base, harmful_instructions, harmless_instructions


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


def get_qwen3_harmless_attention_collection_hook(layer_idx: int, head_indices: List[int], harmless_attention_cache: Dict, sample_indices: List[int]):
    """
    Create hook to collect attention head outputs from harmless instructions.

    WARNING: This implementation is mathematically INCORRECT!
    It collects attention outputs AFTER the Wo linear transformation and tries to
    separate heads by reshaping, which doesn't work correctly.
    Use get_correct_qwen3_harmless_attention_collection_hook() instead.

    Args:
        layer_idx: Layer index (0-39)
        head_indices: List of head indices to collect (0-39)
        harmless_attention_cache: Cache to store harmless attention outputs
        sample_indices: List of sample indices for current batch

    Returns:
        Hook function for collecting harmless attention head outputs
    """
    def hook_fn(module, input, output):
        # Qwen3 self_attn output: (attention_output, attention_weights, past_key_value)
        if isinstance(output, tuple):
            attn_output = output[0]  # [batch, seq, hidden_size=5120]
        else:
            attn_output = output

        batch_size, seq_len, hidden_size = attn_output.shape
        num_heads = 40
        head_dim = hidden_size // num_heads  # 128

        # Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)

        # Initialize cache structure if needed
        if layer_idx not in harmless_attention_cache:
            harmless_attention_cache[layer_idx] = {}

        # Collect specified attention heads for each sample in batch
        for batch_idx in range(batch_size):
            if batch_idx < len(sample_indices):
                sample_idx = sample_indices[batch_idx]
                if sample_idx not in harmless_attention_cache[layer_idx]:
                    harmless_attention_cache[layer_idx][sample_idx] = {}

                for head_idx in head_indices:
                    if 0 <= head_idx < num_heads:
                        # Store head output: [seq, head_dim] -> move to CPU to save GPU memory
                        head_output = reshaped_output[batch_idx, :, head_idx, :].detach().cpu()
                        harmless_attention_cache[layer_idx][sample_idx][head_idx] = head_output

    return hook_fn


def get_qwen3_attention_patching_hook(layer_idx: int, head_indices: List[int], harmless_attention_cache: Dict, sample_indices: List[int]):
    """
    Create attention head patching hook that replaces harmful attention heads with harmless ones.

    WARNING: This implementation is mathematically INCORRECT!
    It performs patching AFTER the Wo linear transformation, which doesn't properly
    replace individual head outputs. Use get_correct_qwen3_attention_patching_hook() instead.

    Args:
        layer_idx: Layer index (0-39)
        head_indices: List of head indices to patch (0-39)
        harmless_attention_cache: Cache containing harmless attention outputs
        sample_indices: List of sample indices for current batch

    Returns:
        Hook function for attention head patching
    """
    print(f"⚠️  WARNING: Using mathematically incorrect patching for layer {layer_idx}!")
    print("   This patches AFTER Wo transformation, not individual heads.")
    print("   Use --use_correct_patching for mathematically correct implementation.")
    def hook_fn(module, input, output):
        # Qwen3 self_attn output: (attention_output, attention_weights, past_key_value)
        if isinstance(output, tuple):
            attn_output = output[0]  # [batch, seq, hidden_size=5120]
            other_outputs = output[1:]
        else:
            attn_output = output
            other_outputs = ()

        batch_size, seq_len, hidden_size = attn_output.shape
        num_heads = 40
        head_dim = hidden_size // num_heads  # 128

        # Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)

        # Replace specified heads with harmless attention outputs
        if layer_idx in harmless_attention_cache:
            for batch_idx in range(batch_size):
                if batch_idx < len(sample_indices):
                    sample_idx = sample_indices[batch_idx]
                    if sample_idx in harmless_attention_cache[layer_idx]:
                        for head_idx in head_indices:
                            if (0 <= head_idx < num_heads and
                                head_idx in harmless_attention_cache[layer_idx][sample_idx]):
                                # Get harmless attention output and move to current device
                                harmless_head_output = harmless_attention_cache[layer_idx][sample_idx][head_idx]
                                harmless_head_output = harmless_head_output.to(device=attn_output.device, dtype=attn_output.dtype)

                                # Ensure sequence length matches (truncate or pad as needed)
                                if harmless_head_output.shape[0] != seq_len:
                                    if harmless_head_output.shape[0] > seq_len:
                                        harmless_head_output = harmless_head_output[:seq_len, :]
                                    else:
                                        # Pad with zeros if harmless sequence is shorter
                                        pad_length = seq_len - harmless_head_output.shape[0]
                                        padding = torch.zeros(pad_length, head_dim, device=attn_output.device, dtype=attn_output.dtype)
                                        harmless_head_output = torch.cat([harmless_head_output, padding], dim=0)

                                # Replace the head output
                                reshaped_output[batch_idx, :, head_idx, :] = harmless_head_output

        # Reshape back to original format: [batch, seq, hidden_size]
        patched_output = reshaped_output.view(batch_size, seq_len, hidden_size)

        # Return modified output
        if other_outputs:
            return (patched_output, *other_outputs)
        else:
            return patched_output

    return hook_fn


def get_correct_qwen3_harmless_attention_collection_hook(layer_idx: int, head_indices: List[int], harmless_attention_cache: Dict, sample_indices: List[int]):
    """
    Create a monkey patch function that collects attention heads correctly.
    Each layer gets its own independent monkey patch with fixed parameters.

    Args:
        layer_idx: Layer index (0-39)
        head_indices: List of head indices to collect (0-39)
        harmless_attention_cache: Cache to store harmless attention outputs
        sample_indices: List of sample indices for current batch

    Returns:
        Monkey patch function that replaces the forward method
    """
    def create_patched_forward(original_forward):
        def patched_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Handle past_key_value if present
            past_key_value = kwargs.get('past_key_value')
            cache_position = kwargs.get('cache_position')
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Get attention interface and compute attention
            from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            # Collect attention head outputs for THIS SPECIFIC layer only
            # Using closure variables: layer_idx, head_indices, harmless_attention_cache, sample_indices
            batch_size = attn_output.shape[0]

            # Initialize cache structure if needed
            if layer_idx not in harmless_attention_cache:
                harmless_attention_cache[layer_idx] = {}

            # Collect specified attention heads for each sample in batch
            for batch_idx in range(batch_size):
                if batch_idx < len(sample_indices):
                    sample_idx = sample_indices[batch_idx]
                    if sample_idx not in harmless_attention_cache[layer_idx]:
                        harmless_attention_cache[layer_idx][sample_idx] = {}

                    for head_idx in head_indices:
                        if 0 <= head_idx < attn_output.shape[1]:
                            # Store head output: [seq, head_dim] -> move to CPU to save GPU memory
                            head_output = attn_output[batch_idx, head_idx, :, :].detach().cpu()
                            harmless_attention_cache[layer_idx][sample_idx][head_idx] = head_output

            # Continue with original processing
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        return patched_forward

    return create_patched_forward


def get_correct_qwen3_attention_patching_hook(layer_idx: int, head_indices: List[int], harmless_attention_cache: Dict, sample_indices: List[int]):
    """
    Create a monkey patch function that performs attention head patching correctly.
    Each layer gets its own independent monkey patch with fixed parameters.

    Args:
        layer_idx: Layer index (0-39)
        head_indices: List of head indices to patch (0-39)
        harmless_attention_cache: Cache containing harmless attention outputs
        sample_indices: List of sample indices for current batch

    Returns:
        Monkey patch function that replaces the forward method
    """
    def create_patched_forward(original_forward):
        def patched_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Handle past_key_value if present
            past_key_value = kwargs.get('past_key_value')
            cache_position = kwargs.get('cache_position')
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Get attention interface and compute attention
            from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            # Perform attention head patching for THIS SPECIFIC layer only
            # Using closure variables: layer_idx, head_indices, harmless_attention_cache, sample_indices
            if layer_idx in harmless_attention_cache:
                batch_size = attn_output.shape[0]
                seq_len = attn_output.shape[2]
                head_dim = attn_output.shape[3]

                for batch_idx in range(batch_size):
                    if batch_idx < len(sample_indices):
                        sample_idx = sample_indices[batch_idx]
                        if sample_idx in harmless_attention_cache[layer_idx]:
                            for head_idx in head_indices:
                                if (0 <= head_idx < attn_output.shape[1] and
                                    head_idx in harmless_attention_cache[layer_idx][sample_idx]):
                                    # Get harmless attention output and move to current device
                                    harmless_head_output = harmless_attention_cache[layer_idx][sample_idx][head_idx]
                                    harmless_head_output = harmless_head_output.to(device=attn_output.device, dtype=attn_output.dtype)

                                    # Ensure sequence length matches (truncate or pad as needed)
                                    if harmless_head_output.shape[0] != seq_len:
                                        if harmless_head_output.shape[0] > seq_len:
                                            harmless_head_output = harmless_head_output[:seq_len, :]
                                        else:
                                            # Pad with zeros if harmless sequence is shorter
                                            pad_length = seq_len - harmless_head_output.shape[0]
                                            padding = torch.zeros(pad_length, head_dim, device=attn_output.device, dtype=attn_output.dtype)
                                            harmless_head_output = torch.cat([harmless_head_output, padding], dim=0)

                                    # Replace the head output
                                    attn_output[batch_idx, head_idx, :, :] = harmless_head_output

            # Continue with original processing
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        return patched_forward

    return create_patched_forward




def collect_harmless_attention_outputs(model_base, harmless_formatted_instructions: List[str],
                                     head_config: Dict[int, List[int]], batch_size: int = 8) -> Dict:
    """
    Collect attention head outputs from harmless instructions.

    WARNING: This implementation is mathematically INCORRECT!
    It collects attention outputs AFTER the Wo linear transformation and tries to
    separate heads by reshaping, which doesn't work correctly.
    Use collect_correct_harmless_attention_outputs() instead.

    Args:
        model_base: Model instance
        harmless_formatted_instructions: List of formatted harmless instructions
        head_config: Dictionary of layer->heads to collect
        batch_size: Batch size for processing

    Returns:
        Dictionary containing harmless attention outputs: {layer_idx: {sample_idx: {head_idx: tensor}}}
    """
    print("⚠️  WARNING: Using legacy harmless attention collection (AFTER Wo transformation)")
    print("   This is mathematically incorrect! Use --use_correct_patching for correct results.")
    print("Collecting attention outputs from harmless instructions...")
    harmless_attention_cache = {}

    for i in tqdm(range(0, len(harmless_formatted_instructions), batch_size), desc="Collecting harmless attention"):
        batch_instructions = harmless_formatted_instructions[i:i+batch_size]
        batch_sample_indices = list(range(i, i + len(batch_instructions)))

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

        # Create collection hooks for each layer
        collection_hooks = []
        for layer_idx, head_indices in head_config.items():
            if 0 <= layer_idx < len(model_base.model_block_modules):
                attn_module = model_base.model_block_modules[layer_idx].self_attn
                collection_hook = get_qwen3_harmless_attention_collection_hook(
                    layer_idx, head_indices, harmless_attention_cache, batch_sample_indices
                )
                collection_hooks.append((attn_module, collection_hook))

        # Forward pass with collection hooks
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=collection_hooks):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)

        # Clear GPU memory
        torch.cuda.empty_cache()

    print(f"Collected attention outputs for {len(harmless_formatted_instructions)} harmless samples")
    return harmless_attention_cache


def create_attention_patching_hooks(model_base, head_config: Dict[int, List[int]],
                                  harmless_attention_cache: Dict, sample_indices: List[int]):
    """
    Create attention head patching hooks based on configuration.

    Args:
        model_base: Qwen3 model instance
        head_config: {layer_idx: [head_indices_to_patch]}
        harmless_attention_cache: Cache of harmless attention outputs
        sample_indices: List of sample indices for current batch

    Returns:
        List[Tuple[module, hook_fn]]: Hook list for add_hooks
    """
    patching_hooks = []

    for layer_idx, head_indices in head_config.items():
        if 0 <= layer_idx < len(model_base.model_block_modules):
            # Get self_attn module for specified layer
            attn_module = model_base.model_block_modules[layer_idx].self_attn

            # Create patching hook for this layer
            patching_hook = get_qwen3_attention_patching_hook(
                layer_idx, head_indices, harmless_attention_cache, sample_indices
            )

            # Add to hooks list
            patching_hooks.append((attn_module, patching_hook))

    return patching_hooks






def collect_activations_with_attention_patching(model_base, harmful_formatted_instructions: List[str],
                                              harmless_formatted_instructions: List[str],
                                              direction: torch.Tensor, head_config: Dict[int, List[int]],
                                              batch_size: int = 8) -> Dict:
    """
    Collect activations with attention head patching:
    1. First collect harmless attention head outputs
    2. Then process harmful instructions with patched attention heads

    Args:
        model_base: Model instance
        harmful_formatted_instructions: List of formatted harmful instructions
        harmless_formatted_instructions: List of formatted harmless instructions
        direction: Refusal direction tensor
        head_config: Dictionary of layer->heads to patch
        batch_size: Batch size for processing

    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with attention head patching...")

    # Ensure equal length for one-to-one pairing
    min_len = min(len(harmful_formatted_instructions), len(harmless_formatted_instructions))
    harmful_instructions = harmful_formatted_instructions[:min_len]
    harmless_instructions = harmless_formatted_instructions[:min_len]

    print(f"Using {min_len} paired samples (harmful-harmless)")

    # Stage 1: Collect harmless attention outputs
    print("\n=== Stage 1: Collecting harmless attention head outputs ===")
    harmless_attention_cache = collect_harmless_attention_outputs(
        model_base, harmless_instructions, head_config, batch_size
    )

    # Stage 2: Process harmful instructions with patched attention heads
    print("\n=== Stage 2: Processing harmful instructions with patched attention ===")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers

    for i in tqdm(range(0, len(harmful_instructions), batch_size), desc="Processing harmful with patching"):
        batch_instructions = harmful_instructions[i:i+batch_size]
        batch_sample_indices = list(range(i, i + len(batch_instructions)))

        # Tokenize harmful instructions
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

        # Create attention patching hooks
        patching_hooks = create_attention_patching_hooks(
            model_base, head_config, harmless_attention_cache, batch_sample_indices
        )

        # Forward pass with both patching hooks and component collection hooks
        with add_hooks(module_forward_pre_hooks=component_hooks,
                      module_forward_hooks=patching_hooks):
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)

        # Clear GPU memory
        torch.cuda.empty_cache()

    print(f"Completed attention patching for {len(harmful_instructions)} samples")
    return results_cache


def save_patching_metadata(head_config: Dict, output_dir: str, length_suffix: str):
    """Save attention patching configuration metadata."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "intervention_type": "attention_head_patching",
        "model": "Qwen3-14B",
        "total_layers": 40,
        "total_heads_per_layer": 40,
        "patching_config": head_config,
        "patched_layers": list(head_config.keys()),
        "total_patched_heads": sum(len(heads) for heads in head_config.values()),
        "length_suffix": length_suffix
    }

    metadata_path = os.path.join(output_dir, "patching_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved patching metadata to {metadata_path}")


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

    # Attention Head Patching parameters
    parser.add_argument('--enable_attention_patching', action='store_true',
                       help='Enable attention head patching (replace harmful with harmless)')
    parser.add_argument('--patching_output_dir', type=str, default='./results/attention_patching_results',
                       help='Output directory for attention patching results')
    parser.add_argument('--use_correct_patching', action='store_true',
                       help='Use mathematically correct patching (BEFORE o_proj transformation)')

    args = parser.parse_args()
    
    print("Starting template-based harmful component analysis...")

    if args.enable_attention_patching:
        print("🔄 ATTENTION HEAD PATCHING ENABLED")
        print(f"Will patch {len(HEAD_PATCHING_CONFIG)} layers with {sum(len(heads) for heads in HEAD_PATCHING_CONFIG.values())} total heads")
        print("Harmless attention outputs will replace harmful ones")
        if args.use_correct_patching:
            print("✅ Using CORRECT implementation (patching BEFORE o_proj)")
        else:
            print("⚠️  Using LEGACY implementation (patching AFTER o_proj)")
            print("   Add --use_correct_patching for mathematically correct results")

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

    # Setup model and data
    if args.enable_attention_patching:
        # Load both harmful and harmless data for patching
        model_base, harmful_instructions, harmless_instructions = setup_model_and_data(args.model_path, cfg)

        # Limit samples if specified
        if args.n_samples > 0:
            harmful_instructions = harmful_instructions[:args.n_samples]
            harmless_instructions = harmless_instructions[:args.n_samples]

        print(f"Processing {len(harmful_instructions)} harmful samples")
        print(f"Using {len(harmless_instructions)} harmless samples for patching")

        # Format both harmful and harmless instructions with template
        harmful_formatted_instructions = format_instructions_with_template(harmful_instructions, format_thinking_template)
        harmless_formatted_instructions = format_instructions_with_template(harmless_instructions, format_thinking_template)

    else:
        # Load only harmful data for normal processing
        model_base, harmful_instructions = setup_model_and_harmful_data(args.model_path, cfg)

        # Limit samples if specified
        if args.n_samples > 0:
            harmful_instructions = harmful_instructions[:args.n_samples]

        print(f"Processing {len(harmful_instructions)} harmful samples")

        # Format instructions with template
        harmful_formatted_instructions = format_instructions_with_template(harmful_instructions, format_thinking_template)

    # 🔥 Critical branch: Choose processing method
    if args.enable_attention_patching:
        if args.use_correct_patching:
            print("\n=== 🔄 Running with CORRECT Attention Head Patching ===")
            print("✅ Using mathematically correct implementation (patching BEFORE o_proj)")
            component_values = collect_activations_with_correct_attention_patching(
                model_base, harmful_formatted_instructions, harmless_formatted_instructions,
                direction, HEAD_PATCHING_CONFIG, args.batch_size
            )
            intervention_suffix = "correct_attention_patched"
        else:
            print("\n=== 🔄 Running with Legacy Attention Head Patching ===")
            print("⚠️  WARNING: Using legacy implementation (patching AFTER o_proj)")
            print("   This is mathematically incorrect! Use --use_correct_patching for correct results.")
            component_values = collect_activations_with_attention_patching(
                model_base, harmful_formatted_instructions, harmless_formatted_instructions,
                direction, HEAD_PATCHING_CONFIG, args.batch_size
            )
            intervention_suffix = "legacy_attention_patched"

        # Use dedicated output directory and filename
        output_dir = args.patching_output_dir
        final_length_suffix = f"{length_suffix}_{intervention_suffix}"

        # Save patching metadata
        save_patching_metadata(HEAD_PATCHING_CONFIG, output_dir, final_length_suffix)

        # Update metadata for saving
        metadata.update({
            "intervention_applied": "attention_head_patching",
            "patching_implementation": "correct" if args.use_correct_patching else "legacy",
            "patching_config": HEAD_PATCHING_CONFIG,
            "template_file": args.template_file,
            "template_length": length_suffix,
            "harmless_samples_used": len(harmless_formatted_instructions)
        })

    else:
        print("\n=== Running without Intervention ===")
        component_values = collect_activations_with_hooks(
            model_base, harmful_formatted_instructions, direction, args.batch_size
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
    results = organize_results(harmful_instructions, harmful_formatted_instructions,
                              component_values, n_layers, metadata)

    save_results(results, output_dir, final_length_suffix)

    print(f"\n=== ✅ Analysis Complete ===")
    print(f"Results saved to {output_dir}")
    if args.enable_attention_patching:
        print(f"🔄 Attention patching applied to {len(HEAD_PATCHING_CONFIG)} layers")
        print(f"Used {len(harmless_formatted_instructions)} harmless samples for patching")
    print("Template-based harmful component calculation finished")


def collect_correct_harmless_attention_outputs(model_base, harmless_formatted_instructions: List[str],
                                              head_config: Dict[int, List[int]], batch_size: int = 8) -> Dict:
    """
    Collect attention head outputs from harmless instructions using correct implementation.
    This collects attention head outputs BEFORE the o_proj transformation.

    Args:
        model_base: Model instance
        harmless_formatted_instructions: List of formatted harmless instructions
        head_config: Dictionary of layer->heads to collect
        batch_size: Batch size for processing

    Returns:
        Dictionary containing harmless attention outputs: {layer_idx: {sample_idx: {head_idx: tensor}}}
    """
    print("Collecting attention outputs from harmless instructions using CORRECT implementation...")
    harmless_attention_cache = {}

    # Store original forward methods for restoration
    original_forwards = {}
    for layer_idx in head_config.keys():
        if 0 <= layer_idx < len(model_base.model_block_modules):
            attn_module = model_base.model_block_modules[layer_idx].self_attn
            original_forwards[layer_idx] = attn_module.forward

    try:
        for i in tqdm(range(0, len(harmless_formatted_instructions), batch_size), desc="Collecting harmless attention"):
            batch_instructions = harmless_formatted_instructions[i:i+batch_size]
            batch_sample_indices = list(range(i, i + len(batch_instructions)))

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

            # Apply monkey patches for collection (no hooks needed)
            for layer_idx, head_indices in head_config.items():
                if 0 <= layer_idx < len(model_base.model_block_modules):
                    attn_module = model_base.model_block_modules[layer_idx].self_attn
                    patch_forward_fn = get_correct_qwen3_harmless_attention_collection_hook(
                        layer_idx, head_indices, harmless_attention_cache, batch_sample_indices
                    )

                    # Apply monkey patch with proper method binding
                    import types
                    patched_method = patch_forward_fn(original_forwards[layer_idx])
                    attn_module.forward = types.MethodType(patched_method, attn_module)

            # Forward pass with patched attention methods (no additional hooks needed)
            with torch.no_grad():
                _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)

            # Clear GPU memory
            torch.cuda.empty_cache()

    finally:
        # Restore original forward methods
        for layer_idx, original_forward in original_forwards.items():
            if 0 <= layer_idx < len(model_base.model_block_modules):
                attn_module = model_base.model_block_modules[layer_idx].self_attn
                attn_module.forward = original_forward

    print(f"Collected attention outputs for {len(harmless_formatted_instructions)} harmless samples")
    return harmless_attention_cache


def create_correct_attention_patching_hooks(model_base, head_config: Dict[int, List[int]],
                                          harmless_attention_cache: Dict, sample_indices: List[int]):
    """
    Create correct attention head patching using monkey patches.
    This performs patching BEFORE the o_proj transformation.

    Args:
        model_base: Qwen3 model instance
        head_config: {layer_idx: [head_indices_to_patch]}
        harmless_attention_cache: Cache of harmless attention outputs
        sample_indices: List of sample indices for current batch

    Returns:
        Dictionary of original forward methods for restoration
    """
    original_forwards = {}

    for layer_idx, head_indices in head_config.items():
        if 0 <= layer_idx < len(model_base.model_block_modules):
            # Get self_attn module for specified layer
            attn_module = model_base.model_block_modules[layer_idx].self_attn

            # Store original forward method
            original_forwards[layer_idx] = attn_module.forward

            # Create patching function for this layer
            patch_forward_fn = get_correct_qwen3_attention_patching_hook(
                layer_idx, head_indices, harmless_attention_cache, sample_indices
            )

            # Apply monkey patch with proper method binding
            import types
            patched_method = patch_forward_fn(original_forwards[layer_idx])
            attn_module.forward = types.MethodType(patched_method, attn_module)

    return original_forwards


def restore_attention_forward_methods(model_base, original_forwards: Dict):
    """
    Restore original attention forward methods after patching.

    Args:
        model_base: Qwen3 model instance
        original_forwards: Dictionary of layer_idx -> original forward method
    """
    for layer_idx, original_forward in original_forwards.items():
        if 0 <= layer_idx < len(model_base.model_block_modules):
            attn_module = model_base.model_block_modules[layer_idx].self_attn
            attn_module.forward = original_forward


def collect_activations_with_correct_attention_patching(model_base, harmful_formatted_instructions: List[str],
                                                       harmless_formatted_instructions: List[str],
                                                       direction: torch.Tensor, head_config: Dict[int, List[int]],
                                                       batch_size: int = 8) -> Dict:
    """
    Collect activations with CORRECT attention head patching:
    1. First collect harmless attention head outputs (BEFORE o_proj)
    2. Then process harmful instructions with patched attention heads (BEFORE o_proj)

    Args:
        model_base: Model instance
        harmful_formatted_instructions: List of formatted harmful instructions
        harmless_formatted_instructions: List of formatted harmless instructions
        direction: Refusal direction tensor
        head_config: Dictionary of layer->heads to patch
        batch_size: Batch size for processing

    Returns:
        Dictionary with parallel components for each sample and layer
    """
    print("Collecting activations with CORRECT attention head patching...")

    # Ensure equal length for one-to-one pairing
    min_len = min(len(harmful_formatted_instructions), len(harmless_formatted_instructions))
    harmful_instructions = harmful_formatted_instructions[:min_len]
    harmless_instructions = harmless_formatted_instructions[:min_len]

    print(f"Using {min_len} paired samples (harmful-harmless)")

    # Stage 1: Collect harmless attention outputs using CORRECT implementation
    print("\n=== Stage 1: Collecting harmless attention head outputs (CORRECT) ===")
    harmless_attention_cache = collect_correct_harmless_attention_outputs(
        model_base, harmless_instructions, head_config, batch_size
    )

    # Stage 2: Process harmful instructions with patched attention heads
    print("\n=== Stage 2: Processing harmful instructions with CORRECT patched attention ===")
    results_cache = {}
    n_layers = model_base.model.config.num_hidden_layers

    for i in tqdm(range(0, len(harmful_instructions), batch_size), desc="Processing harmful with CORRECT patching"):
        batch_instructions = harmful_instructions[i:i+batch_size]
        batch_sample_indices = list(range(i, i + len(batch_instructions)))

        # Tokenize harmful instructions
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

        # Create CORRECT attention patching hooks
        original_forwards = create_correct_attention_patching_hooks(
            model_base, head_config, harmless_attention_cache, batch_sample_indices
        )

        try:
            # Forward pass with component collection hooks and patched attention methods
            with add_hooks(module_forward_pre_hooks=component_hooks, module_forward_hooks=[]):
                with torch.no_grad():
                    _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            # Restore original forward methods after each batch
            restore_attention_forward_methods(model_base, original_forwards)

        # Clear GPU memory
        torch.cuda.empty_cache()

    print(f"Completed CORRECT attention patching for {len(harmful_instructions)} samples")
    return results_cache


if __name__ == "__main__":
    main()