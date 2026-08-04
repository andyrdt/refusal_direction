"""Utilities for ablating selected Qwen3 attention heads before ``o_proj``."""

from contextlib import contextmanager
from typing import Dict, Iterator, List


# Source: the 60-head experiment configuration previously named
# ``HEAD_PATCHING_CONFIG`` in patching_calculate_harmful_components.py.
QWEN3_14B_60_HEAD_ABLATION_CONFIG: Dict[int, List[int]] = {
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
    35: [6, 12],
}


def validate_qwen3_14b_head_config(config: Dict[int, List[int]]) -> None:
    """Validate Qwen3-14B's 40 layers and 40 query heads per layer."""
    for layer_idx, head_indices in config.items():
        if not 0 <= layer_idx < 40:
            raise ValueError(f"Invalid Qwen3-14B layer index: {layer_idx}")
        for head_idx in head_indices:
            if not 0 <= head_idx < 40:
                raise ValueError(
                    f"Invalid Qwen3-14B head index {head_idx} in layer {layer_idx}"
                )


def _patched_qwen3_attention_forward(original_forward):
    """Return a Qwen3 attention forward which zeros selected head outputs."""
    def patched_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kwargs.get("past_key_value") is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
            key_states, value_states = kwargs["past_key_value"].update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        # Transformers' eager Qwen3 attention returns
        # [batch, sequence, attention_heads, head_dim] at this point.
        for head_idx in self._head_indices_to_ablate:
            attn_output[:, :, head_idx, :] = 0.0

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    return patched_forward


@contextmanager
def qwen3_head_ablation(model_base, config: Dict[int, List[int]]) -> Iterator[None]:
    """Temporarily ablate ``config`` heads and always restore original forwards."""
    validate_qwen3_14b_head_config(config)
    originals = {}
    try:
        for layer_idx, head_indices in config.items():
            if layer_idx >= len(model_base.model_block_modules):
                raise ValueError(f"Model has no layer {layer_idx}")
            attention = model_base.model_block_modules[layer_idx].self_attn
            originals[layer_idx] = attention.forward
            attention._head_indices_to_ablate = list(head_indices)
            attention.forward = _patched_qwen3_attention_forward(attention.forward).__get__(
                attention, type(attention)
            )
        yield
    finally:
        for layer_idx, original_forward in originals.items():
            attention = model_base.model_block_modules[layer_idx].self_attn
            attention.forward = original_forward
            if hasattr(attention, "_head_indices_to_ablate"):
                delattr(attention, "_head_indices_to_ablate")
