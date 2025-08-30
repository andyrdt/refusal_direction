import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Qwen3 chat templates using explicit string definitions
QWEN3_CHAT_TEMPLATE_THINKING = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN3_CHAT_TEMPLATE_NON_THINKING = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

QWEN3_CHAT_TEMPLATE_WITH_SYSTEM_THINKING = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN3_CHAT_TEMPLATE_WITH_SYSTEM_NON_THINKING = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

def format_instruction_qwen3_thinking(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        formatted_instruction = QWEN3_CHAT_TEMPLATE_WITH_SYSTEM_THINKING.format(
            instruction=instruction, system=system
        )
    else:
        formatted_instruction = QWEN3_CHAT_TEMPLATE_THINKING.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def format_instruction_qwen3_non_thinking(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        formatted_instruction = QWEN3_CHAT_TEMPLATE_WITH_SYSTEM_NON_THINKING.format(
            instruction=instruction, system=system
        )
    else:
        formatted_instruction = QWEN3_CHAT_TEMPLATE_NON_THINKING.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen3_thinking(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen3_thinking(
                instruction=instruction, output=output, system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen3_thinking(
                instruction=instruction, system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

def tokenize_instructions_qwen3_non_thinking(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen3_non_thinking(
                instruction=instruction, output=output, system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen3_non_thinking(
                instruction=instruction, system=system, 
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

def orthogonalize_qwen3_weights(model, direction: Float[Tensor, "d_model"]):
    """
    Orthogonalization function adapted for Qwen3 architecture.
    Qwen3 uses model.model.layers structure with self_attn and mlp modules.
    """
    # Check if model has embedding layer (some models may not)
    if hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
            model.model.embed_tokens.weight.data, direction
        )
    
    # Qwen3 structure: model.model.layers[i].self_attn / .mlp
    for block in model.model.layers:
        # Handle attention projection layers - Qwen3 uses o_proj
        if hasattr(block.self_attn, 'o_proj'):
            block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                block.self_attn.o_proj.weight.data.T, direction
            ).T
            
        # Handle MLP projection layers - Qwen3 uses down_proj
        if hasattr(block.mlp, 'down_proj'):
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.down_proj.weight.data.T, direction
            ).T

def act_add_qwen3_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    """
    Activation addition function adapted for Qwen3 architecture.
    """
    target_layer = model.model.layers[layer-1]
    
    if hasattr(target_layer.mlp, 'down_proj'):
        dtype = target_layer.mlp.down_proj.weight.dtype
        device = target_layer.mlp.down_proj.weight.device
        bias = (coeff * direction).to(dtype=dtype, device=device)
        target_layer.mlp.down_proj.bias = torch.nn.Parameter(bias)


class Qwen3Model(ModelBase):

    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        # Create dual tokenize functions for thinking/non-thinking modes
        self.tokenize_instructions_thinking_fn = self._get_tokenize_instructions_thinking_fn()
        self.tokenize_instructions_non_thinking_fn = self._get_tokenize_instructions_non_thinking_fn()

    def _load_model(self, model_path, dtype=torch.float16):
        """
        Load Qwen3 model with appropriate parameters.
        Qwen3 uses standard transformers parameters, not Qwen2-specific ones.
        """
        model_kwargs = {}
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto",
                **model_kwargs,
            ).eval()
        except TypeError as e:
            # Fallback for any unsupported parameters
            print(f"Warning: Model loading failed with {e}, trying with minimal parameters...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

        model.requires_grad_(False) 
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'
        
        # Qwen3 specific pad token handling
        if tokenizer.pad_token is None:
            # Qwen3 may use different special tokens
            if hasattr(tokenizer, 'eod_id'):
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.pad_token = tokenizer.decode([tokenizer.eod_id])
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        # Default returns non-thinking mode (for refusal_scores)
        # This is critical for get_refusal_scores to work correctly
        return self._get_tokenize_instructions_non_thinking_fn()

    def _get_tokenize_instructions_thinking_fn(self):
        return functools.partial(
            tokenize_instructions_qwen3_thinking, 
            tokenizer=self.tokenizer, 
            system=None, 
            include_trailing_whitespace=True
        )

    def _get_tokenize_instructions_non_thinking_fn(self):
        return functools.partial(
            tokenize_instructions_qwen3_non_thinking, 
            tokenizer=self.tokenizer, 
            system=None, 
            include_trailing_whitespace=True
        )

    def get_tokenize_instructions_for_refusal_scores(self):
        """Specialized non-thinking mode for refusal_scores"""
        return self.tokenize_instructions_non_thinking_fn
    
    def get_tokenize_instructions_for_generation_tasks(self):
        """Specialized thinking mode for generation tasks like generate_directions"""
        return self.tokenize_instructions_thinking_fn

    def _get_eoi_toks(self):
        """
        Get end-of-instruction tokens for Qwen3.
        Extract tokens from the template after {instruction}
        """
        template_suffix = QWEN3_CHAT_TEMPLATE_THINKING.split("{instruction}")[-1]
        return self.tokenizer.encode(template_suffix, add_special_tokens=False)

    def _get_refusal_toks(self):
        """
        Dynamically obtain refusal tokens through tokenizer.
        Gets first token of common refusal words.
        """
        refusal_words = ['I', 'As', 'Sorry', 'I\'m']  # Common refusal starters
        refusal_toks = []
        for word in refusal_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if tokens:  # Take first token of each word
                refusal_toks.append(tokens[0])
        return list(set(refusal_toks))  # Remove duplicates

    def _get_thinking_tokens(self):
        """Get special thinking tokens"""
        try:
            think_start = self.tokenizer.convert_tokens_to_ids('<think>')
            think_end = self.tokenizer.convert_tokens_to_ids('</think>')
            return think_start, think_end
        except:
            # Fallback if tokens don't exist as special tokens
            think_start_tokens = self.tokenizer.encode('<think>', add_special_tokens=False)
            think_end_tokens = self.tokenizer.encode('</think>', add_special_tokens=False)
            return think_start_tokens[0] if think_start_tokens else None, \
                   think_end_tokens[0] if think_end_tokens else None

    def _get_model_block_modules(self):
        """Qwen3 uses model.model.layers structure"""
        return self.model.model.layers

    def _get_attn_modules(self):
        """Qwen3 attention modules use self_attn naming"""
        return torch.nn.ModuleList([
            block_module.self_attn for block_module in self.model_block_modules
        ])
    
    def _get_mlp_modules(self):
        """Qwen3 MLP modules use standard mlp naming"""  
        return torch.nn.ModuleList([
            block_module.mlp for block_module in self.model_block_modules
        ])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen3_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen3_weights, direction=direction, coeff=coeff, layer=layer)

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        """
        Override generate_completions to use thinking mode for better generation quality.
        Uses model's own generation_config.json parameters.
        """
        from transformers import GenerationConfig
        from tqdm import tqdm
        from pipeline.utils.hook_utils import add_hooks
        import json
        import os
        
        # Load model's own generation config
        model_config_path = os.path.join(self.model_name_or_path, "generation_config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_gen_config = json.load(f)
            
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=model_gen_config.get('do_sample', True),
                temperature=model_gen_config.get('temperature', 0.6),
                top_k=model_gen_config.get('top_k', 20),
                top_p=model_gen_config.get('top_p', 0.95),
            )
        else:
            # Fallback to model's default generation parameters  
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
            )
        
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            # Use thinking mode tokenize function for completions
            tokenized_instructions = self.tokenize_instructions_thinking_fn(instructions=instructions[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        return completions