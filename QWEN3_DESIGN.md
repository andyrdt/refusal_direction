# Qwen3-14B Model Integration Design

## Overview

This document outlines the design for integrating Qwen3-14B into the refusal direction pipeline, addressing the unique requirement for dual-mode chat template support to handle both thinking mode and refusal score evaluation.

## Core Challenge

The integration faces a critical architectural challenge:

1. **get_refusal_scores function requirement**: 
   - Needs single-token forward pass to detect refusal tokens ('I', 'As', etc.)
   - Must use **non-thinking mode** chat template to ensure direct token output
   - Thinking mode would output `<think>` tokens first, interfering with refusal detection

2. **Other pipeline functions requirement**:
   - Functions like `generate_and_save_candidate_directions()` and `generate_and_save_completions_for_dataset()`
   - Should use **thinking mode** chat template for enhanced reasoning quality

## Solution Architecture

### 1. Dual-Mode Chat Template Design

```python
# Thinking mode template (default mode for most functions)
QWEN3_CHAT_TEMPLATE_THINKING = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Non-thinking mode template (specifically for refusal_scores)
QWEN3_CHAT_TEMPLATE_NON_THINKING = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

# System prompt support (optional)
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
```

### 2. Qwen3Model Class Architecture

```python
class Qwen3Model(ModelBase):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        # Create dual tokenize functions
        self.tokenize_instructions_thinking_fn = self._get_tokenize_instructions_thinking_fn()
        self.tokenize_instructions_non_thinking_fn = self._get_tokenize_instructions_non_thinking_fn()
    
    def _get_tokenize_instructions_fn(self):
        # Default returns thinking mode (for generate_completions etc.)
        return self.tokenize_instructions_thinking_fn
    
    def get_tokenize_instructions_for_refusal_scores(self):
        # Specialized non-thinking mode for refusal_scores
        return self.tokenize_instructions_non_thinking_fn
```

### 3. Dynamic Token Configuration

```python
def _get_refusal_toks(self):
    # Dynamically obtain refusal tokens through tokenizer
    refusal_words = ['I', 'As', 'Sorry', 'I\'m']  # Common refusal starters
    refusal_toks = []
    for word in refusal_words:
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if tokens:  # Take first token of each word
            refusal_toks.append(tokens[0])
    return list(set(refusal_toks))  # Remove duplicates

def _get_thinking_tokens(self):
    # Get special thinking tokens
    think_start = self.tokenizer.convert_tokens_to_ids('<think>')
    think_end = self.tokenizer.convert_tokens_to_ids('</think>')
    return think_start, think_end
```

### 4. Template Functions Implementation (Using Explicit Templates)

```python
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
```

### 5. Model Loading Configuration

```python
def _load_model(self, model_path, dtype=torch.float16):
    model_kwargs = {}
    model_kwargs.update({"use_flash_attn": True})
    if dtype != "auto":
        model_kwargs.update({
            "bf16": dtype == torch.bfloat16,
            "fp16": dtype == torch.float16,
            "fp32": dtype == torch.float32,
        })

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        **model_kwargs,
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
    # Use appropriate pad token for Qwen3
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer
```

### 6. Generation Parameters Optimization

```python
# Thinking mode generation config
QWEN3_THINKING_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "do_sample": True,  # Important: DO NOT use greedy decoding
}

# Non-thinking mode generation config  
QWEN3_NON_THINKING_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "do_sample": True,
}
```

## Integration Points

### 1. Model Factory Update

```python
# In model_factory.py
def construct_model_base(model_path: str) -> ModelBase:
    if 'qwen3' in model_path.lower():
        from pipeline.model_utils.qwen3_model import Qwen3Model
        return Qwen3Model(model_path)
    elif 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    # ... rest of conditions
```

### 2. Refusal Score Function Modification

Option A: Modify select_direction.py to accept custom tokenize function:
```python
def get_refusal_scores(model, instructions, tokenize_instructions_fn, refusal_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    # Use the provided tokenize_instructions_fn
    # No changes needed to existing logic
```

Option B: Add method to ModelBase for refusal-specific tokenization:
```python
# In model_base.py (abstract method)
@abstractmethod  
def get_tokenize_instructions_for_refusal_scores(self):
    pass

# In select_direction.py
tokenize_fn = model_base.get_tokenize_instructions_for_refusal_scores()
refusal_scores = get_refusal_scores(model_base.model, instructions, tokenize_fn, model_base.refusal_toks, ...)
```

## Implementation Priority

### Phase 1: Core Implementation
- [ ] Create qwen3_model.py with dual template support
- [ ] Implement dynamic refusal token detection
- [ ] Add model factory recognition for qwen3
- [ ] Basic integration testing

### Phase 2: Optimization  
- [ ] Fine-tune generation parameters
- [ ] Implement thinking token parsing
- [ ] Add comprehensive error handling
- [ ] Performance benchmarking

### Phase 3: Advanced Features
- [ ] Multi-turn conversation support with thinking history
- [ ] Advanced thinking content parsing
- [ ] Integration with evaluation metrics
- [ ] Documentation and examples

## Key Technical Considerations

1. **Compatibility**: Maintain full backward compatibility with existing ModelBase interface
2. **Performance**: Minimize overhead from dual-mode template switching
3. **Robustness**: Handle edge cases in thinking token detection and parsing
4. **Maintainability**: Clear separation between thinking and non-thinking modes
5. **Extensibility**: Design allows for future Qwen model variants

## Testing Strategy

1. **Unit Tests**: Test both template modes independently  
2. **Integration Tests**: Verify refusal_scores works with non-thinking mode
3. **Functional Tests**: Ensure generation quality with thinking mode
4. **Regression Tests**: Confirm no impact on existing model support
5. **Performance Tests**: Benchmark against existing Qwen implementation

This architecture provides a robust, extensible solution that addresses the dual requirements while maintaining clean separation of concerns and full pipeline compatibility.

## Qwen3-14B Model Structure Compatibility

### Architecture Differences from Qwen2

Based on research, Qwen3-14B has the following key architectural characteristics:

1. **Model Configuration**:
   - 40 layers (vs 32 in some Qwen2 variants)
   - 40 attention heads for queries, 8 for keys/values (Grouped Query Attention)
   - 14.8B total parameters (13.2B non-embedding)
   - 128K context window
   - SwiGLU activation, RoPE with enhanced base frequency, RMSNorm

2. **Transformer Structure Compatibility**:
   ```python
   # Qwen3 uses similar structure to Qwen2, but may have different layer naming
   # Need to verify actual model structure once loaded:
   
   # Expected structure (to be confirmed):
   # model.transformer.h[i]          # transformer layers
   # model.transformer.h[i].attn     # attention modules  
   # model.transformer.h[i].mlp      # MLP modules
   # model.transformer.wte           # word embeddings (if present)
   ```

### Model Structure Adaptation Required

```python
def orthogonalize_qwen3_weights(model, direction: Float[Tensor, "d_model"]):
    """
    Orthogonalization function adapted for Qwen3 architecture.
    May need adjustment based on actual model structure.
    """
    # Check if model has embedding layer (some models may not)
    if hasattr(model.transformer, 'wte'):
        model.transformer.wte.weight.data = get_orthogonalized_matrix(
            model.transformer.wte.weight.data, direction
        )
    
    # Adapt to actual attention/MLP layer naming in Qwen3
    for block in model.transformer.h:
        # These names need verification for Qwen3:
        if hasattr(block.attn, 'c_proj'):
            block.attn.c_proj.weight.data = get_orthogonalized_matrix(
                block.attn.c_proj.weight.data.T, direction
            ).T
        elif hasattr(block.attn, 'o_proj'):  # Alternative naming
            block.attn.o_proj.weight.data = get_orthogonalized_matrix(
                block.attn.o_proj.weight.data.T, direction
            ).T
            
        if hasattr(block.mlp, 'c_proj'):
            block.mlp.c_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.c_proj.weight.data.T, direction
            ).T
        elif hasattr(block.mlp, 'down_proj'):  # Alternative naming
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.down_proj.weight.data.T, direction
            ).T

def act_add_qwen3_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    """
    Activation addition function adapted for Qwen3 architecture.
    """
    # Adapt to actual MLP layer naming in Qwen3
    target_layer = model.transformer.h[layer-1]
    
    if hasattr(target_layer.mlp, 'c_proj'):
        dtype = target_layer.mlp.c_proj.weight.dtype
        device = target_layer.mlp.c_proj.weight.device
        bias = (coeff * direction).to(dtype=dtype, device=device)
        target_layer.mlp.c_proj.bias = torch.nn.Parameter(bias)
    elif hasattr(target_layer.mlp, 'down_proj'):
        dtype = target_layer.mlp.down_proj.weight.dtype
        device = target_layer.mlp.down_proj.weight.device
        bias = (coeff * direction).to(dtype=dtype, device=device)
        target_layer.mlp.down_proj.bias = torch.nn.Parameter(bias)
```

### End-of-Instruction Token Adaptation

```python
def _get_eoi_toks(self):
    """
    Get end-of-instruction tokens for Qwen3.
    Should return tokens from the assistant start position.
    """
    # Extract tokens from the template after {instruction}
    template_suffix = QWEN3_CHAT_TEMPLATE_THINKING.split("{instruction}")[-1]
    return self.tokenizer.encode(template_suffix, add_special_tokens=False)
```

### Tokenizer Configuration for Qwen3

```python
def _load_tokenizer(self, model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False  # May need adjustment for Qwen3
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
```

### Model Block Module Access

```python
def _get_model_block_modules(self):
    """Qwen3 should use similar structure to Qwen2"""
    return self.model.transformer.h

def _get_attn_modules(self):
    """Attention modules - may need naming verification"""
    return torch.nn.ModuleList([
        block_module.attn for block_module in self.model_block_modules
    ])
    
def _get_mlp_modules(self):
    """MLP modules - may need naming verification"""  
    return torch.nn.ModuleList([
        block_module.mlp for block_module in self.model_block_modules
    ])
```

### Critical Implementation Notes

1. **Structure Verification**: The exact layer naming (c_proj vs o_proj, down_proj etc.) needs to be verified by inspecting a loaded Qwen3 model
2. **GQA Support**: Qwen3 uses Grouped Query Attention, ensure compatibility with attention module access
3. **Special Tokens**: Verify the actual special token IDs for `<think>`, `</think>`, and other Qwen3-specific tokens
4. **Flash Attention**: Qwen3 supports Flash Attention, ensure model loading parameters are compatible
5. **Context Length**: With 128K context, ensure memory management is appropriate for the pipeline

### Verification Checklist

- [ ] Confirm transformer layer structure (`model.transformer.h`)
- [ ] Verify attention module naming (`attn.c_proj` vs `attn.o_proj`)  
- [ ] Verify MLP module naming (`mlp.c_proj` vs `mlp.down_proj`)
- [ ] Test embedding layer access (`transformer.wte` existence)
- [ ] Validate special token IDs for thinking mode
- [ ] Confirm pad token configuration
- [ ] Test orthogonalization and activation addition functions
- [ ] Verify EOI token extraction from templates