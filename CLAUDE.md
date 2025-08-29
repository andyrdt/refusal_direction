# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase implements the research paper "Refusal in Language Models Is Mediated by a Single Direction". The core hypothesis is that model refusal behaviors can be characterized by a single direction in activation space, which can be extracted, manipulated, and used to control safety mechanisms in language models.

## Core Algorithm & Data Flow

### 1. Direction Extraction (`pipeline/submodules/generate_directions.py`)
**Algorithm**: Compares mean activations between harmful and harmless instructions to extract candidate refusal directions.

**Key Functions**:
- `get_mean_activations()`: Collects mean activations across layers using PyTorch forward hooks
- `get_mean_diff()`: Computes activation differences: `mean_harmful - mean_harmless`
- `generate_directions()`: Main function that extracts candidate directions from multiple positions/layers

**Implementation Details**:
- Uses high-precision (float64) arithmetic to avoid numerical issues
- Extracts directions from end-of-instruction token positions (`eoi_toks`)
- Returns tensor of shape `(n_positions, n_layers, d_model)` representing candidate directions

### 2. Direction Selection (`pipeline/submodules/select_direction.py`)
**Algorithm**: Evaluates candidate directions using three metrics to select the most effective refusal direction.

**Core Functions**:
- `refusal_score()`: Computes log-ratio of refusal vs non-refusal token probabilities
- `select_direction()`: Main selection function using multi-criteria evaluation

**Selection Criteria**:
1. **Ablation Score**: How much removing the direction reduces refusal on harmful instructions (lower is better)
2. **Steering Score**: How much adding the direction increases refusal on harmless instructions (higher is better)  
3. **KL Divergence**: How much the intervention changes overall model behavior (lower is better)

**Filtering Logic**:
```python
def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer):
    # Filter out NaN values, late layers, high KL divergence, poor steering
```

### 3. Activation Intervention (`pipeline/utils/hook_utils.py`)
**Core Technology**: PyTorch forward hooks for real-time activation manipulation during inference.

**Hook Types**:
- `get_direction_ablation_input_pre_hook()`: Removes direction component from activations
  - `activation -= (activation @ direction) * direction`
- `get_activation_addition_input_pre_hook()`: Adds scaled direction to activations
  - `activation += coeff * vector`
- `get_all_direction_ablation_hooks()`: Applies ablation across all transformer layers

**Context Manager**:
```python
with add_hooks(module_forward_pre_hooks, module_forward_hooks):
    # Model forward pass with interventions applied
```

## Architecture Deep Dive

### Model Abstraction Layer (`pipeline/model_utils/`)

**Base Class** (`model_base.py`):
```python
class ModelBase(ABC):
    def __init__(self, model_name_or_path):
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()  # End-of-instruction tokens
        self.refusal_toks = self._get_refusal_toks()  # Refusal indicator tokens
        self.model_block_modules = self._get_model_block_modules()  # Transformer blocks
```

**Model-Specific Implementations**:
- **Llama3** (`llama3_model.py`): 
  - Chat template: `<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|>`
  - Refusal tokens: `[40]` (token for "I")
  - Orthogonalization: `orthogonalize_llama3_weights()` modifies embedding/projection weights
- **Qwen** (`qwen_model.py`):
  - Chat template: `<|im_start|>user\n{instruction}<|im_end|>`
  - Refusal tokens: `[40, 2121]` (tokens for "I", "As")

### Data Management (`dataset/`)

**Data Loading** (`load_dataset.py`):
- `load_dataset_split(harmtype, split)`: Loads train/val/test splits for harmful/harmless data
- `load_dataset(dataset_name)`: Loads specific evaluation datasets (jailbreakbench, advbench, etc.)

**Data Structure**:
```python
{
    "instruction": str,  # The prompt/instruction
    "category": str      # Harm category or dataset identifier  
}
```

### Evaluation Systems

**Jailbreak Evaluation** (`evaluate_jailbreak.py`):
- **Substring Matching**: Checks for refusal phrases ("I'm sorry", "I cannot", etc.)
- **LlamaGuard2**: Uses AI classifier to assess whether response contains harmful content
- **Implementation**: Uses VLLM for efficient batch inference and litellm for API calls

**Loss Evaluation** (`evaluate_loss.py`):
- Measures cross-entropy loss on model's own completions to assess behavioral drift
- Masks loss computation to focus on assistant response tokens only
- Uses custom completions from baseline model as ground truth

### Main Pipeline Execution (`pipeline/run_pipeline.py`)

**Complete Workflow**:
```python
def run_pipeline(model_path):
    # 1. Load model and datasets
    model_base = construct_model_base(model_path)
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets()
    
    # 2. Generate candidate directions
    candidate_directions = generate_directions(model_base, harmful_train, harmless_train)
    
    # 3. Select optimal direction
    pos, layer, direction = select_direction(model_base, harmful_val, harmless_val, candidate_directions)
    
    # 4. Create intervention hooks
    baseline_hooks = []  # No intervention
    ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)  # Remove direction
    actadd_hooks = get_activation_addition_input_pre_hook(direction, coeff=-1.0)  # Add negative direction
    
    # 5. Generate completions with different interventions
    for dataset_name in evaluation_datasets:
        generate_completions(model_base, dataset, baseline_hooks)
        generate_completions(model_base, dataset, ablation_hooks)  
        generate_completions(model_base, dataset, actadd_hooks)
    
    # 6. Evaluate all completions
    evaluate_jailbreak(completions, methodologies=["substring_matching", "llamaguard2"])
    evaluate_loss(model_base, intervention_hooks)
```

## Technical Implementation Details

### Mathematical Operations

**Orthogonalization** (`utils.py`):
```python
def get_orthogonalized_matrix(matrix, vec):
    vec = vec / torch.norm(vec)  # Normalize direction
    proj = einops.einsum(matrix, vec, '... d_model, d_model -> ...') * vec
    return matrix - proj  # Remove direction component
```

**Refusal Score Computation**:
```python
def refusal_score(logits, refusal_toks):
    probs = F.softmax(logits[:, -1, :], dim=-1)  # Last token probabilities
    refusal_prob = probs[:, refusal_toks].sum(dim=-1)
    nonrefusal_prob = 1 - refusal_prob
    return torch.log(refusal_prob + ε) - torch.log(nonrefusal_prob + ε)
```

### Hook Mechanism Details

**Direction Ablation Implementation**:
```python
def get_direction_ablation_input_pre_hook(direction):
    def hook_fn(module, input):
        activation = input[0]  # Shape: (batch, seq, d_model)
        direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        # Project out the direction: activation - (activation · direction) * direction
        activation -= (activation @ direction_norm).unsqueeze(-1) * direction_norm
        return (activation, *input[1:])
    return hook_fn
```

## Artifacts and Results Structure

**Directory Layout**:
```
pipeline/runs/{model_alias}/
├── direction.pt                    # Selected refusal direction tensor
├── direction_metadata.json         # {"pos": int, "layer": int}
├── generate_directions/
│   └── mean_diffs.pt               # All candidate directions
├── select_direction/
│   ├── direction_evaluations.json  # All direction scores
│   ├── ablation_scores.png         # Visualization plots
│   └── actadd_scores.png
├── completions/
│   ├── {dataset}_{intervention}_completions.json
│   └── {dataset}_{intervention}_evaluations.json  
└── loss_evals/
    └── {intervention}_loss_eval.json
```

## Research Applications

This codebase enables:

1. **Safety Mechanism Analysis**: Understanding how refusal behaviors emerge in transformer models
2. **Controlled Safety Bypass**: Temporarily disabling safety mechanisms for research purposes  
3. **Safety Enhancement**: Strengthening refusal behaviors by amplifying refusal directions
4. **Mechanistic Interpretability**: Studying the geometric structure of model behaviors in activation space
5. **Defensive Research**: Developing better understanding of model safety mechanisms and potential vulnerabilities