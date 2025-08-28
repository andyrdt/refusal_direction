# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase accompanying the paper "Refusal in Language Models Is Mediated by a Single Direction". The project analyzes how language models implement refusal behaviors and provides methods to extract and manipulate "refusal directions" in model activations for defensive security research.

## Setup and Environment

### Initial Setup
```bash
source setup.sh
```

The setup script creates a virtual environment, installs dependencies, and optionally configures HuggingFace and Together AI tokens.

### Python Requirements
- Python 3.10 or higher required
- Uses PyTorch, Transformers, and other ML libraries
- Virtual environment created in `venv/` directory

### Environment Variables
- `HF_TOKEN`: HuggingFace token for accessing gated models
- `TOGETHER_API_KEY`: Together AI token for jailbreak evaluations

## Core Architecture

### Pipeline Structure
The main pipeline (`pipeline/run_pipeline.py`) performs these sequential steps:
1. **Generate Directions**: Extract candidate refusal directions from model activations
2. **Select Direction**: Choose the most effective refusal direction using validation data
3. **Evaluate Completions**: Generate and evaluate model outputs on harmful/harmless prompts
4. **Loss Evaluation**: Analyze cross-entropy loss metrics

### Model Support
Multiple model families supported via factory pattern (`pipeline/model_utils/model_factory.py`):
- Llama 2 and Llama 3 variants
- Qwen models
- Gemma models
- Yi models

Each model has specific tokenization and prompt formatting in their respective model classes.

### Dataset Pipeline
- **Raw datasets**: Stored in `dataset/raw/` (AdvBench, HarmBench, JailbreakBench, etc.)
- **Processed datasets**: JSON format in `dataset/processed/`
- **Data splits**: Train/val/test splits in `dataset/splits/`
- **Loading**: Use `dataset/load_dataset.py` functions

## Running the Pipeline

### Main Command
```bash
python3 -m pipeline.run_pipeline --model_path {huggingface_model_path}
```

Example:
```bash
python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

### Configuration
Pipeline settings configured in `pipeline/config.py`:
- Training/validation/test set sizes
- Evaluation methodologies (substring matching, LlamaGuard2)
- Token limits and batch sizes

### Artifacts
Results saved in `pipeline/runs/{model_alias}/`:
- `direction.pt`: Selected refusal direction tensor
- `direction_metadata.json`: Layer and position metadata
- `generate_directions/`: Candidate directions and mean diffs
- `select_direction/`: Direction selection results and plots
- `completions/`: Generated completions and evaluations
- `loss_evals/`: Cross-entropy loss analysis

## Key Components

### Activation Manipulation
- **Hook utilities** (`pipeline/utils/hook_utils.py`): PyTorch forward hooks for activation intervention
- **Direction ablation**: Remove refusal direction from activations
- **Activation addition**: Add/subtract refusal vectors with coefficients

### Evaluation Methods
- **Jailbreak evaluation** (`pipeline/submodules/evaluate_jailbreak.py`): Safety assessment using multiple methodologies
- **Loss evaluation** (`pipeline/submodules/evaluate_loss.py`): Cross-entropy loss analysis
- **Refusal scoring**: Logit analysis for refusal token probabilities

### Model Abstractions
Base class `ModelBase` provides unified interface:
- Tokenization with model-specific prompt formatting
- Completion generation with hook support
- Refusal token identification per model family

## Development Notes

### Pre-computed Results
Pipeline artifacts included for smaller models in each family:
- `qwen-1_8b-chat`, `gemma-2b-it`, `yi-6b-chat`
- `llama-2-7b-chat-hf`, `meta-llama-3-8b-instruct`

### Data Processing
Use `dataset/generate_datasets.ipynb` for processing raw datasets into required JSON formats.

### Testing and Validation
The pipeline performs extensive validation through:
- Refusal score filtering on training/validation data
- Multiple evaluation methodologies for safety assessment
- Loss evaluation to ensure model coherence

## Research Context

This codebase implements techniques for understanding and manipulating model refusal behaviors for defensive security research. The methods enable analysis of how safety mechanisms work in language models and can be used for improving AI safety and alignment research.