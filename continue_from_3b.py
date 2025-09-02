#!/usr/bin/env python3
"""
Continue pipeline from step 3b onwards.
This script resumes the pipeline after step 3a (completions generation) has been completed.
"""

import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

# Import functions from run_pipeline.py
from pipeline.run_pipeline import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset,
    evaluate_loss_for_datasets
)

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Continue pipeline from step 3b onwards.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def continue_pipeline_from_3b(model_path):
    """Continue the pipeline from step 3b onwards."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    
    # Check if required artifacts exist
    direction_path = f'{cfg.artifact_path()}/direction.pt'
    metadata_path = f'{cfg.artifact_path()}/direction_metadata.json'
    
    if not os.path.exists(direction_path):
        raise FileNotFoundError(f"Direction file not found: {direction_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(cfg.model_path)
    
    # Load direction and metadata
    print("Loading direction and metadata...")
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    pos, layer = metadata['pos'], metadata['layer']
    print(f"Using direction from position {pos}, layer {layer}")
    
    # Prepare hooks for later steps
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []
    
    print("Starting step 3b: Evaluate completions and save results on harmful evaluation datasets")
    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        print(f"Evaluating {dataset_name} completions...")
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    
    print("Starting step 4a: Generate and save completions on harmless evaluation dataset")
    # 4a. Generate and save completions on harmless evaluation dataset
    random.seed(42)
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
    
    # Generate baseline completions for harmless data
    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
    
    # Generate actadd completions with positive coefficient (increase refusal)
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)
    
    print("Starting step 4b: Evaluate completions and save results on harmless evaluation dataset")
    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    
    print("Starting step 5: Evaluate loss on harmless datasets")
    # 5. Evaluate loss on harmless datasets
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')
    
    print("Pipeline continuation completed successfully!")

if __name__ == "__main__":
    # python3 continue_from_3b.py --model_path Qwen/Qwen3-14B
    args = parse_arguments()
    continue_pipeline_from_3b(model_path=args.model_path)