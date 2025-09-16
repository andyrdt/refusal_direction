#!/usr/bin/env python3
"""
Visualize comparison between baseline and head-ablated refusal components
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def load_and_compare():
    # Load data
    baseline_path = "/root/autodl-tmp/Jianli_work/refusal_direction/results/template_harmful_components_1k.pt"
    ablated_path = "/root/autodl-tmp/Jianli_work/refusal_direction/results/head_ablation_results/template_harmful_components_1k_head_ablated.pt"

    baseline_data = torch.load(baseline_path, map_location='cpu')
    ablated_data = torch.load(ablated_path, map_location='cpu')

    # Extract parallel components (correct key name)
    baseline_components = baseline_data['parallel_components']  # [n_samples, n_layers]
    ablated_components = ablated_data['parallel_components']    # [n_samples, n_layers]

    # Calculate mean across samples
    baseline_mean = baseline_components.mean(dim=0)  # [n_layers]
    ablated_mean = ablated_components.mean(dim=0)    # [n_layers]

    n_layers = len(baseline_mean)
    layers = list(range(n_layers))

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot lines
    plt.plot(layers, baseline_mean, 'b-', linewidth=2, label='Baseline', alpha=0.8)
    plt.plot(layers, ablated_mean, 'r--', linewidth=2, label='Head Ablated', alpha=0.8)

    # Highlight ablated layers (15-35)
    ablated_layers = list(range(15, 36))
    for layer in ablated_layers:
        if layer < n_layers:
            plt.axvline(x=layer, color='gray', alpha=0.3, linestyle=':')

    # Add difference visualization
    diff = ablated_mean - baseline_mean
    plt.fill_between(layers, baseline_mean, ablated_mean, alpha=0.2, color='orange', label='Difference')

    plt.xlabel('Layer Index')
    plt.ylabel('Parallel Component Magnitude')
    plt.title('Parallel Components: Baseline vs Head Ablation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics text
    max_diff = torch.max(torch.abs(diff)).item()
    mean_diff = torch.mean(torch.abs(diff)).item()
    plt.text(0.02, 0.98, f'Max |Diff|: {max_diff:.3f}\nMean |Diff|: {mean_diff:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/Jianli_work/refusal_direction/results/head_ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved to: results/head_ablation_comparison.png")
    print(f"Max absolute difference: {max_diff:.4f}")
    print(f"Mean absolute difference: {mean_diff:.4f}")

    return baseline_mean, ablated_mean, diff

if __name__ == "__main__":
    baseline, ablated, difference = load_and_compare()