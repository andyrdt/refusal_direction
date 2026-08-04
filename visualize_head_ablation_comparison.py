#!/usr/bin/env python3
"""
Visualize comparison between baseline and different types of head-ablated refusal components
Generates two separate comparison plots:
1. Baseline vs Predefined Head Ablation
2. Baseline vs Random Head Ablation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_data(data_path: str) -> torch.Tensor:
    """
    Load parallel components data from .pt file.

    Args:
        data_path: Path to the .pt file

    Returns:
        Tensor of parallel components [n_samples, n_layers]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    data = torch.load(data_path, map_location='cpu')
    return data['parallel_components']


def create_comparison_plot(baseline_components: torch.Tensor,
                          ablated_components: torch.Tensor,
                          ablation_type: str,
                          output_path: str,
                          template_length: str = "1k") -> tuple:
    """
    Create comparison plot between baseline and ablated components.

    Args:
        baseline_components: Baseline parallel components [n_samples, n_layers]
        ablated_components: Ablated parallel components [n_samples, n_layers]
        ablation_type: Type of ablation ("predefined" or "random")
        output_path: Path to save the plot
        template_length: Template length suffix (e.g., "1k", "3k")

    Returns:
        Tuple of (baseline_mean, ablated_mean, difference)
    """
    # Calculate mean across samples
    baseline_mean = baseline_components.mean(dim=0)  # [n_layers]
    ablated_mean = ablated_components.mean(dim=0)    # [n_layers]

    n_layers = len(baseline_mean)
    layers = list(range(n_layers))

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot lines
    plt.plot(layers, baseline_mean, 'b-', linewidth=2, label='Baseline', alpha=0.8)

    if ablation_type == "predefined":
        plt.plot(layers, ablated_mean, 'r--', linewidth=2, label='Predefined Head Ablation', alpha=0.8)
        title_suffix = "Predefined Head Ablation"
        color = 'red'
    else:  # random
        plt.plot(layers, ablated_mean, 'g--', linewidth=2, label='Random Head Ablation', alpha=0.8)
        title_suffix = "Random Head Ablation"
        color = 'green'

    # Highlight ablated layers (15-35)
    ablated_layers = list(range(15, 36))
    for layer in ablated_layers:
        if layer < n_layers:
            plt.axvline(x=layer, color='gray', alpha=0.3, linestyle=':')

    # Add difference visualization
    diff = ablated_mean - baseline_mean
    plt.fill_between(layers, baseline_mean, ablated_mean, alpha=0.2, color='orange', label='Difference')

    plt.xlabel('Layer Index', fontsize=28)
    plt.ylabel('Refusal Component Magnitude', fontsize=28)
    plt.title(f'Refusal Components: Baseline vs {title_suffix} ({template_length})', fontsize=24)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    max_diff = torch.max(torch.abs(diff)).item()
    mean_diff = torch.mean(torch.abs(diff)).item()

    # Calculate reduction statistics for ablated layers (15-35)
    ablated_indices = [i for i in range(15, min(36, n_layers))]
    if ablated_indices:
        baseline_ablated_region = baseline_mean[ablated_indices]
        ablated_ablated_region = ablated_mean[ablated_indices]
        region_reduction = torch.mean(baseline_ablated_region - ablated_ablated_region).item()

        stats_text = (f'Max |Diff|: {max_diff:.3f}\n'
                     f'Mean |Diff|: {mean_diff:.3f}\n'
                     f'Mean Reduction (L15-35): {region_reduction:.3f}')
    else:
        stats_text = f'Max |Diff|: {max_diff:.3f}\nMean |Diff|: {mean_diff:.3f}'

    plt.text(0.02, 0.98, stats_text, fontsize=16,
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved to: {output_path}")
    print(f"Max absolute difference: {max_diff:.4f}")
    print(f"Mean absolute difference: {mean_diff:.4f}")

    return baseline_mean, ablated_mean, diff


def main():
    """Main function to generate both comparison plots."""
    parser = argparse.ArgumentParser(description="Visualize head ablation comparisons")
    parser.add_argument('--template_length', type=str, default='1k',
                       help='Template length suffix (e.g., 1k, 3k, 11k)')
    parser.add_argument('--results_dir', type=str,
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/results',
                       help='Base results directory')
    parser.add_argument('--ablation_dir', type=str,
                       default='/root/autodl-tmp/Jianli_work/refusal_direction/results/head_ablation_results',
                       help='Head ablation results directory')

    args = parser.parse_args()

    # Define file paths
    baseline_path = os.path.join(args.results_dir, f'template_harmful_components_{args.template_length}.pt')
    predefined_ablation_path = os.path.join(args.ablation_dir, f'template_harmful_components_{args.template_length}_head_ablated.pt')
    random_ablation_path = os.path.join(args.ablation_dir, f'template_harmful_components_{args.template_length}_random_head_ablated.pt')

    print("=== Head Ablation Comparison Visualization ===")
    print(f"Template length: {args.template_length}")
    print(f"Baseline path: {baseline_path}")
    print(f"Predefined ablation path: {predefined_ablation_path}")
    print(f"Random ablation path: {random_ablation_path}")

    # Load baseline data (required for both comparisons)
    try:
        baseline_components = load_data(baseline_path)
        print(f"✅ Loaded baseline data: {baseline_components.shape}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run the baseline analysis first (without ablation)")
        return

    # 1. Generate baseline vs predefined ablation comparison
    if os.path.exists(predefined_ablation_path):
        print(f"\n--- Generating Baseline vs Predefined Ablation Comparison ---")
        try:
            predefined_components = load_data(predefined_ablation_path)
            print(f"✅ Loaded predefined ablation data: {predefined_components.shape}")

            output_path = os.path.join(args.results_dir, f'baseline_vs_predefined_ablation_{args.template_length}.png')
            baseline_mean, predefined_mean, predefined_diff = create_comparison_plot(
                baseline_components, predefined_components, "predefined", output_path, args.template_length
            )

        except Exception as e:
            print(f"❌ Error processing predefined ablation: {e}")
    else:
        print(f"⚠️ Predefined ablation file not found: {predefined_ablation_path}")
        print("Run with --enable_head_ablation to generate predefined ablation data")

    # 2. Generate baseline vs random ablation comparison
    if os.path.exists(random_ablation_path):
        print(f"\n--- Generating Baseline vs Random Ablation Comparison ---")
        try:
            random_components = load_data(random_ablation_path)
            print(f"✅ Loaded random ablation data: {random_components.shape}")

            output_path = os.path.join(args.results_dir, f'baseline_vs_random_ablation_{args.template_length}.png')
            baseline_mean, random_mean, random_diff = create_comparison_plot(
                baseline_components, random_components, "random", output_path, args.template_length
            )

        except Exception as e:
            print(f"❌ Error processing random ablation: {e}")
    else:
        print(f"⚠️ Random ablation file not found: {random_ablation_path}")
        print("Run with --enable_random_head_ablation to generate random ablation data")

    print(f"\n=== ✅ Visualization Complete ===")
    print("Generated comparison plots:")
    if os.path.exists(predefined_ablation_path):
        print(f"  - Baseline vs Predefined: baseline_vs_predefined_ablation_{args.template_length}.png")
    if os.path.exists(random_ablation_path):
        print(f"  - Baseline vs Random: baseline_vs_random_ablation_{args.template_length}.png")


if __name__ == "__main__":
    main()