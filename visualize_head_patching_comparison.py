#!/usr/bin/env python3
"""
Visualize comparison between baseline, head-ablated, and attention-patched refusal components
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_data_file(file_path: str, description: str):
    """
    Load a data file and return its components.

    Args:
        file_path: Path to the .pt file
        description: Description of the data for error messages

    Returns:
        Tuple of (parallel_components, metadata) or (None, None) if file doesn't exist
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: {description} file not found at {file_path}")
        return None, None

    try:
        data = torch.load(file_path, map_location='cpu')
        parallel_components = data['parallel_components']
        metadata = data.get('metadata', {})
        print(f"✅ Loaded {description}: {parallel_components.shape}")
        return parallel_components, metadata
    except Exception as e:
        print(f"❌ Error loading {description} from {file_path}: {e}")
        return None, None


def compare_interventions(template_length: str = "1k", results_base_dir: str = "./results"):
    """
    Compare baseline, ablated, and patched refusal components.

    Args:
        template_length: Template length suffix (e.g., "1k", "3k", "11k")
        results_base_dir: Base directory containing results
    """
    print(f"Comparing interventions for template length: {template_length}")
    print(f"Results base directory: {results_base_dir}")

    # Define file paths
    baseline_path = os.path.join(results_base_dir, f'template_harmful_components_{template_length}.pt')
    ablated_path = os.path.join(results_base_dir, 'head_ablation_results', f'template_harmful_components_{template_length}_head_ablated.pt')
    patched_path = os.path.join(results_base_dir, 'attention_patching_results', f'template_harmful_components_{template_length}_attention_patched.pt')

    # Load data files
    baseline_components, baseline_metadata = load_data_file(baseline_path, "Baseline")
    ablated_components, ablated_metadata = load_data_file(ablated_path, "Head Ablated")
    patched_components, patched_metadata = load_data_file(patched_path, "Attention Patched")

    # Check if we have at least baseline data
    if baseline_components is None:
        print("❌ Cannot proceed without baseline data")
        return None

    # Calculate means
    baseline_mean = baseline_components.mean(dim=0)  # [n_layers]
    n_layers = len(baseline_mean)
    layers = list(range(n_layers))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Attention Head Interventions Comparison (Template: {template_length})', fontsize=16, fontweight='bold')

    # Colors and styles
    colors = {'baseline': '#1f77b4', 'ablated': '#ff7f0e', 'patched': '#2ca02c'}

    # Plot 1: All comparisons together
    ax1 = axes[0, 0]
    ax1.plot(layers, baseline_mean, '-', linewidth=2, label='Baseline', alpha=0.8, color=colors['baseline'])

    if ablated_components is not None:
        ablated_mean = ablated_components.mean(dim=0)
        ax1.plot(layers, ablated_mean, '--', linewidth=2, label='Head Ablated', alpha=0.8, color=colors['ablated'])

    if patched_components is not None:
        patched_mean = patched_components.mean(dim=0)
        ax1.plot(layers, patched_mean, '-.', linewidth=2, label='Attention Patched', alpha=0.8, color=colors['patched'])

    # Highlight intervention layers (15-35 based on HEAD_ABLATION_CONFIG)
    intervention_layers = list(range(15, 36))
    for layer in intervention_layers:
        if layer < n_layers:
            ax1.axvline(x=layer, color='gray', alpha=0.2, linestyle=':')

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Parallel Component Magnitude')
    ax1.set_title('All Interventions Overview')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Baseline vs Ablation
    ax2 = axes[0, 1]
    ax2.plot(layers, baseline_mean, '-', linewidth=2, label='Baseline', alpha=0.8, color=colors['baseline'])

    if ablated_components is not None:
        ablated_mean = ablated_components.mean(dim=0)
        ax2.plot(layers, ablated_mean, '--', linewidth=2, label='Head Ablated', alpha=0.8, color=colors['ablated'])

        # Fill difference
        ax2.fill_between(layers, baseline_mean, ablated_mean, alpha=0.2, color='orange', label='Difference')

        # Statistics
        diff_ablation = ablated_mean - baseline_mean
        max_diff_ablation = torch.max(torch.abs(diff_ablation)).item()
        mean_diff_ablation = torch.mean(torch.abs(diff_ablation)).item()
        ax2.text(0.02, 0.98, f'Max |Diff|: {max_diff_ablation:.3f}\nMean |Diff|: {mean_diff_ablation:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for layer in intervention_layers:
        if layer < n_layers:
            ax2.axvline(x=layer, color='gray', alpha=0.2, linestyle=':')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Parallel Component Magnitude')
    ax2.set_title('Baseline vs Head Ablation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Baseline vs Patching
    ax3 = axes[1, 0]
    ax3.plot(layers, baseline_mean, '-', linewidth=2, label='Baseline', alpha=0.8, color=colors['baseline'])

    if patched_components is not None:
        patched_mean = patched_components.mean(dim=0)
        ax3.plot(layers, patched_mean, '-.', linewidth=2, label='Attention Patched', alpha=0.8, color=colors['patched'])

        # Fill difference
        ax3.fill_between(layers, baseline_mean, patched_mean, alpha=0.2, color='purple', label='Difference')

        # Statistics
        diff_patching = patched_mean - baseline_mean
        max_diff_patching = torch.max(torch.abs(diff_patching)).item()
        mean_diff_patching = torch.mean(torch.abs(diff_patching)).item()
        ax3.text(0.02, 0.98, f'Max |Diff|: {max_diff_patching:.3f}\nMean |Diff|: {mean_diff_patching:.3f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for layer in intervention_layers:
        if layer < n_layers:
            ax3.axvline(x=layer, color='gray', alpha=0.2, linestyle=':')

    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Parallel Component Magnitude')
    ax3.set_title('Baseline vs Attention Patching')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Direct comparison of interventions
    ax4 = axes[1, 1]

    if ablated_components is not None and patched_components is not None:
        ablated_mean = ablated_components.mean(dim=0)
        patched_mean = patched_components.mean(dim=0)

        ax4.plot(layers, ablated_mean, '--', linewidth=2, label='Head Ablated', alpha=0.8, color=colors['ablated'])
        ax4.plot(layers, patched_mean, '-.', linewidth=2, label='Attention Patched', alpha=0.8, color=colors['patched'])

        # Fill difference
        ax4.fill_between(layers, ablated_mean, patched_mean, alpha=0.2, color='red', label='Ablation vs Patching')

        # Statistics
        diff_interventions = patched_mean - ablated_mean
        max_diff_interventions = torch.max(torch.abs(diff_interventions)).item()
        mean_diff_interventions = torch.mean(torch.abs(diff_interventions)).item()
        ax4.text(0.02, 0.98, f'Max |Diff|: {max_diff_interventions:.3f}\nMean |Diff|: {mean_diff_interventions:.3f}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax4.set_title('Head Ablation vs Attention Patching')
    else:
        ax4.text(0.5, 0.5, 'Both intervention methods\nneeded for comparison',
                transform=ax4.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Intervention Comparison (Data Missing)')

    for layer in intervention_layers:
        if layer < n_layers:
            ax4.axvline(x=layer, color='gray', alpha=0.2, linestyle=':')

    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Parallel Component Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(results_base_dir, f'head_interventions_comparison_{template_length}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Visualization saved to: {output_path}")

    # Print summary statistics
    print(f"\n📊 Summary Statistics (Template: {template_length}):")
    print(f"Baseline - Mean: {baseline_mean.mean():.4f}, Std: {baseline_mean.std():.4f}")

    if ablated_components is not None:
        ablated_mean = ablated_components.mean(dim=0)
        diff_ablation = ablated_mean - baseline_mean
        print(f"Ablation vs Baseline - Max |Diff|: {torch.max(torch.abs(diff_ablation)):.4f}, Mean |Diff|: {torch.mean(torch.abs(diff_ablation)):.4f}")

    if patched_components is not None:
        patched_mean = patched_components.mean(dim=0)
        diff_patching = patched_mean - baseline_mean
        print(f"Patching vs Baseline - Max |Diff|: {torch.max(torch.abs(diff_patching)):.4f}, Mean |Diff|: {torch.mean(torch.abs(diff_patching)):.4f}")

    if ablated_components is not None and patched_components is not None:
        diff_interventions = patched_mean - ablated_mean
        print(f"Patching vs Ablation - Max |Diff|: {torch.max(torch.abs(diff_interventions)):.4f}, Mean |Diff|: {torch.mean(torch.abs(diff_interventions)):.4f}")

    return {
        'baseline': baseline_mean,
        'ablated': ablated_mean if ablated_components is not None else None,
        'patched': patched_mean if patched_components is not None else None
    }


def load_and_compare():
    """Legacy function for backward compatibility"""
    return compare_interventions(template_length="1k", results_base_dir="./results")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Visualize comparison between baseline, head-ablated, and attention-patched refusal components")
    parser.add_argument('--template_length', type=str, default='1k',
                       help='Template length suffix (e.g., 1k, 3k, 11k)')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Base directory containing results')
    parser.add_argument('--show_legacy', action='store_true',
                       help='Run legacy comparison function')

    args = parser.parse_args()

    if args.show_legacy:
        print("Running legacy comparison function...")
        results = load_and_compare()
    else:
        print(f"Running enhanced comparison for template: {args.template_length}")
        results = compare_interventions(template_length=args.template_length,
                                      results_base_dir=args.results_dir)

    return results


if __name__ == "__main__":
    results = main()