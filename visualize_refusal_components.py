#!/usr/bin/env python3
"""
Visualize refusal direction parallel components analysis results.
This script generates various plots and analysis reports from the calculated results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import argparse
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd


def load_results(results_path: str) -> Dict:
    """
    Load results from JSON or PyTorch format.
    
    Args:
        results_path: Path to results file (.json or .pt)
        
    Returns:
        Results dictionary
    """
    print(f"Loading results from {results_path}")
    
    if results_path.endswith('.json'):
        with open(results_path, 'r') as f:
            results = json.load(f)
    elif results_path.endswith('.pt'):
        # Load PyTorch format and convert to expected structure
        data = torch.load(results_path, map_location='cpu')
        results = {
            'harmful': {
                'parallel_components': data['harmful_components'].tolist(),
            },
            'harmless': {
                'parallel_components': data['harmless_components'].tolist(),
            },
            'metadata': data['metadata']
        }
    else:
        raise ValueError("Results file must be .json or .pt format")
    
    # Convert to numpy arrays for easier processing
    results['harmful']['parallel_components'] = np.array(results['harmful']['parallel_components'])
    results['harmless']['parallel_components'] = np.array(results['harmless']['parallel_components'])
    
    print(f"Loaded data shapes:")
    print(f"  Harmful: {results['harmful']['parallel_components'].shape}")
    print(f"  Harmless: {results['harmless']['parallel_components'].shape}")
    
    return results


def create_layer_comparison_plot(results: Dict, output_path: str):
    """
    Create layer-wise comparison plot showing mean parallel components.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    print("Creating layer comparison plot...")
    
    harmful_components = results['harmful']['parallel_components']  # [n_samples, n_layers]
    harmless_components = results['harmless']['parallel_components']
    n_layers = harmful_components.shape[1]
    
    # Calculate mean and std for each layer
    harmful_means = np.mean(harmful_components, axis=0)
    harmful_stds = np.std(harmful_components, axis=0)
    harmless_means = np.mean(harmless_components, axis=0)
    harmless_stds = np.std(harmless_components, axis=0)
    
    layers = np.arange(n_layers)
    
    plt.figure(figsize=(15, 8))
    
    # Plot lines with error bars
    plt.errorbar(layers, harmful_means, yerr=harmful_stds, 
                label='Harmful Instructions', color='red', alpha=0.7, 
                capsize=3, capthick=1, linewidth=2, marker='o', markersize=4)
    plt.errorbar(layers, harmless_means, yerr=harmless_stds, 
                label='Harmless Instructions', color='blue', alpha=0.7,
                capsize=3, capthick=1, linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Parallel Component Value', fontsize=14)
    plt.title('Layer-wise Parallel Component Analysis\n(Mean ± Std, with Sign)', fontsize=16)
    
    # Add zero line for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for max difference layers
    diff = harmful_means - harmless_means
    max_diff_layer = np.argmax(diff)
    plt.annotate(f'Max Diff: Layer {max_diff_layer}\n({diff[max_diff_layer]:.3f})', 
                xy=(max_diff_layer, harmful_means[max_diff_layer]),
                xytext=(max_diff_layer + n_layers*0.1, harmful_means[max_diff_layer] + 0.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layer comparison plot to {output_path}")


def create_distribution_comparison_plot(results: Dict, output_path: str):
    """
    Create distribution comparison plot using violin plots.
    
    Args:
        results: Results dictionary  
        output_path: Output file path
    """
    print("Creating distribution comparison plot...")
    
    harmful_components = results['harmful']['parallel_components']
    harmless_components = results['harmless']['parallel_components']
    n_layers = harmful_components.shape[1]
    
    # Select key layers for visualization (every 5th layer + first/last)
    key_layers = [0] + list(range(4, n_layers, 5)) + [n_layers-1]
    key_layers = sorted(list(set(key_layers)))  # Remove duplicates and sort
    
    fig, axes = plt.subplots(2, len(key_layers)//2 + len(key_layers)%2, 
                            figsize=(4*len(key_layers)//2 + 4*len(key_layers)%2, 8))
    if len(key_layers) == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, layer in enumerate(key_layers):
        if i >= len(axes):
            break
            
        data_to_plot = [harmful_components[:, layer], harmless_components[:, layer]]
        labels = ['Harmful', 'Harmless']
        colors = ['red', 'blue']
        
        # Create violin plot
        parts = axes[i].violinplot(data_to_plot, positions=[0, 1], widths=0.7, 
                                  showmeans=True, showextrema=True)
        
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        
        axes[i].set_title(f'Layer {layer}', fontsize=12)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(labels)
        axes[i].set_ylabel('Parallel Component Value', fontsize=10)
        
        # Add zero line for reference
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistical test
        harmful_layer = harmful_components[:, layer]
        harmless_layer = harmless_components[:, layer]
        statistic, p_value = stats.mannwhitneyu(harmful_layer, harmless_layer, 
                                               alternative='two-sided')
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        axes[i].text(0.5, max(np.max(harmful_layer), np.max(harmless_layer)) * 0.9, 
                    f'p{significance}', ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Hide empty subplots
    for i in range(len(key_layers), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Distribution Comparison Across Selected Layers', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution comparison plot to {output_path}")


def create_heatmaps(results: Dict, output_dir: str):
    """
    Create heatmaps for harmful and harmless datasets with signed values.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    print("Creating heatmaps...")
    
    harmful_components = results['harmful']['parallel_components']
    harmless_components = results['harmless']['parallel_components']
    
    # Find global min/max for consistent color scaling
    global_max = max(np.max(harmful_components), np.max(harmless_components))
    global_min = min(np.min(harmful_components), np.min(harmless_components))
    vmax = max(abs(global_max), abs(global_min))
    vmin = -vmax
    
    # Create heatmap for harmful instructions
    plt.figure(figsize=(20, 12))
    sns.heatmap(harmful_components, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Parallel Component Value'})
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Sample Index', fontsize=14)
    plt.title('Parallel Components Heatmap - Harmful Instructions\n(Red=Positive, Blue=Negative)', fontsize=16)
    plt.tight_layout()
    harmful_heatmap_path = os.path.join(output_dir, 'heatmap_harmful.png')
    plt.savefig(harmful_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved harmful heatmap to {harmful_heatmap_path}")
    
    # Create heatmap for harmless instructions
    plt.figure(figsize=(20, 12))
    sns.heatmap(harmless_components, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Parallel Component Value'})
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Sample Index', fontsize=14)
    plt.title('Parallel Components Heatmap - Harmless Instructions\n(Red=Positive, Blue=Negative)', fontsize=16)
    plt.tight_layout()
    harmless_heatmap_path = os.path.join(output_dir, 'heatmap_harmless.png')
    plt.savefig(harmless_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved harmless heatmap to {harmless_heatmap_path}")
    
    # Create difference heatmap
    max_samples = min(harmful_components.shape[0], harmless_components.shape[0])
    diff_components = harmful_components[:max_samples] - harmless_components[:max_samples]
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(diff_components, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Difference (Harmful - Harmless)'})
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Sample Index', fontsize=14)
    plt.title('Parallel Components Difference Heatmap\n(Harmful - Harmless)', fontsize=16)
    plt.tight_layout()
    diff_heatmap_path = os.path.join(output_dir, 'heatmap_difference.png')
    plt.savefig(diff_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved difference heatmap to {diff_heatmap_path}")


def create_statistical_analysis_plot(results: Dict, output_path: str):
    """
    Create statistical analysis plot with significance testing.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    print("Creating statistical analysis plot...")
    
    harmful_components = results['harmful']['parallel_components']
    harmless_components = results['harmless']['parallel_components']
    n_layers = harmful_components.shape[1]
    
    # Statistical tests for each layer
    p_values = []
    effect_sizes = []  # Cohen's d
    mean_differences = []
    
    for layer in range(n_layers):
        harmful_layer = harmful_components[:, layer]
        harmless_layer = harmless_components[:, layer]
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(harmful_layer, harmless_layer, 
                                               alternative='two-sided')
        p_values.append(p_value)
        
        # Cohen's d for effect size
        pooled_std = np.sqrt((np.var(harmful_layer) + np.var(harmless_layer)) / 2)
        cohens_d = (np.mean(harmful_layer) - np.mean(harmless_layer)) / (pooled_std + 1e-8)
        effect_sizes.append(cohens_d)
        
        mean_differences.append(np.mean(harmful_layer) - np.mean(harmless_layer))
    
    # Multiple comparison correction (Bonferroni)
    corrected_p_values = [p * n_layers for p in p_values]
    corrected_p_values = [min(p, 1.0) for p in corrected_p_values]  # Cap at 1.0
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean differences
    axes[0,0].bar(range(n_layers), mean_differences, color='purple', alpha=0.7)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,0].set_xlabel('Layer Index')
    axes[0,0].set_ylabel('Mean Difference (Harmful - Harmless)')
    axes[0,0].set_title('Mean Parallel Component Differences by Layer')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Effect sizes (Cohen's d)
    colors = ['red' if abs(d) > 0.8 else 'orange' if abs(d) > 0.5 else 'yellow' if abs(d) > 0.2 else 'gray' 
              for d in effect_sizes]
    axes[0,1].bar(range(n_layers), effect_sizes, color=colors, alpha=0.7)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,1].axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='Small')
    axes[0,1].axhline(y=0.5, color='blue', linestyle=':', alpha=0.7, label='Medium') 
    axes[0,1].axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Large')
    axes[0,1].set_xlabel('Layer Index')
    axes[0,1].set_ylabel("Cohen's d")
    axes[0,1].set_title('Effect Sizes by Layer')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: P-values (log scale)
    log_p_values = [-np.log10(p + 1e-16) for p in p_values]  # Add small epsilon to avoid log(0)
    axes[1,0].bar(range(n_layers), log_p_values, color='green', alpha=0.7)
    axes[1,0].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[1,0].axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
    axes[1,0].axhline(y=-np.log10(0.001), color='purple', linestyle='--', alpha=0.7, label='p=0.001')
    axes[1,0].set_xlabel('Layer Index')
    axes[1,0].set_ylabel('-log10(p-value)')
    axes[1,0].set_title('Statistical Significance by Layer')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Corrected p-values
    log_corrected_p = [-np.log10(p + 1e-16) for p in corrected_p_values]
    axes[1,1].bar(range(n_layers), log_corrected_p, color='brown', alpha=0.7)
    axes[1,1].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[1,1].set_xlabel('Layer Index')
    axes[1,1].set_ylabel('-log10(corrected p-value)')
    axes[1,1].set_title('Bonferroni Corrected P-values')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistical analysis plot to {output_path}")
    
    return {
        'p_values': p_values,
        'corrected_p_values': corrected_p_values,
        'effect_sizes': effect_sizes,
        'mean_differences': mean_differences
    }


def analyze_positive_negative_components(results: Dict) -> Dict:
    """
    Analyze the positive and negative components separately.
    
    Args:
        results: Results dictionary
        
    Returns:
        Dictionary with positive/negative analysis
    """
    harmful_components = results['harmful']['parallel_components']
    harmless_components = results['harmless']['parallel_components']
    
    # Separate positive and negative components
    harmful_pos = harmful_components[harmful_components > 0]
    harmful_neg = harmful_components[harmful_components < 0]
    harmless_pos = harmless_components[harmless_components > 0]
    harmless_neg = harmless_components[harmless_components < 0]
    
    # Calculate proportions
    harmful_pos_ratio = len(harmful_pos) / harmful_components.size
    harmful_neg_ratio = len(harmful_neg) / harmful_components.size
    harmless_pos_ratio = len(harmless_pos) / harmless_components.size
    harmless_neg_ratio = len(harmless_neg) / harmless_components.size
    
    return {
        'harmful': {
            'positive_mean': np.mean(harmful_pos) if len(harmful_pos) > 0 else 0,
            'negative_mean': np.mean(harmful_neg) if len(harmful_neg) > 0 else 0,
            'positive_ratio': harmful_pos_ratio,
            'negative_ratio': harmful_neg_ratio,
            'positive_count': len(harmful_pos),
            'negative_count': len(harmful_neg)
        },
        'harmless': {
            'positive_mean': np.mean(harmless_pos) if len(harmless_pos) > 0 else 0,
            'negative_mean': np.mean(harmless_neg) if len(harmless_neg) > 0 else 0,
            'positive_ratio': harmless_pos_ratio,
            'negative_ratio': harmless_neg_ratio,
            'positive_count': len(harmless_pos),
            'negative_count': len(harmless_neg)
        }
    }


def generate_analysis_report(results: Dict, stats_results: Dict, output_path: str):
    """
    Generate text analysis report with positive/negative component analysis.
    
    Args:
        results: Results dictionary
        stats_results: Statistical analysis results
        output_path: Output file path
    """
    print("Generating analysis report...")
    
    harmful_components = results['harmful']['parallel_components']
    harmless_components = results['harmless']['parallel_components']
    n_layers = harmful_components.shape[1]
    
    # Analyze positive/negative components
    pos_neg_analysis = analyze_positive_negative_components(results)
    
    with open(output_path, 'w') as f:
        f.write("# Refusal Direction Parallel Component Analysis Report\n")
        f.write("## (With Signed Values - Positive/Negative Analysis)\n\n")
        
        # Dataset summary
        f.write("## Dataset Summary\n")
        f.write(f"- Harmful samples: {harmful_components.shape[0]}\n")
        f.write(f"- Harmless samples: {harmless_components.shape[0]}\n")
        f.write(f"- Number of layers: {n_layers}\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n")
        harmful_mean = np.mean(harmful_components)
        harmless_mean = np.mean(harmless_components)
        harmful_std = np.std(harmful_components)
        harmless_std = np.std(harmless_components)
        
        f.write(f"- Harmful instructions - Mean: {harmful_mean:.4f}, Std: {harmful_std:.4f}\n")
        f.write(f"- Harmless instructions - Mean: {harmless_mean:.4f}, Std: {harmless_std:.4f}\n")
        f.write(f"- Overall difference: {harmful_mean - harmless_mean:.4f}\n\n")
        
        # Positive/Negative Component Analysis
        f.write("## Positive/Negative Component Analysis\n")
        f.write("### Harmful Instructions:\n")
        f.write(f"- Positive components: {pos_neg_analysis['harmful']['positive_count']} ({pos_neg_analysis['harmful']['positive_ratio']:.1%})\n")
        f.write(f"  - Mean of positive values: {pos_neg_analysis['harmful']['positive_mean']:.4f}\n")
        f.write(f"- Negative components: {pos_neg_analysis['harmful']['negative_count']} ({pos_neg_analysis['harmful']['negative_ratio']:.1%})\n")
        f.write(f"  - Mean of negative values: {pos_neg_analysis['harmful']['negative_mean']:.4f}\n\n")
        
        f.write("### Harmless Instructions:\n")
        f.write(f"- Positive components: {pos_neg_analysis['harmless']['positive_count']} ({pos_neg_analysis['harmless']['positive_ratio']:.1%})\n")
        f.write(f"  - Mean of positive values: {pos_neg_analysis['harmless']['positive_mean']:.4f}\n")
        f.write(f"- Negative components: {pos_neg_analysis['harmless']['negative_count']} ({pos_neg_analysis['harmless']['negative_ratio']:.1%})\n")
        f.write(f"  - Mean of negative values: {pos_neg_analysis['harmless']['negative_mean']:.4f}\n\n")
        
        # Layer-wise analysis
        f.write("## Layer-wise Analysis\n")
        
        # Find most significant layers
        p_values = stats_results['p_values']
        effect_sizes = stats_results['effect_sizes']
        mean_differences = stats_results['mean_differences']
        
        # Top 5 layers by effect size
        top_effect_layers = np.argsort(np.abs(effect_sizes))[-5:][::-1]
        f.write("### Top 5 Layers by Effect Size:\n")
        for i, layer in enumerate(top_effect_layers):
            f.write(f"{i+1}. Layer {layer}: Cohen's d = {effect_sizes[layer]:.3f}, ")
            f.write(f"p-value = {p_values[layer]:.2e}, mean diff = {mean_differences[layer]:.4f}\n")
        
        f.write("\n### Top 5 Layers by Statistical Significance:\n")
        top_sig_layers = np.argsort(p_values)[:5]
        for i, layer in enumerate(top_sig_layers):
            f.write(f"{i+1}. Layer {layer}: p-value = {p_values[layer]:.2e}, ")
            f.write(f"Cohen's d = {effect_sizes[layer]:.3f}, mean diff = {mean_differences[layer]:.4f}\n")
        
        # Significant layers after correction
        significant_layers = [i for i, p in enumerate(stats_results['corrected_p_values']) if p < 0.05]
        f.write(f"\n### Significant Layers (Bonferroni corrected, p < 0.05): {len(significant_layers)}\n")
        if significant_layers:
            for layer in significant_layers:
                f.write(f"- Layer {layer}: corrected p = {stats_results['corrected_p_values'][layer]:.3f}\n")
        else:
            f.write("- No layers remain significant after multiple comparison correction\n")
        
        # Effect size interpretation
        large_effects = sum(1 for d in effect_sizes if abs(d) > 0.8)
        medium_effects = sum(1 for d in effect_sizes if 0.5 < abs(d) <= 0.8)
        small_effects = sum(1 for d in effect_sizes if 0.2 < abs(d) <= 0.5)
        
        f.write(f"\n## Effect Size Summary\n")
        f.write(f"- Large effects (|d| > 0.8): {large_effects} layers\n")
        f.write(f"- Medium effects (0.5 < |d| <= 0.8): {medium_effects} layers\n")
        f.write(f"- Small effects (0.2 < |d| <= 0.5): {small_effects} layers\n")
        
        # Conclusions with positive/negative interpretation
        f.write(f"\n## Conclusions (Signed Values)\n")
        
        # Mean comparison
        if harmful_mean > harmless_mean:
            f.write("- Harmful instructions show higher average parallel components with the refusal direction\n")
        elif harmful_mean < harmless_mean:
            f.write("- Harmless instructions show higher average parallel components (interesting pattern)\n")
        else:
            f.write("- Harmful and harmless instructions show similar average parallel components\n")
            
        f.write(f"- The strongest differences are observed in layers: {', '.join(map(str, top_effect_layers[:3]))}\n")
        
        # Effect size interpretation  
        if large_effects > 0:
            f.write(f"- {large_effects} layers show large effect sizes, indicating strong differentiation\n")
        elif medium_effects > 0:
            f.write(f"- {medium_effects} layers show medium effect sizes, indicating moderate differentiation\n")
        else:
            f.write("- Most effects are small, indicating limited differentiation between datasets\n")
        
        # Positive/negative interpretation
        f.write(f"\n## Signed Values Interpretation\n")
        f.write("- **Positive values**: Activations aligned with refusal direction (potential refusal signals)\n")
        f.write("- **Negative values**: Activations opposite to refusal direction (potential safety confirmation signals)\n")
        
        harmful_pos_ratio = pos_neg_analysis['harmful']['positive_ratio']
        harmless_pos_ratio = pos_neg_analysis['harmless']['positive_ratio']
        
        if harmful_pos_ratio > harmless_pos_ratio:
            f.write(f"- Harmful instructions have more positive components ({harmful_pos_ratio:.1%} vs {harmless_pos_ratio:.1%})\n")
            f.write("- This suggests harmful instructions more strongly activate refusal-aligned patterns\n")
        else:
            f.write(f"- Harmless instructions have more positive components ({harmless_pos_ratio:.1%} vs {harmful_pos_ratio:.1%})\n")
            f.write("- This suggests a more complex bidirectional safety mechanism\n")
            
        f.write(f"- The model appears to use both positive and negative components for safety assessment\n")
    
    print(f"Saved analysis report to {output_path}")


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(description="Visualize refusal direction parallel components")
    parser.add_argument('--results_path', type=str, 
                       default='./results/refusal_components.json',
                       help='Path to results file (.json or .pt)')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("Starting visualization generation...")
    print(f"Results path: {args.results_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_path)
    
    # Generate all visualizations
    print("\n=== Creating Layer Comparison Plot ===")
    create_layer_comparison_plot(results, os.path.join(args.output_dir, 'layer_comparison.png'))
    
    print("\n=== Creating Distribution Comparison Plot ===")
    create_distribution_comparison_plot(results, os.path.join(args.output_dir, 'distribution_comparison.png'))
    
    print("\n=== Creating Heatmaps ===")
    create_heatmaps(results, args.output_dir)
    
    print("\n=== Creating Statistical Analysis Plot ===")
    stats_results = create_statistical_analysis_plot(results, os.path.join(args.output_dir, 'statistical_analysis.png'))
    
    print("\n=== Generating Analysis Report ===")
    generate_analysis_report(results, stats_results, os.path.join(args.output_dir, 'analysis_report.txt'))
    
    print("\n=== Visualization Complete ===")
    print(f"All visualizations saved to {args.output_dir}")
    print("Generated files:")
    print("- layer_comparison.png")
    print("- distribution_comparison.png")
    print("- heatmap_harmful.png")
    print("- heatmap_harmless.png")
    print("- heatmap_difference.png")
    print("- statistical_analysis.png")
    print("- analysis_report.txt")


if __name__ == "__main__":
    main()