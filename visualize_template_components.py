#!/usr/bin/env python3
"""
Visualization script for comparing template harmful components across different lengths.
Compares parallel components from different template lengths (3k, 11k, 21k, 31k, 47k).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import pearsonr

def load_all_template_data(results_dir: str = "results") -> Dict[str, Dict]:
    """Load all template component data from JSON files."""
    results_path = Path(results_dir)
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    
    all_data = {}
    
    for length in template_lengths:
        file_path = results_path / f"template_harmful_components_{length}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data[length] = data
                print(f"Loaded {length} template data with {len(data['harmful']['parallel_components'])} samples")
        else:
            print(f"Warning: {file_path} not found")
    
    return all_data

def aggregate_components(all_data: Dict[str, Dict]) -> pd.DataFrame:
    """Aggregate component statistics for each template length."""
    aggregated = []
    
    for length, data in all_data.items():
        components = np.array(data['harmful']['parallel_components'])  # Shape: (n_samples, n_layers)
        
        # Calculate statistics across samples for each layer
        mean_components = np.mean(components, axis=0)
        std_components = np.std(components, axis=0)
        median_components = np.median(components, axis=0)
        
        for layer_idx in range(len(mean_components)):
            aggregated.append({
                'template_length': length,
                'layer': layer_idx,
                'mean_component': mean_components[layer_idx],
                'std_component': std_components[layer_idx],
                'median_component': median_components[layer_idx],
                'all_samples': components[:, layer_idx].tolist()
            })
    
    return pd.DataFrame(aggregated)

def plot_layer_comparison_mean(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "template_layer_comparison_mean.png"):
    """Plot layer-wise comparison of mean components across template lengths."""
    plt.figure(figsize=(15, 8))
    
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    colors = plt.cm.Set1(np.linspace(0, 1, len(template_lengths)))
    
    for i, length in enumerate(template_lengths):
        length_data = df[df['template_length'] == length]
        if len(length_data) > 0:
            plt.plot(length_data['layer'], length_data['mean_component'], 
                    label=f'{length} tokens', color=colors[i], 
                    marker='o', markersize=4, linewidth=2, alpha=0.8)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Parallel Component')
    plt.title('Template Component Comparison Across Layers (Mean Values)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Mean comparison plot saved to {save_path}")

def plot_layer_comparison_median(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "template_layer_comparison_median.png"):
    """Plot layer-wise comparison of median components across template lengths."""
    plt.figure(figsize=(15, 8))
    
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    colors = plt.cm.Set1(np.linspace(0, 1, len(template_lengths)))
    
    for i, length in enumerate(template_lengths):
        length_data = df[df['template_length'] == length]
        if len(length_data) > 0:
            plt.plot(length_data['layer'], length_data['median_component'], 
                    label=f'{length} tokens', color=colors[i], 
                    marker='s', markersize=4, linewidth=2, alpha=0.8)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Median Parallel Component')
    plt.title('Template Component Comparison Across Layers (Median Values)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Median comparison plot saved to {save_path}")

def plot_distribution_comparison(all_data: Dict[str, Dict], save_dir: str = "visualization_results", save_filename: str = "template_distributions.png"):
    """Plot distribution comparison using multiple subplots for detailed layer analysis."""
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    
    # Define layer groups for different subplots
    layer_groups = [
        [5, 10, 15],           # Early layers
        [20, 22, 24],          # Mid-early layers  
        [25, 27, 29],          # Mid layers
        [30, 32, 34],          # Mid-late layers
        [35, 37, 39],          # Late layers
        [28, 31, 33, 36]       # Mixed critical layers
    ]
    
    group_titles = [
        "Early Layers (5, 10, 15)",
        "Mid-Early Layers (20, 22, 24)", 
        "Mid Layers (25, 27, 29)",
        "Mid-Late Layers (30, 32, 34)",
        "Late Layers (35, 37, 39)",
        "Critical Layers (28, 31, 33, 36)"
    ]
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for group_idx, (layers, title) in enumerate(zip(layer_groups, group_titles)):
        # Prepare data for this group
        plot_data = []
        
        for length in template_lengths:
            if length in all_data:
                components = np.array(all_data[length]['harmful']['parallel_components'])
                
                for layer in layers:
                    if layer < components.shape[1]:
                        layer_values = components[:, layer]
                        for value in layer_values:
                            plot_data.append({
                                'template_length': f'{length} tokens',
                                'layer': f'L{layer}',
                                'component_value': float(value)
                            })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create box plot for this group
            sns.boxplot(data=plot_df, x='layer', y='component_value', hue='template_length',
                       palette='Set1', showfliers=True, ax=axes[group_idx])
            
            axes[group_idx].set_title(title, fontsize=12, fontweight='bold')
            axes[group_idx].set_xlabel('Layer', fontsize=10)
            axes[group_idx].set_ylabel('Parallel Component Value', fontsize=10)
            axes[group_idx].grid(True, alpha=0.3)
            axes[group_idx].tick_params(axis='x', rotation=0)
            
            # Only show legend on first subplot
            if group_idx == 0:
                axes[group_idx].legend(title='Template Length', fontsize=8, title_fontsize=9)
            else:
                axes[group_idx].legend().remove()
        else:
            axes[group_idx].text(0.5, 0.5, 'No data available', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[group_idx].transAxes, fontsize=12)
            axes[group_idx].set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Component Value Distributions Across Different Layer Groups', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Distribution comparison plot saved to {save_path}")
    
    # Print summary statistics
    print("\\nLayer group analysis summary:")
    for length in template_lengths:
        if length in all_data:
            components = np.array(all_data[length]['harmful']['parallel_components'])
            print(f"\\n{length} tokens:")
            for group_idx, layers in enumerate(layer_groups):
                group_values = []
                for layer in layers:
                    if layer < components.shape[1]:
                        group_values.extend(components[:, layer].tolist())
                if group_values:
                    mean_val = np.mean(group_values)
                    std_val = np.std(group_values)
                    print(f"  {group_titles[group_idx]}: mean={mean_val:.3f}, std={std_val:.3f}")

def plot_correlation_heatmap(all_data: Dict[str, Dict], save_path: str = "template_component_correlation.png"):
    """Plot correlation heatmap between different template lengths."""
    template_lengths = ["3k", "11k", "21k", "31k", "47k"]
    
    # Calculate layer-wise mean components for each template
    mean_components = {}
    for length in template_lengths:
        if length in all_data:
            components = np.array(all_data[length]['harmful']['parallel_components'])
            mean_components[length] = np.mean(components, axis=0)
    
    # Create correlation matrix
    correlation_matrix = np.zeros((len(mean_components), len(mean_components)))
    labels = list(mean_components.keys())
    
    for i, length1 in enumerate(labels):
        for j, length2 in enumerate(labels):
            if i <= j:  # Only calculate upper triangle
                corr, _ = pearsonr(mean_components[length1], mean_components[length2])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr  # Make symmetric
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, xticklabels=labels, yticklabels=labels,
                mask=mask, cbar_kws={"shrink": .8})
    plt.title('Template Component Correlation Matrix\n(Layer-wise Mean Components)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Correlation heatmap saved to {save_path}")

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for each template length."""
    summary = df.groupby('template_length').agg({
        'mean_component': ['mean', 'std', 'min', 'max'],
        'std_component': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Add overall variation metric (coefficient of variation)
    for length in df['template_length'].unique():
        length_data = df[df['template_length'] == length]
        cv = np.std(length_data['mean_component']) / np.abs(np.mean(length_data['mean_component']))
        summary.loc[length, 'coefficient_of_variation'] = cv
    
    return summary

def plot_summary_statistics(summary_df: pd.DataFrame, save_path: str = "template_component_summary.png"):
    """Plot summary statistics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    template_lengths = summary_df.index.tolist()
    
    # Plot 1: Mean component across templates
    axes[0, 0].bar(template_lengths, summary_df['mean_component_mean'])
    axes[0, 0].errorbar(template_lengths, summary_df['mean_component_mean'], 
                       yerr=summary_df['mean_component_std'], fmt='none', color='red')
    axes[0, 0].set_title('Overall Mean Component by Template Length')
    axes[0, 0].set_ylabel('Mean Component Value')
    
    # Plot 2: Component range (max - min)
    component_range = summary_df['mean_component_max'] - summary_df['mean_component_min']
    axes[0, 1].bar(template_lengths, component_range)
    axes[0, 1].set_title('Component Range by Template Length')
    axes[0, 1].set_ylabel('Range (Max - Min)')
    
    # Plot 3: Average standard deviation
    axes[1, 0].bar(template_lengths, summary_df['std_component_mean'])
    axes[1, 0].set_title('Average Component Std Dev by Template Length')
    axes[1, 0].set_ylabel('Mean Std Dev')
    
    # Plot 4: Coefficient of variation
    axes[1, 1].bar(template_lengths, summary_df['coefficient_of_variation'])
    axes[1, 1].set_title('Component Variability by Template Length')
    axes[1, 1].set_ylabel('Coefficient of Variation')
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary statistics plot saved to {save_path}")

def main():
    """Main visualization pipeline."""
    print("=== Template Component Visualization Pipeline ===")
    
    # Load all data
    print("\n1. Loading template data...")
    all_data = load_all_template_data()
    
    if not all_data:
        print("No template data found!")
        return
    
    # Aggregate components
    print("\n2. Aggregating component statistics...")
    df = aggregate_components(all_data)
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    save_dir = "visualization_results"
    
    plot_layer_comparison_mean(df, save_dir, "template_layer_comparison_mean.png")
    plot_layer_comparison_median(df, save_dir, "template_layer_comparison_median.png") 
    plot_distribution_comparison(all_data, save_dir, "template_distributions.png")
    
    print("\n=== Visualization Complete ===")
    print(f"All files saved to '{save_dir}/' directory:")
    print("- template_layer_comparison_mean.png: Layer-wise component comparison (mean values)")
    print("- template_layer_comparison_median.png: Layer-wise component comparison (median values)")
    print("- template_distributions.png: Component value distributions")

if __name__ == "__main__":
    main()