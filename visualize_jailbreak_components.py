#!/usr/bin/env python3
"""
Visualization script for jailbreak component analysis.
Compares parallel components between successful and failed jailbreak attempts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

def load_jailbreak_data(results_dir: str = "jailbreak_results") -> Dict[str, Dict]:
    """Load jailbreak component data from JSON files."""
    results_path = Path(results_dir)
    
    data = {}
    
    # Load successful jailbreaks
    successful_path = results_path / "successful_jailbreaks_components.json"
    if successful_path.exists():
        with open(successful_path, 'r') as f:
            data['successful'] = json.load(f)
            n_samples = len(data['successful']['successful_jailbreaks']['parallel_components'])
            print(f"Loaded {n_samples} successful jailbreak samples")
    else:
        print(f"Warning: {successful_path} not found")
    
    # Load failed jailbreaks
    failed_path = results_path / "failed_jailbreaks_components.json"
    if failed_path.exists():
        with open(failed_path, 'r') as f:
            data['failed'] = json.load(f)
            n_samples = len(data['failed']['failed_jailbreaks']['parallel_components'])
            print(f"Loaded {n_samples} failed jailbreak samples")
    else:
        print(f"Warning: {failed_path} not found")
    
    return data

def aggregate_jailbreak_components(data: Dict[str, Dict]) -> pd.DataFrame:
    """Aggregate component statistics for successful vs failed jailbreaks."""
    aggregated = []
    
    for group_type in ['successful', 'failed']:
        if group_type in data:
            group_key = f"{group_type}_jailbreaks"
            components = np.array(data[group_type][group_key]['parallel_components'])
            
            # Calculate statistics across samples for each layer
            mean_components = np.mean(components, axis=0)
            std_components = np.std(components, axis=0)
            median_components = np.median(components, axis=0)
            q25_components = np.percentile(components, 25, axis=0)
            q75_components = np.percentile(components, 75, axis=0)
            
            for layer_idx in range(len(mean_components)):
                aggregated.append({
                    'group': group_type,
                    'layer': layer_idx,
                    'mean_component': mean_components[layer_idx],
                    'std_component': std_components[layer_idx],
                    'median_component': median_components[layer_idx],
                    'q25_component': q25_components[layer_idx],
                    'q75_component': q75_components[layer_idx],
                    'all_samples': components[:, layer_idx].tolist()
                })
    
    return pd.DataFrame(aggregated)

def plot_layer_comparison_mean(df: pd.DataFrame, save_dir: str = "jailbreak_visualization_results", 
                              save_filename: str = "jailbreak_layer_comparison_mean.png"):
    """Plot layer-wise comparison of mean components between successful and failed jailbreaks."""
    plt.figure(figsize=(15, 8))
    
    groups = ['successful', 'failed']
    colors = {'successful': '#e74c3c', 'failed': '#3498db'}  # Red for successful, blue for failed
    
    for group in groups:
        group_data = df[df['group'] == group]
        if len(group_data) > 0:
            plt.plot(group_data['layer'], group_data['mean_component'], 
                    label=f'{group.capitalize()} Jailbreaks', color=colors[group], 
                    marker='o', markersize=4, linewidth=2, alpha=0.8)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Mean Parallel Component', fontsize=12)
    plt.title('Jailbreak Component Comparison Across Layers (Mean Values)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Mean comparison plot saved to {save_path}")

def plot_layer_comparison_with_error(df: pd.DataFrame, save_dir: str = "jailbreak_visualization_results",
                                   save_filename: str = "jailbreak_layer_comparison_with_error.png"):
    """Plot layer-wise comparison with error bars (std dev)."""
    plt.figure(figsize=(15, 8))
    
    groups = ['successful', 'failed']
    colors = {'successful': '#e74c3c', 'failed': '#3498db'}
    
    for group in groups:
        group_data = df[df['group'] == group]
        if len(group_data) > 0:
            plt.errorbar(group_data['layer'], group_data['mean_component'], 
                        yerr=group_data['std_component'],
                        label=f'{group.capitalize()} Jailbreaks', color=colors[group], 
                        marker='o', markersize=4, linewidth=2, alpha=0.8, capsize=3)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Mean Parallel Component ± Std Dev', fontsize=12)
    plt.title('Jailbreak Component Comparison with Error Bars', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Error bar plot saved to {save_path}")

def plot_distribution_comparison(data: Dict[str, Dict], save_dir: str = "jailbreak_visualization_results", 
                               save_filename: str = "jailbreak_distributions.png"):
    """Plot distribution comparison using box plots for detailed layer analysis."""
    
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
        
        for group_type in ['successful', 'failed']:
            if group_type in data:
                group_key = f"{group_type}_jailbreaks"
                components = np.array(data[group_type][group_key]['parallel_components'])
                
                for layer in layers:
                    if layer < components.shape[1]:
                        layer_values = components[:, layer]
                        for value in layer_values:
                            plot_data.append({
                                'group': group_type.capitalize(),
                                'layer': f'L{layer}',
                                'component_value': float(value)
                            })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create box plot for this group
            sns.boxplot(data=plot_df, x='layer', y='component_value', hue='group',
                       palette={'Successful': '#e74c3c', 'Failed': '#3498db'}, 
                       showfliers=True, ax=axes[group_idx])
            
            axes[group_idx].set_title(title, fontsize=12, fontweight='bold')
            axes[group_idx].set_xlabel('Layer', fontsize=10)
            axes[group_idx].set_ylabel('Parallel Component Value', fontsize=10)
            axes[group_idx].grid(True, alpha=0.3)
            axes[group_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[group_idx].tick_params(axis='x', rotation=0)
            
            # Only show legend on first subplot
            if group_idx == 0:
                axes[group_idx].legend(title='Jailbreak Type', fontsize=8, title_fontsize=9)
            else:
                axes[group_idx].legend().remove()
        else:
            axes[group_idx].text(0.5, 0.5, 'No data available', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[group_idx].transAxes, fontsize=12)
            axes[group_idx].set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Component Value Distributions: Successful vs Failed Jailbreaks', 
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

def plot_difference_heatmap(df: pd.DataFrame, save_dir: str = "jailbreak_visualization_results",
                          save_filename: str = "jailbreak_difference_heatmap.png"):
    """Plot heatmap showing difference between successful and failed jailbreaks."""
    
    successful_data = df[df['group'] == 'successful']
    failed_data = df[df['group'] == 'failed']
    
    if len(successful_data) == 0 or len(failed_data) == 0:
        print("Warning: Missing data for one or both groups")
        return
    
    # Calculate differences
    differences = successful_data['mean_component'].values - failed_data['mean_component'].values
    layer_indices = successful_data['layer'].values
    
    # Reshape for heatmap (create 2D representation)
    n_layers = len(layer_indices)
    n_rows = 8  # Number of rows in heatmap
    n_cols = (n_layers + n_rows - 1) // n_rows  # Calculate columns needed
    
    # Pad differences to fill the grid
    padded_differences = np.zeros(n_rows * n_cols)
    padded_differences[:len(differences)] = differences
    diff_matrix = padded_differences.reshape(n_rows, n_cols)
    
    plt.figure(figsize=(15, 8))
    
    # Create custom colormap centered at 0
    vmax = max(abs(differences.min()), abs(differences.max()))
    
    sns.heatmap(diff_matrix, annot=False, cmap='RdBu_r', center=0, 
                vmin=-vmax, vmax=vmax, cbar_kws={'label': 'Component Difference\n(Successful - Failed)'})
    
    plt.title('Layer-wise Component Differences\n(Successful Jailbreaks - Failed Jailbreaks)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer Group', fontsize=12)
    plt.ylabel('Layer Subgroup', fontsize=12)
    
    # Add layer indices as text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            layer_idx = i * n_cols + j
            if layer_idx < len(layer_indices):
                plt.text(j + 0.5, i + 0.5, f'L{layer_indices[layer_idx]}', 
                        ha='center', va='center', fontsize=8, color='black')
    
    plt.tight_layout()
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Difference heatmap saved to {save_path}")

def perform_statistical_analysis(df: pd.DataFrame, save_dir: str = "jailbreak_visualization_results"):
    """Perform statistical analysis comparing successful vs failed jailbreaks."""
    
    successful_data = df[df['group'] == 'successful']
    failed_data = df[df['group'] == 'failed']
    
    if len(successful_data) == 0 or len(failed_data) == 0:
        print("Warning: Missing data for statistical analysis")
        return
    
    print("\n=== Statistical Analysis ===")
    
    # Overall statistics
    print(f"Successful jailbreaks - Mean: {successful_data['mean_component'].mean():.4f}, "
          f"Std: {successful_data['mean_component'].std():.4f}")
    print(f"Failed jailbreaks - Mean: {failed_data['mean_component'].mean():.4f}, "
          f"Std: {failed_data['mean_component'].std():.4f}")
    
    # Layer-wise t-tests
    significant_layers = []
    p_values = []
    
    for layer in successful_data['layer'].values:
        succ_samples = successful_data[successful_data['layer'] == layer]['all_samples'].iloc[0]
        fail_samples = failed_data[failed_data['layer'] == layer]['all_samples'].iloc[0]
        
        # Perform t-test
        t_stat, p_val = ttest_ind(succ_samples, fail_samples)
        p_values.append(p_val)
        
        if p_val < 0.05:
            significant_layers.append(layer)
    
    print(f"\nLayers with significant differences (p < 0.05): {significant_layers}")
    print(f"Total significant layers: {len(significant_layers)}")
    print(f"Minimum p-value: {min(p_values):.6f}")
    
    # Save statistical results
    stats_results = {
        'successful_mean': float(successful_data['mean_component'].mean()),
        'successful_std': float(successful_data['mean_component'].std()),
        'failed_mean': float(failed_data['mean_component'].mean()),
        'failed_std': float(failed_data['mean_component'].std()),
        'significant_layers': [int(layer) for layer in significant_layers],
        'layer_p_values': {int(k): float(v) for k, v in zip(successful_data['layer'].values.tolist(), p_values)},
        'min_p_value': float(min(p_values))
    }
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    stats_path = save_dir_path / "statistical_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    print(f"Statistical analysis results saved to {stats_path}")

def plot_summary_statistics(df: pd.DataFrame, save_dir: str = "jailbreak_visualization_results", 
                           save_filename: str = "jailbreak_summary_stats.png"):
    """Plot summary statistics comparison."""
    
    # Calculate summary stats
    summary_stats = df.groupby('group').agg({
        'mean_component': ['mean', 'std', 'min', 'max'],
        'std_component': ['mean']
    }).round(4)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    groups = ['successful', 'failed']
    colors = {'successful': '#e74c3c', 'failed': '#3498db'}
    
    # Plot 1: Overall mean component
    means = [summary_stats.loc[group, ('mean_component', 'mean')] for group in groups]
    stds = [summary_stats.loc[group, ('mean_component', 'std')] for group in groups]
    
    bars = axes[0, 0].bar(groups, means, color=[colors[g] for g in groups])
    axes[0, 0].errorbar(groups, means, yerr=stds, fmt='none', color='black', capsize=5)
    axes[0, 0].set_title('Overall Mean Component by Group')
    axes[0, 0].set_ylabel('Mean Component Value')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Component range (max - min)
    ranges = [summary_stats.loc[group, ('mean_component', 'max')] - 
              summary_stats.loc[group, ('mean_component', 'min')] for group in groups]
    
    axes[0, 1].bar(groups, ranges, color=[colors[g] for g in groups])
    axes[0, 1].set_title('Component Range by Group')
    axes[0, 1].set_ylabel('Range (Max - Min)')
    
    # Plot 3: Average standard deviation
    avg_stds = [summary_stats.loc[group, ('std_component', 'mean')] for group in groups]
    
    axes[1, 0].bar(groups, avg_stds, color=[colors[g] for g in groups])
    axes[1, 0].set_title('Average Component Std Dev by Group')
    axes[1, 0].set_ylabel('Mean Std Dev')
    
    # Plot 4: Min and Max values
    mins = [summary_stats.loc[group, ('mean_component', 'min')] for group in groups]
    maxs = [summary_stats.loc[group, ('mean_component', 'max')] for group in groups]
    
    x = np.arange(len(groups))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, mins, width, label='Min', color=[colors[g] for g in groups], alpha=0.7)
    axes[1, 1].bar(x + width/2, maxs, width, label='Max', color=[colors[g] for g in groups])
    axes[1, 1].set_title('Min/Max Component Values by Group')
    axes[1, 1].set_ylabel('Component Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(groups)
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary statistics plot saved to {save_path}")

def main():
    """Main visualization pipeline."""
    print("=== Jailbreak Component Visualization Pipeline ===")
    
    # Load data
    print("\n1. Loading jailbreak data...")
    data = load_jailbreak_data()
    
    if not data:
        print("No jailbreak data found!")
        return
    
    # Aggregate components
    print("\n2. Aggregating component statistics...")
    df = aggregate_jailbreak_components(data)
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    save_dir = "jailbreak_visualization_results"
    
    plot_layer_comparison_mean(df, save_dir, "jailbreak_layer_comparison_mean.png")
    plot_layer_comparison_with_error(df, save_dir, "jailbreak_layer_comparison_with_error.png")
    plot_distribution_comparison(data, save_dir, "jailbreak_distributions.png")
    plot_difference_heatmap(df, save_dir, "jailbreak_difference_heatmap.png")
    plot_summary_statistics(df, save_dir, "jailbreak_summary_stats.png")
    
    # Statistical analysis
    print("\n4. Performing statistical analysis...")
    perform_statistical_analysis(df, save_dir)
    
    print("\n=== Visualization Complete ===")
    print(f"All files saved to '{save_dir}/' directory:")
    print("- jailbreak_layer_comparison_mean.png: Layer-wise component comparison (mean values)")
    print("- jailbreak_layer_comparison_with_error.png: Layer-wise comparison with error bars")
    print("- jailbreak_distributions.png: Component value distributions by layer groups")
    print("- jailbreak_difference_heatmap.png: Heatmap of differences between groups")
    print("- jailbreak_summary_stats.png: Summary statistics comparison")
    print("- statistical_analysis.json: Detailed statistical analysis results")

if __name__ == "__main__":
    main()