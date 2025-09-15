#!/usr/bin/env python3
"""
Enhanced visualization script for comparing template harmful, harmless, and stealth harmful components across different lengths.
Compares parallel components from different template lengths (1k, 3k, 11k, 21k, 31k, 47k) for harmful, harmless, and stealth harmful data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import pearsonr

def load_all_template_data(results_dir: str = "results", stealth_results_dir: str = "stealth_results") -> Dict[str, Dict]:
    """Load harmful, harmless, and stealth harmful template component data from JSON files."""
    results_path = Path(results_dir)
    stealth_results_path = Path(stealth_results_dir)
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    
    all_data = {}
    
    for length in template_lengths:
        data_entry = {}
        
        # Load harmful components
        harmful_file = results_path / f"template_harmful_components_{length}.json"
        if harmful_file.exists():
            with open(harmful_file, 'r') as f:
                harmful_data = json.load(f)
                data_entry['harmful'] = harmful_data['harmful']
                print(f"Loaded {length} harmful template data with {len(harmful_data['harmful']['parallel_components'])} samples")
        else:
            print(f"Warning: {harmful_file} not found")
            
        # Load harmless components
        harmless_file = results_path / f"template_harmless_components_{length}.json"
        if harmless_file.exists():
            with open(harmless_file, 'r') as f:
                harmless_data = json.load(f)
                data_entry['harmless'] = harmless_data['harmless']
                print(f"Loaded {length} harmless template data with {len(harmless_data['harmless']['parallel_components'])} samples")
        else:
            print(f"Warning: {harmless_file} not found")
            
        # Load stealth harmful components
        stealth_harmful_file = stealth_results_path / f"stealth_harmful_components_{length}.json"
        if stealth_harmful_file.exists():
            with open(stealth_harmful_file, 'r') as f:
                stealth_harmful_data = json.load(f)
                data_entry['stealth_harmful'] = stealth_harmful_data['stealth_harmful']
                print(f"Loaded {length} stealth harmful template data with {len(stealth_harmful_data['stealth_harmful']['parallel_components'])} samples")
        else:
            print(f"Warning: {stealth_harmful_file} not found")
            
        if data_entry:
            all_data[length] = data_entry
    
    return all_data

def aggregate_components(all_data: Dict[str, Dict]) -> pd.DataFrame:
    """Aggregate component statistics for each template length and data type."""
    aggregated = []
    
    for length, data in all_data.items():
        for data_type in ['harmful', 'harmless', 'stealth_harmful']:
            if data_type in data:
                components = np.array(data[data_type]['parallel_components'])  # Shape: (n_samples, n_layers)
                
                # Calculate statistics across samples for each layer
                mean_components = np.mean(components, axis=0)
                std_components = np.std(components, axis=0)
                median_components = np.median(components, axis=0)
                
                for layer_idx in range(len(mean_components)):
                    aggregated.append({
                        'template_length': length,
                        'data_type': data_type,
                        'layer': layer_idx,
                        'mean_component': mean_components[layer_idx],
                        'std_component': std_components[layer_idx],
                        'median_component': median_components[layer_idx],
                        'all_samples': components[:, layer_idx].tolist()
                    })
    
    return pd.DataFrame(aggregated)

def plot_combined_layer_comparison_mean(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "combined_layer_comparison_mean.png"):
    """Plot layer-wise comparison of mean components for harmful, harmless, and stealth harmful across template lengths."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 8))
    
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    colors = plt.cm.Set1(np.linspace(0, 1, len(template_lengths)))
    
    # Plot harmful components
    for i, length in enumerate(template_lengths):
        harmful_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmful')]
        if len(harmful_data) > 0:
            ax1.plot(harmful_data['layer'], harmful_data['mean_component'], 
                    label=f'{length} tokens', color=colors[i], 
                    marker='o', markersize=4, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Mean Parallel Component')
    ax1.set_title('Harmful Template Component Comparison Across Layers (Mean Values)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot harmless components
    for i, length in enumerate(template_lengths):
        harmless_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmless')]
        if len(harmless_data) > 0:
            ax2.plot(harmless_data['layer'], harmless_data['mean_component'], 
                    label=f'{length} tokens', color=colors[i], 
                    marker='s', markersize=4, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Parallel Component')
    ax2.set_title('Harmless Template Component Comparison Across Layers (Mean Values)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot stealth harmful components
    for i, length in enumerate(template_lengths):
        stealth_data = df[(df['template_length'] == length) & (df['data_type'] == 'stealth_harmful')]
        if len(stealth_data) > 0:
            ax3.plot(stealth_data['layer'], stealth_data['mean_component'], 
                    label=f'{length} tokens', color=colors[i], 
                    marker='^', markersize=4, linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Mean Parallel Component')
    ax3.set_title('Stealth Harmful Template Component Comparison Across Layers (Mean Values)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Combined mean comparison plot saved to {save_path}")

def plot_overlaid_layer_comparison(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "overlaid_layer_comparison.png"):
    """Plot overlaid layer-wise comparison with harmful, harmless, and stealth harmful on the same plot."""
    plt.figure(figsize=(20, 12))
    
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    colors = plt.cm.Set1(np.linspace(0, 1, len(template_lengths)))
    
    # Plot harmful components (solid lines)
    for i, length in enumerate(template_lengths):
        harmful_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmful')]
        if len(harmful_data) > 0:
            plt.plot(harmful_data['layer'], harmful_data['mean_component'], 
                    label=f'Harmful {length}', color=colors[i], 
                    marker='o', markersize=4, linewidth=2.5, alpha=0.8, linestyle='-')
    
    # Plot harmless components (dashed lines)
    for i, length in enumerate(template_lengths):
        harmless_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmless')]
        if len(harmless_data) > 0:
            plt.plot(harmless_data['layer'], harmless_data['mean_component'], 
                    label=f'Harmless {length}', color=colors[i], 
                    marker='s', markersize=4, linewidth=2.5, alpha=0.7, linestyle='--')
    
    # Plot stealth harmful components (dotted lines)
    for i, length in enumerate(template_lengths):
        stealth_data = df[(df['template_length'] == length) & (df['data_type'] == 'stealth_harmful')]
        if len(stealth_data) > 0:
            plt.plot(stealth_data['layer'], stealth_data['mean_component'], 
                    label=f'Stealth {length}', color=colors[i], 
                    marker='^', markersize=4, linewidth=2.5, alpha=0.7, linestyle=':')
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Mean Parallel Component', fontsize=12)
    plt.title('Harmful vs Harmless vs Stealth Harmful Template Component Comparison Across Layers', fontsize=14, fontweight='bold')
    
    # Create unified legend on the left side
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Split into harmful, harmless, and stealth, then combine for vertical arrangement
    harmful_handles = [h for h, l in zip(handles, labels) if 'Harmful' in l and 'Stealth' not in l]
    harmless_handles = [h for h, l in zip(handles, labels) if 'Harmless' in l]
    stealth_handles = [h for h, l in zip(handles, labels) if 'Stealth' in l]
    harmful_labels = [l for l in labels if 'Harmful' in l and 'Stealth' not in l]
    harmless_labels = [l for l in labels if 'Harmless' in l]
    stealth_labels = [l for l in labels if 'Stealth' in l]
    
    # Combine all handles and labels in vertical order
    all_handles = harmful_handles + harmless_handles + stealth_handles
    all_labels = harmful_labels + harmless_labels + stealth_labels
    
    # Create single legend positioned on the left side with better formatting
    plt.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(0.02, 0.5), ncol=1, 
              fontsize=10, markerscale=1.5, handlelength=4.0, handletextpad=1.0, columnspacing=2.0)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overlaid comparison plot saved to {save_path}")

def plot_difference_heatmap(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "harmful_harmless_difference_heatmap.png"):
    """Plot heatmap showing difference between harmful and harmless components."""
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    
    # Get unique layer indices
    max_layers = df['layer'].max() + 1
    
    # Create difference matrix
    difference_matrix = np.full((len(template_lengths), max_layers), np.nan)
    
    for i, length in enumerate(template_lengths):
        harmful_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmful')]
        harmless_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmless')]
        
        # Merge data on layer index to compute differences
        if len(harmful_data) > 0 and len(harmless_data) > 0:
            merged = pd.merge(harmful_data[['layer', 'mean_component']], 
                            harmless_data[['layer', 'mean_component']], 
                            on='layer', suffixes=('_harmful', '_harmless'))
            
            for _, row in merged.iterrows():
                layer_idx = int(row['layer'])
                diff = row['mean_component_harmful'] - row['mean_component_harmless']
                difference_matrix[i, layer_idx] = diff
    
    # Create heatmap
    plt.figure(figsize=(16, 8))
    
    mask = np.isnan(difference_matrix)
    
    sns.heatmap(difference_matrix, 
                xticklabels=range(max_layers),
                yticklabels=template_lengths,
                cmap='RdBu_r', center=0, 
                mask=mask,
                annot=False, fmt='.3f',
                cbar_kws={'label': 'Harmful - Harmless Component'})
    
    plt.title('Difference Between Harmful and Harmless Components Across Templates and Layers', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Template Length', fontsize=12)
    
    plt.tight_layout()
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Difference heatmap saved to {save_path}")

def plot_stealth_difference_heatmap(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "stealth_harmful_harmless_difference_heatmap.png"):
    """Plot heatmap showing difference between stealth harmful and harmless components."""
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    
    # Get unique layer indices
    max_layers = df['layer'].max() + 1
    
    # Create difference matrix
    difference_matrix = np.full((len(template_lengths), max_layers), np.nan)
    
    for i, length in enumerate(template_lengths):
        stealth_data = df[(df['template_length'] == length) & (df['data_type'] == 'stealth_harmful')]
        harmless_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmless')]
        
        # Merge data on layer index to compute differences
        if len(stealth_data) > 0 and len(harmless_data) > 0:
            merged = pd.merge(stealth_data[['layer', 'mean_component']], 
                            harmless_data[['layer', 'mean_component']], 
                            on='layer', suffixes=('_stealth', '_harmless'))
            
            for _, row in merged.iterrows():
                layer_idx = int(row['layer'])
                diff = row['mean_component_stealth'] - row['mean_component_harmless']
                difference_matrix[i, layer_idx] = diff
    
    # Create heatmap
    plt.figure(figsize=(16, 8))
    
    mask = np.isnan(difference_matrix)
    
    sns.heatmap(difference_matrix, 
                xticklabels=range(max_layers),
                yticklabels=template_lengths,
                cmap='RdBu_r', center=0, 
                mask=mask,
                annot=False, fmt='.3f',
                cbar_kws={'label': 'Stealth Harmful - Harmless Component'})
    
    plt.title('Difference Between Stealth Harmful and Harmless Components Across Templates and Layers', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Template Length', fontsize=12)
    
    plt.tight_layout()
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Stealth difference heatmap saved to {save_path}")

def plot_harmful_vs_stealth_heatmap(df: pd.DataFrame, save_dir: str = "visualization_results", save_filename: str = "harmful_vs_stealth_difference_heatmap.png"):
    """Plot heatmap showing difference between harmful and stealth harmful components."""
    template_lengths = ["1k", "3k", "11k", "21k", "31k", "47k"]
    
    # Get unique layer indices
    max_layers = df['layer'].max() + 1
    
    # Create difference matrix
    difference_matrix = np.full((len(template_lengths), max_layers), np.nan)
    
    for i, length in enumerate(template_lengths):
        harmful_data = df[(df['template_length'] == length) & (df['data_type'] == 'harmful')]
        stealth_data = df[(df['template_length'] == length) & (df['data_type'] == 'stealth_harmful')]
        
        # Merge data on layer index to compute differences
        if len(harmful_data) > 0 and len(stealth_data) > 0:
            merged = pd.merge(harmful_data[['layer', 'mean_component']], 
                            stealth_data[['layer', 'mean_component']], 
                            on='layer', suffixes=('_harmful', '_stealth'))
            
            for _, row in merged.iterrows():
                layer_idx = int(row['layer'])
                diff = row['mean_component_stealth'] - row['mean_component_harmful']
                difference_matrix[i, layer_idx] = diff
    
    # Create heatmap
    plt.figure(figsize=(16, 8))
    
    mask = np.isnan(difference_matrix)
    
    sns.heatmap(difference_matrix, 
                xticklabels=range(max_layers),
                yticklabels=template_lengths,
                cmap='RdBu_r', center=0, 
                mask=mask,
                annot=False, fmt='.3f',
                cbar_kws={'label': 'Stealth Harmful - Harmful Component'})
    
    plt.title('Difference Between Stealth Harmful and Harmful Components Across Templates and Layers', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Template Length', fontsize=12)
    
    plt.tight_layout()
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True)
    
    save_path = save_dir_path / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Harmful vs stealth difference heatmap saved to {save_path}")


def main():
    """Main enhanced visualization pipeline."""
    print("=== Enhanced Template Component Visualization Pipeline ===")
    
    # Load all data
    print("\n1. Loading template data (harmful and harmless)...")
    all_data = load_all_template_data()
    
    if not all_data:
        print("No template data found!")
        return
    
    # Aggregate components
    print("\n2. Aggregating component statistics...")
    df = aggregate_components(all_data)
    
    # Generate visualizations
    print("\n3. Generating enhanced visualizations...")
    save_dir = "visualization_results"
    
    # Generate core visualizations
    plot_combined_layer_comparison_mean(df, save_dir, "combined_layer_comparison_mean.png")
    plot_overlaid_layer_comparison(df, save_dir, "overlaid_layer_comparison.png")
    plot_difference_heatmap(df, save_dir, "harmful_harmless_difference_heatmap.png")
    plot_stealth_difference_heatmap(df, save_dir, "stealth_harmful_harmless_difference_heatmap.png")
    plot_harmful_vs_stealth_heatmap(df, save_dir, "harmful_vs_stealth_difference_heatmap.png")
    
    print("\n=== Enhanced Visualization Complete ===")
    print(f"All files saved to '{save_dir}/' directory:")
    print("- combined_layer_comparison_mean.png: Side-by-side harmful vs harmless vs stealth harmful layer comparison")
    print("- overlaid_layer_comparison.png: Overlaid harmful, harmless, and stealth harmful components on same plot")  
    print("- harmful_harmless_difference_heatmap.png: Heatmap showing differences between harmful and harmless")
    print("- stealth_harmful_harmless_difference_heatmap.png: Heatmap showing differences between stealth harmful and harmless")
    print("- harmful_vs_stealth_difference_heatmap.png: Heatmap showing differences between harmful and stealth harmful")

if __name__ == "__main__":
    main()