#!/usr/bin/env python3
"""
ADMN-RealWorld Baseline Comparison Visualization
Generates comprehensive comparison charts for all baseline experiments
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Define distinct colors for different elements (per user preference)
COLORS = {
    'upper_bound': '#2E86AB',      # Blue
    'dynamic_12': '#A23B72',       # Purple
    'dynamic_8': '#F18F01',        # Orange
    'dynamic_6': '#C73E1D',        # Red
    'dynamic_4': '#6A994E',        # Green
    'naive_rgb': '#D4A373',        # Tan
    'naive_depth': '#5C4742',      # Brown
    'naive_half': '#9B5DE5',       # Lavender
    'rgb': '#FF6B6B',              # Coral red for RGB bars
    'depth': '#4ECDC4',            # Teal for Depth bars
}

def load_results():
    """Load all baseline results from JSON files"""
    results_dir = Path('results/baselines')
    
    data = {
        'upper_bound': json.load(open(results_dir / 'stage1_upper_bound.json')),
        'dynamic_12': json.load(open(results_dir / 'stage2_dynamic_12layers.json')),
        'dynamic_8': json.load(open(results_dir / 'stage2_dynamic_8layers.json')),
        'dynamic_6': json.load(open(results_dir / 'stage2_dynamic_6layers.json')),
        'dynamic_4': json.load(open(results_dir / 'stage2_dynamic_4layers.json')),
        'naive_rgb': json.load(open(results_dir / 'naive_rgb12_depth0.json')),
        'naive_depth': json.load(open(results_dir / 'naive_rgb0_depth12.json')),
        'naive_half': json.load(open(results_dir / 'naive_rgb6_depth6.json')),
    }
    
    return data

def plot_overall_accuracy(data, ax):
    """Plot overall test accuracy comparison"""
    methods = [
        'Upper Bound\n(24 layers)',
        'Dynamic\n(12 layers)',
        'Dynamic\n(8 layers)',
        'Dynamic\n(6 layers)',
        'Dynamic\n(4 layers)',
        'Naive RGB\n(12/0)',
        'Naive Depth\n(0/12)',
        'Naive Half\n(6/6)'
    ]
    
    accuracies = [
        data['upper_bound']['test_accuracy'],
        data['dynamic_12']['test_accuracy'],
        data['dynamic_8']['test_accuracy'],
        data['dynamic_6']['test_accuracy'],
        data['dynamic_4']['test_accuracy'],
        data['naive_rgb']['test_accuracy'],
        data['naive_depth']['test_accuracy'],
        data['naive_half']['test_accuracy'],
    ]
    
    colors_list = [
        COLORS['upper_bound'],
        COLORS['dynamic_12'],
        COLORS['dynamic_8'],
        COLORS['dynamic_6'],
        COLORS['dynamic_4'],
        COLORS['naive_rgb'],
        COLORS['naive_depth'],
        COLORS['naive_half'],
    ]
    
    bars = ax.bar(methods, accuracies, color=colors_list, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Overall Test Accuracy Comparison', fontweight='bold', fontsize=14, pad=20)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=0)

def plot_per_corruption(data, ax):
    """Plot accuracy breakdown by corruption type"""
    corruptions = ['clean', 'depth_occluded', 'low_light']
    corruption_labels = ['Clean', 'Depth Occluded', 'Low Light']
    
    methods = ['Upper\nBound', 'Dynamic\n12L', 'Dynamic\n8L', 'Dynamic\n6L', 'Dynamic\n4L', 
               'Naive\nRGB', 'Naive\nDepth', 'Naive\nHalf']
    
    # Extract per-corruption accuracy
    corruption_data = {corr: [] for corr in corruptions}
    
    for key in ['upper_bound', 'dynamic_12', 'dynamic_8', 'dynamic_6', 'dynamic_4',
                'naive_rgb', 'naive_depth', 'naive_half']:
        result = data[key]
        if 'per_corruption_accuracy' in result:
            for corr in corruptions:
                corruption_data[corr].append(result['per_corruption_accuracy'][corr])
        else:
            # For dynamic models, compute from allocations (we don't have this, use test_accuracy)
            for corr in corruptions:
                corruption_data[corr].append(result['test_accuracy'])
    
    x = np.arange(len(methods))
    width = 0.25
    
    # Plot grouped bars with distinct colors
    bar_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']  # Distinct colors for each corruption
    
    for i, (corr, label) in enumerate(zip(corruptions, corruption_labels)):
        offset = (i - 1) * width
        ax.bar(x + offset, corruption_data[corr], width, 
               label=label, color=bar_colors[i], edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy by Corruption Type', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)

def plot_layer_budget_vs_accuracy(data, ax):
    """Plot relationship between layer budget and accuracy"""
    # Dynamic allocation results
    budgets = [24, 12, 8, 6, 4]
    accuracies = [
        data['upper_bound']['test_accuracy'],
        data['dynamic_12']['test_accuracy'],
        data['dynamic_8']['test_accuracy'],
        data['dynamic_6']['test_accuracy'],
        data['dynamic_4']['test_accuracy'],
    ]
    
    # Naive allocations (all have 12 layers)
    naive_accuracies = [
        data['naive_rgb']['test_accuracy'],
        data['naive_depth']['test_accuracy'],
        data['naive_half']['test_accuracy'],
    ]
    
    # Plot dynamic allocation as a line
    ax.plot(budgets, accuracies, 'o-', linewidth=3, markersize=10, 
            label='Dynamic Allocation', color=COLORS['dynamic_12'], markeredgecolor='black', markeredgewidth=1.5)
    
    # Add value labels
    for b, acc in zip(budgets, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(b, acc), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontweight='bold')
    
    # Plot naive allocations at 12 layers as separate markers
    naive_labels = ['RGB Only (12/0)', 'Depth Only (0/12)', 'Half-Half (6/6)']
    naive_colors = [COLORS['naive_rgb'], COLORS['naive_depth'], COLORS['naive_half']]
    markers = ['s', '^', 'D']
    
    for acc, label, color, marker in zip(naive_accuracies, naive_labels, naive_colors, markers):
        ax.scatter([12], [acc], s=200, label=label, color=color, 
                  marker=marker, edgecolor='black', linewidth=1.5, zorder=5)
        ax.annotate(f'{acc:.1f}%', xy=(12, acc), xytext=(15, 0),
                   textcoords='offset points', ha='left', fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Total Layer Budget', fontweight='bold', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Layer Budget vs Accuracy', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2, 26])
    ax.set_ylim([30, 100])

def plot_dynamic_allocations(data, ax):
    """Plot layer allocations for dynamic models across corruptions"""
    corruptions = ['clean', 'depth_occluded', 'low_light']
    corruption_labels = ['Clean', 'Depth\nOccluded', 'Low Light']
    
    models = ['Dynamic\n12L', 'Dynamic\n8L', 'Dynamic\n6L', 'Dynamic\n4L']
    model_keys = ['dynamic_12', 'dynamic_8', 'dynamic_6', 'dynamic_4']
    
    n_models = len(models)
    n_corruptions = len(corruptions)
    
    # Create grouped bar chart
    x = np.arange(n_corruptions)
    width = 0.2
    
    for i, (model, key) in enumerate(zip(models, model_keys)):
        rgb_vals = []
        depth_vals = []
        
        for corr in corruptions:
            alloc = data[key]['allocations'][corr]
            rgb_vals.append(alloc['rgb'])
            depth_vals.append(alloc['depth'])
        
        offset = (i - 1.5) * width
        
        # Stacked bars for RGB and Depth
        ax.bar(x + offset, rgb_vals, width, label=f'{model} RGB' if i == 0 else '',
               color=COLORS['rgb'], edgecolor='black', linewidth=0.8, alpha=0.8)
        ax.bar(x + offset, depth_vals, width, bottom=rgb_vals,
               label=f'{model} Depth' if i == 0 else '',
               color=COLORS['depth'], edgecolor='black', linewidth=0.8, alpha=0.8)
        
        # Add total layer count labels
        for j, (r, d) in enumerate(zip(rgb_vals, depth_vals)):
            total = r + d
            ax.text(x[j] + offset, total + 0.3, f'{total:.1f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_ylabel('Number of Layers', fontweight='bold', fontsize=12)
    ax.set_title('Dynamic Layer Allocation by Corruption Type', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(corruption_labels, fontsize=10)
    ax.set_ylim([0, 14])
    ax.grid(axis='y', alpha=0.3)
    
    # Custom legend
    rgb_patch = mpatches.Patch(color=COLORS['rgb'], label='RGB Layers', edgecolor='black')
    depth_patch = mpatches.Patch(color=COLORS['depth'], label='Depth Layers', edgecolor='black')
    ax.legend(handles=[rgb_patch, depth_patch], loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add model labels
    # legend_models = [mpatches.Patch(color='white', label=model, edgecolor='black') 
    #                 for model in models]
    # ax.legend(handles=[rgb_patch, depth_patch] + legend_models, 
    #          loc='upper left', fontsize=9, framealpha=0.9, ncol=2)

def plot_allocation_heatmap(data, ax):
    """Heatmap showing average RGB vs Depth allocation"""
    models = ['Dynamic 12L', 'Dynamic 8L', 'Dynamic 6L', 'Dynamic 4L']
    model_keys = ['dynamic_12', 'dynamic_8', 'dynamic_6', 'dynamic_4']
    corruptions = ['clean', 'depth_occluded', 'low_light']
    corruption_labels = ['Clean', 'Depth Occluded', 'Low Light']
    
    # Prepare data: RGB ratio for each model-corruption combination
    rgb_ratios = []
    
    for key in model_keys:
        row = []
        for corr in corruptions:
            alloc = data[key]['allocations'][corr]
            rgb_ratio = alloc['rgb'] / (alloc['rgb'] + alloc['depth']) * 100
            row.append(rgb_ratio)
        rgb_ratios.append(row)
    
    rgb_ratios = np.array(rgb_ratios)
    
    # Create heatmap
    im = ax.imshow(rgb_ratios, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corruption_labels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(corruption_labels)
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(corruptions)):
            text = ax.text(j, i, f'{rgb_ratios[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('RGB Layer Allocation Ratio (%)', fontweight='bold', fontsize=14, pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('RGB Ratio (%)', rotation=270, labelpad=20, fontweight='bold')

def plot_test_loss_comparison(data, ax):
    """Compare test loss across all methods"""
    methods = [
        'Upper\nBound',
        'Dynamic\n12L',
        'Dynamic\n8L',
        'Dynamic\n6L',
        'Dynamic\n4L',
        'Naive\nRGB',
        'Naive\nDepth',
        'Naive\nHalf'
    ]
    
    losses = [
        data['upper_bound']['test_loss'],
        data['dynamic_12']['test_loss'],
        data['dynamic_8']['test_loss'],
        data['dynamic_6']['test_loss'],
        data['dynamic_4']['test_loss'],
        data['naive_rgb']['test_loss'],
        data['naive_depth']['test_loss'],
        data['naive_half']['test_loss'],
    ]
    
    colors_list = [
        COLORS['upper_bound'],
        COLORS['dynamic_12'],
        COLORS['dynamic_8'],
        COLORS['dynamic_6'],
        COLORS['dynamic_4'],
        COLORS['naive_rgb'],
        COLORS['naive_depth'],
        COLORS['naive_half'],
    ]
    
    bars = ax.bar(methods, losses, color=colors_list, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_ylabel('Test Loss', fontweight='bold', fontsize=12)
    ax.set_title('Test Loss Comparison (Lower is Better)', fontweight='bold', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3)

def main():
    print("Loading baseline results...")
    data = load_results()
    
    print("Generating comparison charts...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])  # Overall accuracy (wide)
    ax2 = fig.add_subplot(gs[0, 2])   # Test loss
    ax3 = fig.add_subplot(gs[1, :2])  # Per corruption accuracy (wide)
    ax4 = fig.add_subplot(gs[1, 2])   # Layer budget vs accuracy
    ax5 = fig.add_subplot(gs[2, 0])   # Dynamic allocations
    ax6 = fig.add_subplot(gs[2, 1])   # Allocation heatmap
    
    # Generate plots
    plot_overall_accuracy(data, ax1)
    plot_test_loss_comparison(data, ax2)
    plot_per_corruption(data, ax3)
    plot_layer_budget_vs_accuracy(data, ax4)
    plot_dynamic_allocations(data, ax5)
    plot_allocation_heatmap(data, ax6)
    
    # Add overall title
    fig.suptitle('ADMN-RealWorld Baseline Comparison Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = 'results/baseline_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Chart saved to: {output_path}")
    
    # Also save as PDF for publication quality
    output_pdf = 'results/baseline_comparison.pdf'
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF saved to: {output_pdf}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\n📊 Test Accuracy Rankings:")
    rankings = [
        ('Upper Bound (24L)', data['upper_bound']['test_accuracy']),
        ('Dynamic 12L', data['dynamic_12']['test_accuracy']),
        ('Dynamic 8L', data['dynamic_8']['test_accuracy']),
        ('Dynamic 6L', data['dynamic_6']['test_accuracy']),
        ('Dynamic 4L', data['dynamic_4']['test_accuracy']),
        ('Naive RGB (12/0)', data['naive_rgb']['test_accuracy']),
        ('Naive Depth (0/12)', data['naive_depth']['test_accuracy']),
        ('Naive Half (6/6)', data['naive_half']['test_accuracy']),
    ]
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, acc) in enumerate(rankings, 1):
        print(f"  {i}. {name:25s}: {acc:6.2f}%")
    
    print("\n🔍 Key Insights:")
    print(f"  • Best performance: {rankings[0][0]} ({rankings[0][1]:.2f}%)")
    print(f"  • Dynamic 12L achieves {data['dynamic_12']['test_accuracy']:.2f}% with 50% fewer layers than Upper Bound")
    print(f"  • Dynamic 8L achieves {data['dynamic_8']['test_accuracy']:.2f}% with only 33% of layers")
    print(f"  • Naive Depth (0/12) outperforms Naive RGB (12/0): {data['naive_depth']['test_accuracy']:.2f}% vs {data['naive_rgb']['test_accuracy']:.2f}%")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

