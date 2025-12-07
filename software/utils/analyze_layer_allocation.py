"""
Analyze Layer Allocation for Trained Dynamic Models
Load checkpoint and visualize detailed layer allocation patterns
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.adaptive_controller import AdaptiveGestureClassifier
from scripts.train_stage2 import get_corrupted_dataloaders


def analyze_layer_allocation(model, dataloader, device, temperature=0.5):
    """
    Analyze detailed layer allocation patterns
    
    Returns:
        results: dict with detailed allocation info per corruption type
    """
    model.eval()
    
    # Recursively set all submodules to eval mode
    def set_eval_recursive(module):
        module.eval()
        for child in module.children():
            set_eval_recursive(child)
    
    set_eval_recursive(model)
    
    # Storage for detailed analysis
    results = {
        'clean': {'rgb_layers': [], 'depth_layers': [], 'rgb_masks': [], 'depth_masks': [], 'correct': []},
        'depth_occluded': {'rgb_layers': [], 'depth_layers': [], 'rgb_masks': [], 'depth_masks': [], 'correct': []},
        'low_light': {'rgb_layers': [], 'depth_layers': [], 'rgb_masks': [], 'depth_masks': [], 'correct': []}
    }
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels, corruption in tqdm(dataloader, desc="Analyzing allocations"):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            labels = labels.to(device)
            
            # Forward pass with allocation tracking
            logits, layer_allocation = model(
                rgb, depth,
                temperature=temperature,
                return_allocation=True
            )
            
            # Get predictions
            _, predicted = torch.max(logits, 1)
            correct = (predicted == labels).cpu().numpy()
            
            # Process each sample in batch
            for i in range(rgb.size(0)):
                # Determine corruption type
                if corruption[i, 0] == 1.0:  # RGB corrupted
                    corr_type = 'low_light'
                elif corruption[i, 1] == 1.0:  # Depth corrupted
                    corr_type = 'depth_occluded'
                else:
                    corr_type = 'clean'
                
                # Extract allocation for this sample
                rgb_alloc = layer_allocation[i, 0].cpu().numpy()  # [12]
                depth_alloc = layer_allocation[i, 1].cpu().numpy()  # [12]
                
                # Count allocated layers
                rgb_count = rgb_alloc.sum()
                depth_count = depth_alloc.sum()
                
                # Store results
                results[corr_type]['rgb_layers'].append(rgb_count)
                results[corr_type]['depth_layers'].append(depth_count)
                results[corr_type]['rgb_masks'].append(rgb_alloc)
                results[corr_type]['depth_masks'].append(depth_alloc)
                results[corr_type]['correct'].append(correct[i])
                
                total_correct += correct[i]
                total_samples += 1
    
    # Compute statistics
    stats = {}
    for corr_type in results:
        n_samples = len(results[corr_type]['rgb_layers'])
        if n_samples == 0:
            continue
            
        # Average layer counts
        avg_rgb = np.mean(results[corr_type]['rgb_layers'])
        avg_depth = np.mean(results[corr_type]['depth_layers'])
        
        # Layer activation frequencies (how often each layer is activated)
        rgb_masks = np.array(results[corr_type]['rgb_masks'])  # [N, 12]
        depth_masks = np.array(results[corr_type]['depth_masks'])  # [N, 12]
        
        rgb_freq = rgb_masks.mean(axis=0)  # [12]
        depth_freq = depth_masks.mean(axis=0)  # [12]
        
        # Find most frequently activated layers
        rgb_top_layers = np.argsort(rgb_freq)[::-1][:int(np.ceil(avg_rgb))]
        depth_top_layers = np.argsort(depth_freq)[::-1][:int(np.ceil(avg_depth))]
        
        # Accuracy
        accuracy = np.mean(results[corr_type]['correct']) * 100
        
        stats[corr_type] = {
            'n_samples': n_samples,
            'avg_rgb_layers': float(avg_rgb),
            'avg_depth_layers': float(avg_depth),
            'rgb_layer_frequencies': rgb_freq.tolist(),
            'depth_layer_frequencies': depth_freq.tolist(),
            'rgb_top_layers': rgb_top_layers.tolist(),
            'depth_top_layers': depth_top_layers.tolist(),
            'accuracy': float(accuracy)
        }
    
    overall_accuracy = 100 * total_correct / total_samples
    
    return results, stats, overall_accuracy


def visualize_layer_allocation(stats, output_dir, total_layers):
    """
    Visualize layer allocation patterns
    """
    corruption_types = ['clean', 'depth_occluded', 'low_light']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Layer Allocation Analysis (Dynamic {total_layers} Layers)', fontsize=16, fontweight='bold')
    
    colors_rgb = ['#e74c3c', '#c0392b', '#e67e22']  # Red tones for RGB
    colors_depth = ['#3498db', '#2980b9', '#1abc9c']  # Blue tones for Depth
    
    for idx, corr_type in enumerate(corruption_types):
        if corr_type not in stats:
            continue
        
        stat = stats[corr_type]
        
        # RGB layer frequencies (top subplot)
        ax_rgb = axes[0, idx]
        rgb_freq = np.array(stat['rgb_layer_frequencies'])
        bars_rgb = ax_rgb.bar(range(12), rgb_freq, color=colors_rgb[idx], alpha=0.7, edgecolor='black')
        ax_rgb.set_xlabel('Layer Index', fontsize=11)
        ax_rgb.set_ylabel('Activation Frequency', fontsize=11)
        ax_rgb.set_title(f'{corr_type.replace("_", " ").title()}\nRGB Layers (Avg: {stat["avg_rgb_layers"]:.2f})', 
                         fontsize=12, fontweight='bold')
        ax_rgb.set_xticks(range(12))
        ax_rgb.set_ylim([0, 1.0])
        ax_rgb.grid(axis='y', alpha=0.3)
        ax_rgb.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Highlight top layers
        for layer_idx in stat['rgb_top_layers']:
            bars_rgb[layer_idx].set_edgecolor('red')
            bars_rgb[layer_idx].set_linewidth(2.5)
        
        # Depth layer frequencies (bottom subplot)
        ax_depth = axes[1, idx]
        depth_freq = np.array(stat['depth_layer_frequencies'])
        bars_depth = ax_depth.bar(range(12), depth_freq, color=colors_depth[idx], alpha=0.7, edgecolor='black')
        ax_depth.set_xlabel('Layer Index', fontsize=11)
        ax_depth.set_ylabel('Activation Frequency', fontsize=11)
        ax_depth.set_title(f'Depth Layers (Avg: {stat["avg_depth_layers"]:.2f})', 
                          fontsize=12, fontweight='bold')
        ax_depth.set_xticks(range(12))
        ax_depth.set_ylim([0, 1.0])
        ax_depth.grid(axis='y', alpha=0.3)
        ax_depth.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Highlight top layers
        for layer_idx in stat['depth_top_layers']:
            bars_depth[layer_idx].set_edgecolor('blue')
            bars_depth[layer_idx].set_linewidth(2.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'layer_allocation_dynamic_{total_layers}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    
    plt.close()
    
    # Create heatmap visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Layer Activation Heatmap (Dynamic {total_layers} Layers)', fontsize=16, fontweight='bold')
    
    # Prepare data for heatmap
    rgb_data = []
    depth_data = []
    labels = []
    
    for corr_type in corruption_types:
        if corr_type not in stats:
            continue
        stat = stats[corr_type]
        rgb_data.append(stat['rgb_layer_frequencies'])
        depth_data.append(stat['depth_layer_frequencies'])
        labels.append(corr_type.replace('_', ' ').title())
    
    rgb_data = np.array(rgb_data)
    depth_data = np.array(depth_data)
    
    # RGB heatmap
    sns.heatmap(rgb_data, ax=axes[0], cmap='Reds', annot=True, fmt='.2f',
                xticklabels=range(12), yticklabels=labels,
                cbar_kws={'label': 'Activation Frequency'},
                vmin=0, vmax=1)
    axes[0].set_xlabel('Layer Index', fontsize=12)
    axes[0].set_ylabel('Corruption Type', fontsize=12)
    axes[0].set_title('RGB Layers', fontsize=14, fontweight='bold')
    
    # Depth heatmap
    sns.heatmap(depth_data, ax=axes[1], cmap='Blues', annot=True, fmt='.2f',
                xticklabels=range(12), yticklabels=labels,
                cbar_kws={'label': 'Activation Frequency'},
                vmin=0, vmax=1)
    axes[1].set_xlabel('Layer Index', fontsize=12)
    axes[1].set_ylabel('Corruption Type', fontsize=12)
    axes[1].set_title('Depth Layers', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, f'layer_heatmap_dynamic_{total_layers}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to {heatmap_path}")
    
    plt.close()


def print_summary(stats, overall_accuracy):
    """Print detailed summary"""
    print("\n" + "="*80)
    print("LAYER ALLOCATION ANALYSIS")
    print("="*80)
    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%\n")
    
    for corr_type in ['clean', 'depth_occluded', 'low_light']:
        if corr_type not in stats:
            continue
        
        stat = stats[corr_type]
        print(f"{corr_type.upper().replace('_', ' ')}:")
        print(f"  Samples: {stat['n_samples']}")
        print(f"  Accuracy: {stat['accuracy']:.2f}%")
        print(f"  Average RGB Layers: {stat['avg_rgb_layers']:.2f}")
        print(f"  Average Depth Layers: {stat['avg_depth_layers']:.2f}")
        print(f"  Total: {stat['avg_rgb_layers'] + stat['avg_depth_layers']:.2f}")
        
        # Show top activated layers
        print(f"  Top RGB Layers: {stat['rgb_top_layers']}")
        print(f"  Top Depth Layers: {stat['depth_top_layers']}")
        
        # Show layer frequencies above 0.5
        rgb_freq = np.array(stat['rgb_layer_frequencies'])
        depth_freq = np.array(stat['depth_layer_frequencies'])
        
        rgb_high = np.where(rgb_freq > 0.5)[0].tolist()
        depth_high = np.where(depth_freq > 0.5)[0].tolist()
        
        print(f"  RGB Layers (>50% activation): {rgb_high}")
        print(f"  Depth Layers (>50% activation): {depth_high}")
        print()
    
    print("="*80)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading test data from {args.data_dir}...")
    _, _, test_loader = get_corrupted_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42
    )
    
    # Create model
    print(f"\nLoading model with {args.total_layers} layers...")
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=args.total_layers,
        qoi_dim=128,
        stage1_checkpoint=args.stage1_checkpoint
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_controller_{args.total_layers}layers.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    print(f"   Checkpoint validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Analyze layer allocation
    print("\nAnalyzing layer allocation patterns...")
    results, stats, overall_accuracy = analyze_layer_allocation(
        model, test_loader, device, temperature=args.temperature
    )
    
    # Print summary
    print_summary(stats, overall_accuracy)
    
    # Save detailed results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f'layer_allocation_analysis_dynamic_{args.total_layers}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'total_layers': args.total_layers,
            'overall_accuracy': overall_accuracy,
            'stats': stats,
            'temperature': args.temperature
        }, f, indent=2)
    print(f"\n✅ Detailed results saved to {results_path}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_layer_allocation(stats, output_dir, args.total_layers)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Layer Allocation for Trained Models')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='checkpoints/stage1/best_model.pth',
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/stage2_6layers',
                        help='Directory containing trained Stage 2 checkpoints')
    parser.add_argument('--total_layers', type=int, default=6,
                        help='Total layer budget (should match checkpoint)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for Gumbel-Softmax')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results and visualizations')
    
    args = parser.parse_args()
    main(args)

