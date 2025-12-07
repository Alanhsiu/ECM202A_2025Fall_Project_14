"""
Simple test to verify model accuracy matches training log
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.adaptive_controller import AdaptiveGestureClassifier
from scripts.train_stage2 import get_corrupted_dataloaders, validate, compute_allocation_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("\nLoading data...")
_, _, test_loader = get_corrupted_dataloaders(
    data_dir='data',
    batch_size=16,
    num_workers=4,
    seed=42
)

print(f"Test samples: {len(test_loader.dataset)}")

# Create model
print("\nLoading model...")
model = AdaptiveGestureClassifier(
    num_classes=4,
    adapter_hidden_dim=256,
    total_layers=6,
    qoi_dim=128,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
).to(device)

# Load checkpoint
checkpoint_path = 'checkpoints/stage2_6layers/best_controller_6layers.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded checkpoint: val_acc = {checkpoint.get('val_acc', 'N/A')}")

# Use the same validate function from training script
criterion = nn.CrossEntropyLoss()

print("\nRunning validation (same as training script)...")
test_loss, test_cls, test_alloc, test_acc, test_allocations = validate(
    model, test_loader, criterion, device, temperature=0.5,
    alpha=1.0, beta=5.0
)

print(f"\n{'='*60}")
print(f"Test Results:")
print(f"  Accuracy: {test_acc:.2f}%")
print(f"  Loss: {test_loss:.4f}")
print(f"  Allocations:")
for corr_type, alloc in test_allocations.items():
    print(f"    {corr_type:15s}: RGB {alloc['rgb']:.1f} | Depth {alloc['depth']:.1f}")
print(f"{'='*60}")


