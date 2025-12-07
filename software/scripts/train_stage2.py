"""
Stage 2 Training: Train Adaptive Controller
Learn to allocate layers based on RGB-D quality
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import rgb_transform, depth_transform


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CorruptedGestureDataset(Dataset):
    """
    Dataset with corruption labels
    Loads from: data/{clean, depth_occluded, low_light}
    """
    
    def __init__(self, root_dir, rgb_transform=None, depth_transform=None):
        """
        Args:
            root_dir: Path to data/ directory
        """
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        
        # Gesture classes
        self.classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Corruption types
        self.corruption_types = ['clean', 'depth_occluded', 'low_light']
        self.corruption_to_idx = {
            'clean': [0.0, 0.0],           # No corruption
            'depth_occluded': [0.0, 1.0],  # Depth corrupted
            'low_light': [1.0, 0.0],       # RGB corrupted
        }
        
        # Load all samples
        print(f"Loading corrupted dataset from {root_dir}...")
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_samples(self):
        """Load all samples with corruption labels"""
        samples = []
        
        for corruption_type in self.corruption_types:
            corruption_dir = os.path.join(self.root_dir, corruption_type)
            
            if not os.path.exists(corruption_dir):
                print(f"⚠️  {corruption_dir} not found, skipping...")
                continue
            
            for class_name in self.classes:
                class_dir = os.path.join(corruption_dir, class_name)
                
                if not os.path.exists(class_dir):
                    continue
                
                # Get all color images and sort
                color_files = sorted([f for f in os.listdir(class_dir) 
                                     if f.startswith('color_image_')])
                
                # Limit depth_occluded to first 20 samples per class
                # if corruption_type == 'depth_occluded':
                #     color_files = color_files[:20]
                
                for color_file in color_files:
                    sample_id = color_file.replace('color_image_', '').replace('.png', '')
                    
                    color_path = os.path.join(class_dir, color_file)
                    depth_path = os.path.join(class_dir, f'depth_image_{sample_id}.png')
                    
                    if os.path.exists(color_path) and os.path.exists(depth_path):
                        samples.append({
                            'color_path': color_path,
                            'depth_path': depth_path,
                            'label': self.class_to_idx[class_name],
                            'corruption_type': corruption_type,
                            'corruption_vector': self.corruption_to_idx[corruption_type],
                            'class_name': class_name
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            data: dict with 'rgb', 'depth'
            label: class label
            corruption: [rgb_corrupt, depth_corrupt] (0 or 1)
        """
        sample = self.samples[idx]
        
        # Load images
        rgb = Image.open(sample['color_path']).convert('RGB')
        depth = Image.open(sample['depth_path']).convert('L')
        
        # Apply transforms
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        data = {'rgb': rgb, 'depth': depth}
        label = sample['label']
        corruption = torch.tensor(sample['corruption_vector'], dtype=torch.float32)
        
        return data, label, corruption


def get_corrupted_dataloaders(data_dir, batch_size=16, num_workers=0, seed=42):
    """
    Create dataloaders for corrupted gesture dataset
    MODIFIED: Use stratified 80/20 split to ensure balanced distribution
    of both corruption types and classes
    """
    # Create dataset
    dataset = CorruptedGestureDataset(
        root_dir=data_dir,
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    print(f"Loading corrupted dataset from {data_dir}...")
    print(f"Loaded {len(dataset)} samples")
    
    # Create stratify labels: combine corruption_type and class_name
    # Format: "{corruption_type}_{class_name}" to ensure balanced split
    stratify_labels = []
    for sample in dataset.samples:
        stratify_label = f"{sample['corruption_type']}_{sample['class_name']}"
        stratify_labels.append(stratify_label)
    
    # Get indices for stratified split
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=stratify_labels,
        random_state=seed,
        shuffle=True
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Print distribution statistics
    print(f"\nStratified Split (80/20):")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples (used as test)")
    
    # Verify balanced distribution
    print("\n  Verifying balanced distribution...")
    
    # Count by corruption type
    train_corr_counts = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    val_corr_counts = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    train_class_counts = {'standing': 0, 'left_hand': 0, 'right_hand': 0, 'both_hands': 0}
    val_class_counts = {'standing': 0, 'left_hand': 0, 'right_hand': 0, 'both_hands': 0}
    
    for idx in train_indices:
        sample = dataset.samples[idx]
        train_corr_counts[sample['corruption_type']] += 1
        train_class_counts[sample['class_name']] += 1
    
    for idx in val_indices:
        sample = dataset.samples[idx]
        val_corr_counts[sample['corruption_type']] += 1
        val_class_counts[sample['class_name']] += 1
    
    print("\n  Train set distribution:")
    print(f"    Corruption: clean={train_corr_counts['clean']}, "
          f"depth_occluded={train_corr_counts['depth_occluded']}, "
          f"low_light={train_corr_counts['low_light']}")
    print(f"    Classes: standing={train_class_counts['standing']}, "
          f"left_hand={train_class_counts['left_hand']}, "
          f"right_hand={train_class_counts['right_hand']}, "
          f"both_hands={train_class_counts['both_hands']}")
    
    print("\n  Val set distribution:")
    print(f"    Corruption: clean={val_corr_counts['clean']}, "
          f"depth_occluded={val_corr_counts['depth_occluded']}, "
          f"low_light={val_corr_counts['low_light']}")
    print(f"    Classes: standing={val_class_counts['standing']}, "
          f"left_hand={val_class_counts['left_hand']}, "
          f"right_hand={val_class_counts['right_hand']}, "
          f"both_hands={val_class_counts['both_hands']}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # MODIFIED: Return val_loader twice (serves as both val and test)
    return train_loader, val_loader, val_loader


def compute_allocation_loss(layer_allocation, corruption_labels):
    """
    Encourage correct layer allocation based on corruption
    
    Args:
        layer_allocation: [batch, 2, 12] - actual allocation
        corruption_labels: [batch, 2] - [rgb_corrupt, depth_corrupt]
    
    Returns:
        loss: scalar
    """
    batch_size = layer_allocation.size(0)
    
    # Count allocated layers per modality
    rgb_layers = layer_allocation[:, 0, :].sum(dim=1)    # [batch]
    depth_layers = layer_allocation[:, 1, :].sum(dim=1)  # [batch]
    
    # Corruption labels: [rgb_corrupt, depth_corrupt]
    rgb_corrupt = corruption_labels[:, 0]    # [batch]
    depth_corrupt = corruption_labels[:, 1]  # [batch]
    
    # Target allocation:
    # - If RGB corrupted: allocate more to Depth
    # - If Depth corrupted: allocate more to RGB
    # - If clean: allocate equally
    
    # Compute ideal ratio (higher corrupt → lower allocation)
    # Use softmax to get allocation ratio
    corruption_inv = 1.0 - corruption_labels  # Inverse corruption
    
    # Add small epsilon to avoid division by zero
    epsilon = 0.1
    corruption_inv = corruption_inv + epsilon
    
    # Normalize to get target ratio
    target_ratio = corruption_inv / corruption_inv.sum(dim=1, keepdim=True)
    target_rgb_ratio = target_ratio[:, 0]    # [batch]
    target_depth_ratio = target_ratio[:, 1]  # [batch]
    
    # Actual ratio
    total_layers = rgb_layers + depth_layers + 1e-6
    actual_rgb_ratio = rgb_layers / total_layers
    actual_depth_ratio = depth_layers / total_layers
    
    # MSE loss between target and actual ratios
    loss = nn.MSELoss()(actual_rgb_ratio, target_rgb_ratio) + \
           nn.MSELoss()(actual_depth_ratio, target_depth_ratio)
    
    return loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                alpha=1.0, beta=0.5, temperature=1.0):
    """
    Train for one epoch
    
    Loss = alpha * classification_loss + beta * allocation_loss
    """
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_alloc_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, labels, corruption) in enumerate(pbar):
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)
        labels = labels.to(device)
        corruption = corruption.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, layer_allocation = model(
            rgb, depth, 
            temperature=temperature, 
            return_allocation=True
        )
        
        # Classification loss
        cls_loss = criterion(logits, labels)
        
        # Allocation loss (encourage corruption-aware allocation)
        alloc_loss = compute_allocation_loss(layer_allocation, corruption)
        
        # Combined loss
        loss = alpha * cls_loss + beta * alloc_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_alloc_loss += alloc_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'alloc': f'{alloc_loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_alloc_loss = total_alloc_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_cls_loss, avg_alloc_loss, accuracy


def validate(model, dataloader, criterion, device, temperature=1.0, alpha=1.0, beta=0.5):
    """Validate the model
    
    MODIFIED: Added alpha and beta parameters to match training
    """
    model.eval()
    
    def set_eval_recursive(module):
        module.eval()
        for child in module.children():
            set_eval_recursive(child)
    
    set_eval_recursive(model)
    
    total_loss = 0
    total_cls_loss = 0
    total_alloc_loss = 0
    correct = 0
    total = 0
    
    # Track allocation statistics
    allocation_stats = {
        'clean': {'rgb': [], 'depth': []},
        'depth_occluded': {'rgb': [], 'depth': []},
        'low_light': {'rgb': [], 'depth': []}
    }
    
    with torch.no_grad():
        for data, labels, corruption in tqdm(dataloader, desc="Validating"):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            labels = labels.to(device)
            corruption = corruption.to(device)
            
            logits, layer_allocation = model(
                rgb, depth,
                temperature=temperature,
                return_allocation=True
            )
            
            cls_loss = criterion(logits, labels)
            alloc_loss = compute_allocation_loss(layer_allocation, corruption)
            
            # MODIFIED: Use same alpha and beta as training
            loss = alpha * cls_loss + beta * alloc_loss
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_alloc_loss += alloc_loss.item()
            
            # Record allocation stats
            for i in range(rgb.size(0)):
                rgb_count = layer_allocation[i, 0].sum().item()
                depth_count = layer_allocation[i, 1].sum().item()
                
                # Determine corruption type
                if corruption[i, 0] == 1.0:  # RGB corrupted
                    corr_type = 'low_light'
                elif corruption[i, 1] == 1.0:  # Depth corrupted
                    corr_type = 'depth_occluded'
                else:
                    corr_type = 'clean'
                
                allocation_stats[corr_type]['rgb'].append(rgb_count)
                allocation_stats[corr_type]['depth'].append(depth_count)
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_alloc_loss = total_alloc_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Compute average allocations
    avg_allocations = {}
    for corr_type in allocation_stats:
        if len(allocation_stats[corr_type]['rgb']) > 0:
            avg_allocations[corr_type] = {
                'rgb': np.mean(allocation_stats[corr_type]['rgb']),
                'depth': np.mean(allocation_stats[corr_type]['depth'])
            }
    
    return avg_loss, avg_cls_loss, avg_alloc_loss, accuracy, avg_allocations

def validate_naive(model, dataloader, criterion, device, rgb_layers, depth_layers):
    """
    Validate with NAIVE (fixed) layer allocation
    
    Args:
        model: Stage 1 GestureClassifier model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        rgb_layers: Number of RGB layers to use (0-12)
        depth_layers: Number of Depth layers to use (0-12)
    
    Returns:
        avg_loss, accuracy, per_corruption_accuracy, avg_allocations
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-corruption accuracy
    corruption_correct = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    corruption_total = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    
    # Create fixed layer masks
    # Strategy: Use LAST N layers (most important for fine-tuned model)
    # Layer 0 is always included when any layers are used (following GTDM convention)
    # Example: 6 layers = [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] (layer 0 + last 5)
    #          12 layers = all 1s
    #          0 layers = all 0s (no processing)
    
    def create_layer_mask(num_layers, type='rgb'):
        mask = torch.zeros(12)
        if num_layers <= 0:
            return mask
        elif num_layers >= 12:
            mask[:] = 1.0
        elif num_layers == 6: # Manually set the layers for 6 layers
            # if type == 'rgb':
            #     mask[[0, 5, 6, 7, 9, 11]] = 1.0
            # elif type == 'depth':
            #     mask[[0, 5, 6, 8, 11]] = 1.0
            mask[[0, 1, 3, 8, 10, 11]] = 1.0
        else:
            # Always include layer 0
            mask[0] = 1.0
            # Include last (num_layers - 1) layers
            remaining = num_layers - 1
            if remaining > 0:
                mask[12-remaining:] = 1.0
        return mask
    
    rgb_mask = create_layer_mask(rgb_layers, type='rgb')
    depth_mask = create_layer_mask(depth_layers, type='depth')
    
    rgb_mask = rgb_mask.to(device)
    depth_mask = depth_mask.to(device)
    
    print(f"  Naive allocation - RGB mask: {rgb_mask.int().tolist()}")
    print(f"  Naive allocation - Depth mask: {depth_mask.int().tolist()}")
    
    with torch.no_grad():
        for data, labels, corruption in tqdm(dataloader, desc="Evaluating (Naive)"):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            labels = labels.to(device)
            
            batch_size = rgb.size(0)
            
            # Forward pass using forward_controller with fixed masks
            # IMPORTANT: When layers=0, return zero vector instead of calling forward_controller
            # because forward_controller still processes patch_embed + positional encoding
            # even when all transformer blocks are masked out
            
            if rgb_layers == 0:
                # Return zero vector when RGB is disabled
                rgb_features = torch.zeros(batch_size, 768).to(device)
                # depth_features += 0.5 * torch.rand_like(depth_features)
                # depth_features = torch.zeros(batch_size, 768).to(device)
            else:
                # Expand mask for batch
                rgb_mask_batch = rgb_mask.unsqueeze(0).expand(batch_size, -1)
                rgb_features = model.vision.forward_controller(rgb, rgb_mask_batch)
                rgb_features = torch.squeeze(rgb_features)
                if len(rgb_features.shape) == 1:
                    rgb_features = torch.unsqueeze(rgb_features, dim=0)
            
            if depth_layers == 0:
                # Return zero vector when Depth is disabled
                depth_features = torch.zeros(batch_size, 768).to(device)
            else:
                # Expand mask for batch
                depth_mask_batch = depth_mask.unsqueeze(0).expand(batch_size, -1)
                depth_features = model.depth.forward_controller(depth, depth_mask_batch)
                depth_features = torch.squeeze(depth_features)
                if len(depth_features.shape) == 1:
                    depth_features = torch.unsqueeze(depth_features, dim=0)
            
            # Fusion (same as model forward)
            from models.vit_dev import positionalencoding1d
            outlist = [model.vision_adapter(rgb_features), model.depth_adapter(depth_features)]
            agg_features = torch.stack(outlist, dim=1)
            b, n, d = agg_features.shape
            agg_features = agg_features + positionalencoding1d(d, n)
            fused = model.encoder(agg_features)
            fused = torch.mean(fused, dim=1)
            logits = model.classifier(fused)
            
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # Track per-corruption accuracy
            for i in range(labels.size(0)):
                if corruption[i, 0] == 1.0:
                    corr_type = 'low_light'
                elif corruption[i, 1] == 1.0:
                    corr_type = 'depth_occluded'
                else:
                    corr_type = 'clean'
                
                corruption_total[corr_type] += 1
                if predicted[i] == labels[i]:
                    corruption_correct[corr_type] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Per-corruption accuracy
    corruption_acc = {}
    for corr_type in ['clean', 'depth_occluded', 'low_light']:
        if corruption_total[corr_type] > 0:
            corruption_acc[corr_type] = 100 * corruption_correct[corr_type] / corruption_total[corr_type]
        else:
            corruption_acc[corr_type] = 0.0
    
    # Fixed allocations
    avg_allocations = {
        'clean': {'rgb': float(rgb_layers), 'depth': float(depth_layers)},
        'depth_occluded': {'rgb': float(rgb_layers), 'depth': float(depth_layers)},
        'low_light': {'rgb': float(rgb_layers), 'depth': float(depth_layers)}
    }
    
    return avg_loss, accuracy, corruption_acc, avg_allocations


def main(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading corrupted data from {args.data_dir}...")
    train_loader, val_loader, test_loader = get_corrupted_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ========================================
    # NAIVE ALLOCATION MODE (Evaluation Only)
    # ========================================
    if args.naive_allocation:
        print("\n" + "="*60)
        print("NAIVE ALLOCATION MODE (Fixed Layer Allocation)")
        print("="*60)
        
        # Parse naive allocation (e.g., "12,0" or "6,6")
        rgb_layers, depth_layers = map(int, args.naive_allocation.split(','))
        print(f"  RGB layers: {rgb_layers}")
        print(f"  Depth layers: {depth_layers}")
        print(f"  Total layers: {rgb_layers + depth_layers}")
        
        # Load Stage 1 model for naive evaluation
        from models.gesture_classifier import GestureClassifier
        model = GestureClassifier(
            num_classes=4,
            adapter_hidden_dim=256,
            vision_vit_layers=12,
            depth_vit_layers=12,
            pretrained_path=None,
            layerdrop=0.0
        ).to(device)
        
        # Load Stage 1 weights
        checkpoint = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded Stage 1 weights from {args.stage1_checkpoint}")
        
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate with naive allocation
        test_loss, test_acc, corruption_acc, allocations = validate_naive(
            model, test_loader, criterion, device, rgb_layers, depth_layers
        )
        
        print(f"\n{'='*60}")
        print(f"NAIVE ALLOCATION RESULTS: RGB={rgb_layers}, Depth={depth_layers}")
        print(f"{'='*60}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"\n  Per-corruption Accuracy:")
        for corr_type, acc in corruption_acc.items():
            print(f"    {corr_type:15s}: {acc:.2f}%")
        print(f"{'='*60}")
        
        # Save results
        import json
        results = {
            'mode': 'naive_allocation',
            'rgb_layers': rgb_layers,
            'depth_layers': depth_layers,
            'total_layers': rgb_layers + depth_layers,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'per_corruption_accuracy': corruption_acc,
            'allocations': allocations
        }
        
        results_path = os.path.join(args.output_dir, f'naive_rgb{rgb_layers}_depth{depth_layers}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {results_path}")
        
        return
    
    # ========================================
    # NORMAL MODE: Train Adaptive Controller
    # ========================================
    
    # Create model
    print("\nCreating adaptive model...")
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=args.total_layers,
        qoi_dim=128,
        stage1_checkpoint=args.stage1_checkpoint
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer (only controller parameters)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    best_val_acc = 0
    
    print(f"\nStarting Stage 2 training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        # Anneal temperature (start from 1.0, decrease to 0.5)
        temperature = max(0.5, 1.0 - (epoch / args.epochs) * 0.5)
        
        # Train
        train_loss, train_cls, train_alloc, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            alpha=args.alpha, beta=args.beta, temperature=temperature
        )
        
        # Validate
        val_loss, val_cls, val_alloc, val_acc, allocations = validate(
            model, val_loader, criterion, device, temperature,
            alpha=args.alpha, beta=args.beta
        )
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_cls', train_cls, epoch)
        writer.add_scalar('Loss/train_alloc', train_alloc, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Temperature', temperature, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f} | Cls: {train_cls:.4f} | Alloc: {train_alloc:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f} | Cls: {val_cls:.4f} | Alloc: {val_alloc:.4f} | Acc: {val_acc:.2f}%")
        print(f"  Temperature: {temperature:.2f}")
        
        # Print allocation stats
        print(f"  Avg Allocations:")
        for corr_type, alloc in allocations.items():
            print(f"    {corr_type:15s}: RGB {alloc['rgb']:.1f} | Depth {alloc['depth']:.1f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, f'best_controller_{args.total_layers}layers.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'allocations': allocations
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print("="*60)
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(args.output_dir, f'best_controller_{args.total_layers}layers.pth'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_cls, test_alloc, test_acc, test_allocations = validate(
        model, test_loader, criterion, device, temperature=0.5,
        alpha=args.alpha, beta=args.beta
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f} | Cls: {test_cls:.4f} | Alloc: {test_alloc:.4f} | Acc: {test_acc:.2f}%")
    print(f"  Allocations:")
    for corr_type, alloc in test_allocations.items():
        print(f"    {corr_type:15s}: RGB {alloc['rgb']:.1f} | Depth {alloc['depth']:.1f}")
    
    print("="*60)
    
    print(f"\n✅ Stage 2 training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Models saved to {args.output_dir}")
    
    # Save results as JSON
    import json
    results = {
        'mode': 'dynamic_allocation',
        'total_layers': args.total_layers,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'allocations': test_allocations,
        'hyperparameters': {
            'alpha': args.alpha,
            'beta': args.beta,
            'lr': args.lr,
            'epochs': args.epochs
        }
    }
    
    results_path = os.path.join(args.output_dir, f'stage2_dynamic_{args.total_layers}layers.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 2 Adaptive Controller')
    
    parser.add_argument('--data_dir', type=str, 
                        default='data',
                        help='Path to corrupted data directory')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='checkpoints/stage1/best_model.pth',
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--total_layers', type=int, default=8,
                        help='Total layer budget for controller')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for allocation loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage2',
                        help='Output directory for checkpoints')
    
    # Naive allocation mode
    parser.add_argument('--naive_allocation', type=str, default=None,
                        help='Naive (fixed) layer allocation as "RGB,DEPTH" (e.g., "12,0", "6,6", "0,12"). '
                             'When set, skips training and only evaluates with fixed allocation.')
    
    args = parser.parse_args()
    main(args)