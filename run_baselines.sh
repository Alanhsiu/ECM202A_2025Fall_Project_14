#!/bin/bash
# ==============================================================================
# ADMN-RealWorld Baseline Generation and Testing Script
# ==============================================================================
# This script runs all baselines for the Adaptive Depth-Modality Network:
#   1. Upper Bound (Stage 1) - All 24 layers
#   2. Stage 2 Dynamic Layer Allocation
#   3. Naive Allocation (Fixed): 12/0, 0/12, 6/6
#   4. Reduced Layer Budget: 4, 6, 8
# ==============================================================================

set -e  # Exit on error

# Create directories
mkdir -p logs
mkdir -p checkpoints/stage1
mkdir -p checkpoints/stage2
mkdir -p results/baselines

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo ""
    echo "=============================================================="
    echo -e "${BLUE}$1${NC}"
    echo "=============================================================="
    echo ""
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to log time
log_time() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# ==============================================================================
# 1. UPPER BOUND: Stage 1 Training
# ==============================================================================
print_header "1. UPPER BOUND: Stage 1 Training (All 24 Layers)"

STAGE1_LOG="logs/stage1_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $STAGE1_LOG"

start_time=$(date +%s)
log_time "Starting Stage 1 training" | tee -a "$STAGE1_LOG"

python scripts/train_stage1.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --layerdrop 0.0 \
    --weight_decay 0.0 \
    --label_smoothing 0.0 \
    --patience 30 \
    --seed 42 \
    --output_dir checkpoints/stage1 2>&1 | tee -a "$STAGE1_LOG"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
log_time "Stage 1 completed in ${elapsed}s" | tee -a "$STAGE1_LOG"

# Copy results to results directory
cp checkpoints/stage1/stage1_upper_bound.json results/baselines/
print_success "Stage 1 (Upper Bound) completed"

# ==============================================================================
# 2. STAGE 2: Dynamic Layer Allocation (12 layers)
# ==============================================================================
print_header "2. STAGE 2: Dynamic Layer Allocation (12 Layers)"

STAGE2_LOG="logs/stage2_dynamic_12_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $STAGE2_LOG"

start_time=$(date +%s)
log_time "Starting Stage 2 training (12 layers)" | tee -a "$STAGE2_LOG"

python scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --total_layers 12 \
    --batch_size 16 \
    --lr 1e-4 \
    --alpha 1.0 \
    --beta 5.0 \
    --seed 42 \
    --epochs 100 \
    --output_dir checkpoints/stage2 2>&1 | tee -a "$STAGE2_LOG"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
log_time "Stage 2 (12 layers) completed in ${elapsed}s" | tee -a "$STAGE2_LOG"

# Copy results
cp checkpoints/stage2/stage2_dynamic_12layers.json results/baselines/
print_success "Stage 2 Dynamic (12 layers) completed"

# ==============================================================================
# 3. NAIVE ALLOCATION: Fixed Layer Allocation
# ==============================================================================
print_header "3. NAIVE ALLOCATION: Fixed Layer Allocation Tests"

# 3a. All RGB (12/0)
echo "Testing Naive Allocation: RGB=12, Depth=0"
NAIVE_LOG="logs/naive_12_0_$(date +%Y%m%d_%H%M%S).log"

python scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --naive_allocation "12,0" \
    --seed 42 \
    --output_dir results/baselines 2>&1 | tee -a "$NAIVE_LOG"

print_success "Naive (12/0) completed"

# 3b. All Depth (0/12)
echo "Testing Naive Allocation: RGB=0, Depth=12"
NAIVE_LOG="logs/naive_0_12_$(date +%Y%m%d_%H%M%S).log"

python scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --naive_allocation "0,12" \
    --seed 42 \
    --output_dir results/baselines 2>&1 | tee -a "$NAIVE_LOG"

print_success "Naive (0/12) completed"

# 3c. Each Half (6/6)
echo "Testing Naive Allocation: RGB=6, Depth=6"
NAIVE_LOG="logs/naive_6_6_$(date +%Y%m%d_%H%M%S).log"

python scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --naive_allocation "6,6" \
    --seed 42 \
    --output_dir results/baselines 2>&1 | tee -a "$NAIVE_LOG"

print_success "Naive (6/6) completed"

# ==============================================================================
# 4. REDUCED LAYER BUDGET: 4, 6, 8 layers
# ==============================================================================
print_header "4. REDUCED LAYER BUDGET: Stage 2 with 4, 6, 8 layers"

for BUDGET in 4 6 8; do
    echo ""
    echo "Training Stage 2 with Layer Budget = $BUDGET"
    BUDGET_LOG="logs/stage2_dynamic_${BUDGET}_$(date +%Y%m%d_%H%M%S).log"
    
    start_time=$(date +%s)
    log_time "Starting Stage 2 training ($BUDGET layers)" | tee -a "$BUDGET_LOG"
    
    python scripts/train_stage2.py \
        --stage1_checkpoint checkpoints/stage1/best_model.pth \
        --data_dir data \
        --total_layers $BUDGET \
        --batch_size 16 \
        --lr 1e-4 \
        --alpha 1.0 \
        --beta 5.0 \
        --seed 42 \
        --epochs 100 \
        --output_dir checkpoints/stage2_${BUDGET}layers 2>&1 | tee -a "$BUDGET_LOG"
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    log_time "Stage 2 ($BUDGET layers) completed in ${elapsed}s" | tee -a "$BUDGET_LOG"
    
    # Copy results
    cp checkpoints/stage2_${BUDGET}layers/stage2_dynamic_${BUDGET}layers.json results/baselines/
    print_success "Stage 2 Dynamic ($BUDGET layers) completed"
done

# ==============================================================================
# SUMMARY
# ==============================================================================
print_header "BASELINE GENERATION COMPLETE"

echo "All results saved to: results/baselines/"
echo ""
echo "Generated files:"
ls -la results/baselines/*.json 2>/dev/null || echo "No JSON files found"

echo ""
echo "Summary of baselines:"
echo "  1. Upper Bound (Stage 1):     24 layers (RGB 12 + Depth 12)"
echo "  2. Dynamic Allocation:        12 layers (adaptive)"
echo "  3. Naive Allocations:"
echo "     - All RGB:                 12 layers (RGB 12 + Depth 0)"
echo "     - All Depth:               12 layers (RGB 0 + Depth 12)"
echo "     - Half-Half:               12 layers (RGB 6 + Depth 6)"
echo "  4. Reduced Budgets (Dynamic): 4, 6, 8 layers"

echo ""
print_success "All baselines generated successfully!"

