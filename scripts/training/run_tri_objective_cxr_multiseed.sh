#!/bin/bash
################################################################################
# Multi-Seed Tri-Objective Training for Chest X-Ray - Phase 7.6
################################################################################
# This script trains the tri-objective model on NIH ChestX-ray14 dataset
# across 3 seeds (42, 123, 456) for statistical rigor.
#
# Expected Runtime: ~45-60 GPU hours total (15-20h per seed)
# Author: Viraj Pankaj Jain
################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/configs/experiments/tri_objective_cxr.yaml"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/train_tri_objective_cxr.py"
LOG_DIR="$PROJECT_ROOT/results/logs/multiseed_cxr"
SEEDS=(42 123 456)
GPU_ID=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$LOG_DIR/multiseed_batch_${TIMESTAMP}.log"

log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$BATCH_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$BATCH_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$BATCH_LOG"
}

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║     Tri-Objective Multi-Label CXR Training - Phase 7.6                ║"
echo "║                    NIH ChestX-ray14 (14 classes)                      ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

log "Starting pre-flight checks..."

# Check Python
if ! command -v python &> /dev/null; then
    log_error "Python not found"
    exit 1
fi
log_success "Python available"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found"
    exit 1
fi
log_success "CUDA available"

# Check config
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi
log_success "Config file found"

# Check training script
if [ ! -f "$TRAINING_SCRIPT" ]; then
    log_error "Training script not found: $TRAINING_SCRIPT"
    exit 1
fi
log_success "Training script found"

log_success "Pre-flight checks completed!"

# User confirmation
echo ""
log "Configuration: ${#SEEDS[@]} seeds, 60 epochs each"
log "Estimated time: ~45-60 hours total"
echo -n "Proceed with training? [y/N] "
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    log "Training cancelled by user."
    exit 0
fi

# Training loop
echo ""
log "Starting multi-seed training..."

declare -A SEED_STATUS
declare -A SEED_BEST_METRIC
SUCCESSFUL_SEEDS=0
FAILED_SEEDS=0

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                         SEED: $seed                                    "
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""

    log "Starting training with seed $seed..."
    SEED_LOG="$LOG_DIR/seed_${seed}_${TIMESTAMP}.log"

    TRAIN_CMD="python $TRAINING_SCRIPT \
        --config $CONFIG_FILE \
        --seed $seed \
        --gpu $GPU_ID"

    if eval "$TRAIN_CMD" 2>&1 | tee "$SEED_LOG"; then
        SEED_STATUS[$seed]="SUCCESS"
        SUCCESSFUL_SEEDS=$((SUCCESSFUL_SEEDS + 1))
        log_success "Seed $seed completed successfully"
    else
        SEED_STATUS[$seed]="FAILED"
        FAILED_SEEDS=$((FAILED_SEEDS + 1))
        log_error "Seed $seed FAILED!"
    fi

    if [ $seed != "${SEEDS[-1]}" ]; then
        log "Pausing 10 seconds before next seed..."
        sleep 10
    fi
done

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                         TRAINING SUMMARY                               ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

log "Multi-seed training completed!"
log "Successful seeds: $SUCCESSFUL_SEEDS / ${#SEEDS[@]}"
log "Failed seeds: $FAILED_SEEDS / ${#SEEDS[@]}"

if [ $FAILED_SEEDS -eq 0 ]; then
    log_success "All seeds completed successfully! 🎉"
    exit 0
else
    log_error "Some seeds failed. Review logs."
    exit 1
fi
