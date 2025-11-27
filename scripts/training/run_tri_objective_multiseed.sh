#!/bin/bash
###############################################################################
# Multi-Seed Tri-Objective Training Script - Phase 7.5
###############################################################################
#
# This script runs tri-objective training sequentially with multiple random
# seeds to ensure statistical significance of results. Running multiple seeds
# is essential for dissertation-quality experiments to:
#
#   1. Quantify variance in model performance
#   2. Compute confidence intervals for metrics
#   3. Ensure reproducibility and robustness of findings
#   4. Meet A1+ grade statistical rigor standards
#
# Usage:
#   bash scripts/training/run_tri_objective_multiseed.sh
#
# Configuration:
#   - Seeds: 42, 123, 456 (three independent runs)
#   - Config: configs/experiments/tri_objective.yaml
#   - Checkpoints: checkpoints/tri_objective/seed_<SEED>/
#   - MLflow: Logs to mlruns/ directory
#
# Expected Runtime (RTX 3050, 4GB):
#   - Per seed: ~25-30 hours (60 epochs)
#   - Total: ~75-90 hours (sequential execution)
#
# Output:
#   - Individual checkpoints for each seed
#   - MLflow experiment tracking with seed-specific runs
#   - Aggregated results summary (printed at end)
#
# Notes:
#   - Uses sequential execution to avoid GPU memory conflicts
#   - Each seed gets independent checkpoint directory
#   - Progress is logged to logs/train_tri_objective.log
#   - Can be interrupted and resumed (checkpoints are saved)
#
# Author: Viraj Pankaj Jain
# Institution: University of Glasgow, School of Computing Science
# Date: November 27, 2025
# Version: 1.0.0 (Phase 7.5)
#
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

###############################################################################
# Configuration
###############################################################################

# Experiment configuration file
CONFIG="configs/experiments/tri_objective.yaml"

# Random seeds for statistical significance (3 runs)
SEEDS=(42 123 456)

# Project root directory (auto-detected)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Python script path
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/training/train_tri_objective.py"

# Checkpoint base directory
CHECKPOINT_BASE_DIR="${PROJECT_ROOT}/checkpoints/tri_objective"

# Log file
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/multi_seed_training_$(date +%Y%m%d_%H%M%S).log"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

###############################################################################
# Helper Functions
###############################################################################

# Print colored message
print_msg() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print section header
print_header() {
    local title=$1
    echo ""
    print_msg "${CYAN}" "=============================================================================="
    print_msg "${CYAN}" "  ${title}"
    print_msg "${CYAN}" "=============================================================================="
    echo ""
}

# Print progress bar
print_progress() {
    local current=$1
    local total=$2
    local seed=$3
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled=$((percent * bar_length / 100))
    local empty=$((bar_length - filled))

    printf "\r${BLUE}Progress:${NC} ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] ${percent}%% (Seed ${seed}: ${current}/${total})${NC}"
}

# Log message to file and console
log_msg() {
    local message=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${message}" | tee -a "${LOG_FILE}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

###############################################################################
# Pre-flight Checks
###############################################################################

print_header "Pre-flight Checks"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"
log_msg "Log file: ${LOG_FILE}"

# Check if Python is available
if ! command_exists python; then
    print_msg "${RED}" "ERROR: Python not found in PATH"
    exit 1
fi
print_msg "${GREEN}" "✓ Python found: $(python --version)"

# Check if Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    print_msg "${RED}" "ERROR: Training script not found: ${PYTHON_SCRIPT}"
    exit 1
fi
print_msg "${GREEN}" "✓ Training script found: ${PYTHON_SCRIPT}"

# Check if config file exists
if [ ! -f "${CONFIG}" ]; then
    print_msg "${RED}" "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi
print_msg "${GREEN}" "✓ Config file found: ${CONFIG}"

# Check if CUDA is available
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEMORY=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')" 2>/dev/null)
    print_msg "${GREEN}" "✓ CUDA available: ${GPU_NAME} (${GPU_MEMORY} GB)"
else
    print_msg "${YELLOW}" "⚠ CUDA not available, training will use CPU (very slow)"
fi

# Create checkpoint directory
mkdir -p "${CHECKPOINT_BASE_DIR}"
print_msg "${GREEN}" "✓ Checkpoint directory: ${CHECKPOINT_BASE_DIR}"

###############################################################################
# Training Summary
###############################################################################

print_header "Multi-Seed Training Configuration"

print_msg "${CYAN}" "Configuration:"
echo "  Config file:        ${CONFIG}"
echo "  Training script:    ${PYTHON_SCRIPT}"
echo "  Seeds:              ${SEEDS[@]}"
echo "  Number of runs:     ${#SEEDS[@]}"
echo "  Checkpoint dir:     ${CHECKPOINT_BASE_DIR}"
echo "  Log file:           ${LOG_FILE}"

print_msg "${CYAN}" "\nExpected Runtime (RTX 3050 4GB):"
echo "  Per seed:           ~25-30 hours"
echo "  Total:              ~75-90 hours"

print_msg "${CYAN}" "\nOutput:"
echo "  Checkpoints:        ${CHECKPOINT_BASE_DIR}/seed_<SEED>/"
echo "  MLflow logs:        ${PROJECT_ROOT}/mlruns/"
echo "  Training logs:      ${LOG_FILE}"

###############################################################################
# Start Training
###############################################################################

print_header "Starting Multi-Seed Training"

# Initialize tracking variables
START_TIME=$(date +%s)
COMPLETED_SEEDS=()
FAILED_SEEDS=()
TOTAL_SEEDS=${#SEEDS[@]}

# Main training loop
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    SEED_IDX=$((i + 1))

    print_header "Training Seed ${SEED_IDX}/${TOTAL_SEEDS}: ${SEED}"

    log_msg "Starting training with seed ${SEED}"

    # Checkpoint directory for this seed
    SEED_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/seed_${SEED}"
    mkdir -p "${SEED_CHECKPOINT_DIR}"

    # Training start time for this seed
    SEED_START_TIME=$(date +%s)

    # Run training
    print_msg "${BLUE}" "Executing: python ${PYTHON_SCRIPT} --config ${CONFIG} --seed ${SEED}"
    echo ""

    if python "${PYTHON_SCRIPT}" \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --checkpoint-dir "${SEED_CHECKPOINT_DIR}" \
        2>&1 | tee -a "${LOG_FILE}"; then

        # Training succeeded
        SEED_END_TIME=$(date +%s)
        SEED_DURATION=$((SEED_END_TIME - SEED_START_TIME))
        SEED_HOURS=$((SEED_DURATION / 3600))
        SEED_MINUTES=$(((SEED_DURATION % 3600) / 60))

        COMPLETED_SEEDS+=("${SEED}")

        print_msg "${GREEN}" "\n✓ Seed ${SEED} completed successfully in ${SEED_HOURS}h ${SEED_MINUTES}m"
        log_msg "Seed ${SEED} completed in ${SEED_HOURS}h ${SEED_MINUTES}m"

    else
        # Training failed
        FAILED_SEEDS+=("${SEED}")

        print_msg "${RED}" "\n✗ Seed ${SEED} failed!"
        log_msg "ERROR: Seed ${SEED} failed"

        # Ask user if they want to continue
        read -p "$(print_msg ${YELLOW} 'Continue with remaining seeds? [y/N]: ')" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_msg "${YELLOW}" "Training interrupted by user"
            break
        fi
    fi

    # Print progress
    print_progress "${SEED_IDX}" "${TOTAL_SEEDS}" "${SEED}"
    echo ""

    # Small delay between seeds to ensure clean GPU state
    if [ "${SEED_IDX}" -lt "${TOTAL_SEEDS}" ]; then
        print_msg "${CYAN}" "\nWaiting 30 seconds before next seed..."
        sleep 30
    fi
done

###############################################################################
# Final Summary
###############################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

print_header "Multi-Seed Training Complete"

print_msg "${CYAN}" "Summary:"
echo "  Total seeds:        ${TOTAL_SEEDS}"
echo "  Completed:          ${#COMPLETED_SEEDS[@]}"
echo "  Failed:             ${#FAILED_SEEDS[@]}"
echo "  Total time:         ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"

if [ ${#COMPLETED_SEEDS[@]} -gt 0 ]; then
    print_msg "${GREEN}" "\nSuccessfully completed seeds:"
    for seed in "${COMPLETED_SEEDS[@]}"; do
        echo "  ✓ Seed ${seed}: ${CHECKPOINT_BASE_DIR}/seed_${seed}/best.pt"
    done
fi

if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
    print_msg "${RED}" "\nFailed seeds:"
    for seed in "${FAILED_SEEDS[@]}"; do
        echo "  ✗ Seed ${seed}"
    done

    print_msg "${YELLOW}" "\nTo retry failed seeds, run:"
    for seed in "${FAILED_SEEDS[@]}"; do
        echo "  python ${PYTHON_SCRIPT} --config ${CONFIG} --seed ${seed}"
    done
fi

print_msg "${CYAN}" "\nResults:"
echo "  Checkpoints:        ${CHECKPOINT_BASE_DIR}/"
echo "  MLflow logs:        ${PROJECT_ROOT}/mlruns/"
echo "  Training log:       ${LOG_FILE}"

# Generate aggregated results summary (if Python + pandas available)
if command_exists python && python -c "import pandas" 2>/dev/null; then
    print_msg "${CYAN}" "\nGenerating aggregated results summary..."

    # Create a simple Python script to aggregate results
    cat > /tmp/aggregate_results.py << 'EOF'
import sys
import json
from pathlib import Path
import mlflow

def aggregate_results(mlflow_experiment_name):
    """Aggregate results from multiple MLflow runs."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(mlflow_experiment_name)

        if experiment is None:
            print(f"Experiment '{mlflow_experiment_name}' not found")
            return

        runs = client.search_runs(experiment.experiment_id)

        if not runs:
            print("No runs found")
            return

        print("\nAggregated Results Across Seeds:")
        print("="*70)

        metrics = [
            "val/accuracy_clean",
            "val/accuracy_robust",
            "val/ssim_mean",
            "val/auroc_macro",
            "val/ece",
        ]

        for metric in metrics:
            values = []
            for run in runs:
                if metric in run.data.metrics:
                    values.append(run.data.metrics[metric])

            if values:
                import numpy as np
                mean = np.mean(values)
                std = np.std(values)
                print(f"{metric:30s}: {mean:.4f} ± {std:.4f}")

        print("="*70)

    except Exception as e:
        print(f"Error aggregating results: {e}")

if __name__ == "__main__":
    aggregate_results("tri-objective-isic2018")
EOF

    python /tmp/aggregate_results.py 2>/dev/null || true
    rm /tmp/aggregate_results.py
fi

print_msg "${GREEN}" "\n✓ All done!"
log_msg "Multi-seed training completed"

###############################################################################
# Exit
###############################################################################

if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
