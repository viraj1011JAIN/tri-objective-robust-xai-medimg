"""
Batch Training Script for Phase 4.4 - All Seeds
================================================

Train EfficientNet-B0 baseline models for all three seeds (42, 123, 456)
to enable statistical analysis of transferability patterns.

This script automates the training process for all seeds sequentially.

Author: Viraj Pankaj Jain
Date: November 24, 2025
"""

import subprocess
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Seeds to train
SEEDS = [42, 123, 456]

# Training configuration
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = "cuda"

print("=" * 80)
print("BATCH TRAINING: EfficientNet-B0 Baseline (All Seeds)")
print("=" * 80)
print(f"Seeds: {SEEDS}")
print(f"Epochs per seed: {EPOCHS}")
print(f"Total training time estimate: {len(SEEDS) * 30}-{len(SEEDS) * 60} minutes")
print("=" * 80)

start_time = time.time()

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"Training Seed {seed}")
    print(f"{'=' * 60}")

    seed_start = time.time()

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "training" / "train_baseline_efficientnet.py"),
        "--seed",
        str(seed),
        "--num-epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--learning-rate",
        str(LEARNING_RATE),
        "--device",
        DEVICE,
    ]

    # Run training
    try:
        result = subprocess.run(cmd, check=True)

        seed_time = time.time() - seed_start
        print(f"\n✅ Seed {seed} training complete in {seed_time / 60:.1f} minutes")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Seed {seed} training failed!")
        print(f"Error: {e}")
        sys.exit(1)

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("BATCH TRAINING COMPLETE")
print("=" * 80)
print(f"Total time: {total_time / 60:.1f} minutes")
print(f"Average time per seed: {total_time / len(SEEDS) / 60:.1f} minutes")
print("\nCheckpoints saved to:")
for seed in SEEDS:
    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "efficientnet_baseline"
        / f"seed_{seed}"
        / "best.pt"
    )
    if checkpoint_path.exists():
        print(f"  ✅ Seed {seed}: {checkpoint_path}")
    else:
        print(f"  ❌ Seed {seed}: NOT FOUND")
print("=" * 80)
