# Quick Start Training Scripts
# =============================================================================
# Use these commands to run training with the correct Python environment

## OPTION 1: Source the environment setup (Recommended)
## This sets up Python 3.11 as default for your current session
.\setup_python_env.ps1

# Then you can use python normally:
python scripts/test_environment.py
python scripts/validate_repository.py
python -m src.training.train_baseline --config configs/experiments/debug.yaml


## OPTION 2: Use wrapper scripts directly (Alternative)
.\train_baseline.ps1 --config configs/experiments/debug.yaml


## OPTION 3: Use Python 3.11 explicitly (Always works)
& "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe" -m src.training.train_baseline --config configs/experiments/debug.yaml


# =============================================================================
# Available Experiment Configs
# =============================================================================

# 1. Debug experiment (CIFAR-10)
python -m src.training.train_baseline --config configs/experiments/debug.yaml

# 2. CIFAR-10 baseline
python -m src.training.train_baseline --config configs/experiments/cifar10_debug_baseline.yaml

# 3. ISIC 2018 with ResNet50 (Medical imaging)
python -m src.training.train_baseline --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml


# =============================================================================
# Quick Test Commands
# =============================================================================

# Test environment
python scripts/test_environment.py

# Validate repository
python scripts/validate_repository.py

# Train debug (1 epoch, small dataset)
python scripts/train_cifar10_debug.py --epochs 1 --batch-size 32


# =============================================================================
# Common Training Options
# =============================================================================

# Specify device
python -m src.training.train_baseline --config <config.yaml> --device cuda

# Custom seed
python -m src.training.train_baseline --config <config.yaml> --seed 42

# Custom checkpoint directory
python -m src.training.train_baseline --config <config.yaml> --checkpoint-dir checkpoints/my_exp

# Show help
python -m src.training.train_baseline --help
