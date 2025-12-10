# Quick Start Guide: Phase 5.2 - PGD Adversarial Training

This guide provides quick instructions for running Phase 5.2 experiments on PGD-based adversarial training.

## Prerequisites

- Python environment activated (`.venv`)
- ISIC2018 dataset prepared
- Configuration files in `configs/experiments/`

## Quick Commands

### 1. Single-Seed Training (Testing)

```powershell
python scripts/training/train_pgd_at.py `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at/seed_42 `
    --single_seed `
    --seeds 42
```

### 2. Multi-Seed Training (Full Experiment)

```powershell
python scripts/training/train_pgd_at.py `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at `
    --seeds 42 123 456 `
    --mlflow_experiment "Phase5.2-PGD-AT"
```

### 3. Resume Training from Checkpoint

```powershell
python scripts/training/train_pgd_at.py `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at/seed_42 `
    --resume results/pgd_at/seed_42/checkpoints/last.pt `
    --single_seed
```

### 4. Evaluate Trained Models

```powershell
# Evaluate all seeds
python scripts/evaluation/evaluate_pgd_at.py `
    --model_paths `
        results/pgd_at/seed_42/checkpoints/best.pt `
        results/pgd_at/seed_123/checkpoints/best.pt `
        results/pgd_at/seed_456/checkpoints/best.pt `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at/evaluation

# Evaluate single model
python scripts/evaluation/evaluate_pgd_at.py `
    --model_paths results/pgd_at/seed_42/checkpoints/best.pt `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at/single_eval
```

## Expected Outputs

- **Checkpoints**: `results/pgd_at/seed_*/checkpoints/`
- **Training logs**: `results/pgd_at/seed_*/training.log`
- **Evaluation results**: `results/pgd_at/evaluation/`
- **MLflow tracking**: `mlruns/`

## Configuration

The main configuration file is `configs/experiments/pgd_at_isic.yaml`. Key parameters:

- **Attack parameters**: `epsilon`, `alpha`, `num_steps`
- **Training**: `epochs`, `batch_size`, `learning_rate`
- **Model**: `architecture`, `pretrained`

## Troubleshooting

- **Out of memory**: Reduce `batch_size` in config
- **Slow training**: Ensure CUDA is available
- **Missing checkpoints**: Check `output_dir` path

For detailed documentation, see `PHASE_5.2_COMMANDS.ps1` and Phase 5.2 notebooks.
