# Phase 5.2: PGD Adversarial Training
# ====================================
# Quick reference script for training and evaluation

# Setup environment
Set-Location C:\Users\Dissertation\tri-objective-robust-xai-medimg
& .venv\Scripts\Activate.ps1

# =============================================================================
# Training Commands
# =============================================================================

# Single-seed training (for testing)
python scripts/training/train_pgd_at.py `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at/seed_42 `
    --single_seed `
    --seeds 42

# Multi-seed training (3 seeds for statistical significance)
python scripts/training/train_pgd_at.py `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at `
    --seeds 42 123 456 `
    --mlflow_experiment "Phase5.2-PGD-AT"

# Resume training from checkpoint
python scripts/training/train_pgd_at.py `
    --config configs/experiments/pgd_at_isic.yaml `
    --output_dir results/pgd_at/seed_42 `
    --resume results/pgd_at/seed_42/checkpoints/last.pt `
    --single_seed

# =============================================================================
# Evaluation Commands
# =============================================================================

# Evaluate all trained models (3 seeds)
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
    --output_dir results/pgd_at/evaluation_single

# =============================================================================
# Monitor Training (MLflow)
# =============================================================================

# Start MLflow UI
mlflow ui --backend-store-uri mlruns --port 5000

# Then open: http://localhost:5000

# =============================================================================
# Results Location
# =============================================================================

# Training results:
#   results/pgd_at/seed_*/checkpoints/       - Model checkpoints
#   results/pgd_at/seed_*/results/           - Training history
#   results/pgd_at/statistical_summary.json  - Aggregated statistics

# Evaluation results:
#   results/pgd_at/evaluation/pgd_at_summary.csv     - Summary table
#   results/pgd_at/evaluation/pgd_at_detailed.csv    - Detailed results
#   results/pgd_at/evaluation/figures/               - Visualizations
#   results/metrics/rq1_robustness/pgd_at.csv        - RQ1 results

# =============================================================================
# Expected Results (ISIC 2018)
# =============================================================================

# Clean Accuracy: 72-78% (slight drop vs baseline ~80-85%)
# Robust Accuracy (PGD-10): 45-55%
# Training Time: ~3-4 hours per seed on GPU (50 epochs)

# Cross-site Performance (Hypothesis for RQ1):
#   ISIC 2019: Similar (~2-3% drop)
#   ISIC 2020: Similar (~2-3% drop)
#   Derm7pt: Larger drop (~5-7% drop)
#   Expected: PGD-AT alone does NOT significantly improve cross-site generalization

# =============================================================================
# Troubleshooting
# =============================================================================

# CUDA out of memory:
#   - Reduce batch_size in config (32 -> 16)
#   - Disable AMP: use_amp: false
#   - Reduce num_workers: num_workers: 2

# Training too slow:
#   - Enable AMP: use_amp: true
#   - Increase num_workers: num_workers: 8
#   - Use mixed training: mix_clean: 0.5

# Poor convergence:
#   - Reduce learning rate: learning_rate: 5.0e-5
#   - Increase gradient_clip: gradient_clip: 2.0
#   - Adjust scheduler patience: patience: 10

Write-Host "`nPhase 5.2 PGD-AT setup complete!" -ForegroundColor Green
Write-Host "Run commands above to train and evaluate PGD-AT models" -ForegroundColor Cyan
