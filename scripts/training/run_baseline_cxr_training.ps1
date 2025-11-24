# Phase 3.6: Baseline Training - Chest X-Ray (NIH ChestX-ray14)
# PowerShell script to train baseline models with 3 random seeds

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Phase 3.6: Baseline Training - Chest X-Ray" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if datasets are accessible
if (-not (Test-Path "/content/drive/MyDrive/data/NIH_ChestXray14")) {
    Write-Host "⚠️  ERROR: Dataset not found at/content/drive/MyDrive/data/NIH_ChestXray14" -ForegroundColor Red
    Write-Host "⚠️  Please ensure the external HDD is connected and accessible" -ForegroundColor Red
    Write-Host "`nSkipping training (will run when dataset is available)..." -ForegroundColor Yellow
    exit 0
}

# Training parameters
$MODEL = "resnet50"
$DATASET = "nih_chestxray14"
$CONFIG = "configs/experiments/rq1_robustness/baseline_nih_resnet50.yaml"
$SEEDS = @(42, 123, 456)

# Train with 3 different seeds
foreach ($SEED in $SEEDS) {
    Write-Host "`n[Training with seed $SEED]" -ForegroundColor Green
    Write-Host "Model: $MODEL" -ForegroundColor White
    Write-Host "Dataset: $DATASET" -ForegroundColor White
    Write-Host "Random seed: $SEED" -ForegroundColor White
    Write-Host ""

    # Run training
    python src/training/train_baseline.py `
        --config $CONFIG `
        --seed $SEED `
        --device cuda

    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Training failed for seed $SEED" -ForegroundColor Red
        Write-Host "Continuing with next seed..." -ForegroundColor Yellow
        continue
    }

    Write-Host "✅ Training complete for seed $SEED" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Training Runs Complete" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Aggregate results across seeds
Write-Host "Aggregating results across seeds..." -ForegroundColor Green
python scripts/training/aggregate_seed_results.py `
    --experiment-name rq1_baseline_nih_resnet50 `
    --output-dir results/metrics/rq1_robustness

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Phase 3.6 Training Complete!" -ForegroundColor Green
    Write-Host "`nCheckpoints saved to:" -ForegroundColor Cyan
    Write-Host "  - results/checkpoints/rq1_robustness/baseline_nih_resnet50/" -ForegroundColor White
    Write-Host "`nNext step: Run evaluation with:" -ForegroundColor Cyan
    Write-Host "  .\scripts\evaluation\run_baseline_cxr_evaluation.ps1" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Aggregation failed" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Phase 3.6 Training Script Complete" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
