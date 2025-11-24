# Train PGD-AT Models (3 Seeds)
# ==============================
# This script trains PGD adversarial training models for Phase 5.2

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  TRAIN PGD-AT MODELS (3 SEEDS)" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

$config = "configs/experiments/pgd_at_isic.yaml"

if (-not (Test-Path $config)) {
    Write-Host "ERROR: Config file not found: $config" -ForegroundColor Red
    exit 1
}

Write-Host "Training configuration: $config" -ForegroundColor White
Write-Host ""
Write-Host "This will train 3 models with seeds: 42, 123, 456" -ForegroundColor Yellow
Write-Host "Estimated time: 6-12 hours total (2-4 hours per seed)" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Continue? (y/n)"
if ($response -ne "y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  TRAINING SEED 42" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

python scripts/training/train_pgd_at.py --config $config --seeds 42 --single_seed

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training failed for seed 42" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  TRAINING SEED 123" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

python scripts/training/train_pgd_at.py --config $config --seeds 123 --single_seed

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training failed for seed 123" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  TRAINING SEED 456" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

python scripts/training/train_pgd_at.py --config $config --seeds 456 --single_seed

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training failed for seed 456" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "  ALL TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Checkpoints saved to:" -ForegroundColor Cyan
Write-Host "  - checkpoints/pgd_at/seed_42/best.pt" -ForegroundColor White
Write-Host "  - checkpoints/pgd_at/seed_123/best.pt" -ForegroundColor White
Write-Host "  - checkpoints/pgd_at/seed_456/best.pt" -ForegroundColor White
Write-Host ""
Write-Host "Next step: Run evaluation pipeline" -ForegroundColor Cyan
Write-Host "  .\RUN_PHASE_5_2_COMPLETE.ps1" -ForegroundColor White
Write-Host ""
