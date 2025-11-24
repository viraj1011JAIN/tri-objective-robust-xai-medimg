# Phase 3.5: Baseline Evaluation - Dermoscopy
# PowerShell script to run all baseline evaluations

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

# Set variables
$CHECKPOINT = "results/checkpoints/rq1_robustness/baseline_isic2018_resnet50/best.pt"
$MODEL = "resnet50"
$DEVICE = "cuda"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Phase 3.5: Baseline Evaluation - Dermoscopy" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if checkpoint exists
if (-not (Test-Path $CHECKPOINT)) {
    Write-Host "⚠️  ERROR: Checkpoint not found: $CHECKPOINT" -ForegroundColor Red
    Write-Host "⚠️  Please complete Phase 3.4 training first!" -ForegroundColor Red
    Write-Host "`nSkipping evaluation (will run when training complete)..." -ForegroundColor Yellow
    exit 0
}

# 1. Evaluate on ISIC 2018 Test Set (Same-Site)
Write-Host "`n[1/4] Evaluating on ISIC 2018 test set (same-site)..." -ForegroundColor Green
python scripts/evaluation/evaluate_baseline.py `
    --checkpoint $CHECKPOINT `
    --model $MODEL `
    --dataset isic2018 `
    --split test `
    --batch-size 32 `
    --n-bootstrap 1000 `
    --output-dir results/evaluation/baseline_isic2018 `
    --device $DEVICE

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation failed for ISIC 2018" -ForegroundColor Red
    Write-Host "⚠️  This is expected if dataset is not accessible (/content/drive/MyDrive/data)" -ForegroundColor Yellow
}

# 2. Evaluate on ISIC 2019 (Cross-Site)
Write-Host "`n[2/4] Evaluating on ISIC 2019 (cross-site)..." -ForegroundColor Green
python scripts/evaluation/evaluate_baseline.py `
    --checkpoint $CHECKPOINT `
    --model $MODEL `
    --dataset isic2019 `
    --split test `
    --batch-size 32 `
    --n-bootstrap 1000 `
    --output-dir results/evaluation/baseline_isic2019 `
    --device $DEVICE

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation failed for ISIC 2019" -ForegroundColor Red
}

# 3. Evaluate on ISIC 2020 (Cross-Site)
Write-Host "`n[3/4] Evaluating on ISIC 2020 (cross-site)..." -ForegroundColor Green
python scripts/evaluation/evaluate_baseline.py `
    --checkpoint $CHECKPOINT `
    --model $MODEL `
    --dataset isic2020 `
    --split test `
    --batch-size 32 `
    --n-bootstrap 1000 `
    --output-dir results/evaluation/baseline_isic2020 `
    --device $DEVICE

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation failed for ISIC 2020" -ForegroundColor Red
}

# 4. Evaluate on Derm7pt (Cross-Site)
Write-Host "`n[4/4] Evaluating on Derm7pt (cross-site)..." -ForegroundColor Green
python scripts/evaluation/evaluate_baseline.py `
    --checkpoint $CHECKPOINT `
    --model $MODEL `
    --dataset derm7pt `
    --split test `
    --batch-size 32 `
    --n-bootstrap 1000 `
    --output-dir results/evaluation/baseline_derm7pt `
    --device $DEVICE

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation failed for Derm7pt" -ForegroundColor Red
}

# 5. Aggregate Results
Write-Host "`n[5/5] Aggregating results across all datasets..." -ForegroundColor Green
python scripts/evaluation/aggregate_baseline_results.py `
    --results-dir results/evaluation `
    --output-dir results/metrics/rq1_robustness

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Phase 3.5 Evaluation Complete!" -ForegroundColor Green
    Write-Host "`nResults saved to:" -ForegroundColor Cyan
    Write-Host "  - results/metrics/rq1_robustness/baseline.csv" -ForegroundColor White
    Write-Host "  - results/metrics/rq1_robustness/plots/" -ForegroundColor White
    Write-Host "  - results/evaluation/baseline_*/" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Aggregation failed or incomplete" -ForegroundColor Yellow
    Write-Host "This is expected if datasets are not accessible" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Phase 3.5 Evaluation Script Complete" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
