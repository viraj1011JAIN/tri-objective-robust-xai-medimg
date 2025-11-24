# Phase 3.6: Baseline Evaluation - Chest X-Ray
# PowerShell script to run all baseline chest X-ray evaluations

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

# Set variables
$CHECKPOINT = "results/checkpoints/rq1_robustness/baseline_nih_resnet50/best.pt"
$MODEL = "resnet50"
$DEVICE = "cuda"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Phase 3.6: Baseline Evaluation - Chest X-Ray" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if checkpoint exists
if (-not (Test-Path $CHECKPOINT)) {
    Write-Host "⚠️  ERROR: Checkpoint not found: $CHECKPOINT" -ForegroundColor Red
    Write-Host "⚠️  Please complete Phase 3.6 training first!" -ForegroundColor Red
    Write-Host "`nSkipping evaluation (will run when training complete)..." -ForegroundColor Yellow
    exit 0
}

# 1. Evaluate on NIH ChestX-ray14 Test Set (Same-Site)
Write-Host "`n[1/2] Evaluating on NIH ChestX-ray14 test set (same-site)..." -ForegroundColor Green
python scripts/evaluation/evaluate_baseline_cxr.py `
    --checkpoint $CHECKPOINT `
    --model $MODEL `
    --dataset nih_chestxray14 `
    --split test `
    --batch-size 32 `
    --n-bootstrap 1000 `
    --threshold 0.5 `
    --output-dir results/evaluation/baseline_nih `
    --device $DEVICE

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation failed for NIH ChestX-ray14" -ForegroundColor Red
    Write-Host "⚠️  This is expected if dataset is not accessible (F:/data)" -ForegroundColor Yellow
}

# 2. Evaluate on PadChest (Cross-Site)
Write-Host "`n[2/2] Evaluating on PadChest (cross-site)..." -ForegroundColor Green
python scripts/evaluation/evaluate_baseline_cxr.py `
    --checkpoint $CHECKPOINT `
    --model $MODEL `
    --dataset padchest `
    --split test `
    --batch-size 32 `
    --n-bootstrap 1000 `
    --threshold 0.5 `
    --output-dir results/evaluation/baseline_padchest `
    --device $DEVICE

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation failed for PadChest" -ForegroundColor Red
    Write-Host "⚠️  This is expected if dataset is not accessible (F:/data)" -ForegroundColor Yellow
}

# 3. Aggregate Results and Compute Cross-Site AUROC Drop
Write-Host "`n[3/3] Aggregating results and computing cross-site AUROC drop..." -ForegroundColor Green
python scripts/evaluation/aggregate_baseline_cxr_results.py `
    --results-dir results/evaluation `
    --output-dir results/metrics/rq1_robustness

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Phase 3.6 Evaluation Complete!" -ForegroundColor Green
    Write-Host "`nResults saved to:" -ForegroundColor Cyan
    Write-Host "  - results/metrics/rq1_robustness/baseline_cxr.csv" -ForegroundColor White
    Write-Host "  - results/metrics/rq1_robustness/baseline_cxr_auroc_drop.json" -ForegroundColor White
    Write-Host "  - results/metrics/rq1_robustness/plots/" -ForegroundColor White
    Write-Host "  - results/evaluation/baseline_nih/" -ForegroundColor White
    Write-Host "  - results/evaluation/baseline_padchest/" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Aggregation failed or incomplete" -ForegroundColor Yellow
    Write-Host "This is expected if datasets are not accessible" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Phase 3.6 Evaluation Script Complete" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
