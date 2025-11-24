# Phase 5.2: Complete Pipeline Execution
# ======================================
# This script runs the complete Phase 5.2 evaluation using your existing checkpoints

# Activate virtual environment first!
# .\.venv\Scripts\Activate.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  PHASE 5.2 - COMPLETE PIPELINE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if baseline checkpoints exist
$baseline_paths = @(
    "checkpoints\baseline\seed_42\best.pt",
    "checkpoints\baseline\seed_123\best.pt",
    "checkpoints\baseline\seed_456\best.pt"
)

$missing_baseline = $false
foreach ($path in $baseline_paths) {
    if (-not (Test-Path $path)) {
        Write-Host "WARNING: Missing baseline checkpoint: $path" -ForegroundColor Yellow
        $missing_baseline = $true
    }
}

# For now, use baseline checkpoints as both baseline and PGD-AT
# (You'll need to train PGD-AT models or specify correct paths)
$pgd_at_paths = @(
    "checkpoints\baseline\seed_42\best.pt",
    "checkpoints\baseline\seed_123\best.pt",
    "checkpoints\baseline\seed_456\best.pt"
)

if ($missing_baseline) {
    Write-Host ""
    Write-Host "Some checkpoints are missing. Please specify correct paths." -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage example:" -ForegroundColor Cyan
    Write-Host "python scripts/phase_5_2_complete_pipeline.py \" -ForegroundColor White
    Write-Host "    --baseline-checkpoints checkpoints/baseline/seed_42/best.pt checkpoints/baseline/seed_123/best.pt checkpoints/baseline/seed_456/best.pt \" -ForegroundColor White
    Write-Host "    --pgd-at-checkpoints checkpoints/pgd_at/seed_42/best.pt checkpoints/pgd_at/seed_123/best.pt checkpoints/pgd_at/seed_456/best.pt" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "SUCCESS: Found all baseline checkpoints" -ForegroundColor Green
Write-Host ""
Write-Host "Running complete Phase 5.2 pipeline..." -ForegroundColor Cyan
Write-Host ""

# Run the pipeline
python scripts/phase_5_2_complete_pipeline.py `
    --config configs/base.yaml `
    --baseline-checkpoints $baseline_paths `
    --pgd-at-checkpoints $pgd_at_paths `
    --device cuda

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "  PHASE 5.2 COMPLETE!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: results/phase_5_2_complete/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Key files:" -ForegroundColor Cyan
    Write-Host "  - rq1_hypothesis_test.json  (RQ1 answer)" -ForegroundColor White
    Write-Host "  - results_table.csv         (Complete results)" -ForegroundColor White
    Write-Host "  - results_table.tex         (For dissertation)" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "WARNING: Pipeline failed. Check logs above." -ForegroundColor Red
    Write-Host ""
}
