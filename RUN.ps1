#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Single-command launcher for Phase 7.7 tri-objective training.

.DESCRIPTION
    Ultra-simple wrapper that just works. No configuration needed.

.EXAMPLE
    .\RUN.ps1

.NOTES
    This is the simplest way to start Phase 7.7 training.
#>

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                      ║" -ForegroundColor Cyan
Write-Host "║              PHASE 7.7 - TRI-OBJECTIVE TRAINING                      ║" -ForegroundColor Cyan
Write-Host "║                                                                      ║" -ForegroundColor Cyan
Write-Host "║              Tri-Objective Robust XAI for Medical Imaging           ║" -ForegroundColor Cyan
Write-Host "║              University of Glasgow                                   ║" -ForegroundColor Cyan
Write-Host "║                                                                      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting automated training pipeline..." -ForegroundColor Yellow
Write-Host ""

# Execute the main training script
& ".\START_PHASE_7.7_TRAINING.ps1"

Write-Host ""
Write-Host "Training session complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review results in MLflow: mlflow ui --port 5000" -ForegroundColor Yellow
Write-Host "  2. Check checkpoints: results/checkpoints/tri_objective" -ForegroundColor Yellow
Write-Host "  3. Read logs: results/logs/training" -ForegroundColor Yellow
Write-Host ""
