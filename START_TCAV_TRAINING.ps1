# Quick Start Script - Train CAVs and Tri-Objective Model with TCAV
# ===================================================================
#
# This script runs the complete TCAV training pipeline:
# 1. Tests integration (30 seconds)
# 2. Trains CAVs (10-15 minutes)
# 3. Trains tri-objective model with TCAV (25-30 hours)
#
# Author: Viraj Pankaj Jain
# Date: December 7, 2025

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "TCAV TRAINING PIPELINE - QUICK START" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

Write-Host "✓ Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
Write-Host ""

# Step 1: Test Integration
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Testing Integration (30 seconds)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

python test_tcav_integration.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Integration test FAILED!" -ForegroundColor Red
    Write-Host "Please fix the issues before continuing." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Integration test PASSED!" -ForegroundColor Green
Write-Host ""

# Step 2: Train CAVs
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Training CAVs (10-15 minutes)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$cav_checkpoint = "checkpoints\cavs\trained_cavs.pt"

if (Test-Path $cav_checkpoint) {
    Write-Host "⚠️  CAV checkpoint already exists: $cav_checkpoint" -ForegroundColor Yellow
    $response = Read-Host "Retrain CAVs? (y/N)"

    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Skipping CAV training, using existing checkpoint." -ForegroundColor Yellow
    } else {
        Write-Host "Retraining CAVs..." -ForegroundColor Yellow
        python scripts\training\train_cavs_for_training.py `
            --data_dir data\processed\isic2018 `
            --model_checkpoint checkpoints\baseline\seed_42\best.pt `
            --output_dir checkpoints\cavs `
            --n_concept_samples 50 `
            --n_random_samples 100

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "❌ CAV training FAILED!" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Write-Host "Training CAVs for the first time..." -ForegroundColor Cyan
    python scripts\training\train_cavs_for_training.py `
        --data_dir data\processed\isic2018 `
        --model_checkpoint checkpoints\baseline\seed_42\best.pt `
        --output_dir checkpoints\cavs `
        --n_concept_samples 50 `
        --n_random_samples 100

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ CAV training FAILED!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "✅ CAV training COMPLETE!" -ForegroundColor Green
Write-Host "   Checkpoint: $cav_checkpoint" -ForegroundColor Green
Write-Host ""

# Step 3: Train Tri-Objective with TCAV
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Training Tri-Objective Model with TCAV" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Expected duration: 25-30 hours (RTX 3050)" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Start full training now? (y/N)"

if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "Starting tri-objective training with TCAV..." -ForegroundColor Cyan
    Write-Host ""

    python scripts\training\train_tri_objective.py `
        --config configs\experiments\tri_objective.yaml `
        --cavs-checkpoint $cav_checkpoint `
        --seed 42

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Training FAILED!" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "✅ TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Open notebooks\phase6_hypothesis_evaluation.ipynb" -ForegroundColor White
    Write-Host "2. Load the trained model: checkpoints\tri_objective\seed_42\best.pt" -ForegroundColor White
    Write-Host "3. Re-run all hypotheses → should see H2.1-H2.4 all PASS ✅" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Skipping full training." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To run training manually:" -ForegroundColor Cyan
    Write-Host "  python scripts\training\train_tri_objective.py ``" -ForegroundColor White
    Write-Host "      --config configs\experiments\tri_objective.yaml ``" -ForegroundColor White
    Write-Host "      --cavs-checkpoint $cav_checkpoint ``" -ForegroundColor White
    Write-Host "      --seed 42" -ForegroundColor White
    Write-Host ""
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "QUICK START SCRIPT COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
