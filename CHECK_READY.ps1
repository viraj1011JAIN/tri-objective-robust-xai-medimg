#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Pre-flight check for Phase 7.7 training readiness.

.DESCRIPTION
    Verifies all requirements are met before starting training.
#>

$script:allChecks = $true

function Test-Requirement {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$SuccessMessage,
        [string]$FailureMessage,
        [string]$FixCommand = ""
    )

    Write-Host "Checking $Name... " -NoNewline

    try {
        $result = & $Test
        if ($result) {
            Write-Host "✅" -ForegroundColor Green
            if ($SuccessMessage) {
                Write-Host "  $SuccessMessage" -ForegroundColor DarkGray
            }
            return $true
        } else {
            throw "Check failed"
        }
    } catch {
        Write-Host "❌" -ForegroundColor Red
        Write-Host "  $FailureMessage" -ForegroundColor Red
        if ($FixCommand) {
            Write-Host "  Fix: $FixCommand" -ForegroundColor Yellow
        }
        $script:allChecks = $false
        return $false
    }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     PHASE 7.7 PRE-FLIGHT CHECK                      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check 1: Python
Test-Requirement `
    -Name "Python Installation" `
    -Test { (python --version 2>&1) -match "Python 3\." } `
    -SuccessMessage "Python 3.x detected" `
    -FailureMessage "Python 3.x not found in PATH" `
    -FixCommand "Install Python 3.8+ or add to PATH"

# Check 2: Virtual Environment
Test-Requirement `
    -Name "Virtual Environment" `
    -Test { Test-Path ".venv/Scripts/Activate.ps1" } `
    -SuccessMessage "Virtual environment found" `
    -FailureMessage "Virtual environment not found" `
    -FixCommand "python -m venv .venv"

# Check 3: PyTorch
Test-Requirement `
    -Name "PyTorch" `
    -Test { (python -c "import torch; print('OK')" 2>&1) -eq "OK" } `
    -SuccessMessage "PyTorch installed" `
    -FailureMessage "PyTorch not installed" `
    -FixCommand "pip install torch torchvision"

# Check 4: CUDA
Test-Requirement `
    -Name "CUDA Availability" `
    -Test { (python -c "import torch; print(torch.cuda.is_available())" 2>&1) -eq "True" } `
    -SuccessMessage "CUDA available" `
    -FailureMessage "CUDA not available (will use CPU - much slower)" `
    -FixCommand "Install CUDA toolkit or train on CPU"

# Check 5: MLflow
Test-Requirement `
    -Name "MLflow" `
    -Test { (python -c "import mlflow; print('OK')" 2>&1) -eq "OK" } `
    -SuccessMessage "MLflow installed" `
    -FailureMessage "MLflow not installed" `
    -FixCommand "pip install mlflow"

# Check 6: Configuration File
Test-Requirement `
    -Name "Configuration File" `
    -Test { Test-Path "configs/experiments/tri_objective.yaml" } `
    -SuccessMessage "tri_objective.yaml found" `
    -FailureMessage "Configuration file not found" `
    -FixCommand "Ensure configs/experiments/tri_objective.yaml exists"

# Check 7: Training Script
Test-Requirement `
    -Name "Training Script" `
    -Test { Test-Path "scripts/train_tri_objective.py" } `
    -SuccessMessage "train_tri_objective.py found" `
    -FailureMessage "Training script not found" `
    -FixCommand "Ensure scripts/train_tri_objective.py exists"

# Check 8: Data Directory
Test-Requirement `
    -Name "Data Directory" `
    -Test { Test-Path "data/processed/isic2018" } `
    -SuccessMessage "ISIC2018 data found" `
    -FailureMessage "ISIC2018 data not found" `
    -FixCommand "Run data preparation script"

# Check 9: Disk Space
Test-Requirement `
    -Name "Disk Space (>10GB)" `
    -Test {
        $drive = Get-PSDrive -Name (Get-Location).Drive.Name
        ($drive.Free / 1GB) -gt 10
    } `
    -SuccessMessage "Sufficient disk space" `
    -FailureMessage "Less than 10GB free space" `
    -FixCommand "Free up disk space"

# Check 10: GPU Memory
if ((python -c "import torch; print(torch.cuda.is_available())" 2>&1) -eq "True") {
    Test-Requirement `
        -Name "GPU Memory (>6GB)" `
        -Test {
            $mem = python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1024**3)" 2>&1
            [float]$mem -gt 6
        } `
        -SuccessMessage "Sufficient GPU memory" `
        -FailureMessage "Less than 6GB GPU memory" `
        -FixCommand "Reduce batch size in config or use gradient accumulation"
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($script:allChecks) {
    Write-Host ""
    Write-Host "✅ ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host ""
    Write-Host "You're ready to start training!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To begin, run:" -ForegroundColor Cyan
    Write-Host "  .\RUN.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or:" -ForegroundColor Cyan
    Write-Host "  .\START_PHASE_7.7_TRAINING.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 0
} else {
    Write-Host ""
    Write-Host "❌ SOME CHECKS FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the issues above before starting training." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
