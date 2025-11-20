# Quick Start Training Script
# This script sets up Python 3.11 environment and starts training

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Tri-Objective Robust XAI Medical Imaging - Training Startup" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Define Python 3.11 path
$PYTHON311 = "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe"

# Check if Python 3.11 exists
if (-not (Test-Path $PYTHON311)) {
    Write-Host "[ERROR] Python 3.11 not found at: $PYTHON311" -ForegroundColor Red
    Write-Host "Please install Python 3.11 or update the path in this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] Verifying Python 3.11 environment..." -ForegroundColor Green
& $PYTHON311 --version
Write-Host ""

Write-Host "[2/4] Checking PyTorch installation..." -ForegroundColor Green
& $PYTHON311 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Write-Host ""

Write-Host "[3/4] Verifying GPU..." -ForegroundColor Green
& $PYTHON311 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
Write-Host ""

Write-Host "[4/4] Running environment tests..." -ForegroundColor Green
& $PYTHON311 scripts\test_environment.py
Write-Host ""

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Environment Ready! Choose training option:" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "Available Training Configs:" -ForegroundColor Yellow
Write-Host "  1. CIFAR-10 Debug (Quick test):" -ForegroundColor White
Write-Host "     & '$PYTHON311' -m src.training.train_baseline --config configs/experiments/debug.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. ISIC 2018 ResNet50 Baseline:" -ForegroundColor White
Write-Host "     & '$PYTHON311' -m src.training.train_baseline --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Custom Config:" -ForegroundColor White
Write-Host "     & '$PYTHON311' -m src.training.train_baseline --config <your_config.yaml>" -ForegroundColor Cyan
Write-Host ""

# Ask user which training to run
Write-Host "=" * 80 -ForegroundColor Cyan
$choice = Read-Host "Enter choice (1-3) or 'q' to quit"

switch ($choice) {
    "1" {
        Write-Host "`nStarting CIFAR-10 Debug Training..." -ForegroundColor Green
        & $PYTHON311 -m src.training.train_baseline --config configs/experiments/debug.yaml
    }
    "2" {
        Write-Host "`nStarting ISIC 2018 ResNet50 Training..." -ForegroundColor Green
        & $PYTHON311 -m src.training.train_baseline --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml
    }
    "3" {
        $config_path = Read-Host "Enter config path"
        Write-Host "`nStarting training with $config_path..." -ForegroundColor Green
        & $PYTHON311 -m src.training.train_baseline --config $config_path
    }
    "q" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
