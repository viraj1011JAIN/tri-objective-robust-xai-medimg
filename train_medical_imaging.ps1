# One-Command Medical Imaging Training
# Quick start for ISIC 2018 ResNet50 baseline training

$PYTHON311 = "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe"

Write-Host "`n🚀 Starting Medical Imaging Training - ISIC 2018 with ResNet50" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

# Verify environment
Write-Host "`n✓ Verifying environment..." -ForegroundColor Yellow
& $PYTHON311 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# Start training
Write-Host "`n✓ Starting training..." -ForegroundColor Yellow
Write-Host "Config: configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml`n" -ForegroundColor Cyan

& $PYTHON311 -m src.training.train_baseline --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml --device cuda

Write-Host "`n✅ Training session ended" -ForegroundColor Green
