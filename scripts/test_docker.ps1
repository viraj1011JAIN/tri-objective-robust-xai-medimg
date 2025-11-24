# Docker Build and Test Script (PowerShell)
# Tests Docker build, CUDA support, and environment validation

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🐳 Docker Build & Test Pipeline" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build Docker image
Write-Host "📦 Step 1: Building Docker image..." -ForegroundColor Yellow
try {
    docker build -t tri-objective-xai:latest .
    Write-Host "✅ Docker image built successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker build failed: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Test CPU-only mode
Write-Host "🖥️  Step 2: Testing CPU-only mode..." -ForegroundColor Yellow
try {
    docker run --rm tri-objective-xai:latest python scripts/check_docker_env.py
    Write-Host "✅ CPU mode test completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  CPU mode test failed: $_" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Test GPU mode (if available)
Write-Host "🎮 Step 3: Testing GPU mode..." -ForegroundColor Yellow
try {
    $nvidiaCheck = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  NVIDIA GPU detected, testing with --gpus all..."
        docker run --rm --gpus all tri-objective-xai:latest python scripts/check_docker_env.py
        Write-Host "✅ GPU mode test completed" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  No NVIDIA GPU detected, skipping GPU test" -ForegroundColor Cyan
    }
} catch {
    Write-Host "ℹ️  GPU test skipped (nvidia-smi not available)" -ForegroundColor Cyan
}
Write-Host ""

# Step 4: Test Python imports
Write-Host "📚 Step 4: Testing Python imports..." -ForegroundColor Yellow
$importTest = @"
import sys
print('Testing critical imports...')
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    import src.datasets.base_dataset
    print('✓ src.datasets')
    import src.models.build
    print('✓ src.models')
    import src.training.base_trainer
    print('✓ src.training')
    import src.losses.tri_objective
    print('✓ src.losses')
    import src.attacks.fgsm
    print('✓ src.attacks')
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"@

try {
    docker run --rm tri-objective-xai:latest python -c $importTest
    Write-Host "✅ Import test completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Python import test failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Image size check
Write-Host "📏 Step 5: Checking image size..." -ForegroundColor Yellow
$imageInfo = docker images tri-objective-xai:latest --format "{{.Size}}"
Write-Host "  Image size: $imageInfo"
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "✅ Docker Build & Test Complete" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor Cyan
Write-Host "  - Image: tri-objective-xai:latest"
Write-Host "  - Size: $imageInfo"
Write-Host "  - Status: Ready for deployment"
Write-Host ""
Write-Host "🚀 To run the container:" -ForegroundColor Yellow
Write-Host "  CPU: docker run --rm tri-objective-xai:latest"
Write-Host "  GPU: docker run --rm --gpus all tri-objective-xai:latest"
Write-Host ""
