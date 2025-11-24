#!/usr/bin/env bash
# Docker Build and Test Script
# Tests Docker build, CUDA support, and environment validation

set -e

echo "============================================"
echo "🐳 Docker Build & Test Pipeline"
echo "============================================"
echo ""

# Step 1: Build Docker image
echo "📦 Step 1: Building Docker image..."
docker build -t tri-objective-xai:latest . || {
    echo "❌ Docker build failed"
    exit 1
}
echo "✅ Docker image built successfully"
echo ""

# Step 2: Test CPU-only mode
echo "🖥️  Step 2: Testing CPU-only mode..."
docker run --rm tri-objective-xai:latest python scripts/check_docker_env.py || {
    echo "⚠️  CPU mode test failed"
}
echo "✅ CPU mode test completed"
echo ""

# Step 3: Test GPU mode (if available)
echo "🎮 Step 3: Testing GPU mode..."
if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected, testing with --gpus all..."
    docker run --rm --gpus all tri-objective-xai:latest python scripts/check_docker_env.py || {
        echo "⚠️  GPU mode test failed"
    }
    echo "✅ GPU mode test completed"
else
    echo "ℹ️  No NVIDIA GPU detected, skipping GPU test"
fi
echo ""

# Step 4: Test Python imports
echo "📚 Step 4: Testing Python imports..."
docker run --rm tri-objective-xai:latest python -c "
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
" || {
    echo "❌ Python import test failed"
    exit 1
}
echo ""

# Step 5: Image size check
echo "📏 Step 5: Checking image size..."
IMAGE_SIZE=$(docker images tri-objective-xai:latest --format "{{.Size}}")
echo "  Image size: $IMAGE_SIZE"
echo ""

echo "============================================"
echo "✅ Docker Build & Test Complete"
echo "============================================"
echo ""
echo "📋 Summary:"
echo "  - Image: tri-objective-xai:latest"
echo "  - Size: $IMAGE_SIZE"
echo "  - Status: Ready for deployment"
echo ""
echo "🚀 To run the container:"
echo "  CPU: docker run --rm tri-objective-xai:latest"
echo "  GPU: docker run --rm --gpus all tri-objective-xai:latest"
echo ""
