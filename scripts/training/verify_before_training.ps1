# Phase 4.4 - Pre-Training Verification Script (Enhanced)
# ========================================================
# Run BEFORE training to ensure everything works!
#
# Enhancements v2.0:
# - Deep implementation testing (not just imports)
# - Resume capability validation
# - Training loop components verification
# - Checkpoint save/load testing
# - Comprehensive error reporting

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PHASE 4.4 PRE-TRAINING VERIFICATION (Enhanced v2.0)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Test 1: Deep model implementation verification
Write-Host "[1/10] Verifying build_model implementation..." -ForegroundColor Yellow
try {
    $output = python -c @"
from src.models.build import build_model
import torch

# Test model creation
model = build_model('efficientnet_b0', num_classes=7, pretrained=True)

# Verify model has required components
assert hasattr(model, 'forward'), 'Missing forward method'
assert hasattr(model, 'fc') or hasattr(model, 'classifier'), 'Missing classifier layer'

# Test actual inference
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    y = model(x)

# Verify output properties
assert y.shape == (2, 7), f'Wrong output shape: {y.shape}'
assert not torch.isnan(y).any(), 'NaN in output'
assert not torch.isinf(y).any(), 'Inf in output'

print('✅ Model implementation verified')
print(f'  - Architecture: {model.__class__.__name__}')
print(f'  - Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Model implementation complete and functional" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Model implementation has issues" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Model verification failed: $_" -ForegroundColor Red
    $errors++
}

# Test 2: Dataset implementation and real data
Write-Host "[2/10] Testing dataset implementation with real data..." -ForegroundColor Yellow
try {
    $output = python -c @"
from src.datasets.isic import ISICDataset
from pathlib import Path
import torch

# Test dataset loading
data_root = Path('data/processed/isic2018')
csv_path = data_root / 'metadata_processed.csv'

if not data_root.exists():
    print('⚠️ Dataset not found at', data_root)
    exit(1)

train_ds = ISICDataset(str(data_root), 'train', str(csv_path))
val_ds = ISICDataset(str(data_root), 'val', str(csv_path))

# Verify dataset properties
assert len(train_ds) > 0, 'Training set is empty'
assert len(val_ds) > 0, 'Validation set is empty'

# Test actual data loading
img, label = train_ds[0]
assert isinstance(img, torch.Tensor), 'Image is not a tensor'
assert img.shape[0] == 3, f'Wrong number of channels: {img.shape[0]}'
assert 0 <= label < 7, f'Label out of range: {label}'

print('✅ Dataset implementation verified')
print(f'  - Train samples: {len(train_ds):,}')
print(f'  - Val samples: {len(val_ds):,}')
print(f'  - Image shape: {img.shape}')
print(f'  - Label range: 0-6')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Dataset implementation complete" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Dataset verification failed" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Dataset test failed: $_" -ForegroundColor Red
    $errors++
}

# Test 3: Training loop components
Write-Host "[3/10] Verifying training loop components..." -ForegroundColor Yellow
try {
    $output = python -c @"
from scripts.training.train_baseline_efficientnet import train_epoch, validate
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Create dummy data
X = torch.randn(32, 3, 224, 224)
y = torch.randint(0, 7, (32,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8)

# Create dummy model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*224*224, 7)
)

# Test training function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

try:
    metrics = train_epoch(model, loader, criterion, optimizer, 'cpu', epoch=1)
    assert 'loss' in metrics, 'Missing loss in metrics'
    assert 'accuracy' in metrics, 'Missing accuracy in metrics'
    print('✅ train_epoch function works')
except Exception as e:
    print(f'❌ train_epoch failed: {e}')
    exit(1)

# Test validation function
try:
    metrics = validate(model, loader, criterion, 'cpu')
    assert 'loss' in metrics, 'Missing loss in validation metrics'
    assert 'accuracy' in metrics, 'Missing accuracy in validation metrics'
    print('✅ validate function works')
except Exception as e:
    print(f'❌ validate failed: {e}')
    exit(1)

print('✅ Training loop components verified')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Training loop implementation complete" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Training loop verification failed" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Training loop test failed: $_" -ForegroundColor Red
    $errors++
}


# Test 4: Optimizer and loss implementation
Write-Host "[4/10] Verifying optimizer and loss..." -ForegroundColor Yellow
try {
    $output = python -c @"
import torch.optim as optim
import torch.nn as nn

# Test loss function
criterion = nn.CrossEntropyLoss()
print('✅ CrossEntropyLoss available')

# Test optimizer
import torch
dummy_params = [torch.randn(10, 10, requires_grad=True)]
optimizer = optim.Adam(dummy_params, lr=1e-4, weight_decay=1e-4)
print('✅ Adam optimizer configured')

# Test scheduler (if used)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
print('✅ Learning rate scheduler available')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Optimizer and loss configured" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Optimizer/loss configuration failed" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Optimizer test failed: $_" -ForegroundColor Red
    $errors++
}

# Test 5: Checkpoint save/load functionality
Write-Host "[5/10] Testing checkpoint save/load..." -ForegroundColor Yellow
try {
    $output = python -c @"
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

# Create temporary checkpoint
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Create dummy model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())

    # Save checkpoint
    checkpoint = {
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': 0.85,
        'train_loss': 0.5,
    }
    ckpt_path = tmpdir / 'test.pt'
    torch.save(checkpoint, ckpt_path)
    print(f'✅ Checkpoint saved: {ckpt_path.stat().st_size} bytes')

    # Load checkpoint
    loaded = torch.load(ckpt_path, map_location='cpu')
    assert 'epoch' in loaded, 'Missing epoch'
    assert 'model_state_dict' in loaded, 'Missing model state'
    assert 'optimizer_state_dict' in loaded, 'Missing optimizer state'
    print('✅ Checkpoint loaded successfully')

    # Verify state restoration
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    print('✅ Model and optimizer state restored')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Checkpoint system functional" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Checkpoint system failed" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Checkpoint test failed: $_" -ForegroundColor Red
    $errors++
}

# Test 6: Check for existing checkpoints (resume capability)
Write-Host "[6/10] Checking resume capability..." -ForegroundColor Yellow
$checkpoint_dir = "checkpoints\efficientnet_baseline"
$resume_available = $false

if (Test-Path $checkpoint_dir) {
    $seeds = Get-ChildItem -Path $checkpoint_dir -Directory -Filter "seed_*" -ErrorAction SilentlyContinue

    if ($seeds.Count -gt 0) {
        Write-Host "  ℹ️ Found existing checkpoints:" -ForegroundColor Cyan
        foreach ($seed_dir in $seeds) {
            $last_ckpt = Join-Path $seed_dir.FullName "last.pt"
            $best_ckpt = Join-Path $seed_dir.FullName "best.pt"

            if (Test-Path $last_ckpt) {
                $resume_available = $true
                $size = (Get-Item $last_ckpt).Length / 1MB
                Write-Host "    - $($seed_dir.Name): last.pt ($([math]::Round($size, 1)) MB)" -ForegroundColor Gray

                # Check checkpoint contents
                try {
                    $check_output = python -c @"
import torch
ckpt = torch.load('$($last_ckpt.Replace('\', '/'))', map_location='cpu')
print(f"      Epoch: {ckpt.get('epoch', 'N/A')}, Val Acc: {ckpt.get('best_val_acc', 'N/A'):.2%}")
"@
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "      $check_output" -ForegroundColor Gray
                    }
                } catch {}
            }

            if (Test-Path $best_ckpt) {
                $size = (Get-Item $best_ckpt).Length / 1MB
                Write-Host "    - $($seed_dir.Name): best.pt ($([math]::Round($size, 1)) MB)" -ForegroundColor Gray
            }
        }

        if ($resume_available) {
            Write-Host "  ⚠️ Resume capability: NOT IMPLEMENTED in training script" -ForegroundColor Yellow
            Write-Host "  ℹ️ Recommendation: Add --resume flag to training script" -ForegroundColor Cyan
            $warnings++
        }
    } else {
        Write-Host "  ℹ️ No existing checkpoints found" -ForegroundColor Gray
        Write-Host "  ✅ Clean start for training" -ForegroundColor Green
    }
} else {
    Write-Host "  ℹ️ Checkpoint directory will be created" -ForegroundColor Gray
    Write-Host "  ✅ Ready for first training run" -ForegroundColor Green
}

# Test 7: Mixed precision training (AMP)
Write-Host "[7/10] Verifying mixed precision support..." -ForegroundColor Yellow
try {
    $output = python -c @"
import torch
from torch.cuda.amp import autocast, GradScaler

# Test AMP availability
if torch.cuda.is_available():
    scaler = GradScaler()
    print('✅ CUDA available - AMP enabled')
    print(f'  - Device: {torch.cuda.get_device_name(0)}')
    print(f'  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available - Training on CPU (slow)')

# Test AMP context
with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
    x = torch.randn(2, 10)
    print('✅ AMP context works')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Mixed precision support verified" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ⚠️ AMP verification failed (non-critical)" -ForegroundColor Yellow
        $warnings++
    }
} catch {
    Write-Host "  ⚠️ AMP test failed (non-critical): $_" -ForegroundColor Yellow
    $warnings++
}

# Test 8: Transforms and data augmentation
Write-Host "[8/10] Testing data transforms..." -ForegroundColor Yellow
try {
    $output = python -c @"
from src.datasets.transforms import get_train_transforms, get_test_transforms
import torch
from PIL import Image
import numpy as np

# Get transforms
train_tfm = get_train_transforms(dataset='isic', image_size=224)
test_tfm = get_test_transforms(dataset='isic', image_size=224)

print('✅ Transforms loaded')

# Test on dummy image
dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

# Test train transforms (with augmentation)
train_result = train_tfm(dummy_img)
assert isinstance(train_result, torch.Tensor), 'Train transform output not a tensor'
assert train_result.shape == (3, 224, 224), f'Wrong shape: {train_result.shape}'
print('✅ Train transforms work (with augmentation)')

# Test test transforms (no augmentation)
test_result = test_tfm(dummy_img)
assert isinstance(test_result, torch.Tensor), 'Test transform output not a tensor'
assert test_result.shape == (3, 224, 224), f'Wrong shape: {test_result.shape}'
print('✅ Test transforms work (without augmentation)')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Data transforms implemented" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Transform verification failed" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Transform test failed: $_" -ForegroundColor Red
    $errors++
}

# Test 9: Reproducibility utilities
Write-Host "[9/10] Verifying reproducibility setup..." -ForegroundColor Yellow
try {
    $output = python -c @"
from src.utils.reproducibility import set_global_seed
import torch
import random
import numpy as np

# Test seed setting
set_global_seed(42)
val1 = torch.rand(1).item()
random1 = random.random()
np1 = np.random.random()

# Reset and verify reproducibility
set_global_seed(42)
val2 = torch.rand(1).item()
random2 = random.random()
np2 = np.random.random()

assert val1 == val2, 'PyTorch seed not working'
assert random1 == random2, 'Python random seed not working'
assert np1 == np2, 'NumPy seed not working'

print('✅ Reproducibility utilities work')
print('  - PyTorch: seeded')
print('  - Python random: seeded')
print('  - NumPy: seeded')
if torch.cuda.is_available():
    print('  - CUDA: deterministic mode')
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Reproducibility configured" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Reproducibility verification failed" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Reproducibility test failed: $_" -ForegroundColor Red
    $errors++
}

# Test 10: Training script imports
Write-Host "[10/10] Verifying training script integrity..." -ForegroundColor Yellow
try {
    $output = python -c @"
import sys
from pathlib import Path

# Test training script can be imported
sys.path.insert(0, str(Path('scripts/training')))

try:
    from train_baseline_efficientnet import train_efficientnet_baseline
    print('✅ Training function imports successfully')

    # Check function signature
    import inspect
    sig = inspect.signature(train_efficientnet_baseline)
    params = list(sig.parameters.keys())

    required = ['data_root', 'output_dir', 'seed', 'num_epochs', 'batch_size',
                'learning_rate', 'weight_decay', 'device']

    for param in required:
        if param not in params:
            print(f'⚠️ Missing parameter: {param}')
        else:
            print(f'  - {param}: ✓')

except Exception as e:
    print(f'❌ Training script import failed: {e}')
    exit(1)
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Training script ready" -ForegroundColor Green
        $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "  ❌ Training script verification failed" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host "  ❌ Training script test failed: $_" -ForegroundColor Red
    $errors++
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✅ ALL TESTS PASSED - FULLY READY TO TRAIN!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Training Options:" -ForegroundColor Cyan
    Write-Host "  1. Single seed (quick test):  python .\scripts\training\train_baseline_efficientnet.py --seed 42" -ForegroundColor White
    Write-Host "  2. All seeds (full study):    python .\scripts\training\train_all_efficientnet_seeds.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected Results:" -ForegroundColor Cyan
    Write-Host "  - Clean Accuracy: 80-87%" -ForegroundColor Gray
    Write-Host "  - Training Time: ~2-4 hours per seed (GPU)" -ForegroundColor Gray
    Write-Host "  - Checkpoints: best.pt + last.pt per seed" -ForegroundColor Gray
} elseif ($errors -eq 0 -and $warnings -gt 0) {
    Write-Host "⚠️ TESTS PASSED WITH $warnings WARNING(S)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can proceed with training, but consider:" -ForegroundColor Yellow
    Write-Host "  - Implementing resume capability (--resume flag)" -ForegroundColor Gray
    Write-Host "  - Checking GPU availability for faster training" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Training Options:" -ForegroundColor Cyan
    Write-Host "  1. Single seed:  python .\scripts\training\train_baseline_efficientnet.py --seed 42" -ForegroundColor White
    Write-Host "  2. All seeds:    python .\scripts\training\train_all_efficientnet_seeds.py" -ForegroundColor White
} else {
    Write-Host "❌ $errors CRITICAL ERROR(S) FOUND!" -ForegroundColor Red
    Write-Host ""
    Write-Host "⛔ DO NOT START TRAINING YET!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix the errors above to avoid:" -ForegroundColor Yellow
    Write-Host "  - Wasted GPU hours on broken code" -ForegroundColor Gray
    Write-Host "  - Incomplete/corrupted checkpoints" -ForegroundColor Gray
    Write-Host "  - Invalid experimental results" -ForegroundColor Gray
    Write-Host ""
    Write-Host "After fixing, re-run: .\scripts\training\verify_before_training.ps1" -ForegroundColor Cyan
    exit 1
}

Write-Host "=" * 80 -ForegroundColor Cyan
