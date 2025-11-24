# Python Environment Setup Guide
## Tri-Objective Robust XAI for Medical Imaging

---

## Current Status ✅

Your repository validation shows:
- ✅ **All 6 medical datasets found** in `/content/drive/MyDrive/data`
- ✅ **Configuration files validated** (8 dataset configs, 2 model configs)
- ✅ **Environment variable set**: `DATA_ROOT=/content/drive/MyDrive/data`
- ✅ **Python 3.13 detected**

The torch import errors are **expected** when running outside a virtual environment.

---

## Python Environment Setup

### Option 1: Using Conda (Recommended)

#### Create Environment
```powershell
# Create new environment with Python 3.10+
conda create -n triobj-xai python=3.10 -y

# Activate environment
conda activate triobj-xai

# Install PyTorch with CUDA support (if you have GPU)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Or CPU-only version
# conda install pytorch torchvision cpuonly -c pytorch -y

# Install other dependencies
pip install -r requirements.txt
```

#### Verify Installation
```powershell
# Should show your conda environment Python
python -c "import sys; print(sys.executable)"

# Should show torch version and CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Using Python venv

#### Create Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (visit pytorch.org for your specific command)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

---

## Validation After Environment Setup

Once your environment is activated, run the validation script:

```powershell
# Activate environment first
conda activate triobj-xai  # or .\venv\Scripts\Activate.ps1

# Run validation
python scripts/validate_repository.py
```

### Expected Output (Success)
```
[1/4] Checking imports...
INFO: ✓ Dataset imports successful
INFO: ✓ Model imports successful
INFO: ✓ Training imports successful
INFO: ✓ Utils imports successful

[2/4] Checking dataset paths...
INFO: ✓ Data root exists: /content/drive/MyDrive/data
INFO: Found datasets:
INFO:   ✓ isic_2018 (ISIC 2018 dermoscopy dataset)
INFO:   ✓ isic_2019 (ISIC 2019 dermoscopy dataset)
INFO:   ✓ isic_2020 (ISIC 2020 dermoscopy dataset)
INFO:   ✓ derm7pt (Derm7pt dermoscopy dataset)
INFO:   ✓ nih_cxr (NIH ChestX-ray14 dataset)
INFO:   ✓ padchest (PadChest dataset)

[3/4] Checking configuration files...
INFO: ✓ Base config exists
INFO: ✓ Found 8 dataset configs
INFO: ✓ Found 2 model configs

[4/4] Checking environment setup...
INFO: ✓ DATA_ROOT environment variable set: /content/drive/MyDrive/data
INFO: ✓ Python version: 3.13

✓ All validation checks passed!
```

---

## Quick Environment Check Commands

### Check Python Environment
```powershell
# Which Python is active?
python -c "import sys; print(sys.executable)"

# Python version
python --version

# Check if in virtual environment
python -c "import sys; print('Virtual Env' if sys.prefix != sys.base_prefix else 'System Python')"
```

### Check Key Packages
```powershell
# PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# NumPy
python -c "import numpy; print(f'NumPy {numpy.__version__}')"

# Check all installed packages
pip list | Select-String "torch|numpy|pandas|sklearn|mlflow|dvc"
```

### Check Project Imports
```powershell
# Test dataset imports
python -c "from src.datasets.isic import ISICDataset; print('✓ Dataset imports OK')"

# Test model imports
python -c "from src.models.resnet import ResNet50Classifier; print('✓ Model imports OK')"

# Test training imports
python -c "from src.training.baseline_trainer import BaselineTrainer; print('✓ Training imports OK')"

# Test config loading
python -c "from src.utils.config import load_experiment_config; print('✓ Config imports OK')"
```

---

## Dataset Validation Details

Your datasets in `/content/drive/MyDrive/data`:

| Dataset | Path | Status |
|---------|------|--------|
| ISIC 2018 | `/content/drive/MyDrive/data\isic_2018` | ✅ Found |
| ISIC 2019 | `/content/drive/MyDrive/data\isic_2019` | ✅ Found |
| ISIC 2020 | `/content/drive/MyDrive/data\isic_2020` | ✅ Found |
| Derm7pt | `/content/drive/MyDrive/data\derm7pt` | ✅ Found |
| NIH CXR | `/content/drive/MyDrive/data\nih_cxr` | ✅ Found |
| PadChest | `/content/drive/MyDrive/data\padchest` | ✅ Found |

### Dataset Structure Requirements

Each dataset should have:
```
/content/drive/MyDrive/data\{dataset_name}\
├── metadata.csv           # Required: Image paths, labels, splits
├── images/                # Required: Image files
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── (other dataset files)
```

### Check Dataset Metadata
```powershell
# Check if metadata files exist
Test-Path /content/drive/MyDrive/data\isic_2018\metadata.csv
Test-Path /content/drive/MyDrive/data\isic_2019\metadata.csv
Test-Path /content/drive/MyDrive/data\isic_2020\metadata.csv
Test-Path /content/drive/MyDrive/data\derm7pt\metadata.csv
Test-Path /content/drive/MyDrive/data\nih_cxr\metadata.csv
Test-Path /content/drive/MyDrive/data\padchest\metadata.csv

# Count images in a dataset
(Get-ChildItem /content/drive/MyDrive/data\isic_2018\images -File).Count
```

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** Activate your virtual environment first
```powershell
conda activate triobj-xai
# or
.\venv\Scripts\Activate.ps1
```

### Issue: "CUDA not available"
**Check GPU:**
```powershell
nvidia-smi
```

**Install CUDA-enabled PyTorch:**
```powershell
# Visit pytorch.org/get-started and get the correct command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Import errors for project modules
**Solution:** Ensure you're in the project root
```powershell
cd C:\Users\Dissertation\tri-objective-robust-xai-medimg
python -c "import sys; sys.path.insert(0, '.'); from src.datasets.isic import ISICDataset"
```

### Issue: "Cannot import name 'ResNetClassifier'"
**Status:** ✅ Fixed in validation script
- Changed to correct class names: `ResNet50Classifier`, `EfficientNetB0Classifier`, `ViTB16Classifier`

---

## Environment Variables

### Required
```powershell
# Data root (already set)
$env:DATA_ROOT = "/content/drive/MyDrive/data"
```

### Optional
```powershell
# MLflow tracking
$env:MLFLOW_TRACKING_URI = "./mlruns"

# CUDA device selection
$env:CUDA_VISIBLE_DEVICES = "0"  # Use GPU 0

# Python path (if needed)
$env:PYTHONPATH = "C:\Users\Dissertation\tri-objective-robust-xai-medimg"
```

### Make Permanent (Windows)
```powershell
# Set permanently for user
[System.Environment]::SetEnvironmentVariable('DATA_ROOT', '/content/drive/MyDrive/data', 'User')
```

---

## Next Steps

1. **Activate your Python environment**
   ```powershell
   conda activate triobj-xai
   ```

2. **Validate everything works**
   ```powershell
   python scripts/validate_repository.py
   ```

3. **Run tests** (optional but recommended)
   ```powershell
   pytest tests/ -v
   ```

4. **Start training**
   ```powershell
   # Small debug run
   python scripts/train_cifar10_debug.py --epochs 1

   # Or use baseline training
   python -m src.training.train_baseline --config configs/experiments/your_experiment.yaml
   ```

---

## Summary of Fixes

✅ **Model import errors fixed**
- Corrected class names: `ResNet50Classifier`, `EfficientNetB0Classifier`, `ViTB16Classifier`

✅ **Dataset paths validated**
- All 6 datasets found in `/content/drive/MyDrive/data`
- CIFAR-10 removed from expected datasets (debug only, can auto-download)

✅ **Configuration files validated**
- All configs point to correct paths
- `${DATA_ROOT}` environment variable expansion working

✅ **Validation script updated**
- Dynamically scans `/content/drive/MyDrive/data` for datasets
- Better error messages
- Correct model class names

---

**Status:** Ready for development after environment activation ✅
