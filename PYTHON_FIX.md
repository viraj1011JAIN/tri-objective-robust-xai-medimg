# Python Environment Fix Guide
## Resolving "No module named 'torch'" Error

---

## Problem

You have two Python installations:
- **Python 3.13** (default, no PyTorch) - `C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python313\python.exe`
- **Python 3.11** (has PyTorch + CUDA) - `C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe`

When you run `python`, it uses Python 3.13 which doesn't have PyTorch.

---

## Solution: Use Python 3.11 Environment

### Method 1: Setup Environment for Current Session (Recommended)

Run this command once per PowerShell session:

```powershell
.\setup_python_env.ps1
```

This will:
- ✅ Set Python 3.11 as `python` alias for current session
- ✅ Verify PyTorch installation
- ✅ Show CUDA availability

After running this, you can use `python` normally:
```powershell
python scripts/test_environment.py
python -m src.training.train_baseline --config <config.yaml>
```

### Method 2: Use Full Path (Always Works)

```powershell
& "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe" -m src.training.train_baseline --config <config.yaml>
```

### Method 3: Use Wrapper Scripts

```powershell
.\train_baseline.ps1 --config <config.yaml>
```

---

## Quick Fix for Training

Instead of:
```powershell
python -m src.training.train_baseline --config configs/experiments/your_experiment.yaml
```

Use one of these:

### Option A: Setup environment first
```powershell
.\setup_python_env.ps1
python -m src.training.train_baseline --config configs/experiments/debug.yaml
```

### Option B: Use Python 3.11 directly
```powershell
& "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe" -m src.training.train_baseline --config configs/experiments/debug.yaml
```

### Option C: Use wrapper script
```powershell
.\train_baseline.ps1 --config configs/experiments/debug.yaml
```

---

## Available Experiment Configs

Located in `configs/experiments/`:

1. **debug.yaml** - CIFAR-10 debug experiment
2. **cifar10_debug_baseline.yaml** - CIFAR-10 baseline
3. **rq1_robustness/baseline_isic2018_resnet50.yaml** - ISIC 2018 with ResNet50

---

## Permanent Solution (Optional)

### Option 1: Add Python 3.11 to PATH

1. Open System Properties → Environment Variables
2. Edit "Path" under User variables
3. Add: `C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311`
4. Move it above Python313 entry
5. Restart PowerShell

### Option 2: Create Python 3.11 Virtual Environment

```powershell
# Create venv using Python 3.11
& "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Now 'python' will use Python 3.11 with PyTorch
python --version

# Install remaining dependencies
pip install -r requirements.txt
```

After activating venv, you can use `python` normally until you close the terminal.

### Option 3: Use Conda (Recommended for ML Projects)

```powershell
# Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Create environment with Python 3.11
conda create -n triobj python=3.11 -y

# Activate
conda activate triobj

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

# Now 'python' will use the conda environment
python --version
```

---

## Verification

After setting up, verify it works:

```powershell
# Check Python version
python --version
# Should show: Python 3.11.x

# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"
# Should show: PyTorch: 2.5.1+cu121

# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should show: CUDA: True

# Run full test
python scripts/test_environment.py
```

---

## Common Errors and Solutions

### Error: "No module named 'torch'"
**Cause:** Using wrong Python (3.13 instead of 3.11)
**Fix:** Use one of the methods above to use Python 3.11

### Error: "ModuleNotFoundError: No module named 'X'"
**Cause:** Missing package
**Fix:**
```powershell
& "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe" -m pip install X
```

### Error: Permission denied extracting CIFAR-10
**Cause:** File in use or permission issue
**Fix:**
```powershell
# Delete and retry
Remove-Item -Recurse -Force .\data\cifar10
# Then rerun training
```

---

## Summary

✅ **Immediate Fix:** Run `.\setup_python_env.ps1` before using `python`
✅ **Best Practice:** Create a virtual environment or use conda
✅ **Quick Start:** Use the wrapper scripts or full Python path

Your PyTorch installation is correct - you just need to use the right Python!
