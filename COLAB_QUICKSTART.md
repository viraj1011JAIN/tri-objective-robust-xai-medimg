# Running HPO on Google Colab - Quick Start

**Date:** November 24, 2025
**Purpose:** Run medical imaging HPO on Google Colab
**Status:** Ready for testing

---

## Problem: Dataset Not Found on Colab

When you run on Colab, you see:
```
FileNotFoundError: ISIC metadata CSV not found at data/processed/isic2018/metadata_processed.csv
```

**Reason:** The dataset files are on your **local Windows machine**, not on Colab's cloud VM.

---

## Solution: 3 Options

### Option 1: Use Mock Data (Quick Test - 5 minutes)

**Best for:** Testing if HPO infrastructure works

```python
# 1. Create mock datasets
!python scripts/setup_colab_data.py

# 2. Run quick HPO test
!python scripts/run_hpo_medical.py --dataset isic2018 --quick-test --device cuda

# Result: 10 trials with 100 mock samples (not real medical data)
```

**⚠️ Warning:** Mock data is for **testing only**. Not suitable for dissertation.

---

### Option 2: Upload from Local Machine (Medium - 30 min)

**Best for:** One-time full HPO run

```python
# 1. Zip your datasets on Windows
# In PowerShell:
Compress-Archive -Path "data\processed\isic2018" -DestinationPath "isic2018.zip"

# 2. Upload to Colab
from google.colab import files
uploaded = files.upload()  # Select isic2018.zip

# 3. Extract
!mkdir -p data/processed
!unzip isic2018.zip -d data/processed/

# 4. Run HPO
!python scripts/run_hpo_medical.py --dataset isic2018 --n-trials 50 --n-epochs 10 --pretrained
```

**Pros:** Full dataset available
**Cons:** Upload time (~10-20 min for ISIC 2018), lost if Colab disconnects

---

### Option 3: Google Drive Mount (Best - Persistent)

**Best for:** Multiple HPO runs, experiments

```python
# 1. Upload datasets to Google Drive (one-time)
#    On Windows: Upload data/processed/ folder to Google Drive

# 2. Mount Drive in Colab
from google.colab import drive
drive.mount('/content/drive')

# 3. Create symlink to datasets
!ln -s /content/drive/MyDrive/dissertation/data/processed data/processed

# 4. Run HPO
!python scripts/run_hpo_medical.py --dataset isic2018 --n-trials 50 --n-epochs 10 --pretrained
```

**Pros:** Persistent, fast access, no re-upload
**Cons:** One-time setup required

---

## Recommended Workflow for Colab

### Phase 1: Quick Validation (5 minutes)

**Goal:** Verify HPO script works

```python
# Clone repo
!git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git
%cd tri-objective-robust-xai-medimg

# Install dependencies
!pip install -q optuna albumentations torch torchvision pandas numpy Pillow

# Create mock data
!python scripts/setup_colab_data.py

# Test HPO
!python scripts/run_hpo_medical.py --dataset isic2018 --quick-test --device cuda
```

**Expected output:**
```
Loaded: 100 train, 50 val, 50 test samples
Starting HPO study with 10 trials...
Trial 0 Complete. Objective: 0.3122
...
BEST TRIAL RESULTS
Trial number: 5
Objective value: 0.3456
```

---

### Phase 2: Real Data Setup (30 minutes)

**Option A: Upload via Google Drive**

```python
# 1. On Windows, upload to Drive:
#    - Create folder: MyDrive/dissertation/data/processed/
#    - Upload: isic2018/, derm7pt/, nih_cxr/ folders

# 2. On Colab:
from google.colab import drive
drive.mount('/content/drive')

# 3. Check datasets exist
!ls /content/drive/MyDrive/dissertation/data/processed/

# 4. Link to working directory
!ln -s /content/drive/MyDrive/dissertation/data/processed data/processed

# 5. Verify
!ls data/processed/isic2018/
# Should see: metadata_processed.csv, images/
```

**Option B: Direct Upload**

```python
# 1. Zip on Windows
# PowerShell: Compress-Archive -Path "data\processed\isic2018" -DestinationPath "isic2018.zip"

# 2. Upload
from google.colab import files
uploaded = files.upload()

# 3. Extract
!mkdir -p data/processed
!unzip -q isic2018.zip -d data/processed/

# 4. Verify
!ls data/processed/isic2018/
```

---

### Phase 3: Full HPO Run (2-3 hours per dataset)

```python
# ISIC 2018 - Full HPO
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --n-trials 50 \
    --n-epochs 10 \
    --pretrained \
    --batch-size 32 \
    --device cuda \
    --output-dir results/hpo_isic2018 \
    --checkpoint-dir checkpoints/hpo_isic2018

# Expected time: 2-3 hours on Tesla T4
# Output: Best hyperparameters, visualizations, study database
```

---

## Complete Colab Notebook Template

```python
# ============================================================
# MEDICAL IMAGING HPO - GOOGLE COLAB
# ============================================================

# 1. SETUP
# --------------------------------------------------------
!git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git
%cd tri-objective-robust-xai-medimg

# 2. INSTALL DEPENDENCIES
# --------------------------------------------------------
!pip install -q optuna albumentations torch torchvision pandas numpy Pillow matplotlib seaborn scikit-learn

# 3. CHECK GPU
# --------------------------------------------------------
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 4. MOUNT GOOGLE DRIVE (if using Drive)
# --------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Option A: Link from Drive
!ln -s /content/drive/MyDrive/dissertation/data/processed data/processed

# Option B: Use mock data for testing
# !python scripts/setup_colab_data.py

# 5. VERIFY DATA
# --------------------------------------------------------
!ls data/processed/isic2018/
!head -n 5 data/processed/isic2018/metadata_processed.csv

# 6. QUICK TEST (10 trials, 2 epochs)
# --------------------------------------------------------
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --quick-test \
    --device cuda

# 7. FULL HPO (50 trials, 10 epochs)
# --------------------------------------------------------
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --n-trials 50 \
    --n-epochs 10 \
    --pretrained \
    --device cuda

# 8. CHECK RESULTS
# --------------------------------------------------------
!ls results/phase_5_4_medical/isic2018/analysis/
# Expected: best_params.json, trials_dataframe.csv, plots/

# 9. DOWNLOAD RESULTS
# --------------------------------------------------------
from google.colab import files
!zip -r hpo_results.zip results/ checkpoints/ hpo_medical.db
files.download('hpo_results.zip')
```

---

## Troubleshooting

### Issue 1: "No GPU available"

```python
# Check runtime
# Runtime > Change runtime type > Hardware accelerator > GPU (T4)

# Verify
import torch
print(torch.cuda.is_available())  # Should be True
```

### Issue 2: "Dataset not found"

```python
# Check what's actually there
!ls -la data/processed/isic2018/

# Common issues:
# - Forgot to mount Drive: run drive.mount()
# - Wrong path: check Drive folder structure
# - Not uploaded: use setup_colab_data.py for testing
```

### Issue 3: "Out of memory"

```python
# Reduce batch size
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --batch-size 16 \  # Reduced from 32
    --quick-test
```

### Issue 4: "Colab disconnected mid-run"

```python
# Use persistent storage (Drive) and resume study
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --study-name trades_hpo_isic2018 \  # Same name resumes
    --storage sqlite:///hpo_medical.db \
    --n-trials 50
```

---

## Dataset Size Estimates

| Dataset | Size | Upload Time (50 Mbps) | HPO Time (50 trials) |
|---------|------|----------------------|---------------------|
| ISIC 2018 | ~5 GB | 15 min | 2-3 hours |
| ISIC 2019 | ~4 GB | 12 min | 2-3 hours |
| ISIC 2020 | ~30 GB | 1.5 hours | 2-3 hours |
| Derm7pt | ~500 MB | 2 min | 1-2 hours |
| NIH CXR | ~50 GB | 3-4 hours | 3-4 hours |
| PadChest | ~80 GB | 5-6 hours | 4-5 hours |

**Recommendation:** Start with ISIC 2018 and Derm7pt (smaller datasets).

---

## Google Drive Organization

```
MyDrive/
└── dissertation/
    └── data/
        └── processed/
            ├── isic2018/
            │   ├── metadata_processed.csv
            │   └── images/
            ├── derm7pt/
            │   ├── metadata_processed.csv
            │   └── images/
            └── nih_cxr/
                ├── metadata_processed.csv
                └── images/
```

---

## Time Budget Planning

### Quick Validation (Mock Data)
- Setup: 2 min
- Mock data creation: 1 min
- Quick HPO: 5 min
- **Total: 8 minutes**

### Single Dataset HPO (ISIC 2018)
- Drive mount: 1 min
- Data verification: 1 min
- Quick test: 5 min
- Full HPO (50 trials): 2-3 hours
- **Total: ~3 hours**

### Full Dissertation (3 Datasets)
- ISIC 2018: 3 hours
- Derm7pt: 2 hours
- NIH CXR: 4 hours
- **Total: ~9 hours**

**Tip:** Run overnight or use Colab Pro for longer sessions.

---

## Next Steps

1. **✅ Test infrastructure**: Run with mock data (5 min)
2. **📤 Upload datasets**: Use Google Drive (30 min one-time)
3. **🔬 Run HPO**: ISIC 2018 first (3 hours)
4. **📊 Analyze results**: Check best hyperparameters
5. **🔁 Repeat**: Derm7pt, NIH CXR
6. **📝 Document**: Save results for dissertation

---

## Important Notes

1. **Mock data is NOT for dissertation** - Only for testing infrastructure
2. **Colab has time limits** - Free: 12 hours, Pro: 24 hours
3. **Save to Drive frequently** - Colab VMs are ephemeral
4. **Download results** - Before session ends
5. **Use `--pretrained`** - Faster convergence with ImageNet weights

---

**Ready to start?** Run the Quick Validation first! ⚡
