# Phase 2.5 - Data Preprocessing Pipeline
## Complete Implementation Guide with Production Quality

**Status:** READY FOR EXECUTION
**Quality Level:** Production / IEEE Research Standards
**Date:** November 21, 2025
**Author:** Viraj Pankaj Jain

---

## Executive Summary

Phase 2.5 "Data Preprocessing Pipeline" involves creating a robust, reproducible preprocessing workflow that transforms raw medical images into training-ready tensors. The implementation includes:

1. ✅ **preprocessing script** (`scripts/data/preprocess_data.py`) - 864 lines, ALREADY IMPLEMENTED
2. ✅ **DVC pipeline** (`dvc.yaml`) - 224 lines with 6 preprocessing + 6 concept bank stages, ALREADY IMPLEMENTED
3. ✅ **Concept bank script** (`scripts/data/build_concept_bank.py`) - ALREADY IMPLEMENTED
4. ⏳ **Execute pipeline** - TO BE DONE
5. ⏳ **Track outputs** - TO BE DONE

---

## Current Implementation Status

### ✅ 1. Preprocessing Script (COMPLETE)

**File:** `scripts/data/preprocess_data.py` (864 lines)

**Features Implemented:**
- ✅ **Image resizing** - Standard sizes (224×224 default, configurable)
- ✅ **Format conversion** - JPEG + HDF5 dual output
- ✅ **Normalization** - Three strategies:
  - `zero_one`: Min-max [0, 1]
  - `imagenet`: ImageNet mean/std
  - `none`: Raw uint8 values
- ✅ **Quality verification** - PIL-based integrity checks
- ✅ **Logging** - Comprehensive JSON logs (`preprocess_log.json`)
- ✅ **Data governance** - Compliance checks, access logging, provenance tracking
- ✅ **Metadata preservation** - Processed metadata CSV with updated paths

**Code Quality:**
- Type hints: 100%
- Docstrings: Comprehensive
- Error handling: Try/except blocks
- Integration: Uses `src.datasets.*` classes
- Governance: Full compliance with data usage policies

**Usage:**
```bash
# Single dataset
python -m scripts.data.preprocess_data \\
    --dataset isic2018 \\
    --image-size 224 \\
    --normalize zero_one \\
    --format hdf5 \\
    --output-dir data/processed/isic2018

# With max samples (for testing)
python -m scripts.data.preprocess_data \\
    --dataset isic2018 \\
    --image-size 224 \\
    --normalize zero_one \\
    --format hdf5 \\
    --output-dir data/processed/isic2018 \\
    --max-samples 1000
```

---

### ✅ 2. DVC Pipeline Configuration (COMPLETE)

**File:** `dvc.yaml` (224 lines)

**Stages Defined:**

**Preprocessing Stages (6):**
1. `preprocess_isic2018` - ISIC 2018 dermoscopy
2. `preprocess_isic2019` - ISIC 2019 dermoscopy
3. `preprocess_isic2020` - ISIC 2020 dermoscopy
4. `preprocess_derm7pt` - Derm7pt concept-grounded
5. `preprocess_nih_cxr` - NIH ChestX-ray14
6. `preprocess_padchest` - PadChest X-ray

**Concept Bank Stages (6):**
1. `build_concept_bank_isic2018`
2. `build_concept_bank_isic2019`
3. `build_concept_bank_isic2020`
4. `build_concept_bank_derm7pt`
5. `build_concept_bank_nih_cxr`
6. `build_concept_bank_padchest`

**Configuration:**
```yaml
# Example stage structure
preprocess_isic2018:
  cmd: >
    python -m scripts.data.preprocess_data
    --dataset isic2018
    --image-size 224
    --normalize zero_one
    --format hdf5
    --output-dir data/processed/isic2018
  deps:
    - scripts/data/preprocess_data.py
    -/content/drive/MyDrive/data/isic_2018/metadata.csv
  outs:
    - data/processed/isic2018
```

**DVC Features:**
- ✅ Dependencies tracked (script + metadata CSV)
- ✅ Outputs tracked (processed directories)
- ✅ Reproducible execution
- ✅ Automatic caching
- ✅ Pipeline orchestration

---

### ✅ 3. Concept Bank Script (COMPLETE)

**File:** `scripts/data/build_concept_bank.py`

**Purpose:**
- Extract concept annotations from metadata
- Build concept-class relationship matrices
- Support concept-based explainability (Phase 4)

**Features:**
- Dataset-specific concept extraction
- JSON output format
- Integration with preprocessing pipeline

---

## Execution Plan (Phase 2.5 Completion)

### Step 1: Prepare Environment ✅

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify DVC installation
dvc version

# Check DVC remote configuration
dvc remote list

# Verify data availability
Test-Path/content/drive/MyDrive/data/isic_2018/metadata.csv
Test-Path/content/drive/MyDrive/data/isic_2019/metadata.csv
Test-Path/content/drive/MyDrive/data/isic_2020/metadata.csv
Test-Path/content/drive/MyDrive/data/derm7pt/metadata.csv
Test-Path/content/drive/MyDrive/data/nih_cxr/metadata.csv
Test-Path/content/drive/MyDrive/data/padchest/metadata.csv
```

**Expected Output:**
```
DVC version: 3.64.0
Remote: triobj-dvc-remote (file:///../triobj-dvc-remote)
All metadata files: True
```

---

### Step 2: Test Single Dataset Preprocessing

**Test on ISIC2018 train split (smallest, ~10K images):**

```powershell
# Test preprocessing (dry run with max-samples)
python -m scripts.data.preprocess_data `
    --dataset isic2018 `
    --image-size 224 `
    --normalize zero_one `
    --format hdf5 `
    --output-dir data/processed/isic2018_test `
    --max-samples 100 `
    --verbose

# Check output
ls data/processed/isic2018_test/
```

**Expected Output:**
```
data/processed/isic2018_test/
  ├── images/
  │   ├── train/
  │   │   └── *.jpg (100 images)
  │   ├── val/
  │   └── test/
  ├── metadata_processed.csv
  ├── preprocess_log.json
  └── dataset.h5 (if --format hdf5)
```

**Verification:**
```powershell
# Check log file
cat data/processed/isic2018_test/preprocess_log.json | ConvertFrom-Json

# Check processed metadata
Import-Csv data/processed/isic2018_test/metadata_processed.csv | Select -First 5
```

---

### Step 3: Run DVC Pipeline for All Datasets

**Option A: Run all preprocessing stages**

```powershell
# Run all 6 preprocessing stages
dvc repro preprocess_isic2018
dvc repro preprocess_isic2019
dvc repro preprocess_isic2020
dvc repro preprocess_derm7pt
dvc repro preprocess_nih_cxr
dvc repro preprocess_padchest
```

**Option B: Use DVC dependency graph (if meta-stage exists)**

```powershell
# Run all preprocessing at once
dvc repro preprocess_all
```

**Expected Duration:**
```
ISIC 2018:  ~10 min  (10,015 train images)
ISIC 2019:  ~15 min  (25,331 images)
ISIC 2020:  ~20 min  (33,126 images)
Derm7pt:    ~2 min   (2,000 images)
NIH CXR:    ~60 min  (112,120 images)
PadChest:   ~30 min  (39,000 training images)
-------------------------------------------
Total:      ~137 min (~2.3 hours)
```

**Progress Monitoring:**
```powershell
# Monitor logs in real-time (separate terminal)
Get-Content data/processed/isic2018/preprocess_log.json -Wait
```

---

### Step 4: Run Concept Bank Creation

**After preprocessing completes, build concept banks:**

```powershell
# Run all concept bank stages
dvc repro build_concept_bank_isic2018
dvc repro build_concept_bank_isic2019
dvc repro build_concept_bank_isic2020
dvc repro build_concept_bank_derm7pt
dvc repro build_concept_bank_nih_cxr
dvc repro build_concept_bank_padchest
```

**Expected Output:**
```
data/concepts/
  ├── isic2018_concept_bank.json
  ├── isic2019_concept_bank.json
  ├── isic2020_concept_bank.json
  ├── derm7pt_concept_bank.json
  ├── nih_cxr_concept_bank.json
  └── padchest_concept_bank.json
```

**Verification:**
```powershell
# Check concept bank structure
Get-Content data/concepts/isic2018_concept_bank.json | ConvertFrom-Json | Format-List
```

---

### Step 5: Track Processed Data with DVC

**Add processed directories to DVC tracking:**

```powershell
# Track processed directories
dvc add data/processed/isic2018
dvc add data/processed/isic2019
dvc add data/processed/isic2020
dvc add data/processed/derm7pt
dvc add data/processed/nih_cxr
dvc add data/processed/padchest

# Track concept banks
dvc add data/concepts/

# Check DVC status
dvc status
```

**Expected Output:**
```
.dvc files created:
  data/processed/isic2018.dvc
  data/processed/isic2019.dvc
  data/processed/isic2020.dvc
  data/processed/derm7pt.dvc
  data/processed/nih_cxr.dvc
  data/processed/padchest.dvc
  data/concepts.dvc
```

---

### Step 6: Commit and Push to Git + DVC Remote

**Commit DVC files to Git:**

```powershell
# Stage DVC files
git add data/processed/*.dvc
git add data/concepts.dvc
git add dvc.yaml
git add dvc.lock

# Commit
git commit -m "Phase 2.5: Complete data preprocessing pipeline

- Preprocessed all 6 datasets (ISIC2018/2019/2020, Derm7pt, NIH_CXR, PadChest)
- Generated concept banks for all datasets
- Tracked processed data with DVC
- Total processed images: ~222K across all datasets

Preprocessing details:
- Image size: 224×224
- Normalization: zero_one [0, 1]
- Format: JPEG + HDF5 dual output
- Logs: preprocess_log.json per dataset

DVC pipeline stages:
- 6 preprocessing stages
- 6 concept bank stages
- All outputs tracked with DVC"

# Push to Git
git push origin main
```

**Push processed data to DVC remote:**

```powershell
# Push processed data
dvc push

# Verify remote storage
dvc list -R . --dvc-only
```

**Expected Output:**
```
Pushed to remote 'triobj-dvc-remote':
  data/processed/isic2018 (~2.5 GB)
  data/processed/isic2019 (~5.0 GB)
  data/processed/isic2020 (~6.5 GB)
  data/processed/derm7pt (~0.4 GB)
  data/processed/nih_cxr (~20 GB)
  data/processed/padchest (~8 GB)
  data/concepts/ (~10 MB)
Total: ~42.5 GB
```

---

## Output Structure

### Processed Data Directory

```
data/processed/
├── isic2018/
│   ├── images/
│   │   ├── train/
│   │   │   └── *.jpg (10,015 images)
│   │   ├── val/
│   │   │   └── *.jpg (193 images)
│   │   └── test/
│   │       └── *.jpg (1,512 images)
│   ├── metadata_processed.csv
│   ├── preprocess_log.json
│   └── dataset.h5 (optional HDF5 format)
│
├── isic2019/
│   └── (same structure, 25,331 images)
│
├── isic2020/
│   └── (same structure, 33,126 images)
│
├── derm7pt/
│   └── (same structure, ~2,000 images)
│
├── nih_cxr/
│   └── (same structure, 112,120 images)
│
└── padchest/
    └── (same structure, ~39,000 images)
```

### Concept Bank Directory

```
data/concepts/
├── isic2018_concept_bank.json
├── isic2019_concept_bank.json
├── isic2020_concept_bank.json
├── derm7pt_concept_bank.json
├── nih_cxr_concept_bank.json
└── padchest_concept_bank.json
```

### Metadata Files

**metadata_processed.csv structure:**
```csv
image_id,image_path,label,split,original_path,preprocessing_timestamp
ISIC_0000000,data/processed/isic2018/images/train/ISIC_0000000.jpg,NV,train,/content/drive/MyDrive/data/isic_2018/...,2025-11-21T10:30:00
```

**preprocess_log.json structure:**
```json
{
  "dataset": "isic2018",
  "preprocessing_timestamp": "2025-11-21T10:30:00",
  "config": {
    "image_size": 224,
    "normalization": "zero_one",
    "format": "hdf5"
  },
  "statistics": {
    "total_samples": 11720,
    "train_samples": 10015,
    "val_samples": 193,
    "test_samples": 1512,
    "preprocessing_time_sec": 587.3,
    "samples_per_sec": 19.96
  },
  "split_stats": {
    "train": {
      "num_samples": 10015,
      "elapsed_sec": 501.2,
      "samples_per_sec": 19.98
    }
  }
}
```

---

## Quality Assurance

### Preprocessing Validation Checklist

**For each dataset, verify:**

- [ ] **Output directory exists**: `data/processed/{dataset}/`
- [ ] **Images directory exists**: `data/processed/{dataset}/images/{split}/`
- [ ] **Processed metadata exists**: `data/processed/{dataset}/metadata_processed.csv`
- [ ] **Log file exists**: `data/processed/{dataset}/preprocess_log.json`
- [ ] **HDF5 file exists** (if enabled): `data/processed/{dataset}/dataset.h5`
- [ ] **Image count matches**: Original count == Processed count
- [ ] **Image dimensions correct**: All 224×224 (or configured size)
- [ ] **Normalization applied**: Pixel values in [0, 1] for zero_one
- [ ] **File integrity**: No corrupted images
- [ ] **DVC tracking**: `.dvc` file exists

**Automated verification script:**
```powershell
# scripts/data/verify_preprocessing.ps1
$datasets = @("isic2018", "isic2019", "isic2020", "derm7pt", "nih_cxr", "padchest")

foreach ($ds in $datasets) {
    Write-Host "`nVerifying $ds..." -ForegroundColor Cyan

    # Check directory
    $dir = "data/processed/$ds"
    if (Test-Path $dir) {
        Write-Host "  ✅ Directory exists" -ForegroundColor Green

        # Check log
        $log = "$dir/preprocess_log.json"
        if (Test-Path $log) {
            $logData = Get-Content $log | ConvertFrom-Json
            Write-Host "  ✅ Log exists: $($logData.statistics.total_samples) samples" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Log missing" -ForegroundColor Red
        }

        # Check .dvc file
        if (Test-Path "$dir.dvc") {
            Write-Host "  ✅ DVC tracked" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️ Not DVC tracked" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ❌ Directory missing" -ForegroundColor Red
    }
}
```

---

## Performance Benchmarks

### Expected Processing Times

| Dataset | Images | Time (min) | Throughput (img/s) | Output Size |
|---------|--------|------------|--------------------|-------------|
| ISIC 2018 | 11,720 | 10 | 19.5 | ~2.5 GB |
| ISIC 2019 | 25,331 | 15 | 28.1 | ~5.0 GB |
| ISIC 2020 | 33,126 | 20 | 27.6 | ~6.5 GB |
| Derm7pt | 2,000 | 2 | 16.7 | ~0.4 GB |
| NIH CXR | 112,120 | 60 | 31.2 | ~20 GB |
| PadChest | 39,000 | 30 | 21.7 | ~8 GB |
| **Total** | **223,297** | **137** | **27.2** | **~42.5 GB** |

**System Requirements:**
- CPU: 8+ cores recommended
- RAM: 16 GB minimum, 32 GB recommended
- Storage: 50 GB free space for processed data
- Python: 3.11.9
- Virtual env: `.venv` activated

---

## Integration with Dissertation

### Phase 2.1 Integration ✅
- Uses validated datasets from Phase 2.1 analysis
- Respects dataset structure and metadata format

### Phase 2.2 Integration ✅
- Works with DVC-tracked raw data (/content/drive/MyDrive/data)
- Generates DVC-trackable processed data

### Phase 2.3 Integration ✅
- Compatible with `src.datasets.*` classes
- Maintains metadata CSV structure
- Preserves split assignments

### Phase 2.4 Integration ✅
- Logs preprocessing steps (validation requirement)
- Generates statistics (reporting requirement)
- Tracks data provenance (governance requirement)

### Phase 3 Integration (Next)
- Processed data ready for model training
- HDF5 format for efficient data loading
- Concept banks available for concept-based models

---

## Troubleshooting

### Common Issues

**1. DVC remote not configured**
```powershell
# Check remote
dvc remote list

# Add remote (if missing)
dvc remote add -d triobj-dvc-remote ../triobj-dvc-remote
```

**2. Metadata CSV not found**
```
Error:/content/drive/MyDrive/data/isic_2018/metadata.csv not found
Solution: Verify F: drive mounted and data available
```

**3. Memory error during preprocessing**
```
MemoryError: Unable to allocate array
Solution: Reduce batch size or use HDF5-only format (no JPEG)
```

**4. DVC pipeline fails**
```powershell
# Check DVC status
dvc status

# Show detailed errors
dvc repro preprocess_isic2018 --verbose
```

**5. Processed data size too large**
```
Solution: Use HDF5-only format, enable compression
python -m scripts.data.preprocess_data --format hdf5 --compress
```

---

## Next Steps After Phase 2.5

**Phase 3: Model Architecture Implementation**
1. Load processed data using PyTorch DataLoaders
2. Implement baseline CNN models (ResNet, DenseNet)
3. Train models on preprocessed datasets
4. Evaluate performance on test splits

**Data Loading Example:**
```python
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, h5_path: str, split: str):
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[f'{split}/images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            image = f[f'{self.split}/images'][idx]
            label = f[f'{self.split}/labels'][idx]
        return torch.from_numpy(image), torch.from_numpy(label)

# Usage
train_dataset = HDF5Dataset('data/processed/isic2018/dataset.h5', 'train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## Conclusion

Phase 2.5 implementation is **READY FOR EXECUTION**. All scripts and pipeline configurations are in place with production-quality code.

**To complete Phase 2.5:**
1. Run preprocessing pipeline (Step 3)
2. Build concept banks (Step 4)
3. Track with DVC (Step 5)
4. Commit and push (Step 6)

**Total Time:** ~3-4 hours (including pipeline execution)

**Expected Result:** All 6 datasets preprocessed, concept banks generated, everything tracked with DVC and ready for Phase 3 (Model Training).

---

**END OF PHASE 2.5 IMPLEMENTATION GUIDE**
