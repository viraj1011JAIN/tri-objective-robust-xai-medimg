# Phase 2.5 Execution Checklist
## Data Preprocessing Pipeline - Commands to Run

**Date:** November 21, 2025
**Status:** Ready for Execution
**Virtual Environment:** `.venv` (MUST BE ACTIVATED)
**Working Directory:** `C:\Users\Dissertation\tri-objective-robust-xai-medimg`

---

## ✅ Pre-Execution Checklist

Before running any commands, verify:

```powershell
# 1. Navigate to project root
cd C:\Users\Dissertation\tri-objective-robust-xai-medimg

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify you see:
# (.venv) PS C:\Users\Dissertation\tri-objective-robust-xai-medimg>

# 4. Verify DVC is available
dvc version

# 5. Verify data is available
Test-Path/content/drive/MyDrive/data/isic_2018/metadata.csv
Test-Path/content/drive/MyDrive/data/isic_2019/metadata.csv
Test-Path/content/drive/MyDrive/data/isic_2020/metadata.csv
```

---

## 📋 Step 1: Run Preprocessing for All Datasets

### Option A: Run Each Dataset Individually (Recommended)

```powershell
# 1. ISIC 2018 (11,720 images, ~10 min)
dvc repro preprocess_isic2018

# 2. ISIC 2019 (25,331 images, ~15 min)
dvc repro preprocess_isic2019

# 3. ISIC 2020 (33,126 images, ~20 min)
dvc repro preprocess_isic2020

# 4. Derm7pt (2,000 images, ~2 min)
dvc repro preprocess_derm7pt

# 5. NIH CXR (112,120 images, ~60 min)
dvc repro preprocess_nih_cxr

# 6. PadChest (39,000 images, ~30 min)
dvc repro preprocess_padchest
```

### Option B: Run All at Once (use meta-stage if available)

```powershell
# Run all preprocessing stages
dvc repro preprocess
```

### ✅ After Each Dataset Completes

Verify the output:

```powershell
# Check the dataset was created
ls data\processed\isic2018

# View the preprocessing log
cat data\processed\isic2018\preprocess_log.json | ConvertFrom-Json | ConvertTo-Json -Depth 5

# Check sample count
Import-Csv data\processed\isic2018\metadata_processed.csv | Measure-Object
```

---

## 📋 Step 2: Build Concept Banks

After preprocessing completes, build concept banks:

```powershell
# 1. ISIC 2018
dvc repro build_concept_bank_isic2018

# 2. ISIC 2019
dvc repro build_concept_bank_isic2019

# 3. ISIC 2020
dvc repro build_concept_bank_isic2020

# 4. Derm7pt
dvc repro build_concept_bank_derm7pt

# 5. NIH CXR
dvc repro build_concept_bank_nih_cxr

# 6. PadChest
dvc repro build_concept_bank_padchest
```

### ✅ Verify Concept Banks

```powershell
# Check all concept banks were created
ls data\concepts\

# View a concept bank
cat data\concepts\isic2018_concept_bank.json | ConvertFrom-Json | ConvertTo-Json -Depth 3
```

---

## 📋 Step 3: Run Verification Script

```powershell
# Run the automated verification script
.\scripts\data\verify_preprocessing.ps1

# This will:
# - Check all datasets
# - Verify logs and metadata
# - Generate verification report
# - Show summary table
```

---

## 📋 Step 4: Track Processed Data with DVC

After all preprocessing is complete and verified:

```powershell
# Track each processed dataset
dvc add data\processed\isic2018
dvc add data\processed\isic2019
dvc add data\processed\isic2020
dvc add data\processed\derm7pt
dvc add data\processed\nih_cxr
dvc add data\processed\padchest

# Track concept banks
dvc add data\concepts

# Check what was tracked
ls data\processed\*.dvc
ls data\concepts.dvc
```

---

## 📋 Step 5: Commit to Git

```powershell
# Stage DVC files
git add data\processed\*.dvc
git add data\concepts.dvc
git add dvc.yaml
git add dvc.lock
git add .gitignore

# Commit
git commit -m "Phase 2.5: Complete data preprocessing pipeline

- Preprocessed all 6 datasets (ISIC2018/2019/2020, Derm7pt, NIH_CXR, PadChest)
- Generated concept banks for all datasets
- Tracked processed data with DVC
- Total processed images: ~223K across all datasets

Preprocessing configuration:
- Image size: 224×224
- Normalization: zero_one [0, 1]
- Output format: JPEG images + HDF5 files
- Logs: preprocess_log.json per dataset

DVC pipeline stages:
- 6 preprocessing stages (all datasets)
- 6 concept bank stages (all datasets)
- All outputs tracked with DVC"

# Verify commit
git log -1 --stat
```

---

## 📋 Step 6: Push to DVC Remote

```powershell
# Push processed data to DVC remote
dvc push

# This will upload:
# - data/processed/isic2018 (~2.5 GB)
# - data/processed/isic2019 (~5.0 GB)
# - data/processed/isic2020 (~6.5 GB)
# - data/processed/derm7pt (~0.4 GB)
# - data/processed/nih_cxr (~20 GB)
# - data/processed/padchest (~8 GB)
# - data/concepts/ (~10 MB)
# Total: ~42.5 GB

# Verify DVC remote
dvc list -R . --dvc-only
```

---

## 📋 Step 7: Push to Git Remote

```powershell
# Push Git changes
git push origin main

# Verify
git status
```

---

## 🔍 Troubleshooting

### If preprocessing fails:

```powershell
# Check error logs
cat data\processed\{dataset}\preprocess_log.json

# Re-run specific stage
dvc repro --force preprocess_{dataset}

# Check DVC status
dvc status
```

### If out of disk space:

```powershell
# Remove test directory
Remove-Item data\processed\isic2018_test -Recurse -Force

# Check available space
Get-Volume C
```

### If memory error:

The preprocessing script processes images one at a time, so memory errors are unlikely. If they occur:

```powershell
# Run with smaller batches (modify dvc.yaml if needed)
# Or reboot and run again
```

---

## 📊 Expected Results

After completion:

### Directory Structure

```
data/processed/
├── isic2018/
│   ├── images/
│   │   ├── train/ (10,015 images)
│   │   ├── val/ (193 images)
│   │   └── test/ (1,512 images)
│   ├── metadata_processed.csv
│   ├── preprocess_log.json
│   └── dataset.h5
├── isic2019/ (same structure)
├── isic2020/ (same structure)
├── derm7pt/ (same structure)
├── nih_cxr/ (same structure)
└── padchest/ (same structure)

data/concepts/
├── isic2018_concept_bank.json
├── isic2019_concept_bank.json
├── isic2020_concept_bank.json
├── derm7pt_concept_bank.json
├── nih_cxr_concept_bank.json
└── padchest_concept_bank.json
```

### File Counts

- **Total processed images:** ~223,297
- **Total storage:** ~42.5 GB
- **Processing time:** ~137 minutes (~2.3 hours)

---

## ✅ Completion Criteria

Phase 2.5 is 100% complete when:

- ✅ All 6 datasets preprocessed successfully
- ✅ All 6 concept banks generated
- ✅ All preprocessing logs exist and valid
- ✅ All outputs tracked with DVC (.dvc files created)
- ✅ Changes committed to Git
- ✅ Processed data pushed to DVC remote
- ✅ Verification script passes (exit code 0)
- ✅ Documentation complete

---

## 📝 Next Steps After Completion

Once Phase 2.5 is 100% complete:

1. Run verification script: `.\scripts\data\verify_preprocessing.ps1`
2. I will create Phase 2.5 completion report
3. Update main README.md
4. Ready to proceed to **Phase 3: Model Architecture Implementation**

---

**Current Status:** Awaiting your command execution
**You run the commands, I'll handle the rest!** ✅
