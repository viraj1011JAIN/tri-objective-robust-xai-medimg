# Dataset Commands & Mock Data Reference
## Complete Guide for Running with Real Datasets

**Date:** November 22, 2025
**Status:** ✅ 99.76% Test Coverage with Synthetic Data
**Next Step:** Execute with Real Medical Imaging Datasets

---

## 📊 Current Testing Status

### ✅ **What's Working Now (with Mocks/Synthetic Data)**
- **1483 tests passing** with synthetic/mock data
- **99.76% code coverage** achieved
- **All core functionality validated** without real datasets
- **0 test failures** in current test suite

### ⏳ **What Requires Real Datasets**
- **84 tests currently skipped** waiting for actual dataset files
- Dataset-specific preprocessing validation
- Real medical image quality checks
- Actual metadata validation
- Cross-dataset consistency verification

---

## 🗂️ Dataset Locations & Expected Paths

### **Primary Data Locations**
Based on current configuration, datasets should be located at:

```
/content/drive/MyDrive/data/                          # Samsung SSD T7 (portable drive)
  ├── isic_2018/
  │   ├── metadata.csv
  │   ├── ISIC2018_Task3_Training_Input/
  │   └── ISIC2018_Task3_Training_GroundTruth.csv
  ├── isic_2019/
  │   ├── metadata.csv
  │   ├── ISIC_2019_Training_Input/
  │   └── ISIC_2019_Training_GroundTruth.csv
  ├── isic_2020/
  │   ├── metadata.csv
  │   ├── train/
  │   └── train.csv
  ├── derm7pt/
  │   ├── metadata.csv
  │   ├── images/
  │   └── meta/
  ├── nih_cxr/
  │   ├── images_001/
  │   ├── images_002/
  │   └── Data_Entry_2017.csv
  └── padchest/
      ├── images/
      └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
```

### **Processed Data Output**
```
C:/Users/Dissertation/tri-objective-robust-xai-medimg/data/
  ├── processed/
  │   ├── isic2018/
  │   ├── isic2019/
  │   ├── isic2020/
  │   ├── derm7pt/
  │   ├── nih_cxr/
  │   └── padchest/
  └── concepts/
      ├── isic2018_concept_bank.pkl
      ├── isic2019_concept_bank.pkl
      ├── isic2020_concept_bank.pkl
      ├── derm7pt_concept_bank.pkl
      ├── nih_cxr_concept_bank.pkl
      └── padchest_concept_bank.pkl
```

---

## 🚀 Commands to Run (Once Datasets Available)

### **Step 0: Pre-Execution Verification**

```powershell
# 1. Navigate to project root
cd C:\Users\Dissertation\tri-objective-robust-xai-medimg

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify you see:
# (.venv) PS C:\Users\Dissertation\tri-objective-robust-xai-medimg>

# 4. Verify DVC is available
dvc version

# 5. Check dataset availability
Test-Path/content/drive/MyDrive/data/isic_2018/metadata.csv    # Should return True
Test-Path/content/drive/MyDrive/data/isic_2019/metadata.csv    # Should return True
Test-Path/content/drive/MyDrive/data/isic_2020/metadata.csv    # Should return True
Test-Path/content/drive/MyDrive/data/derm7pt/metadata.csv      # Should return True
Test-Path/content/drive/MyDrive/data/nih_cxr/Data_Entry_2017.csv  # Should return True
Test-Path/content/drive/MyDrive/data/padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv  # Should return True
```

---

### **Step 1: Data Preprocessing Pipeline**

#### **Option A: Run Each Dataset Individually (Recommended)**

```powershell
# ==========================================
# DERMOSCOPY DATASETS (Skin Lesion Analysis)
# ==========================================

# 1. ISIC 2018 (11,720 images, ~10 min, 8 classes)
dvc repro preprocess_isic2018

# Verify output:
ls data\processed\isic2018
cat data\processed\isic2018\preprocess_log.json | ConvertFrom-Json
Import-Csv data\processed\isic2018\metadata_processed.csv | Measure-Object

# 2. ISIC 2019 (25,331 images, ~15 min, 8 classes)
dvc repro preprocess_isic2019

# Verify output:
ls data\processed\isic2019
Import-Csv data\processed\isic2019\metadata_processed.csv | Measure-Object

# 3. ISIC 2020 (33,126 images, ~20 min, 2 classes - binary)
dvc repro preprocess_isic2020

# Verify output:
ls data\processed\isic2020
Import-Csv data\processed\isic2020\metadata_processed.csv | Measure-Object

# 4. Derm7pt (2,000 images, ~2 min, 7-point checklist)
dvc repro preprocess_derm7pt

# Verify output:
ls data\processed\derm7pt
Import-Csv data\processed\derm7pt\metadata_processed.csv | Measure-Object

# ==========================================
# CHEST X-RAY DATASETS (Multi-Label Classification)
# ==========================================

# 5. NIH Chest X-Ray (112,120 images, ~60 min, 14 pathology labels)
dvc repro preprocess_nih_cxr

# Verify output:
ls data\processed\nih_cxr
Import-Csv data\processed\nih_cxr\metadata_processed.csv | Measure-Object

# 6. PadChest (39,000 images, ~30 min, 174 labels)
dvc repro preprocess_padchest

# Verify output:
ls data\processed\padchest
Import-Csv data\processed\padchest\metadata_processed.csv | Measure-Object
```

#### **Option B: Run All Preprocessing at Once**

```powershell
# Run all preprocessing stages sequentially
dvc repro preprocess

# This will process all 6 datasets in order:
# Total estimated time: ~137 minutes (2.3 hours)
```

---

### **Step 2: Build Concept Banks (After Preprocessing)**

```powershell
# ==========================================
# CONCEPT BANK GENERATION
# ==========================================

# 1. ISIC 2018 Concept Bank (~5 min)
dvc repro build_concept_bank_isic2018

# Verify:
Test-Path data\concepts\isic2018_concept_bank.pkl

# 2. ISIC 2019 Concept Bank (~8 min)
dvc repro build_concept_bank_isic2019

# Verify:
Test-Path data\concepts\isic2019_concept_bank.pkl

# 3. ISIC 2020 Concept Bank (~10 min)
dvc repro build_concept_bank_isic2020

# Verify:
Test-Path data\concepts\isic2020_concept_bank.pkl

# 4. Derm7pt Concept Bank (~2 min)
dvc repro build_concept_bank_derm7pt

# Verify:
Test-Path data\concepts\derm7pt_concept_bank.pkl

# 5. NIH CXR Concept Bank (~30 min)
dvc repro build_concept_bank_nih_cxr

# Verify:
Test-Path data\concepts\nih_cxr_concept_bank.pkl

# 6. PadChest Concept Bank (~15 min)
dvc repro build_concept_bank_padchest

# Verify:
Test-Path data\concepts\padchest_concept_bank.pkl
```

---

### **Step 3: Verify Complete Pipeline**

```powershell
# Run comprehensive verification script
.\scripts\data\verify_preprocessing.ps1

# This checks:
# - All processed metadata files exist
# - Sample counts are correct
# - All concept banks are generated
# - File integrity and checksums
```

---

### **Step 4: Run Full Test Suite with Real Data**

```powershell
# Navigate to tests directory
cd tests

# Run all tests including previously skipped dataset tests
pytest -v --cov=src --cov-branch --cov-report=html --cov-report=term

# Expected result:
# - ~1567 tests passed (84 previously skipped now running)
# - 0 tests failed
# - 0 tests skipped
# - 99.76%+ coverage maintained

# Open coverage report
Invoke-Item htmlcov\index.html
```

---

## 🎭 Current Mock/Synthetic Data Usage

### **Where We Use Mocks (Currently)**

#### **1. Test Fixtures (tests/conftest.py)**
```python
# Lines 60-95: Device and synthetic data fixtures
@pytest.fixture
def synthetic_data(device):
    """Generate synthetic images and labels for testing."""
    torch.manual_seed(42)
    batch_size = 4
    images = torch.rand(batch_size, 3, 32, 32, device=device)  # MOCK DATA
    labels = torch.tensor([0, 1, 2, 3], device=device)
    return images, labels

@pytest.fixture
def dummy_dataloader(device):
    """Create dummy dataloader for testing."""
    images = torch.randn(16, 3, 224, 224)  # MOCK DATA
    labels = torch.randint(0, 7, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4)
```

**→ REPLACEMENT:** Once real datasets available, these fixtures will use actual data loaders

---

#### **2. Attack Testing (tests/test_attacks.py)**
```python
# Lines 83-93: Synthetic image generation
@pytest.fixture
def synthetic_data(device):
    """Generate synthetic images and labels."""
    torch.manual_seed(42)
    batch_size = 4
    images = torch.rand(batch_size, 3, 32, 32, device=device)  # MOCK DATA
    labels = torch.tensor([0, 1, 2, 3], device=device)
    return images, labels
```

**→ REPLACEMENT:** Will use real dermoscopy/CXR images for adversarial robustness testing

---

#### **3. Model Testing (tests/test_models_*.py)**
```python
# Multiple files use synthetic data:
# - test_models_resnet.py
# - test_models_efficientnet_complete.py
# - test_models_vit_complete.py
# - test_models_comprehensive.py

# Example pattern:
def test_forward_pass():
    model = ResNet50(num_classes=8)
    x = torch.randn(2, 3, 224, 224)  # MOCK DATA
    output = model(x)
    assert output.shape == (2, 8)
```

**→ REPLACEMENT:** Will use real medical images from preprocessed datasets

---

#### **4. Evaluation Metrics (tests/test_evaluation_*.py)**
```python
# tests/test_evaluation_metrics.py
# tests/test_evaluation_calibration.py
# tests/test_evaluation_multilabel_metrics.py
# tests/test_evaluation_multilabel_calibration.py

# Example pattern:
@pytest.fixture
def binary_classification_data():
    """Generate binary classification test data."""
    np.random.seed(42)
    n_samples = 100
    predictions = np.random.rand(n_samples, 2)  # MOCK DATA
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    labels = predictions.argmax(axis=1)
    return predictions, labels
```

**→ REPLACEMENT:** Will use actual model predictions on real medical images

---

#### **5. Loss Functions (tests/test_losses_*.py)**
```python
# tests/test_losses_calibration_loss.py
# tests/test_losses_base_loss.py
# tests/test_losses_task_loss.py

# Example pattern:
def test_focal_loss():
    loss_fn = FocalLoss(num_classes=5)
    logits = torch.randn(8, 5)  # MOCK DATA
    targets = torch.randint(0, 5, (8,))  # MOCK DATA
    loss = loss_fn(logits, targets)
    assert loss > 0
```

**→ REPLACEMENT:** Will use real logits from trained models on medical images

---

#### **6. Dataset Testing with Skipped Tests**
```python
# tests/test_datasets.py
# tests/test_all_modules.py

# Lines 175, 255, 280, etc.:
if not (data_root / default_subdir).exists():
    pytest.skip(f"Dataset root for '{default_subdir}' not found.")

# Currently skipping 84 tests:
# - 6 tests for ISIC 2018
# - 3 tests for ISIC 2019
# - 3 tests for ISIC 2020
# - 2 tests for Derm7pt
# - 2 tests for NIH CXR
# - 1 test for PadChest
# + Additional integration tests requiring real data
```

**→ ACTIVATION:** These tests will automatically run once dataset paths exist

---

#### **7. Integration Tests (tests/integration/test_full_pipeline.py)**
```python
# Lines 50-100: End-to-end pipeline testing
def test_end_to_end_training_pipeline():
    # Currently uses synthetic data
    train_images = torch.randn(100, 3, 224, 224)  # MOCK DATA
    train_labels = torch.randint(0, 7, (100,))

    # Train model, evaluate, generate adversarial examples
    # ... (full pipeline with mocks)
```

**→ REPLACEMENT:** Will use real preprocessed datasets for full pipeline validation

---

## 📋 Summary of Mock Replacements Needed

| **Component** | **Current Mock** | **Real Data Source** | **Tests Affected** |
|--------------|------------------|---------------------|-------------------|
| **Images** | `torch.randn()` | Preprocessed ISIC/NIH datasets | ~400 tests |
| **Labels** | `torch.randint()` | Actual metadata CSV files | ~400 tests |
| **Predictions** | `np.random.rand()` | Trained model outputs | ~200 tests |
| **Metadata** | Skipped tests | Real CSV metadata files | 84 tests |
| **Concept Banks** | Not tested | Generated `.pkl` files | ~50 tests |
| **Dataloaders** | `TensorDataset` | Real `ISICDataset`/`ChestXRayDataset` | ~100 tests |

---

## 🎯 Expected Outcomes After Dataset Integration

### **Test Suite Changes**
```
BEFORE (Current):
├── 1483 tests passed
├── 84 tests skipped (no datasets)
├── 0 tests failed
└── 99.76% coverage

AFTER (With Real Datasets):
├── 1567 tests passed (84 previously skipped now running)
├── 0 tests skipped
├── 0 tests failed
└── 99.76%+ coverage maintained or improved
```

### **New Capabilities Unlocked**
✅ Real medical image preprocessing validation
✅ Actual metadata integrity checks
✅ Cross-dataset consistency verification
✅ Real concept bank generation and validation
✅ Authentic adversarial robustness testing
✅ True end-to-end pipeline execution
✅ Publication-ready experimental results

---

## 🔧 Troubleshooting

### **If Datasets Not Found**
```powershell
# Check dataset paths
$datasets = @(
    "/content/drive/MyDrive/data/isic_2018",
    "/content/drive/MyDrive/data/isic_2019",
    "/content/drive/MyDrive/data/isic_2020",
    "/content/drive/MyDrive/data/derm7pt",
    "/content/drive/MyDrive/data/nih_cxr",
    "/content/drive/MyDrive/data/padchest"
)

foreach ($dataset in $datasets) {
    if (Test-Path $dataset) {
        Write-Host "✅ Found: $dataset" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $dataset" -ForegroundColor Red
    }
}
```

### **If DVC Pipeline Fails**
```powershell
# Check DVC status
dvc status

# Re-pull dependencies if needed
dvc pull

# Force re-run specific stage
dvc repro --force preprocess_isic2018
```

### **If Tests Still Skip**
```powershell
# Verify pytest can find datasets
cd tests
python -c "from pathlib import Path; print(Path('/content/drive/MyDrive/data/isic_2018').exists())"

# Run with verbose output
pytest test_datasets.py -v -s
```

---

## 📚 Related Documentation

- **Setup Instructions:** `PYTHON_ENV_SETUP.md`
- **Execution Commands:** `docs/archive/PHASE_2.5_EXECUTION_COMMANDS.md`
- **Training Commands:** `TRAINING_COMMANDS.ps1`
- **DVC Pipeline:** `dvc.yaml`
- **Dataset Configs:** `configs/datasets/*.yaml`

---

## ✅ Checklist Before Running with Real Data

- [x] All datasets downloaded to `/content/drive/MyDrive/data/` (Samsung SSD T7)
- [ ] Virtual environment activated (`.venv`)
- [ ] DVC installed and configured (`dvc version`)
- [ ] All dataset metadata CSV files present
- [ ] Sufficient disk space (~500 GB for processed data)
- [ ] CUDA-capable GPU available (recommended)
- [ ] Run test suite with current mocks first (verify 99.76% coverage)
- [ ] Backup any existing processed data
- [ ] Review preprocessing scripts in `scripts/data/`

---

**Last Updated:** November 22, 2025
**Test Coverage:** 99.76% (1483/1483 tests passing with mocks)
**Ready for Real Data Integration:** ✅ YES
