# Phase 5.4 Medical Dataset Integration Guide

**Date:** November 24, 2025
**Purpose:** Adapt Phase 5.4 HPO for Medical Imaging Datasets

---

## 🚨 **Critical Issue Identified**

Phase 5.4 HPO currently uses **CIFAR-10** (toy dataset) instead of your actual medical imaging datasets:
- ISIC 2018/2019/2020 (Dermatology)
- Derm7pt (Dermoscopic images)
- PadChest (Chest X-rays)
- NIH ChestX-ray14

**This must be fixed before dissertation submission!**

---

## ✅ **What You Have:**

Your project already has:
- ✅ Medical dataset configurations in `configs/datasets/`
- ✅ Dataset classes in `src/datasets/`
- ✅ Data processing pipelines
- ✅ DVC tracking for medical data

**Files confirmed:**
- `configs/datasets/isic_2018.yaml`
- `configs/datasets/isic_2019.yaml`
- `configs/datasets/isic_2020.yaml`
- `configs/datasets/derm7pt.yaml`
- `configs/datasets/padchest.yaml`
- `configs/datasets/nih_cxr.yaml`

---

## 🔧 **Solution: Create Medical Dataset HPO Script**

### **Step 1: Create run_hpo_medical.py**

```python
# scripts/run_hpo_medical.py
"""
HPO Study for Medical Imaging Datasets.
Supports ISIC, Derm7pt, PadChest, NIH CXR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import torch
from src.datasets import get_dataset
from src.training.hpo_trainer import TRADESHPOTrainer
from src.training.hpo_config import create_default_hpo_config
from src.training.hpo_objective import WeightedTriObjective

# Medical dataset configurations
MEDICAL_DATASETS = {
    "isic2018": {"num_classes": 7, "img_size": 224},
    "isic2019": {"num_classes": 8, "img_size": 224},
    "isic2020": {"num_classes": 1, "img_size": 224},  # Binary
    "derm7pt": {"num_classes": 2, "img_size": 224},
    "padchest": {"num_classes": 14, "img_size": 224},  # Multi-label
    "nih_cxr": {"num_classes": 14, "img_size": 224},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(MEDICAL_DATASETS.keys()),
        help="Medical dataset to use"
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick-test", action="store_true")

    args = parser.parse_args()

    # Auto-generate study name
    if args.study_name is None:
        args.study_name = f"trades_hpo_{args.dataset}"

    # Load medical dataset
    dataset_config = MEDICAL_DATASETS[args.dataset]
    train_dataset = get_dataset(
        name=args.dataset,
        split="train",
        img_size=dataset_config["img_size"]
    )
    val_dataset = get_dataset(
        name=args.dataset,
        split="val",
        img_size=dataset_config["img_size"]
    )

    # Quick test mode
    if args.quick_test:
        args.n_trials = 10
        args.n_epochs = 2
        train_dataset = Subset(train_dataset, range(min(800, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(200, len(val_dataset))))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model factory
    def model_factory():
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, dataset_config["num_classes"])
        return model

    # Setup HPO
    hpo_config = create_default_hpo_config()
    hpo_config.study_name = args.study_name
    hpo_config.n_trials = args.n_trials

    objective = WeightedTriObjective(
        robust_weight=0.4,
        clean_weight=0.3,
        auroc_weight=0.3
    )

    # Run HPO
    trainer = TRADESHPOTrainer(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        config=hpo_config,
        objective_function=objective,
        n_epochs=args.n_epochs,
        device=args.device
    )

    trainer.create_study()
    trainer.run_study(n_trials=args.n_trials)

    print(f"✅ HPO completed for {args.dataset}")
    print(f"Best trial: {trainer.study.best_trial.number}")
    print(f"Best value: {trainer.study.best_value:.4f}")
    print(f"Best params: {trainer.study.best_params}")

if __name__ == "__main__":
    main()
```

---

## 📋 **Usage Commands for Google Colab:**

### **Quick Test (Each Dataset):**

```python
# ISIC 2018 (Skin Lesions - 7 classes)
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --quick-test \
    --device cuda

# ISIC 2019 (Skin Lesions - 8 classes)
!python scripts/run_hpo_medical.py \
    --dataset isic2019 \
    --quick-test \
    --device cuda

# Derm7pt (Dermoscopy - Binary)
!python scripts/run_hpo_medical.py \
    --dataset derm7pt \
    --quick-test \
    --device cuda

# NIH CXR (Chest X-ray - 14 classes)
!python scripts/run_hpo_medical.py \
    --dataset nih_cxr \
    --quick-test \
    --device cuda

# PadChest (Chest X-ray - 14 classes)
!python scripts/run_hpo_medical.py \
    --dataset padchest \
    --quick-test \
    --device cuda
```

### **Full HPO (Production):**

```python
# Full HPO on ISIC 2018 (2-3 hours)
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --n-trials 50 \
    --n-epochs 10 \
    --device cuda

# Full HPO on NIH CXR (2-3 hours)
!python scripts/run_hpo_medical.py \
    --dataset nih_cxr \
    --n-trials 50 \
    --n-epochs 10 \
    --device cuda
```

---

## 🎯 **Recommended Workflow:**

### **Phase 1: Quick Validation (30 minutes)**
Test HPO on each medical dataset:

```python
datasets = ["isic2018", "derm7pt", "nih_cxr"]

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Testing HPO on {dataset}")
    print('='*60)
    !python scripts/run_hpo_medical.py --dataset {dataset} --quick-test
```

### **Phase 2: Select Primary Datasets (Your Choice)**
Choose 2-3 datasets for full HPO based on:
- Data quality
- Dataset size
- Clinical relevance
- Time constraints

**Recommendation:**
1. **ISIC 2018** - Dermatology (10,015 images, 7 classes)
2. **NIH CXR** - Radiology (112,120 images, 14 classes)

### **Phase 3: Full HPO on Selected Datasets**

```python
# Run full HPO on selected datasets
!python scripts/run_hpo_medical.py \
    --dataset isic2018 \
    --n-trials 50 \
    --n-epochs 10

!python scripts/run_hpo_medical.py \
    --dataset nih_cxr \
    --n-trials 50 \
    --n-epochs 10
```

### **Phase 4: Cross-Site Generalization**
Train on one dataset, test on another:
- Train on ISIC 2018 → Test on Derm7pt
- Train on NIH CXR → Test on PadChest

---

## 📊 **Expected Results:**

### **ISIC 2018 (After full training):**
- Clean accuracy: 75-85%
- Robust accuracy: 50-65%
- Cross-site AUROC: 70-80%

### **NIH ChestX-ray14:**
- Clean AUC: 75-85%
- Robust AUC: 55-70%
- Cross-site AUROC: 70-80%

---

## ⚠️ **Important Notes:**

### **1. Data Availability**
Ensure datasets are downloaded and processed:
```python
# Check if data exists
from pathlib import Path

datasets_to_check = {
    "ISIC 2018": Path("data/processed/isic_2018"),
    "Derm7pt": Path("data/processed/derm7pt"),
    "NIH CXR": Path("data/processed/nih_cxr"),
}

for name, path in datasets_to_check.items():
    if path.exists():
        print(f"✅ {name}: {path}")
    else:
        print(f"❌ {name}: NOT FOUND - Need to download/process")
```

### **2. Computational Cost**
Medical images (224×224) take ~3-5x longer than CIFAR-10 (32×32):
- CIFAR-10: ~10 seconds per trial
- Medical: ~30-50 seconds per trial
- Full HPO (50 trials): 2-4 hours

### **3. Memory Requirements**
- Batch size 32: ~8GB GPU memory
- Batch size 64: ~12GB GPU memory
- Batch size 128: ~20GB GPU memory (may OOM on Colab)

---

## 🚀 **Action Plan:**

### **Immediate (Today):**
1. ✅ Acknowledge CIFAR-10 was for testing only
2. ⏳ Create `run_hpo_medical.py` script
3. ⏳ Test on ISIC 2018 (quick test)
4. ⏳ Verify medical datasets are accessible

### **Short-term (This Week):**
1. Run full HPO on ISIC 2018
2. Run full HPO on NIH CXR or Derm7pt
3. Compare results with CIFAR-10 baseline

### **Dissertation:**
1. Document why CIFAR-10 was used (infrastructure testing)
2. Present main results on medical datasets
3. Include cross-site generalization analysis
4. Show CIFAR-10 as ablation/validation study

---

## 📝 **Dissertation Section Structure:**

```markdown
## 5.4 Hyperparameter Optimization

### 5.4.1 Infrastructure Validation
Initial testing on CIFAR-10 to validate HPO pipeline...

### 5.4.2 Medical Dataset HPO
Applied validated pipeline to medical imaging datasets:
- ISIC 2018 (Dermatology)
- NIH ChestX-ray14 (Radiology)

### 5.4.3 Results
[Present medical dataset results here]

### 5.4.4 Cross-Site Generalization
Evaluated optimal hyperparameters across datasets...
```

---

## ✅ **Next Steps:**

**Choose one:**

**Option A: Quick Medical Test (30 min)**
```python
!python scripts/run_hpo_medical.py --dataset isic2018 --quick-test
```

**Option B: Full Medical HPO (3 hours)**
```python
!python scripts/run_hpo_medical.py --dataset isic2018 --n-trials 50 --n-epochs 10
```

**Option C: Keep CIFAR-10, Add Medical Later**
- Document CIFAR-10 as infrastructure validation
- Proceed to Phase 5.5 with CIFAR-10
- Come back to medical datasets after Phase 5.5

---

**Which option do you prefer? Should I create the `run_hpo_medical.py` script now?**
