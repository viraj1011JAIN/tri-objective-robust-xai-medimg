# Phase 5.4 - Google Colab Execution Guide

**Date:** November 24, 2025
**Purpose:** Run Phase 5.4 HPO on Google Colab with GPU acceleration

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Setup Colab Environment

```python
# Cell 1: Clone repository and setup
!git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git
%cd tri-objective-robust-xai-medimg

# Check GPU availability
!nvidia-smi

# Install dependencies
!pip install -q torch torchvision optuna pandas matplotlib seaborn plotly mlflow openpyxl
```

### Step 2: Verify Installation

```python
# Cell 2: Verify all modules are installed
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path.cwd()))

# Test imports
try:
    import torch
    import optuna
    from src.training import hpo_config, hpo_objective, hpo_trainer, hpo_analysis
    print("✅ All modules successfully imported!")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ Optuna version: {optuna.__version__}")
except Exception as e:
    print(f"❌ Import error: {e}")
```

---

## 🧪 Testing Phase 5.4 (Choose One)

### Option 1: Quick Test (RECOMMENDED for testing - 5 minutes)

```python
# Cell 3: Quick test with minimal resources
!python scripts/run_hpo_study.py \
    --study-name test_phase54 \
    --n-trials 5 \
    --n-epochs 2 \
    --dataset cifar10 \
    --model resnet18 \
    --batch-size 64 \
    --device cuda \
    --output-dir results/quick_test \
    --quick-test

print("\n✅ Quick test completed! Check results/quick_test/")
```

### Option 2: Mini Pipeline (15-20 minutes)

```python
# Cell 4: Small but realistic test
!python scripts/run_hpo_study.py \
    --study-name mini_phase54 \
    --n-trials 10 \
    --n-epochs 5 \
    --dataset cifar10 \
    --model resnet18 \
    --batch-size 128 \
    --device cuda \
    --output-dir results/mini_test

print("\n✅ Mini pipeline completed!")
```

### Option 3: Full Quick Test (30-40 minutes)

```python
# Cell 5: Full quick test (10 trials, 2 epochs per trial)
!python scripts/run_hpo_study.py \
    --quick-test \
    --study-name quick_phase54 \
    --n-trials 10 \
    --n-epochs 2 \
    --dataset cifar10 \
    --model resnet18 \
    --batch-size 128 \
    --device cuda \
    --output-dir results/quick_phase54

print("\n✅ Full quick test completed!")
```

---

## 🎯 Production Pipeline (2-3 hours on Colab GPU)

### Full HPO Study (50 trials)

```python
# Cell 6: Full production HPO
!python scripts/run_hpo_study.py \
    --study-name trades_hpo_phase54 \
    --n-trials 50 \
    --n-epochs 10 \
    --dataset cifar10 \
    --model resnet18 \
    --batch-size 128 \
    --device cuda \
    --output-dir results/phase_5_4 \
    --robust-weight 0.4 \
    --clean-weight 0.3 \
    --auroc-weight 0.3

print("\n✅ HPO study completed!")
print("📊 Check results/phase_5_4/analysis/ for visualizations")
```

### Retrain with Optimal Hyperparameters

```python
# Cell 7: Retrain with best hyperparameters (200 epochs, ~2 hours)
!python scripts/retrain_optimal.py \
    --study-name trades_hpo_phase54 \
    --storage sqlite:///hpo_study.db \
    --dataset cifar10 \
    --model resnet18 \
    --n-epochs 200 \
    --batch-size 128 \
    --device cuda \
    --checkpoint-dir checkpoints/final_model \
    --output-dir results/phase_5_4/final_model \
    --use-scheduler \
    --save-frequency 10

print("\n✅ Retraining completed!")
print("📦 Model saved in checkpoints/final_model/")
```

### With MLflow Tracking

```python
# Cell 8: Full pipeline with MLflow
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("phase_5_4_colab")

# Run HPO with MLflow
!python scripts/run_hpo_study.py \
    --study-name trades_hpo_mlflow \
    --n-trials 50 \
    --n-epochs 10 \
    --dataset cifar10 \
    --model resnet18 \
    --device cuda

# Retrain with MLflow
!python scripts/retrain_optimal.py \
    --study-name trades_hpo_mlflow \
    --n-epochs 200 \
    --use-scheduler \
    --use-mlflow

print("\n✅ Pipeline with MLflow completed!")
```

---

## 📊 Verify Results

### Check HPO Results

```python
# Cell 9: Load and display results
import json
import pandas as pd
from pathlib import Path

# Load HPO summary
summary_path = Path("results/phase_5_4/analysis/hpo_summary.json")
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)

    print("=" * 60)
    print("HPO SUMMARY")
    print("=" * 60)
    print(f"Study Name: {summary['study_name']}")
    print(f"Total Trials: {summary['n_trials']}")
    print(f"Completed: {summary['n_complete']}")
    print(f"Pruned: {summary['n_pruned']}")
    print(f"\nBest Value: {summary['best_value']:.4f}")
    print(f"Best Trial: #{summary['best_trial_number']}")
    print(f"\nOptimal Hyperparameters:")
    for param, value in summary['best_params'].items():
        print(f"  {param}: {value}")
else:
    print("❌ Summary not found. Run HPO first.")
```

### Display Visualizations

```python
# Cell 10: Show plots
from IPython.display import Image, display
import matplotlib.pyplot as plt

plots_dir = Path("results/phase_5_4/analysis")
plot_files = [
    "optimization_history.png",
    "parameter_importance.png",
    "parameter_relationships.png",
    "objective_tradeoffs.png",
    "convergence_analysis.png"
]

for plot_file in plot_files:
    plot_path = plots_dir / plot_file
    if plot_path.exists():
        print(f"\n{'=' * 60}")
        print(f"{plot_file.replace('_', ' ').title()}")
        print('=' * 60)
        display(Image(filename=str(plot_path)))
    else:
        print(f"⚠️ {plot_file} not found")
```

### Check Final Model Performance

```python
# Cell 11: Display final model metrics
summary_path = Path("results/phase_5_4/final_model/training_summary.json")
if summary_path.exists():
    with open(summary_path) as f:
        metrics = json.load(f)

    print("=" * 60)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Best Epoch: {metrics['best_epoch']}")
    print(f"Clean Accuracy: {metrics['final_clean_accuracy']:.2%}")
    print(f"Robust Accuracy: {metrics['final_robust_accuracy']:.2%}")
    print(f"Best Robust Accuracy: {metrics['best_robust_accuracy']:.2%}")
    print(f"\nOptimal Hyperparameters:")
    for param, value in metrics['hyperparameters'].items():
        print(f"  {param}: {value}")
else:
    print("❌ Final model summary not found. Run retraining first.")
```

---

## ✅ Verification Checklist

### Phase 5.4 is Complete When:

```python
# Cell 12: Automated verification
from pathlib import Path

def verify_phase54():
    """Check if Phase 5.4 is complete"""

    checks = {
        "HPO Study Database": Path("hpo_study.db").exists(),
        "HPO Summary": Path("results/phase_5_4/analysis/hpo_summary.json").exists(),
        "Optimization History": Path("results/phase_5_4/analysis/optimization_history.png").exists(),
        "Parameter Importance": Path("results/phase_5_4/analysis/parameter_importance.png").exists(),
        "Final Model Checkpoint": Path("checkpoints/final_model/best_model.pt").exists(),
        "Training Summary": Path("results/phase_5_4/final_model/training_summary.json").exists(),
    }

    print("=" * 60)
    print("PHASE 5.4 VERIFICATION")
    print("=" * 60)

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and passed

    print("=" * 60)
    if all_passed:
        print("🎉 PHASE 5.4 COMPLETE - Ready for Phase 5.5!")
        return True
    else:
        print("⚠️ PHASE 5.4 INCOMPLETE - Complete missing items above")
        return False

verify_phase54()
```

---

## 📦 Download Results

```python
# Cell 13: Package results for download
!zip -r phase_5_4_results.zip \
    results/phase_5_4/ \
    checkpoints/final_model/ \
    hpo_study.db \
    logs/

print("\n✅ Results packaged!")
print("📥 Download 'phase_5_4_results.zip' from Files panel")
```

---

## 🎯 Recommended Testing Workflow

### For Quick Testing (Choose One):

#### A. Ultra-Fast Test (2-3 minutes)
```python
# Minimal test to verify code works
!python scripts/run_hpo_study.py --quick-test --n-trials 3 --n-epochs 1
```

#### B. Standard Quick Test (5-10 minutes)
```python
# Balanced test with reasonable metrics
!python scripts/run_hpo_study.py --quick-test --n-trials 5 --n-epochs 2
```

#### C. Comprehensive Quick Test (20-30 minutes)
```python
# Full quick test with analysis
!python scripts/run_hpo_study.py --quick-test --n-trials 10 --n-epochs 2
```

### For Production Run:

```python
# Step 1: HPO Study (2-3 hours)
!python scripts/run_hpo_study.py \
    --study-name trades_hpo_phase54 \
    --n-trials 50 \
    --n-epochs 10 \
    --dataset cifar10 \
    --model resnet18 \
    --device cuda

# Step 2: Retrain (2-3 hours)
!python scripts/retrain_optimal.py \
    --study-name trades_hpo_phase54 \
    --n-epochs 200 \
    --use-scheduler

# Step 3: Verify
verify_phase54()
```

---

## 🚦 Progress to Phase 5.5

### Prerequisites for Phase 5.5:

✅ **Must Have:**
1. Completed HPO study with 50 trials
2. Trained final model (200 epochs)
3. HPO analysis visualizations
4. Optimal hyperparameters identified

✅ **Verification:**
```python
# Run verification cell (Cell 12)
if verify_phase54():
    print("\n🎉 Ready to proceed to Phase 5.5!")
    print("\nPhase 5.5 will focus on:")
    print("  • Advanced XAI methods (GradCAM, LIME, SHAP)")
    print("  • Explanation robustness evaluation")
    print("  • Cross-site generalization analysis")
    print("  • Integrated evaluation framework")
else:
    print("\n⚠️ Complete Phase 5.4 verification first")
```

---

## 💡 Pro Tips

### 1. Save Checkpoints Regularly
```python
# Colab disconnects after 12 hours, so save frequently
from google.colab import drive
drive.mount('/content/drive')

# Copy important files to Drive
!cp -r results/ /content/drive/MyDrive/phase_5_4_backup/
!cp hpo_study.db /content/drive/MyDrive/phase_5_4_backup/
```

### 2. Resume Interrupted Study
```python
# If Colab disconnects, resume from saved database
!python scripts/run_hpo_study.py \
    --study-name trades_hpo_phase54 \
    --storage sqlite:///hpo_study.db \
    --n-trials 50  # Will skip completed trials
```

### 3. Monitor GPU Usage
```python
# Check GPU utilization
!nvidia-smi -l 5  # Update every 5 seconds
```

### 4. Reduce Memory if Needed
```python
# If running out of memory
!python scripts/run_hpo_study.py \
    --batch-size 64 \  # Reduce from 128
    --num-workers 2    # Reduce from 4
```

---

## 🐛 Troubleshooting

### Issue 1: "CUDA out of memory"
```python
# Solution: Reduce batch size
!python scripts/run_hpo_study.py --batch-size 64 --quick-test
```

### Issue 2: "ModuleNotFoundError: No module named 'src'"
```python
# Solution: Add to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Issue 3: KeyError: 'learning_rate' (FIXED)
```python
# This issue has been fixed in hpo_config.py
# The to_dict() method now normalizes "log_float" to "float"
# If you still see this error, ensure you have the latest code:
!git pull origin main
```

### Issue 4: "Study already exists"
```python
# Solution: Use different name or delete database
!rm hpo_study.db
# Then re-run with same study name
```

### Issue 5: Colab Disconnected
```python
# Solution: Copy database from Drive and resume
!cp /content/drive/MyDrive/phase_5_4_backup/hpo_study.db .
!python scripts/run_hpo_study.py --study-name trades_hpo_phase54 --n-trials 50
```

---

## 📝 Summary Commands

### Quick Test Sequence:
```python
# 1. Setup (1 min)
!pip install -q torch torchvision optuna pandas matplotlib seaborn plotly

# 2. Quick test (5 min)
!python scripts/run_hpo_study.py --quick-test --n-trials 5 --n-epochs 2

# 3. Verify (1 min)
verify_phase54()
```

### Full Pipeline Sequence:
```python
# 1. HPO Study (2-3 hours)
!python scripts/run_hpo_study.py --study-name trades_hpo_phase54 --n-trials 50 --n-epochs 10

# 2. Retrain (2-3 hours)
!python scripts/retrain_optimal.py --study-name trades_hpo_phase54 --n-epochs 200 --use-scheduler

# 3. Verify (1 min)
verify_phase54()

# 4. Download results
!zip -r phase_5_4_results.zip results/ checkpoints/ hpo_study.db
```

---

## ✨ Next Steps to Phase 5.5

Once verification passes:

1. **Document Results:**
   - Save HPO summary JSON
   - Export visualizations
   - Record optimal hyperparameters

2. **Prepare for Phase 5.5:**
   - Ensure final model checkpoint exists
   - Verify model loads correctly
   - Test inference on sample images

3. **Ready for Phase 5.5 when:**
   - ✅ All verification checks pass
   - ✅ Final model achieves >60% robust accuracy
   - ✅ HPO visualizations generated
   - ✅ Optimal hyperparameters documented

---

**🎓 Phase 5.4 Complete → Ready for Phase 5.5 XAI Integration!**
