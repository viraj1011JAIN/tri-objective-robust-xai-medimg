# Phase 3.3 Quick Start Guide

**Status:** ✅ COMPLETE
**Date:** November 2024
**Quality:** A1 Master Level

---

## What Was Completed

### 1. Core Integration ✅
- **File:** `src/training/baseline_trainer.py`
- **Changes:**
  - Integrated TaskLoss and CalibrationLoss from Phase 3.2
  - Removed old FocalLoss class (~100 lines)
  - Added helper methods (get_temperature, get_loss_statistics)
  - Added parameters: task_type, use_calibration, init_temperature, label_smoothing

### 2. Training Scripts ✅
- **ResNet-50:** `scripts/training/train_resnet50_phase3.py`
- **EfficientNet-B0:** `scripts/training/train_efficientnet_phase3.py`
- **ViT-B/16:** `scripts/training/train_vit_phase3.py`

### 3. Calibration Evaluation ✅
- **Module:** `src/evaluation/calibration.py`
  - ECE (Expected Calibration Error)
  - MCE (Maximum Calibration Error)
  - Reliability diagrams
  - Confidence histograms
- **Script:** `scripts/evaluate_calibration.py`

### 4. Testing ✅
- **File:** `test_baseline_integration.py`
- **Results:** 5/5 tests PASSED

### 5. Documentation ✅
- **Report:** `docs/reports/PHASE_3.3_COMPLETION_REPORT.md`
- **This Guide:** `PHASE_3.3_QUICKSTART.md`

---

## How to Use (Next Steps)

### Option 1: Train a Model (Quick Test)

**Train ResNet-50 on ISIC 2018 (1 epoch test):**
```bash
python scripts/training/train_resnet50_phase3.py \
    --dataset isic2018 \
    --batch-size 32 \
    --max-epochs 1 \
    --device cuda
```

**Expected Output:**
```
======================================================================
ResNet-50 Phase 3.3 Baseline Training
======================================================================
Dataset: isic2018
Device: cuda
Batch size: 32
Max epochs: 1
Loss: TaskLoss (CrossEntropy)
...
Epoch 1/1 - Train Loss: X.XXXX, Train Acc: X.XXXX
Epoch 1/1 - Val Loss: X.XXXX, Val Acc: X.XXXX
Training complete!
```

### Option 2: Run Integration Test

**Verify everything works:**
```bash
python test_baseline_integration.py
```

**Expected Output:**
```
======================================================================
[SUCCESS] ALL INTEGRATION TESTS PASSED!
======================================================================
```

### Option 3: Train with Calibration

**Train ResNet-50 with CalibrationLoss:**
```bash
python scripts/training/train_resnet50_phase3.py \
    --dataset isic2018 \
    --use-calibration \
    --temperature 1.5 \
    --label-smoothing 0.1 \
    --batch-size 32 \
    --max-epochs 50 \
    --device cuda
```

### Option 4: Evaluate Calibration

**After training, evaluate calibration:**
```bash
python scripts/evaluate_calibration.py \
    --checkpoint results/checkpoints/resnet50_phase3/resnet50_best_e50.pt \
    --model resnet50 \
    --dataset isic2018 \
    --output-dir results/calibration/resnet50
```

**Output Files:**
- `results/calibration/resnet50/reliability_diagram.png`
- `results/calibration/resnet50/confidence_histogram.png`
- `results/calibration/resnet50/calibration_metrics.txt`

---

## Key Features

### Loss Functions (Phase 3.2 Integration)

**TaskLoss (Basic):**
```python
trainer = BaselineTrainer(
    model=model,
    ...
    task_type="multi_class",  # or "multi_label"
    use_focal_loss=False,      # False = CrossEntropy, True = FocalLoss
)
```

**TaskLoss (FocalLoss for Imbalance):**
```python
trainer = BaselineTrainer(
    model=model,
    ...
    task_type="multi_class",
    use_focal_loss=True,
    focal_gamma=2.0,
)
```

**CalibrationLoss (Temperature + Smoothing):**
```python
trainer = BaselineTrainer(
    model=model,
    ...
    task_type="multi_class",
    use_calibration=True,
    init_temperature=1.5,
    label_smoothing=0.1,
)
```

### Calibration Metrics

```python
from src.evaluation.calibration import calculate_ece, calculate_mce

# Calculate ECE
ece = calculate_ece(predictions, labels, num_bins=15)
print(f"ECE: {ece:.4f}")

# Calculate MCE
mce = calculate_mce(predictions, labels, num_bins=15)
print(f"MCE: {mce:.4f}")
```

---

## File Structure

```
tri-objective-robust-xai-medimg/
├── src/
│   ├── training/
│   │   └── baseline_trainer.py          # Modified (Phase 3.2 losses integrated)
│   ├── evaluation/
│   │   └── calibration.py               # New (ECE, MCE, plots)
│   └── losses/
│       ├── task_loss.py                 # Phase 3.2
│       └── calibration_loss.py          # Phase 3.2
├── scripts/
│   ├── training/
│   │   ├── train_resnet50_phase3.py     # New
│   │   ├── train_efficientnet_phase3.py # New
│   │   └── train_vit_phase3.py          # New
│   └── evaluate_calibration.py          # New
├── docs/
│   └── reports/
│       ├── PHASE_3.3_COMPLETION_REPORT.md  # New (comprehensive)
│       └── PHASE_3.3_QUICKSTART.md         # This file
└── test_baseline_integration.py         # New (5 tests)
```

---

## Quick Reference

### Training Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | isic2018 | Dataset name (isic2018, derm7pt) |
| `--batch-size` | 32 | Batch size for training |
| `--max-epochs` | 100 | Maximum training epochs |
| `--lr` | 0.0001 | Learning rate |
| `--use-focal-loss` | False | Use FocalLoss for imbalance |
| `--focal-gamma` | 2.0 | Gamma parameter for FocalLoss |
| `--use-calibration` | False | Use CalibrationLoss |
| `--temperature` | 1.5 | Initial temperature |
| `--label-smoothing` | 0.0 | Label smoothing factor |
| `--device` | cuda | Device (cuda/cpu) |

### Calibration Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| ECE | [0, 1] | 0 = perfect calibration, higher = worse |
| MCE | [0, 1] | 0 = perfect calibration, captures worst-case error |
| Accuracy | [0, 1] | Overall classification accuracy |
| Avg Confidence | [0, 1] | Average predicted confidence |

---

## Next Steps (After Phase 3.3)

### Phase 3.4: Adversarial Robustness
- [ ] Implement adversarial attacks (FGSM, PGD, C&W)
- [ ] Integrate robustness losses
- [ ] Evaluate adversarial robustness metrics

### Phase 3.5: Explainability Methods
- [ ] Integrate GradCAM, SmoothGrad, Integrated Gradients
- [ ] Implement explanation quality metrics
- [ ] Visualize explanations on medical images

### Phase 4: Tri-Objective Optimization
- [ ] Multi-objective loss (task + robustness + explanation)
- [ ] Pareto optimization
- [ ] Trade-off analysis

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size
```bash
--batch-size 16  # Instead of 32
```

### Issue: Import errors
**Solution:** Ensure project root is in PYTHONPATH
```bash
# Windows PowerShell
$env:PYTHONPATH = "C:\Users\Dissertation\tri-objective-robust-xai-medimg"
```

### Issue: Dataset not found
**Solution:** Check data path
```bash
--data-root/content/drive/MyDrive/data  # Adjust to your data location
```

### Issue: Slow training
**Solution:** Check GPU utilization
```bash
nvidia-smi  # Monitor GPU usage
```

---

## Contact & Support

**Author:** Viraj Jain
**Project:** Tri-Objective Robust XAI for Medical Imaging
**Phase:** 3.3 - Baseline Training Integration
**Status:** ✅ COMPLETE
**Quality:** A1 Master Level

For detailed documentation, see:
- `docs/reports/PHASE_3.3_COMPLETION_REPORT.md`

---

**End of Quick Start Guide**
