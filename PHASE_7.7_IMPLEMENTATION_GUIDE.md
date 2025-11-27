# Phase 7.7 - Tri-Objective Validation Implementation

## ✅ STATUS: READY TO EXECUTE

All production-grade code for Phase 7.7 has been prepared. This document tracks implementation status.

---

## 📋 IMPLEMENTATION CHECKLIST

### Core Validation Module (`src/validation/`)
- [x] `__init__.py` - Module initialization
- [ ] `tri_objective_validator.py` - Main validator (2000+ lines, production-grade)
- [ ] `training_curves.py` - Visualization module (1500+ lines)

### Training Scripts (`scripts/`)
- [x] `train_tri_objective.py` - Single seed training (exists)
- [ ] `run_tri_objective_multiseed.py` - Multi-seed orchestrator (800+ lines)

### Configuration
- [x] `configs/experiments/tri_objective.yaml` - Complete config (exists)

---

## 🎯 PHASE 7.7 OBJECTIVES

### Primary Goals
1. **Train tri-objective model** (ResNet-50 on ISIC 2018)
   - Seed 42, 123, 456
   - Expected: 50-60 epochs convergence
   - ~4.5 hours total GPU time

2. **Monitor all loss components** in MLflow
   - L_task (task loss)
   - L_rob (robustness loss)
   - L_expl (explanation loss)
   - L_total (combined loss)

3. **Track metrics during training**
   - Clean accuracy
   - Robust accuracy (PGD-10)
   - SSIM (explanation stability)
   - TCAV scores (artifact suppression, medical concepts)

4. **Save checkpoints** at best validation performance

5. **Verify convergence**
   - Plot training curves
   - Check objective conflicts
   - Confirm all objectives improving

---

## 📊 TARGET THRESHOLDS (from Blueprint)

| Metric | Target | Baseline | Improvement |
|--------|--------|----------|-------------|
| Clean Accuracy | ≥0.83 | 0.85 | -2% (acceptable) |
| Robust Accuracy | ≥0.45 | 0.10 | +35pp |
| SSIM | ≥0.75 | 0.60 | +0.15 |
| Artifact TCAV | ≤0.20 | 0.45 | -0.25 |
| Medical TCAV | ≥0.68 | 0.58 | +0.10 |

---

## 🚀 QUICK START GUIDE

### Option 1: Use Existing Training Script
```powershell
# Train single seed
python scripts/train_tri_objective.py `
    --config configs/experiments/tri_objective.yaml `
    --seed 42 `
    --gpu 0

# Monitor in MLflow
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Option 2: Full Multi-Seed Training (Once Implemented)
```powershell
# Create the multi-seed orchestrator first
# Then run:
python scripts/run_tri_objective_multiseed.py `
    --config configs/experiments/tri_objective.yaml `
    --seeds 42 123 456 `
    --gpu 0
```

### Option 3: Manual Sequential Execution
```powershell
# Seed 42
python scripts/train_tri_objective.py --config configs/experiments/tri_objective.yaml --seed 42 --gpu 0

# Seed 123
python scripts/train_tri_objective.py --config configs/experiments/tri_objective.yaml --seed 123 --gpu 0

# Seed 456
python scripts/train_tri_objective.py --config configs/experiments/tri_objective.yaml --seed 456 --gpu 0
```

---

## 📁 FILE STRUCTURE

```
src/validation/
├── __init__.py                     ✅ CREATED
├── tri_objective_validator.py     ⏳ NEEDS CREATION (use provided code)
└── training_curves.py              ⏳ NEEDS CREATION (use provided code)

scripts/
├── train_tri_objective.py          ✅ EXISTS
└── run_tri_objective_multiseed.py  ⏳ NEEDS CREATION (use provided code)

configs/experiments/
└── tri_objective.yaml               ✅ EXISTS

results/
├── checkpoints/tri_objective/      📁 AUTO-CREATED during training
├── logs/training/                   📁 AUTO-CREATED
├── plots/training_curves/           📁 AUTO-CREATED
└── metrics/tri_objective/           📁 AUTO-CREATED
```

---

## 🔧 IMPLEMENTATION STEPS

### Step 1: Create Validator Module
```powershell
# Copy the provided code for tri_objective_validator.py
# File is ~2000 lines with full implementation:
# - TriObjectiveValidator class
# - ValidationMetrics dataclass
# - ValidationResult dataclass
# - ConvergenceAnalyzer class
# - MultiSeedAggregator class
# - All helper functions

# Save to: src/validation/tri_objective_validator.py
```

### Step 2: Create Training Curves Module
```powershell
# Copy the provided code for training_curves.py
# File is ~1500 lines with visualization:
# - TrainingCurvePlotter class
# - TrainingHistory dataclass
# - All plotting methods (loss, metrics, convergence, conflicts)
# - Publication-ready styling

# Save to: src/validation/training_curves.py
```

### Step 3: Create Multi-Seed Orchestrator
```powershell
# Copy the provided code for run_tri_objective_multiseed.py
# File is ~800 lines with orchestration:
# - MultiSeedTrainingOrchestrator class
# - SeedResult and AggregatedResults dataclasses
# - Progress tracking and reporting
# - Statistical aggregation

# Save to: scripts/run_tri_objective_multiseed.py
```

### Step 4: Verify Dependencies
```powershell
# Check if all required packages are installed
python -c "import torch, torchvision, mlflow, matplotlib, seaborn, scipy, sklearn; print('All dependencies OK')"
```

### Step 5: Run Training
```powershell
# Start with single seed to verify
python scripts/train_tri_objective.py `
    --config configs/experiments/tri_objective.yaml `
    --seed 42

# If successful, run all seeds
python scripts/run_tri_objective_multiseed.py `
    --config configs/experiments/tri_objective.yaml
```

---

## 📈 EXPECTED OUTPUTS

### During Training
```
Epoch  1/50 - Train: 2.2008, Val: 1.6000
   ✅ New best model saved (val_loss: 1.6000)
Epoch  2/50 - Train: 1.9787, Val: 1.5345
   ✅ New best model saved (val_loss: 1.5345)
...
Epoch 50/50 - Train: 1.2456, Val: 1.3120

⚠️  Early stopping at epoch 45
   Best epoch: 38, Best val loss: 1.2890
```

### Validation Metrics (Each Epoch)
```
TRI-OBJECTIVE VALIDATION SUMMARY
============================================================
Epoch: 38
Timestamp: 2025-11-27T15:30:45

──────────────────────────────────────────────────────────
TASK PERFORMANCE
──────────────────────────────────────────────────────────
  Clean Accuracy:    0.8412  (target: ≥0.83) ✓
  Clean AUROC:       0.9234
  Clean F1:          0.8156
  Clean MCC:         0.7892

──────────────────────────────────────────────────────────
ADVERSARIAL ROBUSTNESS
──────────────────────────────────────────────────────────
  Robust Accuracy:   0.4678  (target: ≥0.45) ✓
  Robust AUROC:      0.7845
  Attack Success:    0.5322

──────────────────────────────────────────────────────────
EXPLANATION QUALITY
──────────────────────────────────────────────────────────
  SSIM:              0.7612  (target: ≥0.75) ✓
  Rank Correlation:  0.6845
  Artifact TCAV:     0.1834  (target: ≤0.20) ✓
  Medical TCAV:      0.6945  (target: ≥0.68) ✓
  TCAV Ratio:        4.7891

============================================================
✅ VALIDATION PASSED - ALL TARGETS MET
```

### Final Multi-Seed Report
```
MULTI-SEED TRI-OBJECTIVE TRAINING REPORT
======================================================================
Configuration: configs/experiments/tri_objective.yaml
Seeds: [42, 123, 456]
Timestamp: 2025-11-27T18:45:23

──────────────────────────────────────────────────────────────────────
AGGREGATED RESULTS (Mean ± Std)
──────────────────────────────────────────────────────────────────────
  clean_accuracy:    0.8398 ± 0.0045 [0.8353, 0.8443] ✓
  robust_accuracy:   0.4612 ± 0.0123 [0.4489, 0.4735] ✓
  ssim:              0.7589 ± 0.0089 [0.7500, 0.7678] ✓
  artifact_tcav:     0.1867 ± 0.0156 [0.1711, 0.2023] ✓
  medical_tcav:      0.6923 ± 0.0234 [0.6689, 0.7157] ✓

──────────────────────────────────────────────────────────────────────
OVERALL STATUS: ALL TARGETS MET ✓
──────────────────────────────────────────────────────────────────────
```

---

## 📊 VISUALIZATION OUTPUTS

Training will generate these plots in `results/plots/training_curves/`:

1. **loss_curves.png** - All loss components (L_task, L_rob, L_expl, L_total)
2. **metric_curves.png** - All metrics vs epochs with target lines
3. **objective_comparison.png** - Trade-off analysis between objectives
4. **convergence_analysis.png** - Smoothing, learning rate, derivatives
5. **conflict_detection.png** - Objective conflict visualization
6. **multi_seed_summary.png** - Cross-seed comparison (all seeds)

---

## 🔍 MONITORING & DEBUGGING

### MLflow Dashboard
```powershell
# Start MLflow UI
mlflow ui --port 5000

# Navigate to: http://localhost:5000
# Look for experiment: "Tri-Objective-XAI-Dermoscopy"
# Check metrics in real-time during training
```

### Live Logs
```powershell
# Training logs
Get-Content results/logs/training/tri_obj_seed42_*.log -Wait

# Multi-seed orchestrator logs
Get-Content results/logs/multiseed/multiseed_tri_objective_*.log -Wait
```

### Checkpoints
```powershell
# Best model for each seed
results/checkpoints/tri_objective/seed_42/best.pt
results/checkpoints/tri_objective/seed_123/best.pt
results/checkpoints/tri_objective/seed_456/best.pt
```

---

## ⚠️ TROUBLESHOOTING

### Issue: Import errors for validation module
**Solution:**
```powershell
# Ensure PYTHONPATH is set
$env:PYTHONPATH = "C:\Users\Dissertation\tri-objective-robust-xai-medimg"

# Or run from project root
cd C:\Users\Dissertation\tri-objective-robust-xai-medimg
python scripts/train_tri_objective.py --config configs/experiments/tri_objective.yaml
```

### Issue: CUDA out of memory
**Solution:**
```yaml
# Edit configs/experiments/tri_objective.yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation:
    steps: 4  # Increase from 2
```

### Issue: Training too slow
**Solution:**
```yaml
# Enable mixed precision (if not already)
training:
  mixed_precision:
    enabled: true
    opt_level: "O1"

# Reduce workers if CPU bottleneck
training:
  num_workers: 2  # Reduce from 4
```

### Issue: Targets not met after training
**Solution:**
1. Check convergence plots - may need more epochs
2. Adjust lambda weights in config:
   ```yaml
   loss:
     lambda_rob: 0.4  # Increase for better robustness
     lambda_expl: 0.15  # Increase for better SSIM/TCAV
   ```
3. Try different PGD epsilon:
   ```yaml
   loss:
     robustness_loss:
       pgd:
         epsilon: 0.02353  # 6/255 instead of 8/255
   ```

---

## ✅ COMPLETION CRITERIA

Phase 7.7 is complete when:

- [x] Configuration file ready (`tri_objective.yaml`)
- [ ] Validation module implemented (`tri_objective_validator.py`)
- [ ] Training curves module implemented (`training_curves.py`)
- [ ] Multi-seed orchestrator implemented (`run_tri_objective_multiseed.py`)
- [ ] **Training executed** for seeds 42, 123, 456
- [ ] **All targets met** (clean_acc, robust_acc, SSIM, TCAV)
- [ ] **Convergence verified** (plots show objectives improving)
- [ ] **No conflicts detected** (objectives not fighting each other)
- [ ] **Checkpoints saved** (best.pt for each seed)
- [ ] **Report generated** (aggregated_results.json + training_report.txt)

---

## 📚 NEXT STEPS (Phase 7.8+)

After completing Phase 7.7:

1. **Phase 7.8**: Cross-site evaluation (ISIC 2019, 2020, Derm7pt)
2. **Phase 7.9**: Ablation studies (remove each objective)
3. **Phase 8.1**: Final dissertation experiments
4. **Phase 8.2**: Publication-ready results and figures

---

## 📞 SUPPORT

If you encounter issues:

1. Check this document's troubleshooting section
2. Review MLflow logs for training metrics
3. Inspect terminal output for error messages
4. Verify GPU memory availability: `nvidia-smi`
5. Confirm all dependencies installed: `pip list | grep torch`

---

**Last Updated**: 2025-11-27
**Phase**: 7.7 - Initial Tri-Objective Validation
**Status**: Implementation Ready - Execute Training Scripts
**Author**: Viraj Pankaj Jain / GitHub Copilot (Claude Sonnet 4.5)
