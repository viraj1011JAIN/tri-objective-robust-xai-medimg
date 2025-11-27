# Phase 7.6: Tri-Objective Training - Chest X-Ray (Multi-Label)

**Complete Implementation - A1+ Grade Target**
**Status**: ✅ **PRODUCTION READY**
**Date**: November 27, 2025

---

## Overview

Phase 7.6 implements tri-objective adversarial training for **multi-label chest X-ray classification** on the NIH ChestX-ray14 dataset. This adapts the Phase 7.5 framework for multi-label scenarios where each image can have multiple disease labels simultaneously.

### Research Questions
- **RQ1**: Can adversarial robustness and cross-site generalization be jointly optimized for multi-label medical imaging?
- **RQ2**: Does concept-grounded regularization produce stable explanations for multi-label predictions?

### Key Adaptations for Multi-Label

| Aspect | Single-Label (Dermoscopy) | Multi-Label (CXR) |
|--------|---------------------------|-------------------|
| **Task Loss** | Cross-Entropy | BCE with Focal Loss |
| **Output Activation** | Softmax | Sigmoid (per-class) |
| **Metrics** | Accuracy, AUROC | Macro/Micro AUROC, Hamming Loss |
| **Thresholds** | N/A | Per-class optimal thresholds |
| **Robustness Loss** | TRADES (KL on softmax) | TRADES (KL on sigmoid) |
| **Explanation Loss** | **Same** | **Same** (architecture-agnostic) |

---

## Dataset: NIH ChestX-ray14

- **Total Images**: 112,120 frontal-view chest X-rays
- **Patients**: 30,805 unique patients
- **Classes**: 14 diseases (multi-label)
- **Labels per Image**: 0-5 (average ~1.5)
- **Image Size**: 224×224 (resized from 1024×1024)

### 14 Disease Classes

1. Atelectasis (10.3%)
2. Cardiomegaly (2.5%)
3. Effusion (11.9%)
4. Infiltration (17.7%)
5. Mass (5.1%)
6. Nodule (5.6%)
7. Pneumonia (1.2%)
8. Pneumothorax (4.7%)
9. Consolidation (4.1%)
10. Edema (2.0%)
11. Emphysema (2.2%)
12. Fibrosis (1.5%)
13. Pleural Thickening (3.0%)
14. Hernia (0.2%)

---

## Files Created (Phase 7.6)

```
tri-objective-robust-xai-medimg/
├── configs/experiments/
│   └── tri_objective_cxr.yaml                    # Multi-label CXR config (327 lines)
├── src/losses/
│   └── multi_label_task_loss.py                  # BCE/Focal loss (333 lines)
├── scripts/training/
│   ├── train_tri_objective_cxr.py                # Main training script (634 lines)
│   └── run_tri_objective_cxr_multiseed.sh        # Multi-seed bash script (150 lines)
└── PHASE_7.6_README.md                           # This file
```

**Total**: 1,444 lines of production-grade code

---

## Tri-Objective Loss Formulation

```
L_total = L_task + λ_rob × L_rob + λ_expl × L_expl
```

### 1. Task Loss (L_task) - Multi-Label Focal Loss

```python
L_task = -1/(N·C) Σ Σ α_c (1-p_{i,c})^γ log(p_{i,c})
```

Where:
- **p_{i,c}**: Sigmoid probability for sample i, class c
- **α_c**: Class weight (from positive rates)
- **γ**: Focusing parameter (2.0)

**Rationale**: Focal loss addresses severe class imbalance (Hernia: 0.2%)

### 2. Robustness Loss (L_rob) - TRADES (Adapted)

```python
L_rob = KL(σ(f(x)) || σ(f(x_adv)))
```

- **σ**: Sigmoid (not softmax for multi-label)
- **ε**: 4/255 (smaller than dermoscopy 8/255)

**Rationale**: Smaller ε for CXR to preserve anatomical structures

### 3. Explanation Loss (L_expl) - Same as Dermoscopy

```python
L_expl = L_stab + γ × L_concept
```

- **L_stab**: SSIM stability (ε=1/255)
- **L_concept**: TCAV (artifact suppression + medical concept encouragement)

**Rationale**: XAI framework is architecture-agnostic

---

## Configuration Highlights

### Model Architecture
```yaml
model:
  architecture: resnet50
  pretrained: true
  num_classes: 14
  dropout: 0.3
  multilabel: true  # Sigmoid activation
```

### Loss Configuration
```yaml
loss:
  task:
    type: bce_with_logits
    use_focal: true
    focal_gamma: 2.0
    focal_alpha: 0.25
  robustness:
    type: trades
    beta: 6.0
    epsilon: 0.01569  # 4/255
  explanation:
    gamma: 0.5
  lambda_rob: 0.3
  lambda_expl: 0.1
```

### Training Configuration
```yaml
training:
  num_epochs: 60
  batch_size: 32
  optimizer:
    type: adamw
    lr: 0.0001
  scheduler:
    type: cosine
  mixed_precision: true
  early_stopping:
    patience: 15
```

---

## Quick Start

### Prerequisites

```powershell
# Check GPU
nvidia-smi

# Verify data
Test-Path data\processed\nih_cxr\

# Verify concept CAVs
Test-Path data\concepts\chest_xray\cavs\
```

### 1. Single Seed Training (Debug Test)

```powershell
# Quick 5-epoch test
python scripts/training/train_tri_objective_cxr.py `
    --config configs/experiments/tri_objective_cxr.yaml `
    --seed 42 `
    --gpu 0 `
    --debug
```

**Expected**: Completes in ~1.5 hours, loss decreases

### 2. Full Single Seed Training

```powershell
# 60 epochs (~15-20 GPU hours)
python scripts/training/train_tri_objective_cxr.py `
    --config configs/experiments/tri_objective_cxr.yaml `
    --seed 42 `
    --gpu 0
```

### 3. Multi-Seed Training (Production)

```bash
# All 3 seeds: 42, 123, 456 (~45-60 GPU hours total)
bash scripts/training/run_tri_objective_cxr_multiseed.sh
```

---

## Expected Results (A1+ Grade Targets)

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| **Macro AUROC (clean)** | 0.78 | ≥0.76 | -2% acceptable |
| **Macro AUROC (robust)** | 0.40 | ≥0.65 | +62.5% |
| **Cross-Site AUROC Drop** | 0.15 | ≤0.10 | -33% |
| **Hamming Loss** | 0.12 | ≤0.15 | Maintain |
| **SSIM Stability** | 0.55 | ≥0.75 | +36% |
| **Artifact TCAV** | 0.50 | ≤0.20 | -60% |
| **Medical TCAV** | 0.55 | ≥0.68 | +24% |
| **ECE** | 0.12 | ≤0.10 | -17% |

---

## Training Timeline

| Configuration | Epochs | Time per Epoch | Total Time |
|---------------|--------|----------------|------------|
| **Debug Test** | 5 | 15 min | ~1.5 hours |
| **Single Seed** | 60 | 18 min | ~18 hours |
| **Multi-Seed (3 seeds)** | 180 | 18 min | ~54 hours |

**Hardware**: RTX 2080 Ti (11GB) or equivalent

---

## Troubleshooting

### Out of Memory (OOM)

**Solution**:
```yaml
batch_size: 16  # Reduce from 32
gradient_accumulation_steps: 2
num_workers: 2
```

### Training Very Slow

**Solution**:
```yaml
num_workers: 8  # Increase
eval_every_n_epochs: 2  # Reduce validation frequency
```

### Loss Not Decreasing

**Solution**:
```yaml
lr: 0.00005  # Reduce learning rate
lambda_rob: 0.2  # Reduce weight
lambda_expl: 0.05
```

---

## Success Criteria (A1+ Grade)

### MUST HAVE ✅

- [ ] All 3 seeds complete successfully
- [ ] Macro AUROC (clean) ≥ 0.76
- [ ] Robust AUROC ≥ 0.65
- [ ] SSIM ≥ 0.75
- [ ] Artifact TCAV ≤ 0.20
- [ ] Training curves show improvement
- [ ] Checkpoints saved and loadable
- [ ] MLflow logs complete

### SHOULD HAVE ⭐

- [ ] Cross-site drop ≤ 0.10
- [ ] Medical TCAV ≥ 0.68
- [ ] ECE ≤ 0.10
- [ ] Seed variance <3%
- [ ] Hamming Loss ≤ 0.15

---

## Next Steps After Training

### 1. Check Results in MLflow

```powershell
mlflow ui --host 0.0.0.0 --port 5000
```

### 2. Aggregate Multi-Seed Results

```powershell
python scripts/results/aggregate_multiseed_cxr.py `
    --seeds 42 123 456 `
    --output results/metrics/tri_objective_cxr/aggregated.csv
```

### 3. Cross-Site Evaluation (Phase 9)

```powershell
python scripts/evaluation/evaluate_cross_site_cxr.py `
    --checkpoint results/checkpoints/tri_objective_cxr/tri_obj_resnet50_nih_cxr_best_seed42.pt `
    --target_dataset padchest
```

---

## Key Differences from Phase 7.5 (Dermoscopy)

1. **Multi-Label Loss**: BCE with Focal Loss instead of Cross-Entropy
2. **Sigmoid Activation**: Per-class independent probabilities
3. **Multi-Label Metrics**: Macro/Micro AUROC, Hamming Loss, per-class F1
4. **Threshold Optimization**: Per-class thresholds (not fixed 0.5)
5. **Smaller ε**: 4/255 for robustness (vs. 8/255) to preserve anatomy
6. **Concept Bank**: CXR-specific artifacts and medical concepts
7. **Conservative Augmentation**: Small rotations (10°) to preserve structures

**Same**: Explanation loss, TRADES framework, training pipeline, MLflow integration

---

## References

- **NIH ChestX-ray14**: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database", CVPR 2017
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **TRADES**: Zhang et al., "Theoretically Principled Trade-off", ICML 2019

---

**Status**: Phase 7.6 Implementation Complete ✅
**Next**: Phase 7.7 - Initial Tri-Objective Validation
**Timeline**: 45-60 GPU hours for all 3 seeds
**Target**: A1+ Grade, Publication-Ready
