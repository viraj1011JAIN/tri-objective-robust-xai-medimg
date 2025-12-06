# Phase 7 Critical Bug Fix Summary

## 🚨 BUG IDENTIFIED AND FIXED

### The Problem
Tri-objective models claimed **78.7% validation accuracy** but actual accuracy was **3.22%**.
Models collapsed to predicting class 5 (NV) for 81% of samples.

### Root Cause Found
In `Phase7_TriObjective_Training.ipynb`, the `ISICDatasetColab` class was creating
class-to-index mappings **independently for each split**:

```python
# BUGGY CODE (line 1009):
self.class_names = sorted(df['label'].unique().tolist())  # Per-split unique!
self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
```

**Why this causes collapse:**
- If validation split is missing even ONE class, indices shift
- Example: If val set doesn't have 'AKIEC':
  - Train: AKIEC=0, BCC=1, BKL=2, DF=3, MEL=4, NV=5, VASC=6
  - Val:   BCC=0, BKL=1, DF=2, MEL=3, NV=4, VASC=5  ← SHIFTED!
- Model predicts "NV" (train idx=5), val interprets as "VASC" (val idx=5)
- Validation accuracy computed wrong, "best" model is actually collapsed

### The Fix Applied
Added **global** class mapping at module level (commit 78c4138):

```python
# FIXED: Global constants (new lines 28-29)
ISIC_CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
ISIC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(ISIC_CLASS_NAMES)}

class ISICDatasetColab(Dataset):
    ...
    def __init__(self, ...):
        ...
        # FIXED: Use GLOBAL mapping (new lines 79-80)
        self.class_names = ISIC_CLASS_NAMES
        self.class_to_idx = ISIC_CLASS_TO_IDX

        # Added validation (new lines 82-87)
        unknown_labels = set(df['label'].unique()) - set(ISIC_CLASS_NAMES)
        if unknown_labels:
            raise ValueError(f"Unknown labels: {unknown_labels}")
```

### Why TRADES Worked (71.9%)
- TRADES notebook (`Phase_5_Adversarial_Training.ipynb`) uses `src/datasets/isic.py`
- That module **correctly** builds class vocabulary from ALL data before filtering:
  ```python
  # src/datasets/isic.py line 112-114
  all_label_values = df[label_col].astype(str).tolist()
  label_names = sorted(set(all_label_values))  # From ALL data!
  ```

---

## 📋 NEXT STEPS FOR PHASE 9A EVALUATION

### 1. Retrain Tri-Objective Model (MANDATORY)

**In Google Colab:**

```python
# Pull the fix
!cd /content/drive/MyDrive/tri-objective-robust-xai-medimg && git pull

# Open Phase7_TriObjective_Training.ipynb
# Run ALL cells from top to bottom
# Training takes ~2-3 hours with class weights + consistent mapping

# Checkpoints save to: /content/drive/MyDrive/checkpoints/tri-objective/seed_42/
```

### 2. Verify New Checkpoint

After training completes, the new checkpoint should show:
- Validation accuracy: 40-60% (realistic for imbalanced data)
- NOT 78.7% (that was the bug)

### 3. Run Phase 9A Evaluation

**After retraining:**

```python
# Open PHASE_9A_TriObjective_Robust_Evaluation.ipynb
# Pull latest: !cd /content/drive/MyDrive/tri-objective-robust-xai-medimg && git pull
# Run evaluation cells

# Expected results (with proper training):
# - Baseline: ~40-50% clean accuracy (no robustness training)
# - TRADES: ~70% clean, ~40% robust (robustness focused)
# - Tri-objective: ~50-65% clean, ~30-50% robust (balanced approach)
```

---

## ✅ VERIFICATION CHECKLIST

Before running Phase 9A, verify:

1. [ ] Git pull completed in Colab
2. [ ] Phase 7 notebook shows "Using global class mapping: ['AKIEC', 'BCC', ...]"
3. [ ] Phase 7 training completed with realistic val_acc (40-60%, NOT 78%)
4. [ ] New checkpoint saved at `/content/drive/MyDrive/checkpoints/tri-objective/seed_42/best.pt`
5. [ ] Phase 9A sanity check shows non-zero batch accuracy

---

## 📊 Technical Details

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 7 ISICDatasetColab | ✅ FIXED | Uses global ISIC_CLASS_TO_IDX |
| Phase 9A ISIC2018Dataset | ✅ OK | Uses consistent CLASS_NAMES |
| src/datasets/isic.py | ✅ OK | Builds vocab from ALL data |
| Phase 5 TRADES | ✅ OK | Uses correct ISICDataset |

---

## 🔑 Key Lesson

When working with multiple data splits, **always define class vocabulary from the FULL dataset**
before filtering by split. Never derive class mappings from per-split unique values.

---

*Fix committed: 78c4138*
*Date: Session ongoing*
