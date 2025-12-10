# 🔍 Interactive Demo Debug Guide

## ❓ Issue: "All amplified images looking same"

You reported that perturbations from all 3 models appear identical. This guide shows **how to verify** if models are properly loaded and differentiated.

---

## ✅ What to Check When Running Cell D1 (Setup)

After running Cell D1, you should see **MODEL VERIFICATION** output:

```
🔍 VERIFYING MODELS ARE DISTINCT:
--------------------------------------------------------------------------------
   Baseline parameter sum:     X.XXXXXXe+XX
   TRADES parameter sum:       Y.YYYYYYe+YY
   Tri-Objective parameter sum: Z.ZZZZZZe+ZZ

   ✅ Models are distinct (different trained weights)
```

### 🚨 RED FLAG #1: Identical Parameter Sums
If all three values are **EXACTLY THE SAME**, this means:
- All models loaded the same checkpoint
- Models are not differentiated
- Expected behavior: **Different numbers** (models trained differently)

**Fix**: Check checkpoint paths in Google Drive:
```bash
# In Colab, run this cell:
!ls -lh /content/drive/MyDrive/checkpoints/baseline/seed_42/
!ls -lh /content/drive/MyDrive/checkpoints/phase5_adversarial/
!ls -lh /content/drive/MyDrive/checkpoints/tri-objective/seed_42/
```

Look for:
- `best.pt` files exist in all 3 directories
- File sizes are different (models trained with different objectives)
- Timestamps show they were saved at different times

---

## ✅ What to Check in Cells D2-D5 (Testing)

### 📊 Perturbation Magnitude Output

After each cell (D2, D3, D4), you'll now see:

```
📏 Perturbation Magnitude:
   L2 norm:  X.XXXXXX
   L∞ norm:  Y.YYYYYY (max: 0.031373)
   L1 norm:  Z.ZZZZZZ
```

### 🚨 RED FLAG #2: Identical L2/L∞ Norms
If all three models show **EXACTLY the same L2 and L∞ values**, this means:
- Models are producing identical gradients
- Attack is using the same model for all 3 tests
- Expected behavior: **Different values** (different decision boundaries)

**Expected Pattern**:
- **Baseline**: Largest perturbations (easy to fool, weak gradients)
- **TRADES**: Medium perturbations (moderate robustness)
- **Tri-Objective**: Smallest perturbations (hardened gradients)

---

## ✅ What to Check in Cell D5 (Comparison)

### 📊 Perturbation Comparison Table

Cell D5 now shows:

```
📊 PERTURBATION NORMS (Different models should produce different values):
--------------------------------------------------------------------------------
   Baseline       : L2=X.XXXXXX  L∞=Y.YYYYYY  L1=Z.ZZZZZZ
   TRADES         : L2=A.AAAAAA  L∞=B.BBBBBB  L1=C.CCCCCC
   Tri-Objective  : L2=D.DDDDDD  L∞=E.EEEEEE  L1=F.FFFFFF
```

### 🚨 RED FLAG #3: All Values Identical
If you see:
```
   ⚠️  WARNING: All L2 norms are IDENTICAL!
   This suggests all models are using the same decision boundary.
```

This confirms the bug - models are not properly differentiated.

---

## 🔬 Root Cause Analysis

### Scenario 1: Checkpoint Files Are Identical
**Symptom**: Parameter sums are the same, L2 norms are the same

**Diagnosis**: All 3 paths point to the same checkpoint file

**Fix**:
1. Check checkpoint directory structure in Google Drive
2. Verify you have 3 **different** trained models from Phases 3, 5, 7
3. If missing, re-run training for each phase

### Scenario 2: Model Loading Bug
**Symptom**: Parameter sums different, but L2 norms identical

**Diagnosis**: Models loaded correctly, but attack using wrong model reference

**Fix**: Check PGD attack is using `self.model` correctly (already verified in code)

### Scenario 3: Visualization Scaling Issue
**Symptom**: L2 norms are different, but visuals look the same

**Diagnosis**: Perturbations ARE different, but amplification masks differences

**Fix**: Look at numerical values (L2/L∞) instead of visual comparison

---

## 📋 Expected Behavior (Real Models)

### ✅ Clean Predictions Should Differ

Run the same image through all 3 models in Cell D5. Expected:

| Model | Clean Accuracy | Robust Accuracy | Expected Confidence |
|-------|----------------|-----------------|---------------------|
| Baseline | 86.7% | 0.0% | **Highest** |
| TRADES | 60.5% | 33.9% | **Lowest** |
| Tri-Objective | 76.4% | 54.7% | **Medium** |

If all 3 models give **identical** clean predictions with **identical** confidence scores → models are the same.

### ✅ Adversarial Robustness Should Differ

Expected attack success rates (based on Phase 9A):

| Model | Attack Success Rate | Expected Behavior |
|-------|---------------------|-------------------|
| Baseline | 100% | **Always fails** (0% robust accuracy) |
| TRADES | 66.1% | **Sometimes survives** (33.9% robust) |
| Tri-Objective | 45.3% | **Often survives** (54.7% robust) |

If all 3 models **consistently fail or consistently survive** → models are the same.

### ✅ Perturbations Should Differ Numerically

Expected pattern:

```
Baseline        L2=0.150000  L∞=0.031373  (Weak gradients, easy to fool)
TRADES          L2=0.120000  L∞=0.031373  (Moderate gradients)
Tri-Objective   L2=0.090000  L∞=0.031373  (Strong gradients, hard to fool)
```

**Note**: L∞ may be capped at ε=8/255=0.031373 (PGD constraint), but L2 should differ.

If all models show **identical L2 values** → same decision boundary.

---

## 🛠️ Quick Diagnostic Checklist

Run these checks in order:

- [ ] **Cell D1 Output**: Do parameter sums differ? ✅ Different = Good | ❌ Same = Bug
- [ ] **Cell D2-D4 Output**: Do L2 norms differ across cells? ✅ Different = Good | ❌ Same = Bug
- [ ] **Cell D5 Output**: Does perturbation table show variation? ✅ Yes = Good | ❌ No = Bug
- [ ] **Cell D5 Clean Predictions**: Do confidence scores differ? ✅ Yes = Good | ❌ No = Bug
- [ ] **Cell D5 Attack Success**: Do some models survive while others fail? ✅ Yes = Good | ❌ No = Bug

**If ANY checkbox shows ❌ Bug**: Models are not properly loaded/differentiated.

---

## 📧 What to Report

Please share:

1. **Model Verification Output** from Cell D1:
   ```
   Baseline parameter sum:     ?.??????e+??
   TRADES parameter sum:       ?.??????e+??
   Tri-Objective parameter sum: ?.??????e+??
   ```

2. **Perturbation Norms** from Cell D5:
   ```
   Baseline       : L2=?.?????? L∞=?.??????
   TRADES         : L2=?.?????? L∞=?.??????
   Tri-Objective  : L2=?.?????? L∞=?.??????
   ```

3. **Clean Predictions** from Cell D5 (do they differ?)

This will help diagnose if:
- Models are properly loaded (different parameters)
- Models have different decision boundaries (different perturbations)
- Issue is visualization-only (numbers differ but visuals look same)

---

## 🎯 Expected Outcome

**If models are correctly loaded**:
- ✅ Parameter sums DIFFER
- ✅ L2 norms DIFFER
- ✅ Clean predictions DIFFER (confidence scores vary)
- ✅ Attack success rates DIFFER (Baseline always fails, Tri-Objective often survives)
- ✅ Visual perturbations may LOOK similar (all target same features) but **numbers prove they're different**

**Key Insight**: Perturbations may look visually similar because all models target the same diagnostic features (e.g., lesion borders, pigment networks). The **numerical differences** (L2/L∞ norms) prove models have different robustness levels.

---

## 🚀 Next Steps

1. **Re-run Cell D1** to see model verification output
2. **Check parameter sums** - are they different?
3. **Upload test image** to Cell D5 (side-by-side comparison)
4. **Check perturbation norms** - do L2 values differ?
5. **Report findings** so we can diagnose the root cause

If all checks pass but visuals still look identical → this is expected! Different models may produce visually similar perturbations (targeting same features) even though the **magnitude** differs (shown by L2 norms).

The key evidence is:
- **Different L2 norms** → different perturbation strength
- **Different attack success rates** → different robustness
- **Different clean confidences** → different model calibration
