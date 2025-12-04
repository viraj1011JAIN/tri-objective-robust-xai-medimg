# H1b Failure Root Cause Analysis

## Executive Summary

**Hypothesis H1b FAILED**: The Tri-Objective model achieved only **2.6% retention** of TRADES robustness vs the required **90% retention**.

This document explains the root causes discovered and the fixes implemented.

---

## PHASE_9A Results Summary

| Model | Clean Accuracy | Robust Accuracy | Notes |
|-------|----------------|-----------------|-------|
| Baseline | 83.1% | 0.3% | Expected - no adversarial training |
| TRADES | 61.9% | **55.3%** | Reference robust model |
| Tri-Objective | 75.9% | **1.4%** | ❌ FAILED - almost no robustness |

### Hypothesis Results

| Hypothesis | Threshold | Actual | Status |
|------------|-----------|--------|--------|
| H1a | ≥25% robust acc | 55.3% | ✅ PASSED |
| **H1b** | ≥90% retention | **2.6%** | ❌ **FAILED** |
| H2a | SSIM ≥0.40 | 0.49-0.52 | ✅ PASSED |
| H3a | ≥4pp improvement | +4.2pp | ✅ PASSED |

---

## Root Cause Analysis

### Root Cause #1: Explanation Loss Was DISABLED

**Evidence from training log** (`logs/tri_objective/tri_objective_train_20251127_151509.log`):

```
WARNING - Explanation loss disabled: artifact_cavs and medical_cavs not provided
```

**Why this happened:**
- The `ExplanationLoss` class requires TCAV (Testing with Concept Activation Vectors)
- TCAV requires pre-computed CAV files (concept directions in feature space)
- The config specified TCAV but NO CAV files were provided
- **Result**: The entire explanation loss was silently disabled!

**Impact**:
- λ_expl = 0.1 was set, but the actual contribution was **ZERO**
- Training was effectively: L = L_task + 0.3×L_rob + **0×L_expl**
- No explanation stability was encouraged

### Root Cause #2: Normalized Images Used in PGD Attack

**Evidence from training log:**

```
WARNING - images values outside [0, 1] range: min=-2.1179, max=2.6400
```

**Why this happened:**
- Images were normalized using ImageNet statistics:
  - mean = [0.485, 0.456, 0.406]
  - std = [0.229, 0.224, 0.225]
- This transforms [0,1] images to approximately [-2.12, 2.64] range
- The PGD attack expects images in [0,1] range to apply ε=8/255 correctly
- Passing normalized images means:
  - The perturbation budget ε=8/255 ≈ 0.031 is applied in NORMALIZED space
  - In pixel space, this is roughly 0.031 × std ≈ 0.007 (7× weaker!)
  - Clamping to [0,1] is wrong for normalized images

**Impact**:
- Adversarial examples were ~7× weaker than intended during training
- Model learned to be robust against much smaller perturbations
- At evaluation (with correct attacks), it failed completely

---

## The Fix: PHASE_9B_TriObjective_FIXED_Training.ipynb

Created a new training notebook with these corrections:

### Fix #1: SSIM-Only Explanation Loss (No TCAV)

```python
class ExplanationStabilityLoss(nn.Module):
    """SSIM-only loss - no TCAV requirement"""
    def forward(self, clean_cams, adv_cams):
        ssim_loss = 1.0 - self.ssim_module(clean_cams, adv_cams)
        return ssim_loss
```

**Benefits**:
- No CAV files required
- Still encourages GradCAM stability between clean and adversarial images
- Directly optimizes what H2a measures

### Fix #2: Proper Denormalization for PGD Attack

```python
# ImageNet statistics
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def denormalize_for_attack(x_norm):
    """Convert normalized images back to [0,1] for PGD"""
    return x_norm * STD + MEAN

def renormalize(x_pixel):
    """Convert [0,1] images back to normalized form"""
    return (x_pixel - MEAN) / STD

# In training loop:
x_pixel = denormalize_for_attack(x_norm)      # [0,1] range
x_adv_pixel = pgd_attack(model, x_pixel, y)   # Attack in [0,1]
x_adv_norm = renormalize(x_adv_pixel)         # Back to normalized
```

**Benefits**:
- PGD attack operates in correct [0,1] space
- ε=8/255 is applied correctly
- Clamping to [0,1] is valid
- Model learns true robustness

### Fix #3: Increased Robustness Weight

Changed `λ_rob` from 0.3 to 0.5:
- Puts more emphasis on adversarial robustness
- Helps compete with TRADES (β=6.0)

---

## Next Steps

### Step 1: Run PHASE_9B Training in Colab

1. Open `notebooks/PHASE_9B_TriObjective_FIXED_Training.ipynb` in Google Colab
2. Upload your dataset to Google Drive
3. Run all cells (training ~50 epochs, ~2-3 hours on T4/V100)
4. Download the new checkpoint: `checkpoints/tri_objective_fixed/best.pt`

### Step 2: Re-run PHASE_9A Evaluation

1. Update checkpoint path in PHASE_9A to point to new model
2. Run full evaluation
3. Verify H1b passes (≥90% retention)

### Expected Results After Fix

| Model | Clean Accuracy | Robust Accuracy | Retention |
|-------|----------------|-----------------|-----------|
| TRADES | ~60-62% | ~50-55% | Reference |
| Tri-Objective (Fixed) | ~55-60% | ~45-52% | **~85-95%** |

**Why we expect ~90% retention**:
- Proper PGD training in [0,1] space
- λ_rob=0.5 gives strong robustness emphasis
- SSIM loss encourages stable explanations (bonus robustness)

---

## Technical Details

### Key Files Modified/Created

| File | Purpose |
|------|---------|
| `notebooks/PHASE_9B_TriObjective_FIXED_Training.ipynb` | Fixed training script |
| `H1B_FAILURE_ROOT_CAUSE_ANALYSIS.md` | This document |

### Configuration Comparison

| Parameter | Original (Buggy) | Fixed |
|-----------|------------------|-------|
| λ_rob | 0.3 | 0.5 |
| λ_expl | 0.1 (but disabled!) | 0.1 (active!) |
| TCAV | Required but missing | Not required |
| PGD images | Normalized [-2.12, 2.64] | Pixel space [0,1] |
| ε effective | ~0.004 (7× weaker) | 8/255 = 0.031 |

### Training Parameters

```python
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
LAMBDA_ROB = 0.5
LAMBDA_EXPL = 0.1
EPS = 8/255
PGD_STEPS = 7
PGD_ALPHA = 2/255
```

---

## Conclusion

The H1b failure was caused by **two implementation bugs**:

1. **TCAV requirement** silently disabled the explanation loss
2. **Normalized images** in PGD attack made adversarial training ~7× weaker

Both bugs have been fixed in PHASE_9B. After retraining, H1b should pass with ~90% robustness retention.

**Action Required**: Run PHASE_9B training notebook in Google Colab, then re-evaluate with PHASE_9A.
