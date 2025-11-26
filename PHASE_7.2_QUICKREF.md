# Phase 7.2 Quick Reference Guide

## 🎯 Implementation Complete: Tri-Objective Loss

**Status:** ✅ Production-ready
**Tests:** 38/38 passing (100%)
**Coverage:** 80% line, 70% branch
**Files:** 2,403 lines (1,647 implementation + 756 tests)

---

## Quick Import

```python
from src.losses import (
    TriObjectiveLoss,
    TriObjectiveConfig,
    create_tri_objective_loss,
    LossMetrics,
)
```

---

## 30-Second Usage

```python
# 1. Create loss function
loss_fn = create_tri_objective_loss(
    model=resnet50,
    num_classes=10,
    artifact_cavs=[blur_cav, noise_cav],  # List[Tensor(512)]
    medical_cavs=[skin_cav, lesion_cav],  # List[Tensor(512)]
    lambda_rob=0.3,      # 30% robustness
    lambda_expl=0.2,     # 20% explanation
)

# 2. Training loop
model.train()
loss, metrics = loss_fn(images, labels, return_metrics=True)
loss.backward()
optimizer.step()

# 3. Check metrics
print(f"Total: {metrics.loss_total:.4f}")
print(f"Task: {metrics.loss_task:.4f}")
print(f"Robust: {metrics.loss_rob:.4f}")
print(f"Explain: {metrics.loss_expl:.4f}")
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_rob` | 0.3 | Robustness weight (0.0-1.0) |
| `lambda_expl` | 0.2 | Explanation weight (0.0-1.0) |
| `temperature` | 1.5 | Temperature scaling (learnable) |
| `trades_beta` | 6.0 | TRADES emphasis (higher = more robust) |
| `pgd_epsilon` | 8/255 | L∞ perturbation bound |
| `pgd_num_steps` | 7 | PGD attack steps |
| `gamma` | 0.5 | Explanation composition (0.0=SSIM, 1.0=TCAV) |
| `use_ms_ssim` | False | Use MS-SSIM for Grad-CAM |

---

## Loss Components

### Mathematical Formula
```
L_total = L_CE(f(x), y) / T + λ_rob × β × KL(f(x) || f(x_adv))
          + λ_expl × [(1 - SSIM) + γ × L_TCAV]
```

### Component Breakdown
1. **Task Loss** (40-60%): Temperature-scaled cross-entropy
2. **Robustness Loss** (20-40%): PGD-based TRADES with KL divergence
3. **Explanation Loss** (10-30%): Grad-CAM SSIM + TCAV concept alignment

---

## Metrics Returned

```python
@dataclass
class LossMetrics:
    # Main losses
    loss_total: float       # Combined loss value
    loss_task: float        # Classification component
    loss_rob: float         # Robustness component (0 in eval)
    loss_expl: float        # Explanation component

    # Task details
    loss_ce: float          # Raw cross-entropy
    temperature: float      # Current temp value

    # Robustness details
    pgd_epsilon: float      # Perturbation bound
    pgd_num_steps: int      # PGD iterations
    kl_divergence: float    # KL(clean || adv)

    # Explanation details
    loss_gradcam: float     # Grad-CAM SSIM
    loss_tcav: float        # TCAV alignment

    # Weights
    lambda_rob: float       # Current robustness weight
    lambda_expl: float      # Current explanation weight

    # Optional
    grad_norm: float        # Gradient L2 norm
    forward_time_ms: float  # Computation time
```

---

## Custom Configuration

```python
from src.losses import TriObjectiveConfig

config = TriObjectiveConfig(
    lambda_rob=0.5,           # More robustness
    lambda_expl=0.1,          # Less explanation
    temperature=2.0,          # Higher temp
    trades_beta=8.0,          # Stronger adversarial
    pgd_epsilon=16.0 / 255.0, # Larger perturbations
    pgd_num_steps=10,         # More PGD steps
    gamma=0.7,                # More TCAV emphasis
    use_ms_ssim=True,         # Better SSIM
    grad_clip_value=1.0,      # Gradient clipping
)

loss_fn = TriObjectiveLoss(
    model=model,
    num_classes=num_classes,
    artifact_cavs=artifact_cavs,
    medical_cavs=medical_cavs,
    config=config,
)
```

---

## CAV Requirements

**Format:** `List[Tensor]` (NOT Dict!)

```python
# ✅ CORRECT
artifact_cavs = [
    torch.randn(512),  # CAV 1
    torch.randn(512),  # CAV 2
]

# ❌ INCORRECT
artifact_cavs = {
    "blur": torch.randn(512),   # Wrong format!
    "noise": torch.randn(512),
}
```

**Generation:**
```python
from src.xai import generate_concept_activation_vectors

artifact_cavs = generate_concept_activation_vectors(
    model=model,
    concept_images=artifact_images,  # Blur, noise samples
    target_layer="layer4",
)

medical_cavs = generate_concept_activation_vectors(
    model=model,
    concept_images=medical_images,  # Skin types, lesion types
    target_layer="layer4",
)
```

---

## Training Mode vs. Eval Mode

### Training Mode
```python
model.train()
loss_fn.train()

loss, metrics = loss_fn(images, labels, return_metrics=True)
# - Generates adversarial examples (PGD-7)
# - Computes all loss components
# - metrics.loss_rob > 0
```

### Eval Mode
```python
model.eval()
loss_fn.eval()

with torch.no_grad():
    loss, metrics = loss_fn(images, labels, return_metrics=True)
# - Skips adversarial generation
# - metrics.loss_rob = 0.0
# - Faster evaluation
```

---

## Verification Utilities

### Gradient Flow Check
```python
from src.losses import verify_gradient_flow

results = verify_gradient_flow(
    loss_fn,
    batch_size=4,
    image_size=224
)

assert results["forward_pass_successful"]
assert results["gradients_exist"]
assert results["loss_is_finite"]
```

### Performance Benchmark
```python
from src.losses import benchmark_computational_overhead

stats = benchmark_computational_overhead(
    loss_fn,
    batch_size=8,
    image_size=224,
    num_iterations=10
)

print(f"Forward: {stats['forward_mean_ms']:.2f} ms")
print(f"Backward: {stats['backward_mean_ms']:.2f} ms")
print(f"Total: {stats['total_mean_ms']:.2f} ms")
```

---

## Common Issues & Solutions

### Issue 1: CAV Format Error
```python
TypeError: cannot assign 'str' object to buffer 'artifact_cav_0'
```

**Solution:** Use `List[Tensor]` not `Dict[str, Tensor]`
```python
# Change from:
cavs = {"blur": tensor1, "noise": tensor2}
# To:
cavs = [tensor1, tensor2]
```

### Issue 2: Target Layer Error
```python
TypeError: got unexpected keyword argument 'target_layer'
```

**Solution:** Removed in Phase 7.2, use default layer4
```python
# Don't pass target_layer parameter
loss_fn = create_tri_objective_loss(
    model=model,
    # target_layer="layer4",  # ❌ Remove this
)
```

### Issue 3: High Memory Usage
```python
CUDA out of memory. Tried to allocate...
```

**Solution:** Reduce PGD steps or batch size
```python
config = TriObjectiveConfig(
    pgd_num_steps=3,      # Reduce from 7
    pgd_epsilon=4.0/255,  # Smaller perturbations
)
```

### Issue 4: Zero Robustness Loss
```python
metrics.loss_rob == 0.0  # In training mode
```

**Solution:** Check model is in train mode
```python
model.train()      # ✅ Must be training mode
loss_fn.train()    # ✅ Loss also needs training mode
```

---

## Testing

### Run All Tests
```bash
pytest tests/losses/test_tri_objective_loss.py -v
# 38 tests, ~5 seconds
```

### Run Specific Test Class
```bash
pytest tests/losses/test_tri_objective_loss.py::TestTriObjectiveLoss -v
```

### Run with Coverage
```bash
pytest tests/losses/test_tri_objective_loss.py \
  --cov=src.losses.tri_objective \
  --cov-report=html \
  --cov-report=term-missing
```

---

## Integration with Phase 7.1

```python
# Phase 7.1: ExplanationLoss (standalone)
from src.losses import ExplanationLoss, create_explanation_loss

expl_loss = create_explanation_loss(
    model=model,
    artifact_cavs=[cav1, cav2],
    medical_cavs=[cav3, cav4],
)

# Phase 7.2: TriObjectiveLoss (integrated)
from src.losses import TriObjectiveLoss, create_tri_objective_loss

tri_loss = create_tri_objective_loss(
    model=model,
    num_classes=10,
    artifact_cavs=[cav1, cav2],  # Same CAVs
    medical_cavs=[cav3, cav4],   # Same CAVs
    lambda_rob=0.3,              # New: robustness
    lambda_expl=0.2,             # Explanation weight
)

# TriObjectiveLoss internally uses ExplanationLoss
# No need to create ExplanationLoss separately
```

---

## Performance Tips

### 1. Adjust PGD Steps
```python
# Fast (training):    pgd_num_steps=3-5
# Balanced:           pgd_num_steps=7 (default)
# Strong (eval):      pgd_num_steps=10-20
```

### 2. Use Gradient Accumulation
```python
config = TriObjectiveConfig(
    gradient_accumulation_steps=4,  # Effective batch = 4x
)
```

### 3. Enable Mixed Precision
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss, metrics = loss_fn(images, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Disable Explanation in Eval
```python
# For faster evaluation
config = TriObjectiveConfig(
    lambda_expl=0.0,  # Skip explanation in eval
)
```

---

## Next Steps: Phase 7.3

### Update Trainer
1. Open `src/training/tri_objective_trainer.py`
2. Replace existing loss with new `TriObjectiveLoss`
3. Add metrics logging to MLflow
4. Configure checkpoint saving

### Example Integration
```python
# In TriObjectiveTrainer.__init__()
self.loss_fn = create_tri_objective_loss(
    model=self.model,
    num_classes=self.num_classes,
    artifact_cavs=self.artifact_cavs,
    medical_cavs=self.medical_cavs,
    lambda_rob=self.config.lambda_rob,
    lambda_expl=self.config.lambda_expl,
)

# In training_step()
loss, metrics = self.loss_fn(images, labels, return_metrics=True)

# Log to MLflow
mlflow.log_metrics({
    "loss/total": metrics.loss_total,
    "loss/task": metrics.loss_task,
    "loss/robust": metrics.loss_rob,
    "loss/explain": metrics.loss_expl,
}, step=self.global_step)
```

---

## Documentation Links

- **Implementation:** `src/losses/tri_objective.py`
- **Tests:** `tests/losses/test_tri_objective_loss.py`
- **Complete Report:** `PHASE_7.2_COMPLETION_REPORT.md`
- **Phase 7.1 (ExplanationLoss):** `PHASE_7.1_COMPLETION_REPORT.md`

---

## Quick Command Reference

```bash
# Run tests
pytest tests/losses/test_tri_objective_loss.py -v

# Generate coverage
pytest tests/losses/test_tri_objective_loss.py --cov=src.losses.tri_objective --cov-report=html

# Check imports
python -c "from src.losses import TriObjectiveLoss; print('✅ Import OK')"

# View coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux
```

---

**Phase 7.2 Status:** ✅ **COMPLETE**
**Quality:** Production-ready
**Next:** Phase 7.3 - Trainer Integration
