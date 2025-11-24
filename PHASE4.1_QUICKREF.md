# Phase 4.1 Quick Reference Card
# ================================

## 🚀 Quick Start

### Import Attacks
```python
from src.attacks import fgsm_attack, pgd_attack, cw_attack, autoattack
```

### Run Attacks
```python
# FGSM (fastest - single step)
x_adv = fgsm_attack(model, images, labels, epsilon=8/255)

# PGD (strong - 40 steps)
x_adv = pgd_attack(model, images, labels, epsilon=8/255, num_steps=40)

# C&W (strongest - optimization)
x_adv = cw_attack(model, images, labels, max_iterations=1000)

# AutoAttack (comprehensive - ensemble)
x_adv = autoattack(model, images, labels, epsilon=8/255, num_classes=10)
```

---

## 📊 Attack Comparison

| Attack | Type | Norm | Steps | Speed | Strength |
|--------|------|------|-------|-------|----------|
| FGSM | Gradient | L∞ | 1 | ⚡⚡⚡ | Baseline |
| PGD | Iterative | L∞ | 40 | ⚡⚡ | Strong |
| C&W | Optimization | L2 | 1000 | ⚡ | Strongest |
| AutoAttack | Ensemble | L∞/L2 | Varies | ⚡ | Very Strong |

---

## 🎯 Medical Imaging Recommendations

### Dermoscopy (RGB, 10 classes)
```python
# Subtle
x_adv = fgsm_attack(model, images, labels, epsilon=2/255)

# Moderate
x_adv = pgd_attack(model, images, labels, epsilon=4/255, num_steps=40)

# Strong
x_adv = autoattack(model, images, labels, epsilon=8/255, num_classes=10)
```

### Chest X-ray (Grayscale, 2 classes)
```python
# Subtle
x_adv = fgsm_attack(model, images, labels, epsilon=2/255)

# Moderate
x_adv = pgd_attack(model, images, labels, epsilon=4/255, num_steps=40)

# Strong
x_adv = autoattack(model, images, labels, epsilon=4/255, num_classes=2)
```

---

## 🔧 Configuration Files

Located in `configs/attacks/`:
- `fgsm_default.yaml` - FGSM settings
- `pgd_default.yaml` - PGD settings (steps, step_size, random_start)
- `cw_default.yaml` - C&W settings (confidence, iterations)
- `autoattack_standard.yaml` - AutoAttack settings (norm, version)

---

## 🧪 Class-Based API (Advanced)

```python
from src.attacks import FGSM, PGD, FGSMConfig, PGDConfig

# Configure attack
config = PGDConfig(
    epsilon=8/255,
    num_steps=40,
    step_size=2/255,
    random_start=True,
    verbose=True
)

# Create attack object
attack = PGD(config)

# Run attack and get detailed results
result = attack(model, images, labels)

# Access metrics
print(f"Success rate: {result.success_rate:.2%}")
print(f"Mean L2: {result.mean_l2:.4f}")
print(f"Mean L∞: {result.mean_linf:.4f}")

# Get statistics
stats = attack.get_statistics()
print(f"Total attacks: {stats['attack_count']}")
print(f"Average time: {stats['total_time'] / stats['attack_count']:.3f}s")
```

---

## 📝 With Normalization

```python
from torchvision import transforms

# ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Run attack
x_adv = fgsm_attack(
    model,
    images,
    labels,
    epsilon=8/255,
    normalize=normalize
)
```

---

## 🧪 Testing

```bash
# Run all attack tests
pytest tests/test_attacks.py -v

# Run specific attack
pytest tests/test_attacks.py::TestFGSM -v

# With coverage
pytest tests/test_attacks.py --cov=src/attacks
```

---

## 📊 Expected Performance

### CUDA (RTX 3050 Laptop, 4GB)
- FGSM: ~0.01s per batch (32 images)
- PGD: ~0.3s per batch (40 steps)
- C&W: ~10s per batch (1000 iterations)
- AutoAttack: ~0.5s per batch

### CPU (slower)
- FGSM: ~0.1s per batch
- PGD: ~2s per batch
- C&W: ~60s per batch
- AutoAttack: ~4s per batch

---

## 🔍 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
config = PGDConfig(epsilon=8/255, batch_size=16)
```

### Attack not working
```python
# Check model is in eval mode
model.eval()

# Check images in [0, 1] range
assert images.min() >= 0 and images.max() <= 1
```

### C&W too slow
```python
# Reduce iterations for testing
x_adv = cw_attack(
    model, images, labels,
    max_iterations=100,
    binary_search_steps=3
)
```

---

## 📚 Documentation

- **Comprehensive**: [PHASE4.1_COMPLETE.md](./PHASE4.1_COMPLETE.md)
- **Summary**: [PHASE4.1_SUMMARY.md](./PHASE4.1_SUMMARY.md)
- **Validation**: Run `python validate_phase4_1.py`

---

## ✅ Checklist for Using Attacks

- [ ] Model in eval mode: `model.eval()`
- [ ] Images in [0, 1]: `images = images / 255.0`
- [ ] Correct device: `images.to(device)`
- [ ] Labels correct shape: `[batch_size]`
- [ ] Optional normalization: Pass `normalize` function
- [ ] Check results: `assert x_adv.shape == images.shape`

---

**Status**: ✅ PRODUCTION READY
**Validation**: All 4 attacks tested on CUDA ✅
**Tests**: 60+ comprehensive tests ✅
**Documentation**: Complete ✅
