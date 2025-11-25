# Phase 6.1 Quick Reference

**Status**: ✅ Production Ready | **Version**: 6.1.0 | **Date**: Nov 25, 2025

---

## One-Line Summary

Production-grade Grad-CAM visual explanations for medical image CNNs and Vision Transformers.

---

## Quick Commands

### Python API

```python
# Basic Grad-CAM
from src.xai import GradCAM, GradCAMConfig

config = GradCAMConfig(target_layers=["layer4"], use_cuda=True)
gradcam = GradCAM(model, config)
heatmap = gradcam.generate_heatmap(image, class_idx=1)
overlay = gradcam.visualize(image, heatmap)

# Auto-detect layers
from src.xai import create_gradcam
gradcam = create_gradcam(model, method="gradcam")

# Grad-CAM++
gradcam_pp = create_gradcam(model, method="gradcam++")
```

### CLI

```bash
# Single image
python scripts/run_gradcam.py \
    --checkpoint checkpoints/best.pt \
    --image data/test/image1.jpg \
    --output results/xai/gradcam

# Batch
python scripts/run_gradcam.py \
    --checkpoint checkpoints/best.pt \
    --image-dir data/test/ \
    --output results/xai/batch

# Grad-CAM++
python scripts/run_gradcam.py \
    --checkpoint checkpoints/best.pt \
    --image data/test/image1.jpg \
    --method gradcam++ \
    --target-layer layer3
```

---

## Key Files

| File | Purpose | Size |
|------|---------|------|
| `src/xai/gradcam.py` | Grad-CAM implementation | 850 lines |
| `src/xai/attention_rollout.py` | ViT support | 350 lines |
| `tests/xai/test_gradcam.py` | Tests (>95% coverage) | 650 lines |
| `scripts/run_gradcam.py` | CLI wrapper | 350 lines |

---

## Core Classes

```python
# Configuration
GradCAMConfig(
    target_layers=["layer4"],
    use_cuda=True,
    batch_size=32,
    output_size=(224, 224),
    normalize_heatmap=True
)

# Grad-CAM
GradCAM(model, config)
    .generate_heatmap(image, class_idx)
    .generate_batch_heatmaps(batch)
    .get_multi_layer_heatmap(image, aggregation="mean")
    .visualize(image, heatmap, alpha=0.5)

# Grad-CAM++
GradCAMPlusPlus(model, config)

# ViT
AttentionRollout(vit_model, head_fusion="mean")
```

---

## Common Options

### GradCAMConfig

```python
target_layers=["layer4"]              # Layers to explain
use_cuda=True                         # GPU acceleration
batch_size=32                         # Batch processing
output_size=(224, 224)                # Heatmap size
normalize_heatmap=True                # Normalize [0,1]
interpolation_mode="bilinear"         # Resize method
```

### Aggregation Methods

- `"mean"` - Average all layers
- `"max"` - Take maximum activation
- `"weighted"` - Weight by layer depth

### Visualization

```python
alpha=0.5                    # Blending (0=image, 1=heatmap)
colormap=cv2.COLORMAP_JET    # Colormap
return_pil=False             # Return PIL Image
```

---

## Supported Architectures

| Architecture | Recommended Layer | Method |
|--------------|-------------------|--------|
| ResNet | `layer4` | Grad-CAM |
| DenseNet | `features.norm5` | Grad-CAM |
| EfficientNet | `features.8` | Grad-CAM |
| VGG | `features.30` | Grad-CAM |
| MobileNet | `features.18` | Grad-CAM |
| ViT | N/A | AttentionRollout |

---

## Testing

```bash
# Run all tests
pytest tests/xai/test_gradcam.py -v

# With coverage
pytest tests/xai/test_gradcam.py --cov=src.xai.gradcam

# Specific test class
pytest tests/xai/test_gradcam.py::TestGradCAM -v

# Expected: 41 tests passed, >95% coverage
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Target layers not found" | List layers with `model.named_modules()` |
| "No attention maps" (ViT) | Use `AttentionRollout` instead |
| Heatmap all zeros | Check model.eval(), use predicted class |
| CUDA out of memory | Reduce `batch_size` or use CPU |

---

## Integration

✅ Compatible with:
- `src.models.model_registry` - Model loading
- `src.training.base_trainer` - Trained models
- `src.training.adversarial_trainer` - Robust models
- `src.evaluation` - XAI evaluation metrics

---

## Performance

| Operation | GPU Time | CPU Time |
|-----------|----------|----------|
| Single heatmap (224×224) | ~15ms | ~80ms |
| Batch (16) | ~120ms | ~1.2s |
| Grad-CAM++ | ~25ms | ~120ms |

---

## Output Structure

```
results/xai/gradcam/
├── image1_heatmap.png      # Colored heatmap
├── image1_overlay.png      # Overlay visualization
└── image1_metadata.json    # Prediction info
```

**metadata.json**:
```json
{
  "image": "data/test/image1.jpg",
  "target_class": 1,
  "prediction": {
    "predicted_class": 1,
    "confidence": 0.9234
  }
}
```

---

## Advanced Usage

### Multi-Layer

```python
config = GradCAMConfig(target_layers=["layer3", "layer4"])
gradcam = GradCAM(model, config)
heatmap = gradcam.get_multi_layer_heatmap(image, aggregation="weighted")
```

### Auto-Detect

```python
from src.xai import get_recommended_layers

layers = get_recommended_layers(model)
config = GradCAMConfig(target_layers=layers)
```

### Batch Processing

```python
batch = torch.stack([img1, img2, img3, img4])
heatmaps = gradcam.generate_batch_heatmaps(batch)
```

---

## Next Steps

1. ✅ Phase 6.1 complete (Grad-CAM)
2. → Phase 6.2: Integrated Gradients
3. → Phase 6.3: SmoothGrad
4. → Phase 6.4: Concept-based (TCAV)
5. → Phase 6.5: XAI robustness evaluation

---

## Quick Links

- **Full Docs**: `PHASE_6_1_PRODUCTION.md`
- **Complete Summary**: `PHASE_6_1_COMPLETE.md`
- **Code**: `src/xai/gradcam.py`
- **Tests**: `tests/xai/test_gradcam.py`
- **CLI**: `scripts/run_gradcam.py`

---

**Status**: ✅ Ready to Use
**Updated**: November 25, 2025
