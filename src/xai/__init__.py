"""Explainable AI (XAI) module for medical imaging.

Provides production-ready implementations of visual explanation methods:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Grad-CAM++: Improved Grad-CAM with weighted gradients
- AttentionRollout: For Vision Transformers
- Stability Metrics: SSIM, MS-SSIM, Spearman, L2, Cosine (Phase 6.2)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 25, 2025
"""

from src.xai.gradcam import (
    GradCAM,
    GradCAMConfig,
    GradCAMPlusPlus,
    create_gradcam,
    get_recommended_layers,
)
from src.xai.stability_metrics import (
    SSIM,
    MultiScaleSSIM,
    StabilityMetrics,
    StabilityMetricsConfig,
    cosine_similarity,
    create_stability_metrics,
    normalized_l2_distance,
    spearman_correlation,
)

try:
    from src.xai.attention_rollout import AttentionRollout, create_vit_explainer
except ImportError:
    AttentionRollout = None
    create_vit_explainer = None

__all__ = [
    # Grad-CAM
    "GradCAM",
    "GradCAMConfig",
    "GradCAMPlusPlus",
    "create_gradcam",
    "get_recommended_layers",
    # ViT
    "AttentionRollout",
    "create_vit_explainer",
    # Stability Metrics
    "StabilityMetrics",
    "StabilityMetricsConfig",
    "create_stability_metrics",
    "SSIM",
    "MultiScaleSSIM",
    "spearman_correlation",
    "normalized_l2_distance",
    "cosine_similarity",
]

__version__ = "6.2.0"
