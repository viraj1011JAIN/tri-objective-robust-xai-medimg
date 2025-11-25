"""Explainable AI (XAI) module for medical imaging.

Provides production-ready implementations of visual explanation methods:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Grad-CAM++: Improved Grad-CAM with weighted gradients
- AttentionRollout: For Vision Transformers

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

try:
    from src.xai.attention_rollout import AttentionRollout, create_vit_explainer
except ImportError:
    AttentionRollout = None
    create_vit_explainer = None

__all__ = [
    "GradCAM",
    "GradCAMConfig",
    "GradCAMPlusPlus",
    "create_gradcam",
    "get_recommended_layers",
    "AttentionRollout",
    "create_vit_explainer",
]

__version__ = "6.1.0"
