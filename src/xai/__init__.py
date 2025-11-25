"""Explainable AI (XAI) module for medical imaging.

Provides production-ready implementations of visual explanation methods:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Grad-CAM++: Improved Grad-CAM with weighted gradients
- AttentionRollout: For Vision Transformers
- Stability Metrics: SSIM, MS-SSIM, Spearman, L2, Cosine (Phase 6.2)
- Faithfulness Metrics: Deletion/Insertion Curves, Pointing Game (Phase 6.3)
- Baseline Quality: Evaluation framework for baseline explanations (Phase 6.4)
- Concept Bank: TCAV concept extraction for semantic alignment (Phase 6.5)
- TCAV: Testing with Concept Activation Vectors (Phase 6.6)
- Baseline TCAV Evaluation: Baseline concept reliance analysis (Phase 6.7)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 25, 2025
"""

from src.xai.baseline_explanation_quality import (
    BaselineExplanationQuality,
    BaselineQualityConfig,
    create_baseline_quality_evaluator,
)
from src.xai.baseline_tcav_evaluation import (
    BaselineTCAVConfig,
    BaselineTCAVEvaluator,
    ConceptCategory,
    create_baseline_tcav_evaluator,
)
from src.xai.concept_bank import (
    CHEST_XRAY_ARTIFACT_CONCEPTS,
    CHEST_XRAY_MEDICAL_CONCEPTS,
    DERMOSCOPY_ARTIFACT_CONCEPTS,
    DERMOSCOPY_MEDICAL_CONCEPTS,
    ConceptBankConfig,
    ConceptBankCreator,
    create_concept_bank_creator,
)
from src.xai.faithfulness import (
    DeletionMetric,
    FaithfulnessConfig,
    FaithfulnessMetrics,
    InsertionMetric,
    PointingGame,
    create_faithfulness_metrics,
)
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
from src.xai.tcav import (
    TCAV,
    ActivationExtractor,
    CAVTrainer,
    ConceptDataset,
    TCAVConfig,
    create_tcav,
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
    # Faithfulness Metrics
    "FaithfulnessMetrics",
    "FaithfulnessConfig",
    "DeletionMetric",
    "InsertionMetric",
    "PointingGame",
    "create_faithfulness_metrics",
    # Baseline Quality
    "BaselineExplanationQuality",
    "BaselineQualityConfig",
    "create_baseline_quality_evaluator",
    # Concept Bank
    "ConceptBankCreator",
    "ConceptBankConfig",
    "create_concept_bank_creator",
    "DERMOSCOPY_MEDICAL_CONCEPTS",
    "DERMOSCOPY_ARTIFACT_CONCEPTS",
    "CHEST_XRAY_MEDICAL_CONCEPTS",
    "CHEST_XRAY_ARTIFACT_CONCEPTS",
    # TCAV
    "TCAV",
    "TCAVConfig",
    "ActivationExtractor",
    "CAVTrainer",
    "ConceptDataset",
    "create_tcav",
    # Baseline TCAV Evaluation
    "BaselineTCAVEvaluator",
    "BaselineTCAVConfig",
    "ConceptCategory",
    "create_baseline_tcav_evaluator",
]

__version__ = "6.7.0"
