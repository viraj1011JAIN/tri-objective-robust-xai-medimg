"""
Loss Functions for Tri-Objective Robust XAI Framework.

This package exposes publication-grade loss functions for medical image
classification and robust explainability:

- BaseLoss:
    Abstract superclass with:
        * input validation
        * reduction handling
        * statistics tracking (mean / min / max / calls)

- Task losses:
    * TaskLoss
        - Smart wrapper routing to:
            - CalibratedCrossEntropyLoss (multi-class)
            - MultiLabelBCELoss (multi-label CXR-style tasks)
            - FocalLoss (for severe class imbalance)
    * CalibratedCrossEntropyLoss
    * MultiLabelBCELoss
    * FocalLoss

- Calibration losses:
    * TemperatureScaling      (post-hoc Guo et al. calibration)
    * LabelSmoothingLoss      (training-time regularisation)
    * CalibrationLoss         (combined CE + temperature + smoothing)

These losses are used throughout the Tri-Objective Robust XAI training loop
(task accuracy, robustness, and explanation quality) and are designed to be:

- numerically stable on CPU/GPU
- fully differentiable with proper gradient flow
- compatible with mixed-precision training (autocast)
- rigorously validated via unit tests with 100% coverage

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

from .base_loss import BaseLoss
from .calibration_loss import CalibrationLoss, LabelSmoothingLoss, TemperatureScaling

# Phase 7.1: Explanation loss for tri-objective training
from .explanation_loss import (
    ExplanationLoss,
    ExplanationLossConfig,
    SSIMKernelType,
    SSIMStabilityLoss,
    TCavConceptLoss,
    create_explanation_loss,
)

# Phase 5.1: Robust losses for adversarial training
from .robust_loss import (
    AdversarialTrainingLoss,
    MARTLoss,
    TRADESLoss,
    adversarial_training_loss,
    mart_loss,
    trades_loss,
)
from .task_loss import (
    CalibratedCrossEntropyLoss,
    FocalLoss,
    MultiLabelBCELoss,
    TaskLoss,
)

# Phase 7.2: Tri-objective loss integrating task, robustness, and explanation
from .tri_objective import LossMetrics
from .tri_objective import TRADESLoss as TriObjectiveTRADESLoss
from .tri_objective import (
    TriObjectiveConfig,
    TriObjectiveLoss,
    benchmark_computational_overhead,
    create_tri_objective_loss,
    verify_gradient_flow,
)

__all__ = [
    # Base
    "BaseLoss",
    # Task Losses
    "TaskLoss",
    "CalibratedCrossEntropyLoss",
    "MultiLabelBCELoss",
    "FocalLoss",
    # Calibration
    "TemperatureScaling",
    "LabelSmoothingLoss",
    "CalibrationLoss",
    # Robust Losses (Phase 5.1)
    "TRADESLoss",
    "MARTLoss",
    "AdversarialTrainingLoss",
    "trades_loss",
    "mart_loss",
    "adversarial_training_loss",
    # Explanation Losses (Phase 7.1)
    "ExplanationLoss",
    "ExplanationLossConfig",
    "SSIMStabilityLoss",
    "SSIMKernelType",
    "TCavConceptLoss",
    "create_explanation_loss",
    # Tri-Objective Loss (Phase 7.2)
    "TriObjectiveLoss",
    "TriObjectiveConfig",
    "TriObjectiveTRADESLoss",
    "LossMetrics",
    "create_tri_objective_loss",
    "verify_gradient_flow",
    "benchmark_computational_overhead",
]

# Simple semantic version for the loss subsystem; useful in logs / reports.
__version__: str = "1.3.0"  # Updated for Phase 7.2
