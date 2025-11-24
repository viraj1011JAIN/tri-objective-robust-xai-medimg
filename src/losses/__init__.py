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
]

# Simple semantic version for the loss subsystem; useful in logs / reports.
__version__: str = "1.1.0"  # Updated for Phase 5.1
