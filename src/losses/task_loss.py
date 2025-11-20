"""
Task loss functions for Tri-Objective Robust XAI.

Implements primary task losses for medical image classification:
- TaskLoss: high-level wrapper choosing the right loss per task
- CalibratedCrossEntropyLoss: cross-entropy with learnable temperature + class weights
- MultiLabelBCELoss: multi-label BCE with logits, class weights, and pos_weight
- FocalLoss: focal loss for severe class imbalance (multi-class)

These losses are designed for:
- Medical image classification (dermatology, CXR, etc.)
- Strong validation and clear error messages (for testing and debugging)
- Compatibility with the BaseLoss statistics utilities
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_loss import BaseLoss

# ---------------------------------------------------------------------------
# High-level task loss wrapper
# ---------------------------------------------------------------------------


class TaskLoss(BaseLoss):
    """
    Generic task loss wrapper.

    Automatically selects appropriate loss based on task type:
    - Multi-class: CalibratedCrossEntropyLoss (with optional focal loss)
    - Multi-label: MultiLabelBCELoss

    Example
    -------
    >>> # Multi-class classification
    >>> loss_fn = TaskLoss(num_classes=7, task_type="multi_class")
    >>> logits = torch.randn(32, 7)
    >>> targets = torch.randint(0, 7, (32,))
    >>> loss = loss_fn(logits, targets)
    >>>
    >>> # Multi-label classification
    >>> loss_fn = TaskLoss(num_classes=14, task_type="multi_label")
    >>> logits = torch.randn(32, 14)
    >>> targets = torch.randint(0, 2, (32, 14)).float()
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        num_classes: int,
        task_type: str = "multi_class",
        class_weights: Optional[Tensor] = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction, name="TaskLoss")

        if task_type not in ("multi_class", "multi_label"):
            raise ValueError(
                f"Invalid task_type '{task_type}'. Must be 'multi_class' or 'multi_label'."
            )

        if task_type == "multi_label" and use_focal:
            # tests expect message to contain "only supported for multi_class"
            raise ValueError(
                "Focal loss is only supported for multi_class tasks; "
                f"got use_focal=True with task_type='{task_type}'."
            )

        self.num_classes = int(num_classes)
        self.task_type = task_type
        self.use_focal = bool(use_focal)

        # Register class weights as a buffer and expose as attribute
        if class_weights is not None:
            if class_weights.shape[0] != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

        # Underlying loss implementation
        if task_type == "multi_class":
            if use_focal:
                self.loss_fn = FocalLoss(
                    num_classes=num_classes,
                    class_weights=self.class_weights,
                    gamma=focal_gamma,
                    alpha=0.25,
                    reduction=reduction,
                )
            else:
                self.loss_fn = CalibratedCrossEntropyLoss(
                    num_classes=num_classes,
                    class_weights=self.class_weights,
                    reduction=reduction,
                )
        else:
            # Multi-label (e.g., CXR)
            self.loss_fn = MultiLabelBCELoss(
                num_classes=num_classes,
                class_weights=self.class_weights,
                reduction=reduction,
            )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        """
        Compute task loss.

        Multi-class:
            predictions: (B, C) logits
            targets:     (B,)   class indices

        Multi-label:
            predictions: (B, C) logits
            targets:     (B, C) binary labels
        """
        # Shape validations specific to task_type must run BEFORE BaseLoss._validate_inputs
        if self.task_type == "multi_class":
            # tests expect these error messages
            if predictions.dim() != 2:
                msg = (
                    f"Multi-class predictions must be 2D (B, C), "
                    f"got shape {tuple(predictions.shape)}"
                )
                raise ValueError(msg)
            if targets.dim() != 1:
                msg = (
                    f"Multi-class targets must be 1D (B,), "
                    f"got shape {tuple(targets.shape)}"
                )
                raise ValueError(msg)
        else:  # multi_label
            if predictions.dim() != 2:
                msg = (
                    f"Multi-label predictions must be 2D (B, C), "
                    f"got shape {tuple(predictions.shape)}"
                )
                raise ValueError(msg)
            if targets.dim() != 2:
                raise ValueError(
                    f"Multi-label targets must be 2D (B, C), got shape {tuple(targets.shape)}"
                )

        # Now run common validation (types, batch size, NaNs/Infs)
        self._validate_inputs(predictions, targets)

        # Class-dimension checks (tests look for "expected <num_classes>")
        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"Predictions have {predictions.shape[1]} classes, expected {self.num_classes}"
            )

        if self.task_type == "multi_label":
            if targets.shape[1] != self.num_classes:
                raise ValueError(
                    f"Targets have {targets.shape[1]} classes, expected {self.num_classes}"
                )

        loss = self.loss_fn(predictions, targets)
        self._update_statistics(loss)
        return loss


# ---------------------------------------------------------------------------
# Calibrated Cross-Entropy Loss
# ---------------------------------------------------------------------------


class CalibratedCrossEntropyLoss(BaseLoss):
    """
    Cross-entropy loss with learnable temperature scaling for calibration.

    - Learns a scalar temperature T > 0 in log-space for stability.
    - Supports per-class weights `class_weights`.

    Tests expect:
    - `temperature` to be an nn.Parameter with a gradient.
    - `get_temperature()` to return a positive float.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Tensor] = None,
        init_temperature: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction, name="CalibratedCrossEntropyLoss")

        if init_temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {init_temperature}")

        self.num_classes = int(num_classes)

        # Temperature stored in log-space but exposed as `temperature` Parameter
        # tests access `loss_fn.temperature.grad`
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32).log()
        )

        # Class weights
        if class_weights is not None:
            if class_weights.shape[0] != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        self._validate_inputs(predictions, targets)

        if predictions.dim() != 2:
            raise ValueError(
                f"Predictions must have shape (B, C), got {tuple(predictions.shape)}"
            )
        if targets.dim() != 1:
            raise ValueError(
                f"Targets must have shape (B,), got {tuple(targets.shape)}"
            )

        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"Predictions have {predictions.shape[1]} classes, expected {self.num_classes}"
            )

        # Convert log-temperature to positive temperature
        temp = self.temperature.exp()
        scaled_logits = predictions / temp

        # Compute cross entropy with scaled predictions
        loss = F.cross_entropy(
            scaled_logits, targets, weight=self.class_weights, reduction="mean"
        )

        self._update_statistics(loss)
        return loss

    def get_temperature(self) -> float:
        """Return the current temperature T as a positive float."""
        return float(self.temperature.exp().item())


# ---------------------------------------------------------------------------
# Multi-Label BCE Loss
# ---------------------------------------------------------------------------


class MultiLabelBCELoss(BaseLoss):
    """
    Multi-label binary cross entropy with logits.

    Used for multi-label classification tasks (e.g. chest X-ray with multiple diseases).

    Supports:
    - `class_weights`: per-class weights applied after BCE
    - `pos_weight`: positive-class weights passed to BCEWithLogitsLoss
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction, name="MultiLabelBCELoss")

        self.num_classes = int(num_classes)

        # Class weights
        if class_weights is not None:
            if class_weights.shape[0] != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

        # Positive-class weights
        if pos_weight is not None:
            if pos_weight.shape[0] != num_classes:
                raise ValueError(
                    f"pos_weight length {pos_weight.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.register_buffer("pos_weight", torch.ones(num_classes))

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        self._validate_inputs(predictions, targets)

        if predictions.dim() != 2:
            raise ValueError(
                f"Predictions must have shape (B, C), got {tuple(predictions.shape)}"
            )
        if targets.dim() != 2:
            raise ValueError(
                f"Targets must have shape (B, C), got {tuple(targets.shape)}"
            )

        if predictions.shape[1] != self.num_classes:
            # tests look for "Predictions have"
            raise ValueError(
                f"Predictions have {predictions.shape[1]} classes, expected {self.num_classes}"
            )

        if targets.shape[1] != self.num_classes:
            # tests look for "Targets have"
            raise ValueError(
                f"Targets have {targets.shape[1]} classes, expected {self.num_classes}"
            )

        # Per-sample, per-class BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )  # (B, C)

        # Apply class weights
        weighted_loss = bce_loss * self.class_weights.unsqueeze(0)  # (B, C)

        if self.reduction == "mean":
            loss = weighted_loss.mean()
        elif self.reduction == "sum":
            loss = weighted_loss.sum()
        else:  # "none"
            loss = weighted_loss

        self._update_statistics(loss)
        return loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(BaseLoss):
    """
    Focal Loss for addressing severe class imbalance (multi-class).

    Formula (for each sample):
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the true class.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    class_weights : Optional[Tensor]
        Per-class weights (C,).
    gamma : float
        Focusing parameter (γ >= 0). Higher γ focuses more on hard examples.
    alpha : float
        Balancing factor in [0, 1]. Balances positive/negative examples.
    reduction : {"none", "mean", "sum"}
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Tensor] = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction, name="FocalLoss")

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

        # Class weights
        if class_weights is not None:
            if class_weights.shape[0] != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        self._validate_inputs(predictions, targets)

        if predictions.dim() != 2:
            raise ValueError(
                f"Predictions must have shape (B, C), got {tuple(predictions.shape)}"
            )
        if targets.dim() != 1:
            raise ValueError(
                f"Targets must have shape (B,), got {tuple(targets.shape)}"
            )

        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"Predictions have {predictions.shape[1]} classes, expected {self.num_classes}"
            )

        # Probabilities and log-probabilities
        log_probs = F.log_softmax(predictions, dim=1)  # (B, C)
        probs = log_probs.exp()

        targets = targets.long()
        if targets.min() < 0 or targets.max() >= self.num_classes:
            raise ValueError("Target indices out of range for given logits")

        # Gather p_t and log p_t
        log_p_t = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)

        # Alpha factor (per-sample)
        alpha_vec = self.class_weights / self.class_weights.sum()
        alpha_t = alpha_vec[targets] * self.alpha + (1.0 - self.alpha)

        # Focal modulation (1 - p_t)^gamma
        focal_factor = (1.0 - p_t).clamp(min=0.0, max=1.0) ** self.gamma

        loss = -alpha_t * focal_factor * log_p_t  # (B,)

        if self.reduction == "none":
            final_loss = loss
        elif self.reduction == "sum":
            final_loss = loss.sum()
        else:  # "mean"
            final_loss = loss.mean()

        self._update_statistics(final_loss)
        return final_loss
