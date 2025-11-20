from __future__ import annotations

"""
Calibration Loss Functions for Tri-Objective Robust XAI.

Implements calibration techniques to improve model confidence:
- TemperatureScaling (post-hoc calibration)
- LabelSmoothingLoss (regularisation during training)
- CalibrationLoss (combined calibrated task loss)

These are critical for medical AI where confidence must be interpretable.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_loss import BaseLoss


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling Module for post-hoc calibration.

    Learns a single scalar temperature parameter to calibrate model outputs.
    This is the simplest and most effective calibration method.

    Usage:
        1. Train model normally.
        2. Create TemperatureScaling module.
        3. Fit temperature on validation set.
        4. Apply at inference: probs = module(logits)

    Reference:
        Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)

    Example:
        >>> temp_module = TemperatureScaling(init_temperature=1.5)
        >>> optimizer = torch.optim.LBFGS([temp_module.log_temperature], lr=0.01)
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     loss = temp_module.fit_step(logits, targets)
        ...     loss.backward()
        ...     return loss
        >>> optimizer.step(closure)
        >>> calibrated_probs = temp_module(logits)
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        """
        Initialize temperature scaling.

        Args:
            init_temperature: Initial temperature (typically 1.0–2.0).

        Raises:
            ValueError: If init_temperature <= 0.
        """
        super().__init__()

        if init_temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {init_temperature}")

        # Learnable temperature (stored as log for stability)
        self.log_temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32).log()
        )

    def forward(self, logits: Tensor) -> Tensor:  # type: ignore[override]
        """
        Apply temperature scaling to logits and return probabilities.

        Args:
            logits: Model logits (B, C).

        Returns:
            Temperature-scaled probabilities (B, C).
        """
        temperature = self.log_temperature.exp()
        scaled_logits = logits / temperature
        return F.softmax(scaled_logits, dim=1)

    def fit_step(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute NLL loss for fitting temperature.

        Use this in a separate optimisation loop after training:

        >>> optimizer = torch.optim.LBFGS([temp_module.log_temperature], lr=0.01)
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     loss = temp_module.fit_step(logits, targets)
        ...     loss.backward()
        ...     return loss
        >>> optimizer.step(closure)

        Args:
            logits: Model logits (B, C).
            targets: Ground truth class indices (B,).

        Returns:
            Negative log-likelihood loss.
        """
        temperature = self.log_temperature.exp()
        scaled_logits = logits / temperature
        return F.cross_entropy(scaled_logits, targets.long())

    def get_temperature(self) -> float:
        """Get current temperature value as a Python float."""
        return float(self.log_temperature.exp().item())


class LabelSmoothingLoss(BaseLoss):
    """
    Label Smoothing Loss for regularisation.

    Prevents the model from becoming over-confident by smoothing the target
    distribution. Instead of hard targets [0, 0, 1, 0], uses soft targets
    [ε/K, ε/K, 1-ε, ε/K] where K is number of classes and ε is smoothing factor.

    This improves calibration and generalisation.

    Reference:
        Szegedy et al. "Rethinking the Inception Architecture" (CVPR 2016)

    Example:
        >>> loss_fn = LabelSmoothingLoss(num_classes=7, smoothing=0.1)
        >>> logits = torch.randn(32, 7)
        >>> targets = torch.randint(0, 7, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
        class_weights: Optional[Tensor] = None,
    ) -> None:
        """
        Initialize label smoothing loss.

        Args:
            num_classes: Number of classes.
            smoothing: Smoothing factor ε ∈ [0, 1). Typically 0.1.
            reduction: Loss reduction method ('none', 'mean', 'sum').
            class_weights: Optional per-class weights (C,).

        Raises:
            ValueError: If smoothing not in [0, 1).
        """
        super().__init__(reduction=reduction, name="LabelSmoothingLoss")

        if not 0 <= smoothing < 1:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - self.smoothing

        # Register class weights as buffer (used only if provided)
        if class_weights is not None:
            if class_weights.shape[0] != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        """
        Compute label smoothing loss.

        Args:
            predictions: Model logits (B, C).
            targets: Ground truth class indices (B,).

        Returns:
            Loss value (scalar if reduction != 'none').
        """
        self._validate_inputs(predictions, targets)

        if predictions.dim() != 2:
            raise ValueError(
                f"Predictions must have shape (B, C), got {tuple(predictions.shape)}"
            )

        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"Predictions have {predictions.shape[1]} classes, expected {self.num_classes}"
            )

        if targets.dim() != 1:
            raise ValueError(
                f"Targets must have shape (B,), got {tuple(targets.shape)}"
            )

        # Compute log probabilities
        log_probs = F.log_softmax(predictions, dim=1)  # (B, C)

        batch_size = predictions.shape[0]
        num_classes = self.num_classes

        # True class gets (1 - ε), others get ε / (K - 1)
        smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.view(batch_size, 1), self.confidence)

        # Apply class weights if not all ones
        if self.class_weights is not None:
            smooth_targets = smooth_targets * self.class_weights.unsqueeze(0)

        # Compute: - sum(q * log p)
        loss_per_sample = -(smooth_targets * log_probs).sum(dim=1)  # (B,)

        # Reduce according to specified method
        if self.reduction == "mean":
            loss = loss_per_sample.mean()
        elif self.reduction == "sum":
            loss = loss_per_sample.sum()
        else:  # "none"
            loss = loss_per_sample

        self._update_statistics(loss)
        return loss


class CalibrationLoss(BaseLoss):
    """
    Combined calibration loss with multiple techniques.

    Combines:
    - Base task loss (cross-entropy with optional class weights)
    - Label smoothing (optional)
    - Temperature scaling (learnable, applied during training)

    Example:
        >>> loss_fn = CalibrationLoss(
        ...     num_classes=7,
        ...     use_label_smoothing=True,
        ...     smoothing=0.1,
        ...     init_temperature=1.5,
        ... )
        >>> logits = torch.randn(32, 7)
        >>> targets = torch.randint(0, 7, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Tensor] = None,
        use_label_smoothing: bool = True,
        smoothing: float = 0.1,
        init_temperature: float = 1.5,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize calibration loss.

        Args:
            num_classes: Number of classes.
            class_weights: Optional per-class weights (C,).
            use_label_smoothing: Whether to use label smoothing.
            smoothing: Label smoothing factor.
            init_temperature: Initial temperature for scaling.
            reduction: Loss reduction method.

        Raises:
            ValueError: If class_weights have wrong length.
        """
        super().__init__(reduction=reduction, name="CalibrationLoss")

        self.num_classes = int(num_classes)
        self.use_label_smoothing = bool(use_label_smoothing)

        if init_temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {init_temperature}")

        # Learnable temperature (log-space)
        self.log_temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32).log()
        )

        # Register class weights
        if class_weights is not None:
            if class_weights.shape[0] != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

        # Create label smoothing loss if enabled
        if self.use_label_smoothing:
            smoothing_loss = LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=smoothing,
                reduction=reduction,
                class_weights=self.class_weights,
            )
            self.label_smoothing: Optional[LabelSmoothingLoss] = smoothing_loss
        else:
            self.label_smoothing = None  # type: ignore[assignment]

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        """
        Compute calibrated loss.

        Args:
            predictions: Model logits (B, C).
            targets: Ground truth class indices (B,).

        Returns:
            Loss value (scalar if reduction != 'none').
        """
        self._validate_inputs(predictions, targets)

        if predictions.dim() != 2:
            raise ValueError(
                f"Predictions must have shape (B, C), got {tuple(predictions.shape)}"
            )

        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                f"Predictions have {predictions.shape[1]} classes, expected {self.num_classes}"
            )

        if targets.dim() != 1:
            raise ValueError(
                f"Targets must have shape (B,), got {tuple(targets.shape)}"
            )

        # Apply temperature scaling
        temperature = self.log_temperature.exp()
        scaled_logits = predictions / temperature

        # Compute loss (with or without label smoothing)
        if self.use_label_smoothing and self.label_smoothing is not None:
            loss = self.label_smoothing(scaled_logits, targets)
        else:
            loss = F.cross_entropy(
                scaled_logits,
                targets.long(),
                weight=self.class_weights,
                reduction=self.reduction,
            )

        self._update_statistics(loss)
        return loss

    def get_temperature(self) -> float:
        """Get current temperature value as a Python float."""
        return float(self.log_temperature.exp().item())


class SomeCalibrationClass:
    def __init__(self, config):
        # Fix line ~307 - allow None
        self.label_smoothing_loss: Optional[LabelSmoothingLoss] = None

        if config.label_smoothing > 0:
            self.label_smoothing_loss = LabelSmoothingLoss(
                smoothing=config.label_smoothing,
                num_classes=config.num_classes,
            )
