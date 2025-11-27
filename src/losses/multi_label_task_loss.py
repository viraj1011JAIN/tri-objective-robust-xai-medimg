"""
Multi-Label Task Loss for Chest X-Ray Classification
====================================================

This module implements task losses specifically designed for multi-label
classification in chest X-ray imaging, where each image can have multiple
disease labels simultaneously.

Key Features:
- Binary Cross-Entropy with Logits (numerically stable)
- Automatic pos_weight computation from class imbalance
- Focal loss for extreme imbalance (rare diseases)
- Per-class thresholds optimization
- Temperature scaling for calibration

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 27, 2025
Target: A1+ Grade, Publication-Ready

Mathematical Formulation:
-------------------------
For multi-label with C classes and N samples:

Standard BCE:
    L = -1/(N*C) Σ Σ [y_{i,c} log(σ(z_{i,c})) + (1-y_{i,c}) log(1-σ(z_{i,c}))]

Weighted BCE (for imbalance):
    L = -1/(N*C) Σ Σ [w_c · y_{i,c} log(σ(z_{i,c})) + (1-y_{i,c}) log(1-σ(z_{i,c}))]

Focal Loss (for severe imbalance):
    L = -1/(N*C) Σ Σ α_c (1-p_{i,c})^γ log(p_{i,c})
    where p_{i,c} = σ(z_{i,c}) if y_{i,c}=1, else 1-σ(z_{i,c})
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelTaskLoss(nn.Module):
    """
    Multi-label task loss with support for:
    - Binary Cross-Entropy with automatic pos_weight
    - Focal loss for class imbalance
    - Temperature scaling for calibration

    This loss is designed for chest X-ray where:
    - Multiple diseases can be present simultaneously
    - Severe class imbalance (some diseases are rare)
    - Need for well-calibrated predictions
    """

    def __init__(
        self,
        num_classes: int,
        positive_rates: Optional[List[float]] = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Initialize multi-label task loss.

        Args:
            num_classes: Number of disease classes (14 for NIH ChestX-ray14)
            positive_rates: Positive rate for each class (for pos_weight)
                           If None, uses equal weights
            use_focal: Whether to use focal loss instead of BCE
            focal_gamma: Focusing parameter for focal loss (default: 2.0)
            focal_alpha: Weighting factor for focal loss (default: 0.25)
            temperature: Temperature for calibration (default: 1.0)
            label_smoothing: Label smoothing factor (typically 0 for multi-label)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Temperature for calibration (can be learned)
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Compute pos_weight from positive rates
        if positive_rates is not None:
            # pos_weight = (1 - positive_rate) / positive_rate
            # This balances the loss for imbalanced classes
            pos_weight = torch.tensor(
                [(1.0 - pr) / (pr + 1e-8) for pr in positive_rates], dtype=torch.float32
            )

            # Clamp to prevent extreme weights
            pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)

            self.register_buffer("pos_weight", pos_weight)
        else:
            self.register_buffer("pos_weight", torch.ones(num_classes))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_per_class: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-label task loss.

        Args:
            logits: Model output logits (B, C) where C is num_classes
            targets: Binary target labels (B, C) where each element is 0 or 1
            return_per_class: If True, returns loss per class

        Returns:
            loss: Scalar loss value (or per-class if return_per_class=True)
        """
        # Apply temperature scaling
        logits_scaled = logits / self.temperature

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets)

        if self.use_focal:
            loss = self._focal_loss(logits_scaled, targets)
        else:
            loss = self._bce_loss(logits_scaled, targets)

        if return_per_class:
            return loss  # (C,) per-class losses
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:  # 'none'
                return loss

    def _bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy loss.

        Args:
            logits: (B, C) logits
            targets: (B, C) binary targets

        Returns:
            loss: (C,) per-class losses
        """
        # Use BCEWithLogitsLoss for numerical stability
        # Manual implementation for per-class weighting

        # Sigmoid
        probs = torch.sigmoid(logits)

        # Per-class BCE
        # L_c = -[w_c * y_c * log(p_c) + (1-y_c) * log(1-p_c)]
        pos_loss = targets * torch.log(probs + 1e-8)
        neg_loss = (1 - targets) * torch.log(1 - probs + 1e-8)

        # Apply pos_weight to positive class
        pos_loss = pos_loss * self.pos_weight.unsqueeze(0)

        # Combine
        loss = -(pos_loss + neg_loss)  # (B, C)

        # Average over batch dimension
        loss = loss.mean(dim=0)  # (C,)

        return loss

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for multi-label classification.

        Focal loss: FL(p_t) = -α_t (1-p_t)^γ log(p_t)
        where p_t = p if y=1, else 1-p

        This down-weights easy examples and focuses on hard ones.

        Args:
            logits: (B, C) logits
            targets: (B, C) binary targets

        Returns:
            loss: (C,) per-class losses
        """
        # Sigmoid
        probs = torch.sigmoid(logits)

        # Compute p_t (probability of true class)
        # If y=1: p_t = p
        # If y=0: p_t = 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Cross-entropy term: -log(p_t)
        ce = -torch.log(p_t + 1e-8)

        # Focal loss: α * focal_weight * ce
        loss = self.focal_alpha * focal_weight * ce  # (B, C)

        # Apply pos_weight
        loss = loss * self.pos_weight.unsqueeze(0)

        # Average over batch dimension
        loss = loss.mean(dim=0)  # (C,)

        return loss

    def _apply_label_smoothing(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing: y' = y * (1 - ε) + ε / 2

        For multi-label, smoothing is applied to both 0 and 1 labels.

        Args:
            targets: (B, C) binary targets

        Returns:
            smoothed_targets: (B, C) smoothed targets
        """
        # For multi-label:
        # 0 → ε / 2
        # 1 → 1 - ε / 2
        epsilon = self.label_smoothing
        smoothed = targets * (1 - epsilon) + epsilon / 2
        return smoothed

    def get_pos_weight(self) -> torch.Tensor:
        """Get the current pos_weight values."""
        return self.pos_weight

    def get_temperature(self) -> float:
        """Get the current temperature value."""
        return self.temperature.item()

    def set_temperature(self, temperature: float) -> None:
        """Set a new temperature value."""
        with torch.no_grad():
            self.temperature.fill_(temperature)


class OptimalThresholdFinder:
    """
    Find optimal classification thresholds for multi-label classification.

    Instead of using 0.5 for all classes, this finds the optimal threshold
    per class that maximizes F1 score on the validation set.

    This is critical for imbalanced multi-label problems.
    """

    def __init__(self, num_classes: int):
        """
        Initialize threshold finder.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.thresholds = np.ones(num_classes) * 0.5  # Default: 0.5

    def find_optimal_thresholds(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        threshold_range: np.ndarray = np.linspace(0.1, 0.9, 81),
    ) -> np.ndarray:
        """
        Find optimal threshold per class that maximizes F1 score.

        Args:
            probabilities: (N, C) predicted probabilities
            targets: (N, C) binary targets
            threshold_range: Range of thresholds to try

        Returns:
            optimal_thresholds: (C,) optimal threshold per class
        """
        optimal_thresholds = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            class_probs = probabilities[:, c]
            class_targets = targets[:, c]

            best_f1 = 0.0
            best_threshold = 0.5

            for threshold in threshold_range:
                # Make predictions
                preds = (class_probs >= threshold).astype(int)

                # Compute F1 score
                tp = ((preds == 1) & (class_targets == 1)).sum()
                fp = ((preds == 1) & (class_targets == 0)).sum()
                fn = ((preds == 0) & (class_targets == 1)).sum()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds[c] = best_threshold

        self.thresholds = optimal_thresholds
        return optimal_thresholds

    def apply_thresholds(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply optimal thresholds to probabilities.

        Args:
            probabilities: (N, C) predicted probabilities

        Returns:
            predictions: (N, C) binary predictions
        """
        predictions = np.zeros_like(probabilities, dtype=int)

        for c in range(self.num_classes):
            predictions[:, c] = (probabilities[:, c] >= self.thresholds[c]).astype(int)

        return predictions

    def get_thresholds(self) -> np.ndarray:
        """Get current thresholds."""
        return self.thresholds


__all__ = [
    "MultiLabelTaskLoss",
    "OptimalThresholdFinder",
]
