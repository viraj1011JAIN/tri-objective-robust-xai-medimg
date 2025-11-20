from __future__ import annotations

"""
Standard Baseline Trainer for Tri-Objective Robust XAI Medical Imaging.

Implements standard supervised training with:
- Cross-entropy or focal loss
- Class-imbalance handling via class weights
- Epoch-level accuracy metrics
- Compatible with the generic BaseTrainer

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Project: Tri-Objective Robust XAI for Medical Imaging

Location: src/training/baseline_trainer.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainingConfig, TrainingMetrics

logger = logging.getLogger(__name__)


class BaselineTrainer(BaseTrainer):
    """
    Standard baseline trainer using cross-entropy loss.

    Features
    --------
    - Task loss only (no robustness or explainability terms).
    - Optional focal loss to handle class imbalance.
    - Class weight support.
    - Epoch-level accuracy tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        config: TrainingConfig,
        num_classes: int,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        """
        Initialise the baseline trainer.

        Parameters
        ----------
        model:
            Neural network model.
        train_loader:
            Training DataLoader yielding (images, labels).
        val_loader:
            Validation DataLoader yielding (images, labels).
        optimizer:
            Optimiser instance.
        config:
            Training configuration.
        num_classes:
            Number of output classes.
        scheduler:
            Optional learning rate scheduler.
        device:
            Device on which to run the model.
        checkpoint_dir:
            Directory in which to store checkpoints.
        class_weights:
            Optional per-class weights for cross-entropy / focal loss.
        use_focal_loss:
            If True, use FocalLoss instead of standard cross-entropy.
        focal_gamma:
            Gamma parameter for focal loss.
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            scheduler=scheduler,
            device=str(device) if device is not None else "cuda",
            checkpoint_dir=checkpoint_dir,
        )

        self.num_classes = int(num_classes)
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_gamma = float(focal_gamma)

        # Ensure weights live on the correct device
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        if self.use_focal_loss:
            self.criterion: nn.Module = FocalLoss(
                num_classes=self.num_classes,
                gamma=self.focal_gamma,
                weight=class_weights,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.criterion = self.criterion.to(self.device)

        # Metric buffers (per-epoch)
        self.train_predictions: List[torch.Tensor] = []
        self.train_targets: List[torch.Tensor] = []
        self.val_predictions: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Per-batch steps
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss and basic metrics for a training batch.

        Parameters
        ----------
        batch:
            (images, labels) tuple from the DataLoader.
        batch_idx:
            Index of the batch in the current epoch.

        Returns
        -------
        loss:
            Scalar loss tensor.
        metrics:
            Dictionary with per-batch metrics (currently accuracy only).
        """
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits = self.model(images)
        loss = self.criterion(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

            # Store predictions for epoch-level metrics
            self.train_predictions.append(preds.detach().cpu())
            self.train_targets.append(labels.detach().cpu())

        return loss, {"accuracy": accuracy}

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss and basic metrics for a validation batch.

        Parameters
        ----------
        batch:
            (images, labels) tuple from the DataLoader.
        batch_idx:
            Index of the batch in the current epoch.

        Returns
        -------
        loss:
            Scalar loss tensor.
        metrics:
            Dictionary with per-batch metrics (currently accuracy only).
        """
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits = self.model(images)
        loss = self.criterion(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

            self.val_predictions.append(preds.detach().cpu())
            self.val_targets.append(labels.detach().cpu())

        return loss, {"accuracy": accuracy}

    # ------------------------------------------------------------------
    # Epoch-level overrides for richer metrics
    # ------------------------------------------------------------------

    def train_epoch(self) -> TrainingMetrics:
        """Run one training epoch and compute epoch-level accuracy."""
        self.train_predictions.clear()
        self.train_targets.clear()

        metrics = super().train_epoch()

        if self.train_predictions:
            all_preds = torch.cat(self.train_predictions)
            all_targets = torch.cat(self.train_targets)
            metrics.accuracy = (all_preds == all_targets).float().mean().item()

        logger.info(
            "Epoch %d Train Accuracy: %.4f", self.current_epoch, metrics.accuracy
        )
        return metrics

    def validate(self) -> TrainingMetrics:
        """Run validation epoch and compute epoch-level accuracy."""
        self.val_predictions.clear()
        self.val_targets.clear()

        metrics = super().validate()

        if self.val_predictions:
            all_preds = torch.cat(self.val_predictions)
            all_targets = torch.cat(self.val_targets)
            metrics.accuracy = (all_preds == all_targets).float().mean().item()

        logger.info("Epoch %d Val Accuracy: %.4f", self.current_epoch, metrics.accuracy)
        return metrics


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.

    Reference
    ---------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """
        Parameters
        ----------
        num_classes:
            Number of classes.
        gamma:
            Focusing parameter; larger values emphasise hard examples.
        alpha:
            Optional scalar balance parameter.
        weight:
            Optional per-class weight tensor.
        reduction:
            One of {"none", "mean", "sum"}.
        """
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction '{reduction}'. "
                "Expected one of {'none', 'mean', 'sum'}."
            )

        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        logits:
            Logits of shape (batch_size, num_classes).
        targets:
            Ground-truth class indices of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if logits.ndim != 2:
            raise ValueError("logits must be 2D (batch_size, num_classes)")
        if targets.ndim != 1:
            raise ValueError("targets must be 1D (batch_size,)")

        if logits.size(0) != targets.size(0):
            raise ValueError("Batch size of logits and targets must match")

        if logits.size(1) != self.num_classes:
            raise ValueError(
                "The second dimension of logits must equal num_classes "
                f"({logits.size(1)} != {self.num_classes})"
            )

        # Compute standard cross-entropy loss per sample
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")

        # Compute probabilities of the true class
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
