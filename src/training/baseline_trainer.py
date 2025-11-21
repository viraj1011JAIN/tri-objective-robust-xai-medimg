from __future__ import annotations

"""
Standard Baseline Trainer for Tri-Objective Robust XAI Medical Imaging.

Implements standard supervised training with:
- TaskLoss (production-grade CE/BCE/Focal from Phase 3.2)
- CalibrationLoss (temperature scaling + label smoothing)
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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..losses import TaskLoss, CalibrationLoss
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
        task_type: str = "multi_class",
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_calibration: bool = False,
        init_temperature: float = 1.5,
        label_smoothing: float = 0.0,
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
            Optional per-class weights for TaskLoss.
        task_type:
            Task type: "multi_class" or "multi_label" (for CXR).
        use_focal_loss:
            If True, use FocalLoss (via TaskLoss).
        focal_gamma:
            Gamma parameter for focal loss (default 2.0).
        use_calibration:
            If True, use CalibrationLoss (temperature + smoothing).
        init_temperature:
            Initial temperature for calibration (default 1.5).
        label_smoothing:
            Label smoothing factor (0.0 = no smoothing, 0.1 = 10%).
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
        self.task_type = task_type
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_gamma = float(focal_gamma)
        self.use_calibration = bool(use_calibration)

        # Ensure weights live on the correct device
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        # Use production-grade loss functions from Phase 3.2
        if self.use_calibration:
            # CalibrationLoss: Temperature scaling + label smoothing
            self.criterion: nn.Module = CalibrationLoss(
                num_classes=self.num_classes,
                class_weights=class_weights,
                use_label_smoothing=(label_smoothing > 0.0),
                smoothing=label_smoothing,
                init_temperature=init_temperature,
                reduction="mean",
            )
            logger.info(
                "Using CalibrationLoss (temp=%.2f, smoothing=%.2f)",
                init_temperature,
                label_smoothing,
            )
        else:
            # TaskLoss: Auto-selects CE/BCE/Focal based on task_type
            self.criterion = TaskLoss(
                num_classes=self.num_classes,
                task_type=task_type,
                class_weights=class_weights,
                use_focal=use_focal_loss,
                focal_gamma=focal_gamma,
                reduction="mean",
            )
            loss_type = "FocalLoss" if use_focal_loss else "CE/BCE"
            logger.info(
                "Using TaskLoss (%s, task_type=%s)", loss_type, task_type
            )

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
            "Epoch %d Train Accuracy: %.4f",
            self.current_epoch,
            metrics.accuracy,
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

        logger.info(
            "Epoch %d Val Accuracy: %.4f",
            self.current_epoch,
            metrics.accuracy,
        )
        return metrics

    def get_temperature(self) -> Optional[float]:
        """
        Get current temperature from calibration loss.

        Returns
        -------
        float or None:
            Current temperature value, or None if not using calibration.
        """
        if hasattr(self.criterion, "get_temperature"):
            return self.criterion.get_temperature()
        return None

    def get_loss_statistics(self) -> Dict[str, float]:
        """
        Get loss statistics from Phase 3.2 loss functions.

        Returns
        -------
        dict:
            Loss statistics (mean, min, max, num_calls).
        """
        if hasattr(self.criterion, "get_statistics"):
            return self.criterion.get_statistics()
        return {}
