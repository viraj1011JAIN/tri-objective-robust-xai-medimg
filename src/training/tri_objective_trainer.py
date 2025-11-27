"""
Tri-Objective Trainer for Robust XAI Medical Imaging.

Implements the complete training loop for the tri-objective optimization:
    L_total = L_task + λ_rob * L_rob + λ_expl * L_expl

This trainer orchestrates:
1. PGD adversarial example generation during training
2. Grad-CAM heatmap computation for explanation stability
3. Feature embedding extraction for TCAV
4. MLflow logging of all objectives and metrics
5. Checkpoint management and early stopping
6. Multi-seed reproducibility

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Project: Tri-Objective Robust XAI for Medical Imaging
Target: A1+ Grade | Publication-Ready (NeurIPS/MICCAI/TMI)
Deadline: November 28, 2025

Usage
-----
>>> from src.training.tri_objective_trainer import TriObjectiveTrainer, TriObjectiveConfig
>>> from src.losses.tri_objective import TriObjectiveLoss
>>> from src.attacks.pgd import PGD, PGDConfig
>>>
>>> config = TriObjectiveConfig(
...     max_epochs=100,
...     learning_rate=1e-4,
...     lambda_rob=0.3,
...     lambda_expl=0.2,
... )
>>>
>>> trainer = TriObjectiveTrainer(
...     model=model,
...     optimizer=optimizer,
...     train_loader=train_loader,
...     val_loader=val_loader,
...     config=config,
... )
>>>
>>> history = trainer.train()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..attacks.pgd import PGD, PGDConfig
from ..losses.tri_objective import TriObjectiveLoss
from .base_trainer import BaseTrainer, TrainingConfig, TrainingMetrics

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Logging disabled.")


@dataclass
class TriObjectiveConfig(TrainingConfig):
    """
    Configuration for tri-objective training.

    Extends TrainingConfig with tri-objective specific parameters.

    Attributes
    ----------
    lambda_rob : float
        Weight for robustness loss (TRADES)
    lambda_expl : float
        Weight for explanation loss (SSIM + TCAV)
    lambda_ssim : float
        Weight for SSIM within explanation loss
    lambda_tcav : float
        Weight for TCAV within explanation loss
    temperature : float
        Initial temperature for calibration
    trades_beta : float
        Beta parameter for TRADES
    pgd_epsilon : float
        PGD attack epsilon (L∞ norm)
    pgd_step_size : float
        PGD step size per iteration
    pgd_num_steps : int
        Number of PGD iterations
    pgd_random_start : bool
        Whether to use random initialization for PGD
    generate_heatmaps : bool
        Whether to generate Grad-CAM heatmaps (expensive)
    heatmap_layer : str
        Layer name for Grad-CAM (e.g., "layer4" for ResNet)
    extract_embeddings : bool
        Whether to extract embeddings for TCAV
    embedding_layer : str
        Layer name for embeddings (e.g., "avgpool" for ResNet)
    """

    # Tri-objective weights
    lambda_rob: float = 0.3
    lambda_expl: float = 0.2
    lambda_ssim: float = 0.7
    lambda_tcav: float = 0.3
    temperature: float = 1.5
    trades_beta: float = 6.0

    # PGD attack parameters
    pgd_epsilon: float = 8.0 / 255.0
    pgd_step_size: float = 2.0 / 255.0
    pgd_num_steps: int = 10
    pgd_random_start: bool = True

    # Explanation parameters
    generate_heatmaps: bool = False  # Disabled by default (Phase 4.3)
    heatmap_layer: str = "layer4"
    extract_embeddings: bool = True
    embedding_layer: str = "avgpool"


class TriObjectiveTrainer(BaseTrainer):
    """
    Trainer for tri-objective optimization.

    Implements the complete training loop with:
    - PGD adversarial training (TRADES)
    - Grad-CAM heatmap generation (optional)
    - Feature embedding extraction
    - MLflow logging of all objectives
    - Checkpoint management

    This trainer is the execution engine for:
    - Hypothesis H1: Robustness under TRADES
    - Hypothesis H2: Explanation stability (SSIM)
    - Hypothesis H3: Selective prediction (via embeddings)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TriObjectiveConfig,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = "cuda",
    ):
        """
        Initialize tri-objective trainer.

        Parameters
        ----------
        model : nn.Module
            Model to train (must have get_embeddings() method)
        optimizer : Optimizer
            PyTorch optimizer
        train_loader : DataLoader
            Training data loader
        config : TriObjectiveConfig
            Training configuration
        val_loader : DataLoader, optional
            Validation data loader
        scheduler : LRScheduler, optional
            Learning rate scheduler
        device : str
            Device to train on ("cuda" or "cpu")
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            val_loader=val_loader,
            scheduler=scheduler,
        )

        self.device = device
        self.model = model.to(device)

        # Determine task type from model
        self.num_classes = self._infer_num_classes()
        self.task_type = config.__dict__.get("task_type", "multi_class")

        # Create TriObjectiveConfig from trainer config
        from ..losses.tri_objective import TriObjectiveConfig as LossConfig

        # Calculate gamma from lambda_ssim and lambda_tcav
        gamma = (
            config.lambda_tcav / config.lambda_ssim if config.lambda_ssim > 0 else 0.5
        )

        loss_config = LossConfig(
            lambda_rob=config.lambda_rob,
            lambda_expl=config.lambda_expl,
            temperature=config.temperature,
            trades_beta=config.trades_beta,
            pgd_epsilon=config.pgd_epsilon,
            pgd_num_steps=config.pgd_num_steps,
            pgd_step_size=config.pgd_step_size,
            gamma=gamma,
        )

        # Initialize tri-objective loss with new API
        self.criterion = TriObjectiveLoss(
            model=self.model,
            num_classes=self.num_classes,
            task_type=self.task_type,
            artifact_cavs=None,  # Will be set later if needed
            medical_cavs=None,  # Will be set later if needed
            config=loss_config,
        ).to(device)

        # Initialize PGD attack
        pgd_config = PGDConfig(
            epsilon=config.pgd_epsilon,
            num_steps=config.pgd_num_steps,
            step_size=config.pgd_step_size,
            random_start=config.pgd_random_start,
            targeted=False,
            device=str(device),  # Pass device to PGD config
        )
        self.pgd_attack = PGD(pgd_config)

        # Grad-CAM setup (if enabled)
        self.generate_heatmaps = config.generate_heatmaps
        if self.generate_heatmaps:
            logger.info("Grad-CAM heatmap generation ENABLED (expensive!)")
        else:
            logger.info("Grad-CAM heatmap generation DISABLED (Phase 4.3 baseline)")

        # Embeddings setup
        self.extract_embeddings = config.extract_embeddings

        logger.info(
            f"TriObjectiveTrainer initialized: "
            f"λ_rob={config.lambda_rob}, λ_expl={config.lambda_expl}, "
            f"ε={config.pgd_epsilon:.4f}, steps={config.pgd_num_steps}"
        )

    def _infer_num_classes(self) -> int:
        """Infer number of classes from model output."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
            if isinstance(output, dict):
                output = output["logits"]
        return output.shape[1]

    def _generate_adversarial_examples(
        self,
        images: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Generate adversarial examples using PGD.

        Parameters
        ----------
        images : Tensor
            Clean images, shape (B, C, H, W)
        labels : Tensor
            Ground truth labels

        Returns
        -------
        images_adv : Tensor
            Adversarial images, shape (B, C, H, W)
        """
        self.model.eval()  # Set to eval for attack

        with torch.enable_grad():
            images_adv = self.pgd_attack(self.model, images, labels)

        self.model.train()  # Back to train mode
        return images_adv

    def _extract_embeddings(self, images: Tensor) -> Tensor:
        """
        Extract feature embeddings for TCAV.

        For ResNet, this is the output of avgpool layer.

        Parameters
        ----------
        images : Tensor
            Input images, shape (B, C, H, W)

        Returns
        -------
        embeddings : Tensor
            Feature embeddings, shape (B, D)
        """
        if hasattr(self.model, "get_embeddings"):
            embeddings = self.model.get_embeddings(images)
        else:
            # Fallback: use penultimate layer
            logger.warning("Model does not have get_embeddings(), using forward pass")
            with torch.no_grad():
                output = self.model(images)
                if isinstance(output, dict) and "embeddings" in output:
                    embeddings = output["embeddings"]
                else:
                    # Use feature map before classifier
                    embeddings = None

        return embeddings

    def _generate_heatmaps(
        self,
        images: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Generate Grad-CAM heatmaps (placeholder for Phase 4.3).

        Full implementation will be added in Phase 5.
        For now, return dummy heatmaps.

        Parameters
        ----------
        images : Tensor
            Input images, shape (B, C, H, W)
        labels : Tensor
            Ground truth labels

        Returns
        -------
        heatmaps : Tensor
            Heatmaps, shape (B, 1, H, W)
        """
        # Placeholder: return dummy heatmaps
        # Real Grad-CAM will be implemented in Phase 5.1
        batch_size = images.shape[0]
        h, w = images.shape[2:]
        heatmaps = torch.rand(batch_size, 1, h, w, device=images.device)
        return heatmaps

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Dict[str, float]:
        """
        Single training step.

        Implements:
        1. Generate adversarial examples via PGD
        2. Forward pass on clean and adversarial
        3. Extract embeddings (optional: heatmaps)
        4. Compute tri-objective loss
        5. Backward pass

        Parameters
        ----------
        batch : Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Dict]
            (images, labels) or (images, labels, meta)
        batch_idx : int
            Batch index

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics for this batch
        """
        # Handle both (images, labels) and (images, labels, meta)
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        batch_size = images.shape[0]

        # 1. Generate adversarial examples
        images_adv = self._generate_adversarial_examples(images, labels)

        # 2. Forward pass on clean images
        self.model.train()
        outputs_clean = self.model(images)
        if isinstance(outputs_clean, dict):
            logits_clean = outputs_clean["logits"]
        else:
            logits_clean = outputs_clean

        # 3. Forward pass on adversarial images
        outputs_adv = self.model(images_adv)
        if isinstance(outputs_adv, dict):
            logits_adv = outputs_adv["logits"]
        else:
            logits_adv = outputs_adv

        # 4. Extract embeddings (for TCAV)
        embeddings = None
        if self.extract_embeddings:
            embeddings = self._extract_embeddings(images)

        # 5. Generate heatmaps (optional, expensive)
        heatmap_clean = None
        heatmap_adv = None
        if self.generate_heatmaps:
            heatmap_clean = self._generate_heatmaps(images, labels)
            heatmap_adv = self._generate_heatmaps(images_adv, labels)

        # 6. Compute tri-objective loss (new API)
        # The new TriObjectiveLoss handles adversarial generation internally
        loss, loss_metrics = self.criterion(
            images=images,
            labels=labels,
            return_metrics=True,
        )

        # NOTE: BaseTrainer handles backward pass, optimizer step,
        # and gradient clipping. Do NOT do backward() here -
        # return loss tensor for BaseTrainer to handle

        # 7. Compute accuracy
        with torch.no_grad():
            model_output = self.model(images)
            # Handle dict vs tensor output
            if isinstance(model_output, dict):
                logits_clean = model_output.get("logits", model_output.get("out"))
            else:
                logits_clean = model_output

            if self.task_type == "multi_class":
                preds = torch.argmax(logits_clean, dim=1)
                correct = (preds == labels).sum().item()
                accuracy = correct / batch_size
            else:
                # Multi-label: threshold at 0.5
                probs = torch.sigmoid(logits_clean)
                preds = (probs > 0.5).float()
                correct = (preds == labels).sum().item()
                total = batch_size * self.num_classes
                accuracy = correct / total

        # 8. Return (loss_tensor, metrics) - BaseTrainer signature
        metrics = {
            "task_loss": loss_metrics.loss_task,
            "robustness_loss": loss_metrics.loss_rob,
            "explanation_loss": loss_metrics.loss_expl,
            "temperature": loss_metrics.temperature,
            "accuracy": accuracy,
        }

        return loss, metrics

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Dict[str, float]:
        """
        Single validation step.

        Parameters
        ----------
        batch : Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Dict]
            (images, labels) or (images, labels, meta)
        batch_idx : int
            Batch index

        Returns
        -------
        metrics : Dict[str, float]
            Validation metrics for this batch
        """
        # Handle both (images, labels) and (images, labels, meta)
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        batch_size = images.shape[0]

        with torch.no_grad():
            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            # Compute accuracy
            if self.task_type == "multi_class":
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                accuracy = correct / batch_size

                # Compute loss (CE only, no robustness/explanation)
                loss = F.cross_entropy(logits, labels)
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct = (preds == labels).sum().item()
                total = batch_size * self.num_classes
                accuracy = correct / total

                # Compute loss (BCE)
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        metrics = {
            "accuracy": accuracy,
        }

        return loss, metrics

    def on_train_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Callback after training epoch.

        Log metrics to MLflow if enabled.
        """
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(f"train_{key}", value, step=epoch)

        # Log to console
        logger.info(
            f"Epoch {epoch}/{self.config.max_epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Task: {metrics.get('task_loss', 0.0):.4f} | "
            f"Rob: {metrics.get('robustness_loss', 0.0):.4f} | "
            f"Expl: {metrics.get('explanation_loss', 0.0):.4f} | "
            f"Acc: {metrics['accuracy']:.4f}"
        )

    def on_val_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Callback after validation epoch.

        Log metrics to MLflow if enabled.
        """
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)

        logger.info(
            f"Val Epoch {epoch} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f}"
        )


def create_tri_objective_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    lambda_rob: float = 0.3,
    lambda_expl: float = 0.2,
    pgd_epsilon: float = 8.0 / 255.0,
    pgd_num_steps: int = 10,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints/tri_objective",
    use_mlflow: bool = True,
    **kwargs,
) -> TriObjectiveTrainer:
    """
    Factory function to create a tri-objective trainer.

    Simplifies trainer creation with sensible defaults.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay
    max_epochs : int
        Maximum epochs
    lambda_rob : float
        Robustness weight
    lambda_expl : float
        Explanation weight
    pgd_epsilon : float
        PGD epsilon
    pgd_num_steps : int
        PGD steps
    device : str
        Device
    checkpoint_dir : str
        Checkpoint directory
    use_mlflow : bool
        Enable MLflow logging
    **kwargs
        Additional config parameters

    Returns
    -------
    trainer : TriObjectiveTrainer
        Configured trainer
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create config
    config = TriObjectiveConfig(
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lambda_rob=lambda_rob,
        lambda_expl=lambda_expl,
        pgd_epsilon=pgd_epsilon,
        pgd_num_steps=pgd_num_steps,
        checkpoint_dir=checkpoint_dir,
        use_mlflow=use_mlflow,
        device=device,
        **kwargs,
    )

    # Create scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=learning_rate * 0.01,
    )

    # Create trainer
    trainer = TriObjectiveTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
    )

    return trainer
