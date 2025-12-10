#!/usr/bin/env python3
"""
Tri-Objective Training Script for Medical Imaging - Phase 7.5
==============================================================

Production-grade training script implementing the complete tri-objective
optimization framework for robust and explainable medical image classification.

This script integrates:
    1. Task Loss (L_task): Calibrated cross-entropy with temperature scaling
    2. Robustness Loss (L_rob): TRADES adversarial robustness via KL divergence
    3. Explanation Loss (L_expl): Stability (SSIM) + semantic alignment (TCAV)

Mathematical Formulation:
    L_total = L_task + λ_rob * L_rob + λ_expl * L_expl

Key Features:
    - Multi-objective optimization with gradient balancing
    - Mixed precision training (AMP) for GPU efficiency
    - MLflow experiment tracking with comprehensive metrics
    - Adversarial evaluation during validation (PGD-20)
    - Explanation quality monitoring (SSIM, TCAV)
    - Calibration metrics (ECE, MCE, Brier score)
    - Early stopping with patience
    - Multi-seed experimental support
    - Production-level error handling and logging

Usage:
    # Single seed training
    python scripts/training/train_tri_objective.py \\
        --config configs/experiments/tri_objective.yaml \\
        --seed 42

    # Multi-seed training (use bash script)
    bash scripts/training/run_tri_objective_multiseed.sh

Expected Performance (ISIC 2018, ResNet-50):
    - Clean Accuracy: ≥ 82%
    - Robust Accuracy (PGD-20): ≥ 65%
    - SSIM Stability: ≥ 0.70
    - TCAV Medical Alignment: ≥ 0.60
    - Calibration ECE: ≤ 0.10

Training Time (RTX 3050, 4GB):
    - Per epoch: ~25-30 minutes
    - Total (60 epochs): ~25-30 hours
    - Expected convergence: ~35-40 epochs

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 27, 2025
Version: 1.0.0 (Phase 7.5 - Production Release)
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Disable albumentations version check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
from src.attacks import PGD
from src.datasets.isic import ISICDataset
from src.evaluation.calibration import evaluate_calibration
from src.evaluation.metrics import compute_classification_metrics
from src.losses.tri_objective import TriObjectiveLoss
from src.models import build_model
from src.utils.config import ExperimentConfig, load_experiment_config
from src.utils.mlflow_utils import init_mlflow
from src.utils.reproducibility import get_reproducibility_state, set_global_seed
from src.xai.gradcam import GradCAM
from src.xai.stability_metrics import SSIM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train_tri_objective.log"),
    ],
)
logger = logging.getLogger(__name__)


class TriObjectiveTrainer:
    """
    Production-grade trainer for tri-objective optimization.

    This trainer implements the complete tri-objective framework with:
        - Multi-objective loss computation and gradient balancing
        - Mixed precision training with gradient scaling
        - Comprehensive metric tracking and logging
        - Adversarial evaluation during validation
        - Explanation quality monitoring
        - Early stopping and checkpoint management
        - MLflow experiment tracking

    Args:
        config: Experiment configuration loaded from YAML.
        model: Neural network model (e.g., ResNet-50).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader (optional).
        optimizer: Optimizer (e.g., AdamW).
        scheduler: Learning rate scheduler (e.g., CosineAnnealingLR).
        device: Device to use for training ('cuda' or 'cpu').
        mixed_precision: Whether to use mixed precision training.
        checkpoint_dir: Directory to save checkpoints.
        log_interval: Logging interval (number of batches).

    Attributes:
        config: Experiment configuration.
        model: Neural network model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device for training.
        mixed_precision: Whether to use mixed precision.
        checkpoint_dir: Directory for checkpoints.
        log_interval: Logging interval.
        tri_objective_loss: Tri-objective loss module.
        scaler: Gradient scaler for mixed precision.
        best_val_loss: Best validation loss seen so far.
        patience_counter: Counter for early stopping patience.
        current_epoch: Current training epoch.
        global_step: Global training step counter.
        gradcam: GradCAM explainer for validation.
        pgd_val: PGD attack for validation.

    Example:
        >>> config = load_experiment_config(
        ...     "configs/experiments/tri_objective.yaml"
        ... )
        >>> model = build_model(config.model)
        >>> trainer = TriObjectiveTrainer(
        ...     config=config,
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     device="cuda",
        ... )
        >>> trainer.train(num_epochs=60)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str = "cuda",
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints/tri_objective",
        log_interval: int = 10,
        cavs_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the tri-objective trainer."""
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.cavs_data = cavs_data

        # Initialize tri-objective loss
        self.tri_objective_loss = self._create_tri_objective_loss()
        self.tri_objective_loss.to(device)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=mixed_precision)

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0
        self.global_step = 0

        # Validation tools
        self.gradcam = GradCAM(
            model=model,
            target_layer=self._get_target_layer(),
        )
        self.ssim = SSIM(reduction="mean")  # For explanation stability
        self.pgd_val = PGD(
            epsilon=config.validation.adversarial_eval.attack.epsilon,
            num_steps=config.validation.adversarial_eval.attack.num_steps,
            step_size=config.validation.adversarial_eval.attack.step_size,
            random_start=True,
            norm="linf",
        )

        logger.info("TriObjectiveTrainer initialized successfully")
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {mixed_precision}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")

    def _create_tri_objective_loss(self) -> TriObjectiveLoss:
        """Create tri-objective loss module from config.

        Returns:
            Configured tri-objective loss module.
        """
        from src.losses.tri_objective import TriObjectiveConfig

        loss_config = TriObjectiveConfig(
            lambda_rob=self.config.loss.lambda_rob,
            lambda_expl=self.config.loss.lambda_expl,
            beta=self.config.loss.robustness_loss.beta,
            epsilon=self.config.loss.robustness_loss.attack.epsilon,
            pgd_steps=self.config.loss.robustness_loss.attack.num_steps,
            pgd_step_size=self.config.loss.robustness_loss.attack.step_size,
            temperature=self.config.loss.task_loss.temperature,
            learnable_temperature=(self.config.loss.task_loss.learnable_temperature),
            gamma=self.config.loss.explanation_loss.gamma,
            label_smoothing=self.config.loss.task_loss.label_smoothing,
        )

        # Extract CAVs if provided
        artifact_cavs = None
        medical_cavs = None

        if self.cavs_data is not None:
            logger.info("Loading CAVs from checkpoint...")
            cavs = self.cavs_data.get("cavs", {})

            # Separate artifact and medical CAVs (convert numpy to torch)
            artifact_cavs = [
                torch.from_numpy(cav_data["vector"]).float()
                for cav_name, cav_data in cavs.items()
                if cav_data.get("type") == "artifact"
            ]
            medical_cavs = [
                torch.from_numpy(cav_data["vector"]).float()
                for cav_name, cav_data in cavs.items()
                if cav_data.get("type") == "medical"
            ]

            logger.info(f"✓ Loaded {len(artifact_cavs)} artifact CAVs")
            logger.info(f"✓ Loaded {len(medical_cavs)} medical CAVs")

            # Log CAV details
            for cav_name, cav_data in cavs.items():
                acc = cav_data.get("accuracy", 0.0)
                cav_type = cav_data.get("type", "unknown")
                logger.info(f"  - {cav_name} ({cav_type}): accuracy={acc:.3f}")
        else:
            logger.warning(
                "⚠️  No CAVs provided! TCAV component of explanation loss will be disabled. "
                "To enable TCAV (H2.2-H2.4), train CAVs first:\n"
                "    python scripts/training/train_cavs_for_training.py \\\n"
                "        --data_dir data/processed/isic2018 \\\n"
                "        --model_checkpoint checkpoints/baseline/seed_42/best.pt \\\n"
                "        --output_dir checkpoints/cavs\n"
                "Then pass --cavs-checkpoint checkpoints/cavs/trained_cavs.pt"
            )

        return TriObjectiveLoss(
            config=loss_config,
            model=self.model,
            num_classes=self.config.model.num_classes,
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

    def _get_target_layer(self) -> nn.Module:
        """Get the target layer for GradCAM.

        Returns:
            Target layer module (e.g., model.layer4 for ResNet).
        """
        if hasattr(self.model, "layer4"):
            return self.model.layer4
        elif hasattr(self.model, "features"):
            return self.model.features[-1]
        else:
            logger.warning("Could not identify target layer, using last conv layer")
            # Find last convolutional layer
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
            raise ValueError("No convolutional layer found for GradCAM")

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train.

        Returns:
            Dictionary containing training history and final metrics.
        """
        logger.info(f"Starting tri-objective training for {num_epochs} epochs")
        logger.info(f"Total training steps: {len(self.train_loader) * num_epochs}")

        training_start_time = time.time()
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_robust_acc": [],
            "val_ssim": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self._train_epoch()
            history["train_loss"].append(train_metrics["loss_total"])
            history["train_acc"].append(train_metrics["accuracy"])

            # Validation phase
            val_metrics = self._validate_epoch()
            history["val_loss"].append(val_metrics["loss_total"])
            history["val_acc"].append(val_metrics["accuracy_clean"])
            history["val_robust_acc"].append(val_metrics["accuracy_robust"])
            history["val_ssim"].append(val_metrics["ssim_mean"])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary")
            logger.info(f"{'='*80}")
            logger.info(f"Train Loss: {train_metrics['loss_total']:.4f}")
            logger.info(f"Train Acc:  {train_metrics['accuracy']:.2%}")
            logger.info(f"Val Loss:   {val_metrics['loss_total']:.4f}")
            logger.info(f"Val Acc:    {val_metrics['accuracy_clean']:.2%}")
            logger.info(f"Val Robust: {val_metrics['accuracy_robust']:.2%}")
            logger.info(f"Val SSIM:   {val_metrics['ssim_mean']:.4f}")
            logger.info(f"Epoch Time: {epoch_time:.1f}s")
            logger.info(f"{'='*80}\n")

            # MLflow logging
            if hasattr(self.config, "logging") and self.config.logging.use_mlflow:
                self._log_epoch_metrics(train_metrics, val_metrics, epoch)

            # Checkpoint management
            is_best = val_metrics["loss_total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss_total"]
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
                logger.info("✓ New best model saved!")
            else:
                self.patience_counter += 1

            # Save last checkpoint
            if self.config.training.save_last:
                self._save_checkpoint(is_best=False)

            # Early stopping check
            if self._check_early_stopping():
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Training complete
        training_time = time.time() - training_start_time
        logger.info(f"\n{'='*80}")
        logger.info("Training Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Total training time: {training_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Final model saved at: {self.checkpoint_dir}")
        logger.info(f"{'='*80}\n")

        # Test evaluation (if test loader provided)
        if self.test_loader is not None:
            logger.info("Evaluating on test set...")
            test_metrics = self._test_epoch()
            history["test_metrics"] = test_metrics
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
            logger.info(
                f"Test Robust Accuracy: " f"{test_metrics['accuracy_robust']:.2%}"
            )

        return history

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics for the epoch.
        """
        self.model.train()
        self.tri_objective_loss.train()

        metrics = {
            "loss_total": 0.0,
            "loss_task": 0.0,
            "loss_robustness": 0.0,
            "loss_explanation": 0.0,
            "accuracy": 0.0,
        }
        num_batches = len(self.train_loader)
        num_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = images.size(0)

            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                loss_dict = self.tri_objective_loss(
                    model=self.model,
                    images=images,
                    labels=labels,
                    return_dict=True,
                )
                loss_total = loss_dict["total"]

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss_total).backward()

            # Gradient clipping
            if self.config.training.grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip_max_norm,
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute accuracy
            with torch.no_grad():
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / batch_size

            # Update metrics
            metrics["loss_total"] += loss_total.item() * batch_size
            metrics["loss_task"] += loss_dict["task"].item() * batch_size
            metrics["loss_robustness"] += loss_dict["robustness"].item() * batch_size
            metrics["loss_explanation"] += loss_dict["explanation"].item() * batch_size
            metrics["accuracy"] += correct
            num_samples += batch_size

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_total.item():.4f}",
                    "acc": f"{accuracy:.2%}",
                }
            )

            # Batch logging
            if (batch_idx + 1) % self.log_interval == 0:
                self._log_batch_metrics(loss_dict, accuracy, batch_idx)

            self.global_step += 1

        # Normalize metrics
        for key in metrics:
            if key == "accuracy":
                metrics[key] = metrics[key] / num_samples
            else:
                metrics[key] = metrics[key] / num_samples

        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics for the epoch.
        """
        self.model.eval()
        self.tri_objective_loss.eval()

        metrics = {
            "loss_total": 0.0,
            "loss_task": 0.0,
            "loss_robustness": 0.0,
            "loss_explanation": 0.0,
            "accuracy_clean": 0.0,
            "accuracy_robust": 0.0,
            "ssim_mean": 0.0,
        }
        num_samples = 0

        all_labels = []
        all_preds_clean = []
        all_preds_robust = []
        all_probs_clean = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                batch_size = images.size(0)

                # Tri-objective loss computation
                with autocast(enabled=self.mixed_precision):
                    loss_dict = self.tri_objective_loss(
                        model=self.model,
                        images=images,
                        labels=labels,
                        return_dict=True,
                    )

                # Clean accuracy
                outputs_clean = self.model(images)
                probs_clean = F.softmax(outputs_clean, dim=1)
                _, preds_clean = torch.max(outputs_clean, 1)
                correct_clean = (preds_clean == labels).sum().item()

                # Adversarial accuracy (PGD-20)
                images_adv = self.pgd_val(self.model, images, labels)
                outputs_adv = self.model(images_adv)
                _, preds_adv = torch.max(outputs_adv, 1)
                correct_robust = (preds_adv == labels).sum().item()

                # SSIM stability (subset for efficiency)
                if num_samples < 100:  # Evaluate on first 100 samples
                    cams_clean = self.gradcam(images, labels)
                    cams_adv = self.gradcam(images_adv, labels)
                    ssim = self.ssim(cams_clean, cams_adv)
                    metrics["ssim_mean"] += ssim.item() * batch_size

                # Update metrics
                metrics["loss_total"] += loss_dict["total"].item() * batch_size
                metrics["loss_task"] += loss_dict["task"].item() * batch_size
                metrics["loss_robustness"] += (
                    loss_dict["robustness"].item() * batch_size
                )
                metrics["loss_explanation"] += (
                    loss_dict["explanation"].item() * batch_size
                )
                metrics["accuracy_clean"] += correct_clean
                metrics["accuracy_robust"] += correct_robust
                num_samples += batch_size

                # Store for metrics computation
                all_labels.append(labels.cpu())
                all_preds_clean.append(preds_clean.cpu())
                all_preds_robust.append(preds_adv.cpu())
                all_probs_clean.append(probs_clean.cpu())

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total'].item():.4f}",
                        "acc": f"{correct_clean / batch_size:.2%}",
                    }
                )

        # Normalize metrics
        for key in metrics:
            if "accuracy" in key:
                metrics[key] = metrics[key] / num_samples
            elif key == "ssim_mean":
                metrics[key] = metrics[key] / min(num_samples, 100)
            else:
                metrics[key] = metrics[key] / num_samples

        # Compute additional metrics
        all_labels = torch.cat(all_labels).numpy()
        all_preds_clean = torch.cat(all_preds_clean).numpy()
        all_probs_clean = torch.cat(all_probs_clean).numpy()

        # Classification metrics
        clf_metrics = compute_classification_metrics(
            y_true=all_labels,
            y_pred=all_preds_clean,
            y_probs=all_probs_clean,
            num_classes=self.config.model.num_classes,
        )
        metrics["auroc_macro"] = clf_metrics["auroc_macro"]
        metrics["f1_macro"] = clf_metrics["f1_macro"]

        # Calibration metrics
        cal_metrics = evaluate_calibration(
            y_true=all_labels,
            y_probs=all_probs_clean,
            num_bins=15,
        )
        metrics["ece"] = cal_metrics["ece"]
        metrics["mce"] = cal_metrics["mce"]

        return metrics

    def _test_epoch(self) -> Dict[str, float]:
        """
        Evaluate on test set.

        Returns:
            Dictionary of test metrics.
        """
        self.model.eval()

        metrics = {
            "accuracy": 0.0,
            "accuracy_robust": 0.0,
            "auroc_macro": 0.0,
            "f1_macro": 0.0,
            "ece": 0.0,
        }
        num_samples = 0

        all_labels = []
        all_preds_clean = []
        all_preds_robust = []
        all_probs_clean = []

        logger.info("Running test evaluation...")

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test", leave=False):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                batch_size = images.size(0)

                # Clean predictions
                outputs_clean = self.model(images)
                probs_clean = F.softmax(outputs_clean, dim=1)
                _, preds_clean = torch.max(outputs_clean, 1)
                correct_clean = (preds_clean == labels).sum().item()

                # Adversarial predictions
                images_adv = self.pgd_val(self.model, images, labels)
                outputs_adv = self.model(images_adv)
                _, preds_adv = torch.max(outputs_adv, 1)
                correct_robust = (preds_adv == labels).sum().item()

                # Update metrics
                metrics["accuracy"] += correct_clean
                metrics["accuracy_robust"] += correct_robust
                num_samples += batch_size

                # Store for metrics
                all_labels.append(labels.cpu())
                all_preds_clean.append(preds_clean.cpu())
                all_preds_robust.append(preds_adv.cpu())
                all_probs_clean.append(probs_clean.cpu())

        # Normalize metrics
        metrics["accuracy"] = metrics["accuracy"] / num_samples
        metrics["accuracy_robust"] = metrics["accuracy_robust"] / num_samples

        # Compute additional metrics
        all_labels = torch.cat(all_labels).numpy()
        all_preds_clean = torch.cat(all_preds_clean).numpy()
        all_probs_clean = torch.cat(all_probs_clean).numpy()

        clf_metrics = compute_classification_metrics(
            y_true=all_labels,
            y_pred=all_preds_clean,
            y_probs=all_probs_clean,
            num_classes=self.config.model.num_classes,
        )
        metrics["auroc_macro"] = clf_metrics["auroc_macro"]
        metrics["f1_macro"] = clf_metrics["f1_macro"]

        cal_metrics = evaluate_calibration(
            y_true=all_labels,
            y_probs=all_probs_clean,
            num_bins=15,
        )
        metrics["ece"] = cal_metrics["ece"]

        return metrics

    def _log_batch_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        accuracy: float,
        batch_idx: int,
    ) -> None:
        """Log batch-level metrics to MLflow.

        Args:
            loss_dict: Dictionary of loss components.
            accuracy: Batch accuracy.
            batch_idx: Current batch index.
        """
        if not hasattr(self.config, "logging") or not self.config.logging.use_mlflow:
            return

        step = self.global_step
        mlflow.log_metric("batch/loss_total", loss_dict["total"].item(), step=step)
        mlflow.log_metric("batch/loss_task", loss_dict["task"].item(), step=step)
        mlflow.log_metric(
            "batch/loss_robustness",
            loss_dict["robustness"].item(),
            step=step,
        )
        mlflow.log_metric(
            "batch/loss_explanation",
            loss_dict["explanation"].item(),
            step=step,
        )
        mlflow.log_metric("batch/accuracy", accuracy, step=step)
        mlflow.log_metric(
            "batch/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            step=step,
        )

    def _log_epoch_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """Log epoch-level metrics to MLflow.

        Args:
            train_metrics: Training metrics for the epoch.
            val_metrics: Validation metrics for the epoch.
            epoch: Current epoch number.
        """
        if not hasattr(self.config, "logging") or not self.config.logging.use_mlflow:
            return

        # Training metrics
        mlflow.log_metric("train/loss_total", train_metrics["loss_total"], step=epoch)
        mlflow.log_metric("train/loss_task", train_metrics["loss_task"], step=epoch)
        mlflow.log_metric(
            "train/loss_robustness",
            train_metrics["loss_robustness"],
            step=epoch,
        )
        mlflow.log_metric(
            "train/loss_explanation",
            train_metrics["loss_explanation"],
            step=epoch,
        )
        mlflow.log_metric("train/accuracy", train_metrics["accuracy"], step=epoch)

        # Validation metrics
        mlflow.log_metric("val/loss_total", val_metrics["loss_total"], step=epoch)
        mlflow.log_metric("val/loss_task", val_metrics["loss_task"], step=epoch)
        mlflow.log_metric(
            "val/loss_robustness",
            val_metrics["loss_robustness"],
            step=epoch,
        )
        mlflow.log_metric(
            "val/loss_explanation",
            val_metrics["loss_explanation"],
            step=epoch,
        )
        mlflow.log_metric(
            "val/accuracy_clean",
            val_metrics["accuracy_clean"],
            step=epoch,
        )
        mlflow.log_metric(
            "val/accuracy_robust",
            val_metrics["accuracy_robust"],
            step=epoch,
        )
        mlflow.log_metric("val/ssim_mean", val_metrics["ssim_mean"], step=epoch)
        mlflow.log_metric("val/auroc_macro", val_metrics["auroc_macro"], step=epoch)
        mlflow.log_metric("val/f1_macro", val_metrics["f1_macro"], step=epoch)
        mlflow.log_metric("val/ece", val_metrics["ece"], step=epoch)
        mlflow.log_metric("val/mce", val_metrics["mce"], step=epoch)

        # Learning rate
        mlflow.log_metric(
            "learning_rate",
            self.optimizer.param_groups[0]["lr"],
            step=epoch,
        )

        # Temperature (if learnable)
        if hasattr(self.tri_objective_loss, "temperature"):
            temp = self.tri_objective_loss.temperature.item()
            mlflow.log_metric("temperature", temp, step=epoch)

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            checkpoint_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Best checkpoint saved: {checkpoint_path}")

            # Log to MLflow
            if (
                hasattr(self.config, "logging")
                and self.config.logging.use_mlflow
                and hasattr(self.config.logging, "log_best_model")
                and self.config.logging.log_best_model
            ):
                mlflow.pytorch.log_model(self.model, "best_model")

        if self.config.training.save_last:
            checkpoint_path = self.checkpoint_dir / "last.pt"
            torch.save(checkpoint, checkpoint_path)

    def _check_early_stopping(self) -> bool:
        """Check if early stopping should be triggered.

        Returns:
            True if early stopping should be triggered, False otherwise.
        """
        patience = self.config.training.early_stopping_patience
        if self.patience_counter >= patience:
            logger.info(f"No improvement for {patience} epochs, stopping early")
            return True
        return False


def create_data_loaders(
    config: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import WeightedRandomSampler

    logger.info("Creating data loaders...")

    # Create transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    image_size = config.dataset.image_size

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    # Training dataset
    train_dataset = ISICDataset(
        root=config.dataset.root,
        split="train",
        transforms=train_transforms,
    )

    # Validation dataset
    val_dataset = ISICDataset(
        root=config.dataset.root,
        split="val",
        transforms=val_transforms,
    )

    # Test dataset (optional)
    test_dataset = None
    try:
        test_dataset = ISICDataset(
            root=config.dataset.root,
            split="test",
            transforms=val_transforms,
        )
    except Exception as e:
        logger.warning(f"Test dataset not available: {e}")

    # Weighted sampler for class imbalance
    if config.dataset.use_class_weights:
        class_weights = torch.tensor(config.dataset.class_weights)
        sample_weights = class_weights[train_dataset.labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def main() -> None:
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Tri-Objective Training for Medical Imaging"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/tri_objective.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--cavs-checkpoint",
        type=str,
        default=None,
        help="Path to trained CAVs checkpoint (enables TCAV in explanation loss)",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_experiment_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config.reproducibility.seed = args.seed
        logger.info(f"Seed overridden to: {args.seed}")

    # Set random seed
    set_global_seed(config.reproducibility.seed)
    logger.info(f"Random seed set to: {config.reproducibility.seed}")

    # Load CAVs if provided
    cavs_data = None
    if args.cavs_checkpoint:
        logger.info(f"Loading CAVs from: {args.cavs_checkpoint}")
        try:
            # PyTorch 2.6+ requires weights_only=False for numpy arrays
            cavs_data = torch.load(
                args.cavs_checkpoint, map_location="cpu", weights_only=False
            )
            logger.info("✅ CAVs loaded successfully")

            # Log CAV metadata
            if "metadata" in cavs_data:
                metadata = cavs_data["metadata"]
                logger.info(f"CAV training info:")
                for concept_name, info in metadata.items():
                    logger.info(
                        f"  - {concept_name}: "
                        f"accuracy={info['accuracy']:.3f}, "
                        f"type={info['type']}"
                    )
        except Exception as e:
            logger.error(f"Failed to load CAVs: {e}")
            logger.warning("Continuing without CAVs (TCAV will be disabled)")
            cavs_data = None
    else:
        logger.warning(
            "⚠️  No CAVs checkpoint provided! "
            "TCAV component (H2.2-H2.4) will be disabled. "
            "To enable TCAV, first train CAVs:\n"
            "    python scripts/training/train_cavs_for_training.py \\\n"
            "        --data_dir data/processed/isic2018 \\\n"
            "        --model_checkpoint checkpoints/baseline/seed_42/best.pt \\\n"
            "        --output_dir checkpoints/cavs\n"
            "Then add: --cavs-checkpoint checkpoints/cavs/trained_cavs.pt"
        )

    # Initialize MLflow (if logging config exists)
    if hasattr(config, "logging") and config.logging.use_mlflow:
        init_mlflow(
            experiment_name=config.logging.mlflow_experiment_name,
            run_name=(
                f"{config.experiment.name}_" f"seed{config.reproducibility.seed}"
            ),
            tracking_uri=config.logging.mlflow_tracking_uri,
        )
        # Log configuration
        mlflow.log_params(
            {
                "seed": config.reproducibility.seed,
                "lambda_rob": config.loss.lambda_rob,
                "lambda_expl": config.loss.lambda_expl,
                "learning_rate": config.training.learning_rate,
                "batch_size": config.dataset.batch_size,
                "max_epochs": config.training.max_epochs,
            }
        )
        # Log reproducibility state
        repro_state = get_reproducibility_state()
        mlflow.log_params(
            {
                "torch_version": repro_state.torch_version,
                "cuda_version": repro_state.cuda_version,
                "cudnn_version": repro_state.cudnn_version,
            }
        )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # Build model
    logger.info("Building model...")
    model = build_model(config.model)
    logger.info(f"Model: {config.model.name}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.max_epochs,
        eta_min=config.training.scheduler_params.eta_min,
    )

    # Checkpoint directory
    checkpoint_dir = args.checkpoint_dir or config.training.checkpoint_dir
    checkpoint_dir = Path(checkpoint_dir) / f"seed_{config.reproducibility.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = TriObjectiveTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.training.device,
        mixed_precision=config.training.mixed_precision,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=(
            getattr(config.logging, "log_interval", 10)
            if hasattr(config, "logging")
            else 10
        ),
        cavs_data=cavs_data,  # Pass CAVs to trainer
    )

    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(
            args.resume, map_location=config.training.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"] + 1
        trainer.global_step = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    # Train
    try:
        history = trainer.train(num_epochs=config.training.max_epochs)

        # Log final results
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        if test_loader:
            test_acc = history["test_metrics"]["accuracy"]
            test_robust = history["test_metrics"]["accuracy_robust"]
            logger.info(f"Test accuracy: {test_acc:.2%}")
            logger.info(f"Test robust accuracy: {test_robust:.2%}")
        logger.info("=" * 80 + "\n")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        if hasattr(config, "logging") and config.logging.use_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()
