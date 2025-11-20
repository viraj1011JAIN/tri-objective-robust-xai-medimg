from __future__ import annotations

"""
Abstract Base Trainer for Tri-Objective Robust XAI Medical Imaging.

Provides training loop skeleton with:
- Training and validation loops
- Checkpoint management
- Early stopping
- Learning rate scheduling
- Optional MLflow integration
- Multi-seed experiment support

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Project: Tri-Objective Robust XAI for Medical Imaging
Target: A1+ Grade | Publication-Ready (NeurIPS/MICCAI/TMI)

Location: src/training/base_trainer.py
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

try:
    import mlflow
except Exception:
    mlflow = None


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    max_epochs: int = 100
    eval_every_n_epochs: int = 1
    log_every_n_steps: int = 50
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    gradient_clip_val: float = 1.0
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None

    # Add missing fields for train_baseline.py
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    loss: float = 0.0
    accuracy: float = 0.0
    num_batches: int = 0
    num_samples: int = 0
    extra_metrics: Dict[str, float] = field(default_factory=dict)


class BaseTrainer(ABC):
    """
    Abstract base trainer for all models.

    Provides:
    - Training and validation loops
    - Checkpoint saving/loading
    - Early stopping
    - Learning rate scheduling
    - Optional MLflow logging
    - Multi-seed support

    Subclasses must implement:
    - training_step()
    - validation_step()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainingConfig,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """Initialize trainer."""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Handle checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = (
            float("inf") if config.monitor_mode == "min" else float("-inf")
        )
        self.best_epoch = 0
        self.patience_counter = 0

        # Add this for mypy
        self.best_val_loss: float = float("inf")

        # History tracking
        self.train_metrics_history: List[TrainingMetrics] = []
        self.val_metrics_history: List[TrainingMetrics] = []

        # MLflow setup
        if config.use_mlflow and mlflow is not None:
            self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        if self.config.mlflow_experiment_name:
            mlflow.set_experiment(self.config.mlflow_experiment_name)

        mlflow.start_run()
        mlflow.log_params(
            {
                "max_epochs": self.config.max_epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )

    def _log_mlflow_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to MLflow."""
        if self.config.use_mlflow and mlflow is not None and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    @abstractmethod
    def training_step(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute single training step.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (loss, metrics_dict)
        """
        pass

    @abstractmethod
    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute single validation step.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (loss, metrics_dict)
        """
        pass

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch."""
        if isinstance(batch, (tuple, list)):
            return len(batch[0])
        elif isinstance(batch, dict):
            return len(next(iter(batch.values())))
        else:
            return len(batch)

    def train_epoch(self) -> TrainingMetrics:
        """Run one training epoch."""
        self.model.train()
        metrics = TrainingMetrics()

        for batch_idx, batch in enumerate(self.train_loader):
            loss, batch_metrics = self.training_step(batch, batch_idx)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.config.gradient_clip_val and self.config.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )

            self.optimizer.step()

            batch_size = self._get_batch_size(batch)
            metrics.loss += float(loss.item()) * batch_size
            metrics.num_batches += 1
            metrics.num_samples += batch_size

            for key, val in batch_metrics.items():
                if hasattr(metrics, key):
                    current = getattr(metrics, key)
                    setattr(metrics, key, current + (float(val) * batch_size))

            self.global_step += 1

            if (
                self.config.log_every_n_steps > 0
                and (batch_idx + 1) % self.config.log_every_n_steps == 0
            ):
                avg_loss = metrics.loss / max(metrics.num_samples, 1)
                logger.info(
                    "Epoch %d | Step %d | Loss: %.4f",
                    self.current_epoch,
                    self.global_step,
                    avg_loss,
                )
                self._log_mlflow_metrics(
                    {"train/loss": avg_loss}, step=self.global_step
                )

        if metrics.num_samples > 0:
            metrics.loss /= metrics.num_samples
            metrics.accuracy /= metrics.num_samples

        logger.info("Epoch %d Train Loss: %.4f", self.current_epoch, metrics.loss)
        return metrics

    def validate(self) -> TrainingMetrics:
        """Run validation loop."""
        if self.val_loader is None:
            logger.warning("validate() called but no val_loader was provided.")
            return TrainingMetrics()

        self.model.eval()
        metrics = TrainingMetrics()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                loss, batch_metrics = self.validation_step(batch, batch_idx)

                batch_size = self._get_batch_size(batch)
                metrics.loss += float(loss.item()) * batch_size
                metrics.num_batches += 1
                metrics.num_samples += batch_size

                for key, val in batch_metrics.items():
                    if hasattr(metrics, key):
                        current = getattr(metrics, key)
                        setattr(metrics, key, current + (float(val) * batch_size))

        if metrics.num_samples > 0:
            metrics.loss /= metrics.num_samples
            metrics.accuracy /= metrics.num_samples

        logger.info("Epoch %d Val Loss: %.4f", self.current_epoch, metrics.loss)
        return metrics

    def _check_early_stopping(self, val_metrics: TrainingMetrics) -> bool:
        """Check early stopping criterion."""
        current_metric = val_metrics.loss

        improved = False
        if self.config.monitor_mode == "min":
            if current_metric < (
                self.best_metric - self.config.early_stopping_min_delta
            ):
                improved = True
        else:
            if current_metric > (
                self.best_metric + self.config.early_stopping_min_delta
            ):
                improved = True

        if improved:
            self.best_metric = current_metric
            self.best_epoch = self.current_epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.config.early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d (no improvement for %d epochs)",
                self.current_epoch,
                self.patience_counter,
            )

        return improved

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best model to %s", best_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)

        logger.info("Loaded checkpoint from epoch %d", self.current_epoch)

    def fit(self) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info("Starting training for %d epochs", self.config.max_epochs)

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.train_metrics_history.append(train_metrics)

            if self.val_loader is not None and (
                (epoch + 1) % self.config.eval_every_n_epochs == 0
            ):
                val_metrics = self.validate()
                self.val_metrics_history.append(val_metrics)

                self._log_mlflow_metrics(
                    {"val/loss": float(val_metrics.loss)}, step=epoch
                )

                improved = self._check_early_stopping(val_metrics)
                self.save_checkpoint(is_best=improved)

            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Training complete. Best epoch: %d", self.best_epoch)

        return {
            "train_loss": [m.loss for m in self.train_metrics_history],
            "train_acc": [m.accuracy for m in self.train_metrics_history],
            "val_loss": [m.loss for m in self.val_metrics_history],
            "val_acc": [m.accuracy for m in self.val_metrics_history],
        }
