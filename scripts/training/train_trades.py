#!/usr/bin/env python3
"""
================================================================================
TRADES Training Script - Phase 5.3
================================================================================
TRADES: TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization

Loss Formulation:
    L_TRADES = L_CE(f(x), y) + β × KL(f(x) || f(x_adv))

    where:
    - L_CE: Standard cross-entropy loss on clean samples
    - KL: KL divergence between clean and adversarial predictions
    - β: Trade-off hyperparameter (controls clean vs robust accuracy)

Key Advantages over PGD-AT:
    1. Explicit clean loss term → better clean accuracy preservation
    2. KL divergence in adversarial term → smoother decision boundaries
    3. Controllable trade-off via beta parameter
    4. Better calibration and generalization

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 2025
================================================================================
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.pgd import PGDAttack
from src.data.isic_dataset import ISICDataset
from src.models.model_factory import get_model
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.logging_utils import setup_logger
from src.utils.metrics import compute_metrics

# =============================================================================
# TRADES Loss Implementation
# =============================================================================


class TRADESLoss(nn.Module):
    """
    TRADES loss function for adversarial training.

    Combines cross-entropy loss on clean samples with KL divergence
    between clean and adversarial predictions.

    Formula:
        L_TRADES = L_CE(f(x), y) + β × KL(f(x) || f(x_adv))

    Args:
        beta: Trade-off parameter controlling clean vs robust accuracy
        reduction: Loss reduction method ('mean' or 'sum')
    """

    def __init__(self, beta: float = 6.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

    def kl_divergence(
        self, logits_clean: torch.Tensor, logits_adv: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between clean and adversarial predictions.

        KL(P || Q) = Σ P(x) log(P(x) / Q(x))

        Args:
            logits_clean: Clean logits [batch_size, num_classes]
            logits_adv: Adversarial logits [batch_size, num_classes]

        Returns:
            KL divergence scalar
        """
        # Convert logits to log probabilities
        log_probs_clean = F.log_softmax(logits_clean, dim=1)
        log_probs_adv = F.log_softmax(logits_adv, dim=1)

        # KL divergence with numerical stability
        # KL(clean || adv) = Σ exp(log_p_clean) * (log_p_clean - log_p_adv)
        kl = torch.sum(
            torch.exp(log_probs_clean) * (log_probs_clean - log_probs_adv), dim=1
        )

        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        else:
            return kl

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_adv: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute TRADES loss.

        Args:
            logits_clean: Clean logits [batch_size, num_classes]
            logits_adv: Adversarial logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            total_loss: Combined TRADES loss
            loss_dict: Dictionary with individual loss components
        """
        # Clean loss: standard cross-entropy
        clean_loss = self.ce_loss(logits_clean, targets)

        # Robust loss: KL divergence between clean and adversarial predictions
        robust_loss = self.kl_divergence(logits_clean, logits_adv)

        # Total TRADES loss
        total_loss = clean_loss + self.beta * robust_loss

        # Return loss and components for logging
        loss_dict = {
            "total_loss": total_loss.item(),
            "clean_loss": clean_loss.item(),
            "robust_loss": robust_loss.item(),
            "beta": self.beta,
        }

        return total_loss, loss_dict


# =============================================================================
# TRADES Trainer Implementation
# =============================================================================


class TRADESTrainer:
    """
    Complete TRADES training pipeline with production-grade features.

    Features:
        - Mixed precision training (AMP)
        - Gradient clipping
        - Learning rate scheduling
        - MLflow experiment tracking
        - Comprehensive metrics logging
        - Memory-efficient training
        - Multi-seed reproducibility
        - Checkpoint management

    Args:
        config: Configuration dictionary
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        logger: Logger instance
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_attack()
        self._setup_amp()
        self._setup_tracking()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_robust_acc = 0.0
        self.global_step = 0

        self.logger.info("TRADESTrainer initialized successfully")

    def _setup_optimizer(self) -> None:
        """Initialize optimizer with configuration."""
        train_cfg = self.config["training"]

        self.optimizer = SGD(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
            nesterov=train_cfg.get("nesterov", True),
        )

        self.logger.info(
            f"Optimizer: SGD (lr={train_cfg['learning_rate']}, "
            f"momentum={train_cfg['momentum']}, "
            f"weight_decay={train_cfg['weight_decay']})"
        )

    def _setup_scheduler(self) -> None:
        """Initialize learning rate scheduler."""
        lr_cfg = self.config["training"]["lr_scheduler"]

        if lr_cfg["type"] == "multistep":
            self.scheduler = MultiStepLR(
                self.optimizer, milestones=lr_cfg["milestones"], gamma=lr_cfg["gamma"]
            )
            self.logger.info(
                f"Scheduler: MultiStepLR (milestones={lr_cfg['milestones']}, "
                f"gamma={lr_cfg['gamma']})"
            )
        else:
            raise ValueError(f"Unknown scheduler type: {lr_cfg['type']}")

    def _setup_loss(self) -> None:
        """Initialize TRADES loss function."""
        beta = self.config["trades"]["beta"]
        self.criterion = TRADESLoss(beta=beta, reduction="mean")
        self.logger.info(f"TRADES Loss initialized (beta={beta})")

    def _setup_attack(self) -> None:
        """Initialize PGD attack for adversarial example generation."""
        attack_cfg = self.config["trades"]["train_attack"]

        self.attack = PGDAttack(
            model=self.model,
            epsilon=attack_cfg["epsilon"],
            num_steps=attack_cfg["num_steps"],
            step_size=attack_cfg["step_size"],
            random_start=attack_cfg["random_start"],
            loss_type=attack_cfg["loss_type"],
            targeted=attack_cfg["targeted"],
            clip_min=attack_cfg["clip_min"],
            clip_max=attack_cfg["clip_max"],
        )

        self.logger.info(
            f"Attack: PGD (ε={attack_cfg['epsilon']}, "
            f"steps={attack_cfg['num_steps']}, "
            f"step_size={attack_cfg['step_size']}, "
            f"loss_type={attack_cfg['loss_type']})"
        )

    def _setup_amp(self) -> None:
        """Initialize automatic mixed precision training."""
        amp_cfg = self.config["training"]["amp"]

        self.use_amp = amp_cfg["enabled"]
        if self.use_amp:
            self.scaler = GradScaler(enabled=True)
            self.logger.info(f"AMP enabled (opt_level={amp_cfg['opt_level']})")
        else:
            self.scaler = None
            self.logger.info("AMP disabled")

    def _setup_tracking(self) -> None:
        """Initialize experiment tracking."""
        mlflow_cfg = self.config["output"]["mlflow"]

        if mlflow_cfg["enabled"]:
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
            mlflow.set_experiment(mlflow_cfg["experiment_name"])
            self.logger.info(
                f"MLflow tracking enabled: {mlflow_cfg['experiment_name']}"
            )
        else:
            self.logger.info("MLflow tracking disabled")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()

        # Metrics tracking
        epoch_metrics = {
            "total_loss": 0.0,
            "clean_loss": 0.0,
            "robust_loss": 0.0,
            "clean_accuracy": 0.0,
            "robust_accuracy": 0.0,
            "num_samples": 0,
        }

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
            ncols=120,
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)

            # Generate adversarial examples
            with torch.no_grad():
                adv_images = self.attack(images, labels)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type="cuda"):
                    # Clean predictions
                    logits_clean = self.model(images)

                    # Adversarial predictions
                    logits_adv = self.model(adv_images)

                    # TRADES loss
                    loss, loss_dict = self.criterion(logits_clean, logits_adv, labels)

                # Backward pass with gradient scaling
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config["training"]["grad_clip"]["enabled"]:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config["training"]["grad_clip"]["max_norm"],
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Without AMP
                logits_clean = self.model(images)
                logits_adv = self.model(adv_images)
                loss, loss_dict = self.criterion(logits_clean, logits_adv, labels)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if self.config["training"]["grad_clip"]["enabled"]:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config["training"]["grad_clip"]["max_norm"],
                    )

                self.optimizer.step()

            # Compute accuracies
            with torch.no_grad():
                clean_preds = logits_clean.argmax(dim=1)
                clean_acc = (clean_preds == labels).float().mean().item()

                adv_preds = logits_adv.argmax(dim=1)
                robust_acc = (adv_preds == labels).float().mean().item()

            # Update metrics
            epoch_metrics["total_loss"] += loss_dict["total_loss"] * batch_size
            epoch_metrics["clean_loss"] += loss_dict["clean_loss"] * batch_size
            epoch_metrics["robust_loss"] += loss_dict["robust_loss"] * batch_size
            epoch_metrics["clean_accuracy"] += clean_acc * batch_size
            epoch_metrics["robust_accuracy"] += robust_acc * batch_size
            epoch_metrics["num_samples"] += batch_size

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "clean_acc": f"{clean_acc:.4f}",
                    "robust_acc": f"{robust_acc:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            self.global_step += 1

        # Average metrics
        for key in [
            "total_loss",
            "clean_loss",
            "robust_loss",
            "clean_accuracy",
            "robust_accuracy",
        ]:
            epoch_metrics[key] /= epoch_metrics["num_samples"]

        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()

        # Metrics tracking
        val_metrics = {"clean_accuracy": 0.0, "robust_accuracy": 0.0, "num_samples": 0}

        pbar = tqdm(
            self.val_loader,
            desc=f"Validation {epoch}/{self.config['training']['num_epochs']}",
            ncols=120,
        )

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)

            # Clean predictions
            logits_clean = self.model(images)
            clean_preds = logits_clean.argmax(dim=1)
            clean_acc = (clean_preds == labels).float().mean().item()

            # Adversarial predictions
            adv_images = self.attack(images, labels)
            logits_adv = self.model(adv_images)
            adv_preds = logits_adv.argmax(dim=1)
            robust_acc = (adv_preds == labels).float().mean().item()

            # Update metrics
            val_metrics["clean_accuracy"] += clean_acc * batch_size
            val_metrics["robust_accuracy"] += robust_acc * batch_size
            val_metrics["num_samples"] += batch_size

            pbar.set_postfix(
                {"clean_acc": f"{clean_acc:.4f}", "robust_acc": f"{robust_acc:.4f}"}
            )

        # Average metrics
        val_metrics["clean_accuracy"] /= val_metrics["num_samples"]
        val_metrics["robust_accuracy"] /= val_metrics["num_samples"]

        return val_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = (
            Path(self.config["output"]["base_dir"])
            / self.config["output"]["checkpoints_dir"]
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_robust_acc": self.best_val_robust_acc,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")

        # Save last checkpoint
        last_path = checkpoint_dir / "last.pt"
        torch.save(checkpoint, last_path)

    def train(self) -> None:
        """Main training loop."""
        num_epochs = self.config["training"]["num_epochs"]

        self.logger.info("=" * 80)
        self.logger.info("Starting TRADES Training")
        self.logger.info("=" * 80)

        # Start MLflow run
        if self.config["output"]["mlflow"]["enabled"]:
            mlflow.start_run(
                run_name=f"{self.config['output']['mlflow']['run_name_prefix']}_seed_{self.config['reproducibility']['seed']}"
            )
            mlflow.log_params(self.config)

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Train Clean Acc: {train_metrics['clean_accuracy']:.4f}, "
                f"Train Robust Acc: {train_metrics['robust_accuracy']:.4f}, "
                f"Val Clean Acc: {val_metrics['clean_accuracy']:.4f}, "
                f"Val Robust Acc: {val_metrics['robust_accuracy']:.4f}"
            )

            # MLflow logging
            if self.config["output"]["mlflow"]["enabled"]:
                mlflow.log_metrics(
                    {
                        "train_total_loss": train_metrics["total_loss"],
                        "train_clean_loss": train_metrics["clean_loss"],
                        "train_robust_loss": train_metrics["robust_loss"],
                        "train_clean_accuracy": train_metrics["clean_accuracy"],
                        "train_robust_accuracy": train_metrics["robust_accuracy"],
                        "val_clean_accuracy": val_metrics["clean_accuracy"],
                        "val_robust_accuracy": val_metrics["robust_accuracy"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

            # Save checkpoint
            is_best = val_metrics["robust_accuracy"] > self.best_val_robust_acc
            if is_best:
                self.best_val_robust_acc = val_metrics["robust_accuracy"]
                self.best_val_acc = val_metrics["clean_accuracy"]

            if epoch % self.config["training"]["save_freq"] == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # Memory cleanup
            if self.config["resources"].get("clear_memory_between_epochs", True):
                torch.cuda.empty_cache()

        # End MLflow run
        if self.config["output"]["mlflow"]["enabled"]:
            mlflow.end_run()

        self.logger.info("=" * 80)
        self.logger.info("Training Complete")
        self.logger.info(f"Best Validation Clean Accuracy: {self.best_val_acc:.4f}")
        self.logger.info(
            f"Best Validation Robust Accuracy: {self.best_val_robust_acc:.4f}"
        )
        self.logger.info("=" * 80)


# =============================================================================
# Main Training Function
# =============================================================================


def main():
    """Main function for TRADES training."""
    parser = argparse.ArgumentParser(description="TRADES Training - Phase 5.3")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/trades_isic.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model architecture (overrides config)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="TRADES beta parameter (overrides config)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.seed is not None:
        config["reproducibility"]["seed"] = args.seed
    if args.model is not None:
        config["model"]["architecture"] = args.model
    if args.beta is not None:
        config["trades"]["beta"] = args.beta

    # Set random seeds
    seed = config["reproducibility"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = config["reproducibility"][
            "use_deterministic_algorithms"
        ]
        torch.backends.cudnn.benchmark = config["resources"].get("benchmark", False)

    # Setup logging
    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / config["output"]["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="TRADES",
        log_file=log_dir / f"train_seed_{seed}.log",
        level=config["output"]["logging"]["level"],
    )

    logger.info("=" * 80)
    logger.info("TRADES Training - Phase 5.3")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Model: {config['model']['architecture']}")
    logger.info(f"Beta: {config['trades']['beta']}")
    logger.info("=" * 80)

    # Setup device
    device = torch.device(
        f"cuda:{config['resources']['cuda_device']}"
        if torch.cuda.is_available() and config["resources"]["device"] == "cuda"
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # Load datasets
    logger.info("Loading datasets...")

    train_dataset = ISICDataset(
        data_dir=config["data"]["data_dir"],
        metadata_csv=config["data"]["metadata_csv"],
        split="train",
        transform=True,
        image_size=config["data"]["image_size"],
    )

    val_dataset = ISICDataset(
        data_dir=config["data"]["data_dir"],
        metadata_csv=config["data"]["metadata_csv"],
        split="val",
        transform=False,
        image_size=config["data"]["image_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        persistent_workers=config["training"].get("persistent_workers", False),
        prefetch_factor=config["training"].get("prefetch_factor", 2),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Load model
    logger.info("Loading model...")
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )
    model = model.to(device)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize trainer
    trainer = TRADESTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
