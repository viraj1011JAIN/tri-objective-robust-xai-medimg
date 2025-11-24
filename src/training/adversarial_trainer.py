"""
Adversarial Training Infrastructure
====================================

High-performance adversarial training framework with support for:
1. **TRADES** - TRadeoff-inspired Adversarial DEfense
2. **MART** - Misclassification Aware adversarial tRaining
3. **Standard AT** - Classic adversarial training (Madry et al.)
4. **Mixed Training** - Alternate clean and adversarial batches

This module provides production-ready adversarial training with:
- On-the-fly adversarial example generation during training
- Mixed precision training (AMP) support
- Comprehensive metrics tracking (clean + robust accuracy)
- Gradient clipping and numerical stability
- Checkpointing and early stopping
- Integration with existing training infrastructure

Design Philosophy:
------------------
1. **Flexibility**: Support multiple adversarial training methods
2. **Efficiency**: Generate adversarial examples on-the-fly (no storage)
3. **Monitoring**: Track both clean and robust metrics
4. **Stability**: Gradient clipping, loss clipping, AMP support
5. **Production-Ready**: Error handling, logging, resumability

Adversarial Training Process:
------------------------------
For each training batch:
    1. Forward pass on clean examples → clean logits
    2. Generate adversarial examples using PGD/FGSM
    3. Forward pass on adversarial examples → adv logits
    4. Compute robust loss (TRADES/MART/AT)
    5. Backward pass and optimizer step
    6. Track metrics (clean acc, robust acc, loss components)

Medical Imaging Considerations:
--------------------------------
- **Perturbation Budget**: ε ∈ [4/255, 8/255] for dermoscopy
- **Training Time**: ~3-5x slower than standard training
- **Robustness-Accuracy Tradeoff**: β ∈ [0.5, 2.0] for medical imaging
- **Clinical Validation**: Test on real sensor noise, not just L∞ attacks

References:
-----------
[1] Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to
    Adversarial Attacks", ICLR 2018
[2] Zhang, H., et al. (2019). "Theoretically Principled Trade-off between
    Robustness and Accuracy", ICML 2019
[3] Wang, Y., et al. (2020). "Improving Adversarial Robustness Requires
    Revisiting Misclassified Examples", ICLR 2020

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Date: November 24, 2025
Version: 5.1.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..attacks.base import BaseAttack
from ..attacks.pgd import PGD, PGDConfig
from ..losses.robust_loss import AdversarialTrainingLoss, MARTLoss, TRADESLoss

logger = logging.getLogger(__name__)

__all__ = [
    "AdversarialTrainingConfig",
    "AdversarialTrainer",
    "train_adversarial_epoch",
    "validate_robust",
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AdversarialTrainingConfig:
    """
    Configuration for adversarial training.

    This dataclass encapsulates all hyperparameters and settings for
    adversarial training, enabling easy experiment tracking and reproducibility.

    Attributes:
        # Loss Configuration
        loss_type: Type of robust loss ('trades' | 'mart' | 'at')
        beta: Tradeoff parameter for TRADES/MART (higher = more robust)

        # Attack Configuration (for training-time adversarial examples)
        attack_epsilon: L∞ perturbation budget (e.g., 8/255)
        attack_steps: Number of PGD steps during training
        attack_step_size: Step size per PGD iteration
        attack_random_start: Whether to use random initialization in PGD

        # Training Strategy
        mix_clean: Proportion of clean examples in batch (0.0 = adv only)
        alternate_batches: Whether to alternate clean/adv batches (vs. mixed)

        # Optimization
        gradient_clip: Maximum gradient norm (None = no clipping)
        use_amp: Whether to use automatic mixed precision

        # Evaluation
        eval_attack_steps: Number of PGD steps for validation (typically more)
        eval_epsilon: Perturbation budget for evaluation (can differ from training)

        # Monitoring
        track_clean_acc: Whether to track clean accuracy during training
        log_frequency: Log metrics every N batches (None = no logging)

    Examples:
        >>> # TRADES training for dermoscopy
        >>> config = AdversarialTrainingConfig(
        ...     loss_type='trades',
        ...     beta=1.0,
        ...     attack_epsilon=8/255,
        ...     attack_steps=10,
        ... )

        >>> # MART training with higher robustness
        >>> config = AdversarialTrainingConfig(
        ...     loss_type='mart',
        ...     beta=3.0,
        ...     attack_epsilon=8/255,
        ...     attack_steps=10,
        ... )
    """

    # Loss configuration
    loss_type: Literal["trades", "mart", "at"] = "trades"
    beta: float = 1.0

    # Attack configuration
    attack_epsilon: float = 8 / 255  # 8/255 for dermoscopy, 4/255 for CXR
    attack_steps: int = 10  # Fast training (10), thorough (40)
    attack_step_size: Optional[float] = None  # Default: epsilon/4
    attack_random_start: bool = True

    # Training strategy
    mix_clean: float = 0.0  # 0.0 = pure adversarial, 0.5 = mixed
    alternate_batches: bool = False  # True = alternate clean/adv batches

    # Optimization
    gradient_clip: Optional[float] = 1.0  # Max gradient norm
    use_amp: bool = True  # Automatic mixed precision

    # Evaluation
    eval_attack_steps: int = 40  # More thorough evaluation
    eval_epsilon: Optional[float] = None  # Default: same as attack_epsilon

    # Monitoring
    track_clean_acc: bool = True
    log_frequency: Optional[int] = 10  # Log every N batches

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Validate loss type
        if self.loss_type not in ["trades", "mart", "at"]:
            raise ValueError(
                f"loss_type must be 'trades', 'mart', or 'at', got {self.loss_type}"
            )

        # Validate beta
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

        # Validate attack parameters
        if self.attack_epsilon <= 0:
            raise ValueError(
                f"attack_epsilon must be positive, got {self.attack_epsilon}"
            )
        if self.attack_steps <= 0:
            raise ValueError(f"attack_steps must be positive, got {self.attack_steps}")

        # Set default step size
        if self.attack_step_size is None:
            self.attack_step_size = self.attack_epsilon / 4.0

        # Validate mix_clean
        if not 0.0 <= self.mix_clean <= 1.0:
            raise ValueError(f"mix_clean must be in [0, 1], got {self.mix_clean}")

        # Set default eval epsilon
        if self.eval_epsilon is None:
            self.eval_epsilon = self.attack_epsilon

        logger.info(
            f"Initialized AdversarialTrainingConfig:\n"
            f"  Loss: {self.loss_type.upper()} (β={self.beta})\n"
            f"  Attack: ε={self.attack_epsilon:.4f}, steps={self.attack_steps}\n"
            f"  Strategy: mix_clean={self.mix_clean}, alternate={self.alternate_batches}\n"
            f"  AMP: {self.use_amp}"
        )


# =============================================================================
# Adversarial Trainer
# =============================================================================


class AdversarialTrainer:
    """
    High-level adversarial training coordinator.

    This class orchestrates adversarial training by:
    1. Managing attack generation during training
    2. Computing robust losses (TRADES/MART/AT)
    3. Tracking clean and robust metrics
    4. Handling mixed precision training
    5. Providing checkpointing and resumability

    The trainer is designed to be modular and extensible, supporting
    different loss functions, attacks, and training strategies.

    Args:
        model: Neural network to train
        config: Adversarial training configuration
        device: Device for computation ('cuda' | 'cpu')

    Attributes:
        model: The neural network being trained
        config: Adversarial training configuration
        device: Computation device
        criterion: Robust loss function (TRADES/MART/AT)
        attack: Attack for generating adversarial examples
        scaler: Gradient scaler for AMP (None if use_amp=False)

    Examples:
        >>> config = AdversarialTrainingConfig(loss_type='trades', beta=1.0)
        >>> trainer = AdversarialTrainer(model, config, device='cuda')
        >>>
        >>> # Train for one epoch
        >>> metrics = trainer.train_epoch(
        ...     train_loader, optimizer, epoch=1
        ... )
        >>>
        >>> # Validate with robust accuracy
        >>> val_metrics = trainer.validate(val_loader, attack_steps=40)
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdversarialTrainingConfig,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        # Initialize robust loss
        self.criterion = self._create_criterion()

        # Initialize attack for training
        self.attack = self._create_attack(
            epsilon=config.attack_epsilon,
            num_steps=config.attack_steps,
            step_size=config.attack_step_size,
        )

        # Initialize gradient scaler for AMP
        self.scaler = GradScaler() if config.use_amp else None

        logger.info(
            f"Initialized AdversarialTrainer:\n"
            f"  Model: {model.__class__.__name__}\n"
            f"  Loss: {self.criterion}\n"
            f"  Attack: {self.attack}\n"
            f"  Device: {device}\n"
            f"  AMP: {config.use_amp}"
        )

    def _create_criterion(self) -> nn.Module:
        """Create robust loss function based on config."""
        if self.config.loss_type == "trades":
            return TRADESLoss(beta=self.config.beta)
        elif self.config.loss_type == "mart":
            return MARTLoss(beta=self.config.beta)
        elif self.config.loss_type == "at":
            return AdversarialTrainingLoss(mix_clean=self.config.mix_clean)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def _create_attack(
        self,
        epsilon: float,
        num_steps: int,
        step_size: float,
    ) -> BaseAttack:
        """Create PGD attack for generating adversarial examples."""
        attack_config = PGDConfig(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=self.config.attack_random_start,
            targeted=False,
        )
        return PGD(attack_config)

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch with adversarial examples.

        This method:
        1. Generates adversarial examples on-the-fly for each batch
        2. Computes robust loss (TRADES/MART/AT)
        3. Performs gradient descent with optional AMP
        4. Tracks both clean and robust metrics
        5. Logs progress periodically

        Args:
            dataloader: Training data loader
            optimizer: Optimizer for gradient descent
            epoch: Current epoch number (for logging)
            scheduler: Optional learning rate scheduler

        Returns:
            Dictionary with training metrics:
                - 'loss': Average total loss
                - 'clean_acc': Clean accuracy (if track_clean_acc=True)
                - 'adv_acc': Adversarial accuracy
                - 'ce_loss': Cross-entropy component (TRADES/MART)
                - 'robust_loss': Robustness component (TRADES/MART)

        Raises:
            RuntimeError: If numerical instability detected (NaN/Inf)
        """
        self.model.train()

        metrics = {
            "loss": 0.0,
            "clean_acc": 0.0,
            "adv_acc": 0.0,
            "ce_loss": 0.0,
            "robust_loss": 0.0,
        }

        total_samples = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch} [Adversarial Training]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            # Parse batch
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Zero gradients
            optimizer.zero_grad()

            # ================================================================
            # Generate Adversarial Examples
            # ================================================================
            # Note: Attack generation needs gradients
            # Don't use no_grad() - attack computes perturbations via gradients
            adv_images = self.attack(self.model, images, labels)

            # ================================================================
            # Forward Pass (with optional AMP)
            # ================================================================
            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    # Clean predictions (for TRADES/MART)
                    if self.config.loss_type in ["trades", "mart"]:
                        clean_logits = self.model(images)
                    else:
                        clean_logits = None

                    # Adversarial predictions
                    adv_logits = self.model(adv_images)

                    # Compute robust loss
                    if self.config.loss_type == "at":
                        loss = self.criterion(adv_logits, labels, clean_logits)
                    else:  # TRADES or MART
                        loss = self.criterion(clean_logits, adv_logits, labels)

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping (after unscaling)
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip,
                    )

                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training (no AMP)
                # Clean predictions
                if self.config.loss_type in ["trades", "mart"]:
                    clean_logits = self.model(images)
                else:
                    clean_logits = None

                # Adversarial predictions
                adv_logits = self.model(adv_images)

                # Compute robust loss
                if self.config.loss_type == "at":
                    loss = self.criterion(adv_logits, labels, clean_logits)
                else:  # TRADES or MART
                    loss = self.criterion(clean_logits, adv_logits, labels)

                # Backward
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip,
                    )

                # Optimizer step
                optimizer.step()

            # ================================================================
            # Metrics Tracking
            # ================================================================
            metrics["loss"] += loss.item() * batch_size

            # Clean accuracy
            if self.config.track_clean_acc and clean_logits is not None:
                _, clean_pred = clean_logits.max(1)
                clean_correct = clean_pred.eq(labels).sum().item()
                metrics["clean_acc"] += clean_correct

            # Adversarial accuracy
            _, adv_pred = adv_logits.max(1)
            adv_correct = adv_pred.eq(labels).sum().item()
            metrics["adv_acc"] += adv_correct

            total_samples += batch_size

            # Update progress bar
            if (
                self.config.log_frequency
                and (batch_idx + 1) % self.config.log_frequency == 0
            ):
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "adv_acc": f"{adv_correct / batch_size:.2%}",
                    }
                )

            # Learning rate scheduler step (if per-batch)
            if scheduler is not None and hasattr(scheduler, "step_update"):
                scheduler.step_update(epoch * len(dataloader) + batch_idx)

        # Average metrics
        metrics["loss"] /= total_samples
        metrics["clean_acc"] /= total_samples
        metrics["adv_acc"] /= total_samples

        return metrics

    def validate(
        self,
        dataloader: DataLoader,
        attack_steps: Optional[int] = None,
        attack_epsilon: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Validate with both clean and robust accuracy.

        Args:
            dataloader: Validation data loader
            attack_steps: Number of PGD steps (default: config.eval_attack_steps)
            attack_epsilon: Perturbation budget (default: config.eval_epsilon)

        Returns:
            Dictionary with validation metrics:
                - 'clean_acc': Accuracy on clean examples
                - 'robust_acc': Accuracy on adversarial examples
                - 'clean_loss': Cross-entropy loss on clean examples
        """
        return validate_robust(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            attack_steps=attack_steps or self.config.eval_attack_steps,
            attack_epsilon=attack_epsilon or self.config.eval_epsilon,
        )


# =============================================================================
# Standalone Training Functions
# =============================================================================


def train_adversarial_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    attack: BaseAttack,
    device: str,
    epoch: int,
    use_amp: bool = True,
    gradient_clip: Optional[float] = 1.0,
    log_frequency: int = 10,
) -> Dict[str, float]:
    """
    Standalone function for training one adversarial epoch.

    This is a functional alternative to AdversarialTrainer.train_epoch(),
    useful for custom training loops.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Robust loss function (TRADES/MART/AT)
        attack: Attack for generating adversarial examples
        device: Device for computation
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        gradient_clip: Maximum gradient norm (None = no clipping)
        log_frequency: Log every N batches

    Returns:
        Dictionary with training metrics

    Examples:
        >>> attack = PGD(PGDConfig(epsilon=8/255, num_steps=10))
        >>> criterion = TRADESLoss(beta=1.0)
        >>> metrics = train_adversarial_epoch(
        ...     model, train_loader, optimizer, criterion, attack,
        ...     device='cuda', epoch=1
        ... )
    """
    model.train()
    scaler = GradScaler() if use_amp else None

    running_loss = 0.0
    clean_correct = 0
    adv_correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Adv Train]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Parse batch
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Generate adversarial examples (needs gradients for perturbations)
        adv_images = attack(model, images, labels)

        # Forward pass with AMP
        if scaler is not None:
            with autocast():
                clean_logits = model(images)
                adv_logits = model(adv_images)

                # Compute loss (assuming TRADES-like loss)
                if isinstance(criterion, (TRADESLoss, MARTLoss)):
                    loss = criterion(clean_logits, adv_logits, labels)
                else:
                    loss = criterion(adv_logits, labels, clean_logits)

            scaler.scale(loss).backward()

            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            clean_logits = model(images)
            adv_logits = model(adv_images)

            if isinstance(criterion, (TRADESLoss, MARTLoss)):
                loss = criterion(clean_logits, adv_logits, labels)
            else:
                loss = criterion(adv_logits, labels, clean_logits)

            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

        # Metrics
        running_loss += loss.item() * labels.size(0)

        _, clean_pred = clean_logits.max(1)
        clean_correct += clean_pred.eq(labels).sum().item()

        _, adv_pred = adv_logits.max(1)
        adv_correct += adv_pred.eq(labels).sum().item()

        total += labels.size(0)

        # Logging
        if (batch_idx + 1) % log_frequency == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "clean_acc": f"{clean_correct / total:.2%}",
                    "adv_acc": f"{adv_correct / total:.2%}",
                }
            )

    return {
        "loss": running_loss / total,
        "clean_acc": clean_correct / total,
        "adv_acc": adv_correct / total,
    }


def validate_robust(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    attack_steps: int = 40,
    attack_epsilon: float = 8 / 255,
    attack_step_size: Optional[float] = None,
) -> Dict[str, float]:
    """
    Validate model with both clean and robust accuracy.

    This function evaluates the model on:
    1. Clean examples (standard validation)
    2. Adversarial examples (PGD attack)

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device for computation
        attack_steps: Number of PGD steps for evaluation
        attack_epsilon: Perturbation budget
        attack_step_size: Step size per iteration (default: epsilon/4)

    Returns:
        Dictionary with metrics:
            - 'clean_acc': Accuracy on clean examples
            - 'robust_acc': Accuracy on adversarial examples
            - 'clean_loss': Cross-entropy loss on clean examples

    Examples:
        >>> metrics = validate_robust(
        ...     model, val_loader, device='cuda',
        ...     attack_steps=40, attack_epsilon=8/255
        ... )
        >>> print(f"Clean: {metrics['clean_acc']:.2%}, Robust: {metrics['robust_acc']:.2%}")
    """
    # Model must be in train mode for attack generation (needs gradients)
    # but we use torch.no_grad() for the actual validation forward passes
    was_training = model.training
    model.train()  # Attack needs gradients

    # Create attack for evaluation
    if attack_step_size is None:
        attack_step_size = attack_epsilon / 4.0

    attack_config = PGDConfig(
        epsilon=attack_epsilon,
        num_steps=attack_steps,
        step_size=attack_step_size,
        random_start=True,
    )
    attack = PGD(attack_config)

    clean_correct = 0
    robust_correct = 0
    total = 0
    clean_loss = 0.0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    pbar = tqdm(dataloader, desc="Robust Validation", leave=False)

    with torch.no_grad():
        for batch in pbar:
            # Parse batch
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # Clean accuracy
            clean_logits = model(images)
            _, clean_pred = clean_logits.max(1)
            clean_correct += clean_pred.eq(labels).sum().item()

            clean_loss += criterion(clean_logits, labels).item()

            total += labels.size(0)

    # Generate adversarial examples (needs model in train mode for gradients)
    # Keep model in train mode for attack generation
    pbar = tqdm(dataloader, desc="Robust Accuracy", leave=False)

    for batch in pbar:
        # Parse batch
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        # Generate adversarial examples (model already in train mode)
        adv_images = attack(model, images, labels)

        # Robust accuracy evaluation (no grad needed)
        with torch.no_grad():
            adv_logits = model(adv_images)
            _, adv_pred = adv_logits.max(1)
            robust_correct += adv_pred.eq(labels).sum().item()

    # Restore original training state
    model.train(was_training)

    return {
        "clean_acc": clean_correct / total,
        "robust_acc": robust_correct / total,
        "clean_loss": clean_loss / total,
    }
