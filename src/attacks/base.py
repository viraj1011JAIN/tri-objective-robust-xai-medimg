"""
Base Attack Classes and Utilities
==================================

Provides abstract base classes and common utilities for adversarial attacks.

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """
    Base configuration for adversarial attacks.

    Attributes:
        epsilon: Maximum perturbation magnitude (L∞ norm)
        clip_min: Minimum pixel value (default: 0.0)
        clip_max: Maximum pixel value (default: 1.0)
        targeted: Whether to perform targeted attack
        device: Computation device ('cuda' or 'cpu')
        batch_size: Batch size for attack generation
        verbose: Enable verbose logging
        random_seed: Random seed for reproducibility
    """

    epsilon: float = 8.0 / 255.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    targeted: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    verbose: bool = True
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.clip_min >= self.clip_max:
            raise ValueError(
                f"clip_min ({self.clip_min}) must be < clip_max ({self.clip_max})"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "epsilon": self.epsilon,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "targeted": self.targeted,
            "device": self.device,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            "random_seed": self.random_seed,
        }


@dataclass
class AttackResult:
    """
    Results from an adversarial attack.

    Attributes:
        x_adv: Adversarial examples [B, C, H, W]
        success: Boolean mask indicating successful attacks [B]
        l2_dist: L2 distances per sample [B]
        linf_dist: L∞ distances per sample [B]
        pred_clean: Clean predictions [B]
        pred_adv: Adversarial predictions [B]
        time_elapsed: Time taken for attack generation
        iterations: Number of iterations used (if applicable)
        metadata: Additional attack-specific metadata
    """

    x_adv: torch.Tensor
    success: torch.Tensor
    l2_dist: torch.Tensor
    linf_dist: torch.Tensor
    pred_clean: torch.Tensor
    pred_adv: torch.Tensor
    time_elapsed: float
    iterations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate attack success rate."""
        return self.success.float().mean().item()

    @property
    def mean_l2(self) -> float:
        """Calculate mean L2 distance."""
        return self.l2_dist.mean().item()

    @property
    def mean_linf(self) -> float:
        """Calculate mean L∞ distance."""
        return self.linf_dist.mean().item()

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "success_rate": self.success_rate,
            "mean_l2_dist": self.mean_l2,
            "mean_linf_dist": self.mean_linf,
            "time_elapsed": self.time_elapsed,
            "iterations": self.iterations,
            **self.metadata,
        }


class BaseAttack(ABC, nn.Module):
    """
    Abstract base class for adversarial attacks.

    All attacks should inherit from this class and implement
    the `generate` method.
    """

    def __init__(self, config: AttackConfig, name: str = "BaseAttack"):
        """
        Initialize base attack.

        Args:
            config: Attack configuration
            name: Attack name for logging
        """
        super().__init__()
        self.config = config
        self.name = name
        self.device = torch.device(config.device)

        # Statistics tracking
        self.attack_count = 0
        self.success_count = 0
        self.total_time = 0.0

        # Set random seed
        torch.manual_seed(config.random_seed)

        if config.verbose:
            logger.info(f"{self.name} initialized on {self.device}")

    @abstractmethod
    def generate(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            model: Target model (should be in eval mode)
            x: Clean input images [B, C, H, W]
            y: True labels [B] for untargeted, or target labels for targeted
            **kwargs: Additional attack-specific parameters

        Returns:
            Adversarial examples [B, C, H, W]
        """
        pass

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        return_result: bool = False,
        **kwargs,
    ) -> torch.Tensor | AttackResult:
        """
        Forward pass that wraps generate() with timing and statistics.

        Args:
            model: Target model
            x: Clean images
            y: Labels
            return_result: If True, return AttackResult object
            **kwargs: Additional parameters

        Returns:
            Adversarial examples or AttackResult object
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Store original model mode
        was_training = model.training
        model.eval()

        # Get clean predictions
        with torch.no_grad():
            outputs = model(x)
            if isinstance(outputs, dict):
                logits_clean = outputs["logits"]
            else:
                logits_clean = outputs

            # Handle multi-class vs multi-label
            if y.ndim == 1:
                pred_clean = logits_clean.argmax(dim=1)
            else:
                pred_clean = (logits_clean > 0).float()

        # Generate adversarial examples with timing
        start_time = time.time()
        x_adv = self.generate(model, x, y, **kwargs)
        time_elapsed = time.time() - start_time

        # Get adversarial predictions
        with torch.no_grad():
            outputs_adv = model(x_adv)
            if isinstance(outputs_adv, dict):
                logits_adv = outputs_adv["logits"]
            else:
                logits_adv = outputs_adv

            # Handle multi-class vs multi-label
            if y.ndim == 1:
                # Multi-class
                pred_adv = logits_adv.argmax(dim=1)
                if self.config.targeted:
                    success = pred_adv == y
                else:
                    success = pred_adv != y
            else:
                # Multi-label (y is 2D)
                pred_adv = (logits_adv > 0).float()
                # Success if any label changes
                changes = (pred_adv != y).any(dim=1)
                if self.config.targeted:
                    # For targeted, success if all match
                    success = (pred_adv == y).all(dim=1)
                else:
                    # For untargeted, success if any change
                    success = changes

        # Calculate distances
        l2_dist = torch.norm((x_adv - x).view(x.size(0), -1), p=2, dim=1)
        linf_dist = torch.norm((x_adv - x).view(x.size(0), -1), p=float("inf"), dim=1)

        # Update statistics
        self.attack_count += x.size(0)
        self.success_count += success.sum().item()
        self.total_time += time_elapsed

        # Restore original model mode
        if was_training:
            model.train()

        if return_result:
            result = AttackResult(
                x_adv=x_adv.detach(),
                success=success,
                l2_dist=l2_dist,
                linf_dist=linf_dist,
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                time_elapsed=time_elapsed,
                metadata={"attack_name": self.name},
            )
            return result

        return x_adv.detach()

    def __call__(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Make attack callable."""
        return self.forward(model, x, y, return_result=False, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get attack statistics."""
        if self.attack_count == 0:
            return {"attack_count": 0, "success_rate": 0.0, "avg_time": 0.0}

        return {
            "attack_count": self.attack_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.attack_count,
            "total_time": self.total_time,
            "avg_time": self.total_time / self.attack_count,
        }

    def reset_statistics(self):
        """Reset attack statistics."""
        self.attack_count = 0
        self.success_count = 0
        self.total_time = 0.0

    @staticmethod
    def _infer_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> nn.Module:
        """
        Infer appropriate loss function from logits and labels.

        Args:
            logits: Model output logits
            labels: Target labels

        Returns:
            Loss function
        """
        if labels.dtype in (torch.int64, torch.int32, torch.int16):
            # Integer labels -> multi-class classification
            return nn.CrossEntropyLoss(reduction="mean")
        elif labels.ndim == logits.ndim:
            # Same shape -> multi-label or soft labels
            return nn.BCEWithLogitsLoss(reduction="mean")
        else:
            # Fallback to cross-entropy
            return nn.CrossEntropyLoss(reduction="mean")

    @staticmethod
    def project_linf(
        x_adv: torch.Tensor,
        x: torch.Tensor,
        epsilon: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> torch.Tensor:
        """
        Project onto L∞ ball and clip to valid range.

        Args:
            x_adv: Adversarial examples
            x: Original images
            epsilon: L∞ radius
            clip_min: Minimum pixel value
            clip_max: Maximum pixel value

        Returns:
            Projected adversarial examples
        """
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = x + delta
        x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
        return x_adv

    @staticmethod
    def project_l2(
        x_adv: torch.Tensor,
        x: torch.Tensor,
        epsilon: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> torch.Tensor:
        """
        Project onto L2 ball and clip to valid range.

        Args:
            x_adv: Adversarial examples
            x: Original images
            epsilon: L2 radius
            clip_min: Minimum pixel value
            clip_max: Maximum pixel value

        Returns:
            Projected adversarial examples
        """
        delta = x_adv - x
        delta = delta.view(delta.size(0), -1)
        norm = torch.norm(delta, p=2, dim=1, keepdim=True)
        factor = torch.min(torch.ones_like(norm), epsilon / (norm + 1e-12))
        delta = delta * factor
        delta = delta.view_as(x_adv)
        x_adv = x + delta
        x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
        return x_adv
