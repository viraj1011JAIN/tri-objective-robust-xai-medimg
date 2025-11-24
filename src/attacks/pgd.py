"""
Projected Gradient Descent (PGD)
=================================

Multi-step iterative adversarial attack for L∞ norm.

PGD is a stronger variant of FGSM that takes multiple smaller steps,
projecting onto the L∞ ball after each step:

    x_{t+1} = Π_{x + S}(x_t + α · sign(∇_x L(θ, x_t, y)))

where:
- x is the clean input
- S is the L∞ ball of radius ε
- α is the step size
- Π projects onto the constraint set
- The process repeats for T iterations

PGD with random initialization is considered one of the strongest
first-order adversaries for adversarial training.

Reference:
    Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
    "Towards Deep Learning Models Resistant to Adversarial Attacks"
    ICLR 2018, arXiv:1706.06083

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from .base import AttackConfig, BaseAttack

logger = logging.getLogger(__name__)


@dataclass
class PGDConfig(AttackConfig):
    """
    Configuration for PGD attack.

    Attributes (additional to AttackConfig):
        num_steps: Number of PGD iterations (default: 40)
        step_size: Step size per iteration (default: epsilon/4)
        random_start: Whether to start from random perturbation (default: True)
        early_stop: Stop if all examples are misclassified (default: False)
    """

    num_steps: int = 40
    step_size: Optional[float] = None  # If None, defaults to epsilon/4
    random_start: bool = True
    early_stop: bool = False

    def __post_init__(self):
        """Validate and set default parameters."""
        super().__post_init__()

        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

        # Set default step size
        if self.step_size is None:
            self.step_size = self.epsilon / 4.0

        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")


class PGD(BaseAttack):
    """
    Projected Gradient Descent attack.

    PGD is an iterative extension of FGSM that:
    1. Optionally starts from a random perturbation (random_start=True)
    2. Takes multiple small steps in the gradient direction
    3. Projects back onto the L∞ ball after each step
    4. Optionally stops early if all examples are misclassified

    PGD is stronger than FGSM and is commonly used in adversarial training.

    For medical imaging:
    - Dermoscopy: epsilon=8/255, steps=40, step_size=2/255, random_start=True
    - Chest X-ray: epsilon=4/255, steps=40, step_size=1/255, random_start=True

    Examples:
        >>> config = PGDConfig(
        ...     epsilon=8/255,
        ...     num_steps=40,
        ...     step_size=2/255,
        ...     random_start=True
        ... )
        >>> attack = PGD(config)
        >>> x_adv = attack(model, images, labels)
        >>>
        >>> # With early stopping
        >>> config = PGDConfig(epsilon=8/255, early_stop=True)
        >>> attack = PGD(config)
        >>> x_adv = attack(model, images, labels)
    """

    def __init__(self, config: PGDConfig):
        """
        Initialize PGD attack.

        Args:
            config: PGD configuration
        """
        super().__init__(config, name="PGD")
        self.config: PGDConfig = config

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Generate PGD adversarial examples.

        Args:
            model: Target model (should be in eval mode)
            x: Clean input images [B, C, H, W] in [clip_min, clip_max]
            y: True labels [B] for untargeted, target labels for targeted
            loss_fn: Optional loss function. If None, inferred from data
            normalize: Optional normalization function applied before model

        Returns:
            Adversarial examples [B, C, H, W]
        """
        if self.config.epsilon <= 0:
            return x.detach()

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        # Initialize adversarial examples
        if self.config.random_start:
            # Start from random point in L∞ ball
            delta = torch.empty_like(x).uniform_(
                -self.config.epsilon, self.config.epsilon
            )
            x_adv = torch.clamp(
                x + delta, min=self.config.clip_min, max=self.config.clip_max
            )
        else:
            x_adv = x.clone()

        # Infer loss function if not provided
        if loss_fn is None:
            # Forward pass to get logits shape
            with torch.no_grad():
                if normalize is not None:
                    outputs = model(normalize(x))
                else:
                    outputs = model(x)
                # Handle dict outputs
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                loss_fn = self._infer_loss_fn(logits, y)

        # Track success for early stopping
        if self.config.early_stop:
            success_mask = torch.zeros(len(x), dtype=torch.bool, device=self.device)

        # PGD iterations
        for step in range(self.config.num_steps):
            x_adv.requires_grad = True

            # Forward pass
            if normalize is not None:
                outputs = model(normalize(x_adv))
            else:
                outputs = model(x_adv)

            # Handle dict outputs
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            # Compute loss
            loss = loss_fn(logits, y)

            # Reverse loss for targeted attacks
            if self.config.targeted:
                loss = -loss

            # Compute gradient
            model.zero_grad(set_to_none=True)
            loss.backward()

            grad_sign = x_adv.grad.detach().sign()

            # Take step
            x_adv = x_adv.detach() + self.config.step_size * grad_sign

            # Project onto L∞ ball around x
            x_adv = self.project_linf(
                x_adv,
                x,
                self.config.epsilon,
                self.config.clip_min,
                self.config.clip_max,
            )

            # Early stopping check
            if self.config.early_stop:
                with torch.no_grad():
                    if normalize is not None:
                        pred = model(normalize(x_adv)).argmax(dim=1)
                    else:
                        pred = model(x_adv).argmax(dim=1)

                    if self.config.targeted:
                        success_mask |= pred == y
                    else:
                        success_mask |= pred != y

                    if success_mask.all():
                        if self.config.verbose:
                            logger.info(
                                f"PGD early stop at step {step+1}/{self.config.num_steps}"
                            )
                        break

        return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    *,
    num_steps: int = 40,
    step_size: Optional[float] = None,
    random_start: bool = True,
    loss_fn: Optional[nn.Module] = None,
    targeted: bool = False,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Functional API for PGD attack.

    Args:
        model: Target model
        x: Clean images [B, C, H, W]
        y: Labels [B]
        epsilon: Perturbation magnitude (default: 8/255)
        num_steps: Number of iterations (default: 40)
        step_size: Step size (default: epsilon/4)
        random_start: Random initialization (default: True)
        loss_fn: Optional loss function
        targeted: Whether to perform targeted attack
        clip_min: Minimum pixel value (default: 0.0)
        clip_max: Maximum pixel value (default: 1.0)
        normalize: Optional normalization function
        device: Computation device

    Returns:
        Adversarial examples [B, C, H, W]

    Examples:
        >>> x_adv = pgd_attack(model, images, labels, epsilon=8/255)
        >>>
        >>> # Custom step size and iterations
        >>> x_adv = pgd_attack(
        ...     model, images, labels,
        ...     epsilon=8/255,
        ...     num_steps=100,
        ...     step_size=1/255
        ... )
    """
    config = PGDConfig(
        epsilon=epsilon,
        num_steps=num_steps,
        step_size=step_size,
        random_start=random_start,
        targeted=targeted,
        clip_min=clip_min,
        clip_max=clip_max,
        device=device,
        verbose=False,
    )
    attack = PGD(config)
    return attack.generate(model, x, y, loss_fn=loss_fn, normalize=normalize)
