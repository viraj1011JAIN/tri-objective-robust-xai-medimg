"""
Fast Gradient Sign Method (FGSM)
=================================

Single-step gradient-based adversarial attack for L∞ norm.

FGSM generates adversarial examples by taking a single step in the direction
of the gradient of the loss with respect to the input:

    x_adv = x + ε · sign(∇_x L(θ, x, y))

where:
- x is the clean input
- ε is the perturbation magnitude (epsilon)
- L is the loss function
- θ are the model parameters
- y is the true label (untargeted) or target label (targeted)

Reference:
    Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
    "Explaining and Harnessing Adversarial Examples"
    ICLR 2015, arXiv:1412.6572

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable
import logging

import torch
import torch.nn as nn

from .base import BaseAttack, AttackConfig

logger = logging.getLogger(__name__)


@dataclass
class FGSMConfig(AttackConfig):
    """
    Configuration for FGSM attack.
    
    Inherits all parameters from AttackConfig.
    No additional FGSM-specific parameters needed.
    """
    pass


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method attack.
    
    FGSM is a simple yet effective single-step attack that adds perturbations
    in the direction of the gradient sign. Despite its simplicity, it remains
    a strong baseline for adversarial robustness evaluation.
    
    For medical imaging:
    - Dermoscopy (RGB): epsilon typically in [2/255, 4/255, 8/255]
    - Chest X-ray (grayscale): epsilon typically in [2/255, 4/255]
    
    Examples:
        >>> config = FGSMConfig(epsilon=8/255)
        >>> attack = FGSM(config)
        >>> x_adv = attack(model, images, labels)
        >>> 
        >>> # With normalization
        >>> normalize = transforms.Normalize(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> x_adv = attack(model, images, labels, normalize=normalize)
    """
    
    def __init__(self, config: FGSMConfig):
        """
        Initialize FGSM attack.
        
        Args:
            config: FGSM configuration
        """
        super().__init__(config, name="FGSM")
        self.config: FGSMConfig = config
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate FGSM adversarial examples.
        
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
        
        # Enable gradient computation
        x.requires_grad = True
        
        # Forward pass (with optional normalization)
        if normalize is not None:
            logits = model(normalize(x))
        else:
            logits = model(x)
        
        # Infer loss function if not provided
        if loss_fn is None:
            loss_fn = self._infer_loss_fn(logits, y)
        
        # Compute loss
        loss = loss_fn(logits, y)
        
        # Reverse loss for targeted attacks
        if self.config.targeted:
            loss = -loss
        
        # Compute gradient
        model.zero_grad(set_to_none=True)
        loss.backward()
        
        # Get gradient sign
        grad_sign = x.grad.detach().sign()
        
        # Generate adversarial examples
        x_adv = x + self.config.epsilon * grad_sign
        
        # Project onto valid range
        x_adv = torch.clamp(
            x_adv,
            min=self.config.clip_min,
            max=self.config.clip_max
        )
        
        return x_adv.detach()


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    *,
    loss_fn: Optional[nn.Module] = None,
    targeted: bool = False,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Functional API for FGSM attack.
    
    Convenience function for one-off FGSM attacks without creating
    an attack object.
    
    Args:
        model: Target model
        x: Clean images [B, C, H, W]
        y: Labels [B]
        epsilon: Perturbation magnitude (default: 8/255)
        loss_fn: Optional loss function
        targeted: Whether to perform targeted attack
        clip_min: Minimum pixel value (default: 0.0)
        clip_max: Maximum pixel value (default: 1.0)
        normalize: Optional normalization function
        device: Computation device
    
    Returns:
        Adversarial examples [B, C, H, W]
    
    Examples:
        >>> x_adv = fgsm_attack(model, images, labels, epsilon=8/255)
        >>> 
        >>> # Targeted attack
        >>> x_adv = fgsm_attack(
        ...     model, images, target_labels,
        ...     epsilon=4/255,
        ...     targeted=True
        ... )
    """
    config = FGSMConfig(
        epsilon=epsilon,
        targeted=targeted,
        clip_min=clip_min,
        clip_max=clip_max,
        device=device,
        verbose=False
    )
    attack = FGSM(config)
    return attack.generate(model, x, y, loss_fn=loss_fn, normalize=normalize)
