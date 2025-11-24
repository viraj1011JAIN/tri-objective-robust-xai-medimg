"""
Carlini & Wagner (C&W) L2 Attack
=================================

Optimization-based adversarial attack minimizing L2 distance.

The C&W attack formulates adversarial example generation as an optimization
problem that minimizes the L2 distance while ensuring misclassification:

    minimize ||δ||_2 + c · f(x + δ)
    
where:
- δ is the perturbation
- c is a penalty parameter (tuned via binary search)
- f is an objective function ensuring misclassification
- x + δ is parameterized in tanh space for box constraints

The attack uses the logit difference objective:
    f(x') = max(max{Z(x')_i : i ≠ t} - Z(x')_t, -κ)

where Z(x') are the logits, t is the target class, and κ is the confidence.

Reference:
    Carlini, N., & Wagner, D. (2017).
    "Towards Evaluating the Robustness of Neural Networks"
    IEEE S&P 2017, arXiv:1608.04644

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseAttack, AttackConfig

logger = logging.getLogger(__name__)


@dataclass
class CWConfig(AttackConfig):
    """
    Configuration for C&W L2 attack.
    
    Attributes (additional to AttackConfig):
        confidence: Confidence parameter κ (default: 0.0)
        learning_rate: Adam optimizer learning rate (default: 0.01)
        max_iterations: Maximum optimization iterations (default: 1000)
        binary_search_steps: Binary search steps for c (default: 9)
        initial_c: Initial value of c (default: 1e-3)
        abort_early: Abort if loss increases (default: True)
    """
    confidence: float = 0.0
    learning_rate: float = 0.01
    max_iterations: int = 1000
    binary_search_steps: int = 9
    initial_c: float = 1e-3
    abort_early: bool = True
    
    def __post_init__(self):
        """Validate parameters."""
        super().__post_init__()
        
        if self.confidence < 0:
            raise ValueError(f"confidence must be >= 0, got {self.confidence}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations}")
        if self.binary_search_steps < 0:
            raise ValueError(
                f"binary_search_steps must be >= 0, got {self.binary_search_steps}"
            )
        if self.initial_c <= 0:
            raise ValueError(f"initial_c must be > 0, got {self.initial_c}")


class CarliniWagner(BaseAttack):
    """
    Carlini & Wagner L2 attack.
    
    C&W is an optimization-based attack that finds minimal L2 perturbations
    by solving an optimization problem. It uses:
    
    1. Tanh-space parameterization to handle box constraints automatically
    2. Binary search over penalty parameter c
    3. Adam optimizer for efficient optimization
    4. Early abort if loss stops decreasing
    
    C&W typically produces imperceptible perturbations with high success rates,
    making it one of the strongest attacks for robustness evaluation.
    
    For medical imaging:
    - Start with default parameters
    - Increase max_iterations for higher quality (e.g., 10000)
    - Adjust confidence for stronger attacks (e.g., κ=20)
    
    Examples:
        >>> config = CWConfig(
        ...     confidence=0,
        ...     max_iterations=1000,
        ...     binary_search_steps=9
        ... )
        >>> attack = CarliniWagner(config)
        >>> x_adv = attack(model, images, labels)
        >>>
        >>> # High confidence attack
        >>> config = CWConfig(confidence=20, max_iterations=5000)
        >>> attack = CarliniWagner(config)
        >>> x_adv = attack(model, images, labels)
    """
    
    def __init__(self, config: CWConfig):
        """
        Initialize C&W attack.
        
        Args:
            config: C&W configuration
        """
        super().__init__(config, name="C&W-L2")
        self.config: CWConfig = config
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate C&W adversarial examples.
        
        Args:
            model: Target model (should be in eval mode)
            x: Clean input images [B, C, H, W] in [clip_min, clip_max]
            y: True labels [B] for untargeted, target labels for targeted
            loss_fn: Not used (C&W uses logit-based objective)
            normalize: Optional normalization function applied before model
        
        Returns:
            Adversarial examples [B, C, H, W]
        """
        batch_size = x.size(0)
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # Initialize tracking variables
        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Binary search bounds for c
        c_lower = torch.zeros(batch_size, device=self.device)
        c_upper = torch.full((batch_size,), 1e10, device=self.device)
        c = torch.full((batch_size,), self.config.initial_c, device=self.device)
        
        # Tanh-space inverse mapping
        # x = (tanh(w) + 1) / 2 * (clip_max - clip_min) + clip_min
        # w = atanh((x - clip_min) / (clip_max - clip_min) * 2 - 1)
        x_scaled = (x - self.config.clip_min) / (self.config.clip_max - self.config.clip_min)
        x_scaled = torch.clamp(x_scaled, 1e-6, 1 - 1e-6)  # Avoid atanh boundaries
        w = torch.atanh(x_scaled * 2 - 1)
        
        # Binary search over c
        for binary_step in range(self.config.binary_search_steps):
            # Optimization variable (perturbation in tanh space)
            delta_w = torch.zeros_like(w, requires_grad=True)
            
            # Adam optimizer
            optimizer = optim.Adam([delta_w], lr=self.config.learning_rate)
            
            # Track best result for this c value
            best_loss = torch.full((batch_size,), float('inf'), device=self.device)
            best_delta = torch.zeros_like(x)
            
            # Optimization loop
            for iteration in range(self.config.max_iterations):
                optimizer.zero_grad()
                
                # Map from tanh space to input space
                x_adv = torch.tanh(w + delta_w)
                x_adv = (x_adv + 1) / 2 * (self.config.clip_max - self.config.clip_min)
                x_adv = x_adv + self.config.clip_min
                
                # Forward pass
                if normalize is not None:
                    logits = model(normalize(x_adv))
                else:
                    logits = model(x_adv)
                
                # Compute L2 distance
                l2_dist = (x_adv - x).view(batch_size, -1).norm(p=2, dim=1)
                
                # Compute logit-based objective
                f = self._logit_objective(logits, y)
                
                # Total loss: L2 + c * f
                loss_per_sample = l2_dist + c * f
                loss = loss_per_sample.mean()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update best adversarial examples
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    if self.config.targeted:
                        success = (pred == y)
                    else:
                        success = (pred != y)
                    
                    # Update best if successful and lower L2
                    improved = success & (l2_dist < best_l2)
                    best_adv[improved] = x_adv[improved].clone()
                    best_l2[improved] = l2_dist[improved]
                
                # Early abort if loss increases
                if self.config.abort_early and iteration % 100 == 0 and iteration > 0:
                    if (loss_per_sample > best_loss).all():
                        if self.config.verbose:
                            logger.info(f"C&W early abort at iteration {iteration}")
                        break
                    best_loss = torch.minimum(best_loss, loss_per_sample)
            
            # Binary search update for c
            with torch.no_grad():
                if normalize is not None:
                    final_logits = model(normalize(best_adv))
                else:
                    final_logits = model(best_adv)
                
                pred = final_logits.argmax(dim=1)
                if self.config.targeted:
                    success = (pred == y)
                else:
                    success = (pred != y)
                
                # Update c bounds
                c_upper[success] = torch.minimum(c_upper[success], c[success])
                c_lower[~success] = torch.maximum(c_lower[~success], c[~success])
                
                # Set new c (geometric mean of bounds)
                c[success] = (c_lower[success] + c_upper[success]) / 2
                c[~success] = (c_lower[~success] + c_upper[~success]) / 2
            
            if self.config.verbose:
                logger.info(
                    f"Binary search step {binary_step+1}/{self.config.binary_search_steps}: "
                    f"Success rate = {success.float().mean():.2%}"
                )
        
        return best_adv.detach()
    
    def _logit_objective(
        self,
        logits: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logit-based objective function.
        
        For untargeted: f(x') = max(Z(x')_y - max{Z(x')_i : i ≠ y}, -κ)
        For targeted:   f(x') = max(max{Z(x')_i : i ≠ t} - Z(x')_t, -κ)
        
        Args:
            logits: Model logits [B, num_classes]
            y: Labels [B]
        
        Returns:
            Objective values [B]
        """
        batch_size, num_classes = logits.shape
        
        # Create one-hot encoding
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        
        # Get target class logit
        target_logit = (logits * y_onehot).sum(dim=1)
        
        # Get max logit of other classes
        other_logits = logits - y_onehot * 1e9  # Mask target class
        max_other_logit = other_logits.max(dim=1)[0]
        
        if self.config.targeted:
            # Want max_other > target
            f = torch.clamp(
                max_other_logit - target_logit,
                min=-self.config.confidence
            )
        else:
            # Want target > max_other
            f = torch.clamp(
                target_logit - max_other_logit,
                min=-self.config.confidence
            )
        
        return f


def cw_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    confidence: float = 0.0,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    binary_search_steps: int = 9,
    targeted: bool = False,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Functional API for C&W L2 attack.
    
    Args:
        model: Target model
        x: Clean images [B, C, H, W]
        y: Labels [B]
        confidence: Confidence parameter κ (default: 0.0)
        learning_rate: Adam learning rate (default: 0.01)
        max_iterations: Max optimization iterations (default: 1000)
        binary_search_steps: Binary search steps (default: 9)
        targeted: Whether to perform targeted attack
        clip_min: Minimum pixel value (default: 0.0)
        clip_max: Maximum pixel value (default: 1.0)
        normalize: Optional normalization function
        device: Computation device
    
    Returns:
        Adversarial examples [B, C, H, W]
    
    Examples:
        >>> x_adv = cw_attack(model, images, labels)
        >>>
        >>> # High confidence attack
        >>> x_adv = cw_attack(
        ...     model, images, labels,
        ...     confidence=20,
        ...     max_iterations=5000
        ... )
    """
    config = CWConfig(
        confidence=confidence,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        binary_search_steps=binary_search_steps,
        targeted=targeted,
        clip_min=clip_min,
        clip_max=clip_max,
        device=device,
        verbose=False
    )
    attack = CarliniWagner(config)
    return attack.generate(model, x, y, normalize=normalize)
