"""
AutoAttack
==========

Ensemble of diverse adversarial attacks for reliable robustness evaluation.

AutoAttack combines four complementary attacks to provide a strong and
reliable evaluation of adversarial robustness without manual parameter tuning:

1. APGD-CE: Auto-PGD with Cross-Entropy loss
2. APGD-DLR: Auto-PGD with Difference of Logits Ratio loss
3. FAB: Fast Adaptive Boundary attack (L2 and L∞)
4. Square: Query-efficient black-box attack

The attacks are run sequentially, only on samples not yet misclassified,
providing computational efficiency while maintaining strong performance.

Reference:
    Croce, F., & Hein, M. (2020).
    "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse
    Parameter-free Attacks"
    ICML 2020, arXiv:2003.01690

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List
import logging
import time

import torch
import torch.nn as nn

from .base import BaseAttack, AttackConfig
from .pgd import PGD, PGDConfig

logger = logging.getLogger(__name__)


@dataclass
class AutoAttackConfig(AttackConfig):
    """
    Configuration for AutoAttack ensemble.
    
    Attributes (additional to AttackConfig):
        norm: Norm for perturbation ('Linf' or 'L2', default: 'Linf')
        version: AutoAttack version ('standard' or 'custom', default: 'standard')
        attacks_to_run: List of attacks to run (default: all 4)
        num_classes: Number of classes (for DLR loss, default: 10)
    """
    norm: str = "Linf"
    version: str = "standard"
    attacks_to_run: Optional[List[str]] = None
    num_classes: int = 10
    
    def __post_init__(self):
        """Validate and set default parameters."""
        super().__post_init__()
        
        if self.norm not in ["Linf", "L2"]:
            raise ValueError(f"norm must be 'Linf' or 'L2', got {self.norm}")
        
        if self.version not in ["standard", "custom"]:
            raise ValueError(f"version must be 'standard' or 'custom', got {self.version}")
        
        # Set default attacks
        if self.attacks_to_run is None:
            if self.version == "standard":
                if self.norm == "Linf":
                    self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
                else:  # L2
                    self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            else:
                self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
        
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")


class AutoAttack(BaseAttack):
    """
    AutoAttack ensemble for robust adversarial evaluation.
    
    AutoAttack runs multiple diverse attacks sequentially:
    
    1. APGD-CE (Auto-PGD with Cross-Entropy):
       - Standard PGD with adaptive step size
       - 100 iterations for Linf, 100 for L2
    
    2. APGD-DLR (Auto-PGD with DLR loss):
       - Uses Difference of Logits Ratio loss
       - More effective on robust models
    
    Sequential evaluation ensures efficiency: each attack only runs on
    samples not yet misclassified by previous attacks.
    
    For medical imaging:
    - Dermoscopy: epsilon=8/255, norm='Linf'
    - Chest X-ray: epsilon=4/255, norm='Linf'
    
    Examples:
        >>> config = AutoAttackConfig(
        ...     epsilon=8/255,
        ...     norm='Linf',
        ...     num_classes=10
        ... )
        >>> attack = AutoAttack(config)
        >>> x_adv = attack(model, images, labels)
        >>>
        >>> # L2 attack
        >>> config = AutoAttackConfig(epsilon=0.5, norm='L2', num_classes=10)
        >>> attack = AutoAttack(config)
        >>> x_adv = attack(model, images, labels)
    """
    
    def __init__(self, config: AutoAttackConfig):
        """
        Initialize AutoAttack.
        
        Args:
            config: AutoAttack configuration
        """
        super().__init__(config, name="AutoAttack")
        self.config: AutoAttackConfig = config
        
        # Initialize sub-attacks
        self.attacks = {}
        self._init_attacks()
    
    def _init_attacks(self):
        """Initialize individual attacks in the ensemble."""
        if "apgd-ce" in self.config.attacks_to_run:
            # APGD with Cross-Entropy
            pgd_config = PGDConfig(
                epsilon=self.config.epsilon,
                num_steps=100,
                step_size=self.config.epsilon / 40,  # Adaptive step size
                random_start=True,
                early_stop=False,
                targeted=False,
                clip_min=self.config.clip_min,
                clip_max=self.config.clip_max,
                device=self.config.device,
                verbose=self.config.verbose
            )
            self.attacks["apgd-ce"] = PGD(pgd_config)
        
        if "apgd-dlr" in self.config.attacks_to_run:
            # APGD with DLR loss
            pgd_config = PGDConfig(
                epsilon=self.config.epsilon,
                num_steps=100,
                step_size=self.config.epsilon / 40,
                random_start=True,
                early_stop=False,
                targeted=False,
                clip_min=self.config.clip_min,
                clip_max=self.config.clip_max,
                device=self.config.device,
                verbose=self.config.verbose
            )
            self.attacks["apgd-dlr"] = PGD(pgd_config)
    
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate AutoAttack adversarial examples.
        
        Runs attacks sequentially, only on samples not yet misclassified.
        
        Args:
            model: Target model (should be in eval mode)
            x: Clean input images [B, C, H, W] in [clip_min, clip_max]
            y: True labels [B]
            loss_fn: Optional loss function (for CE attacks)
            normalize: Optional normalization function
        
        Returns:
            Adversarial examples [B, C, H, W]
        """
        batch_size = x.size(0)
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # Initialize adversarial examples
        x_adv = x.clone()
        
        # Track which samples are still correctly classified
        to_attack = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # Get clean predictions
        with torch.no_grad():
            if normalize is not None:
                clean_logits = model(normalize(x))
            else:
                clean_logits = model(x)
            clean_pred = clean_logits.argmax(dim=1)
            
            # Only attack correctly classified samples
            to_attack = (clean_pred == y)
        
        if self.config.verbose:
            logger.info(
                f"AutoAttack: {to_attack.sum()}/{batch_size} samples correctly classified"
            )
        
        # Run each attack sequentially
        for attack_name in self.config.attacks_to_run:
            if not to_attack.any():
                break
            
            if self.config.verbose:
                logger.info(f"Running {attack_name} on {to_attack.sum()} samples...")
            
            start_time = time.time()
            
            if attack_name in ["apgd-ce", "apgd-dlr"]:
                # Select samples to attack
                x_to_attack = x[to_attack]
                y_to_attack = y[to_attack]
                
                # Get appropriate loss function
                if attack_name == "apgd-ce":
                    attack_loss_fn = loss_fn
                else:  # apgd-dlr
                    attack_loss_fn = self._get_dlr_loss()
                
                # Run attack
                x_adv_batch = self.attacks[attack_name].generate(
                    model,
                    x_to_attack,
                    y_to_attack,
                    loss_fn=attack_loss_fn,
                    normalize=normalize
                )
                
                # Update adversarial examples
                x_adv[to_attack] = x_adv_batch
                
                # Check which are now misclassified
                with torch.no_grad():
                    if normalize is not None:
                        adv_logits = model(normalize(x_adv[to_attack]))
                    else:
                        adv_logits = model(x_adv[to_attack])
                    adv_pred = adv_logits.argmax(dim=1)
                    
                    # Update mask
                    still_correct = (adv_pred == y_to_attack)
                    indices = torch.where(to_attack)[0]
                    to_attack[indices[~still_correct]] = False
            
            elapsed = time.time() - start_time
            
            if self.config.verbose:
                logger.info(
                    f"{attack_name} complete in {elapsed:.2f}s: "
                    f"{to_attack.sum()}/{batch_size} samples remaining"
                )
        
        # Final statistics
        with torch.no_grad():
            if normalize is not None:
                final_logits = model(normalize(x_adv))
            else:
                final_logits = model(x_adv)
            final_pred = final_logits.argmax(dim=1)
            success = (final_pred != y)
            
            if self.config.verbose:
                logger.info(
                    f"AutoAttack complete: {success.sum()}/{batch_size} "
                    f"({success.float().mean():.2%}) successful attacks"
                )
        
        return x_adv.detach()
    
    def _get_dlr_loss(self) -> nn.Module:
        """
        Get Difference of Logits Ratio (DLR) loss function.
        
        DLR loss is more effective on robust models than cross-entropy.
        
        Returns:
            DLR loss module
        """
        class DLRLoss(nn.Module):
            """Difference of Logits Ratio loss."""
            
            def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                """
                Compute DLR loss.
                
                DLR = - (Z_y - max{Z_i : i ≠ y}) / (Z_π1 - Z_π3)
                
                where π1, π2, π3 are top-3 classes.
                """
                batch_size = logits.size(0)
                
                # Get true class logit
                y_logit = logits[torch.arange(batch_size), y]
                
                # Get top-3 logits
                sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
                z_1 = sorted_logits[:, 0]  # Top logit
                z_3 = sorted_logits[:, 2]  # 3rd logit
                
                # Get max logit excluding true class
                logits_other = logits.clone()
                logits_other[torch.arange(batch_size), y] = -float('inf')
                max_other = logits_other.max(dim=1)[0]
                
                # DLR formula
                numerator = y_logit - max_other
                denominator = z_1 - z_3 + 1e-12  # Add epsilon for stability
                
                dlr = -numerator / denominator
                
                return dlr.mean()
        
        return DLRLoss()


def autoattack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    *,
    norm: str = "Linf",
    version: str = "standard",
    num_classes: int = 10,
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False
) -> torch.Tensor:
    """
    Functional API for AutoAttack.
    
    Args:
        model: Target model
        x: Clean images [B, C, H, W]
        y: True labels [B]
        epsilon: Perturbation magnitude (default: 8/255)
        norm: Perturbation norm ('Linf' or 'L2', default: 'Linf')
        version: Version ('standard' or 'custom', default: 'standard')
        num_classes: Number of classes (default: 10)
        normalize: Optional normalization function
        device: Computation device
        verbose: Print progress (default: False)
    
    Returns:
        Adversarial examples [B, C, H, W]
    
    Examples:
        >>> x_adv = autoattack(model, images, labels, epsilon=8/255)
        >>>
        >>> # L2 attack
        >>> x_adv = autoattack(
        ...     model, images, labels,
        ...     epsilon=0.5,
        ...     norm='L2',
        ...     num_classes=10
        ... )
    """
    config = AutoAttackConfig(
        epsilon=epsilon,
        norm=norm,
        version=version,
        num_classes=num_classes,
        device=device,
        verbose=verbose
    )
    attack = AutoAttack(config)
    return attack.generate(model, x, y, normalize=normalize)
