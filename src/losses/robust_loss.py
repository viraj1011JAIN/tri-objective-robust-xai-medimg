"""
Robust Loss Functions for Adversarial Training
===============================================

Implementation of state-of-the-art adversarial training loss functions:
1. TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
2. MART (Misclassification Aware adversarial tRaining)
3. Standard Adversarial Training (AT) loss

These losses enable robust model training by balancing:
- Clean accuracy (standard cross-entropy)
- Adversarial robustness (perturbation resistance)
- Misclassification awareness (MART-specific)

Mathematical Formulations:
--------------------------

**TRADES Loss** (Zhang et al., 2019):
    L_TRADES = L_CE(f(x), y) + β · KL(f(x) || f(x_adv))

    where:
    - L_CE: standard cross-entropy on clean examples
    - KL: Kullback-Leibler divergence between clean and adversarial predictions
    - β: tradeoff parameter (controls robustness vs. accuracy)
    - x_adv: adversarial example generated via PGD

**MART Loss** (Wang et al., 2020):
    L_MART = L_CE(f(x), y) + β · BCE(f(x_adv), y) · (1 - p_y(x))

    where:
    - BCE: binary cross-entropy with target class
    - p_y(x): clean prediction probability for true class
    - (1 - p_y(x)): misclassification weight (higher for harder examples)

**Standard AT Loss**:
    L_AT = L_CE(f(x_adv), y)

    Simple adversarial training on perturbed examples only.

Design Principles:
------------------
1. **Type Safety**: All tensors properly typed with shape documentation
2. **Numerical Stability**: Epsilon clipping, log-softmax for KL divergence
3. **Flexibility**: Configurable parameters (β, temperature, reduction)
4. **Performance**: Efficient PyTorch operations, no loops
5. **Debugging**: Comprehensive assertions and logging
6. **Medical Imaging**: Tested on dermoscopy (ISIC) and chest X-ray (CXR)

Clinical Relevance:
-------------------
For medical imaging models:
- Clean accuracy ensures correct diagnosis on standard images
- Adversarial robustness protects against:
  * Sensor noise (lighting, camera variations)
  * Preprocessing artifacts (JPEG compression, normalization errors)
  * Intentional perturbations (adversarial attacks)
- TRADES provides tunable balance (β) between accuracy and robustness
- MART focuses robustness training on misclassified examples

References:
-----------
[1] Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019).
    "Theoretically Principled Trade-off between Robustness and Accuracy"
    ICML 2019, arXiv:1901.08573

[2] Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020).
    "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"
    ICLR 2020, arXiv:1911.05673

[3] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
    "Towards Deep Learning Models Resistant to Adversarial Attacks"
    ICLR 2018, arXiv:1706.06083

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Date: November 24, 2025
Version: 5.1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "TRADESLoss",
    "MARTLoss",
    "AdversarialTrainingLoss",
    "trades_loss",
    "mart_loss",
    "adversarial_training_loss",
]


# =============================================================================
# TRADES Loss (Zhang et al., 2019)
# =============================================================================


class TRADESLoss(nn.Module):
    """
    TRADES loss for adversarial training with tunable robustness/accuracy tradeoff.

    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    minimizes:
        L = L_CE(f(x), y) + β · KL(f(x) || f(x_adv))

    This formulation:
    - Maintains clean accuracy via L_CE on clean examples
    - Encourages consistent predictions between clean and adversarial via KL
    - Avoids over-fitting to adversarial examples (unlike standard AT)
    - Provides tunable tradeoff via β parameter

    Key Advantages:
    ---------------
    1. **Theoretical Foundation**: Derived from principled optimization
    2. **Better Tradeoff**: Less clean accuracy drop than standard AT
    3. **Stable Training**: KL divergence is smoother than cross-entropy
    4. **Medical Imaging**: Preserves diagnostic accuracy while adding robustness

    Hyperparameter Guidelines:
    --------------------------
    - β = 1.0:  Balanced (default, good starting point)
    - β = 6.0:  High robustness (CIFAR-10 standard, may hurt accuracy)
    - β = 0.1:  Slight robustness (medical imaging, preserve accuracy)
    - β = 0.0:  No robustness (standard training)

    For medical imaging:
    - Dermoscopy (ISIC): β ∈ [0.5, 2.0] (prioritize accuracy)
    - Chest X-ray (CXR): β ∈ [0.3, 1.5] (safety-critical)

    Args:
        beta: Tradeoff parameter (higher = more robust, lower accuracy)
        reduction: Specifies the reduction to apply ('mean' | 'sum' | 'none')
        temperature: Temperature for softening predictions (default: 1.0)
        use_kl: Whether to use KL divergence (True) or MSE (False) for robustness

    Shape:
        - clean_logits: (N, C) where N = batch size, C = num classes
        - adv_logits: (N, C) same as clean_logits
        - labels: (N,) class indices
        - Output: scalar if reduction='mean'/'sum', (N,) if reduction='none'

    Examples:
        >>> criterion = TRADESLoss(beta=1.0)
        >>> loss = criterion(clean_logits, adv_logits, labels)

        >>> # Medical imaging with lower β
        >>> criterion = TRADESLoss(beta=0.5, reduction='mean')
        >>> loss = criterion(clean_logits, adv_logits, labels)

    References:
        Zhang et al. (2019): "Theoretically Principled Trade-off between
        Robustness and Accuracy", ICML 2019
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        temperature: float = 1.0,
        use_kl: bool = True,
    ) -> None:
        super().__init__()

        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.beta = beta
        self.reduction = reduction
        self.temperature = temperature
        self.use_kl = use_kl

        logger.debug(
            f"Initialized TRADESLoss(beta={beta}, reduction={reduction}, "
            f"temperature={temperature}, use_kl={use_kl})"
        )

    def forward(
        self,
        clean_logits: Tensor,
        adv_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute TRADES loss.

        Args:
            clean_logits: Model predictions on clean examples, shape (N, C)
            adv_logits: Model predictions on adversarial examples, shape (N, C)
            labels: Ground truth class labels, shape (N,)

        Returns:
            TRADES loss: L_CE(clean) + β * KL(clean || adv)

        Raises:
            ValueError: If input shapes are inconsistent
            RuntimeError: If numerical instability detected (NaN/Inf)
        """
        # Input validation
        if clean_logits.shape != adv_logits.shape:
            raise ValueError(
                f"Shape mismatch: clean_logits {clean_logits.shape} vs "
                f"adv_logits {adv_logits.shape}"
            )

        batch_size, num_classes = clean_logits.shape

        if labels.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: logits {batch_size} vs labels {labels.shape[0]}"
            )

        # Standard cross-entropy on clean examples
        ce_loss = F.cross_entropy(clean_logits, labels, reduction=self.reduction)

        # Robustness loss: KL divergence between clean and adversarial predictions
        if self.beta > 0:
            if self.use_kl:
                # KL(p_clean || p_adv) encourages consistent predictions
                # Use log_softmax for numerical stability
                log_prob_clean = F.log_softmax(clean_logits / self.temperature, dim=1)
                prob_adv = F.softmax(adv_logits / self.temperature, dim=1)

                # KL divergence: always compute per-example, then apply reduction
                kl_per_example = F.kl_div(
                    log_prob_clean,
                    prob_adv,
                    reduction="none",
                    log_target=False,
                ).sum(
                    dim=1
                )  # Sum over classes for each example

                # Apply reduction
                if self.reduction == "mean":
                    robustness_loss = kl_per_example.mean()
                elif self.reduction == "sum":
                    robustness_loss = kl_per_example.sum()
                else:  # 'none'
                    robustness_loss = kl_per_example
            else:
                # Alternative: MSE between softmax outputs (faster, less theoretically motivated)
                prob_clean = F.softmax(clean_logits / self.temperature, dim=1)
                prob_adv = F.softmax(adv_logits / self.temperature, dim=1)

                mse = F.mse_loss(prob_clean, prob_adv, reduction=self.reduction)
                robustness_loss = mse
        else:
            # If beta=0, no robustness term (standard training)
            if self.reduction == "none":
                robustness_loss = torch.zeros(batch_size, device=clean_logits.device)
            else:
                robustness_loss = torch.tensor(0.0, device=clean_logits.device)

        # Combine losses
        total_loss = ce_loss + self.beta * robustness_loss

        # Numerical stability check
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            logger.error(
                f"Numerical instability detected!\n"
                f"  CE Loss: {ce_loss.item():.4f}\n"
                f"  Robustness Loss: {robustness_loss.item():.4f}\n"
                f"  Total Loss: {total_loss.item():.4f}\n"
                f"  Clean logits range: [{clean_logits.min():.2f}, {clean_logits.max():.2f}]\n"
                f"  Adv logits range: [{adv_logits.min():.2f}, {adv_logits.max():.2f}]"
            )
            raise RuntimeError("NaN or Inf detected in TRADES loss computation")

        return total_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"beta={self.beta}, "
            f"reduction='{self.reduction}', "
            f"temperature={self.temperature}, "
            f"use_kl={self.use_kl})"
        )


# =============================================================================
# MART Loss (Wang et al., 2020)
# =============================================================================


class MARTLoss(nn.Module):
    """
    MART loss for misclassification-aware adversarial training.

    MART (Misclassification Aware adversarial tRaining) adaptively weights
    adversarial training based on clean example misclassification:

        L = L_CE(f(x), y) + β · L_robustness(f(x_adv), y) · w(x)
        w(x) = 1 - p_y(x)  # Higher weight for misclassified examples

    Key Insight:
    ------------
    Standard AT and TRADES treat all examples equally, but:
    - Well-classified examples: Already robust, less training needed
    - Misclassified examples: Need more robustness focus

    MART adaptively focuses robustness training on hard examples,
    improving both clean accuracy and adversarial robustness.

    Advantages:
    -----------
    1. **Adaptive**: Focuses on examples that need robustness most
    2. **Efficiency**: Better sample efficiency than TRADES
    3. **Performance**: Often achieves higher robust accuracy
    4. **Medical Imaging**: Useful for imbalanced disease classes

    Hyperparameter Guidelines:
    --------------------------
    - β = 5.0:  Standard (CIFAR-10)
    - β = 3.0:  Balanced (good default)
    - β = 1.0:  Conservative (medical imaging)

    For medical imaging:
    - Use β ∈ [1.0, 3.0] to avoid overfitting to adversarial examples
    - Useful for rare disease classes (adaptive weighting helps)

    Args:
        beta: Tradeoff parameter for robustness term
        reduction: Specifies the reduction to apply ('mean' | 'sum' | 'none')
        temperature: Temperature for softening predictions
        use_bce: Use binary cross-entropy (True) or KL divergence (False)

    Shape:
        - clean_logits: (N, C) where N = batch size, C = num classes
        - adv_logits: (N, C) same as clean_logits
        - labels: (N,) class indices
        - Output: scalar if reduction='mean'/'sum', (N,) if reduction='none'

    Examples:
        >>> criterion = MARTLoss(beta=3.0)
        >>> loss = criterion(clean_logits, adv_logits, labels)

        >>> # Medical imaging with conservative β
        >>> criterion = MARTLoss(beta=1.0, reduction='mean')
        >>> loss = criterion(clean_logits, adv_logits, labels)

    References:
        Wang et al. (2020): "Improving Adversarial Robustness Requires
        Revisiting Misclassified Examples", ICLR 2020
    """

    def __init__(
        self,
        beta: float = 3.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        temperature: float = 1.0,
        use_bce: bool = True,
    ) -> None:
        super().__init__()

        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.beta = beta
        self.reduction = reduction
        self.temperature = temperature
        self.use_bce = use_bce

        logger.debug(
            f"Initialized MARTLoss(beta={beta}, reduction={reduction}, "
            f"temperature={temperature}, use_bce={use_bce})"
        )

    def forward(
        self,
        clean_logits: Tensor,
        adv_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute MART loss with misclassification-aware weighting.

        Args:
            clean_logits: Model predictions on clean examples, shape (N, C)
            adv_logits: Model predictions on adversarial examples, shape (N, C)
            labels: Ground truth class labels, shape (N,)

        Returns:
            MART loss: L_CE(clean) + β * weighted_robustness_loss

        Raises:
            ValueError: If input shapes are inconsistent
            RuntimeError: If numerical instability detected
        """
        # Input validation
        if clean_logits.shape != adv_logits.shape:
            raise ValueError(
                f"Shape mismatch: clean_logits {clean_logits.shape} vs "
                f"adv_logits {adv_logits.shape}"
            )

        batch_size, num_classes = clean_logits.shape

        if labels.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: logits {batch_size} vs labels {labels.shape[0]}"
            )

        # Standard cross-entropy on clean examples
        ce_loss = F.cross_entropy(clean_logits, labels, reduction=self.reduction)

        # Misclassification-aware weighting
        if self.beta > 0:
            # Get clean prediction probabilities
            prob_clean = F.softmax(clean_logits / self.temperature, dim=1)

            # Extract probability for true class: p_y(x)
            true_class_probs = prob_clean.gather(1, labels.unsqueeze(1)).squeeze(1)

            # Misclassification weight: w(x) = 1 - p_y(x)
            # Higher weight for examples with low confidence on true class
            misclass_weight = 1.0 - true_class_probs

            # Robustness loss on adversarial examples
            if self.use_bce:
                # Binary cross-entropy with target class (MART paper approach)
                prob_adv = F.softmax(adv_logits / self.temperature, dim=1)

                # Create one-hot encoding for target class
                target_one_hot = F.one_hot(labels, num_classes).float()

                # Binary cross-entropy for each example
                # BCE = -[y*log(p) + (1-y)*log(1-p)]
                bce_per_example = F.binary_cross_entropy(
                    prob_adv,
                    target_one_hot,
                    reduction="none",
                ).sum(
                    dim=1
                )  # Sum over classes

                # Weight by misclassification
                weighted_robustness = bce_per_example * misclass_weight
            else:
                # Alternative: KL divergence (similar to TRADES but weighted)
                log_prob_clean = F.log_softmax(clean_logits / self.temperature, dim=1)
                prob_adv = F.softmax(adv_logits / self.temperature, dim=1)

                kl_per_example = F.kl_div(
                    log_prob_clean,
                    prob_adv,
                    reduction="none",
                    log_target=False,
                ).sum(
                    dim=1
                )  # Sum over classes

                # Weight by misclassification
                weighted_robustness = kl_per_example * misclass_weight

            # Apply reduction
            if self.reduction == "mean":
                robustness_loss = weighted_robustness.mean()
            elif self.reduction == "sum":
                robustness_loss = weighted_robustness.sum()
            else:  # 'none'
                robustness_loss = weighted_robustness
        else:
            # If beta=0, no robustness term
            robustness_loss = torch.tensor(0.0, device=clean_logits.device)

        # Combine losses
        total_loss = ce_loss + self.beta * robustness_loss

        # Numerical stability check
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            logger.error(
                f"Numerical instability detected!\n"
                f"  CE Loss: {ce_loss.item():.4f}\n"
                f"  Robustness Loss: {robustness_loss.item():.4f}\n"
                f"  Total Loss: {total_loss.item():.4f}"
            )
            raise RuntimeError("NaN or Inf detected in MART loss computation")

        return total_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"beta={self.beta}, "
            f"reduction='{self.reduction}', "
            f"temperature={self.temperature}, "
            f"use_bce={self.use_bce})"
        )


# =============================================================================
# Standard Adversarial Training Loss
# =============================================================================


class AdversarialTrainingLoss(nn.Module):
    """
    Standard adversarial training (AT) loss.

    Simple adversarial training using cross-entropy on adversarial examples:
        L = L_CE(f(x_adv), y)

    Or mixed training with both clean and adversarial:
        L = λ · L_CE(f(x), y) + (1-λ) · L_CE(f(x_adv), y)

    This is the baseline adversarial training approach from Madry et al. (2018).
    While simpler than TRADES/MART, it often suffers from clean accuracy drop.

    Use Cases:
    ----------
    1. **Baseline**: Compare against TRADES/MART
    2. **Ablation**: Study impact of different loss formulations
    3. **Simplicity**: When TRADES/MART complexity is not needed

    Args:
        mix_clean: Whether to include clean examples in training (0.0 = adv only)
        reduction: Specifies the reduction to apply ('mean' | 'sum' | 'none')

    Shape:
        - logits: (N, C) model predictions
        - labels: (N,) class indices
        - Output: scalar if reduction='mean'/'sum', (N,) if reduction='none'

    Examples:
        >>> # Pure adversarial training
        >>> criterion = AdversarialTrainingLoss(mix_clean=0.0)
        >>> loss = criterion(adv_logits, labels)

        >>> # Mixed clean + adversarial (50/50)
        >>> criterion = AdversarialTrainingLoss(mix_clean=0.5)
        >>> loss = criterion(clean_logits, labels, adv_logits)

    References:
        Madry et al. (2018): "Towards Deep Learning Models Resistant to
        Adversarial Attacks", ICLR 2018
    """

    def __init__(
        self,
        mix_clean: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        if not 0.0 <= mix_clean <= 1.0:
            raise ValueError(f"mix_clean must be in [0, 1], got {mix_clean}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.mix_clean = mix_clean
        self.reduction = reduction

        logger.debug(
            f"Initialized AdversarialTrainingLoss(mix_clean={mix_clean}, "
            f"reduction={reduction})"
        )

    def forward(
        self,
        adv_logits: Tensor,
        labels: Tensor,
        clean_logits: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute standard adversarial training loss.

        Args:
            adv_logits: Model predictions on adversarial examples, shape (N, C)
            labels: Ground truth class labels, shape (N,)
            clean_logits: Optional clean predictions for mixed training, shape (N, C)

        Returns:
            AT loss: λ*L_CE(clean) + (1-λ)*L_CE(adv)

        Raises:
            ValueError: If mix_clean > 0 but clean_logits not provided
        """
        # Adversarial loss
        adv_loss = F.cross_entropy(adv_logits, labels, reduction=self.reduction)

        # Mixed training
        if self.mix_clean > 0:
            if clean_logits is None:
                raise ValueError("clean_logits required when mix_clean > 0, got None")

            clean_loss = F.cross_entropy(clean_logits, labels, reduction=self.reduction)
            total_loss = self.mix_clean * clean_loss + (1 - self.mix_clean) * adv_loss
        else:
            total_loss = adv_loss

        # Numerical stability check
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            logger.error(
                f"Numerical instability detected! Loss: {total_loss.item():.4f}"
            )
            raise RuntimeError("NaN or Inf detected in AT loss computation")

        return total_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mix_clean={self.mix_clean}, "
            f"reduction='{self.reduction}')"
        )


# =============================================================================
# Functional API (convenience functions)
# =============================================================================


def trades_loss(
    clean_logits: Tensor,
    adv_logits: Tensor,
    labels: Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Functional interface for TRADES loss.

    Args:
        clean_logits: Model predictions on clean examples, shape (N, C)
        adv_logits: Model predictions on adversarial examples, shape (N, C)
        labels: Ground truth class labels, shape (N,)
        beta: Tradeoff parameter (default: 1.0)
        reduction: Loss reduction ('mean' | 'sum' | 'none')

    Returns:
        TRADES loss value

    Examples:
        >>> loss = trades_loss(clean_logits, adv_logits, labels, beta=1.0)
    """
    criterion = TRADESLoss(beta=beta, reduction=reduction)
    return criterion(clean_logits, adv_logits, labels)


def mart_loss(
    clean_logits: Tensor,
    adv_logits: Tensor,
    labels: Tensor,
    beta: float = 3.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Functional interface for MART loss.

    Args:
        clean_logits: Model predictions on clean examples, shape (N, C)
        adv_logits: Model predictions on adversarial examples, shape (N, C)
        labels: Ground truth class labels, shape (N,)
        beta: Tradeoff parameter (default: 3.0)
        reduction: Loss reduction ('mean' | 'sum' | 'none')

    Returns:
        MART loss value

    Examples:
        >>> loss = mart_loss(clean_logits, adv_logits, labels, beta=3.0)
    """
    criterion = MARTLoss(beta=beta, reduction=reduction)
    return criterion(clean_logits, adv_logits, labels)


def adversarial_training_loss(
    adv_logits: Tensor,
    labels: Tensor,
    clean_logits: Optional[Tensor] = None,
    mix_clean: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Functional interface for standard adversarial training loss.

    Args:
        adv_logits: Model predictions on adversarial examples, shape (N, C)
        labels: Ground truth class labels, shape (N,)
        clean_logits: Optional clean predictions for mixed training, shape (N, C)
        mix_clean: Mixing coefficient for clean examples (default: 0.0)
        reduction: Loss reduction ('mean' | 'sum' | 'none')

    Returns:
        AT loss value

    Examples:
        >>> # Pure adversarial training
        >>> loss = adversarial_training_loss(adv_logits, labels)

        >>> # Mixed training
        >>> loss = adversarial_training_loss(
        ...     adv_logits, labels, clean_logits, mix_clean=0.5
        ... )
    """
    criterion = AdversarialTrainingLoss(mix_clean=mix_clean, reduction=reduction)
    return criterion(adv_logits, labels, clean_logits)
