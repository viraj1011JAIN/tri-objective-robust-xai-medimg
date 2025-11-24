"""
Tri-Objective Loss for Robust XAI Medical Imaging.

Implements the core tri-objective optimization:
    L_total = L_task + λ_rob * L_rob + λ_expl * L_expl

Where:
- L_task: Cross-entropy with temperature scaling (calibration)
- L_rob: TRADES robustness loss (KL divergence on adversarial examples)
- L_expl: Explanation stability loss (SSIM on Grad-CAM heatmaps + TCAV)

This module is the mathematical heart of the dissertation, implementing:
- Hypothesis H1: Robustness improves under TRADES
- Hypothesis H2: Explanation stability (SSIM > 0.75)
- Hypothesis H3: Selective prediction reduces risk

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Project: Tri-Objective Robust XAI for Medical Imaging
Target: A1+ Grade | Publication-Ready (NeurIPS/MICCAI/TMI)
Deadline: November 28, 2025

Mathematical Formulation
------------------------
L_total = L_CE(f(x), y) / T + λ_rob * KL(f(x) || f(x_adv)) + λ_expl * L_stability

Where:
- T: temperature for calibration (learnable parameter)
- x_adv: adversarial example from PGD
- L_stability: (1 - SSIM(GradCAM_clean, GradCAM_adv)) + γ * L_TCAV
- L_TCAV: concept penalty (artifact concepts punished, medical rewarded)

References
----------
- TRADES: Zhang et al., "Theoretically Principled Trade-off between
  Robustness and Accuracy", ICML 2019
- SSIM: Wang et al., "Image Quality Assessment: From Error Visibility
  to Structural Similarity", IEEE TIP 2004
- TCAV: Kim et al., "Interpretability Beyond Feature Attribution", ICML 2018
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_loss import BaseLoss
from .task_loss import TaskLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable SSIM Implementation
# ---------------------------------------------------------------------------


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for heatmap comparison.

    SSIM measures perceptual similarity between two images:
        SSIM(x, y) = (2μ_x μ_y + c1)(2σ_xy + c2) / ((μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2))

    For explanation stability, we compute:
        L_ssim = 1 - SSIM(heatmap_clean, heatmap_adv)

    This is differentiable and can be used in the backward pass.

    Parameters
    ----------
    window_size : int
        Size of the Gaussian window (default: 11)
    sigma : float
        Standard deviation of the Gaussian (default: 1.5)
    channel : int
        Number of channels (1 for grayscale heatmaps)
    data_range : float
        Dynamic range of the heatmaps (default: 1.0)

    Returns
    -------
    loss : Tensor
        1 - SSIM, shape (batch_size,) or scalar if reduced
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 1,
        data_range: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.data_range = data_range
        self.reduction = reduction

        # Create Gaussian window
        self.window = self._create_window(window_size, sigma, channel)

        # SSIM constants (avoid division by zero)
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def _create_window(self, size: int, sigma: float, channel: int) -> Tensor:
        """Create 2D Gaussian window."""
        # 1D Gaussian
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        # 2D Gaussian (outer product)
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)

        # Expand to channel dimension
        window = window_2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, size, size).contiguous()

        return window

    def _ssim(
        self,
        x: Tensor,
        y: Tensor,
        window: Tensor,
    ) -> Tensor:
        """
        Compute SSIM between two tensors.

        Parameters
        ----------
        x, y : Tensor
            Input tensors, shape (B, C, H, W)
        window : Tensor
            Gaussian window, shape (C, 1, K, K)

        Returns
        -------
        ssim : Tensor
            SSIM values, shape (B,)
        """
        # Move window to same device as input
        if window.device != x.device:
            window = window.to(x.device)

        # Compute local means
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=self.channel)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        # Compute local variances and covariance
        sigma_x_sq = (
            F.conv2d(x * x, window, padding=self.window_size // 2, groups=self.channel)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_xy
        )

        # Compute SSIM
        numerator = (2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )

        ssim_map = numerator / denominator

        # Average over spatial dimensions and channels
        ssim_val = ssim_map.mean(dim=[1, 2, 3])

        return ssim_val

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Parameters
        ----------
        x, y : Tensor
            Heatmaps to compare, shape (B, C, H, W)

        Returns
        -------
        loss : Tensor
            1 - SSIM, shape (B,) or scalar
        """
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensors (B, C, H, W), got {x.ndim}D")

        # Compute SSIM
        ssim_val = self._ssim(x, y, self.window)

        # Convert to loss (1 - SSIM)
        loss = 1.0 - ssim_val

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ---------------------------------------------------------------------------
# TCAV Loss (Concept-Based Explanation Loss)
# ---------------------------------------------------------------------------


class TCAVLoss(nn.Module):
    """
    TCAV (Testing with Concept Activation Vectors) Loss.

    Penalizes activations aligned with artifact concepts,
    rewards activations aligned with medical concepts.

    For Phase 4.3 (baseline), we use a simplified placeholder.
    Full TCAV with learned CAVs will be implemented in Phase 5.

    L_TCAV = -1 * (CAV_medical^T · activations) + (CAV_artifact^T · activations)

    For now, we return zero and will replace with real implementation
    when concept vectors are trained (Phase 5.3).

    Parameters
    ----------
    num_concepts : int
        Number of concepts (medical + artifact)
    embedding_dim : int
        Dimension of feature embeddings

    Returns
    -------
    loss : Tensor
        Concept alignment loss, scalar
    """

    def __init__(
        self,
        num_concepts: int = 10,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self._initialized = False

        # Will initialize lazily on first forward pass if embedding_dim is None
        if embedding_dim is not None:
            self._initialize_cavs(embedding_dim)

        logger.warning(
            "TCAVLoss initialized with random CAVs. "
            "Replace with learned CAVs in Phase 5.3."
        )

    def _initialize_cavs(self, embedding_dim: int):
        """Initialize CAV vectors with given embedding dimension."""
        if not self._initialized:
            self.embedding_dim = embedding_dim
            self.register_buffer(
                "medical_cavs",
                torch.randn(self.num_concepts // 2, embedding_dim) * 0.01,
            )
            self.register_buffer(
                "artifact_cavs",
                torch.randn(self.num_concepts // 2, embedding_dim) * 0.01,
            )
            self._initialized = True

    def forward(
        self,
        embeddings: Tensor,
        return_scores: bool = False,
    ) -> Tensor:
        """
        Compute TCAV loss.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings from model, shape (B, D)
        return_scores : bool
            If True, return individual TCAV scores (for logging)

        Returns
        -------
        loss : Tensor
            TCAV loss, scalar
        tcav_scores : Dict[str, Tensor], optional
            Individual TCAV scores (if return_scores=True)
        """
        # Lazy initialization if not done yet
        if not self._initialized:
            self._initialize_cavs(embeddings.size(1))

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Compute alignment with medical concepts (positive is good)
        medical_alignment = torch.matmul(
            embeddings_norm,
            self.medical_cavs.T,
        )
        medical_score = medical_alignment.mean()

        # Compute alignment with artifact concepts (positive is bad)
        artifact_alignment = torch.matmul(
            embeddings_norm,
            self.artifact_cavs.T,
        )
        artifact_score = artifact_alignment.mean()

        # Loss: penalize artifact, reward medical
        loss = artifact_score - medical_score

        if return_scores:
            return loss, {
                "medical_tcav": medical_score.item(),
                "artifact_tcav": artifact_score.item(),
            }

        return loss


# ---------------------------------------------------------------------------
# TRADES Robustness Loss
# ---------------------------------------------------------------------------


class TRADESLoss(nn.Module):
    """
    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization).

    Implements the KL divergence between clean and adversarial predictions:
        L_rob = KL(p(y|x) || p(y|x_adv))

    Where x_adv is generated via PGD attack to maximize KL divergence.

    This loss encourages the model to produce similar predictions on
    clean and adversarially perturbed inputs, improving robustness.

    Reference:
        Zhang et al., "Theoretically Principled Trade-off between
        Robustness and Accuracy", ICML 2019

    Parameters
    ----------
    beta : float
        Weight for the robustness loss (default: 6.0, as per paper)

    Returns
    -------
    loss : Tensor
        KL divergence, scalar
    """

    def __init__(self, beta: float = 6.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logits_clean: Tensor,
        logits_adv: Tensor,
    ) -> Tensor:
        """
        Compute TRADES robustness loss.

        Parameters
        ----------
        logits_clean : Tensor
            Logits on clean inputs, shape (B, C)
        logits_adv : Tensor
            Logits on adversarial inputs, shape (B, C)

        Returns
        -------
        loss : Tensor
            KL divergence, scalar
        """
        # Convert to log probabilities
        log_probs_clean = F.log_softmax(logits_clean, dim=1)
        probs_adv = F.softmax(logits_adv, dim=1)

        # KL divergence: KL(p_adv || p_clean)
        # Note: PyTorch's kl_div expects log_probs as first arg
        kl_div = F.kl_div(
            log_probs_clean,
            probs_adv,
            reduction="batchmean",
        )

        return self.beta * kl_div


# ---------------------------------------------------------------------------
# Main Tri-Objective Loss
# ---------------------------------------------------------------------------


class TriObjectiveLoss(BaseLoss):
    """
    Tri-Objective Loss combining Task + Robustness + Explainability.

    This is the core loss function for the entire dissertation:
        L_total = L_task + λ_rob * L_rob + λ_expl * L_expl

    Where:
    - L_task: Cross-entropy with temperature scaling
    - L_rob: TRADES KL divergence (robustness)
    - L_expl: SSIM on Grad-CAM heatmaps + TCAV (explanation stability)

    The loss enforces three objectives:
    1. Task Accuracy: Correct predictions on clean data
    2. Adversarial Robustness: Similar predictions on clean/adversarial
    3. Explanation Stability: Similar heatmaps on clean/adversarial

    Parameters
    ----------
    num_classes : int
        Number of output classes
    task_type : str
        "multi_class" or "multi_label"
    lambda_rob : float
        Weight for robustness loss (default: 0.3)
    lambda_expl : float
        Weight for explanation loss (default: 0.2)
    lambda_ssim : float
        Weight for SSIM within explanation loss (default: 0.7)
    lambda_tcav : float
        Weight for TCAV within explanation loss (default: 0.3)
    temperature : float
        Initial temperature for calibration (learnable, default: 1.5)
    trades_beta : float
        Beta parameter for TRADES (default: 6.0)

    Example
    -------
    >>> loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
    >>>
    >>> # Forward pass (requires heatmaps and embeddings)
    >>> outputs = loss_fn(
    ...     logits_clean=logits_clean,
    ...     logits_adv=logits_adv,
    ...     labels=labels,
    ...     heatmap_clean=heatmap_clean,
    ...     heatmap_adv=heatmap_adv,
    ...     embeddings=embeddings,
    ... )
    >>>
    >>> loss = outputs["loss"]
    >>> loss.backward()
    """

    def __init__(
        self,
        num_classes: int,
        task_type: str = "multi_class",
        lambda_rob: float = 0.3,
        lambda_expl: float = 0.2,
        lambda_ssim: float = 0.7,
        lambda_tcav: float = 0.3,
        temperature: float = 1.5,
        trades_beta: float = 6.0,
        reduction: str = "mean",
        name: str = "tri_objective",
    ):
        super().__init__(reduction=reduction, name=name)

        self.num_classes = num_classes
        self.task_type = task_type
        self.lambda_rob = lambda_rob
        self.lambda_expl = lambda_expl
        self.lambda_ssim = lambda_ssim
        self.lambda_tcav = lambda_tcav

        # Task loss (with temperature scaling)
        self.task_loss_fn = TaskLoss(
            num_classes=num_classes,
            task_type=task_type,
            reduction=reduction,
        )

        # Temperature parameter (learnable for calibration)
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Robustness loss (TRADES)
        self.robustness_loss_fn = TRADESLoss(beta=trades_beta)

        # Explanation losses
        self.ssim_loss_fn = SSIMLoss(reduction=reduction)
        self.tcav_loss_fn = TCAVLoss()

        logger.info(
            f"Initialized TriObjectiveLoss: λ_rob={lambda_rob}, "
            f"λ_expl={lambda_expl}, T={temperature}, β={trades_beta}"
        )

    def forward(
        self,
        logits_clean: Tensor,
        logits_adv: Tensor,
        labels: Tensor,
        heatmap_clean: Optional[Tensor] = None,
        heatmap_adv: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute tri-objective loss.

        Parameters
        ----------
        logits_clean : Tensor
            Logits on clean inputs, shape (B, C)
        logits_adv : Tensor
            Logits on adversarial inputs, shape (B, C)
        labels : Tensor
            Ground truth labels, shape (B,) for multi-class or (B, C) for multi-label
        heatmap_clean : Tensor, optional
            Grad-CAM heatmap on clean inputs, shape (B, 1, H, W)
        heatmap_adv : Tensor, optional
            Grad-CAM heatmap on adversarial inputs, shape (B, 1, H, W)
        embeddings : Tensor, optional
            Feature embeddings for TCAV, shape (B, D)

        Returns
        -------
        outputs : Dict[str, Tensor]
            Dictionary containing:
            - "loss": total loss (scalar)
            - "task": task loss (scalar)
            - "robustness": robustness loss (scalar)
            - "explanation": explanation loss (scalar)
            - "ssim": SSIM loss (scalar, if heatmaps provided)
            - "tcav": TCAV loss (scalar, if embeddings provided)
            - "temperature": current temperature (scalar)
        """
        # 1. Task Loss (with temperature scaling)
        logits_scaled = logits_clean / self.temperature
        task_loss = self.task_loss_fn(logits_scaled, labels)

        # 2. Robustness Loss (TRADES)
        robustness_loss = self.robustness_loss_fn(logits_clean, logits_adv)

        # 3. Explanation Loss (SSIM + TCAV)
        explanation_loss = torch.tensor(0.0, device=logits_clean.device)
        ssim_loss = torch.tensor(0.0, device=logits_clean.device)
        tcav_loss = torch.tensor(0.0, device=logits_clean.device)

        if heatmap_clean is not None and heatmap_adv is not None:
            ssim_loss = self.ssim_loss_fn(heatmap_clean, heatmap_adv)
            explanation_loss = explanation_loss + self.lambda_ssim * ssim_loss

        if embeddings is not None:
            tcav_loss = self.tcav_loss_fn(embeddings)
            explanation_loss = explanation_loss + self.lambda_tcav * tcav_loss

        # 4. Total Loss
        total_loss = (
            task_loss
            + self.lambda_rob * robustness_loss
            + self.lambda_expl * explanation_loss
        )

        # Return all components for logging
        return {
            "loss": total_loss,
            "task": task_loss,
            "robustness": robustness_loss,
            "explanation": explanation_loss,
            "ssim": ssim_loss,
            "tcav": tcav_loss,
            "temperature": self.temperature,
        }

    def compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compatibility method for BaseLoss interface.

        For tri-objective loss, use forward() with all required inputs.
        This method raises an error if called directly.
        """
        raise NotImplementedError(
            "TriObjectiveLoss requires forward() with logits_clean, "
            "logits_adv, labels, and optional heatmaps/embeddings. "
            "Do not use compute() directly."
        )
