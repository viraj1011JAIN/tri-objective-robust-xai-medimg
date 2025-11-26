"""
Explanation Loss Module for Tri-Objective Robust XAI Framework.

This module implements the explanation quality loss (L_expl) comprising:
1. L_stab: SSIM-based stability loss for explanation consistency
2. L_concept: TCAV-based concept regularization for artifact suppression

Mathematical Formulation:
    L_expl = L_stab + γ × L_concept

    L_stab = 1 - SSIM(H_clean, H_adv)

    L_concept = Σ max(0, TCAV_artifact - τ)
                - λ_med × Σ max(0, τ_med - TCAV_medical)

Key Features:
    - Efficient batch-parallel implementation
    - Multi-Scale SSIM support for ablation studies
    - Differentiable TCAV for end-to-end training
    - Integrated FGSM adversarial perturbation
    - Built-in Grad-CAM computation with activation/gradient extraction
    - Comprehensive gradient flow verification
    - Production-level error handling and validation

Usage:
    >>> from src.losses.explanation_loss import create_explanation_loss
    >>>
    >>> loss_fn = create_explanation_loss(
    ...     model=model,
    ...     artifact_cavs=artifact_cavs,
    ...     medical_cavs=medical_cavs,
    ...     gamma=0.5,
    ...     use_ms_ssim=False
    ... )
    >>>
    >>> loss, metrics = loss_fn(images, labels, return_components=True)
    >>> print(f"Stability: {metrics['loss_stability']:.4f}")
    >>> print(f"Concept: {metrics['loss_concept']:.4f}")

References:
    1. Wang et al. "Image Quality Assessment: From Error Visibility to
       Structural Similarity", IEEE TIP 2004
    2. Kim et al. "Interpretability Beyond Feature Attribution: Quantitative
       Testing with Concept Activation Vectors (TCAV)", ICML 2018
    3. Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks",
       ICCV 2017
    4. Goodfellow et al. "Explaining and Harnessing Adversarial Examples",
       ICLR 2015

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 26, 2025
Version: 1.0.0 (Phase 7.1)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class SSIMKernelType(str, Enum):
    """SSIM kernel types."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


# SSIM constants from Wang et al. 2004
_C1 = (0.01) ** 2  # Stability constant for luminance
_C2 = (0.03) ** 2  # Stability constant for contrast


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class ExplanationLossConfig:
    """Configuration for explanation loss computation.

    Attributes:
        gamma: Weight for concept regularization (default: 0.5)
        tau_artifact: Penalty threshold for artifact TCAV (default: 0.3)
        tau_medical: Reward threshold for medical TCAV (default: 0.5)
        lambda_medical: Weight for medical concept reward (default: 0.5)
        fgsm_epsilon: FGSM perturbation magnitude (default: 2/255)
        use_ms_ssim: Use Multi-Scale SSIM (default: False)
        ssim_window_size: Window size for SSIM (default: 11)
        ssim_sigma: Gaussian sigma for SSIM (default: 1.5)
        ssim_data_range: Data range for SSIM normalization (default: 1.0)
        ssim_k1: Luminance stability constant (default: 0.01)
        ssim_k2: Contrast stability constant (default: 0.03)
        kernel_type: SSIM kernel type (default: "gaussian")
        reduction: Loss reduction method (default: "mean")
        differentiable: Use differentiable TCAV (default: True)
        soft_temperature: Temperature for soft TCAV (default: 10.0)
        gradcam_target_layer: Target layer for Grad-CAM (default: None, auto-detect)
        normalize_gradients: Normalize gradients before TCAV (default: True)
    """

    # Concept regularization weights
    gamma: float = 0.5
    tau_artifact: float = 0.3
    tau_medical: float = 0.5
    lambda_medical: float = 0.5

    # FGSM parameters
    fgsm_epsilon: float = 2.0 / 255.0

    # SSIM parameters
    use_ms_ssim: bool = False
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    ssim_data_range: float = 1.0
    ssim_k1: float = 0.01
    ssim_k2: float = 0.03
    kernel_type: SSIMKernelType = SSIMKernelType.GAUSSIAN

    # Loss computation parameters
    reduction: str = "mean"
    differentiable: bool = True
    soft_temperature: float = 10.0

    # Grad-CAM parameters
    gradcam_target_layer: Optional[str] = None
    normalize_gradients: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")

        if not 0 <= self.tau_artifact <= 1:
            raise ValueError(f"tau_artifact must be in [0, 1], got {self.tau_artifact}")

        if not 0 <= self.tau_medical <= 1:
            raise ValueError(f"tau_medical must be in [0, 1], got {self.tau_medical}")

        if self.lambda_medical < 0:
            raise ValueError(
                f"lambda_medical must be non-negative, got {self.lambda_medical}"
            )

        if self.fgsm_epsilon < 0:
            raise ValueError(
                f"fgsm_epsilon must be non-negative, got {self.fgsm_epsilon}"
            )

        if self.ssim_window_size % 2 == 0 or self.ssim_window_size < 3:
            raise ValueError(
                f"ssim_window_size must be odd and >= 3, got {self.ssim_window_size}"
            )

        if self.ssim_sigma <= 0:
            raise ValueError(f"ssim_sigma must be positive, got {self.ssim_sigma}")

        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {self.reduction}"
            )

        if self.soft_temperature <= 0:
            raise ValueError(
                f"soft_temperature must be positive, got {self.soft_temperature}"
            )


# ============================================================================
# SSIM Stability Loss
# ============================================================================


class SSIMStabilityLoss(nn.Module):
    """SSIM-based stability loss for explanation consistency.

    Computes structural similarity between clean and adversarial heatmaps.
    Loss = 1 - SSIM, where SSIM ∈ [0, 1].

    The SSIM metric compares:
    - Luminance: mean intensity
    - Contrast: standard deviation
    - Structure: correlation

    Reference:
        Wang et al. "Image Quality Assessment: From Error Visibility to
        Structural Similarity", IEEE TIP 2004

    Args:
        window_size: Size of Gaussian window (must be odd)
        sigma: Standard deviation of Gaussian window
        data_range: Expected data range (e.g., 1.0 for normalized [0,1])
        k1: Luminance stability constant (default: 0.01)
        k2: Contrast stability constant (default: 0.03)
        reduction: Reduction method ("mean", "sum", "none")
        use_ms_ssim: Use Multi-Scale SSIM
        kernel_type: Kernel type ("gaussian" or "uniform")

    Example:
        >>> loss_fn = SSIMStabilityLoss(window_size=11, sigma=1.5)
        >>> heatmap1 = torch.rand(4, 1, 56, 56)
        >>> heatmap2 = torch.rand(4, 1, 56, 56)
        >>> loss = loss_fn(heatmap1, heatmap2)
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: str = "mean",
        use_ms_ssim: bool = False,
        kernel_type: Union[str, SSIMKernelType] = SSIMKernelType.GAUSSIAN,
    ) -> None:
        """Initialize SSIM stability loss."""
        super().__init__()

        # Validate parameters
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError(f"window_size must be odd and >= 3, got {window_size}")

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction
        self.use_ms_ssim = use_ms_ssim
        self.kernel_type = (
            SSIMKernelType(kernel_type) if isinstance(kernel_type, str) else kernel_type
        )

        # Stability constants
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

        # Lazy kernel initialization (created on first forward pass)
        self._kernel: Optional[Tensor] = None

    def _create_kernel(self, channels: int, device: torch.device) -> Tensor:
        """Create SSIM kernel.

        Args:
            channels: Number of channels
            device: Target device

        Returns:
            Kernel tensor of shape (channels, 1, window_size, window_size)
        """
        if self.kernel_type == SSIMKernelType.GAUSSIAN:
            return _create_gaussian_kernel(
                self.window_size, self.sigma, channels, device
            )
        else:  # UNIFORM
            return _create_uniform_kernel(self.window_size, channels, device)

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute SSIM between two tensors.

        Args:
            x: First tensor (B, C, H, W)
            y: Second tensor (B, C, H, W)

        Returns:
            SSIM score (B,) if reduction="none", else scalar
        """
        channels = x.size(1)
        device = x.device

        # Create kernel if not exists or device changed
        if self._kernel is None or self._kernel.device != device:
            self._kernel = self._create_kernel(channels, device)

        # Compute local means using depthwise convolution
        mu_x = F.conv2d(x, self._kernel, padding=self.window_size // 2, groups=channels)
        mu_y = F.conv2d(y, self._kernel, padding=self.window_size // 2, groups=channels)

        # Compute local variances and covariance
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(
                x * x, self._kernel, padding=self.window_size // 2, groups=channels
            )
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(
                y * y, self._kernel, padding=self.window_size // 2, groups=channels
            )
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(
                x * y, self._kernel, padding=self.window_size // 2, groups=channels
            )
            - mu_xy
        )

        # SSIM formula
        numerator = (2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )

        ssim_map = numerator / denominator

        # Reduce over spatial and channel dimensions
        if self.reduction == "mean":
            return ssim_map.mean()
        elif self.reduction == "sum":
            return ssim_map.sum()
        else:  # "none"
            return ssim_map.mean(dim=[1, 2, 3])  # Return per-sample scores

    def _ms_ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute Multi-Scale SSIM.

        Args:
            x: First tensor (B, C, H, W)
            y: Second tensor (B, C, H, W)

        Returns:
            MS-SSIM score
        """
        # Default MS-SSIM weights from Wang et al.
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device
        )

        # Determine number of scales based on image size
        min_size = min(x.size(-2), x.size(-1))
        num_scales = min(5, int(np.log2(min_size / self.window_size)) + 1)
        weights = weights[:num_scales]
        weights = weights / weights.sum()  # Normalize

        mssim_list = []
        mcs_list = []

        for i in range(num_scales):
            if i > 0:
                # Downsample by 2x
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

            ssim_val = self._ssim(x, y)
            mssim_list.append(ssim_val)

            # Compute contrast comparison (structural similarity)
            channels = x.size(1)
            if self._kernel is None or self._kernel.device != x.device:
                self._kernel = self._create_kernel(channels, x.device)

            mu_x = F.conv2d(
                x, self._kernel, padding=self.window_size // 2, groups=channels
            )
            mu_y = F.conv2d(
                y, self._kernel, padding=self.window_size // 2, groups=channels
            )
            sigma_x_sq = F.conv2d(
                x * x, self._kernel, padding=self.window_size // 2, groups=channels
            ) - mu_x.pow(2)
            sigma_y_sq = F.conv2d(
                y * y, self._kernel, padding=self.window_size // 2, groups=channels
            ) - mu_y.pow(2)
            sigma_xy = (
                F.conv2d(
                    x * y, self._kernel, padding=self.window_size // 2, groups=channels
                )
                - mu_x * mu_y
            )

            cs = (2 * sigma_xy + self.c2) / (sigma_x_sq + sigma_y_sq + self.c2)
            mcs_list.append(cs.mean())

        # Combine scales
        mssim_val = torch.prod(
            torch.stack([mcs**w for mcs, w in zip(mcs_list[:-1], weights[:-1])])
        )
        mssim_val = mssim_val * (mssim_list[-1] ** weights[-1])

        return mssim_val

    def forward(self, heatmap_clean: Tensor, heatmap_adv: Tensor) -> Tensor:
        """Compute SSIM stability loss.

        Args:
            heatmap_clean: Clean heatmap (B, C, H, W) or (B, H, W)
            heatmap_adv: Adversarial heatmap (B, C, H, W) or (B, H, W)

        Returns:
            Loss = 1 - SSIM

        Raises:
            ValueError: If input shapes don't match or are invalid
        """
        # Handle 3D inputs (B, H, W) -> (B, 1, H, W)
        if heatmap_clean.dim() == 3:
            heatmap_clean = heatmap_clean.unsqueeze(1)
        if heatmap_adv.dim() == 3:
            heatmap_adv = heatmap_adv.unsqueeze(1)

        # Validate inputs
        if heatmap_clean.shape != heatmap_adv.shape:
            raise ValueError(
                f"Input shapes must match: got {heatmap_clean.shape} "
                f"vs {heatmap_adv.shape}"
            )

        if heatmap_clean.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got {heatmap_clean.dim()}D"
            )

        # Ensure spatial dimensions are large enough
        min_size = min(heatmap_clean.size(-2), heatmap_clean.size(-1))
        if min_size < self.window_size:
            raise RuntimeError(
                f"Input spatial size ({min_size}) must be >= window_size "
                f"({self.window_size})"
            )

        # Compute SSIM
        if self.use_ms_ssim:
            ssim_score = self._ms_ssim(heatmap_clean, heatmap_adv)
        else:
            ssim_score = self._ssim(heatmap_clean, heatmap_adv)

        # Return loss = 1 - SSIM
        return 1.0 - ssim_score


# ============================================================================
# TCAV Concept Loss
# ============================================================================


class TCavConceptLoss(nn.Module):
    """TCAV-based concept regularization loss.

    Penalizes artifact concepts and rewards medical concepts using
    Testing with Concept Activation Vectors (TCAV) methodology.

    Loss = Σ max(0, TCAV_artifact - τ)
           - λ_med × Σ max(0, τ_med - TCAV_medical)

    Reference:
        Kim et al. "Interpretability Beyond Feature Attribution: Quantitative
        Testing with Concept Activation Vectors (TCAV)", ICML 2018

    Args:
        artifact_cavs: List of artifact CAV vectors
        medical_cavs: List of medical CAV vectors
        tau_artifact: Penalty threshold for artifact TCAV (default: 0.3)
        tau_medical: Reward threshold for medical TCAV (default: 0.5)
        lambda_medical: Weight for medical concept reward (default: 0.5)
        reduction: Reduction method ("mean", "sum", "none")
        differentiable: Use differentiable (soft) TCAV (default: True)
        soft_temperature: Temperature for soft TCAV (default: 10.0)
        normalize_gradients: Normalize gradients before TCAV (default: True)

    Example:
        >>> artifact_cavs = [torch.randn(2048) for _ in range(4)]
        >>> medical_cavs = [torch.randn(2048) for _ in range(6)]
        >>> loss_fn = TCavConceptLoss(artifact_cavs, medical_cavs)
        >>> activations = torch.randn(8, 2048)
        >>> gradients = torch.randn(8, 2048)
        >>> loss, metrics = loss_fn(activations, gradients)
    """

    def __init__(
        self,
        artifact_cavs: Optional[List[Tensor]] = None,
        medical_cavs: Optional[List[Tensor]] = None,
        tau_artifact: float = 0.3,
        tau_medical: float = 0.5,
        lambda_medical: float = 0.5,
        reduction: str = "mean",
        differentiable: bool = True,
        soft_temperature: float = 10.0,
        normalize_gradients: bool = True,
    ) -> None:
        """Initialize TCAV concept loss."""
        super().__init__()

        # Validate parameters
        if not 0 <= tau_artifact <= 1:
            raise ValueError(f"tau_artifact must be in [0, 1], got {tau_artifact}")

        if not 0 <= tau_medical <= 1:
            raise ValueError(f"tau_medical must be in [0, 1], got {tau_medical}")

        if lambda_medical < 0:
            raise ValueError(
                f"lambda_medical must be non-negative, got {lambda_medical}"
            )

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        if soft_temperature <= 0:
            raise ValueError(
                f"soft_temperature must be positive, got {soft_temperature}"
            )

        self.tau_artifact = tau_artifact
        self.tau_medical = tau_medical
        self.lambda_medical = lambda_medical
        self.reduction = reduction
        self.differentiable = differentiable
        self.soft_temperature = soft_temperature
        self.normalize_gradients = normalize_gradients

        # Store CAVs as buffers (moved to device automatically)
        self._artifact_cavs: List[Tensor] = []
        self._medical_cavs: List[Tensor] = []

        if artifact_cavs is not None:
            for i, cav in enumerate(artifact_cavs):
                self.register_buffer(f"artifact_cav_{i}", cav)
                self._artifact_cavs.append(cav)

        if medical_cavs is not None:
            for i, cav in enumerate(medical_cavs):
                self.register_buffer(f"medical_cav_{i}", cav)
                self._medical_cavs.append(cav)

    def update_cavs(
        self,
        artifact_cavs: Optional[List[Tensor]] = None,
        medical_cavs: Optional[List[Tensor]] = None,
    ) -> None:
        """Update CAVs dynamically during training.

        Args:
            artifact_cavs: New artifact CAV list
            medical_cavs: New medical CAV list
        """
        if artifact_cavs is not None:
            # Clear old buffers
            for i in range(len(self._artifact_cavs)):
                if hasattr(self, f"artifact_cav_{i}"):
                    delattr(self, f"artifact_cav_{i}")

            # Register new buffers
            self._artifact_cavs = []
            for i, cav in enumerate(artifact_cavs):
                self.register_buffer(f"artifact_cav_{i}", cav)
                self._artifact_cavs.append(cav)

        if medical_cavs is not None:
            # Clear old buffers
            for i in range(len(self._medical_cavs)):
                if hasattr(self, f"medical_cav_{i}"):
                    delattr(self, f"medical_cav_{i}")

            # Register new buffers
            self._medical_cavs = []
            for i, cav in enumerate(medical_cavs):
                self.register_buffer(f"medical_cav_{i}", cav)
                self._medical_cavs.append(cav)

    def _compute_tcav_score(
        self, cav: Tensor, activations: Tensor, gradients: Tensor
    ) -> Tensor:
        """Compute TCAV score for a single CAV.

        TCAV(c) = (1/N) Σ_i 𝟙[∇h · v_c > 0]

        For differentiable version:
        TCAV_soft(c) = (1/N) Σ_i sigmoid(temperature × ∇h · v_c)

        Args:
            cav: Concept Activation Vector (D,)
            activations: Feature activations (B, D)
            gradients: Feature gradients (B, D)

        Returns:
            TCAV score (scalar)
        """
        # Normalize CAV
        cav_norm = F.normalize(cav.unsqueeze(0), p=2, dim=1)  # (1, D)

        # Normalize gradients if requested
        if self.normalize_gradients:
            gradients = F.normalize(gradients, p=2, dim=1)  # (B, D)

        # Compute directional derivatives: ∇h · v_c
        directional_derivatives = torch.sum(gradients * cav_norm, dim=1)  # (B,)

        # Compute TCAV score
        if self.differentiable:
            # Soft (differentiable) TCAV using sigmoid
            tcav_score = torch.sigmoid(
                self.soft_temperature * directional_derivatives
            ).mean()
        else:
            # Hard (non-differentiable) TCAV using indicator function
            tcav_score = (directional_derivatives > 0).float().mean()

        return tcav_score

    def forward(
        self, activations: Tensor, gradients: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute TCAV concept loss.

        Args:
            activations: Feature activations (B, D)
            gradients: Feature gradients (B, D)

        Returns:
            Tuple of (loss, metrics_dict)
            metrics_dict contains:
                - artifact_tcav_mean: Mean TCAV score for artifact concepts
                - medical_tcav_mean: Mean TCAV score for medical concepts
                - tcav_ratio: Ratio of medical to artifact TCAV

        Raises:
            ValueError: If input dimensions don't match
        """
        # Validate inputs
        if activations.shape != gradients.shape:
            raise ValueError(
                f"Activations and gradients must have same shape: "
                f"got {activations.shape} vs {gradients.shape}"
            )

        if activations.dim() != 2:
            raise ValueError(
                f"Expected 2D activations (B, D), got {activations.dim()}D"
            )

        # Check if we have any CAVs
        if len(self._artifact_cavs) == 0 and len(self._medical_cavs) == 0:
            warnings.warn(
                "No CAVs registered. Returning zero loss.",
                UserWarning,
                stacklevel=2,
            )
            device = activations.device
            return torch.tensor(0.0, device=device), {}

        # Compute artifact TCAV scores and penalty
        artifact_loss = torch.tensor(0.0, device=activations.device)
        artifact_tcav_scores = []

        for cav in self._artifact_cavs:
            tcav_score = self._compute_tcav_score(cav, activations, gradients)
            artifact_tcav_scores.append(tcav_score.item())

            # Penalty: max(0, TCAV - τ)
            penalty = torch.relu(tcav_score - self.tau_artifact)
            artifact_loss = artifact_loss + penalty

        # Compute medical TCAV scores and reward
        medical_loss = torch.tensor(0.0, device=activations.device)
        medical_tcav_scores = []

        for cav in self._medical_cavs:
            tcav_score = self._compute_tcav_score(cav, activations, gradients)
            medical_tcav_scores.append(tcav_score.item())

            # Reward: -λ × max(0, τ - TCAV)
            reward = -self.lambda_medical * torch.relu(self.tau_medical - tcav_score)
            medical_loss = medical_loss + reward

        # Combine losses
        total_loss = artifact_loss + medical_loss

        # Apply reduction
        if self.reduction == "mean":
            if len(self._artifact_cavs) + len(self._medical_cavs) > 0:
                total_loss = total_loss / (
                    len(self._artifact_cavs) + len(self._medical_cavs)
                )
        elif self.reduction == "sum":
            pass  # Already summed
        # "none" not applicable for concept loss (returns scalar per concept)

        # Prepare metrics
        metrics = {}
        if artifact_tcav_scores:
            metrics["artifact_tcav_mean"] = float(np.mean(artifact_tcav_scores))
        if medical_tcav_scores:
            metrics["medical_tcav_mean"] = float(np.mean(medical_tcav_scores))
        if artifact_tcav_scores and medical_tcav_scores:
            artifact_mean = float(np.mean(artifact_tcav_scores))
            medical_mean = float(np.mean(medical_tcav_scores))
            if artifact_mean > 0:
                metrics["tcav_ratio"] = medical_mean / artifact_mean
            else:
                metrics["tcav_ratio"] = float("inf") if medical_mean > 0 else 1.0

        return total_loss, metrics


# ============================================================================
# Combined Explanation Loss
# ============================================================================


class ExplanationLoss(nn.Module):
    """Combined explanation loss: L_expl = L_stab + γ × L_concept.

    Integrates SSIM-based stability loss with TCAV-based concept regularization.
    Includes built-in FGSM adversarial perturbation and Grad-CAM computation.

    Args:
        model: PyTorch model (optional, can be set later)
        config: ExplanationLossConfig instance
        artifact_cavs: List of artifact CAV vectors
        medical_cavs: List of medical CAV vectors

    Example:
        >>> config = ExplanationLossConfig(gamma=0.5)
        >>> loss_fn = ExplanationLoss(model, config, artifact_cavs, medical_cavs)
        >>> loss, metrics = loss_fn(images, labels, return_components=True)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[ExplanationLossConfig] = None,
        artifact_cavs: Optional[List[Tensor]] = None,
        medical_cavs: Optional[List[Tensor]] = None,
    ) -> None:
        """Initialize explanation loss."""
        super().__init__()

        self.config = config if config is not None else ExplanationLossConfig()
        self._model = model

        # Initialize sub-losses
        self.ssim_loss = SSIMStabilityLoss(
            window_size=self.config.ssim_window_size,
            sigma=self.config.ssim_sigma,
            data_range=self.config.ssim_data_range,
            k1=self.config.ssim_k1,
            k2=self.config.ssim_k2,
            reduction=self.config.reduction,
            use_ms_ssim=self.config.use_ms_ssim,
            kernel_type=self.config.kernel_type,
        )

        self.concept_loss = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            tau_artifact=self.config.tau_artifact,
            tau_medical=self.config.tau_medical,
            lambda_medical=self.config.lambda_medical,
            reduction=self.config.reduction,
            differentiable=self.config.differentiable,
            soft_temperature=self.config.soft_temperature,
            normalize_gradients=self.config.normalize_gradients,
        )

        # Store hyperparameters
        self.gamma = self.config.gamma
        self.fgsm_epsilon = self.config.fgsm_epsilon

        # Hooks for activation/gradient extraction
        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None
        self._hooks: List[Any] = []

    def set_model(self, model: nn.Module) -> None:
        """Set or update the model.

        Args:
            model: PyTorch model
        """
        # Remove old hooks
        self._remove_hooks()

        self._model = model

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _register_hooks(self, target_layer: nn.Module) -> None:
        """Register forward and backward hooks for activation/gradient extraction.

        Args:
            target_layer: Layer to extract from
        """
        self._remove_hooks()

        def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
            self._activations = output

        def backward_hook(
            module: nn.Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]
        ):
            self._gradients = grad_output[0]

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def _get_target_layer(self) -> nn.Module:
        """Get target layer for feature extraction.

        Returns:
            Target layer module

        Raises:
            RuntimeError: If model not set or layer not found
        """
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        # If layer specified in config, use it
        if self.config.gradcam_target_layer is not None:
            # Navigate to layer by name
            parts = self.config.gradcam_target_layer.split(".")
            layer = self._model
            for part in parts:
                if not hasattr(layer, part):
                    raise RuntimeError(
                        f"Layer '{self.config.gradcam_target_layer}' not found in model"
                    )
                layer = getattr(layer, part)
            return layer

        # Auto-detect last convolutional layer
        last_conv = None
        for module in self._model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise RuntimeError("No convolutional layers found in model")

        return last_conv

    def _generate_fgsm_perturbation(
        self, images: Tensor, labels: Tensor, epsilon: float
    ) -> Tensor:
        """Generate FGSM adversarial perturbation.

        Args:
            images: Input images (B, C, H, W)
            labels: True labels (B,)
            epsilon: Perturbation magnitude

        Returns:
            Adversarial images (B, C, H, W)
        """
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        images_copy = images.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self._model(images_copy)

        # Compute loss
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Generate perturbation
        perturbation = epsilon * images_copy.grad.sign()

        # Create adversarial examples
        adv_images = images + perturbation
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach()

    def _generate_gradcam_heatmap(self, images: Tensor, labels: Tensor) -> Tensor:
        """Generate Grad-CAM heatmap.

        Args:
            images: Input images (B, C, H, W)
            labels: Target class labels (B,)

        Returns:
            Heatmap (B, H, W) normalized to [0, 1]
        """
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        batch_size = images.size(0)
        target_layer = self._get_target_layer()
        self._register_hooks(target_layer)

        # Forward pass
        self._model.eval()
        outputs = self._model(images)

        # Select target class scores
        target_scores = outputs[torch.arange(batch_size), labels]

        # Backward pass to get gradients
        self._model.zero_grad()
        target_scores.sum().backward(retain_graph=True)

        # Get activations and gradients
        activations = self._activations  # (B, C, H, W)
        gradients = self._gradients  # (B, C, H, W)

        # Compute weights: global average pool of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        # Compute weighted combination
        cam = torch.sum(weights * activations, dim=1)  # (B, H, W)

        # Apply ReLU and normalize
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)
        cam_max = cam.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Resize to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return cam

    def _extract_features_for_tcav(
        self, images: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Extract activations and gradients for TCAV.

        Args:
            images: Input images (B, C, H, W)
            labels: Target class labels (B,)

        Returns:
            Tuple of (activations, gradients), both (B, D) after global avg pool
        """
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        batch_size = images.size(0)
        target_layer = self._get_target_layer()
        self._register_hooks(target_layer)

        # Forward pass
        self._model.eval()
        outputs = self._model(images)

        # Select target class scores
        target_scores = outputs[torch.arange(batch_size), labels]

        # Backward pass
        self._model.zero_grad()
        target_scores.sum().backward(retain_graph=True)

        # Get activations and gradients
        activations = self._activations  # (B, C, H, W)
        gradients = self._gradients  # (B, C, H, W)

        # Global average pooling
        activations_pooled = (
            F.adaptive_avg_pool2d(activations, (1, 1)).squeeze(-1).squeeze(-1)
        )
        gradients_pooled = (
            F.adaptive_avg_pool2d(gradients, (1, 1)).squeeze(-1).squeeze(-1)
        )

        return activations_pooled, gradients_pooled

    def compute_stability_only(
        self, heatmap_clean: Tensor, heatmap_adv: Tensor
    ) -> Tensor:
        """Compute only stability loss (L_stab).

        Useful for ablation studies or when concept loss is not needed.

        Args:
            heatmap_clean: Clean heatmap (B, C, H, W) or (B, H, W)
            heatmap_adv: Adversarial heatmap (B, C, H, W) or (B, H, W)

        Returns:
            Stability loss
        """
        return self.ssim_loss(heatmap_clean, heatmap_adv)

    def compute_concept_only(
        self, activations: Tensor, gradients: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute only concept loss (L_concept).

        Useful for ablation studies or when stability loss is not needed.

        Args:
            activations: Feature activations (B, D)
            gradients: Feature gradients (B, D)

        Returns:
            Tuple of (concept_loss, metrics_dict)
        """
        return self.concept_loss(activations, gradients)

    def forward(
        self,
        images: Tensor,
        labels: Tensor,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """Compute explanation loss.

        Full pipeline:
        1. Generate adversarial examples using FGSM
        2. Generate Grad-CAM heatmaps for clean and adversarial images
        3. Compute SSIM stability loss
        4. Extract features and compute TCAV concept loss
        5. Combine: L_expl = L_stab + γ × L_concept

        Args:
            images: Input images (B, C, H, W), range [0, 1]
            labels: True class labels (B,)
            return_components: If True, return (loss, metrics_dict)

        Returns:
            If return_components=False: total loss (scalar)
            If return_components=True: (total loss, metrics_dict)

        Raises:
            RuntimeError: If model not set
        """
        if self._model is None:
            raise RuntimeError(
                "Model not set. Call set_model() or pass model in constructor."
            )

        # Set model to eval mode
        was_training = self._model.training
        self._model.eval()

        # 1. Generate adversarial examples
        adv_images = self._generate_fgsm_perturbation(images, labels, self.fgsm_epsilon)

        # 2. Generate Grad-CAM heatmaps
        heatmap_clean = self._generate_gradcam_heatmap(images, labels)
        heatmap_adv = self._generate_gradcam_heatmap(adv_images, labels)

        # 3. Compute stability loss
        loss_stability = self.ssim_loss(heatmap_clean, heatmap_adv)

        # 4. Extract features for TCAV
        activations, gradients = self._extract_features_for_tcav(images, labels)

        # 5. Compute concept loss
        loss_concept, concept_metrics = self.concept_loss(activations, gradients)

        # 6. Combine losses
        loss_total = loss_stability + self.gamma * loss_concept

        # Restore model training state
        if was_training:
            self._model.train()

        if return_components:
            metrics = {
                "loss_stability": loss_stability.item(),
                "loss_concept": loss_concept.item(),
                "loss_total": loss_total.item(),
                "ssim_score": 1.0 - loss_stability.item(),
                **concept_metrics,
            }
            return loss_total, metrics
        else:
            return loss_total

    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


# ============================================================================
# Factory Functions
# ============================================================================


def create_explanation_loss(
    model: Optional[nn.Module] = None,
    artifact_cavs: Optional[List[Tensor]] = None,
    medical_cavs: Optional[List[Tensor]] = None,
    gamma: float = 0.5,
    tau_artifact: float = 0.3,
    tau_medical: float = 0.5,
    lambda_medical: float = 0.5,
    fgsm_epsilon: float = 2.0 / 255.0,
    use_ms_ssim: bool = False,
    **kwargs,
) -> ExplanationLoss:
    """Factory function to create ExplanationLoss.

    Args:
        model: PyTorch model (optional)
        artifact_cavs: List of artifact CAV vectors
        medical_cavs: List of medical CAV vectors
        gamma: Weight for concept regularization
        tau_artifact: Penalty threshold for artifact TCAV
        tau_medical: Reward threshold for medical TCAV
        lambda_medical: Weight for medical concept reward
        fgsm_epsilon: FGSM perturbation magnitude
        use_ms_ssim: Use Multi-Scale SSIM
        **kwargs: Additional config parameters

    Returns:
        ExplanationLoss instance
    """
    config = ExplanationLossConfig(
        gamma=gamma,
        tau_artifact=tau_artifact,
        tau_medical=tau_medical,
        lambda_medical=lambda_medical,
        fgsm_epsilon=fgsm_epsilon,
        use_ms_ssim=use_ms_ssim,
        **kwargs,
    )

    return ExplanationLoss(
        model=model,
        config=config,
        artifact_cavs=artifact_cavs,
        medical_cavs=medical_cavs,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def _create_gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create Gaussian kernel for SSIM.

    Args:
        window_size: Size of Gaussian window (must be odd)
        sigma: Standard deviation
        channels: Number of channels
        device: Target device
        dtype: Data type

    Returns:
        Gaussian kernel (channels, 1, window_size, window_size)
    """
    # Create 1D Gaussian
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords -= window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    # Create 2D Gaussian by outer product
    kernel = g.unsqueeze(1) * g.unsqueeze(0)  # (window_size, window_size)

    # Expand to channels
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = kernel.repeat(channels, 1, 1, 1)  # (C, 1, H, W)

    return kernel


def _create_uniform_kernel(
    window_size: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create uniform (box) kernel for SSIM.

    Args:
        window_size: Size of window
        channels: Number of channels
        device: Target device
        dtype: Data type

    Returns:
        Uniform kernel (channels, 1, window_size, window_size)
    """
    kernel = torch.ones(
        channels, 1, window_size, window_size, dtype=dtype, device=device
    )
    kernel = kernel / (window_size * window_size)
    return kernel


def verify_gradient_flow(
    loss_fn: ExplanationLoss,
    batch_size: int = 4,
    image_size: int = 56,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, bool]:
    """Verify that gradients flow correctly through explanation loss.

    Args:
        loss_fn: ExplanationLoss instance
        batch_size: Batch size for test
        image_size: Image size for test
        num_classes: Number of classes
        device: Device to run test on

    Returns:
        Dictionary with gradient flow results:
            - ssim_grad_flow: SSIM gradient flow status
            - concept_grad_flow: Concept loss gradient flow status
            - combined_grad_flow: Combined loss gradient flow status
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy data
    images = torch.randn(
        batch_size, 3, image_size, image_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Test SSIM gradient flow
    heatmap1 = torch.randn(
        batch_size, 1, image_size, image_size, device=device, requires_grad=True
    )
    heatmap2 = torch.randn(
        batch_size, 1, image_size, image_size, device=device, requires_grad=True
    )

    ssim_loss = loss_fn.compute_stability_only(heatmap1, heatmap2)
    ssim_loss.backward()

    ssim_grad_flow = (
        heatmap1.grad is not None
        and not torch.isnan(heatmap1.grad).any()
        and (heatmap1.grad.abs().sum() > 0)
    )

    # Test concept gradient flow
    feat_dim = 256
    activations = torch.randn(batch_size, feat_dim, device=device, requires_grad=True)
    gradients = torch.randn(batch_size, feat_dim, device=device, requires_grad=True)

    concept_loss, _ = loss_fn.compute_concept_only(activations, gradients)

    if concept_loss.requires_grad:
        concept_loss.backward()
        concept_grad_flow = (
            gradients.grad is not None and not torch.isnan(gradients.grad).any()
        )
    else:
        concept_grad_flow = True  # No CAVs, expected

    # Test combined gradient flow (if model is set)
    combined_grad_flow = False
    if loss_fn._model is not None:
        try:
            images_test = torch.randn(
                2, 3, image_size, image_size, device=device, requires_grad=True
            )
            labels_test = torch.randint(0, num_classes, (2,), device=device)

            total_loss = loss_fn(images_test, labels_test)
            total_loss.backward()

            combined_grad_flow = (
                images_test.grad is not None and not torch.isnan(images_test.grad).any()
            )
        except Exception as e:
            logger.warning(f"Combined gradient flow test failed: {e}")
            combined_grad_flow = False

    return {
        "ssim_grad_flow": ssim_grad_flow,
        "concept_grad_flow": concept_grad_flow,
        "combined_grad_flow": combined_grad_flow,
    }


def benchmark_computational_overhead(
    loss_fn: ExplanationLoss,
    batch_size: int = 8,
    image_size: int = 224,
    num_iterations: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Benchmark computational overhead of explanation loss.

    Args:
        loss_fn: ExplanationLoss instance
        batch_size: Batch size for benchmark
        image_size: Image size for benchmark
        num_iterations: Number of iterations to average
        device: Device to run benchmark on

    Returns:
        Dictionary with timing results (in milliseconds):
            - ssim_time_ms: Time for SSIM computation
            - concept_time_ms: Time for concept loss computation
            - total_time_ms: Total time for combined loss
    """
    import time

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warmup
    heatmap1 = torch.randn(batch_size, 1, image_size, image_size, device=device)
    heatmap2 = torch.randn(batch_size, 1, image_size, image_size, device=device)
    _ = loss_fn.compute_stability_only(heatmap1, heatmap2)

    # Benchmark SSIM
    ssim_times = []
    for _ in range(num_iterations):
        heatmap1 = torch.randn(batch_size, 1, image_size, image_size, device=device)
        heatmap2 = torch.randn(batch_size, 1, image_size, image_size, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = loss_fn.compute_stability_only(heatmap1, heatmap2)

        if device.type == "cuda":
            torch.cuda.synchronize()

        ssim_times.append((time.perf_counter() - start) * 1000)

    # Benchmark concept loss
    feat_dim = 2048
    concept_times = []
    for _ in range(num_iterations):
        activations = torch.randn(batch_size, feat_dim, device=device)
        gradients = torch.randn(batch_size, feat_dim, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _, _ = loss_fn.compute_concept_only(activations, gradients)

        if device.type == "cuda":
            torch.cuda.synchronize()

        concept_times.append((time.perf_counter() - start) * 1000)

    return {
        "ssim_time_ms": float(np.mean(ssim_times)),
        "concept_time_ms": float(np.mean(concept_times)),
        "total_time_ms": float(np.mean(ssim_times)) + float(np.mean(concept_times)),
    }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main classes
    "ExplanationLoss",
    "ExplanationLossConfig",
    "SSIMStabilityLoss",
    "SSIMKernelType",
    "TCavConceptLoss",
    # Factory functions
    "create_explanation_loss",
    # Utility functions
    "verify_gradient_flow",
    "benchmark_computational_overhead",
]
