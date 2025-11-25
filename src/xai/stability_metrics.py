"""
Production-Grade Explanation Stability Metrics for XAI.

This module provides comprehensive metrics for measuring explanation stability
in the context of adversarial robustness and cross-site generalization. These
metrics are crucial for RQ2 (Hypothesis H2) of the dissertation.

Mathematical Foundation
-----------------------
Stability metrics quantify how consistent explanations remain under:
1. Adversarial perturbations (ε-bounded attacks)
2. Domain shifts (cross-site datasets)
3. Model uncertainty (ensemble predictions)

Key Metrics:
    - SSIM: Structural similarity (perceptual consistency)
    - MS-SSIM: Multi-scale structural similarity (robust to scale)
    - Spearman ρ: Rank correlation (attribution ordering)
    - L2 Distance: Euclidean distance (normalized)
    - Cosine Similarity: Angular similarity (direction)

Research Hypothesis (H2)
------------------------
Tri-objective training produces explanations with:
    SSIM(GradCAM_clean, GradCAM_adv) ≥ 0.75

Target: Demonstrate explanation stability improves with λ_expl > 0

Integration:
    - Compatible with src.xai.gradcam heatmaps
    - Used in src.losses.tri_objective.SSIMLoss
    - Logged via src.training.tri_objective_trainer
    - Evaluated in Phase 6.5 robustness experiments

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 6.2 - Explanation Stability Metrics
Date: November 25, 2025
Version: 6.2.0 (Production)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class StabilityMetricsConfig:
    """Configuration for stability metrics computation.

    Attributes:
        ssim_window_size: Gaussian window size for SSIM (default: 11)
        ssim_sigma: Standard deviation for Gaussian window (default: 1.5)
        ssim_data_range: Dynamic range of heatmaps (default: 1.0)
        ms_ssim_weights: Weights for multi-scale SSIM levels
        normalize_heatmaps: Normalize to [0, 1] before computation
        use_cuda: Use GPU acceleration if available
        epsilon: Small constant for numerical stability (default: 1e-8)
    """

    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    ssim_data_range: float = 1.0
    ms_ssim_weights: Optional[List[float]] = None
    normalize_heatmaps: bool = True
    use_cuda: bool = True
    epsilon: float = 1e-8

    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.ms_ssim_weights is None:
            # Default MS-SSIM weights from Wang et al. (2003)
            self.ms_ssim_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        # Validate window size
        if self.ssim_window_size % 2 == 0:
            raise ValueError(
                f"SSIM window size must be odd, got {self.ssim_window_size}"
            )

        # Validate weights
        if not np.isclose(sum(self.ms_ssim_weights), 1.0):
            logger.warning(
                f"MS-SSIM weights sum to {sum(self.ms_ssim_weights)}, "
                "should sum to 1.0. Normalizing."
            )
            total = sum(self.ms_ssim_weights)
            self.ms_ssim_weights = [w / total for w in self.ms_ssim_weights]


# ============================================================================
# SSIM Implementation (Differentiable)
# ============================================================================


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) for heatmap comparison.

    SSIM measures perceptual similarity between two images by comparing:
    - Luminance: Mean intensities
    - Contrast: Standard deviations
    - Structure: Correlation coefficient

    Mathematical Formula:
        SSIM(x, y) = (2μ_x μ_y + c1)(2σ_xy + c2) /
                     ((μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2))

    Where:
        - μ: mean (computed via Gaussian window)
        - σ: standard deviation
        - σ_xy: covariance
        - c1, c2: constants for numerical stability

    Reference:
        Wang et al., "Image Quality Assessment: From Error Visibility to
        Structural Similarity", IEEE TIP 2004

    Parameters
    ----------
    window_size : int
        Size of Gaussian window (default: 11)
    sigma : float
        Standard deviation of Gaussian (default: 1.5)
    channel : int
        Number of channels (1 for grayscale heatmaps)
    data_range : float
        Dynamic range of inputs (default: 1.0)
    reduction : str
        How to reduce batch dimension ('mean', 'sum', 'none')

    Returns
    -------
    ssim_value : Tensor
        SSIM ∈ [-1, 1], where 1 = identical, -1 = inverse
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
        self.register_buffer("window", self._create_window(window_size, sigma, channel))

        # SSIM constants (Wang et al., 2004)
        self.c1 = (0.01 * data_range) ** 2  # Luminance stability
        self.c2 = (0.03 * data_range) ** 2  # Contrast stability

    def _create_window(self, size: int, sigma: float, channel: int) -> Tensor:
        """
        Create 2D Gaussian window for SSIM computation.

        Parameters
        ----------
        size : int
            Window size (must be odd)
        sigma : float
            Standard deviation of Gaussian
        channel : int
            Number of channels to expand window to

        Returns
        -------
        window : Tensor
            Gaussian window, shape (channel, 1, size, size)
        """
        # 1D Gaussian
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        # 2D Gaussian (outer product)
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)

        # Expand to (C, 1, H, W) for conv2d
        window = window_2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, size, size).contiguous()

        return window

    def _apply_window(self, x: Tensor, window: Tensor) -> Tensor:
        """Apply Gaussian window via depthwise convolution."""
        # Conv2d with groups=channel implements depthwise convolution
        return F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute SSIM between two tensors.

        Parameters
        ----------
        x, y : Tensor
            Input tensors, shape (B, C, H, W)
            Should be normalized to [0, data_range]

        Returns
        -------
        ssim : Tensor
            SSIM value(s), shape depends on reduction:
            - 'mean': scalar
            - 'sum': scalar
            - 'none': (B,)
        """
        # Validate inputs
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: x={x.shape}, y={y.shape}")

        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.dim()}D")

        # Move window to input device if needed
        # (register_buffer handles device when module.to(device) is called,
        # but this ensures compatibility when module isn't explicitly moved)
        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        # Compute statistics via Gaussian window
        mu_x = self._apply_window(x, self.window)
        mu_y = self._apply_window(y, self.window)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = self._apply_window(x * x, self.window) - mu_x_sq
        sigma_y_sq = self._apply_window(y * y, self.window) - mu_y_sq
        sigma_xy = self._apply_window(x * y, self.window) - mu_xy

        # SSIM formula
        numerator = (2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )

        ssim_map = numerator / denominator

        # Reduce to per-sample SSIM
        # Average over spatial dimensions and channels
        ssim_per_sample = ssim_map.mean(dim=[1, 2, 3])

        # Apply reduction
        if self.reduction == "mean":
            return ssim_per_sample.mean()
        elif self.reduction == "sum":
            return ssim_per_sample.sum()
        elif self.reduction == "none":
            return ssim_per_sample
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ============================================================================
# Multi-Scale SSIM (MS-SSIM)
# ============================================================================


class MultiScaleSSIM(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM).

    MS-SSIM computes SSIM at multiple scales (resolutions) and combines them
    using learned or fixed weights. This provides robustness to scale changes
    and better perceptual similarity measurement.

    Process:
        1. Compute SSIM at original resolution
        2. Downsample by 2x and compute SSIM
        3. Repeat for N scales
        4. Combine: MS-SSIM = Π_i (SSIM_i)^w_i

    Reference:
        Wang et al., "Multi-scale structural similarity for image quality
        assessment", Asilomar Conference 2003

    Parameters
    ----------
    window_size : int
        Gaussian window size (default: 11)
    sigma : float
        Standard deviation (default: 1.5)
    channel : int
        Number of channels (default: 1)
    data_range : float
        Dynamic range (default: 1.0)
    weights : List[float], optional
        Weights for each scale (should sum to 1.0)
    reduction : str
        Reduction method ('mean', 'sum', 'none')

    Returns
    -------
    ms_ssim : Tensor
        Multi-scale SSIM value
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 1,
        data_range: float = 1.0,
        weights: Optional[List[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()

        if weights is None:
            # Default weights from Wang et al. (2003)
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_scales = len(weights)

        # Create SSIM module
        self.ssim = SSIM(
            window_size=window_size,
            sigma=sigma,
            channel=channel,
            data_range=data_range,
            reduction="none",  # We'll handle reduction ourselves
        )

        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute MS-SSIM between two tensors.

        Parameters
        ----------
        x, y : Tensor
            Input tensors, shape (B, C, H, W)

        Returns
        -------
        ms_ssim : Tensor
            Multi-scale SSIM value
        """
        # Move weights to same device
        if self.weights.device != x.device:
            self.weights = self.weights.to(x.device)

        # Check minimum size (need to downsample num_scales times)
        min_size = 2 ** (self.num_scales - 1) * self.ssim.window_size
        if x.shape[-1] < min_size or x.shape[-2] < min_size:
            logger.warning(
                f"Input size {x.shape[-2:]} too small for {self.num_scales} scales. "
                f"Minimum size: {min_size}. Using fewer scales."
            )
            # Compute max scales that fit
            max_scales = int(np.log2(min(x.shape[-2:]) / self.ssim.window_size)) + 1
            num_scales = min(max_scales, self.num_scales)
            weights = self.weights[:num_scales]
            weights = weights / weights.sum()  # Renormalize
        else:
            num_scales = self.num_scales
            weights = self.weights

        # Compute SSIM at multiple scales
        mssim_values = []
        for i in range(num_scales):
            # Compute SSIM at current scale
            ssim_val = self.ssim(x, y)
            mssim_values.append(ssim_val)

            # Downsample for next scale (skip for last scale)
            if i < num_scales - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        # Stack and combine with weights
        mssim_stack = torch.stack(mssim_values, dim=0)  # (num_scales, B)

        # Weighted geometric mean: MS-SSIM = Π (SSIM_i)^w_i
        # In log space: log(MS-SSIM) = Σ w_i * log(SSIM_i)
        # Clamp to avoid log(0)
        mssim_stack = torch.clamp(mssim_stack, min=1e-8)
        log_mssim = (weights.unsqueeze(1) * torch.log(mssim_stack)).sum(dim=0)
        ms_ssim = torch.exp(log_mssim)

        # Apply reduction
        if self.reduction == "mean":
            return ms_ssim.mean()
        elif self.reduction == "sum":
            return ms_ssim.sum()
        elif self.reduction == "none":
            return ms_ssim
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ============================================================================
# Rank Correlation (Spearman ρ)
# ============================================================================


def spearman_correlation(
    x: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[float, np.ndarray, Tensor]:
    """
    Compute Spearman rank correlation between explanation heatmaps.

    Spearman ρ measures how similar the ranking of pixel importances are
    between two explanations. Unlike Pearson, it's robust to monotonic
    transformations and outliers.

    **Warning**: This implementation uses scipy and is **NOT differentiable**.
    Cannot be used in loss functions requiring gradient flow. Use SSIM,
    L2 distance, or cosine similarity for differentiable alternatives.

    Process:
        1. Flatten heatmaps to 1D
        2. Rank pixels by attribution value
        3. Compute Pearson correlation on ranks

    Mathematical Formula:
        ρ = 1 - (6 Σ d_i²) / (n(n² - 1))

    Where d_i is the difference in ranks for pixel i.

    Parameters
    ----------
    x, y : Tensor or np.ndarray
        Explanation heatmaps, shape (B, C, H, W) or (B, H, W)
        Will be flattened automatically
    reduction : str
        How to reduce batch dimension:
        - 'mean': average across batch
        - 'none': return per-sample correlations

    Returns
    -------
    rho : float or np.ndarray or Tensor
        Spearman correlation coefficient(s)
        Range: [-1, 1] where 1 = perfect agreement

    Examples
    --------
    >>> x = torch.randn(4, 1, 14, 14)  # 4 heatmaps
    >>> y = x + 0.1 * torch.randn_like(x)  # Slightly perturbed
    >>> rho = spearman_correlation(x, y)
    >>> print(f"Correlation: {rho:.3f}")  # Should be close to 1.0
    """
    # Convert to numpy for scipy
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()

    # Flatten spatial dimensions
    if x.ndim == 4:
        x = x.reshape(x.shape[0], -1)  # (B, H*W*C)
    elif x.ndim == 3:
        x = x.reshape(x.shape[0], -1)  # (B, H*W)

    if y.ndim == 4:
        y = y.reshape(y.shape[0], -1)
    elif y.ndim == 3:
        y = y.reshape(y.shape[0], -1)

    # Compute per-sample correlations
    batch_size = x.shape[0]
    correlations = []

    for i in range(batch_size):
        # Compute Spearman correlation for this sample
        rho, _ = scipy_stats.spearmanr(x[i], y[i])

        # Handle NaN (occurs when variance is zero)
        if np.isnan(rho):
            rho = 1.0  # Identical constant maps

        correlations.append(rho)

    correlations = np.array(correlations)

    # Apply reduction
    if reduction == "mean":
        return float(correlations.mean())
    elif reduction == "none":
        return correlations
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================================
# L2 Distance (Normalized)
# ============================================================================


def normalized_l2_distance(
    x: Tensor,
    y: Tensor,
    reduction: str = "mean",
    epsilon: float = 1e-8,
) -> Tensor:
    """
    Compute normalized L2 distance between explanation heatmaps.

    The L2 distance measures pixel-wise Euclidean distance between
    explanations. We normalize by the L2 norms to make it scale-invariant.

    Mathematical Formula:
        d = ||x - y||_2 / (||x||_2 + ||y||_2 + ε)

    Parameters
    ----------
    x, y : Tensor
        Explanation heatmaps, shape (B, C, H, W) or (B, H, W)
    reduction : str
        How to reduce batch dimension ('mean', 'sum', 'none')
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    distance : Tensor
        Normalized L2 distance ∈ [0, 1]
        0 = identical, 1 = maximally different

    Examples
    --------
    >>> x = torch.randn(4, 1, 14, 14)
    >>> y = torch.randn(4, 1, 14, 14)
    >>> dist = normalized_l2_distance(x, y)
    >>> print(f"Distance: {dist:.3f}")
    """
    # Flatten spatial dimensions
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    # Compute L2 norms
    norm_x = torch.norm(x_flat, p=2, dim=1)
    norm_y = torch.norm(y_flat, p=2, dim=1)

    # Compute L2 distance
    diff = x_flat - y_flat
    l2_dist = torch.norm(diff, p=2, dim=1)

    # Normalize
    normalized_dist = l2_dist / (norm_x + norm_y + epsilon)

    # Apply reduction
    if reduction == "mean":
        return normalized_dist.mean()
    elif reduction == "sum":
        return normalized_dist.sum()
    elif reduction == "none":
        return normalized_dist
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================================
# Cosine Similarity
# ============================================================================


def cosine_similarity(
    x: Tensor,
    y: Tensor,
    reduction: str = "mean",
    epsilon: float = 1e-8,
) -> Tensor:
    """
    Compute cosine similarity between explanation heatmaps.

    Cosine similarity measures the angle between two vectors, indicating
    directional similarity regardless of magnitude.

    Mathematical Formula:
        cos(θ) = (x · y) / (||x||_2 * ||y||_2)

    Parameters
    ----------
    x, y : Tensor
        Explanation heatmaps, shape (B, C, H, W) or (B, H, W)
    reduction : str
        How to reduce batch dimension ('mean', 'sum', 'none')
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    similarity : Tensor
        Cosine similarity ∈ [-1, 1]
        1 = same direction, -1 = opposite, 0 = orthogonal

    Examples
    --------
    >>> x = torch.randn(4, 1, 14, 14)
    >>> y = 2 * x  # Same direction, different magnitude
    >>> sim = cosine_similarity(x, y)
    >>> print(f"Similarity: {sim:.3f}")  # Should be ~1.0
    """
    # Flatten spatial dimensions
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    # Compute dot product
    dot_product = (x_flat * y_flat).sum(dim=1)

    # Compute norms
    norm_x = torch.norm(x_flat, p=2, dim=1)
    norm_y = torch.norm(y_flat, p=2, dim=1)

    # Compute cosine similarity
    cos_sim = dot_product / (norm_x * norm_y + epsilon)

    # Apply reduction
    if reduction == "mean":
        return cos_sim.mean()
    elif reduction == "sum":
        return cos_sim.sum()
    elif reduction == "none":
        return cos_sim
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================================
# Unified Stability Metrics Class
# ============================================================================


class StabilityMetrics:
    """
    Unified class for computing all explanation stability metrics.

    This class provides a consistent interface for evaluating explanation
    stability across different metrics. All metrics are computed efficiently
    in a single pass when possible.

    Attributes:
        config: Configuration for metric computation
        ssim_module: SSIM computation module
        ms_ssim_module: Multi-scale SSIM module

    Example Usage:
        >>> from src.xai.stability_metrics import StabilityMetrics
        >>> metrics = StabilityMetrics()
        >>> heatmap_clean = model.gradcam(image_clean)
        >>> heatmap_adv = model.gradcam(image_adv)
        >>> results = metrics.compute_all(heatmap_clean, heatmap_adv)
        >>> print(f"SSIM: {results['ssim']:.3f}")
        >>> print(f"Spearman: {results['spearman']:.3f}")
    """

    def __init__(self, config: Optional[StabilityMetricsConfig] = None):
        """
        Initialize stability metrics computer.

        Parameters
        ----------
        config : StabilityMetricsConfig, optional
            Configuration for metrics. If None, uses defaults.
        """
        self.config = config or StabilityMetricsConfig()

        # Create SSIM modules
        self.ssim_module = SSIM(
            window_size=self.config.ssim_window_size,
            sigma=self.config.ssim_sigma,
            channel=1,  # Grayscale heatmaps
            data_range=self.config.ssim_data_range,
            reduction="mean",
        )

        self.ms_ssim_module = MultiScaleSSIM(
            window_size=self.config.ssim_window_size,
            sigma=self.config.ssim_sigma,
            channel=1,
            data_range=self.config.ssim_data_range,
            weights=self.config.ms_ssim_weights,
            reduction="mean",
        )

        # Move to GPU if requested
        if self.config.use_cuda and torch.cuda.is_available():
            self.ssim_module = self.ssim_module.cuda()
            self.ms_ssim_module = self.ms_ssim_module.cuda()

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"StabilityMetrics("
            f"window_size={self.config.ssim_window_size}, "
            f"sigma={self.config.ssim_sigma}, "
            f"normalize={self.config.normalize_heatmaps}, "
            f"scales={len(self.config.ms_ssim_weights)})"
        )

    def _preprocess_heatmaps(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Preprocess heatmaps before metric computation.

        - Ensures 4D shape (B, C, H, W)
        - Normalizes to [0, 1] if requested
        - Validates shapes match

        Parameters
        ----------
        x, y : Tensor
            Input heatmaps

        Returns
        -------
        x_prep, y_prep : Tensor
            Preprocessed heatmaps
        """
        # Ensure 4D shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        if y.dim() == 3:
            y = y.unsqueeze(1)

        # Validate shapes
        if x.shape != y.shape:
            raise ValueError(f"Heatmap shapes must match: x={x.shape}, y={y.shape}")

        # Normalize if requested
        if self.config.normalize_heatmaps:
            x = (x - x.min()) / (x.max() - x.min() + self.config.epsilon)
            y = (y - y.min()) / (y.max() - y.min() + self.config.epsilon)

        return x, y

    def compute_ssim(self, x: Tensor, y: Tensor) -> float:
        """
        Compute SSIM between two heatmaps.

        Parameters
        ----------
        x, y : Tensor
            Explanation heatmaps

        Returns
        -------
        ssim : float
            SSIM value ∈ [-1, 1]
        """
        x, y = self._preprocess_heatmaps(x, y)
        ssim_val = self.ssim_module(x, y)
        return float(ssim_val.item())

    def compute_ms_ssim(self, x: Tensor, y: Tensor) -> float:
        """
        Compute multi-scale SSIM between two heatmaps.

        Parameters
        ----------
        x, y : Tensor
            Explanation heatmaps

        Returns
        -------
        ms_ssim : float
            Multi-scale SSIM value
        """
        x, y = self._preprocess_heatmaps(x, y)
        ms_ssim_val = self.ms_ssim_module(x, y)
        return float(ms_ssim_val.item())

    def compute_spearman(self, x: Tensor, y: Tensor) -> float:
        """
        Compute Spearman rank correlation.

        Parameters
        ----------
        x, y : Tensor
            Explanation heatmaps

        Returns
        -------
        rho : float
            Spearman correlation ∈ [-1, 1]
        """
        x, y = self._preprocess_heatmaps(x, y)
        return spearman_correlation(x, y, reduction="mean")

    def compute_l2_distance(self, x: Tensor, y: Tensor) -> float:
        """
        Compute normalized L2 distance.

        Parameters
        ----------
        x, y : Tensor
            Explanation heatmaps

        Returns
        -------
        distance : float
            Normalized L2 distance ∈ [0, 1]
        """
        x, y = self._preprocess_heatmaps(x, y)
        dist = normalized_l2_distance(
            x, y, reduction="mean", epsilon=self.config.epsilon
        )
        return float(dist.item())

    def compute_cosine_similarity(self, x: Tensor, y: Tensor) -> float:
        """
        Compute cosine similarity.

        Parameters
        ----------
        x, y : Tensor
            Explanation heatmaps

        Returns
        -------
        similarity : float
            Cosine similarity ∈ [-1, 1]
        """
        x, y = self._preprocess_heatmaps(x, y)
        sim = cosine_similarity(x, y, reduction="mean", epsilon=self.config.epsilon)
        return float(sim.item())

    def compute_all(
        self,
        x: Tensor,
        y: Tensor,
        include_ms_ssim: bool = True,
    ) -> Dict[str, float]:
        """
        Compute all stability metrics in one call.

        This is the recommended method for comprehensive evaluation.

        Parameters
        ----------
        x, y : Tensor
            Explanation heatmaps to compare
        include_ms_ssim : bool
            Whether to compute MS-SSIM (more expensive)

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary with all computed metrics:
            - 'ssim': Structural similarity
            - 'ms_ssim': Multi-scale SSIM (if requested)
            - 'spearman': Rank correlation
            - 'l2_distance': Normalized L2 distance
            - 'cosine_similarity': Cosine similarity

        Examples
        --------
        >>> metrics = StabilityMetrics()
        >>> results = metrics.compute_all(heatmap1, heatmap2)
        >>> for name, value in results.items():
        ...     print(f"{name}: {value:.4f}")
        """
        results = {}

        # SSIM
        results["ssim"] = self.compute_ssim(x, y)

        # MS-SSIM (optional, more expensive)
        if include_ms_ssim:
            results["ms_ssim"] = self.compute_ms_ssim(x, y)

        # Spearman correlation
        results["spearman"] = self.compute_spearman(x, y)

        # L2 distance
        results["l2_distance"] = self.compute_l2_distance(x, y)

        # Cosine similarity
        results["cosine_similarity"] = self.compute_cosine_similarity(x, y)

        return results


# ============================================================================
# Factory Function
# ============================================================================


def create_stability_metrics(
    config: Optional[StabilityMetricsConfig] = None,
) -> StabilityMetrics:
    """
    Factory function to create StabilityMetrics instance.

    Parameters
    ----------
    config : StabilityMetricsConfig, optional
        Configuration for metrics

    Returns
    -------
    metrics : StabilityMetrics
        Configured stability metrics computer

    Examples
    --------
    >>> from src.xai.stability_metrics import create_stability_metrics
    >>> metrics = create_stability_metrics()
    >>> results = metrics.compute_all(heatmap1, heatmap2)
    """
    return StabilityMetrics(config=config)
