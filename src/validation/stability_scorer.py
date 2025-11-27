"""
Production-Grade Explanation Stability Scorer for Selective Prediction.

This module provides stability scoring functionality for Phase 8.2 of the
tri-objective robust XAI framework. Stability scores quantify how consistent
explanations remain under small adversarial perturbations, enabling safe
selective prediction for clinical deployment.

Mathematical Foundation
-----------------------
Stability scoring measures explanation consistency via:
1. Generate GradCAM heatmap on clean image
2. Generate small perturbation (FGSM ε=2/255)
3. Generate GradCAM heatmap on perturbed image
4. Compute SSIM between heatmaps
5. Stability score = SSIM value [0, 1]

Higher SSIM indicates more stable explanations, which correlates with:
- Higher model confidence on correct predictions
- Better generalization across domains
- Safer predictions for clinical use

Research Integration
--------------------
This module implements the stability component for RQ3:
    "Can multi-signal gating enable safe selective prediction?"

Target: Combined confidence + stability gating improves accuracy by
        ≥4pp at 90% coverage vs. confidence-only baseline

Integration:
    - Uses src.xai.gradcam for explanation generation
    - Uses src.xai.stability_metrics for SSIM computation
    - Uses src.attacks.fgsm for adversarial perturbation
    - Complements src.validation.confidence_scorer
    - Part of Phase 8 selective prediction pipeline

Key Classes:
    StabilityMethod: Enum for stability computation methods
    StabilityScore: Dataclass for stability results
    StabilityScorer: Main scorer with SSIM-based stability

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 8.2 - Stability Scoring
Date: November 27, 2025
Version: 8.2.0 (Production)
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

from src.attacks.fgsm import FGSM, FGSMConfig
from src.xai.gradcam import GradCAM, GradCAMConfig
from src.xai.stability_metrics import StabilityMetrics, StabilityMetricsConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class StabilityMethod(Enum):
    """Available stability computation methods.

    Attributes:
        SSIM: Structural Similarity Index (default, recommended)
        SPEARMAN: Spearman rank correlation
        L2: Normalized L2 distance
        COSINE: Cosine similarity
        MS_SSIM: Multi-scale SSIM (more robust to scale changes)
    """

    SSIM = "ssim"
    SPEARMAN = "spearman"
    L2 = "l2"
    COSINE = "cosine"
    MS_SSIM = "ms_ssim"


# Default perturbation magnitude for FGSM (2/255 as per dissertation)
DEFAULT_EPSILON = 2.0 / 255.0


# ============================================================================
# Stability Score Dataclass
# ============================================================================


@dataclass
class StabilityScore:
    """Container for stability scoring results.

    Attributes:
        stability: Stability score in [0, 1], higher = more stable
        instability: Instability score (1 - stability)
        method: Stability computation method used
        clean_explanation: GradCAM heatmap on clean image (H, W)
        perturbed_explanation: GradCAM heatmap on perturbed image (H, W)
        perturbation_norm: L∞ norm of perturbation applied
        epsilon: FGSM epsilon used for perturbation
        metadata: Additional information (e.g., all metrics, timings)
    """

    stability: float
    instability: float
    method: StabilityMethod
    clean_explanation: np.ndarray
    perturbed_explanation: np.ndarray
    perturbation_norm: float
    epsilon: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize stability score."""
        # Clip stability to [0, 1] for numerical stability
        self.stability = float(np.clip(self.stability, 0.0, 1.0))

        # Ensure instability is consistent
        if not np.isclose(self.instability, 1.0 - self.stability):
            self.instability = 1.0 - self.stability

        # Validate arrays
        if self.clean_explanation.ndim != 2:
            raise ValueError(
                f"clean_explanation must be 2D (H, W), got {self.clean_explanation.ndim}D"
            )
        if self.perturbed_explanation.ndim != 2:
            raise ValueError(
                f"perturbed_explanation must be 2D (H, W), got {self.perturbed_explanation.ndim}D"
            )
        if self.clean_explanation.shape != self.perturbed_explanation.shape:
            raise ValueError(
                f"Explanation shapes must match: {self.clean_explanation.shape} vs "
                f"{self.perturbed_explanation.shape}"
            )

    def is_stable(self, threshold: float = 0.75) -> bool:
        """Check if explanation is stable above threshold.

        Args:
            threshold: Stability threshold (default: 0.75 from H2)

        Returns:
            True if stability >= threshold
        """
        return self.stability >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary with all score information
        """
        return {
            "stability": float(self.stability),
            "instability": float(self.instability),
            "method": self.method.value,
            "perturbation_norm": float(self.perturbation_norm),
            "epsilon": float(self.epsilon),
            "explanation_shape": self.clean_explanation.shape,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"StabilityScore(stability={self.stability:.4f}, "
            f"method={self.method.value}, epsilon={self.epsilon:.6f})"
        )


# ============================================================================
# Base Stability Scorer
# ============================================================================


class BaseStabilityScorer:
    """Base class for stability scorers.

    Provides common functionality for generating explanations and
    computing stability across different methods.

    Args:
        model: PyTorch model to generate explanations for
        epsilon: FGSM perturbation magnitude (default: 2/255)
        target_layers: GradCAM target layers (default: ["layer4"])
        device: Computation device (auto-detected if None)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = DEFAULT_EPSILON,
        target_layers: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize base stability scorer."""
        self.model = model
        self.epsilon = epsilon
        self.target_layers = target_layers or ["layer4"]
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize GradCAM
        self.gradcam_config = GradCAMConfig(
            target_layers=self.target_layers,
            use_cuda=self.device.type == "cuda",
            output_size=None,  # Will be set based on input size
        )
        self.gradcam = GradCAM(self.model, self.gradcam_config)

        # Initialize FGSM attack
        self.fgsm_config = FGSMConfig(
            epsilon=self.epsilon,
            targeted=False,
            clip_min=0.0,
            clip_max=1.0,
        )
        self.fgsm = FGSM(self.fgsm_config)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Initialized {self.__class__.__name__} with epsilon={epsilon:.6f}, "
            f"device={self.device}, target_layers={self.target_layers}"
        )

    def _generate_explanation(
        self,
        x: Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate GradCAM explanation for input.

        Args:
            x: Input tensor (1, C, H, W) or (C, H, W)
            class_idx: Target class index (None = predicted class)

        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        # Ensure 4D input (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D")

        # Move to device
        x = x.to(self.device)

        # Generate heatmap
        with torch.no_grad():
            # Get predicted class if not provided
            if class_idx is None:
                outputs = self.model(x)
                class_idx = outputs.argmax(dim=1).item()

        # Generate GradCAM heatmap
        heatmap = self.gradcam.generate_heatmap(x, class_idx=class_idx)

        return heatmap

    def _generate_perturbation(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Generate FGSM adversarial perturbation.

        Args:
            x: Clean input (1, C, H, W)
            y: True label (1,) or scalar

        Returns:
            Perturbed input (1, C, H, W)
        """
        # Ensure correct shapes
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        elif y.ndim == 0:
            y = y.unsqueeze(0)

        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)

        # Generate adversarial example
        x_adv = self.fgsm.generate(self.model, x, y)

        return x_adv

    def _compute_perturbation_norm(
        self,
        x_clean: Tensor,
        x_adv: Tensor,
    ) -> float:
        """Compute L∞ norm of perturbation.

        Args:
            x_clean: Clean input
            x_adv: Perturbed input

        Returns:
            Maximum absolute difference
        """
        delta = (x_adv - x_clean).abs()
        return float(delta.max().item())


# ============================================================================
# SSIM Stability Scorer
# ============================================================================


class SSIMStabilityScorer(BaseStabilityScorer):
    """Stability scorer using SSIM metric.

    This is the recommended method for explanation stability scoring.
    SSIM (Structural Similarity Index) measures perceptual similarity
    and is robust to small pixel-wise differences.

    Args:
        model: PyTorch model to evaluate
        epsilon: FGSM epsilon (default: 2/255)
        target_layers: GradCAM layers (default: ["layer4"])
        use_ms_ssim: Use multi-scale SSIM (default: False)
        ssim_window_size: Window size for SSIM (default: 11)
        device: Computation device

    Example:
        >>> scorer = SSIMStabilityScorer(model, epsilon=2/255)
        >>> score = scorer(image, label)
        >>> print(f"Stability: {score.stability:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = DEFAULT_EPSILON,
        target_layers: Optional[List[str]] = None,
        use_ms_ssim: bool = False,
        ssim_window_size: int = 11,
        device: Optional[torch.device] = None,
    ):
        """Initialize SSIM stability scorer."""
        super().__init__(model, epsilon, target_layers, device)

        self.use_ms_ssim = use_ms_ssim
        self.ssim_window_size = ssim_window_size

        # Initialize stability metrics
        self.metrics_config = StabilityMetricsConfig(
            ssim_window_size=ssim_window_size,
            use_cuda=self.device.type == "cuda",
        )
        self.metrics = StabilityMetrics(self.metrics_config)

        logger.info(
            f"Initialized SSIMStabilityScorer with "
            f"use_ms_ssim={use_ms_ssim}, window_size={ssim_window_size}"
        )

    def __call__(
        self,
        x: Tensor,
        y: Union[int, Tensor],
        class_idx: Optional[int] = None,
        return_all_metrics: bool = False,
    ) -> StabilityScore:
        """Compute stability score for input.

        Args:
            x: Input image (C, H, W) or (1, C, H, W)
            y: True label (scalar or tensor)
            class_idx: Target class for explanation (None = predicted)
            return_all_metrics: Include all stability metrics in metadata

        Returns:
            StabilityScore with stability value and explanations
        """
        # Ensure correct input format
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        elif y.ndim == 0:
            y = y.unsqueeze(0)

        # Step 1: Generate explanation on clean image
        heatmap_clean = self._generate_explanation(x, class_idx)

        # Step 2: Generate adversarial perturbation
        x_adv = self._generate_perturbation(x, y)

        # Step 3: Generate explanation on perturbed image
        heatmap_adv = self._generate_explanation(x_adv, class_idx)

        # Step 4: Compute SSIM between explanations
        # Convert to tensors (B, C, H, W) format
        h_clean = torch.from_numpy(heatmap_clean).unsqueeze(0).unsqueeze(0)
        h_adv = torch.from_numpy(heatmap_adv).unsqueeze(0).unsqueeze(0)
        h_clean = h_clean.to(self.device)
        h_adv = h_adv.to(self.device)

        # Compute SSIM
        if self.use_ms_ssim:
            ssim_value = self.metrics.compute_ms_ssim(h_clean, h_adv)
        else:
            ssim_value = self.metrics.compute_ssim(h_clean, h_adv)

        # Compute perturbation norm
        pert_norm = self._compute_perturbation_norm(x, x_adv)

        # Prepare metadata
        metadata = {
            "method_variant": "MS-SSIM" if self.use_ms_ssim else "SSIM",
            "window_size": self.ssim_window_size,
        }

        # Optionally compute all metrics
        if return_all_metrics:
            all_metrics = self.metrics.compute_all(h_clean, h_adv)
            # Map to concise keys for consistency
            metadata["all_metrics"] = {
                "ssim": all_metrics["ssim"],
                "ms_ssim": all_metrics.get("ms_ssim", 0.0),
                "spearman": all_metrics["spearman"],
                "l2": all_metrics["l2_distance"],
                "cosine": all_metrics["cosine_similarity"],
            }

        # Step 5: Return stability score
        return StabilityScore(
            stability=ssim_value,
            instability=1.0 - ssim_value,
            method=(
                StabilityMethod.MS_SSIM if self.use_ms_ssim else StabilityMethod.SSIM
            ),
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=pert_norm,
            epsilon=self.epsilon,
            metadata=metadata,
        )

    def batch_score(
        self,
        inputs: Tensor,
        labels: Tensor,
        batch_size: int = 32,
        return_all_metrics: bool = False,
    ) -> List[StabilityScore]:
        """Compute stability scores for batch of inputs.

        Args:
            inputs: Input images (N, C, H, W)
            labels: True labels (N,)
            batch_size: Processing batch size (default: 32)
            return_all_metrics: Include all metrics in metadata

        Returns:
            List of StabilityScore objects
        """
        if inputs.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {inputs.ndim}D")
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Batch size mismatch: inputs={inputs.shape[0]}, "
                f"labels={labels.shape[0]}"
            )
        if inputs.shape[0] == 0:
            raise ValueError("Cannot compute scores for empty batch")

        scores = []
        num_samples = inputs.shape[0]

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_inputs = inputs[i:end_idx]
            batch_labels = labels[i:end_idx]

            # Process each sample in batch
            for j in range(batch_inputs.shape[0]):
                score = self(
                    batch_inputs[j],
                    batch_labels[j],
                    return_all_metrics=return_all_metrics,
                )
                scores.append(score)

        return scores


# ============================================================================
# Alternative Stability Scorers
# ============================================================================


class SpearmanStabilityScorer(BaseStabilityScorer):
    """Stability scorer using Spearman rank correlation.

    Spearman ρ measures rank correlation between explanations,
    which is robust to monotonic transformations.

    Args:
        model: PyTorch model to evaluate
        epsilon: FGSM epsilon (default: 2/255)
        target_layers: GradCAM layers (default: ["layer4"])
        device: Computation device
    """

    def __call__(
        self,
        x: Tensor,
        y: Union[int, Tensor],
        class_idx: Optional[int] = None,
    ) -> StabilityScore:
        """Compute Spearman stability score."""
        # Ensure correct format
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        elif y.ndim == 0:
            y = y.unsqueeze(0)

        # Generate explanations
        heatmap_clean = self._generate_explanation(x, class_idx)
        x_adv = self._generate_perturbation(x, y)
        heatmap_adv = self._generate_explanation(x_adv, class_idx)

        # Compute Spearman correlation
        # Flatten heatmaps
        h_clean_flat = heatmap_clean.flatten()
        h_adv_flat = heatmap_adv.flatten()

        # Compute Spearman ρ
        from scipy.stats import spearmanr

        spearman_corr, _ = spearmanr(h_clean_flat, h_adv_flat)

        # Handle NaN (constant heatmaps)
        if np.isnan(spearman_corr):
            spearman_corr = 1.0 if np.allclose(h_clean_flat, h_adv_flat) else 0.0

        # Normalize to [0, 1] (Spearman is in [-1, 1])
        stability = (spearman_corr + 1.0) / 2.0

        # Compute perturbation norm
        pert_norm = self._compute_perturbation_norm(x, x_adv)

        return StabilityScore(
            stability=stability,
            instability=1.0 - stability,
            method=StabilityMethod.SPEARMAN,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=pert_norm,
            epsilon=self.epsilon,
            metadata={"spearman_rho": float(spearman_corr)},
        )


class L2StabilityScorer(BaseStabilityScorer):
    """Stability scorer using normalized L2 distance.

    L2 distance measures Euclidean distance between explanations.
    Lower distance = higher stability.

    Args:
        model: PyTorch model to evaluate
        epsilon: FGSM epsilon (default: 2/255)
        target_layers: GradCAM layers (default: ["layer4"])
        device: Computation device
    """

    def __call__(
        self,
        x: Tensor,
        y: Union[int, Tensor],
        class_idx: Optional[int] = None,
    ) -> StabilityScore:
        """Compute L2 stability score."""
        # Ensure correct format
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        elif y.ndim == 0:
            y = y.unsqueeze(0)

        # Generate explanations
        heatmap_clean = self._generate_explanation(x, class_idx)
        x_adv = self._generate_perturbation(x, y)
        heatmap_adv = self._generate_explanation(x_adv, class_idx)

        # Compute normalized L2 distance
        h_clean_flat = heatmap_clean.flatten()
        h_adv_flat = heatmap_adv.flatten()

        l2_dist = np.linalg.norm(h_clean_flat - h_adv_flat)
        max_dist = np.sqrt(2.0)  # Max distance for [0, 1] normalized heatmaps
        normalized_l2 = l2_dist / max_dist

        # Convert distance to similarity: stability = 1 - normalized_distance
        stability = 1.0 - normalized_l2

        # Compute perturbation norm
        pert_norm = self._compute_perturbation_norm(x, x_adv)

        return StabilityScore(
            stability=stability,
            instability=1.0 - stability,
            method=StabilityMethod.L2,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=pert_norm,
            epsilon=self.epsilon,
            metadata={
                "l2_distance": float(l2_dist),
                "normalized_l2": float(normalized_l2),
            },
        )


class CosineStabilityScorer(BaseStabilityScorer):
    """Stability scorer using cosine similarity.

    Cosine similarity measures angular similarity between explanations,
    which is invariant to scaling.

    Args:
        model: PyTorch model to evaluate
        epsilon: FGSM epsilon (default: 2/255)
        target_layers: GradCAM layers (default: ["layer4"])
        device: Computation device
    """

    def __call__(
        self,
        x: Tensor,
        y: Union[int, Tensor],
        class_idx: Optional[int] = None,
    ) -> StabilityScore:
        """Compute cosine stability score."""
        # Ensure correct format
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if isinstance(y, int):
            y = torch.tensor([y], device=self.device)
        elif y.ndim == 0:
            y = y.unsqueeze(0)

        # Generate explanations
        heatmap_clean = self._generate_explanation(x, class_idx)
        x_adv = self._generate_perturbation(x, y)
        heatmap_adv = self._generate_explanation(x_adv, class_idx)

        # Compute cosine similarity
        h_clean_flat = heatmap_clean.flatten()
        h_adv_flat = heatmap_adv.flatten()

        dot_product = np.dot(h_clean_flat, h_adv_flat)
        norm_clean = np.linalg.norm(h_clean_flat)
        norm_adv = np.linalg.norm(h_adv_flat)

        # Handle zero norms
        if norm_clean < 1e-8 or norm_adv < 1e-8:
            cosine_sim = 1.0 if np.allclose(h_clean_flat, h_adv_flat) else 0.0
        else:
            cosine_sim = dot_product / (norm_clean * norm_adv)

        # Normalize to [0, 1] (cosine is in [-1, 1])
        stability = (cosine_sim + 1.0) / 2.0

        # Compute perturbation norm
        pert_norm = self._compute_perturbation_norm(x, x_adv)

        return StabilityScore(
            stability=stability,
            instability=1.0 - stability,
            method=StabilityMethod.COSINE,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=pert_norm,
            epsilon=self.epsilon,
            metadata={"cosine_similarity": float(cosine_sim)},
        )


# ============================================================================
# Unified Stability Scorer
# ============================================================================


class StabilityScorer:
    """Unified interface for all stability scoring methods.

    This class provides a single API for computing stability scores
    using different metrics. It automatically selects the appropriate
    scorer based on the method parameter.

    Args:
        model: PyTorch model to evaluate
        method: Stability computation method (default: "ssim")
        epsilon: FGSM epsilon (default: 2/255)
        target_layers: GradCAM layers (default: ["layer4"])
        use_ms_ssim: Use multi-scale SSIM (only for SSIM method)
        ssim_window_size: SSIM window size (only for SSIM method)
        device: Computation device

    Example:
        >>> # SSIM (recommended)
        >>> scorer = StabilityScorer(model, method="ssim")
        >>> score = scorer(image, label)
        >>>
        >>> # Multi-scale SSIM
        >>> scorer_ms = StabilityScorer(model, method="ms_ssim", use_ms_ssim=True)
        >>> score_ms = scorer_ms(image, label)
        >>>
        >>> # Alternative metrics
        >>> scorer_spearman = StabilityScorer(model, method="spearman")
        >>> score_sp = scorer_spearman(image, label)
    """

    def __init__(
        self,
        model: nn.Module,
        method: Union[str, StabilityMethod] = "ssim",
        epsilon: float = DEFAULT_EPSILON,
        target_layers: Optional[List[str]] = None,
        use_ms_ssim: bool = False,
        ssim_window_size: int = 11,
        device: Optional[torch.device] = None,
    ):
        """Initialize unified stability scorer."""
        self.model = model
        self.epsilon = epsilon
        self.target_layers = target_layers or ["layer4"]
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Convert method string to enum
        if isinstance(method, str):
            try:
                self.method = StabilityMethod(method.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown stability method: {method}. "
                    f"Choose from: {[m.value for m in StabilityMethod]}"
                )
        else:
            self.method = method

        # Initialize appropriate scorer
        if self.method in (StabilityMethod.SSIM, StabilityMethod.MS_SSIM):
            self.scorer = SSIMStabilityScorer(
                model=model,
                epsilon=epsilon,
                target_layers=self.target_layers,
                use_ms_ssim=(self.method == StabilityMethod.MS_SSIM or use_ms_ssim),
                ssim_window_size=ssim_window_size,
                device=self.device,
            )
        elif self.method == StabilityMethod.SPEARMAN:
            self.scorer = SpearmanStabilityScorer(
                model=model,
                epsilon=epsilon,
                target_layers=self.target_layers,
                device=self.device,
            )
        elif self.method == StabilityMethod.L2:
            self.scorer = L2StabilityScorer(
                model=model,
                epsilon=epsilon,
                target_layers=self.target_layers,
                device=self.device,
            )
        elif self.method == StabilityMethod.COSINE:
            self.scorer = CosineStabilityScorer(
                model=model,
                epsilon=epsilon,
                target_layers=self.target_layers,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported stability method: {self.method}")

        logger.info(
            f"Initialized StabilityScorer with method={self.method.value}, "
            f"epsilon={epsilon:.6f}, device={self.device}"
        )

    def __call__(
        self,
        x: Tensor,
        y: Union[int, Tensor],
        class_idx: Optional[int] = None,
        return_all_metrics: bool = False,
    ) -> StabilityScore:
        """Compute stability score.

        Args:
            x: Input image (C, H, W) or (1, C, H, W)
            y: True label
            class_idx: Target class for explanation
            return_all_metrics: Include all metrics (only for SSIM)

        Returns:
            StabilityScore object
        """
        if hasattr(self.scorer, "__call__"):
            if isinstance(self.scorer, SSIMStabilityScorer):
                return self.scorer(x, y, class_idx, return_all_metrics)
            else:
                return self.scorer(x, y, class_idx)
        else:
            raise RuntimeError(f"Scorer {type(self.scorer)} not callable")

    def batch_score(
        self,
        inputs: Tensor,
        labels: Tensor,
        batch_size: int = 32,
        return_all_metrics: bool = False,
    ) -> List[StabilityScore]:
        """Compute stability scores for batch.

        Args:
            inputs: Input images (N, C, H, W)
            labels: True labels (N,)
            batch_size: Processing batch size
            return_all_metrics: Include all metrics

        Returns:
            List of StabilityScore objects
        """
        if hasattr(self.scorer, "batch_score"):
            return self.scorer.batch_score(
                inputs, labels, batch_size, return_all_metrics
            )
        else:
            # Fallback to sequential processing
            scores = []
            for i in range(inputs.shape[0]):
                score = self(
                    inputs[i], labels[i], return_all_metrics=return_all_metrics
                )
                scores.append(score)
            return scores

    def score_batch(
        self,
        inputs: Tensor,
        labels: Tensor,
        batch_size: int = 32,
        return_all_metrics: bool = False,
    ) -> List[StabilityScore]:
        """Alias for batch_score (more intuitive naming).

        Args:
            inputs: Input images (N, C, H, W)
            labels: True labels (N,)
            batch_size: Processing batch size
            return_all_metrics: Include all metrics

        Returns:
            List of StabilityScore objects
        """
        return self.batch_score(inputs, labels, batch_size, return_all_metrics)


# ============================================================================
# Helper Functions
# ============================================================================


def compute_stability_metrics(
    scores: List[StabilityScore],
) -> Dict[str, float]:
    """Compute aggregate statistics for stability scores.

    Args:
        scores: List of StabilityScore objects

    Returns:
        Dictionary with aggregate metrics:
            - mean_stability: Average stability
            - std_stability: Standard deviation
            - min_stability: Minimum stability
            - max_stability: Maximum stability
            - median_stability: Median stability
            - stable_fraction: Fraction with stability >= 0.75
            - mean_perturbation_norm: Average perturbation L∞ norm
    """
    if not scores:
        return {}

    stabilities = [s.stability for s in scores]
    pert_norms = [s.perturbation_norm for s in scores]

    return {
        "mean_stability": float(np.mean(stabilities)),
        "std_stability": float(np.std(stabilities)),
        "min_stability": float(np.min(stabilities)),
        "max_stability": float(np.max(stabilities)),
        "median_stability": float(np.median(stabilities)),
        "stable_fraction": float(np.mean([s >= 0.75 for s in stabilities])),
        "mean_perturbation_norm": float(np.mean(pert_norms)),
        # Legacy keys for backward compatibility
        "mean": float(np.mean(stabilities)),
        "std": float(np.std(stabilities)),
        "min": float(np.min(stabilities)),
        "max": float(np.max(stabilities)),
        "median": float(np.median(stabilities)),
    }


def create_stability_scorer(
    model: nn.Module,
    method: str = "ssim",
    epsilon: float = DEFAULT_EPSILON,
    **kwargs,
) -> StabilityScorer:
    """Factory function for creating stability scorers.

    Args:
        model: PyTorch model
        method: Stability method ("ssim", "spearman", "l2", "cosine")
        epsilon: FGSM epsilon (default: 2/255)
        **kwargs: Additional arguments for scorer

    Returns:
        StabilityScorer instance

    Example:
        >>> scorer = create_stability_scorer(model, method="ssim", epsilon=2/255)
        >>> score = scorer(image, label)
    """
    return StabilityScorer(model=model, method=method, epsilon=epsilon, **kwargs)


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    "StabilityMethod",
    "StabilityScore",
    "BaseStabilityScorer",
    "SSIMStabilityScorer",
    "SpearmanStabilityScorer",
    "L2StabilityScorer",
    "CosineStabilityScorer",
    "StabilityScorer",
    "compute_stability_metrics",
    "create_stability_scorer",
    "DEFAULT_EPSILON",
]
