"""
Production-Grade Faithfulness Metrics for XAI Evaluation.

This module implements quantitative metrics for measuring how faithful visual
explanations are to the model's actual decision-making process. Faithfulness
measures whether high-attribution regions truly drive predictions (RQ3).

Mathematical Foundation
-----------------------
Faithfulness evaluates explanation quality through perturbation analysis:
1. **Deletion**: Remove high-attribution pixels → score should drop
2. **Insertion**: Add high-attribution pixels → score should rise
3. **Pointing Game**: Max attribution should align with ground truth

Key Metrics:
    - Deletion AUC: Area under deletion curve (lower = more faithful)
    - Insertion AUC: Area under insertion curve (higher = more faithful)
    - Pointing Game Accuracy: Hit rate for ground-truth mask alignment
    - Average Drop (AD): Prediction drop after deletion
    - Average Increase (AI): Prediction rise after insertion

Research Hypothesis (H3)
------------------------
Tri-objective training produces explanations with:
    - Higher Insertion AUC (explanations identify discriminative regions)
    - Lower Deletion AUC (explanations are localized)
    - Better Pointing Game accuracy (explanations align with semantics)

Target: Demonstrate faithfulness improves with λ_expl > 0

Integration:
    - Compatible with src.xai.gradcam heatmaps
    - Works with src.models.model_registry architectures
    - Supports batch processing for efficiency
    - Enables RQ3 hypothesis testing

References
----------
.. [1] Petsiuk et al. "RISE: Randomized Input Sampling for Explanation of
       Black-box Models." BMVC 2018.
.. [2] Samek et al. "Evaluating the Visualization of What a Deep Neural
       Network Has Learned." IEEE TNNLS 2017.
.. [3] Zhang et al. "Top-down Neural Attention by Excitation Backprop."
       ECCV 2016.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 6.3 - Faithfulness Metrics
Date: November 25, 2025
Version: 6.3.0 (Production)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FaithfulnessConfig:
    """
    Configuration for faithfulness metrics computation.

    Attributes
    ----------
    num_steps : int
        Number of perturbation steps for deletion/insertion curves.
        Trade-off: More steps = smoother curves but slower computation.
        Recommended: 50-100 for production, 20-30 for development.

    batch_size : int
        Batch size for model forward passes during perturbation.
        Higher = faster but more memory.

    baseline_mode : str
        How to fill deleted/uninserted regions:
        - 'mean': Dataset mean (standard, model-aware)
        - 'blur': Gaussian blur (preserves structure)
        - 'noise': Random noise (max perturbation)
        - 'zero': Zero pixels (simple baseline)

    interpolation_mode : str
        PyTorch interpolation for resizing heatmaps.
        Options: 'bilinear', 'nearest', 'bicubic'

    normalize_curves : bool
        Whether to normalize curves to [0, 1] range.
        Recommended: True for cross-model comparison.

    use_softmax : bool
        Apply softmax to logits before extracting scores.
        Set False if model already outputs probabilities.

    pointing_game_threshold : float
        Tolerance for pointing game (fraction of max attribution).
        Standard: 0.15 (within 15% of max)

    device : str
        Computation device ('cuda' or 'cpu').

    verbose : bool
        Show progress bars during computation.

    Example
    -------
    >>> config = FaithfulnessConfig(
    ...     num_steps=50,
    ...     baseline_mode='mean',
    ...     batch_size=32
    ... )
    """

    num_steps: int = 50
    batch_size: int = 16
    baseline_mode: str = "mean"
    interpolation_mode: str = "bilinear"
    normalize_curves: bool = True
    use_softmax: bool = True
    pointing_game_threshold: float = 0.15
    device: str = "cuda"
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_steps < 5:
            raise ValueError(
                f"num_steps must be ≥ 5 for meaningful curves, got {self.num_steps}"
            )

        valid_baselines = ["mean", "blur", "noise", "zero"]
        if self.baseline_mode not in valid_baselines:
            raise ValueError(
                f"baseline_mode must be in {valid_baselines}, got {self.baseline_mode}"
            )

        valid_interp = ["bilinear", "nearest", "bicubic"]
        if self.interpolation_mode not in valid_interp:
            raise ValueError(
                f"interpolation_mode must be in {valid_interp}, "
                f"got {self.interpolation_mode}"
            )

        if not 0.0 <= self.pointing_game_threshold <= 1.0:
            raise ValueError(
                f"pointing_game_threshold must be in [0, 1], "
                f"got {self.pointing_game_threshold}"
            )


# ============================================================================
# Deletion Curve
# ============================================================================


class DeletionMetric:
    """
    Deletion curve: Remove high-attribution pixels and measure score drop.

    A faithful explanation should cause a steep score drop when its
    high-attribution regions are removed (deleted pixels were important).

    Algorithm:
        1. Rank pixels by attribution (descending)
        2. For k = 0, Δ, 2Δ, ..., 100%:
            a. Delete top k% pixels (replace with baseline)
            b. Forward pass → prediction score
            c. Record (k, score)
        3. Compute AUC (lower = more faithful)
        4. Compute Average Drop: score₀ - score_final

    Interpretation:
        - AUC close to 0: Explanation highlights critical regions
        - AUC close to 1: Explanation is random/unfaithful
        - Steeper drop = better localization

    Mathematical Formula:
        AUC_del = (1/n) Σᵢ₌₀ⁿ⁻¹ score(x_deleted[i])
        AD = score(x) - score(x_fully_deleted)

    Example
    -------
    >>> deletion = DeletionMetric(model, config)
    >>> curve, auc, ad = deletion.compute(image, heatmap, target_class=1)
    >>> print(f"Deletion AUC: {auc:.3f}, Average Drop: {ad:.3f}")
    """

    def __init__(self, model: nn.Module, config: FaithfulnessConfig):
        """
        Initialize deletion metric.

        Parameters
        ----------
        model : nn.Module
            Trained model to evaluate.
        config : FaithfulnessConfig
            Configuration for computation.
        """
        self.model = model
        self.config = config
        self.model.eval()

    def compute(
        self,
        image: Tensor,
        heatmap: Tensor,
        target_class: int,
        baseline_value: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute deletion curve, AUC, and average drop.

        Parameters
        ----------
        image : Tensor
            Input image, shape (C, H, W) or (1, C, H, W)
        heatmap : Tensor
            Attribution heatmap, shape (H, W) or (1, H, W) or (1, 1, H, W)
        target_class : int
            Class index to track during deletion
        baseline_value : Tensor, optional
            Pre-computed baseline (e.g., dataset mean).
            If None, uses config.baseline_mode.

        Returns
        -------
        curve : np.ndarray
            Deletion curve, shape (num_steps,)
            curve[i] = prediction score after deleting i% of pixels
        auc : float
            Area under deletion curve (normalized to [0, 1])
        average_drop : float
            Score drop: score(original) - score(fully_deleted)

        Examples
        --------
        >>> curve, auc, ad = deletion.compute(img, hmap, target_class=0)
        >>> # Lower AUC and higher AD = more faithful explanation
        """
        # Preprocess inputs
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        if heatmap.dim() == 2:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)  # (B, 1, H, W)

        image = image.to(self.config.device)
        heatmap = heatmap.to(self.config.device)

        # Resize heatmap to match image spatial dimensions
        if heatmap.shape[-2:] != image.shape[-2:]:
            heatmap = F.interpolate(
                heatmap,
                size=image.shape[-2:],
                mode=self.config.interpolation_mode,
                align_corners=(
                    False if self.config.interpolation_mode != "nearest" else None
                ),
            )

        # Get baseline value
        if baseline_value is None:
            baseline_value = self._get_baseline(image)

        # Flatten and rank pixels by attribution
        batch_size, channels, height, width = image.shape
        heatmap_flat = heatmap.view(batch_size, -1)  # (B, H*W)
        sorted_indices = torch.argsort(
            heatmap_flat, dim=1, descending=True
        )  # High → Low

        # Compute deletion curve
        num_pixels = height * width
        step_size = max(1, num_pixels // self.config.num_steps)
        curve = []

        # Initial score (no deletion)
        with torch.no_grad():
            logits = self.model(image)
            if self.config.use_softmax:
                scores = F.softmax(logits, dim=1)
            else:
                scores = logits
            initial_score = scores[0, target_class].item()
            curve.append(initial_score)

        # Progressive deletion
        deleted_image = image.clone()
        iterator = range(step_size, num_pixels + 1, step_size)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Deletion", leave=False)

        for num_deleted in iterator:
            # Get indices to delete (top-k attributions)
            indices_to_delete = sorted_indices[0, :num_deleted]

            # Create mask for deletion
            mask = torch.ones(num_pixels, device=image.device)
            mask[indices_to_delete] = 0
            mask = mask.view(1, 1, height, width).expand_as(image)

            # Apply deletion (replace with baseline)
            deleted_image = image * mask + baseline_value * (1 - mask)

            # Forward pass
            with torch.no_grad():
                logits = self.model(deleted_image)
                if self.config.use_softmax:
                    scores = F.softmax(logits, dim=1)
                else:
                    scores = logits
                score = scores[0, target_class].item()
                curve.append(score)

        curve = np.array(curve)

        # Compute AUC (normalized to [0, 1])
        if self.config.normalize_curves:
            auc = np.trapz(curve, dx=1.0 / len(curve))
        else:
            auc = np.trapz(curve) / len(curve)

        # Average drop
        average_drop = initial_score - curve[-1]

        return curve, auc, average_drop

    def _get_baseline(self, image: Tensor) -> Tensor:
        """
        Get baseline value for deleted pixels.

        Parameters
        ----------
        image : Tensor
            Input image, shape (B, C, H, W)

        Returns
        -------
        baseline : Tensor
            Baseline value, same shape as image
        """
        if self.config.baseline_mode == "mean":
            # Channel-wise mean
            mean = image.mean(dim=(2, 3), keepdim=True)
            return mean.expand_as(image)

        elif self.config.baseline_mode == "blur":
            # Gaussian blur
            kernel_size = min(11, image.shape[-1] // 4 * 2 + 1)  # Adaptive kernel
            return F.avg_pool2d(
                image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )

        elif self.config.baseline_mode == "noise":
            # Random noise (same distribution as image)
            noise = torch.randn_like(image)
            noise = noise * image.std() + image.mean()
            return noise

        elif self.config.baseline_mode == "zero":
            # Zero baseline
            return torch.zeros_like(image)

        else:  # pragma: no cover (validated in FaithfulnessConfig.__post_init__)
            raise ValueError(f"Unknown baseline_mode: {self.config.baseline_mode}")


# ============================================================================
# Insertion Curve
# ============================================================================


class InsertionMetric:
    """
    Insertion curve: Add high-attribution pixels and measure score rise.

    A faithful explanation should cause a steep score rise when its
    high-attribution regions are added to a blank canvas.

    Algorithm:
        1. Start with baseline image (blank canvas)
        2. Rank pixels by attribution (descending)
        3. For k = 0, Δ, 2Δ, ..., 100%:
            a. Insert top k% pixels from original
            b. Forward pass → prediction score
            c. Record (k, score)
        4. Compute AUC (higher = more faithful)
        5. Compute Average Increase: score_final - score₀

    Interpretation:
        - AUC close to 1: Explanation captures discriminative features
        - AUC close to 0: Explanation misses important regions
        - Steeper rise = better feature identification

    Mathematical Formula:
        AUC_ins = (1/n) Σᵢ₌₀ⁿ⁻¹ score(x_inserted[i])
        AI = score(x_fully_inserted) - score(x_baseline)

    Example
    -------
    >>> insertion = InsertionMetric(model, config)
    >>> curve, auc, ai = insertion.compute(image, heatmap, target_class=1)
    >>> print(f"Insertion AUC: {auc:.3f}, Average Increase: {ai:.3f}")
    """

    def __init__(self, model: nn.Module, config: FaithfulnessConfig):
        """
        Initialize insertion metric.

        Parameters
        ----------
        model : nn.Module
            Trained model to evaluate.
        config : FaithfulnessConfig
            Configuration for computation.
        """
        self.model = model
        self.config = config
        self.model.eval()

    def compute(
        self,
        image: Tensor,
        heatmap: Tensor,
        target_class: int,
        baseline_value: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute insertion curve, AUC, and average increase.

        Parameters
        ----------
        image : Tensor
            Input image, shape (C, H, W) or (1, C, H, W)
        heatmap : Tensor
            Attribution heatmap, shape (H, W) or (1, H, W) or (1, 1, H, W)
        target_class : int
            Class index to track during insertion
        baseline_value : Tensor, optional
            Pre-computed baseline (starting canvas).
            If None, uses config.baseline_mode.

        Returns
        -------
        curve : np.ndarray
            Insertion curve, shape (num_steps,)
            curve[i] = prediction score after inserting i% of pixels
        auc : float
            Area under insertion curve (normalized to [0, 1])
        average_increase : float
            Score increase: score(fully_inserted) - score(baseline)

        Examples
        --------
        >>> curve, auc, ai = insertion.compute(img, hmap, target_class=0)
        >>> # Higher AUC and AI = more faithful explanation
        """
        # Preprocess inputs
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        if heatmap.dim() == 2:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)  # (B, 1, H, W)

        image = image.to(self.config.device)
        heatmap = heatmap.to(self.config.device)

        # Resize heatmap to match image spatial dimensions
        if heatmap.shape[-2:] != image.shape[-2:]:
            heatmap = F.interpolate(
                heatmap,
                size=image.shape[-2:],
                mode=self.config.interpolation_mode,
                align_corners=(
                    False if self.config.interpolation_mode != "nearest" else None
                ),
            )

        # Get baseline value
        if baseline_value is None:
            baseline_value = self._get_baseline(image)

        # Flatten and rank pixels by attribution
        batch_size, channels, height, width = image.shape
        heatmap_flat = heatmap.view(batch_size, -1)  # (B, H*W)
        sorted_indices = torch.argsort(
            heatmap_flat, dim=1, descending=True
        )  # High → Low

        # Compute insertion curve
        num_pixels = height * width
        step_size = max(1, num_pixels // self.config.num_steps)
        curve = []

        # Initial score (baseline only, no insertion)
        with torch.no_grad():
            logits = self.model(baseline_value)
            if self.config.use_softmax:
                scores = F.softmax(logits, dim=1)
            else:
                scores = logits
            initial_score = scores[0, target_class].item()
            curve.append(initial_score)

        # Progressive insertion
        inserted_image = baseline_value.clone()
        iterator = range(step_size, num_pixels + 1, step_size)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Insertion", leave=False)

        for num_inserted in iterator:
            # Get indices to insert (top-k attributions)
            indices_to_insert = sorted_indices[0, :num_inserted]

            # Create mask for insertion
            mask = torch.zeros(num_pixels, device=image.device)
            mask[indices_to_insert] = 1
            mask = mask.view(1, 1, height, width).expand_as(image)

            # Apply insertion (copy from original)
            inserted_image = baseline_value * (1 - mask) + image * mask

            # Forward pass
            with torch.no_grad():
                logits = self.model(inserted_image)
                if self.config.use_softmax:
                    scores = F.softmax(logits, dim=1)
                else:
                    scores = logits
                score = scores[0, target_class].item()
                curve.append(score)

        curve = np.array(curve)

        # Compute AUC (normalized to [0, 1])
        if self.config.normalize_curves:
            auc = np.trapz(curve, dx=1.0 / len(curve))
        else:
            auc = np.trapz(curve) / len(curve)

        # Average increase
        average_increase = curve[-1] - initial_score

        return curve, auc, average_increase

    def _get_baseline(self, image: Tensor) -> Tensor:
        """Get baseline (same as DeletionMetric)."""
        if self.config.baseline_mode == "mean":
            mean = image.mean(dim=(2, 3), keepdim=True)
            return mean.expand_as(image)
        elif self.config.baseline_mode == "blur":
            kernel_size = min(11, image.shape[-1] // 4 * 2 + 1)
            return F.avg_pool2d(
                image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )
        elif self.config.baseline_mode == "noise":
            noise = torch.randn_like(image)
            noise = noise * image.std() + image.mean()
            return noise
        elif self.config.baseline_mode == "zero":
            return torch.zeros_like(image)
        else:  # pragma: no cover (validated in FaithfulnessConfig.__post_init__)
            raise ValueError(f"Unknown baseline_mode: {self.config.baseline_mode}")


# ============================================================================
# Pointing Game
# ============================================================================


class PointingGame:
    """
    Pointing game: Check if max attribution aligns with ground-truth mask.

    Evaluates whether the explanation's peak (max attribution) falls within
    the semantically relevant region defined by a ground-truth segmentation.

    Algorithm:
        1. Find location (i, j) of max attribution in heatmap
        2. Check if mask[i, j] = 1 (inside ground-truth region)
        3. Tolerance: Allow nearby pixels within threshold
        4. Hit rate = (# hits) / (# samples)

    Interpretation:
        - High hit rate: Explanation aligns with semantic regions
        - Low hit rate: Explanation is unfocused or incorrect
        - Requires ground-truth masks (not always available)

    Use Cases:
        - Medical imaging with lesion segmentations
        - Object detection datasets with bounding boxes
        - Semantic segmentation benchmarks

    Mathematical Formula:
        Hit_i = 1 if mask[argmax(heatmap)] = 1, else 0
        Accuracy = (1/N) Σᵢ₌₀ᴺ⁻¹ Hit_i

    Example
    -------
    >>> pointing_game = PointingGame(config)
    >>> hits, acc = pointing_game.compute_batch(heatmaps, masks)
    >>> print(f"Pointing Game Accuracy: {acc:.2%}")
    """

    def __init__(self, config: FaithfulnessConfig):
        """
        Initialize pointing game metric.

        Parameters
        ----------
        config : FaithfulnessConfig
            Configuration (mainly for threshold).
        """
        self.config = config

    def compute(
        self,
        heatmap: Tensor,
        mask: Tensor,
        tolerance: Optional[int] = None,
    ) -> Tuple[bool, Tuple[int, int]]:
        """
        Compute pointing game for single sample.

        Parameters
        ----------
        heatmap : Tensor
            Attribution heatmap, shape (H, W) or (1, H, W)
        mask : Tensor
            Ground-truth binary mask, shape (H, W) or (1, H, W)
            1 = relevant region, 0 = background
        tolerance : int, optional
            Pixel tolerance radius. If None, uses strict matching.

        Returns
        -------
        hit : bool
            True if max attribution inside mask (with tolerance)
        max_location : Tuple[int, int]
            (row, col) location of max attribution

        Examples
        --------
        >>> hit, (i, j) = pointing_game.compute(heatmap, mask, tolerance=5)
        >>> if hit:
        ...     print(f"Hit! Max at ({i}, {j})")
        """
        # Preprocess
        if heatmap.dim() == 3:
            heatmap = heatmap.squeeze(0)  # (H, W)
        if mask.dim() == 3:
            mask = mask.squeeze(0)  # (H, W)

        heatmap = heatmap.cpu()
        mask = mask.cpu()

        # Resize if needed
        if heatmap.shape != mask.shape:
            heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=mask.shape,
                mode=self.config.interpolation_mode,
                align_corners=(
                    False if self.config.interpolation_mode != "nearest" else None
                ),
            ).squeeze()

        # Find max location
        max_idx = torch.argmax(heatmap.flatten())
        max_row = max_idx // heatmap.shape[1]
        max_col = max_idx % heatmap.shape[1]
        max_location = (int(max_row), int(max_col))

        # Check if inside mask (with tolerance)
        if tolerance is None or tolerance == 0:
            # Strict: exact pixel must be inside mask
            hit = bool(mask[max_row, max_col] > 0.5)
        else:
            # Tolerant: check neighborhood
            h, w = mask.shape
            row_start = max(0, max_row - tolerance)
            row_end = min(h, max_row + tolerance + 1)
            col_start = max(0, max_col - tolerance)
            col_end = min(w, max_col + tolerance + 1)

            neighborhood = mask[row_start:row_end, col_start:col_end]
            hit = bool(neighborhood.max() > 0.5)

        return hit, max_location

    def compute_batch(
        self,
        heatmaps: Tensor,
        masks: Tensor,
        tolerance: Optional[int] = None,
    ) -> Tuple[List[bool], float]:
        """
        Compute pointing game for batch of samples.

        Parameters
        ----------
        heatmaps : Tensor
            Attribution heatmaps, shape (B, H, W) or (B, 1, H, W)
        masks : Tensor
            Ground-truth masks, shape (B, H, W) or (B, 1, H, W)
        tolerance : int, optional
            Pixel tolerance radius

        Returns
        -------
        hits : List[bool]
            Per-sample hit indicators, length B
        accuracy : float
            Overall hit rate (accuracy)

        Examples
        --------
        >>> hits, acc = pointing_game.compute_batch(heatmaps, masks, tolerance=3)
        >>> print(f"Hits: {sum(hits)}/{len(hits)}, Accuracy: {acc:.2%}")
        """
        batch_size = heatmaps.shape[0]
        hits = []

        for i in range(batch_size):
            hit, _ = self.compute(
                heatmaps[i],
                masks[i],
                tolerance=tolerance,
            )
            hits.append(hit)

        accuracy = sum(hits) / len(hits) if hits else 0.0

        return hits, accuracy


# ============================================================================
# Unified Faithfulness Metrics
# ============================================================================


class FaithfulnessMetrics:
    """
    Unified interface for all faithfulness metrics.

    Computes deletion curves, insertion curves, and pointing game in a
    single cohesive API. Handles batch processing and result aggregation.

    Attributes:
        config: Configuration for metrics
        deletion: Deletion metric computer
        insertion: Insertion metric computer
        pointing_game: Pointing game metric

    Example Usage:
        >>> from src.xai.faithfulness import FaithfulnessMetrics
        >>> metrics = FaithfulnessMetrics(model, config)
        >>> results = metrics.compute_all(
        ...     images, heatmaps, target_classes, masks=masks
        ... )
        >>> print(f"Deletion AUC: {results['deletion_auc']:.3f}")
        >>> print(f"Insertion AUC: {results['insertion_auc']:.3f}")
        >>> print(f"Pointing Game: {results['pointing_acc']:.2%}")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FaithfulnessConfig] = None,
    ):
        """
        Initialize faithfulness metrics computer.

        Parameters
        ----------
        model : nn.Module
            Trained model to evaluate
        config : FaithfulnessConfig, optional
            Configuration. Uses defaults if None.
        """
        self.config = config or FaithfulnessConfig()
        self.model = model
        self.model.eval()

        # Initialize metric computers
        self.deletion = DeletionMetric(model, self.config)
        self.insertion = InsertionMetric(model, self.config)
        self.pointing_game = PointingGame(self.config)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"FaithfulnessMetrics("
            f"num_steps={self.config.num_steps}, "
            f"baseline={self.config.baseline_mode}, "
            f"device={self.config.device})"
        )

    def compute_all(
        self,
        images: Tensor,
        heatmaps: Tensor,
        target_classes: List[int],
        masks: Optional[Tensor] = None,
        baseline_value: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute all faithfulness metrics for a batch.

        Parameters
        ----------
        images : Tensor
            Input images, shape (B, C, H, W)
        heatmaps : Tensor
            Attribution heatmaps, shape (B, H, W) or (B, 1, H, W)
        target_classes : List[int]
            Target class for each sample, length B
        masks : Tensor, optional
            Ground-truth masks for pointing game, shape (B, H, W)
        baseline_value : Tensor, optional
            Pre-computed baseline for efficiency

        Returns
        -------
        results : Dict[str, float]
            Aggregated metrics:
            - 'deletion_auc': Mean deletion AUC
            - 'deletion_ad': Mean average drop
            - 'insertion_auc': Mean insertion AUC
            - 'insertion_ai': Mean average increase
            - 'pointing_acc': Pointing game accuracy (if masks provided)

        Examples
        --------
        >>> results = metrics.compute_all(imgs, hmaps, classes, masks=masks)
        >>> print(f"Faithfulness Score: {results['insertion_auc']:.3f}")
        """
        batch_size = images.shape[0]
        if len(target_classes) != batch_size:
            raise ValueError(
                f"Number of target_classes ({len(target_classes)}) "
                f"must match batch size ({batch_size})"
            )
        if baseline_value is not None and baseline_value.shape[0] != batch_size:
            raise ValueError(
                f"baseline_value batch size ({baseline_value.shape[0]}) "
                f"doesn't match images batch size ({batch_size})"
            )

        deletion_aucs = []
        deletion_ads = []
        insertion_aucs = []
        insertion_ais = []

        # Compute deletion and insertion for each sample
        iterator = range(batch_size)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Computing Faithfulness")

        for i in iterator:
            # Deletion
            _, del_auc, del_ad = self.deletion.compute(
                images[i],
                heatmaps[i],
                target_classes[i],
                baseline_value=(
                    baseline_value[i : i + 1] if baseline_value is not None else None
                ),
            )
            deletion_aucs.append(del_auc)
            deletion_ads.append(del_ad)

            # Insertion
            _, ins_auc, ins_ai = self.insertion.compute(
                images[i],
                heatmaps[i],
                target_classes[i],
                baseline_value=(
                    baseline_value[i : i + 1] if baseline_value is not None else None
                ),
            )
            insertion_aucs.append(ins_auc)
            insertion_ais.append(ins_ai)

        results = {
            "deletion_auc": float(np.mean(deletion_aucs)),
            "deletion_ad": float(np.mean(deletion_ads)),
            "insertion_auc": float(np.mean(insertion_aucs)),
            "insertion_ai": float(np.mean(insertion_ais)),
        }

        # Pointing game (if masks available)
        if masks is not None:
            _, pointing_acc = self.pointing_game.compute_batch(
                heatmaps,
                masks,
                tolerance=int(
                    self.config.pointing_game_threshold * min(masks.shape[-2:])
                ),
            )
            results["pointing_acc"] = pointing_acc

        return results


# ============================================================================
# Factory Function
# ============================================================================


def create_faithfulness_metrics(
    model: nn.Module,
    config: Optional[FaithfulnessConfig] = None,
) -> FaithfulnessMetrics:
    """
    Factory function for creating faithfulness metrics computer.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate
    config : FaithfulnessConfig, optional
        Configuration. Uses defaults if None.

    Returns
    -------
    metrics : FaithfulnessMetrics
        Configured faithfulness metrics computer

    Examples
    --------
    >>> from src.xai.faithfulness import create_faithfulness_metrics
    >>> metrics = create_faithfulness_metrics(model)
    >>> results = metrics.compute_all(images, heatmaps, classes)
    """
    return FaithfulnessMetrics(model, config)


# ============================================================================
# Utility Functions
# ============================================================================


def plot_curves(
    deletion_curve: np.ndarray,
    insertion_curve: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot deletion and insertion curves for visualization.

    Parameters
    ----------
    deletion_curve : np.ndarray
        Deletion curve values
    insertion_curve : np.ndarray
        Insertion curve values
    save_path : str, optional
        Path to save figure. If None, displays interactively.

    Examples
    --------
    >>> from src.xai.faithfulness import plot_curves
    >>> plot_curves(del_curve, ins_curve, "results/faithfulness.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, cannot plot curves")
        return

    plt.figure(figsize=(10, 4))

    # Deletion curve
    plt.subplot(1, 2, 1)
    x = np.linspace(0, 100, len(deletion_curve))
    plt.plot(x, deletion_curve, "r-", linewidth=2, label="Deletion")
    plt.fill_between(x, deletion_curve, alpha=0.3, color="red")
    plt.xlabel("% Pixels Deleted")
    plt.ylabel("Prediction Score")
    plt.title("Deletion Curve (Lower AUC = Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Insertion curve
    plt.subplot(1, 2, 2)
    x = np.linspace(0, 100, len(insertion_curve))
    plt.plot(x, insertion_curve, "g-", linewidth=2, label="Insertion")
    plt.fill_between(x, insertion_curve, alpha=0.3, color="green")
    plt.xlabel("% Pixels Inserted")
    plt.ylabel("Prediction Score")
    plt.title("Insertion Curve (Higher AUC = Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved faithfulness curves to {save_path}")
    else:
        plt.show()

    plt.close()
