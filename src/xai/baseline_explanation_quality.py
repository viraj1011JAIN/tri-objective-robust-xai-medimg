"""
Production-Grade Baseline Explanation Quality Evaluator.

This module provides comprehensive evaluation of baseline (untrained/vanilla)
explanation quality for medical imaging models. It measures both stability
and faithfulness of Grad-CAM explanations under adversarial perturbations,
establishing the baseline for RQ2 and RQ3 hypothesis testing.

Key Functionality
-----------------
1. **Stability Evaluation**: SSIM, Spearman ρ, L2 distance under FGSM
2. **Faithfulness Evaluation**: Deletion/Insertion AUC, Pointing Game
3. **Visualization**: Side-by-side clean vs adversarial heatmaps
4. **Statistical Analysis**: Mean, std, confidence intervals

Research Context (RQ2, RQ3)
---------------------------
This module establishes the baseline against which tri-objective training
improvements are measured:

**Baseline Expectations**:
    - Low stability: SSIM ~0.55-0.60 (below H2 threshold of 0.75)
    - Moderate faithfulness: Deletion AUC ~0.40-0.50
    - Poor pointing game: Accuracy ~30-40%

**Tri-Objective Improvements** (to be demonstrated):
    - H2: SSIM ≥ 0.75 with λ_expl > 0
    - H3: Better deletion/insertion AUC with λ_expl > 0
    - Improved semantic alignment (pointing game)

Integration
-----------
- Uses src.xai.gradcam for explanation generation
- Uses src.xai.stability_metrics for stability evaluation
- Uses src.xai.faithfulness for faithfulness evaluation
- Uses src.attacks.fgsm for adversarial perturbations
- Compatible with src.models architectures
- Logs metrics for MLflow tracking

Typical Usage
-------------
>>> from src.xai.baseline_explanation_quality import (
...     BaselineExplanationQuality,
...     BaselineQualityConfig
... )
>>> config = BaselineQualityConfig(epsilon=2/255, num_samples=100)
>>> evaluator = BaselineExplanationQuality(model, config)
>>> results = evaluator.evaluate_dataset(dataloader)
>>> evaluator.save_visualizations(results, save_dir="results/baseline")

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 6.4 - Baseline Explanation Quality
Date: November 25, 2025
Version: 6.4.0 (Production)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks.fgsm import FGSM, FGSMConfig
from src.xai.faithfulness import FaithfulnessConfig, FaithfulnessMetrics
from src.xai.gradcam import GradCAM, GradCAMConfig
from src.xai.stability_metrics import StabilityMetrics, StabilityMetricsConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BaselineQualityConfig:
    """
    Configuration for baseline explanation quality evaluation.

    Attributes:
        epsilon: FGSM perturbation magnitude (default: 2/255 = 0.00784)
        target_layers: Layers for Grad-CAM (None = auto-detect)
        num_samples: Number of samples to evaluate (None = all)
        batch_size: Batch size for evaluation
        device: Device for computation
        seed: Random seed for reproducibility
        compute_faithfulness: Whether to compute faithfulness metrics
        compute_pointing_game: Whether to compute pointing game (needs masks)
        save_visualizations: Whether to save visualization outputs
        use_ms_ssim: Whether to include MS-SSIM (slower but more robust)
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
    """

    # Attack configuration
    epsilon: float = 2.0 / 255.0  # Standard FGSM perturbation

    # Grad-CAM configuration
    target_layers: Optional[List[str]] = None

    # Evaluation configuration
    num_samples: Optional[int] = None  # None = evaluate all
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Metric flags
    compute_faithfulness: bool = True
    compute_pointing_game: bool = False  # Requires ground-truth masks
    save_visualizations: bool = True
    use_ms_ssim: bool = False  # MS-SSIM is slower

    # Visualization configuration
    num_visualizations: int = 10  # Number of samples to visualize
    figsize: Tuple[int, int] = (15, 5)  # Figure size for side-by-side plots

    # Logging
    verbose: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive or None, got {self.num_samples}"
            )

        if self.num_visualizations < 0:
            raise ValueError(
                f"num_visualizations must be non-negative, "
                f"got {self.num_visualizations}"
            )


# ============================================================================
# Baseline Explanation Quality Evaluator
# ============================================================================


class BaselineExplanationQuality:
    """
    Comprehensive evaluator for baseline explanation quality.

    This class orchestrates the evaluation pipeline:
    1. Generate Grad-CAM heatmaps on clean images
    2. Generate adversarial perturbations (FGSM)
    3. Generate Grad-CAM heatmaps on adversarial images
    4. Compute stability metrics (SSIM, Spearman, L2, Cosine)
    5. Compute faithfulness metrics (Deletion/Insertion AUC)
    6. Visualize results and save reports

    Attributes:
        model: PyTorch model to evaluate
        config: Configuration for evaluation
        gradcam: Grad-CAM instance
        fgsm: FGSM attack instance
        stability_metrics: Stability metrics instance
        faithfulness_metrics: Faithfulness metrics instance (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[BaselineQualityConfig] = None,
    ):
        """
        Initialize baseline explanation quality evaluator.

        Args:
            model: PyTorch model to evaluate
            config: Configuration (uses defaults if None)
        """
        self.model = model.eval()
        self.config = config or BaselineQualityConfig()

        # Set device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Initialize Grad-CAM
        # Auto-detect target layers if not specified
        target_layers = self.config.target_layers
        if target_layers is None:
            # Try to find last convolutional layer
            target_layers = self._auto_detect_target_layers()

        gradcam_config = GradCAMConfig(
            target_layers=target_layers,
            use_cuda=(self.device.type == "cuda"),
        )
        self.gradcam = GradCAM(self.model, gradcam_config)

        # Initialize FGSM attack
        fgsm_config = FGSMConfig(
            epsilon=self.config.epsilon,
            clip_min=0.0,
            clip_max=1.0,
        )
        self.fgsm = FGSM(fgsm_config)

        # Initialize stability metrics
        stability_config = StabilityMetricsConfig(
            normalize_heatmaps=True,
            use_cuda=(self.device.type == "cuda"),
        )
        self.stability_metrics = StabilityMetrics(stability_config)

        # Initialize faithfulness metrics (optional)
        self.faithfulness_metrics = None
        if self.config.compute_faithfulness:
            faithfulness_config = FaithfulnessConfig(
                num_steps=50,
                verbose=(self.config.verbose >= 2),
                device=str(self.device),
            )
            self.faithfulness_metrics = FaithfulnessMetrics(
                self.model, faithfulness_config
            )

        logger.info(
            f"Initialized BaselineExplanationQuality with ε={self.config.epsilon:.5f}"
        )

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"BaselineExplanationQuality(\n"
            f"  epsilon={self.config.epsilon:.5f},\n"
            f"  target_layers={self.config.target_layers},\n"
            f"  compute_faithfulness={self.config.compute_faithfulness},\n"
            f"  device={self.device}\n"
            f")"
        )

    def _auto_detect_target_layers(self) -> List[str]:
        """
        Auto-detect target layers for Grad-CAM.

        Returns:
            List of layer names (prioritizes last conv layer)
        """
        # Try common layer names
        common_names = ["layer4", "conv2", "features"]

        for name in common_names:
            if hasattr(self.model, name):
                return [name]

        # Fallback: find last Conv2d module
        last_conv_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name

        if last_conv_name:
            return [last_conv_name]

        # Ultimate fallback
        logger.warning("Could not auto-detect target layers. Using empty list.")
        return ["conv2"]  # Default fallback

    def evaluate_batch(
        self,
        images: Tensor,
        labels: Tensor,
        masks: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate explanation quality for a single batch.

        Args:
            images: Clean images [B, C, H, W]
            labels: Ground-truth labels [B]
            masks: Ground-truth masks [B, H, W] (optional, for pointing game)

        Returns:
            Dictionary with evaluation metrics:
                - stability: {ssim, ms_ssim, spearman, l2_distance, cosine}
                - faithfulness_clean: {deletion_auc, insertion_auc, ...}
                - faithfulness_adv: {deletion_auc, insertion_auc, ...}
                - heatmaps_clean: Clean heatmaps [B, 1, H', W']
                - heatmaps_adv: Adversarial heatmaps [B, 1, H', W']
                - images_adv: Adversarial images [B, C, H, W]
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        results = {}

        # Step 1: Generate clean heatmaps (sample by sample)
        heatmaps_clean_list = []
        for i in range(images.shape[0]):
            heatmap = self.gradcam.generate_heatmap(
                images[i : i + 1], class_idx=labels[i].item()
            )
            # Convert numpy to tensor
            if isinstance(heatmap, np.ndarray):
                heatmap = torch.from_numpy(heatmap).to(self.device)
            # Ensure 4D shape [1, 1, H, W]
            if heatmap.dim() == 2:
                heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            elif heatmap.dim() == 3:
                heatmap = heatmap.unsqueeze(0)
            heatmaps_clean_list.append(heatmap)
        heatmaps_clean = torch.cat(heatmaps_clean_list, dim=0)
        results["heatmaps_clean"] = heatmaps_clean

        # Step 2: Generate adversarial images
        # Note: Generate adversarials one by one to avoid batch backward issues
        images_adv_list = []
        for i in range(images.shape[0]):
            img_single = images[i : i + 1]
            label_single = labels[i : i + 1]
            img_adv = self.fgsm.generate(self.model, img_single, label_single)
            images_adv_list.append(img_adv)
        images_adv = torch.cat(images_adv_list, dim=0)
        results["images_adv"] = images_adv

        # Step 3: Generate adversarial heatmaps (sample by sample)
        heatmaps_adv_list = []
        for i in range(images_adv.shape[0]):
            heatmap = self.gradcam.generate_heatmap(
                images_adv[i : i + 1], class_idx=labels[i].item()
            )
            # Convert numpy to tensor
            if isinstance(heatmap, np.ndarray):
                heatmap = torch.from_numpy(heatmap).to(self.device)
            # Ensure 4D shape [1, 1, H, W]
            if heatmap.dim() == 2:
                heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            elif heatmap.dim() == 3:
                heatmap = heatmap.unsqueeze(0)
            heatmaps_adv_list.append(heatmap)
        heatmaps_adv = torch.cat(heatmaps_adv_list, dim=0)
        results["heatmaps_adv"] = heatmaps_adv

        # Step 4: Compute stability metrics
        stability = self.stability_metrics.compute_all(
            heatmaps_clean,
            heatmaps_adv,
            include_ms_ssim=self.config.use_ms_ssim,
        )
        results["stability"] = stability

        # Step 5: Compute faithfulness metrics (optional)
        if self.config.compute_faithfulness and self.faithfulness_metrics is not None:
            # Faithfulness on clean images
            faithfulness_clean = self.faithfulness_metrics.compute_all(
                images,
                heatmaps_clean,
                labels.tolist(),
                masks=masks if self.config.compute_pointing_game else None,
            )
            results["faithfulness_clean"] = faithfulness_clean

            # Faithfulness on adversarial images
            faithfulness_adv = self.faithfulness_metrics.compute_all(
                images_adv,
                heatmaps_adv,
                labels.tolist(),
                masks=masks if self.config.compute_pointing_game else None,
            )
            results["faithfulness_adv"] = faithfulness_adv

        return results

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate explanation quality on entire dataset.

        Args:
            dataloader: DataLoader for evaluation
            save_dir: Directory to save visualizations (if enabled)

        Returns:
            Aggregated results dictionary:
                - stability: {metric: [mean, std, ci_low, ci_high]}
                - faithfulness_clean: {metric: [mean, std, ci_low, ci_high]}
                - faithfulness_adv: {metric: [mean, std, ci_low, ci_high]}
                - num_samples: Total samples evaluated
                - visualization_samples: List of sample results for plotting
        """
        self.model.eval()

        # Storage for metrics
        stability_metrics = {
            "ssim": [],
            "spearman": [],
            "l2_distance": [],
            "cosine_similarity": [],
        }
        if self.config.use_ms_ssim:
            stability_metrics["ms_ssim"] = []

        faithfulness_clean_metrics = {
            "deletion_auc": [],
            "deletion_ad": [],
            "insertion_auc": [],
            "insertion_ai": [],
        }
        faithfulness_adv_metrics = {
            "deletion_auc": [],
            "deletion_ad": [],
            "insertion_auc": [],
            "insertion_ai": [],
        }

        if self.config.compute_pointing_game:
            faithfulness_clean_metrics["pointing_acc"] = []
            faithfulness_adv_metrics["pointing_acc"] = []

        # Storage for visualizations
        visualization_samples = []

        # Evaluation loop
        total_samples = 0
        max_samples = self.config.num_samples or float("inf")

        pbar = tqdm(
            dataloader,
            desc="Evaluating baseline quality",
            disable=(self.config.verbose == 0),
        )

        for batch in pbar:
            if total_samples >= max_samples:
                break

            # Extract batch data
            if len(batch) == 2:
                images, labels = batch
                masks = None
            elif len(batch) == 3:
                images, labels, masks = batch
            else:
                raise ValueError(
                    f"Expected batch with 2 or 3 elements, got {len(batch)}"
                )

            # Evaluate batch
            batch_results = self.evaluate_batch(images, labels, masks)

            # Aggregate stability metrics
            for key, value in batch_results["stability"].items():
                if key in stability_metrics:
                    stability_metrics[key].append(value)

            # Aggregate faithfulness metrics
            if "faithfulness_clean" in batch_results:
                for key, value in batch_results["faithfulness_clean"].items():
                    if key in faithfulness_clean_metrics:
                        faithfulness_clean_metrics[key].append(value)

            if "faithfulness_adv" in batch_results:
                for key, value in batch_results["faithfulness_adv"].items():
                    if key in faithfulness_adv_metrics:
                        faithfulness_adv_metrics[key].append(value)

            # Store samples for visualization
            if len(visualization_samples) < self.config.num_visualizations:
                num_to_save = min(
                    self.config.num_visualizations - len(visualization_samples),
                    images.shape[0],
                )
                for i in range(num_to_save):
                    visualization_samples.append(
                        {
                            "image_clean": images[i].cpu(),
                            "image_adv": batch_results["images_adv"][i].cpu(),
                            "heatmap_clean": batch_results["heatmaps_clean"][i].cpu(),
                            "heatmap_adv": batch_results["heatmaps_adv"][i].cpu(),
                            "label": labels[i].item(),
                            "ssim": batch_results["stability"]["ssim"],
                        }
                    )

            total_samples += images.shape[0]

            # Update progress bar
            if self.config.verbose >= 1:
                pbar.set_postfix(
                    {
                        "SSIM": f"{np.mean(stability_metrics['ssim']):.3f}",
                        "Samples": total_samples,
                    }
                )

        # Compute statistics (mean, std, 95% CI)
        def compute_stats(values: List[float]) -> Dict[str, float]:
            """Compute mean, std, and 95% confidence interval."""
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr)
            ci = 1.96 * std / np.sqrt(len(arr))  # 95% CI
            return {
                "mean": float(mean),
                "std": float(std),
                "ci_low": float(mean - ci),
                "ci_high": float(mean + ci),
                "n": len(arr),
            }

        # Aggregate results
        aggregated_results = {
            "stability": {
                key: compute_stats(values)
                for key, values in stability_metrics.items()
                if values
            },
            "num_samples": total_samples,
            "visualization_samples": visualization_samples,
        }

        if self.config.compute_faithfulness:
            aggregated_results["faithfulness_clean"] = {
                key: compute_stats(values)
                for key, values in faithfulness_clean_metrics.items()
                if values
            }
            aggregated_results["faithfulness_adv"] = {
                key: compute_stats(values)
                for key, values in faithfulness_adv_metrics.items()
                if values
            }

        # Save visualizations
        if self.config.save_visualizations and save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self.save_visualizations(aggregated_results, save_path)

        # Log summary
        if self.config.verbose >= 1:
            self._log_summary(aggregated_results)

        return aggregated_results

    def save_visualizations(
        self,
        results: Dict[str, Any],
        save_dir: Union[str, Path],
    ) -> None:
        """
        Save side-by-side visualizations of clean vs adversarial explanations.

        Args:
            results: Aggregated results from evaluate_dataset()
            save_dir: Directory to save visualizations
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        visualization_samples = results["visualization_samples"]

        for idx, sample in enumerate(visualization_samples):
            fig, axes = plt.subplots(1, 4, figsize=self.config.figsize)

            # Clean image
            img_clean = self._denormalize_image(sample["image_clean"])
            axes[0].imshow(img_clean)
            axes[0].set_title("Clean Image")
            axes[0].axis("off")

            # Clean heatmap overlay
            heatmap_clean = sample["heatmap_clean"].squeeze().numpy()
            overlay_clean = self._overlay_heatmap(img_clean, heatmap_clean)
            axes[1].imshow(overlay_clean)
            axes[1].set_title("Clean Grad-CAM")
            axes[1].axis("off")

            # Adversarial image
            img_adv = self._denormalize_image(sample["image_adv"])
            axes[2].imshow(img_adv)
            axes[2].set_title(f"Adversarial (ε={self.config.epsilon:.4f})")
            axes[2].axis("off")

            # Adversarial heatmap overlay
            heatmap_adv = sample["heatmap_adv"].squeeze().numpy()
            overlay_adv = self._overlay_heatmap(img_adv, heatmap_adv)
            axes[3].imshow(overlay_adv)
            axes[3].set_title(f"Adv Grad-CAM (SSIM={sample['ssim']:.3f})")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(
                save_path / f"baseline_quality_sample_{idx:03d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        logger.info(f"Saved {len(visualization_samples)} visualizations to {save_path}")

    def _denormalize_image(self, image: Tensor) -> np.ndarray:
        """
        Denormalize image tensor for visualization.

        Args:
            image: Image tensor [C, H, W]

        Returns:
            RGB numpy array [H, W, 3] in [0, 255]
        """
        # Move to CPU if needed
        image = image.cpu()

        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Denormalize
        img = image * std + mean
        img = torch.clamp(img, 0, 1)

        # Convert to numpy
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        return img

    def _overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay heatmap on image.

        Args:
            image: RGB image [H, W, 3]
            heatmap: Heatmap [H', W']
            alpha: Overlay transparency

        Returns:
            Overlaid image [H, W, 3]
        """
        import matplotlib.cm as cm
        from PIL import Image as PILImage

        # Resize heatmap to match image
        heatmap_resized = PILImage.fromarray(heatmap).resize(
            (image.shape[1], image.shape[0]), PILImage.BILINEAR
        )
        heatmap_resized = np.array(heatmap_resized)

        # Normalize heatmap
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (
            heatmap_resized.max() - heatmap_resized.min() + 1e-8
        )

        # Apply colormap (jet)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # RGB only
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

        return overlay

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log evaluation summary."""
        logger.info("=" * 70)
        logger.info("BASELINE EXPLANATION QUALITY SUMMARY")
        logger.info("=" * 70)

        # Stability metrics
        logger.info("\nStability Metrics (Clean vs Adversarial):")
        logger.info("-" * 70)
        for metric, stats in results["stability"].items():
            logger.info(
                f"  {metric:20s}: {stats['mean']:.4f} ± "
                f"{stats['std']:.4f} [{stats['ci_low']:.4f}, "
                f"{stats['ci_high']:.4f}]"
            )

        # Faithfulness metrics
        if "faithfulness_clean" in results:
            logger.info("\nFaithfulness Metrics (Clean Images):")
            logger.info("-" * 70)
            for metric, stats in results["faithfulness_clean"].items():
                logger.info(
                    f"  {metric:20s}: {stats['mean']:.4f} ± "
                    f"{stats['std']:.4f} [{stats['ci_low']:.4f}, "
                    f"{stats['ci_high']:.4f}]"
                )

            logger.info("\nFaithfulness Metrics (Adversarial Images):")
            logger.info("-" * 70)
            for metric, stats in results["faithfulness_adv"].items():
                logger.info(
                    f"  {metric:20s}: {stats['mean']:.4f} ± "
                    f"{stats['std']:.4f} [{stats['ci_low']:.4f}, "
                    f"{stats['ci_high']:.4f}]"
                )

        logger.info("\n" + "=" * 70)
        logger.info(f"Total samples evaluated: {results['num_samples']}")
        logger.info("=" * 70)


# ============================================================================
# Factory Function
# ============================================================================


def create_baseline_quality_evaluator(
    model: nn.Module,
    config: Optional[BaselineQualityConfig] = None,
) -> BaselineExplanationQuality:
    """
    Factory function to create baseline quality evaluator.

    Args:
        model: PyTorch model to evaluate
        config: Configuration (uses defaults if None)

    Returns:
        BaselineExplanationQuality instance
    """
    return BaselineExplanationQuality(model, config)
