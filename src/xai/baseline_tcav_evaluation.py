"""Baseline TCAV Evaluation for Medical Imaging Models.

This module implements comprehensive evaluation of baseline model reliance on
medical vs artifact concepts using TCAV (Testing with Concept Activation Vectors).
It provides quantitative analysis of concept importance across network layers,
enabling assessment of spurious correlation problems.

Key Components:
    - BaselineTCAVConfig: Configuration for baseline evaluation
    - ConceptCategory: Enum for concept types (medical, artifact)
    - BaselineTCAVEvaluator: Main evaluation class
    - Visualization utilities for TCAV scores

Research Context:
    Baseline TCAV evaluation establishes the problem: Do models rely on spurious
    artifacts (rulers, hair, ink marks) rather than medical features? This
    motivates concept regularization (RQ2) and validates the need for robust
    training approaches.

    Expected Baseline Results:
    - Artifact TCAV scores: 0.40-0.50 (HIGH - indicates problem)
    - Medical TCAV scores: 0.55-0.65 (MODERATE - should be higher)
    - Multi-layer analysis reveals concept emergence patterns

Reference:
    Phase 6.7 of dissertation focusing on baseline concept reliance analysis
    for dermoscopy and chest X-ray classification tasks.

Example:
    >>> from src.xai.baseline_tcav_evaluation import create_baseline_tcav_evaluator
    >>> import torch
    >>>
    >>> # Create evaluator
    >>> evaluator = create_baseline_tcav_evaluator(
    ...     model=my_resnet50,
    ...     target_layers=["layer2", "layer3", "layer4"],
    ...     concept_data_dir="data/concepts/derm7pt",
    ...     medical_concepts=["pigment_network", "atypical_network"],
    ...     artifact_concepts=["ruler", "hair", "ink_mark"]
    ... )
    >>>
    >>> # Evaluate baseline
    >>> results = evaluator.evaluate_baseline(
    ...     images=test_images,
    ...     target_class=1,
    ...     num_random_concepts=10
    ... )
    >>>
    >>> # Analyze results
    >>> print(f"Medical TCAV: {results['medical_mean']:.3f} ± {results['medical_std']:.3f}")
    >>> print(f"Artifact TCAV: {results['artifact_mean']:.3f} ± {results['artifact_std']:.3f}")
    >>>
    >>> # Generate visualization
    >>> evaluator.visualize_concept_scores(results, save_path="baseline_tcav.png")
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats

from src.xai.tcav import create_tcav

logger = logging.getLogger(__name__)


class ConceptCategory(Enum):
    """Concept category enumeration."""

    MEDICAL = "medical"
    ARTIFACT = "artifact"
    RANDOM = "random"


@dataclass
class BaselineTCAVConfig:
    """Configuration for baseline TCAV evaluation.

    Attributes:
        model: PyTorch model to evaluate
        target_layers: List of layer names to analyze (e.g., ["layer2", "layer3", "layer4"])
        concept_data_dir: Directory containing concept datasets
        medical_concepts: List of medical concept names
        artifact_concepts: List of artifact concept names
        cav_dir: Directory to save/load CAVs
        batch_size: Batch size for data loading
        num_random_concepts: Number of random concepts for statistical testing
        min_cav_accuracy: Minimum CAV accuracy threshold
        device: Device to run computations on
        seed: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=info, 2=debug)
    """

    model: nn.Module
    target_layers: List[str]
    concept_data_dir: Union[str, Path]
    medical_concepts: List[str]
    artifact_concepts: List[str]
    cav_dir: Union[str, Path] = Path("checkpoints/baseline_cavs")
    batch_size: int = 32
    num_random_concepts: int = 10
    min_cav_accuracy: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    verbose: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if not self.target_layers:
            raise ValueError("target_layers cannot be empty")

        if not self.medical_concepts:
            raise ValueError("medical_concepts cannot be empty")

        if not self.artifact_concepts:
            raise ValueError("artifact_concepts cannot be empty")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_random_concepts <= 0:
            raise ValueError("num_random_concepts must be positive")

        if not 0 < self.min_cav_accuracy <= 1:
            raise ValueError("min_cav_accuracy must be in (0, 1]")

        # Convert paths
        self.concept_data_dir = Path(self.concept_data_dir)
        self.cav_dir = Path(self.cav_dir)

        # Create CAV directory
        self.cav_dir.mkdir(parents=True, exist_ok=True)


class BaselineTCAVEvaluator:
    """Baseline TCAV evaluation for medical imaging models.

    This class provides comprehensive evaluation of model reliance on medical
    vs artifact concepts, including multi-layer analysis, statistical testing,
    and visualization capabilities.
    """

    def __init__(self, config: BaselineTCAVConfig):
        """Initialize baseline TCAV evaluator.

        Args:
            config: Configuration for baseline evaluation
        """
        self.config = config
        self.model = config.model
        self.model.to(config.device)
        self.model.eval()

        # Create TCAV instance
        self.tcav = create_tcav(
            model=config.model,
            target_layers=config.target_layers,
            concept_data_dir=config.concept_data_dir,
            cav_dir=config.cav_dir,
            batch_size=config.batch_size,
            num_random_concepts=config.num_random_concepts,
            min_cav_accuracy=config.min_cav_accuracy,
            device=config.device,
            seed=config.seed,
            verbose=config.verbose,
        )

        # Storage for results
        self.results: Dict = {}

        logger.info(
            f"Initialized BaselineTCAVEvaluator with "
            f"{len(config.medical_concepts)} medical and "
            f"{len(config.artifact_concepts)} artifact concepts"
        )

    def precompute_cavs(self):
        """Precompute CAVs for all concepts and layers.

        This trains CAVs for all medical and artifact concepts across all
        target layers with multiple random concepts for statistical testing.
        """
        all_concepts = self.config.medical_concepts + self.config.artifact_concepts

        logger.info(f"Precomputing CAVs for {len(all_concepts)} concepts")
        self.tcav.precompute_all_cavs(all_concepts)
        logger.info("CAV precomputation complete")

    def evaluate_baseline(
        self,
        images: torch.Tensor,
        target_class: int,
        precompute: bool = True,
    ) -> Dict[str, Union[float, Dict, np.ndarray]]:
        """Evaluate baseline model reliance on concepts.

        Args:
            images: Input images (N, C, H, W)
            target_class: Target class index for TCAV computation
            precompute: Whether to precompute CAVs first

        Returns:
            Dictionary containing:
                - medical_scores: Dict[str, Dict[str, float]] - per concept, per layer
                - artifact_scores: Dict[str, Dict[str, float]] - per concept, per layer
                - medical_mean: float - mean medical TCAV score
                - medical_std: float - std medical TCAV score
                - artifact_mean: float - mean artifact TCAV score
                - artifact_std: float - std artifact TCAV score
                - medical_layer_means: Dict[str, float] - per layer medical means
                - artifact_layer_means: Dict[str, float] - per layer artifact means
                - statistical_comparison: Dict - t-test and effect size
        """
        if precompute:
            self.precompute_cavs()

        logger.info(f"Evaluating baseline on {len(images)} images")

        # Compute TCAV scores for all concepts
        medical_scores = self._compute_concept_scores(
            images, target_class, self.config.medical_concepts
        )
        artifact_scores = self._compute_concept_scores(
            images, target_class, self.config.artifact_concepts
        )

        # Aggregate statistics
        medical_all_scores = self._flatten_scores(medical_scores)
        artifact_all_scores = self._flatten_scores(artifact_scores)

        # Per-layer aggregation
        medical_layer_means = self._compute_layer_means(medical_scores)
        artifact_layer_means = self._compute_layer_means(artifact_scores)

        # Statistical comparison
        stat_comparison = self._statistical_comparison(
            medical_all_scores, artifact_all_scores
        )

        results = {
            "medical_scores": medical_scores,
            "artifact_scores": artifact_scores,
            "medical_mean": float(np.mean(medical_all_scores)),
            "medical_std": float(np.std(medical_all_scores)),
            "artifact_mean": float(np.mean(artifact_all_scores)),
            "artifact_std": float(np.std(artifact_all_scores)),
            "medical_layer_means": medical_layer_means,
            "artifact_layer_means": artifact_layer_means,
            "statistical_comparison": stat_comparison,
            "num_images": len(images),
            "target_class": target_class,
        }

        self.results = results

        logger.info(
            f"Baseline evaluation complete: "
            f"Medical={results['medical_mean']:.3f}±{results['medical_std']:.3f}, "
            f"Artifact={results['artifact_mean']:.3f}±{results['artifact_std']:.3f}"
        )

        return results

    def _compute_concept_scores(
        self, images: torch.Tensor, target_class: int, concepts: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute TCAV scores for concepts across all layers.

        Args:
            images: Input images
            target_class: Target class
            concepts: List of concept names

        Returns:
            Nested dict: {concept: {layer: score}}
        """
        scores = {}

        for concept in concepts:
            scores[concept] = {}

            # Compute multi-layer TCAV
            try:
                layer_scores = self.tcav.compute_multilayer_tcav(
                    inputs=images, target_class=target_class, concept=concept
                )
                scores[concept] = layer_scores

                if self.config.verbose > 0:
                    mean_score = np.mean(list(layer_scores.values()))
                    logger.info(f"Concept '{concept}': {mean_score:.3f} (avg)")

            except Exception as e:
                logger.warning(f"Failed to compute TCAV for concept '{concept}': {e}")
                # Fill with NaN for missing scores
                for layer in self.config.target_layers:
                    scores[concept][layer] = float("nan")

        return scores

    def _flatten_scores(self, scores: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Flatten nested scores dict to 1D array.

        Args:
            scores: Nested scores dict

        Returns:
            1D array of all scores
        """
        all_scores = []
        for concept_scores in scores.values():
            for score in concept_scores.values():
                if not np.isnan(score):
                    all_scores.append(score / 100.0)  # Convert to [0, 1]

        return np.array(all_scores)

    def _compute_layer_means(
        self, scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute mean scores per layer.

        Args:
            scores: Nested scores dict

        Returns:
            Dict mapping layer to mean score
        """
        layer_means = {}

        for layer in self.config.target_layers:
            layer_scores = []
            for concept_scores in scores.values():
                if layer in concept_scores and not np.isnan(concept_scores[layer]):
                    layer_scores.append(concept_scores[layer] / 100.0)

            if layer_scores:
                layer_means[layer] = float(np.mean(layer_scores))
            else:
                layer_means[layer] = float("nan")

        return layer_means

    def _statistical_comparison(
        self, medical_scores: np.ndarray, artifact_scores: np.ndarray
    ) -> Dict[str, float]:
        """Perform statistical comparison between medical and artifact scores.

        Args:
            medical_scores: Medical TCAV scores
            artifact_scores: Artifact TCAV scores

        Returns:
            Dict with t-statistic, p-value, and Cohen's d
        """
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(medical_scores, artifact_scores)

        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.var(medical_scores) + np.var(artifact_scores)) / 2)
        cohens_d = (np.mean(medical_scores) - np.mean(artifact_scores)) / pooled_std

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
        }

    def analyze_multilayer_activation(
        self, results: Optional[Dict] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze concept activation across network depth.

        Args:
            results: Evaluation results (uses self.results if None)

        Returns:
            Dict with per-layer analysis for medical and artifact concepts
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run evaluate_baseline first.")
            results = self.results

        analysis = {
            "medical": results["medical_layer_means"],
            "artifact": results["artifact_layer_means"],
            "layer_differences": {},
        }

        # Compute difference per layer
        for layer in self.config.target_layers:
            med_score = results["medical_layer_means"].get(layer, float("nan"))
            art_score = results["artifact_layer_means"].get(layer, float("nan"))

            if not np.isnan(med_score) and not np.isnan(art_score):
                analysis["layer_differences"][layer] = med_score - art_score

        logger.info("Multi-layer activation analysis complete")
        return analysis

    def visualize_concept_scores(
        self,
        results: Optional[Dict] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """Generate comprehensive visualization of TCAV scores.

        Creates a multi-panel figure with:
        1. Bar chart: Medical vs Artifact mean scores
        2. Per-concept scores across layers
        3. Multi-layer activation trends
        4. Statistical comparison

        Args:
            results: Evaluation results (uses self.results if None)
            save_path: Path to save figure
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run evaluate_baseline first.")
            results = self.results

        # Set style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Medical vs Artifact comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_category_comparison(ax1, results)

        # 2. Per-concept scores
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_per_concept_scores(ax2, results)

        # 3. Multi-layer activation
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_multilayer_activation(ax3, results)

        # 4. Statistical summary
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_statistical_summary(ax4, results)

        plt.suptitle(
            "Baseline TCAV Evaluation: Medical vs Artifact Concepts",
            fontsize=16,
            fontweight="bold",
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {save_path}")

        return fig

    def _plot_category_comparison(self, ax: plt.Axes, results: Dict):
        """Plot medical vs artifact category comparison."""
        categories = ["Medical", "Artifact"]
        means = [results["medical_mean"], results["artifact_mean"]]
        stds = [results["medical_std"], results["artifact_std"]]

        colors = ["#2ecc71", "#e74c3c"]  # Green for medical, red for artifact

        bars = ax.bar(categories, means, yerr=stds, capsize=10, color=colors, alpha=0.7)

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("TCAV Score", fontweight="bold")
        ax.set_title("Mean TCAV Scores by Category", fontweight="bold")
        ax.set_ylim(0, max(means) + max(stds) + 0.1)

        # Add expected ranges
        ax.axhspan(0.40, 0.50, alpha=0.1, color="red", label="Expected Artifact Range")
        ax.axhspan(0.55, 0.65, alpha=0.1, color="green", label="Expected Medical Range")
        ax.legend(loc="upper right", fontsize=8)

    def _plot_per_concept_scores(self, ax: plt.Axes, results: Dict):
        """Plot per-concept TCAV scores."""
        # Collect all concepts and their mean scores
        medical_concepts = []
        medical_means = []

        for concept, layer_scores in results["medical_scores"].items():
            valid_scores = [s / 100.0 for s in layer_scores.values() if not np.isnan(s)]
            if valid_scores:
                medical_concepts.append(concept)
                medical_means.append(np.mean(valid_scores))

        artifact_concepts = []
        artifact_means = []

        for concept, layer_scores in results["artifact_scores"].items():
            valid_scores = [s / 100.0 for s in layer_scores.values() if not np.isnan(s)]
            if valid_scores:
                artifact_concepts.append(concept)
                artifact_means.append(np.mean(valid_scores))

        # Combine and sort
        all_concepts = medical_concepts + artifact_concepts
        all_means = medical_means + artifact_means
        colors_list = ["#2ecc71"] * len(medical_concepts) + ["#e74c3c"] * len(
            artifact_concepts
        )

        # Sort by score
        sorted_idx = np.argsort(all_means)[::-1]
        all_concepts = [all_concepts[i] for i in sorted_idx]
        all_means = [all_means[i] for i in sorted_idx]
        colors_list = [colors_list[i] for i in sorted_idx]

        # Plot horizontal bar chart
        y_pos = np.arange(len(all_concepts))
        ax.barh(y_pos, all_means, color=colors_list, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_concepts, fontsize=8)
        ax.set_xlabel("Mean TCAV Score", fontweight="bold")
        ax.set_title("Per-Concept TCAV Scores", fontweight="bold")
        ax.invert_yaxis()

    def _plot_multilayer_activation(self, ax: plt.Axes, results: Dict):
        """Plot multi-layer activation trends."""
        layers = self.config.target_layers
        medical_means = [
            results["medical_layer_means"].get(layer, 0) for layer in layers
        ]
        artifact_means = [
            results["artifact_layer_means"].get(layer, 0) for layer in layers
        ]

        x = np.arange(len(layers))
        width = 0.35

        ax.bar(
            x - width / 2,
            medical_means,
            width,
            label="Medical",
            color="#2ecc71",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            artifact_means,
            width,
            label="Artifact",
            color="#e74c3c",
            alpha=0.7,
        )

        ax.set_xlabel("Network Layer", fontweight="bold")
        ax.set_ylabel("Mean TCAV Score", fontweight="bold")
        ax.set_title("Multi-Layer Concept Activation", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    def _plot_statistical_summary(self, ax: plt.Axes, results: Dict):
        """Plot statistical summary text."""
        ax.axis("off")

        stat = results["statistical_comparison"]

        summary_text = f"""
Statistical Comparison
{'=' * 40}

Medical Concepts:
  Mean: {results['medical_mean']:.4f}
  Std:  {results['medical_std']:.4f}

Artifact Concepts:
  Mean: {results['artifact_mean']:.4f}
  Std:  {results['artifact_std']:.4f}

Statistical Test:
  t-statistic: {stat['t_statistic']:.4f}
  p-value:     {stat['p_value']:.4e}
  Cohen's d:   {stat['cohens_d']:.4f}
  Significant: {'Yes' if stat['significant'] else 'No'} (α=0.05)

Interpretation:
"""

        # Add interpretation
        if results["artifact_mean"] >= 0.40:
            summary_text += "\n⚠️  HIGH artifact reliance detected!"
        else:
            summary_text += "\n✓  Low artifact reliance"

        if results["medical_mean"] < 0.60:
            summary_text += "\n⚠️  LOW medical concept usage!"
        else:
            summary_text += "\n✓  Good medical concept usage"

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=9,
            verticalalignment="top",
        )

    def save_results(self, path: Union[str, Path]):
        """Save evaluation results to file.

        Args:
            path: Path to save results (will save as .npz)
        """
        if not self.results:
            raise ValueError("No results to save. Run evaluate_baseline first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert nested dicts to flat structure for npz
        save_dict = {
            "medical_mean": self.results["medical_mean"],
            "medical_std": self.results["medical_std"],
            "artifact_mean": self.results["artifact_mean"],
            "artifact_std": self.results["artifact_std"],
            "num_images": self.results["num_images"],
            "target_class": self.results["target_class"],
        }

        # Add statistical comparison
        for key, val in self.results["statistical_comparison"].items():
            save_dict[f"stat_{key}"] = val

        np.savez(path, **save_dict)
        logger.info(f"Saved results to {path}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BaselineTCAVEvaluator(\n"
            f"  medical_concepts={len(self.config.medical_concepts)},\n"
            f"  artifact_concepts={len(self.config.artifact_concepts)},\n"
            f"  target_layers={self.config.target_layers}\n"
            f")"
        )


def create_baseline_tcav_evaluator(
    model: nn.Module,
    target_layers: List[str],
    concept_data_dir: Union[str, Path],
    medical_concepts: List[str],
    artifact_concepts: List[str],
    **kwargs,
) -> BaselineTCAVEvaluator:
    """Factory function to create baseline TCAV evaluator.

    Args:
        model: PyTorch model to evaluate
        target_layers: List of layer names
        concept_data_dir: Path to concept datasets
        medical_concepts: List of medical concept names
        artifact_concepts: List of artifact concept names
        **kwargs: Additional config parameters

    Returns:
        BaselineTCAVEvaluator instance
    """
    config = BaselineTCAVConfig(
        model=model,
        target_layers=target_layers,
        concept_data_dir=concept_data_dir,
        medical_concepts=medical_concepts,
        artifact_concepts=artifact_concepts,
        **kwargs,
    )

    return BaselineTCAVEvaluator(config)
