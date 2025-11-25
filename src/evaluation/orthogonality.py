"""
Orthogonality Analysis for RQ1: Adversarial Training vs. Generalization.

This module provides production-ready tools to analyze the orthogonality
between robustness and cross-site generalization objectives, confirming that
adversarial training improves robustness but NOT generalization.

Classes:
    OrthogonalityConfig: Configuration for orthogonality analysis
    OrthogonalityResults: Container for analysis results
    OrthogonalityAnalyzer: Main analysis engine

Typical Usage:
    >>> config = OrthogonalityConfig(
    ...     results_dir=Path("results/phase_5_baselines/isic2018"),
    ...     output_dir=Path("results/phase_5_5_analysis"),
    ...     seeds=[42, 123, 456],
    ...     dataset="isic2018"
    ... )
    >>> analyzer = OrthogonalityAnalyzer(config)
    >>> results = analyzer.run_analysis()
    >>> print(f"Orthogonality confirmed: {results.is_orthogonal}")

Integration:
    - Compatible with adversarial_trainer.py metrics format
    - Uses project evaluation utilities (comparison.py, metrics.py)
    - Follows BaseTrainer checkpoint conventions
    - MLflow logging ready
    - Type-safe with comprehensive validation

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.evaluation.comparison import save_comparison_results

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalityConfig:
    """Configuration for orthogonality analysis.

    Attributes:
        results_dir: Directory containing model results (baseline, pgd_at,
            trades)
        output_dir: Directory to save analysis outputs
        seeds: Random seeds used for training (default: [42, 123, 456])
        dataset: Dataset name (e.g., 'isic2018', 'derm7pt')
        models: Models to compare (default: baseline, pgd_at, trades)
        significance_level: Statistical significance threshold (default: 0.05)
        metrics: Metrics to analyze (clean_acc, robust_acc, cross_site_auroc)
        generate_latex: Whether to generate LaTeX tables (default: True)
        save_figures: Whether to save figure files (default: True)
        figure_format: Format for saved figures (default: 'pdf')
        figure_dpi: DPI for saved figures (default: 300)
    """

    results_dir: Path
    output_dir: Path
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    dataset: str = "isic2018"
    models: List[str] = field(default_factory=lambda: ["baseline", "pgd_at", "trades"])
    significance_level: float = 0.05
    metrics: List[str] = field(
        default_factory=lambda: [
            "clean_accuracy",
            "robust_accuracy",
            "cross_site_auroc",
        ]
    )
    generate_latex: bool = True
    save_figures: bool = True
    figure_format: str = "pdf"
    figure_dpi: int = 300

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.results_dir = Path(self.results_dir)
        self.output_dir = Path(self.output_dir)

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        if self.significance_level <= 0 or self.significance_level >= 1:
            raise ValueError(
                f"significance_level must be in (0, 1), "
                f"got {self.significance_level}"
            )

        if len(self.seeds) < 2:
            raise ValueError(
                f"At least 2 seeds required for statistics, got {len(self.seeds)}"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelResults:
    """Container for single model results across seeds.

    Attributes:
        model_name: Name of the model (e.g., 'baseline', 'pgd_at')
        clean_accuracy: List of clean accuracy values (one per seed)
        robust_accuracy: List of robust accuracy values (one per seed)
        cross_site_auroc: List of cross-site AUROC values (one per seed)
        seeds: Random seeds used
    """

    model_name: str
    clean_accuracy: List[float]
    robust_accuracy: List[float]
    cross_site_auroc: List[float]
    seeds: List[int]

    def __post_init__(self) -> None:
        """Validate that all metric lists have same length as seeds."""
        n_seeds = len(self.seeds)
        if len(self.clean_accuracy) != n_seeds:
            raise ValueError(
                f"clean_accuracy length ({len(self.clean_accuracy)}) "
                f"!= seeds length ({n_seeds})"
            )
        if len(self.robust_accuracy) != n_seeds:
            raise ValueError(
                f"robust_accuracy length ({len(self.robust_accuracy)}) "
                f"!= seeds length ({n_seeds})"
            )
        if len(self.cross_site_auroc) != n_seeds:
            raise ValueError(
                f"cross_site_auroc length ({len(self.cross_site_auroc)}) "
                f"!= seeds length ({n_seeds})"
            )

    def get_mean(self, metric: str) -> float:
        """Get mean value for a metric."""
        values = getattr(self, metric)
        return float(np.mean(values))

    def get_std(self, metric: str) -> float:
        """Get standard deviation for a metric."""
        values = getattr(self, metric)
        return float(np.std(values, ddof=1))


@dataclass
class StatisticalTest:
    """Results of statistical significance test.

    Attributes:
        test_name: Name of the test (e.g., 'paired_t_test')
        metric: Metric being tested (e.g., 'robust_accuracy')
        model_a: First model name
        model_b: Second model name
        statistic: Test statistic value
        p_value: P-value
        is_significant: Whether result is statistically significant
        effect_size: Cohen's d effect size
        interpretation: Human-readable interpretation
    """

    test_name: str
    metric: str
    model_a: str
    model_b: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    interpretation: str


@dataclass
class OrthogonalityResults:
    """Complete orthogonality analysis results.

    Attributes:
        config: Configuration used for analysis
        model_results: Dictionary mapping model names to ModelResults
        statistical_tests: List of statistical test results
        is_orthogonal: Whether orthogonality is confirmed
        summary: Human-readable summary
        comparison_table: DataFrame with comparison results
    """

    config: OrthogonalityConfig
    model_results: Dict[str, ModelResults]
    statistical_tests: List[StatisticalTest]
    is_orthogonal: bool
    summary: str
    comparison_table: pd.DataFrame

    def save(self, output_dir: Optional[Path] = None) -> None:
        """Save all results to disk.

        Args:
            output_dir: Output directory (uses config.output_dir if None)
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        summary_data = {
            "dataset": self.config.dataset,
            "is_orthogonal": self.is_orthogonal,
            "summary": self.summary,
            "model_results": {
                name: {
                    "clean_accuracy": {
                        "mean": results.get_mean("clean_accuracy"),
                        "std": results.get_std("clean_accuracy"),
                    },
                    "robust_accuracy": {
                        "mean": results.get_mean("robust_accuracy"),
                        "std": results.get_std("robust_accuracy"),
                    },
                    "cross_site_auroc": {
                        "mean": results.get_mean("cross_site_auroc"),
                        "std": results.get_std("cross_site_auroc"),
                    },
                }
                for name, results in self.model_results.items()
            },
            "statistical_tests": [asdict(test) for test in self.statistical_tests],
        }

        with open(output_dir / "orthogonality_results.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        # Save comparison table
        self.comparison_table.to_csv(output_dir / "comparison_table.csv", index=False)

        if self.config.generate_latex:
            latex_str = self.comparison_table.to_latex(index=False, float_format="%.4f")
            with open(output_dir / "comparison_table.tex", "w") as f:
                f.write(latex_str)

        logger.info(f"Saved orthogonality results to {output_dir}")


class OrthogonalityAnalyzer:
    """Main engine for orthogonality analysis.

    This analyzer loads trained model results, computes statistical comparisons,
    and generates comprehensive reports confirming or rejecting the orthogonality
    hypothesis (adversarial training improves robustness but NOT generalization).

    Example:
        >>> config = OrthogonalityConfig(
        ...     results_dir=Path("results/phase_5_baselines/isic2018"),
        ...     output_dir=Path("results/phase_5_5_analysis")
        ... )
        >>> analyzer = OrthogonalityAnalyzer(config)
        >>> results = analyzer.run_analysis()
    """

    def __init__(self, config: OrthogonalityConfig):
        """Initialize analyzer with configuration.

        Args:
            config: OrthogonalityConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model_results(self, model_name: str) -> ModelResults:
        """Load results for a single model across all seeds.

        Expected structure:
        results_dir/{model_name}_seed{seed}/metrics.json

        Metrics JSON format (from adversarial_trainer.py):
        {
            "clean_accuracy": 0.8523,
            "robust_accuracy": 0.1234,
            "cross_site_auroc": 0.782,
            "epoch": 100
        }

        Args:
            model_name: Name of the model (e.g., 'baseline', 'pgd_at')

        Returns:
            ModelResults containing metrics across seeds

        Raises:
            FileNotFoundError: If any result file is missing
            KeyError: If required metrics are missing from results
        """
        clean_acc = []
        robust_acc = []
        cross_site = []

        for seed in self.config.seeds:
            result_file = (
                self.config.results_dir / f"{model_name}_seed{seed}" / "metrics.json"
            )

            if not result_file.exists():
                raise FileNotFoundError(
                    f"Result file not found: {result_file}\n"
                    f"Expected structure: results_dir/{model_name}_seed{seed}/"
                    f"metrics.json"
                )

            with open(result_file) as f:
                metrics = json.load(f)

            # Extract required metrics with validation
            try:
                clean_acc.append(float(metrics["clean_accuracy"]))
                robust_acc.append(float(metrics["robust_accuracy"]))
                cross_site.append(float(metrics["cross_site_auroc"]))
            except KeyError as e:
                raise KeyError(
                    f"Missing metric {e} in {result_file}\n"
                    f"Required: clean_accuracy, robust_accuracy, cross_site_auroc"
                )

        self.logger.info(
            f"Loaded {model_name}: "
            f"clean_acc={np.mean(clean_acc):.4f}±{np.std(clean_acc):.4f}, "
            f"robust_acc={np.mean(robust_acc):.4f}±{np.std(robust_acc):.4f}, "
            f"cross_site={np.mean(cross_site):.4f}±{np.std(cross_site):.4f}"
        )

        return ModelResults(
            model_name=model_name,
            clean_accuracy=clean_acc,
            robust_accuracy=robust_acc,
            cross_site_auroc=cross_site,
            seeds=self.config.seeds,
        )

    def compute_statistical_test(
        self,
        metric: str,
        model_a: ModelResults,
        model_b: ModelResults,
    ) -> StatisticalTest:
        """Perform paired t-test and compute effect size.

        Args:
            metric: Metric name (e.g., 'robust_accuracy')
            model_a: First model results
            model_b: Second model results

        Returns:
            StatisticalTest with test results and interpretation
        """
        values_a = np.array(getattr(model_a, metric))
        values_b = np.array(getattr(model_b, metric))

        # Paired t-test
        statistic, p_value = stats.ttest_rel(values_a, values_b)

        # Cohen's d effect size
        diff = values_a - values_b
        effect_size = float(np.mean(diff) / np.std(diff, ddof=1))

        is_significant = p_value < self.config.significance_level

        # Interpretation
        if is_significant:
            direction = "higher" if np.mean(values_a) > np.mean(values_b) else "lower"
            interpretation = (
                f"{model_a.model_name} has significantly {direction} "
                f"{metric} than {model_b.model_name} "
                f"(p={p_value:.4f}, d={effect_size:.2f})"
            )
        else:
            interpretation = (
                f"No significant difference in {metric} between "
                f"{model_a.model_name} and {model_b.model_name} "
                f"(p={p_value:.4f}, d={effect_size:.2f})"
            )

        return StatisticalTest(
            test_name="paired_t_test",
            metric=metric,
            model_a=model_a.model_name,
            model_b=model_b.model_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=effect_size,
            interpretation=interpretation,
        )

    def create_comparison_table(
        self, model_results: Dict[str, ModelResults]
    ) -> pd.DataFrame:
        """Create comparison table with all metrics.

        Args:
            model_results: Dictionary mapping model names to ModelResults

        Returns:
            DataFrame with comparison statistics
        """
        rows = []
        for model_name, results in model_results.items():
            row = {
                "Model": model_name.replace("_", " ").title(),
                "Clean Acc (%)": f"{results.get_mean('clean_accuracy') * 100:.2f} "
                f"± {results.get_std('clean_accuracy') * 100:.2f}",
                "Robust Acc (%)": f"{results.get_mean('robust_accuracy') * 100:.2f} "
                f"± {results.get_std('robust_accuracy') * 100:.2f}",
                "Cross-Site AUROC": f"{results.get_mean('cross_site_auroc'):.4f} "
                f"± {results.get_std('cross_site_auroc'):.4f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def determine_orthogonality(
        self, statistical_tests: List[StatisticalTest]
    ) -> Tuple[bool, str]:
        """Determine if orthogonality is confirmed based on statistical tests.

        Orthogonality is confirmed if:
        1. Adversarial models (PGD-AT, TRADES) have significantly higher
           robust accuracy than baseline
        2. Adversarial models do NOT have significantly higher cross-site
           AUROC than baseline

        Args:
            statistical_tests: List of statistical test results

        Returns:
            Tuple of (is_orthogonal, summary_explanation)
        """
        # Find tests comparing adversarial models to baseline
        robust_improved = []
        generalization_improved = []

        for test in statistical_tests:
            if test.model_b == "baseline":
                if test.metric == "robust_accuracy":
                    robust_improved.append(test.is_significant and test.statistic > 0)
                elif test.metric == "cross_site_auroc":
                    generalization_improved.append(
                        test.is_significant and test.statistic > 0
                    )

        # Orthogonality confirmed if robustness improved but NOT generalization
        robustness_ok = any(robust_improved)
        generalization_ok = not any(generalization_improved)
        is_orthogonal = robustness_ok and generalization_ok

        if is_orthogonal:
            summary = (
                "✓ ORTHOGONALITY CONFIRMED: Adversarial training significantly "
                "improves robustness but does NOT improve cross-site generalization, "
                "confirming the need for tri-objective optimization."
            )
        else:
            issues = []
            if not robustness_ok:
                issues.append("adversarial training did not improve robustness")
            if not generalization_ok:
                issues.append(
                    "adversarial training unexpectedly improved generalization"
                )
            summary = f"✗ ORTHOGONALITY NOT CONFIRMED: {', '.join(issues)}"

        return is_orthogonal, summary

    def plot_metric_comparison(
        self, model_results: Dict[str, ModelResults], metric: str
    ) -> None:
        """Create bar plot comparing models on a single metric.

        Args:
            model_results: Dictionary mapping model names to ModelResults
            metric: Metric to plot (e.g., 'robust_accuracy')
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        models = list(model_results.keys())
        means = [model_results[m].get_mean(metric) for m in models]
        stds = [model_results[m].get_std(metric) for m in models]

        x_pos = np.arange(len(models))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace("_", " ").title() for m in models])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if self.config.save_figures:
            filename = f"comparison_{metric}.{self.config.figure_format}"
            filepath = self.config.output_dir / filename
            plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches="tight")
            self.logger.info(f"Saved figure: {filepath}")

        plt.close()

    def plot_orthogonality_scatter(
        self, model_results: Dict[str, ModelResults]
    ) -> None:
        """Create scatter plot showing robustness vs. generalization.

        Args:
            model_results: Dictionary mapping model names to ModelResults
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        colors = {"baseline": "blue", "pgd_at": "red", "trades": "green"}

        for model_name, results in model_results.items():
            robust = results.robust_accuracy
            cross_site = results.cross_site_auroc

            ax.scatter(
                robust,
                cross_site,
                label=model_name.replace("_", " ").title(),
                color=colors.get(model_name, "gray"),
                s=100,
                alpha=0.7,
            )

        ax.set_xlabel("Robust Accuracy (PGD-20)")
        ax.set_ylabel("Cross-Site AUROC")
        ax.set_title("Orthogonality: Robustness vs. Generalization")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if self.config.save_figures:
            filename = f"orthogonality_scatter.{self.config.figure_format}"
            filepath = self.config.output_dir / filename
            plt.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches="tight")
            self.logger.info(f"Saved figure: {filepath}")

        plt.close()

    def run_analysis(self) -> OrthogonalityResults:
        """Run complete orthogonality analysis.

        Returns:
            OrthogonalityResults with all analysis outputs

        Raises:
            FileNotFoundError: If required result files are missing
            ValueError: If analysis prerequisites are not met
        """
        self.logger.info(f"Starting orthogonality analysis for {self.config.dataset}")
        self.logger.info(f"Results directory: {self.config.results_dir}")
        self.logger.info(f"Seeds: {self.config.seeds}")

        # Load all model results
        model_results = {}
        for model_name in self.config.models:
            try:
                model_results[model_name] = self.load_model_results(model_name)
            except FileNotFoundError as e:
                self.logger.error(f"Failed to load {model_name}: {e}")
                raise

        # Run statistical tests
        statistical_tests = []
        baseline = model_results["baseline"]

        for model_name in ["pgd_at", "trades"]:
            if model_name not in model_results:
                continue

            adversarial = model_results[model_name]

            for metric in self.config.metrics:
                test = self.compute_statistical_test(metric, adversarial, baseline)
                statistical_tests.append(test)
                self.logger.info(test.interpretation)

        # Determine orthogonality
        is_orthogonal, summary = self.determine_orthogonality(statistical_tests)
        self.logger.info(summary)

        # Create comparison table
        comparison_table = self.create_comparison_table(model_results)

        # Generate plots
        for metric in self.config.metrics:
            self.plot_metric_comparison(model_results, metric)

        self.plot_orthogonality_scatter(model_results)

        # Create results object
        results = OrthogonalityResults(
            config=self.config,
            model_results=model_results,
            statistical_tests=statistical_tests,
            is_orthogonal=is_orthogonal,
            summary=summary,
            comparison_table=comparison_table,
        )

        # Save results
        results.save()

        self.logger.info("Orthogonality analysis complete")

        return results
