"""
================================================================================
Comparison Utilities - Phase 5.3
================================================================================
Statistical comparison tools for TRADES vs PGD-AT analysis.

Features:
    - Paired statistical tests (t-test, Wilcoxon signed-rank)
    - Effect size computation (Cohen's d, Hedges' g)
    - Bootstrap confidence intervals
    - Multiple comparison correction (Bonferroni, Holm)
    - Performance delta analysis

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 2025
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample


class StatisticalComparator:
    """Statistical comparison between TRADES and baseline methods."""

    def __init__(self, alpha: float = 0.01, logger: Optional[logging.Logger] = None):
        """
        Initialize statistical comparator.

        Args:
            alpha: Significance level for hypothesis testing
            logger: Logger instance
        """
        self.alpha = alpha
        self.logger = logger or logging.getLogger(__name__)

    def paired_ttest(
        self, method1_scores: np.ndarray, method2_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform paired t-test.

        Args:
            method1_scores: Scores from method 1 (e.g., TRADES)
            method2_scores: Scores from method 2 (e.g., PGD-AT)

        Returns:
            Dictionary with test statistics
        """
        statistic, pvalue = stats.ttest_rel(method1_scores, method2_scores)

        return {
            "t_statistic": float(statistic),
            "p_value": float(pvalue),
            "significant": pvalue < self.alpha,
            "mean_diff": float(np.mean(method1_scores - method2_scores)),
        }

    def wilcoxon_test(
        self, method1_scores: np.ndarray, method2_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).

        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2

        Returns:
            Dictionary with test statistics
        """
        statistic, pvalue = stats.wilcoxon(method1_scores, method2_scores)

        return {
            "w_statistic": float(statistic),
            "p_value": float(pvalue),
            "significant": pvalue < self.alpha,
            "median_diff": float(np.median(method1_scores - method2_scores)),
        }

    def cohens_d(self, method1_scores: np.ndarray, method2_scores: np.ndarray) -> float:
        """
        Compute Cohen's d effect size.

        Cohen's d = (mean1 - mean2) / pooled_std

        Interpretation:
            |d| < 0.2: negligible
            0.2 ≤ |d| < 0.5: small
            0.5 ≤ |d| < 0.8: medium
            |d| ≥ 0.8: large

        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2

        Returns:
            Cohen's d value
        """
        n1, n2 = len(method1_scores), len(method2_scores)
        var1, var2 = np.var(method1_scores, ddof=1), np.var(method2_scores, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        d = (np.mean(method1_scores) - np.mean(method2_scores)) / pooled_std

        return float(d)

    def hedges_g(self, method1_scores: np.ndarray, method2_scores: np.ndarray) -> float:
        """
        Compute Hedges' g (bias-corrected Cohen's d).

        Hedges' g = Cohen's d × correction_factor

        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2

        Returns:
            Hedges' g value
        """
        d = self.cohens_d(method1_scores, method2_scores)
        n = len(method1_scores) + len(method2_scores)

        # Correction factor
        correction = 1 - (3 / (4 * n - 9))

        return float(d * correction)

    def bootstrap_ci(
        self,
        method1_scores: np.ndarray,
        method2_scores: np.ndarray,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.99,
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for mean difference.

        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 99%)

        Returns:
            Dictionary with CI bounds
        """
        diffs = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample1 = resample(
                method1_scores, replace=True, n_samples=len(method1_scores)
            )
            sample2 = resample(
                method2_scores, replace=True, n_samples=len(method2_scores)
            )

            # Compute mean difference
            diff = np.mean(sample1) - np.mean(sample2)
            diffs.append(diff)

        diffs = np.array(diffs)

        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(diffs, lower_percentile)
        ci_upper = np.percentile(diffs, upper_percentile)

        return {
            "mean_diff": float(np.mean(diffs)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": confidence_level,
        }

    def bonferroni_correction(
        self, p_values: List[float], alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Adjusted alpha = alpha / num_comparisons

        Args:
            p_values: List of p-values
            alpha: Significance level (uses self.alpha if None)

        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha

        num_comparisons = len(p_values)
        adjusted_alpha = alpha / num_comparisons

        significant = [p < adjusted_alpha for p in p_values]

        return {
            "adjusted_alpha": adjusted_alpha,
            "num_comparisons": num_comparisons,
            "significant": significant,
            "num_significant": sum(significant),
        }

    def holm_correction(
        self, p_values: List[float], alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply Holm-Bonferroni correction (less conservative than Bonferroni).

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha

        # Sort p-values
        num_comparisons = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]

        # Apply Holm correction
        significant = []
        for i, p in enumerate(sorted_pvalues):
            adjusted_alpha = alpha / (num_comparisons - i)
            if p < adjusted_alpha:
                significant.append(True)
            else:
                # Once we fail to reject, stop (step-down procedure)
                significant.extend([False] * (num_comparisons - i))
                break

        # Restore original order
        significant_original_order = [False] * num_comparisons
        for i, sig in zip(sorted_indices, significant):
            significant_original_order[i] = sig

        return {
            "alpha": alpha,
            "num_comparisons": num_comparisons,
            "significant": significant_original_order,
            "num_significant": sum(significant_original_order),
        }

    def compare_methods(
        self,
        trades_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        metrics: List[str],
        seeds: List[int],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive comparison between TRADES and baseline.

        Args:
            trades_results: TRADES results dict (seed -> metrics)
            baseline_results: Baseline results dict (seed -> metrics)
            metrics: List of metrics to compare
            seeds: List of random seeds

        Returns:
            Dictionary with comparison results for each metric
        """
        comparison_results = {}

        for metric in metrics:
            self.logger.info(f"Comparing metric: {metric}")

            # Extract scores for each seed
            trades_scores = np.array([trades_results[seed][metric] for seed in seeds])
            baseline_scores = np.array(
                [baseline_results[seed][metric] for seed in seeds]
            )

            # Compute all statistics
            ttest_result = self.paired_ttest(trades_scores, baseline_scores)
            wilcoxon_result = self.wilcoxon_test(trades_scores, baseline_scores)
            cohens_d = self.cohens_d(trades_scores, baseline_scores)
            hedges_g = self.hedges_g(trades_scores, baseline_scores)
            bootstrap_ci = self.bootstrap_ci(trades_scores, baseline_scores)

            # Aggregate results
            comparison_results[metric] = {
                "trades_mean": float(np.mean(trades_scores)),
                "trades_std": float(np.std(trades_scores)),
                "baseline_mean": float(np.mean(baseline_scores)),
                "baseline_std": float(np.std(baseline_scores)),
                "mean_diff": float(np.mean(trades_scores - baseline_scores)),
                "ttest": ttest_result,
                "wilcoxon": wilcoxon_result,
                "cohens_d": cohens_d,
                "hedges_g": hedges_g,
                "bootstrap_ci": bootstrap_ci,
            }

            # Log summary
            self.logger.info(
                f"  TRADES: {np.mean(trades_scores):.4f} ± {np.std(trades_scores):.4f}"
            )
            self.logger.info(
                f"  Baseline: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}"
            )
            self.logger.info(f"  Diff: {np.mean(trades_scores - baseline_scores):.4f}")
            self.logger.info(f"  p-value: {ttest_result['p_value']:.6f}")
            self.logger.info(f"  Cohen's d: {cohens_d:.4f}")

        return comparison_results


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def save_comparison_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save comparison results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare TRADES vs Baseline")
    parser.add_argument("--trades_results", type=str, required=True)
    parser.add_argument("--baseline_results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load results
    logger.info("Loading results...")
    trades_results = load_results(Path(args.trades_results))
    baseline_results = load_results(Path(args.baseline_results))

    # Initialize comparator
    comparator = StatisticalComparator(alpha=0.01, logger=logger)

    # Compare methods
    metrics = ["clean_accuracy", "robust_accuracy", "ece", "auroc"]
    comparison_results = comparator.compare_methods(
        trades_results, baseline_results, metrics, args.seeds
    )

    # Save results
    save_comparison_results(comparison_results, Path(args.output))
    logger.info(f"Saved comparison results to {args.output}")


if __name__ == "__main__":
    main()
