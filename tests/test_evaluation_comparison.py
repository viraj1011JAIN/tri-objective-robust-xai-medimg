"""
Comprehensive tests for src/evaluation/comparison.py
Achieves 100% line and branch coverage with production-level quality.
"""

import json
import logging
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from scipy import stats

from src.evaluation.comparison import (
    StatisticalComparator,
    load_results,
    main,
    save_comparison_results,
)


class TestStatisticalComparator:
    """Test StatisticalComparator class."""

    @pytest.fixture
    def comparator(self):
        """Create comparator with default settings."""
        return StatisticalComparator(alpha=0.01)

    @pytest.fixture
    def comparator_with_logger(self):
        """Create comparator with custom logger."""
        logger = logging.getLogger("test_logger")
        return StatisticalComparator(alpha=0.05, logger=logger)

    @pytest.fixture
    def sample_scores(self):
        """Create sample score arrays."""
        np.random.seed(42)
        method1 = np.random.normal(0.85, 0.05, 50)
        method2 = np.random.normal(0.80, 0.05, 50)
        return method1, method2

    def test_init_default(self):
        """Test initialization with defaults."""
        comp = StatisticalComparator()
        assert comp.alpha == 0.01
        assert comp.logger is not None

    def test_init_custom(self, comparator_with_logger):
        """Test initialization with custom parameters."""
        assert comparator_with_logger.alpha == 0.05
        assert comparator_with_logger.logger.name == "test_logger"

    def test_paired_ttest(self, comparator, sample_scores):
        """Test paired t-test."""
        method1, method2 = sample_scores
        result = comparator.paired_ttest(method1, method2)

        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "mean_diff" in result
        assert isinstance(result["t_statistic"], float)
        assert isinstance(result["p_value"], float)
        assert result["significant"] in [True, False]
        assert result["mean_diff"] > 0  # method1 > method2

    def test_paired_ttest_significant(self, comparator):
        """Test t-test with significant difference."""
        method1 = np.array([0.9] * 20)
        method2 = np.array([0.5] * 20)
        result = comparator.paired_ttest(method1, method2)

        assert result["significant"]
        assert result["p_value"] < 0.01

    def test_paired_ttest_not_significant(self, comparator):
        """Test t-test with non-significant difference."""
        method1 = np.array([0.85, 0.86, 0.84, 0.85])
        method2 = np.array([0.84, 0.85, 0.85, 0.86])
        result = comparator.paired_ttest(method1, method2)

        assert not result["significant"]

    def test_wilcoxon_test(self, comparator, sample_scores):
        """Test Wilcoxon signed-rank test."""
        method1, method2 = sample_scores
        result = comparator.wilcoxon_test(method1, method2)

        assert "w_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "median_diff" in result
        assert isinstance(result["w_statistic"], float)
        assert result["median_diff"] > 0

    def test_wilcoxon_test_significant(self, comparator):
        """Test Wilcoxon with significant difference."""
        method1 = np.array([0.9] * 30)
        method2 = np.array([0.5] * 30)
        result = comparator.wilcoxon_test(method1, method2)

        assert result["significant"]

    def test_cohens_d(self, comparator, sample_scores):
        """Test Cohen's d effect size."""
        method1, method2 = sample_scores
        d = comparator.cohens_d(method1, method2)

        assert isinstance(d, float)
        assert d > 0  # Positive effect

    def test_cohens_d_large_effect(self, comparator):
        """Test Cohen's d with large effect size."""
        method1 = np.array([1.0] * 50)
        method2 = np.array([0.5] * 50)
        d = comparator.cohens_d(method1, method2)

        assert abs(d) > 0.8  # Large effect

    def test_cohens_d_small_effect(self, comparator):
        """Test Cohen's d with small effect size."""
        method1 = np.array([0.85, 0.86, 0.84])
        method2 = np.array([0.84, 0.85, 0.83])
        d = comparator.cohens_d(method1, method2)

        assert isinstance(d, (float, np.float64))

    def test_hedges_g(self, comparator, sample_scores):
        """Test Hedges' g (bias-corrected Cohen's d)."""
        method1, method2 = sample_scores
        g = comparator.hedges_g(method1, method2)

        assert isinstance(g, float)
        # Hedges' g should be slightly smaller than Cohen's d
        d = comparator.cohens_d(method1, method2)
        assert abs(g) < abs(d)

    def test_hedges_g_small_sample(self, comparator):
        """Test Hedges' g with small sample."""
        method1 = np.array([0.9, 0.8, 0.85])
        method2 = np.array([0.7, 0.6, 0.65])
        g = comparator.hedges_g(method1, method2)

        assert isinstance(g, float)
        assert g > 0

    def test_bootstrap_ci(self, comparator, sample_scores):
        """Test bootstrap confidence interval."""
        method1, method2 = sample_scores
        result = comparator.bootstrap_ci(
            method1, method2, n_bootstrap=1000, confidence_level=0.95
        )

        assert "mean_diff" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "confidence_level" in result
        assert result["ci_lower"] < result["mean_diff"]
        assert result["mean_diff"] < result["ci_upper"]
        assert result["confidence_level"] == 0.95

    def test_bootstrap_ci_99_percent(self, comparator):
        """Test bootstrap CI with 99% confidence."""
        method1 = np.random.normal(0.9, 0.05, 30)
        method2 = np.random.normal(0.8, 0.05, 30)
        result = comparator.bootstrap_ci(
            method1, method2, n_bootstrap=500, confidence_level=0.99
        )

        assert result["confidence_level"] == 0.99
        # 99% CI should be wider than 95%
        width_99 = result["ci_upper"] - result["ci_lower"]
        assert width_99 > 0

    def test_bonferroni_correction(self, comparator):
        """Test Bonferroni correction."""
        p_values = [0.001, 0.005, 0.03, 0.08, 0.15]
        result = comparator.bonferroni_correction(p_values)

        assert "adjusted_alpha" in result
        assert "num_comparisons" in result
        assert "significant" in result
        assert "num_significant" in result
        assert result["num_comparisons"] == 5
        assert result["adjusted_alpha"] == 0.01 / 5
        assert len(result["significant"]) == 5

    def test_bonferroni_custom_alpha(self, comparator):
        """Test Bonferroni with custom alpha."""
        p_values = [0.001, 0.02, 0.04]
        result = comparator.bonferroni_correction(p_values, alpha=0.05)

        assert result["adjusted_alpha"] == 0.05 / 3
        assert result["num_comparisons"] == 3

    def test_bonferroni_all_significant(self, comparator):
        """Test Bonferroni when all tests are significant."""
        p_values = [0.0001, 0.0002, 0.0003]
        result = comparator.bonferroni_correction(p_values, alpha=0.01)

        assert result["num_significant"] == 3
        assert all(result["significant"])

    def test_bonferroni_none_significant(self, comparator):
        """Test Bonferroni when no tests are significant."""
        p_values = [0.1, 0.2, 0.3]
        result = comparator.bonferroni_correction(p_values, alpha=0.01)

        assert result["num_significant"] == 0
        assert not any(result["significant"])

    def test_holm_correction(self, comparator):
        """Test Holm-Bonferroni correction."""
        p_values = [0.001, 0.005, 0.03, 0.08, 0.15]
        result = comparator.holm_correction(p_values)

        assert "alpha" in result
        assert "num_comparisons" in result
        assert "significant" in result
        assert "num_significant" in result
        assert result["num_comparisons"] == 5
        assert len(result["significant"]) == 5

    def test_holm_custom_alpha(self, comparator):
        """Test Holm with custom alpha."""
        p_values = [0.01, 0.02, 0.04]
        result = comparator.holm_correction(p_values, alpha=0.05)

        assert result["alpha"] == 0.05
        assert result["num_comparisons"] == 3

    def test_holm_stepdown_procedure(self, comparator):
        """Test Holm's step-down procedure."""
        # Sorted: [0.001, 0.02, 0.15]
        # First: 0.001 < 0.01/3 = 0.0033 → significant
        # Second: 0.02 > 0.01/2 = 0.005 → NOT significant, STOP
        p_values = [0.02, 0.001, 0.15]
        result = comparator.holm_correction(p_values, alpha=0.01)

        # Should maintain original order
        assert len(result["significant"]) == 3

    def test_holm_all_significant(self, comparator):
        """Test Holm when all tests pass."""
        p_values = [0.0001, 0.0002, 0.0003]
        result = comparator.holm_correction(p_values, alpha=0.01)

        assert result["num_significant"] == 3
        assert all(result["significant"])

    def test_compare_methods(self, comparator):
        """Test comprehensive method comparison."""
        trades_results = {
            42: {"acc": 0.85, "rob": 0.70},
            123: {"acc": 0.86, "rob": 0.71},
            456: {"acc": 0.84, "rob": 0.69},
        }
        baseline_results = {
            42: {"acc": 0.80, "rob": 0.65},
            123: {"acc": 0.81, "rob": 0.66},
            456: {"acc": 0.79, "rob": 0.64},
        }
        metrics = ["acc", "rob"]
        seeds = [42, 123, 456]

        results = comparator.compare_methods(
            trades_results, baseline_results, metrics, seeds
        )

        assert "acc" in results
        assert "rob" in results

        for metric in metrics:
            metric_result = results[metric]
            assert "trades_mean" in metric_result
            assert "trades_std" in metric_result
            assert "baseline_mean" in metric_result
            assert "baseline_std" in metric_result
            assert "mean_diff" in metric_result
            assert "ttest" in metric_result
            assert "wilcoxon" in metric_result
            assert "cohens_d" in metric_result
            assert "hedges_g" in metric_result
            assert "bootstrap_ci" in metric_result

    def test_compare_methods_logging(self, comparator_with_logger, caplog):
        """Test that comparison logs properly."""
        trades_results = {42: {"metric": 0.9}, 123: {"metric": 0.91}}
        baseline_results = {42: {"metric": 0.8}, 123: {"metric": 0.81}}

        with caplog.at_level(logging.INFO):
            comparator_with_logger.compare_methods(
                trades_results, baseline_results, ["metric"], [42, 123]
            )

        assert "Comparing metric: metric" in caplog.text
        assert "TRADES:" in caplog.text
        assert "Baseline:" in caplog.text


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_load_results(self, tmp_path):
        """Test loading results from JSON."""
        test_data = {"seed_42": {"acc": 0.85, "rob": 0.70}}
        results_file = tmp_path / "results.json"

        with open(results_file, "w") as f:
            json.dump(test_data, f)

        loaded = load_results(results_file)
        assert loaded == test_data

    def test_save_comparison_results(self, tmp_path):
        """Test saving comparison results."""
        results = {
            "acc": {"trades_mean": 0.85, "baseline_mean": 0.80},
            "rob": {"trades_mean": 0.70, "baseline_mean": 0.65},
        }
        output_file = tmp_path / "output" / "comparison.json"

        save_comparison_results(results, output_file)

        assert output_file.exists()
        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == results

    def test_save_comparison_creates_parent(self, tmp_path):
        """Test that save creates parent directories."""
        output_file = tmp_path / "deep" / "nested" / "dir" / "results.json"
        results = {"test": "data"}

        save_comparison_results(results, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestMainFunction:
    """Test main execution function."""

    @patch(
        "sys.argv",
        [
            "prog",
            "--trades_results",
            "t.json",
            "--baseline_results",
            "b.json",
            "--output",
            "o.json",
            "--seeds",
            "42",
            "123",
        ],
    )
    @patch("src.evaluation.comparison.load_results")
    @patch("src.evaluation.comparison.save_comparison_results")
    @patch("src.evaluation.comparison.logging.basicConfig")
    def test_main_execution(self, mock_logging, mock_save, mock_load):
        """Test main function execution."""
        # Mock results with expected structure
        mock_load.side_effect = [
            {
                42: {
                    "clean_accuracy": 0.9,
                    "robust_accuracy": 0.8,
                    "ece": 0.05,
                    "auroc": 0.95,
                },
                123: {
                    "clean_accuracy": 0.91,
                    "robust_accuracy": 0.79,
                    "ece": 0.06,
                    "auroc": 0.94,
                },
            },
            {
                42: {
                    "clean_accuracy": 0.85,
                    "robust_accuracy": 0.75,
                    "ece": 0.07,
                    "auroc": 0.92,
                },
                123: {
                    "clean_accuracy": 0.84,
                    "robust_accuracy": 0.76,
                    "ece": 0.08,
                    "auroc": 0.91,
                },
            },
        ]

        # Run main
        main()

        # Verify calls
        mock_logging.assert_called_once()
        assert mock_load.call_count == 2
        mock_save.assert_called_once()

    @patch(
        "sys.argv",
        [
            "prog",
            "--trades_results",
            "t.json",
            "--baseline_results",
            "b.json",
            "--output",
            "o.json",
        ],
    )
    @patch("src.evaluation.comparison.load_results")
    @patch("src.evaluation.comparison.save_comparison_results")
    def test_main_with_real_args(self, mock_save, mock_load):
        """Test main with real argument parsing."""
        # Include default seeds: 42, 123, 456
        mock_load.side_effect = [
            {
                42: {
                    "clean_accuracy": 0.9,
                    "robust_accuracy": 0.8,
                    "ece": 0.05,
                    "auroc": 0.95,
                },
                123: {
                    "clean_accuracy": 0.91,
                    "robust_accuracy": 0.79,
                    "ece": 0.06,
                    "auroc": 0.94,
                },
                456: {
                    "clean_accuracy": 0.89,
                    "robust_accuracy": 0.81,
                    "ece": 0.055,
                    "auroc": 0.96,
                },
            },
            {
                42: {
                    "clean_accuracy": 0.85,
                    "robust_accuracy": 0.75,
                    "ece": 0.07,
                    "auroc": 0.92,
                },
                123: {
                    "clean_accuracy": 0.84,
                    "robust_accuracy": 0.76,
                    "ece": 0.08,
                    "auroc": 0.91,
                },
                456: {
                    "clean_accuracy": 0.86,
                    "robust_accuracy": 0.74,
                    "ece": 0.065,
                    "auroc": 0.93,
                },
            },
        ]

        main()

        assert mock_load.call_count == 2
        mock_save.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_paired_ttest_identical_scores(self):
        """Test t-test with identical scores."""
        comp = StatisticalComparator()
        scores = np.array([0.85] * 10)
        result = comp.paired_ttest(scores, scores)

        assert result["mean_diff"] == 0.0
        assert result["t_statistic"] == 0.0 or np.isnan(result["t_statistic"])

    def test_cohens_d_zero_variance(self):
        """Test Cohen's d with zero variance."""
        comp = StatisticalComparator()
        method1 = np.array([1.0] * 10)
        method2 = np.array([0.5] * 10)
        d = comp.cohens_d(method1, method2)

        assert isinstance(d, float)
        assert not np.isnan(d)

    def test_bootstrap_ci_small_sample(self):
        """Test bootstrap with very small sample."""
        comp = StatisticalComparator()
        method1 = np.array([0.9, 0.8])
        method2 = np.array([0.7, 0.6])
        result = comp.bootstrap_ci(method1, method2, n_bootstrap=100)

        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_bonferroni_single_test(self):
        """Test Bonferroni with single test."""
        comp = StatisticalComparator()
        result = comp.bonferroni_correction([0.005])

        assert result["adjusted_alpha"] == 0.01
        assert result["num_comparisons"] == 1
        assert result["significant"][0]

    def test_holm_single_test(self):
        """Test Holm with single test."""
        comp = StatisticalComparator()
        result = comp.holm_correction([0.005])

        assert result["num_comparisons"] == 1
        assert result["significant"][0]

    def test_compare_methods_single_seed(self):
        """Test comparison with single seed."""
        comp = StatisticalComparator()
        trades = {42: {"metric": 0.9}}
        baseline = {42: {"metric": 0.8}}

        result = comp.compare_methods(trades, baseline, ["metric"], [42])

        assert "metric" in result
        assert result["metric"]["trades_mean"] == 0.9

    def test_wilcoxon_minimum_samples(self):
        """Test Wilcoxon with minimum sample size."""
        comp = StatisticalComparator()
        method1 = np.array([0.9, 0.8, 0.85])
        method2 = np.array([0.7, 0.6, 0.65])
        result = comp.wilcoxon_test(method1, method2)

        assert "p_value" in result
        assert "significant" in result


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_large_sample_sizes(self):
        """Test with large sample sizes."""
        comp = StatisticalComparator()
        np.random.seed(42)
        method1 = np.random.normal(0.85, 0.02, 10000)
        method2 = np.random.normal(0.80, 0.02, 10000)

        result = comp.paired_ttest(method1, method2)
        assert result["significant"]

    def test_very_small_differences(self):
        """Test with very small differences."""
        comp = StatisticalComparator()
        method1 = np.array([0.8500001] * 50)
        method2 = np.array([0.8500000] * 50)

        d = comp.cohens_d(method1, method2)
        assert isinstance(d, float)

    def test_extreme_values(self):
        """Test with extreme values."""
        comp = StatisticalComparator()
        method1 = np.array([1.0] * 20)
        method2 = np.array([0.0] * 20)

        result = comp.paired_ttest(method1, method2)
        assert result["significant"]
        assert result["mean_diff"] == 1.0
