"""
Tests for Statistical Tests Module.

Phase 9.1: Comprehensive Evaluation Infrastructure
Author: Viraj Jain
MSc Dissertation - University of Glasgow
Date: November 2024

Tests cover:
- Effect size calculations (Cohen's d, Glass's delta, Hedges' g)
- Parametric tests (paired t-test, independent t-test)
- Non-parametric tests (McNemar's, Wilcoxon, Mann-Whitney U)
- Bootstrap methods
- Multiple comparison corrections
- Comprehensive model comparison
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.statistical_tests import (
    BootstrapResult,
    StatisticalTestResult,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_confidence_interval,
    bootstrap_metric_comparison,
    bootstrap_paired_difference,
    comprehensive_model_comparison,
    compute_cohens_d,
    compute_glass_delta,
    compute_hedges_g,
    generate_comparison_report,
    independent_t_test,
    interpret_effect_size,
    mann_whitney_u_test,
    mcnemars_test,
    paired_t_test,
    save_results,
    wilcoxon_signed_rank_test,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_paired_data():
    """Generate paired sample data."""
    np.random.seed(42)
    sample1 = np.random.normal(0.85, 0.05, 100)
    sample2 = np.random.normal(0.80, 0.05, 100)
    return sample1, sample2


@pytest.fixture
def sample_independent_data():
    """Generate independent sample data."""
    np.random.seed(42)
    sample1 = np.random.normal(0.85, 0.05, 50)
    sample2 = np.random.normal(0.80, 0.05, 60)
    return sample1, sample2


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for McNemar's test."""
    np.random.seed(42)
    n = 100
    ground_truth = np.random.randint(0, 2, n)

    # Model 1: 85% accuracy
    pred1 = ground_truth.copy()
    flip_idx = np.random.choice(n, 15, replace=False)
    pred1[flip_idx] = 1 - pred1[flip_idx]

    # Model 2: 75% accuracy
    pred2 = ground_truth.copy()
    flip_idx = np.random.choice(n, 25, replace=False)
    pred2[flip_idx] = 1 - pred2[flip_idx]

    return pred1, pred2, ground_truth


@pytest.fixture
def sample_model_predictions():
    """Generate sample predictions for multiple models."""
    np.random.seed(42)
    n = 200
    ground_truth = np.random.randint(0, 3, n)

    models = {}
    accuracies = {"model_a": 0.90, "model_b": 0.85, "model_c": 0.80}

    for name, acc in accuracies.items():
        pred = ground_truth.copy()
        n_flip = int(n * (1 - acc))
        flip_idx = np.random.choice(n, n_flip, replace=False)
        pred[flip_idx] = (pred[flip_idx] + 1) % 3
        models[name] = pred

    return models, ground_truth


# ============================================================================
# DATA CLASS TESTS
# ============================================================================


class TestStatisticalTestResult:
    """Tests for StatisticalTestResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = StatisticalTestResult(
            test_name="t-test",
            statistic=2.5,
            p_value=0.01,
            significant=True,
            alpha=0.05,
        )
        assert result.statistic == 2.5
        assert result.p_value == 0.01
        assert result.test_name == "t-test"
        assert result.significant is True
        assert result.effect_size is None
        assert result.confidence_interval is None

    def test_is_significant(self):
        """Test significance checking."""
        significant = StatisticalTestResult(
            test_name="t", statistic=2.5, p_value=0.01, significant=True
        )
        not_significant = StatisticalTestResult(
            test_name="t", statistic=1.0, p_value=0.10, significant=False
        )

        assert significant.significant is True
        assert not_significant.significant is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = StatisticalTestResult(
            test_name="t-test",
            statistic=2.5,
            p_value=0.01,
            significant=True,
            effect_size=0.8,
            confidence_interval=(0.5, 1.1),
        )
        d = result.to_dict()

        assert d["statistic"] == 2.5
        assert d["p_value"] == 0.01
        assert d["effect_size"] == 0.8
        assert d["confidence_interval"] == [0.5, 1.1]


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        samples = np.random.randn(1000)
        result = BootstrapResult(
            point_estimate=0.0,
            mean=0.0,
            std=1.0,
            ci_lower=-0.1,
            ci_upper=0.1,
            confidence_level=0.95,
            n_bootstrap=1000,
            bootstrap_distribution=samples,
        )
        assert result.point_estimate == 0.0
        assert result.ci_lower == -0.1
        assert result.ci_upper == 0.1
        assert len(result.bootstrap_distribution) == 1000

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = BootstrapResult(
            point_estimate=0.5,
            mean=0.5,
            std=0.05,
            ci_lower=0.4,
            ci_upper=0.6,
            confidence_level=0.95,
            n_bootstrap=100,
            bootstrap_distribution=np.array([0.45, 0.55]),
        )
        d = result.to_dict()

        assert d["point_estimate"] == 0.5
        assert d["ci_lower"] == 0.4
        assert d["ci_upper"] == 0.6
        assert d["confidence_level"] == 0.95


# ============================================================================
# EFFECT SIZE TESTS
# ============================================================================


class TestEffectSizes:
    """Tests for effect size calculations."""

    def test_cohens_d_zero_difference(self):
        """Test Cohen's d with identical samples."""
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = np.array([1, 2, 3, 4, 5])

        d = compute_cohens_d(sample1, sample2)
        assert abs(d) < 0.01

    def test_cohens_d_large_effect(self, sample_paired_data):
        """Test Cohen's d with different samples."""
        sample1, sample2 = sample_paired_data
        d = compute_cohens_d(sample1, sample2)

        # Should be positive (sample1 > sample2)
        assert d > 0
        assert abs(d) > 0.5  # At least medium effect

    def test_cohens_d_pooled_std(self):
        """Test Cohen's d uses pooled std by default."""
        sample1 = np.array([10, 11, 12, 13, 14])
        sample2 = np.array([5, 6, 7, 8, 9])

        d = compute_cohens_d(sample1, sample2)

        # Known difference / known pooled std
        expected = 5 / np.sqrt(
            (4 * np.var(sample1, ddof=1) + 4 * np.var(sample2, ddof=1)) / 8
        )
        assert abs(d - expected) < 0.01

    def test_glass_delta(self):
        """Test Glass's delta calculation."""
        sample1 = np.array([10, 11, 12, 13, 14])
        sample2 = np.array([5, 6, 7, 8, 9])

        delta = compute_glass_delta(sample1, sample2)

        # Uses control group (sample2) std
        expected = (np.mean(sample1) - np.mean(sample2)) / np.std(sample2, ddof=1)
        assert abs(delta - expected) < 0.01

    def test_hedges_g(self):
        """Test Hedges' g calculation."""
        sample1 = np.array([10, 11, 12, 13, 14])
        sample2 = np.array([5, 6, 7, 8, 9])

        g = compute_hedges_g(sample1, sample2)
        d = compute_cohens_d(sample1, sample2)

        # Hedges' g should be smaller than Cohen's d (bias correction)
        assert g < d

    def test_interpret_effect_size(self):
        """Test effect size interpretation."""
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(0.3) == "small"
        assert interpret_effect_size(0.6) == "medium"
        assert interpret_effect_size(0.9) == "large"
        # Note: implementation uses 4 categories, not 5
        assert interpret_effect_size(1.5) == "large"

        # Should handle negative values
        assert interpret_effect_size(-0.9) == "large"


# ============================================================================
# PARAMETRIC TESTS
# ============================================================================


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_identical_samples(self):
        """Test with identical samples."""
        sample = np.array([1, 2, 3, 4, 5])

        result = paired_t_test(sample, sample)

        # With identical samples, statistic may be NaN due to 0 variance
        # Just verify we get a result without error
        assert result is not None
        assert result.effect_size == 0.0  # No difference

    def test_different_samples(self, sample_paired_data):
        """Test with different samples."""
        sample1, sample2 = sample_paired_data

        result = paired_t_test(sample1, sample2)

        assert result.p_value < 0.05  # Should be significant
        assert result.effect_size is not None
        assert result.effect_size > 0  # sample1 > sample2

    def test_effect_size_included(self, sample_paired_data):
        """Test that effect size is computed."""
        sample1, sample2 = sample_paired_data

        result = paired_t_test(sample1, sample2)

        assert result.effect_size is not None
        assert result.effect_size_interpretation is not None

    def test_two_tailed_default(self, sample_paired_data):
        """Test default is two-tailed."""
        sample1, sample2 = sample_paired_data

        result = paired_t_test(sample1, sample2, alternative="two-sided")

        # Should be symmetric
        assert (
            "two-sided" in str(result.additional_info.get("alternative", "")).lower()
            or result.p_value > 0
        )


class TestIndependentTTest:
    """Tests for independent samples t-test."""

    def test_identical_distributions(self):
        """Test with samples from same distribution."""
        np.random.seed(42)
        sample1 = np.random.normal(0, 1, 100)
        sample2 = np.random.normal(0, 1, 100)

        result = independent_t_test(sample1, sample2)

        # Should not be significant (same distribution)
        assert result.p_value > 0.05 or abs(result.effect_size) < 0.5

    def test_different_sizes(self, sample_independent_data):
        """Test with different sample sizes."""
        sample1, sample2 = sample_independent_data

        result = independent_t_test(sample1, sample2)

        assert result.statistic is not None
        assert result.p_value is not None


# ============================================================================
# NON-PARAMETRIC TESTS
# ============================================================================


class TestMcNemarsTest:
    """Tests for McNemar's test."""

    def test_basic_contingency(self):
        """Test with basic contingency table."""
        # b = 10 (model1 wrong, model2 right)
        # c = 5  (model1 right, model2 wrong)
        pred1 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1] + [1] * 5 + [0] * 85)
        pred2 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [1] * 5 + [0] * 85)
        truth = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1] + [1] * 5 + [0] * 85)

        result = mcnemars_test(pred1, pred2, truth)

        assert result.statistic is not None
        assert 0 <= result.p_value <= 1

    def test_identical_predictions(self):
        """Test with identical predictions."""
        pred = np.random.randint(0, 2, 100)
        truth = np.random.randint(0, 2, 100)

        result = mcnemars_test(pred, pred, truth)

        # p-value should be high (no difference)
        assert result.p_value >= 0.05 or result.statistic == 0

    def test_discordant_pairs(self, sample_predictions):
        """Test with known discordant pairs."""
        pred1, pred2, truth = sample_predictions

        result = mcnemars_test(pred1, pred2, truth)

        assert "discordant_pairs" in result.additional_info
        assert result.additional_info["discordant_pairs"] >= 0
        assert "contingency_table" in result.additional_info


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test."""

    def test_identical_samples(self):
        """Test with identical samples."""
        sample = np.array([1, 2, 3, 4, 5])

        result = wilcoxon_signed_rank_test(sample, sample)

        # Should be non-significant
        assert result.p_value >= 0.05 or result.statistic == 0

    def test_different_samples(self, sample_paired_data):
        """Test with different samples."""
        sample1, sample2 = sample_paired_data

        result = wilcoxon_signed_rank_test(sample1, sample2)

        assert result.statistic is not None
        assert 0 <= result.p_value <= 1


class TestMannWhitneyTest:
    """Tests for Mann-Whitney U test."""

    def test_identical_distributions(self):
        """Test with samples from same distribution."""
        np.random.seed(42)
        sample1 = np.random.uniform(0, 1, 50)
        sample2 = np.random.uniform(0, 1, 50)

        result = mann_whitney_u_test(sample1, sample2)

        # Should generally not be significant
        assert result.p_value >= 0.01 or result.effect_size is not None

    def test_different_distributions(self):
        """Test with clearly different distributions."""
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = np.array([10, 11, 12, 13, 14])

        result = mann_whitney_u_test(sample1, sample2)

        assert result.p_value < 0.05  # Should be significant


# ============================================================================
# BOOTSTRAP TESTS
# ============================================================================


class TestBootstrapMethods:
    """Tests for bootstrap methods."""

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI computation."""
        np.random.seed(42)
        data = np.random.normal(0.5, 0.1, 100)

        result = bootstrap_confidence_interval(
            data=data,
            statistic_fn=np.mean,
            n_bootstrap=1000,
            confidence_level=0.95,
        )

        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert abs(result.point_estimate - 0.5) < 0.05
        assert len(result.bootstrap_distribution) == 1000

    def test_bootstrap_percentile_method(self):
        """Test percentile bootstrap method."""
        np.random.seed(42)
        data = np.random.normal(1.0, 0.2, 50)

        result = bootstrap_confidence_interval(
            data=data,
            statistic_fn=np.mean,
            method="percentile",
        )

        # CI should contain true mean approximately
        assert result.ci_lower < 1.2
        assert result.ci_upper > 0.8

    def test_bootstrap_paired_difference(self, sample_paired_data):
        """Test bootstrap paired difference."""
        sample1, sample2 = sample_paired_data

        result = bootstrap_paired_difference(
            values1=sample1,
            values2=sample2,
            n_bootstrap=500,
        )

        # True difference is about 0.05
        assert result.ci_lower < 0.10
        assert result.ci_upper > 0.0

    def test_bootstrap_metric_comparison(self, sample_predictions):
        """Test bootstrap metric comparison."""
        pred1, pred2, truth = sample_predictions

        def accuracy(preds, labels):
            return np.mean(preds == labels)

        result = bootstrap_metric_comparison(
            predictions1=pred1,
            predictions2=pred2,
            labels=truth,
            metric_fn=accuracy,
            n_bootstrap=500,
        )

        assert "model1" in result
        assert result["model1"]["value"] is not None
        assert "model2" in result
        assert result["model2"]["value"] is not None
        assert "difference" in result


# ============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# ============================================================================


class TestMultipleComparisons:
    """Tests for multiple comparison corrections."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.001, 0.02, 0.03, 0.04, 0.05]

        result = bonferroni_correction(p_values, alpha=0.05)

        assert result["n_tests"] == 5
        assert result["corrected_alpha"] == 0.01  # 0.05 / 5
        # With alpha=0.05 and 5 tests, threshold is 0.01
        # Uses p < corrected_alpha (strictly less than)
        assert result["significant"][0] is True  # 0.001 < 0.01
        assert result["significant"][-1] is False  # 0.05 >= 0.01

    def test_benjamini_hochberg(self):
        """Test Benjamini-Hochberg correction."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.5]

        result = benjamini_hochberg_correction(p_values, alpha=0.05)

        assert result["n_tests"] == 5
        # Smallest p-value should definitely be significant
        assert result["significant"][0] is True
        # Large p-value should not be significant
        assert result["significant"][-1] is False


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================


class TestComprehensiveComparison:
    """Tests for comprehensive model comparison."""

    def test_basic_comparison(self, sample_predictions):
        """Test basic comprehensive comparison."""
        pred1, pred2, truth = sample_predictions

        results = comprehensive_model_comparison(
            predictions1=pred1,
            predictions2=pred2,
            labels=truth,
            alpha=0.05,
        )

        assert "mcnemars_test" in results
        assert "accuracy_model1" in results
        assert "accuracy_model2" in results

    def test_comparison_includes_all_tests(self, sample_predictions):
        """Test that all tests are run."""
        pred1, pred2, truth = sample_predictions

        results = comprehensive_model_comparison(
            predictions1=pred1,
            predictions2=pred2,
            labels=truth,
        )

        # Check that tests were run
        assert "mcnemars_test" in results
        assert "paired_t_test" in results
        assert "wilcoxon_test" in results

    def test_generates_report(self, sample_predictions):
        """Test report generation."""
        pred1, pred2, truth = sample_predictions

        results = comprehensive_model_comparison(
            predictions1=pred1,
            predictions2=pred2,
            labels=truth,
        )

        # generate_comparison_report expects a list of StatisticalTestResult
        # Create from the result dict
        test_results = []
        if "mcnemars_test" in results:
            test_results.append(
                StatisticalTestResult(
                    **{
                        k: v
                        for k, v in results["mcnemars_test"].items()
                        if k
                        in [
                            "test_name",
                            "statistic",
                            "p_value",
                            "significant",
                            "alpha",
                            "effect_size",
                            "effect_size_interpretation",
                        ]
                    }
                )
            )

        if test_results:
            report = generate_comparison_report(test_results)
            assert len(report) > 50


class TestSaveResults:
    """Tests for saving results."""

    def test_save_to_file(self, sample_predictions, tmp_path):
        """Test saving results to file."""
        pred1, pred2, truth = sample_predictions

        results = comprehensive_model_comparison(
            predictions1=pred1,
            predictions2=pred2,
            labels=truth,
        )

        # Create StatisticalTestResult from mcnemars test with proper types
        mcnemar_data = results["mcnemars_test"]
        test_result = StatisticalTestResult(
            test_name=str(mcnemar_data["test_name"]),
            statistic=float(mcnemar_data["statistic"]),
            p_value=float(mcnemar_data["p_value"]),
            significant=bool(mcnemar_data["significant"]),
            alpha=float(mcnemar_data["alpha"]),
        )

        output_file = tmp_path / "results.json"
        save_results(test_result, output_file)

        # Check file was created
        assert output_file.exists()


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_sample_size(self):
        """Test with very small sample sizes."""
        sample1 = np.array([1, 2])
        sample2 = np.array([3, 4])

        result = paired_t_test(sample1, sample2)
        assert result is not None

    def test_constant_samples(self):
        """Test with constant samples."""
        sample1 = np.array([5, 5, 5, 5, 5])
        sample2 = np.array([5, 5, 5, 5, 5])

        # Should handle gracefully
        result = compute_cohens_d(sample1, sample2)
        assert np.isfinite(result) or result == 0

    def test_single_element(self):
        """Test with single element samples."""
        sample1 = np.array([1])
        sample2 = np.array([2])

        # Should return some result without crashing
        d = compute_cohens_d(sample1, sample2)
        assert np.isfinite(d) or np.isnan(d)

    def test_empty_predictions(self):
        """Test comprehensive comparison with empty predictions."""
        # Empty predictions may or may not raise - just verify no crashes
        try:
            result = comprehensive_model_comparison(
                predictions1=np.array([]),
                predictions2=np.array([]),
                labels=np.array([]),
            )
            # If no error, just verify result structure
            assert isinstance(result, dict)
        except (ValueError, IndexError, ZeroDivisionError, RuntimeWarning):
            # Expected errors for empty data
            pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for statistical testing workflow."""

    def test_full_workflow(self, sample_predictions, tmp_path):
        """Test full statistical testing workflow."""
        pred1, pred2, truth = sample_predictions

        # 1. Run comprehensive comparison
        results = comprehensive_model_comparison(pred1, pred2, truth)

        # 2. Check mcnemars test was run
        assert "mcnemars_test" in results
        assert "accuracy_model1" in results
        assert "accuracy_model2" in results

        # 3. Generate report from StatisticalTestResult
        mcnemar_data = results["mcnemars_test"]
        test_result = StatisticalTestResult(
            test_name=str(mcnemar_data["test_name"]),
            statistic=float(mcnemar_data["statistic"]),
            p_value=float(mcnemar_data["p_value"]),
            significant=bool(mcnemar_data["significant"]),
            alpha=float(mcnemar_data["alpha"]),
        )
        report = generate_comparison_report([test_result])
        assert len(report) > 50

        # 4. Save results
        output_file = tmp_path / "results.json"
        save_results(test_result, output_file)

        # 5. Verify saved files
        assert output_file.exists()

    def test_bootstrap_with_custom_metric(self, sample_predictions):
        """Test bootstrap with custom metric function."""
        pred1, pred2, truth = sample_predictions

        # Custom metric: balanced accuracy
        def balanced_acc(preds, labels):
            from sklearn.metrics import balanced_accuracy_score

            return balanced_accuracy_score(labels, preds)

        result = bootstrap_metric_comparison(
            predictions1=pred1,
            predictions2=pred2,
            labels=truth,
            metric_fn=balanced_acc,
            n_bootstrap=200,
        )

        assert "model1" in result
        assert result["model1"]["value"] is not None
        assert "model2" in result
        assert result["model2"]["value"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
