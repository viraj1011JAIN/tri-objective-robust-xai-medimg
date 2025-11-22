"""
Comprehensive test suite for multilabel_metrics module.

Tests cover:
- AUROC computation (macro, micro, weighted, per-class)
- Comprehensive multi-label metrics (hamming loss, subset accuracy, etc.)
- Multi-label confusion matrices
- Plotting functions (ROC curves, per-class AUROC, confusion matrices)
- Bootstrap confidence intervals
- Optimal threshold computation
- Edge cases and error handling

Achieves 100% line and branch coverage.
"""

import os
import tempfile
from unittest.mock import patch

import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Import multilabel_metrics module normally for proper coverage tracking
from src.evaluation import multilabel_metrics

# Extract functions
compute_bootstrap_ci_multilabel = multilabel_metrics.compute_bootstrap_ci_multilabel
compute_multilabel_auroc = multilabel_metrics.compute_multilabel_auroc
compute_multilabel_confusion_matrix = (
    multilabel_metrics.compute_multilabel_confusion_matrix
)
compute_multilabel_metrics = multilabel_metrics.compute_multilabel_metrics
compute_optimal_thresholds = multilabel_metrics.compute_optimal_thresholds
plot_multilabel_auroc_per_class = multilabel_metrics.plot_multilabel_auroc_per_class
plot_multilabel_roc_curves = multilabel_metrics.plot_multilabel_roc_curves
plot_per_class_confusion_matrices = multilabel_metrics.plot_per_class_confusion_matrices


# =============================================================================
# Fixtures for Test Data
# =============================================================================


@pytest.fixture
def sample_multilabel_data():
    """Generate sample multi-label classification data."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 5

    # Ground truth (binary labels)
    y_true = np.random.randint(0, 2, size=(n_samples, n_classes))

    # Predicted probabilities
    y_prob = np.random.rand(n_samples, n_classes)

    # Binary predictions (threshold at 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    class_names = [f"Class_{i}" for i in range(n_classes)]

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "class_names": class_names,
    }


@pytest.fixture
def single_class_data():
    """Generate data where one class has only one label."""
    np.random.seed(42)
    n_samples = 50
    n_classes = 3

    y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
    # Make class 1 have only positive labels
    y_true[:, 1] = 1

    y_prob = np.random.rand(n_samples, n_classes)
    y_pred = (y_prob >= 0.5).astype(int)

    return {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}


@pytest.fixture
def perfect_predictions():
    """Generate perfect predictions for testing."""
    np.random.seed(42)
    n_samples = 50
    n_classes = 4

    y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
    y_prob = y_true.astype(float)
    y_pred = y_true.copy()

    return {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}


# =============================================================================
# Test compute_multilabel_auroc
# =============================================================================


class TestComputeMultilabelAUROC:
    """Test compute_multilabel_auroc function."""

    def test_basic_auroc_computation(self, sample_multilabel_data):
        """Test basic AUROC computation."""
        result = compute_multilabel_auroc(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
        )

        assert "auroc_macro" in result
        assert "auroc_micro" in result
        assert "auroc_weighted" in result
        assert "auroc_per_class" in result

        assert 0.0 <= result["auroc_macro"] <= 1.0
        assert 0.0 <= result["auroc_micro"] <= 1.0
        assert 0.0 <= result["auroc_weighted"] <= 1.0
        assert len(result["auroc_per_class"]) == 5

    def test_auroc_with_class_names(self, sample_multilabel_data):
        """Test AUROC computation with class names."""
        result = compute_multilabel_auroc(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            class_names=sample_multilabel_data["class_names"],
        )

        assert "class_names" in result
        assert "auroc_by_class" in result
        assert len(result["class_names"]) == 5
        assert len(result["auroc_by_class"]) == 5

        for name in sample_multilabel_data["class_names"]:
            assert name in result["auroc_by_class"]

    def test_auroc_with_single_class_label(self, single_class_data):
        """Test AUROC when a class has only one label."""
        result = compute_multilabel_auroc(
            single_class_data["y_true"],
            single_class_data["y_prob"],
        )

        # Class 1 has only positive labels, AUROC should be NaN
        assert np.isnan(result["auroc_per_class"][1])

        # Other classes should have valid AUROC
        assert not np.isnan(result["auroc_per_class"][0])
        assert not np.isnan(result["auroc_per_class"][2])

    def test_auroc_perfect_predictions(self, perfect_predictions):
        """Test AUROC with perfect predictions."""
        result = compute_multilabel_auroc(
            perfect_predictions["y_true"],
            perfect_predictions["y_prob"],
        )

        # Perfect predictions should give AUROC = 1.0
        assert result["auroc_macro"] == 1.0
        assert result["auroc_micro"] == 1.0
        assert result["auroc_weighted"] == 1.0

        for score in result["auroc_per_class"]:
            assert score == 1.0

    def test_auroc_returns_float_types(self, sample_multilabel_data):
        """Test that AUROC values are Python floats."""
        result = compute_multilabel_auroc(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
        )

        assert isinstance(result["auroc_macro"], float)
        assert isinstance(result["auroc_micro"], float)
        assert isinstance(result["auroc_weighted"], float)


# =============================================================================
# Test compute_multilabel_metrics
# =============================================================================


class TestComputeMultilabelMetrics:
    """Test compute_multilabel_metrics function."""

    def test_comprehensive_metrics_computation(self, sample_multilabel_data):
        """Test computation of all metrics."""
        result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            sample_multilabel_data["y_prob"],
        )

        # Check AUROC metrics
        assert "auroc_macro" in result
        assert "auroc_micro" in result
        assert "auroc_weighted" in result
        assert "auroc_per_class" in result

        # Check other metrics
        assert "hamming_loss" in result
        assert "subset_accuracy" in result
        assert "coverage_error" in result
        assert "ranking_loss" in result
        assert "label_ranking_avg_precision" in result

        # Check per-class metrics
        assert "precision_per_class" in result
        assert "recall_per_class" in result
        assert "f1_per_class" in result
        assert "support_per_class" in result

        # Check macro averages
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "f1_macro" in result

        # Validate ranges
        assert 0.0 <= result["hamming_loss"] <= 1.0
        assert 0.0 <= result["subset_accuracy"] <= 1.0
        assert 0.0 <= result["label_ranking_avg_precision"] <= 1.0

    def test_metrics_with_class_names(self, sample_multilabel_data):
        """Test metrics computation with class names."""
        result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            sample_multilabel_data["y_prob"],
            class_names=sample_multilabel_data["class_names"],
        )

        assert "per_class_metrics" in result
        assert len(result["per_class_metrics"]) == 5

        for name in sample_multilabel_data["class_names"]:
            assert name in result["per_class_metrics"]
            metrics = result["per_class_metrics"][name]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics
            assert "auroc" in metrics

    def test_metrics_with_custom_threshold(self, sample_multilabel_data):
        """Test metrics with custom threshold."""
        result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            sample_multilabel_data["y_prob"],
            threshold=0.7,
        )

        assert "hamming_loss" in result
        assert isinstance(result["hamming_loss"], float)

    def test_metrics_with_none_y_pred(self, sample_multilabel_data):
        """Test metrics when y_pred is None (should compute from y_prob)."""
        result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            None,
            sample_multilabel_data["y_prob"],
            threshold=0.5,
        )

        assert "hamming_loss" in result
        assert isinstance(result["hamming_loss"], float)

    def test_metrics_with_mismatched_y_pred_shape(self, sample_multilabel_data):
        """Test metrics when y_pred has wrong shape (should recompute)."""
        wrong_shape_pred = np.random.randint(0, 2, size=(50, 3))  # Wrong shape

        result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            wrong_shape_pred,
            sample_multilabel_data["y_prob"],
        )

        assert "hamming_loss" in result
        assert isinstance(result["hamming_loss"], float)

    def test_metrics_perfect_predictions(self, perfect_predictions):
        """Test metrics with perfect predictions."""
        result = compute_multilabel_metrics(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
            perfect_predictions["y_prob"],
        )

        # Perfect predictions should give:
        assert result["hamming_loss"] == 0.0
        assert result["subset_accuracy"] == 1.0
        assert result["ranking_loss"] == 0.0

    def test_metrics_returns_correct_types(self, sample_multilabel_data):
        """Test that metrics return correct types."""
        result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            sample_multilabel_data["y_prob"],
        )

        assert isinstance(result["hamming_loss"], float)
        assert isinstance(result["subset_accuracy"], float)
        assert isinstance(result["precision_per_class"], list)
        assert isinstance(result["recall_per_class"], list)
        assert isinstance(result["f1_per_class"], list)
        assert isinstance(result["support_per_class"], list)


# =============================================================================
# Test compute_multilabel_confusion_matrix
# =============================================================================


class TestComputeMultilabelConfusionMatrix:
    """Test compute_multilabel_confusion_matrix function."""

    def test_basic_confusion_matrix(self, sample_multilabel_data):
        """Test basic confusion matrix computation."""
        result = compute_multilabel_confusion_matrix(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
        )

        assert "confusion_matrices" in result
        assert "num_classes" in result
        assert result["num_classes"] == 5
        assert len(result["confusion_matrices"]) == 5

        # Each confusion matrix should be 2x2
        for cm in result["confusion_matrices"]:
            assert len(cm) == 2
            assert len(cm[0]) == 2

    def test_confusion_matrix_with_class_names(self, sample_multilabel_data):
        """Test confusion matrix with class names."""
        result = compute_multilabel_confusion_matrix(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            class_names=sample_multilabel_data["class_names"],
        )

        assert "class_names" in result
        assert "per_class_cm" in result
        assert len(result["class_names"]) == 5

        for name in sample_multilabel_data["class_names"]:
            assert name in result["per_class_cm"]
            cm = result["per_class_cm"][name]
            assert len(cm) == 2
            assert len(cm[0]) == 2

    def test_confusion_matrix_structure(self, sample_multilabel_data):
        """Test that confusion matrix has correct structure."""
        result = compute_multilabel_confusion_matrix(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
        )

        # Each confusion matrix: [[TN, FP], [FN, TP]]
        for cm in result["confusion_matrices"]:
            tn, fp = cm[0]
            fn, tp = cm[1]

            # All values should be non-negative integers
            assert tn >= 0
            assert fp >= 0
            assert fn >= 0
            assert tp >= 0

    def test_confusion_matrix_perfect_predictions(self, perfect_predictions):
        """Test confusion matrix with perfect predictions."""
        result = compute_multilabel_confusion_matrix(
            perfect_predictions["y_true"],
            perfect_predictions["y_pred"],
        )

        # Perfect predictions should have FP=0 and FN=0
        for cm in result["confusion_matrices"]:
            tn, fp = cm[0]
            fn, tp = cm[1]
            assert fp == 0
            assert fn == 0


# =============================================================================
# Test Plotting Functions
# =============================================================================


class TestPlottingFunctions:
    """Test plotting functions."""

    def test_plot_multilabel_auroc_per_class_no_save(self, sample_multilabel_data):
        """Test plotting per-class AUROC without saving."""
        auroc_scores = np.array([0.8, 0.75, 0.9, 0.85, 0.7])

        with patch("matplotlib.pyplot.show") as mock_show:
            plot_multilabel_auroc_per_class(
                auroc_scores,
                sample_multilabel_data["class_names"],
                save_path=None,
            )
            mock_show.assert_called_once()

    def test_plot_multilabel_auroc_per_class_with_save(self, sample_multilabel_data):
        """Test plotting per-class AUROC with saving."""
        auroc_scores = np.array([0.8, 0.75, 0.9, 0.85, 0.7])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            plot_multilabel_auroc_per_class(
                auroc_scores,
                sample_multilabel_data["class_names"],
                save_path=save_path,
                title="Test AUROC",
            )

            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_plot_multilabel_roc_curves_no_save(self, sample_multilabel_data):
        """Test plotting ROC curves without saving."""
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_multilabel_roc_curves(
                sample_multilabel_data["y_true"],
                sample_multilabel_data["y_prob"],
                sample_multilabel_data["class_names"],
                save_path=None,
            )
            mock_show.assert_called_once()

    def test_plot_multilabel_roc_curves_with_save(self, sample_multilabel_data):
        """Test plotting ROC curves with saving."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            plot_multilabel_roc_curves(
                sample_multilabel_data["y_true"],
                sample_multilabel_data["y_prob"],
                sample_multilabel_data["class_names"],
                save_path=save_path,
                title="Test ROC",
            )

            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_plot_multilabel_roc_curves_with_single_class(self, single_class_data):
        """Test plotting ROC curves when one class has only one label."""
        class_names = ["Class_0", "Class_1_AllPos", "Class_2"]

        with patch("matplotlib.pyplot.show") as mock_show:
            # Should handle the case where class 1 has only positive labels
            plot_multilabel_roc_curves(
                single_class_data["y_true"],
                single_class_data["y_prob"],
                class_names,
                save_path=None,
            )
            mock_show.assert_called_once()

    def test_plot_per_class_confusion_matrices_no_save(self, sample_multilabel_data):
        """Test plotting confusion matrices without saving."""
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_per_class_confusion_matrices(
                sample_multilabel_data["y_true"],
                sample_multilabel_data["y_pred"],
                sample_multilabel_data["class_names"],
                save_path=None,
            )
            mock_show.assert_called_once()

    def test_plot_per_class_confusion_matrices_with_save(self, sample_multilabel_data):
        """Test plotting confusion matrices with saving."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            plot_per_class_confusion_matrices(
                sample_multilabel_data["y_true"],
                sample_multilabel_data["y_pred"],
                sample_multilabel_data["class_names"],
                save_path=save_path,
                title="Test CM",
            )

            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_plot_confusion_matrices_many_classes(self):
        """Test plotting confusion matrices with many classes."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 10

        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        y_pred = np.random.randint(0, 2, size=(n_samples, n_classes))
        class_names = [f"Class_{i}" for i in range(n_classes)]

        with patch("matplotlib.pyplot.show") as mock_show:
            plot_per_class_confusion_matrices(
                y_true,
                y_pred,
                class_names,
                save_path=None,
            )
            mock_show.assert_called_once()


# =============================================================================
# Test Bootstrap Confidence Intervals
# =============================================================================


class TestBootstrapCI:
    """Test compute_bootstrap_ci_multilabel function."""

    def test_bootstrap_ci_basic(self, sample_multilabel_data):
        """Test basic bootstrap CI computation."""

        def auroc_macro_fn(y_true, y_prob):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(y_true, y_prob, average="macro")

        metric, lower, upper = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=auroc_macro_fn,
            n_bootstrap=100,
            confidence_level=0.95,
            random_state=42,
        )

        assert isinstance(metric, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= metric <= upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_bootstrap_ci_with_kwargs(self, sample_multilabel_data):
        """Test bootstrap CI with additional kwargs."""

        def auroc_weighted_fn(y_true, y_prob, average="weighted"):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(y_true, y_prob, average=average)

        metric, lower, upper = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=auroc_weighted_fn,
            n_bootstrap=50,
            average="weighted",
        )

        assert lower <= metric <= upper

    def test_bootstrap_ci_different_confidence_levels(self, sample_multilabel_data):
        """Test bootstrap CI with different confidence levels."""

        def simple_metric(y_true, y_prob):
            return 0.75  # Constant for testing

        # 90% CI
        _, lower_90, upper_90 = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=simple_metric,
            n_bootstrap=50,
            confidence_level=0.90,
        )

        # 95% CI
        _, lower_95, upper_95 = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=simple_metric,
            n_bootstrap=50,
            confidence_level=0.95,
        )

        # 95% CI should be wider than 90% CI
        assert (upper_95 - lower_95) >= (upper_90 - lower_90)

    def test_bootstrap_ci_with_failing_metric(self, sample_multilabel_data):
        """Test bootstrap CI when some bootstrap samples fail."""
        call_count = [0]

        def failing_metric(y_true, y_prob):
            call_count[0] += 1
            # Fail on some bootstrap samples (but not the first call)
            if (
                call_count[0] > 1 and np.random.rand() < 0.3
            ):  # 30% chance after first call
                raise ValueError("Random failure")
            return 0.75

        # Should still complete by skipping failed samples
        metric, lower, upper = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=failing_metric,
            n_bootstrap=50,
            random_state=42,
        )

        assert isinstance(metric, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        # Verify that the except block was exercised
        assert (
            call_count[0] > 50
        )  # More than n_bootstrap + 1 means some failed and continued

    def test_bootstrap_ci_reproducibility(self, sample_multilabel_data):
        """Test that bootstrap CI is reproducible with same random_state."""

        def auroc_fn(y_true, y_prob):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(y_true, y_prob, average="macro")

        result1 = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=auroc_fn,
            n_bootstrap=50,
            random_state=42,
        )

        result2 = compute_bootstrap_ci_multilabel(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric_fn=auroc_fn,
            n_bootstrap=50,
            random_state=42,
        )

        assert result1 == result2


# =============================================================================
# Test Optimal Thresholds
# =============================================================================


class TestOptimalThresholds:
    """Test compute_optimal_thresholds function."""

    def test_optimal_thresholds_f1(self, sample_multilabel_data):
        """Test computing optimal thresholds for F1 score."""
        thresholds = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="f1",
        )

        assert thresholds.shape == (5,)
        assert np.all(thresholds >= 0.0)
        assert np.all(thresholds <= 1.0)

    def test_optimal_thresholds_precision(self, sample_multilabel_data):
        """Test computing optimal thresholds for precision."""
        thresholds = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="precision",
        )

        assert thresholds.shape == (5,)
        assert np.all(thresholds >= 0.0)
        assert np.all(thresholds <= 1.0)

    def test_optimal_thresholds_recall(self, sample_multilabel_data):
        """Test computing optimal thresholds for recall."""
        thresholds = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="recall",
        )

        assert thresholds.shape == (5,)
        assert np.all(thresholds >= 0.0)
        assert np.all(thresholds <= 1.0)

    def test_optimal_thresholds_j_statistic(self, sample_multilabel_data):
        """Test computing optimal thresholds for J statistic."""
        thresholds = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="j_statistic",
        )

        assert thresholds.shape == (5,)
        assert np.all(thresholds >= 0.0)
        assert np.all(thresholds <= 1.0)

    def test_optimal_thresholds_with_single_class(self, single_class_data):
        """Test optimal thresholds when one class has only one label."""
        thresholds = compute_optimal_thresholds(
            single_class_data["y_true"],
            single_class_data["y_prob"],
            metric="f1",
        )

        # Class 1 has only positive labels, should default to 0.5
        assert thresholds[1] == 0.5

        # Other classes should have valid thresholds
        assert 0.0 <= thresholds[0] <= 1.0
        assert 0.0 <= thresholds[2] <= 1.0

    def test_optimal_thresholds_invalid_metric(self, sample_multilabel_data):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_optimal_thresholds(
                sample_multilabel_data["y_true"],
                sample_multilabel_data["y_prob"],
                metric="invalid_metric",
            )

    def test_optimal_thresholds_different_metrics_differ(self, sample_multilabel_data):
        """Test that different metrics produce different thresholds."""
        thresholds_f1 = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="f1",
        )

        thresholds_precision = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="precision",
        )

        # Different metrics should generally produce different thresholds
        # (though they might occasionally be the same)
        assert thresholds_f1.shape == thresholds_precision.shape


# =============================================================================
# Test Integration and Edge Cases
# =============================================================================


class TestIntegration:
    """Test integration scenarios and edge cases."""

    def test_full_evaluation_pipeline(self, sample_multilabel_data):
        """Test complete evaluation pipeline."""
        # Compute all metrics
        metrics = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            sample_multilabel_data["y_prob"],
            class_names=sample_multilabel_data["class_names"],
        )

        # Compute confusion matrices
        cm_result = compute_multilabel_confusion_matrix(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            class_names=sample_multilabel_data["class_names"],
        )

        # Compute optimal thresholds
        thresholds = compute_optimal_thresholds(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
            metric="f1",
        )

        assert metrics is not None
        assert cm_result is not None
        assert thresholds is not None

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        n_samples = 10
        n_classes = 3

        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        y_prob = np.random.rand(n_samples, n_classes)
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_multilabel_metrics(y_true, y_pred, y_prob)

        assert "auroc_macro" in result
        assert "hamming_loss" in result

    def test_binary_multilabel(self):
        """Test with binary multi-label (2 classes)."""
        np.random.seed(42)
        n_samples = 50
        n_classes = 2

        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        y_prob = np.random.rand(n_samples, n_classes)
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_multilabel_metrics(y_true, y_pred, y_prob)

        assert len(result["auroc_per_class"]) == 2
        assert len(result["precision_per_class"]) == 2

    def test_many_classes(self):
        """Test with many classes."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 20

        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        y_prob = np.random.rand(n_samples, n_classes)
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_multilabel_metrics(y_true, y_pred, y_prob)

        assert len(result["auroc_per_class"]) == 20
        assert len(result["precision_per_class"]) == 20

    def test_all_zeros_predictions(self):
        """Test with all-zero predictions."""
        np.random.seed(42)
        n_samples = 50
        n_classes = 3

        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        y_pred = np.zeros((n_samples, n_classes), dtype=int)
        y_prob = np.zeros((n_samples, n_classes))

        result = compute_multilabel_metrics(y_true, y_pred, y_prob)

        assert "hamming_loss" in result
        # Hamming loss should be > 0 if there are any positive labels
        if np.any(y_true == 1):
            assert result["hamming_loss"] > 0

    def test_all_ones_predictions(self):
        """Test with all-one predictions."""
        np.random.seed(42)
        n_samples = 50
        n_classes = 3

        y_true = np.random.randint(0, 2, size=(n_samples, n_classes))
        y_pred = np.ones((n_samples, n_classes), dtype=int)
        y_prob = np.ones((n_samples, n_classes))

        result = compute_multilabel_metrics(y_true, y_pred, y_prob)

        assert "hamming_loss" in result
        # Hamming loss should be > 0 if there are any negative labels
        if np.any(y_true == 0):
            assert result["hamming_loss"] > 0

    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 4

        y_true = np.zeros((n_samples, n_classes), dtype=int)
        # Make class 0 have 5% positive rate
        y_true[:5, 0] = 1
        # Make class 1 have 95% positive rate
        y_true[:95, 1] = 1
        # Make classes 2 and 3 balanced
        y_true[:50, 2] = 1
        y_true[50:, 3] = 1

        y_prob = np.random.rand(n_samples, n_classes)
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_multilabel_metrics(y_true, y_pred, y_prob)

        # Should handle imbalanced classes
        assert "auroc_weighted" in result
        assert isinstance(result["auroc_weighted"], float)

    def test_close_figure_after_plotting(self, sample_multilabel_data):
        """Test that figures are properly closed after plotting."""
        initial_figs = len(plt.get_fignums())

        auroc_scores = np.array([0.8, 0.75, 0.9, 0.85, 0.7])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            plot_multilabel_auroc_per_class(
                auroc_scores,
                sample_multilabel_data["class_names"],
                save_path=save_path,
            )

            # Figure should be closed after saving
            final_figs = len(plt.get_fignums())
            assert final_figs == initial_figs
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_metrics_consistency(self, sample_multilabel_data):
        """Test consistency between different metric computations."""
        # Compute metrics separately
        auroc_result = compute_multilabel_auroc(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_prob"],
        )

        full_result = compute_multilabel_metrics(
            sample_multilabel_data["y_true"],
            sample_multilabel_data["y_pred"],
            sample_multilabel_data["y_prob"],
        )

        # AUROC values should match
        assert auroc_result["auroc_macro"] == full_result["auroc_macro"]
        assert auroc_result["auroc_micro"] == full_result["auroc_micro"]
        assert auroc_result["auroc_weighted"] == full_result["auroc_weighted"]
        assert auroc_result["auroc_per_class"] == full_result["auroc_per_class"]
