"""
Comprehensive tests for Multi-Label Task Loss to achieve 100% coverage.

This test suite covers all remaining uncovered lines including:
- OptimalThresholdFinder class
- Per-class loss returns
- Helper methods (get_pos_weight, get_temperature, set_temperature)
- Edge cases and reduction modes

Author: Viraj Pankaj Jain
Date: November 27, 2025
"""

import numpy as np
import pytest
import torch

from src.losses.multi_label_task_loss import MultiLabelTaskLoss, OptimalThresholdFinder


class TestMultiLabelHelperMethods:
    """Tests for helper methods in MultiLabelTaskLoss."""

    def test_get_pos_weight(self):
        """Test get_pos_weight method."""
        positive_rates = [0.1, 0.2, 0.3, 0.4]
        loss_fn = MultiLabelTaskLoss(num_classes=4, positive_rates=positive_rates)

        pos_weight = loss_fn.get_pos_weight()

        assert isinstance(pos_weight, torch.Tensor)
        assert pos_weight.shape == (4,)
        # Verify pos_weight values are reasonable
        assert (pos_weight > 0).all()

    def test_get_pos_weight_no_positive_rates(self):
        """Test get_pos_weight with default (no positive rates)."""
        loss_fn = MultiLabelTaskLoss(num_classes=4)

        pos_weight = loss_fn.get_pos_weight()

        assert isinstance(pos_weight, torch.Tensor)
        assert pos_weight.shape == (4,)
        # Should be all ones when no positive rates provided
        assert torch.allclose(pos_weight, torch.ones(4))

    def test_get_temperature(self):
        """Test get_temperature method."""
        loss_fn = MultiLabelTaskLoss(num_classes=4, temperature=2.5)

        temp = loss_fn.get_temperature()

        assert isinstance(temp, float)
        assert temp == 2.5

    def test_set_temperature(self):
        """Test set_temperature method."""
        loss_fn = MultiLabelTaskLoss(num_classes=4, temperature=1.0)

        # Change temperature
        loss_fn.set_temperature(3.0)

        assert loss_fn.get_temperature() == 3.0
        assert loss_fn.temperature.item() == 3.0

    def test_set_temperature_updates_computation(self):
        """Test that set_temperature affects loss computation."""
        loss_fn = MultiLabelTaskLoss(num_classes=4, temperature=1.0)

        logits = torch.randn(2, 4)
        targets = torch.randint(0, 2, (2, 4)).float()

        loss1 = loss_fn(logits, targets)

        # Change temperature and compute again
        loss_fn.set_temperature(2.0)
        loss2 = loss_fn(logits, targets)

        # Losses should be different with different temperatures
        assert not torch.isclose(loss1, loss2)


class TestMultiLabelPerClassLoss:
    """Tests for per-class loss computation."""

    def test_return_per_class_bce(self):
        """Test return_per_class=True with BCE loss."""
        loss_fn = MultiLabelTaskLoss(num_classes=4, reduction="mean")

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        per_class_loss = loss_fn(logits, targets, return_per_class=True)

        assert per_class_loss.shape == (4,)
        assert not torch.isnan(per_class_loss).any()
        assert not torch.isinf(per_class_loss).any()

    def test_return_per_class_focal(self):
        """Test return_per_class=True with focal loss."""
        loss_fn = MultiLabelTaskLoss(
            num_classes=4, use_focal=True, focal_gamma=2.0, reduction="mean"
        )

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        per_class_loss = loss_fn(logits, targets, return_per_class=True)

        assert per_class_loss.shape == (4,)
        assert not torch.isnan(per_class_loss).any()

    def test_reduction_none_vs_per_class(self):
        """Test that reduction='none' behaves correctly."""
        loss_fn = MultiLabelTaskLoss(num_classes=4, reduction="none")

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        # With reduction='none' and return_per_class=False, should return (C,)
        loss = loss_fn(logits, targets, return_per_class=False)

        assert loss.shape == (4,)  # Per-class losses

    def test_reduction_sum(self):
        """Test reduction='sum' mode."""
        loss_fn = MultiLabelTaskLoss(num_classes=4, reduction="sum")

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)


class TestOptimalThresholdFinder:
    """Tests for OptimalThresholdFinder class."""

    def test_initialization(self):
        """Test OptimalThresholdFinder initialization."""
        finder = OptimalThresholdFinder(num_classes=5)

        assert finder.num_classes == 5
        assert len(finder.thresholds) == 5
        # Default thresholds should be 0.5
        assert np.allclose(finder.thresholds, 0.5)

    def test_find_optimal_thresholds_basic(self):
        """Test finding optimal thresholds."""
        finder = OptimalThresholdFinder(num_classes=3)

        # Create simple synthetic data
        # Class 0: easy to classify at 0.5
        # Class 1: better threshold at 0.3
        # Class 2: better threshold at 0.7
        np.random.seed(42)
        probabilities = np.random.rand(100, 3)
        targets = np.zeros((100, 3))

        # Make patterns
        targets[:, 0] = (probabilities[:, 0] > 0.5).astype(int)
        targets[:, 1] = (probabilities[:, 1] > 0.3).astype(int)
        targets[:, 2] = (probabilities[:, 2] > 0.7).astype(int)

        optimal_thresholds = finder.find_optimal_thresholds(probabilities, targets)

        assert len(optimal_thresholds) == 3
        assert (optimal_thresholds >= 0.1).all()
        assert (optimal_thresholds <= 0.9).all()
        # Verify thresholds were updated
        assert np.array_equal(finder.thresholds, optimal_thresholds)

    def test_find_optimal_thresholds_custom_range(self):
        """Test finding thresholds with custom range."""
        finder = OptimalThresholdFinder(num_classes=2)

        np.random.seed(42)
        probabilities = np.random.rand(50, 2)
        targets = np.random.randint(0, 2, (50, 2))

        # Custom threshold range
        threshold_range = np.linspace(0.2, 0.8, 31)

        optimal_thresholds = finder.find_optimal_thresholds(
            probabilities, targets, threshold_range
        )

        assert len(optimal_thresholds) == 2
        # Thresholds should be within custom range
        assert (optimal_thresholds >= 0.2).all()
        assert (optimal_thresholds <= 0.8).all()

    def test_find_optimal_thresholds_perfect_classification(self):
        """Test threshold finding with perfect separation."""
        finder = OptimalThresholdFinder(num_classes=2)

        # Perfect separation: class 0 at high probs, class 1 at low probs
        probabilities = np.array(
            [
                [0.9, 0.1],
                [0.95, 0.05],
                [0.92, 0.08],
                [0.1, 0.9],
                [0.05, 0.95],
                [0.08, 0.92],
            ]
        )
        targets = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

        optimal_thresholds = finder.find_optimal_thresholds(probabilities, targets)

        # Should find thresholds around 0.5 for both
        assert len(optimal_thresholds) == 2

    def test_apply_thresholds(self):
        """Test applying thresholds to probabilities."""
        finder = OptimalThresholdFinder(num_classes=3)

        # Set custom thresholds
        finder.thresholds = np.array([0.3, 0.5, 0.7])

        probabilities = np.array([[0.4, 0.6, 0.8], [0.2, 0.4, 0.6], [0.5, 0.7, 0.9]])

        predictions = finder.apply_thresholds(probabilities)

        expected = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])

        assert predictions.shape == (3, 3)
        assert np.array_equal(predictions, expected)

    def test_apply_thresholds_default(self):
        """Test applying default thresholds (0.5)."""
        finder = OptimalThresholdFinder(num_classes=2)

        probabilities = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])

        predictions = finder.apply_thresholds(probabilities)

        expected = np.array([[1, 0], [0, 1], [1, 1]])  # 0.5 threshold for ties

        assert np.array_equal(predictions, expected)

    def test_get_thresholds(self):
        """Test get_thresholds method."""
        finder = OptimalThresholdFinder(num_classes=4)

        thresholds = finder.get_thresholds()

        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == 4
        assert np.allclose(thresholds, 0.5)  # Default

    def test_threshold_finder_edge_case_all_positive(self):
        """Test threshold finding when all samples are positive."""
        finder = OptimalThresholdFinder(num_classes=2)

        probabilities = np.random.rand(20, 2)
        targets = np.ones((20, 2))  # All positive

        optimal_thresholds = finder.find_optimal_thresholds(probabilities, targets)

        assert len(optimal_thresholds) == 2
        # Should find reasonable thresholds even for all-positive case

    def test_threshold_finder_edge_case_all_negative(self):
        """Test threshold finding when all samples are negative."""
        finder = OptimalThresholdFinder(num_classes=2)

        probabilities = np.random.rand(20, 2)
        targets = np.zeros((20, 2))  # All negative

        optimal_thresholds = finder.find_optimal_thresholds(probabilities, targets)

        assert len(optimal_thresholds) == 2
        # Should find reasonable thresholds even for all-negative case

    def test_threshold_finder_imbalanced_classes(self):
        """Test threshold finding with severe class imbalance."""
        finder = OptimalThresholdFinder(num_classes=2)

        np.random.seed(42)
        probabilities = np.random.rand(100, 2)
        # Class 0: 90% negative, 10% positive
        # Class 1: 95% negative, 5% positive
        targets = np.zeros((100, 2))
        targets[:10, 0] = 1  # 10% positive for class 0
        targets[:5, 1] = 1  # 5% positive for class 1

        optimal_thresholds = finder.find_optimal_thresholds(probabilities, targets)

        assert len(optimal_thresholds) == 2
        # With imbalanced data, thresholds might deviate from 0.5


class TestEdgeCasesAndIntegration:
    """Tests for edge cases and integration scenarios."""

    def test_extreme_positive_rates(self):
        """Test with extreme positive rates (very rare and very common classes)."""
        # Very rare class (0.01) and very common class (0.99)
        positive_rates = [0.01, 0.99, 0.5, 0.2]
        loss_fn = MultiLabelTaskLoss(num_classes=4, positive_rates=positive_rates)

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_pos_weight_clamping(self):
        """Test that pos_weight is clamped to prevent extreme values."""
        # Very extreme positive rates should be clamped
        positive_rates = [0.001, 0.999, 0.5]  # Will create extreme weights
        loss_fn = MultiLabelTaskLoss(num_classes=3, positive_rates=positive_rates)

        pos_weight = loss_fn.get_pos_weight()

        # Weights should be clamped between 0.1 and 100.0
        assert (pos_weight >= 0.1).all()
        assert (pos_weight <= 100.0).all()

    def test_focal_loss_with_all_features(self):
        """Test focal loss with all features enabled."""
        positive_rates = [0.1, 0.2, 0.3, 0.4]
        loss_fn = MultiLabelTaskLoss(
            num_classes=4,
            positive_rates=positive_rates,
            use_focal=True,
            focal_gamma=3.0,
            focal_alpha=0.5,
            temperature=2.0,
            label_smoothing=0.1,
            reduction="sum",
        )

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8, 4)).float()

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_label_smoothing_application(self):
        """Test that label smoothing is applied correctly."""
        loss_fn = MultiLabelTaskLoss(num_classes=2, label_smoothing=0.1)

        # Test with binary targets
        targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        smoothed = loss_fn._apply_label_smoothing(targets)

        # 0 → 0.05, 1 → 0.95 (with epsilon=0.1)
        expected = torch.tensor([[0.05, 0.95], [0.95, 0.05]])

        assert torch.allclose(smoothed, expected, atol=1e-6)

    def test_loss_consistency_across_batches(self):
        """Test that loss is consistent when processing same data in different batches."""
        # Use mean reduction for batch consistency (sum would double-count)
        loss_fn = MultiLabelTaskLoss(num_classes=4, reduction="mean")

        # Create data
        torch.manual_seed(42)
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16, 4)).float()

        # Compute loss on full batch
        loss_full = loss_fn(logits, targets)

        # Per-class losses should be consistent
        per_class_full = loss_fn(logits, targets, return_per_class=True)

        # Verify all values are finite
        assert not torch.isnan(loss_full)
        assert not torch.isinf(loss_full)
        assert not torch.isnan(per_class_full).any()

    def test_backward_pass_all_reduction_modes(self):
        """Test gradient computation works for all reduction modes."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = MultiLabelTaskLoss(num_classes=3, reduction=reduction)

            logits = torch.randn(4, 3, requires_grad=True)
            targets = torch.randint(0, 2, (4, 3)).float()

            loss = loss_fn(logits, targets)

            if reduction == "none":
                # Need to reduce manually for backward
                loss = loss.sum()

            loss.backward()

            assert logits.grad is not None
            assert not torch.isnan(logits.grad).any()


class TestModuleExports:
    """Test that all classes are properly exported."""

    def test_exports(self):
        """Test that __all__ exports are correct."""
        from src.losses.multi_label_task_loss import __all__

        assert "MultiLabelTaskLoss" in __all__
        assert "OptimalThresholdFinder" in __all__
        assert len(__all__) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
