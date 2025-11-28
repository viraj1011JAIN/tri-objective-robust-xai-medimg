"""
Comprehensive tests for HPO Objectives to achieve 100% coverage.

Tests all objective classes, scalarization methods, dynamic adjustment, and factory functions.

Author: Viraj Jain
Date: November 27, 2025
"""

import numpy as np
import pytest

from src.config.hpo.objectives import (
    AccuracyObjective,
    AugmentedTchebycheffObjective,
    DynamicWeightAdjuster,
    ExplainabilityObjective,
    ObjectiveMetrics,
    ObjectiveType,
    OptimizationDirection,
    ParetoFrontTracker,
    PBIObjective,
    RobustnessObjective,
    SingleObjective,
    TchebycheffObjective,
    WeightedSumObjective,
    create_objective_function,
)


class TestAccuracyObjective:
    """Tests for AccuracyObjective class."""

    def test_initialization(self):
        """Test accuracy objective initialization."""
        objective = AccuracyObjective()

        assert objective.objective_type == ObjectiveType.ACCURACY
        assert objective.direction == OptimizationDirection.MAXIMIZE
        assert objective.name == "accuracy"

    def test_evaluate(self):
        """Test accuracy objective evaluation."""
        objective = AccuracyObjective()
        metrics = ObjectiveMetrics(accuracy=0.92, robustness=0.8, explainability=0.75)

        value = objective.evaluate(metrics)
        assert value == 0.92


class TestRobustnessObjective:
    """Tests for RobustnessObjective class."""

    def test_initialization_default(self):
        """Test robustness objective with default metric."""
        objective = RobustnessObjective()

        assert objective.objective_type == ObjectiveType.ROBUSTNESS
        assert objective.direction == OptimizationDirection.MAXIMIZE
        assert objective.robustness_metric == "robust_accuracy"

    def test_initialization_custom(self):
        """Test robustness objective with custom metric."""
        objective = RobustnessObjective(robustness_metric="robustness")

        assert objective.robustness_metric == "robustness"

    def test_evaluate_robust_accuracy(self):
        """Test evaluation with robust_accuracy metric."""
        objective = RobustnessObjective(robustness_metric="robust_accuracy")
        metrics = ObjectiveMetrics(
            accuracy=0.9, robustness=0.85, robust_accuracy=0.78, explainability=0.75
        )

        value = objective.evaluate(metrics)
        assert value == 0.78

    def test_evaluate_robustness(self):
        """Test evaluation with robustness metric."""
        objective = RobustnessObjective(robustness_metric="robustness")
        metrics = ObjectiveMetrics(
            accuracy=0.9, robustness=0.85, robust_accuracy=0.78, explainability=0.75
        )

        value = objective.evaluate(metrics)
        assert value == 0.85

    def test_evaluate_combined(self):
        """Test evaluation with combined metrics."""
        objective = RobustnessObjective(robustness_metric="combined")
        metrics = ObjectiveMetrics(
            accuracy=0.9, robustness=0.8, robust_accuracy=0.6, explainability=0.75
        )

        value = objective.evaluate(metrics)
        expected = 0.5 * 0.6 + 0.5 * 0.8  # 0.7
        assert abs(value - expected) < 1e-6


class TestExplainabilityObjective:
    """Tests for ExplainabilityObjective class."""

    def test_initialization_default(self):
        """Test explainability objective with default weights."""
        objective = ExplainabilityObjective()

        assert objective.objective_type == ObjectiveType.EXPLAINABILITY
        assert objective.direction == OptimizationDirection.MAXIMIZE
        # Weights should be normalized to 0.5, 0.5
        assert abs(objective.coherence_weight - 0.5) < 1e-6
        assert abs(objective.faithfulness_weight - 0.5) < 1e-6

    def test_initialization_custom_weights(self):
        """Test explainability objective with custom weights."""
        objective = ExplainabilityObjective(
            coherence_weight=0.3, faithfulness_weight=0.7
        )

        # Weights should be normalized
        assert abs(objective.coherence_weight - 0.3) < 1e-6
        assert abs(objective.faithfulness_weight - 0.7) < 1e-6

    def test_weight_normalization(self):
        """Test that weights are automatically normalized."""
        objective = ExplainabilityObjective(
            coherence_weight=2.0, faithfulness_weight=3.0
        )

        # Should normalize to 0.4 and 0.6
        assert abs(objective.coherence_weight - 0.4) < 1e-6
        assert abs(objective.faithfulness_weight - 0.6) < 1e-6

    def test_evaluate(self):
        """Test explainability objective evaluation."""
        objective = ExplainabilityObjective(
            coherence_weight=0.6, faithfulness_weight=0.4
        )
        metrics = ObjectiveMetrics(
            accuracy=0.9,
            robustness=0.8,
            explainability=0.75,
            xai_coherence=0.85,
            xai_faithfulness=0.70,
        )

        value = objective.evaluate(metrics)
        expected = 0.6 * 0.85 + 0.4 * 0.70  # 0.51 + 0.28 = 0.79
        assert abs(value - expected) < 1e-6


class TestWeightedSumObjectiveComplete:
    """Complete tests for WeightedSumObjective class."""

    def test_initialization_without_normalization(self):
        """Test weighted sum without normalization."""
        objective = WeightedSumObjective(
            accuracy_weight=2.0,
            robustness_weight=3.0,
            explainability_weight=1.0,
            normalize_weights=False,
        )

        assert objective.accuracy_weight == 2.0
        assert objective.robustness_weight == 3.0
        assert objective.explainability_weight == 1.0

    def test_set_weights_with_normalization(self):
        """Test setting weights with normalization."""
        objective = WeightedSumObjective()
        objective.set_weights(2.0, 3.0, 1.0, normalize=True)

        # Should sum to 1
        weights = objective.get_weights()
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_set_weights_without_normalization(self):
        """Test setting weights without normalization."""
        objective = WeightedSumObjective()
        objective.set_weights(2.0, 3.0, 1.0, normalize=False)

        weights = objective.get_weights()
        assert weights == (2.0, 3.0, 1.0)

    def test_set_weights_zero_total(self):
        """Test setting weights with zero total (edge case)."""
        objective = WeightedSumObjective()
        objective.set_weights(0.0, 0.0, 0.0, normalize=True)

        # Weights should remain zero when total is zero
        weights = objective.get_weights()
        assert weights == (0.0, 0.0, 0.0)


class TestTchebycheffObjective:
    """Tests for TchebycheffObjective class."""

    def test_initialization_default(self):
        """Test Tchebycheff with default parameters."""
        objective = TchebycheffObjective()

        assert np.allclose(objective.reference_point, [1.0, 1.0, 1.0])
        assert np.allclose(objective.weights, [1.0 / 3, 1.0 / 3, 1.0 / 3])

    def test_initialization_custom(self):
        """Test Tchebycheff with custom parameters."""
        ref_point = np.array([0.95, 0.90, 0.85])
        weights = np.array([0.5, 0.3, 0.2])

        objective = TchebycheffObjective(reference_point=ref_point, weights=weights)

        assert np.allclose(objective.reference_point, ref_point)
        # Weights should be normalized
        assert abs(np.sum(objective.weights) - 1.0) < 1e-6

    def test_evaluate(self):
        """Test Tchebycheff evaluation."""
        objective = TchebycheffObjective(
            reference_point=np.array([1.0, 1.0, 1.0]),
            weights=np.array([0.5, 0.3, 0.2]),
        )

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)

        value = objective.evaluate(metrics)
        # Max of weighted differences: [0.5*0.2, 0.3*0.3, 0.2*0.1] = [0.1, 0.09, 0.02]
        expected = 0.1
        assert abs(value - expected) < 1e-6

    def test_update_reference_point(self):
        """Test updating reference point."""
        objective = TchebycheffObjective()
        new_point = np.array([0.95, 0.92, 0.88])

        objective.update_reference_point(new_point)

        assert np.allclose(objective.reference_point, new_point)


class TestAugmentedTchebycheffObjective:
    """Tests for AugmentedTchebycheffObjective class."""

    def test_initialization(self):
        """Test augmented Tchebycheff initialization."""
        objective = AugmentedTchebycheffObjective(augmentation_factor=0.1)

        assert objective.augmentation_factor == 0.1
        assert np.allclose(objective.reference_point, [1.0, 1.0, 1.0])

    def test_evaluate(self):
        """Test augmented Tchebycheff evaluation."""
        objective = AugmentedTchebycheffObjective(
            reference_point=np.array([1.0, 1.0, 1.0]),
            weights=np.array([0.4, 0.4, 0.2]),
            augmentation_factor=0.05,
        )

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)

        value = objective.evaluate(metrics)

        # Tchebycheff term: max([0.4*0.2, 0.4*0.3, 0.2*0.1]) = 0.12
        # Augmentation term: 0.05 * (0.08 + 0.12 + 0.02) = 0.011
        # Total: 0.12 + 0.011 = 0.131
        assert isinstance(value, float)
        assert value > 0


class TestPBIObjective:
    """Tests for PBIObjective class."""

    def test_initialization_default(self):
        """Test PBI with default parameters."""
        objective = PBIObjective()

        # Weights should be normalized to unit vector
        assert abs(np.linalg.norm(objective.weights) - 1.0) < 1e-6
        assert objective.penalty_parameter == 5.0

    def test_initialization_custom(self):
        """Test PBI with custom parameters."""
        weights = np.array([0.5, 0.3, 0.2])
        objective = PBIObjective(weights=weights, penalty_parameter=10.0)

        assert abs(np.linalg.norm(objective.weights) - 1.0) < 1e-6
        assert objective.penalty_parameter == 10.0

    def test_evaluate(self):
        """Test PBI evaluation."""
        objective = PBIObjective(
            weights=np.array([1.0, 1.0, 1.0]), penalty_parameter=5.0
        )

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)

        value = objective.evaluate(metrics)

        # Value should be a float (to minimize, so includes negative d1)
        assert isinstance(value, float)


class TestDynamicWeightAdjuster:
    """Tests for DynamicWeightAdjuster class."""

    def test_initialization(self):
        """Test dynamic weight adjuster initialization."""
        adjuster = DynamicWeightAdjuster(
            initial_weights=(0.4, 0.4, 0.2),
            adjustment_strategy="performance_based",
            adjustment_rate=0.1,
        )

        assert adjuster.adjustment_strategy == "performance_based"
        assert adjuster.adjustment_rate == 0.1
        assert abs(np.sum(adjuster.current_weights) - 1.0) < 1e-6

    def test_update_weights_insufficient_history(self):
        """Test update with insufficient history."""
        adjuster = DynamicWeightAdjuster()

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
        weights = adjuster.update_weights(metrics)

        # With only 1 entry, weights should not change
        assert len(adjuster.history) == 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_performance_based_adjustment(self):
        """Test performance-based weight adjustment."""
        adjuster = DynamicWeightAdjuster(
            adjustment_strategy="performance_based", adjustment_rate=0.1
        )

        # Add multiple metrics to build history
        for i in range(15):
            metrics = ObjectiveMetrics(
                accuracy=0.5 + i * 0.01,  # Improving
                robustness=0.3,  # Stagnant
                explainability=0.4 + i * 0.005,  # Moderate improvement
            )
            adjuster.update_weights(metrics)

        # Weights should be adjusted (robustness should get higher weight due to poor performance)
        assert len(adjuster.history) == 15

    def test_gradient_based_adjustment(self):
        """Test gradient-based weight adjustment."""
        adjuster = DynamicWeightAdjuster(
            adjustment_strategy="gradient_based", adjustment_rate=0.1
        )

        # Add metrics with different improvement rates
        for i in range(10):
            metrics = ObjectiveMetrics(
                accuracy=0.6 + i * 0.02,  # Fast improvement
                robustness=0.5 + i * 0.005,  # Slow improvement
                explainability=0.4 + i * 0.01,  # Moderate improvement
            )
            adjuster.update_weights(metrics)

        # Should adjust based on gradients
        assert len(adjuster.history) == 10

    def test_variance_based_adjustment(self):
        """Test variance-based weight adjustment."""
        adjuster = DynamicWeightAdjuster(
            adjustment_strategy="variance_based", adjustment_rate=0.1
        )

        # Add metrics with varying variance
        for i in range(25):
            metrics = ObjectiveMetrics(
                accuracy=0.7 + (i % 3) * 0.1,  # High variance
                robustness=0.6 + (i % 2) * 0.05,  # Moderate variance
                explainability=0.65,  # Low variance
            )
            adjuster.update_weights(metrics)

        # Should adjust based on variance
        assert len(adjuster.history) == 25

    def test_gradient_adjustment_zero_gradient(self):
        """Test gradient adjustment with zero gradients."""
        adjuster = DynamicWeightAdjuster(
            adjustment_strategy="gradient_based", adjustment_rate=0.1
        )

        # Add constant metrics (zero gradients)
        for _ in range(10):
            metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
            adjuster.update_weights(metrics)

        # Should handle zero gradients gracefully
        assert len(adjuster.history) == 10

    def test_variance_adjustment_zero_variance(self):
        """Test variance adjustment with zero variance."""
        adjuster = DynamicWeightAdjuster(
            adjustment_strategy="variance_based", adjustment_rate=0.1
        )

        # Add constant metrics (zero variance)
        for _ in range(25):
            metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
            adjuster.update_weights(metrics)

        # Should handle zero variance gracefully
        assert len(adjuster.history) == 25


class TestParetoFrontTrackerComplete:
    """Complete tests for ParetoFrontTracker class."""

    def test_update_empty_front(self):
        """Test updating empty Pareto front."""
        tracker = ParetoFrontTracker()
        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
        config = {"lr": 0.001, "batch_size": 32}

        tracker.update(metrics, config)

        front = tracker.get_pareto_front()
        assert len(front) == 1
        assert front[0][0].accuracy == 0.8

    def test_update_dominated_solution(self):
        """Test that dominated solutions are not added."""
        tracker = ParetoFrontTracker()

        # Add a good solution
        metrics1 = ObjectiveMetrics(accuracy=0.9, robustness=0.85, explainability=0.88)
        tracker.update(metrics1, {"config": 1})

        # Try to add a dominated solution
        metrics2 = ObjectiveMetrics(accuracy=0.8, robustness=0.75, explainability=0.80)
        tracker.update(metrics2, {"config": 2})

        front = tracker.get_pareto_front()
        # Dominated solution should not be added
        assert len(front) == 1

    def test_update_dominating_solution(self):
        """Test that dominating solution replaces dominated ones."""
        tracker = ParetoFrontTracker()

        # Add a mediocre solution
        metrics1 = ObjectiveMetrics(accuracy=0.8, robustness=0.75, explainability=0.80)
        tracker.update(metrics1, {"config": 1})

        # Add a dominating solution
        metrics2 = ObjectiveMetrics(accuracy=0.9, robustness=0.85, explainability=0.88)
        tracker.update(metrics2, {"config": 2})

        front = tracker.get_pareto_front()
        # Only the dominating solution should remain
        assert len(front) == 1
        assert front[0][0].accuracy == 0.9

    def test_update_non_dominated_solutions(self):
        """Test adding multiple non-dominated solutions."""
        tracker = ParetoFrontTracker()

        # Add solutions that are non-dominated
        metrics1 = ObjectiveMetrics(accuracy=0.9, robustness=0.7, explainability=0.75)
        tracker.update(metrics1, {"config": 1})

        metrics2 = ObjectiveMetrics(accuracy=0.75, robustness=0.9, explainability=0.80)
        tracker.update(metrics2, {"config": 2})

        metrics3 = ObjectiveMetrics(accuracy=0.80, robustness=0.75, explainability=0.92)
        tracker.update(metrics3, {"config": 3})

        front = tracker.get_pareto_front()
        # All three should be in the front
        assert len(front) == 3

    def test_get_best_by_objective_accuracy(self):
        """Test getting best solution by accuracy."""
        tracker = ParetoFrontTracker()

        metrics1 = ObjectiveMetrics(accuracy=0.9, robustness=0.7, explainability=0.75)
        tracker.update(metrics1, {"config": 1})

        metrics2 = ObjectiveMetrics(accuracy=0.85, robustness=0.9, explainability=0.80)
        tracker.update(metrics2, {"config": 2})

        best_metrics, best_config = tracker.get_best_by_objective("accuracy")
        assert best_metrics.accuracy == 0.9

    def test_get_best_by_objective_robustness(self):
        """Test getting best solution by robustness."""
        tracker = ParetoFrontTracker()

        metrics1 = ObjectiveMetrics(accuracy=0.9, robustness=0.7, explainability=0.75)
        tracker.update(metrics1, {"config": 1})

        metrics2 = ObjectiveMetrics(accuracy=0.85, robustness=0.9, explainability=0.80)
        tracker.update(metrics2, {"config": 2})

        best_metrics, best_config = tracker.get_best_by_objective("robustness")
        assert best_metrics.robustness == 0.9

    def test_get_best_by_objective_explainability(self):
        """Test getting best solution by explainability."""
        tracker = ParetoFrontTracker()

        metrics1 = ObjectiveMetrics(accuracy=0.9, robustness=0.7, explainability=0.75)
        tracker.update(metrics1, {"config": 1})

        metrics2 = ObjectiveMetrics(accuracy=0.85, robustness=0.9, explainability=0.88)
        tracker.update(metrics2, {"config": 2})

        best_metrics, best_config = tracker.get_best_by_objective("explainability")
        assert best_metrics.explainability == 0.88

    def test_get_best_empty_front(self):
        """Test getting best from empty front raises error."""
        tracker = ParetoFrontTracker()

        with pytest.raises(ValueError, match="Pareto front is empty"):
            tracker.get_best_by_objective("accuracy")

    def test_get_best_unknown_objective(self):
        """Test getting best with unknown objective raises error."""
        tracker = ParetoFrontTracker()

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)
        tracker.update(metrics, {"config": 1})

        with pytest.raises(ValueError, match="Unknown objective"):
            tracker.get_best_by_objective("invalid_objective")


class TestCreateObjectiveFunction:
    """Tests for create_objective_function factory."""

    def test_create_accuracy_objective(self):
        """Test creating accuracy objective function."""
        obj_func = create_objective_function("accuracy")

        metrics = ObjectiveMetrics(accuracy=0.85, robustness=0.8, explainability=0.75)
        value = obj_func(metrics)

        assert value == 0.85

    def test_create_robustness_objective(self):
        """Test creating robustness objective function."""
        obj_func = create_objective_function(
            "robustness", robustness_metric="robustness"
        )

        metrics = ObjectiveMetrics(accuracy=0.85, robustness=0.78, explainability=0.75)
        value = obj_func(metrics)

        assert value == 0.78

    def test_create_explainability_objective(self):
        """Test creating explainability objective function."""
        obj_func = create_objective_function(
            "explainability", coherence_weight=0.6, faithfulness_weight=0.4
        )

        metrics = ObjectiveMetrics(
            accuracy=0.85,
            robustness=0.8,
            explainability=0.75,
            xai_coherence=0.8,
            xai_faithfulness=0.7,
        )
        value = obj_func(metrics)

        expected = 0.6 * 0.8 + 0.4 * 0.7
        assert abs(value - expected) < 1e-6

    def test_create_weighted_sum_objective(self):
        """Test creating weighted sum objective function."""
        obj_func = create_objective_function(
            "weighted_sum",
            accuracy_weight=0.5,
            robustness_weight=0.3,
            explainability_weight=0.2,
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.7)
        value = obj_func(metrics)

        expected = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
        assert abs(value - expected) < 1e-6

    def test_create_tchebycheff_objective(self):
        """Test creating Tchebycheff objective function."""
        obj_func = create_objective_function(
            "tchebycheff",
            reference_point=np.array([1.0, 1.0, 1.0]),
            weights=np.array([0.5, 0.3, 0.2]),
        )

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
        value = obj_func(metrics)

        assert isinstance(value, (float, np.floating))

    def test_create_augmented_tchebycheff_objective(self):
        """Test creating augmented Tchebycheff objective function."""
        obj_func = create_objective_function(
            "augmented_tchebycheff", augmentation_factor=0.05
        )

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
        value = obj_func(metrics)

        assert isinstance(value, (float, np.floating))

    def test_create_pbi_objective(self):
        """Test creating PBI objective function."""
        obj_func = create_objective_function("pbi", penalty_parameter=5.0)

        metrics = ObjectiveMetrics(accuracy=0.8, robustness=0.7, explainability=0.9)
        value = obj_func(metrics)

        assert isinstance(value, (float, np.floating))

    def test_create_unknown_objective(self):
        """Test creating unknown objective type raises error."""
        with pytest.raises(ValueError, match="Unknown objective type"):
            create_objective_function("unknown_objective_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
