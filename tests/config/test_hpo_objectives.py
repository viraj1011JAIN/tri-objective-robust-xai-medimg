"""
Comprehensive tests for HPO optimization objectives.

Tests cover:
- Objective metrics
- Single objective configuration
- Weighted sum objective computation
- Pareto front tracking
- Objective types and directions

Author: Viraj Pankaj Jain
"""

import pytest

from src.config.hpo.objectives import (
    ObjectiveMetrics,
    ObjectiveType,
    OptimizationDirection,
    ParetoFrontTracker,
    SingleObjective,
    WeightedSumObjective,
)


class TestObjectiveMetrics:
    """Tests for ObjectiveMetrics dataclass."""

    def test_default_creation(self):
        """Test creating metrics with defaults."""
        metrics = ObjectiveMetrics()

        assert metrics.accuracy == 0.0
        assert metrics.robustness == 0.0
        assert metrics.explainability == 0.0

    def test_custom_values(self):
        """Test creating metrics with custom values."""
        metrics = ObjectiveMetrics(
            accuracy=0.85,
            robustness=0.75,
            explainability=0.80,
        )

        assert metrics.accuracy == 0.85
        assert metrics.robustness == 0.75
        assert metrics.explainability == 0.80

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ObjectiveMetrics(
            accuracy=0.85,
            robustness=0.75,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["accuracy"] == 0.85
        assert metrics_dict["robustness"] == 0.75


class TestSingleObjective:
    """Tests for SingleObjective class."""

    def test_single_objective_creation(self):
        """Test creating single objective."""
        objective = SingleObjective(
            objective_type=ObjectiveType.ACCURACY,
            direction=OptimizationDirection.MAXIMIZE,
        )

        assert objective.objective_type == ObjectiveType.ACCURACY
        assert objective.direction == OptimizationDirection.MAXIMIZE


class TestWeightedSumObjective:
    """Tests for WeightedSumObjective class."""

    def test_weighted_objective_creation(self):
        """Test creating weighted sum objective."""
        objective = WeightedSumObjective(
            accuracy_weight=0.5,
            robustness_weight=0.3,
            explainability_weight=0.2,
        )

        assert objective.accuracy_weight == 0.5
        assert objective.robustness_weight == 0.3
        assert objective.explainability_weight == 0.2

    def test_weighted_objective_evaluation(self):
        """Test evaluating weighted objective."""
        objective = WeightedSumObjective(
            accuracy_weight=0.5,
            robustness_weight=0.3,
            explainability_weight=0.2,
        )

        metrics = ObjectiveMetrics(
            accuracy=0.9,
            robustness=0.8,
            explainability=0.7,
        )

        score = objective.evaluate(metrics)

        assert isinstance(score, float)
        expected = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
        assert abs(score - expected) < 1e-6

    def test_weight_normalization(self):
        """Test automatic weight normalization."""
        objective = WeightedSumObjective(
            accuracy_weight=2.0,
            robustness_weight=1.0,
            explainability_weight=1.0,
            normalize_weights=True,
        )

        # Weights should sum to 1
        weights = objective.get_weights()
        total = sum(weights)
        assert abs(total - 1.0) < 1e-6


class TestParetoFrontTracker:
    """Tests for ParetoFrontTracker class."""

    def test_pareto_tracker_creation(self):
        """Test creating Pareto front tracker."""
        tracker = ParetoFrontTracker()

        assert isinstance(tracker, ParetoFrontTracker)


class TestSingleObjectiveExtended:
    """Extended tests for SingleObjective."""

    def test_evaluate_accuracy(self):
        """Test evaluate method for accuracy objective."""
        objective = SingleObjective(
            objective_type=ObjectiveType.ACCURACY,
            direction=OptimizationDirection.MAXIMIZE,
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.7)

        value = objective.evaluate(metrics)
        assert value == 0.9

    def test_evaluate_robustness(self):
        """Test evaluate method for robustness objective."""
        objective = SingleObjective(
            objective_type=ObjectiveType.ROBUSTNESS,
            direction=OptimizationDirection.MAXIMIZE,
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.7)

        value = objective.evaluate(metrics)
        assert value == 0.8

    def test_evaluate_explainability(self):
        """Test evaluate method for explainability objective."""
        objective = SingleObjective(
            objective_type=ObjectiveType.EXPLAINABILITY,
            direction=OptimizationDirection.MAXIMIZE,
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.7)

        value = objective.evaluate(metrics)
        assert value == 0.7

    def test_evaluate_invalid_type(self):
        """Test evaluate with invalid objective type."""
        # Create objective with mock invalid type
        objective = SingleObjective(
            objective_type=ObjectiveType.ACCURACY,
            direction=OptimizationDirection.MAXIMIZE,
        )
        # Manually set invalid type for testing
        objective.objective_type = "invalid"

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.7)

        with pytest.raises(ValueError, match="Invalid objective type"):
            objective.evaluate(metrics)

    def test_is_better_maximize(self):
        """Test is_better for maximization."""
        objective = SingleObjective(
            objective_type=ObjectiveType.ACCURACY,
            direction=OptimizationDirection.MAXIMIZE,
        )

        assert objective.is_better(0.9, 0.8)
        assert not objective.is_better(0.7, 0.8)

    def test_is_better_minimize(self):
        """Test is_better for minimization."""
        objective = SingleObjective(
            objective_type=ObjectiveType.ACCURACY,
            direction=OptimizationDirection.MINIMIZE,
        )

        assert objective.is_better(0.7, 0.8)
        assert not objective.is_better(0.9, 0.8)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
