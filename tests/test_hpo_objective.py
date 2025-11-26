"""
Comprehensive tests for TRADES HPO objective functions.

Tests cover:
- MetricType enum
- TrialMetrics dataclass
- ObjectiveConfig validation
- ObjectiveFunction abstract base class
- WeightedTriObjective implementation
- AdaptiveWeightedObjective dynamic weighting
- MultiObjectiveEvaluator Pareto analysis

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 24, 2025
Version: 5.4.0
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.hpo_objective import (
    AdaptiveWeightedObjective,
    MetricType,
    MultiObjectiveEvaluator,
    ObjectiveConfig,
    ObjectiveFunction,
    TrialMetrics,
    WeightedTriObjective,
)


class TestMetricType:
    """Test MetricType enum."""

    def test_enum_values(self):
        """Test all metric type enum values."""
        assert MetricType.ROBUST_ACCURACY.value == "robust_accuracy"
        assert MetricType.CLEAN_ACCURACY.value == "clean_accuracy"
        assert MetricType.CROSS_SITE_AUROC.value == "cross_site_auroc"
        assert MetricType.LOSS.value == "loss"
        assert MetricType.TRADES_LOSS.value == "trades_loss"
        assert MetricType.NATURAL_LOSS.value == "natural_loss"
        assert MetricType.ROBUST_LOSS.value == "robust_loss"
        assert MetricType.EXPLANATION_STABILITY.value == "explanation_stability"

    def test_enum_membership(self):
        """Test enum membership."""
        assert MetricType.ROBUST_ACCURACY in MetricType
        assert "invalid_type" not in [m.value for m in MetricType]

    def test_enum_iteration(self):
        """Test iterating over metric types."""
        types = list(MetricType)
        assert len(types) == 8


class TestTrialMetrics:
    """Test TrialMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = TrialMetrics()
        assert metrics.robust_accuracy == 0.0
        assert metrics.clean_accuracy == 0.0
        assert metrics.cross_site_auroc == 0.0
        assert metrics.loss == float("inf")
        assert metrics.natural_loss == float("inf")
        assert metrics.robust_loss == float("inf")
        assert metrics.explanation_stability == 0.0
        assert metrics.epoch == 0
        assert metrics.step == 0
        assert isinstance(metrics.timestamp, float)

    def test_custom_metrics(self):
        """Test custom metric values."""
        timestamp = time.time()
        metrics = TrialMetrics(
            robust_accuracy=0.85,
            clean_accuracy=0.92,
            cross_site_auroc=0.88,
            loss=0.5,
            natural_loss=0.3,
            robust_loss=0.7,
            explanation_stability=0.75,
            epoch=10,
            step=500,
            timestamp=timestamp,
        )

        assert metrics.robust_accuracy == 0.85
        assert metrics.clean_accuracy == 0.92
        assert metrics.cross_site_auroc == 0.88
        assert metrics.loss == 0.5
        assert metrics.timestamp == timestamp

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        metrics_dict = metrics.to_dict()

        assert metrics_dict["robust_accuracy"] == 0.8
        assert metrics_dict["clean_accuracy"] == 0.9
        assert metrics_dict["cross_site_auroc"] == 0.85
        assert "loss" in metrics_dict
        assert "timestamp" in metrics_dict

    def test_is_valid_with_valid_metrics(self):
        """Test validation with valid metrics."""
        metrics = TrialMetrics(
            robust_accuracy=0.85, clean_accuracy=0.92, cross_site_auroc=0.88, loss=0.5
        )
        assert metrics.is_valid() is True

    def test_is_valid_with_nan(self):
        """Test validation with NaN values."""
        metrics = TrialMetrics(
            robust_accuracy=float("nan"),
            clean_accuracy=0.92,
            cross_site_auroc=0.88,
        )
        assert metrics.is_valid() is False

    def test_is_valid_with_inf_loss(self):
        """Test that inf loss is treated specially."""
        # loss defaults to inf, which should be handled
        metrics = TrialMetrics(
            robust_accuracy=0.85,
            clean_accuracy=0.92,
            cross_site_auroc=0.88,
        )
        assert metrics.is_valid() is True

    def test_is_valid_with_negative_inf(self):
        """Test validation with negative infinity."""
        metrics = TrialMetrics(
            robust_accuracy=float("-inf"),
            clean_accuracy=0.92,
            cross_site_auroc=0.88,
            loss=0.5,
        )
        assert metrics.is_valid() is False

    def test_timestamp_auto_generation(self):
        """Test that timestamp is auto-generated."""
        before = time.time()
        metrics = TrialMetrics()
        after = time.time()
        assert before <= metrics.timestamp <= after


class TestObjectiveConfig:
    """Test ObjectiveConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ObjectiveConfig()
        assert config.weights["robust_accuracy"] == 0.4
        assert config.weights["clean_accuracy"] == 0.3
        assert config.weights["cross_site_auroc"] == 0.3
        assert config.direction == "maximize"
        assert config.use_log_scale is False
        assert config.clip_range == (0.0, 1.0)
        assert config.penalty_invalid == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        custom_weights = {
            "robust_accuracy": 0.5,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.2,
        }
        config = ObjectiveConfig(
            weights=custom_weights,
            direction="minimize",
            use_log_scale=True,
            clip_range=(0.1, 0.9),
            penalty_invalid=-1.0,
        )

        assert config.weights == custom_weights
        assert config.direction == "minimize"
        assert config.use_log_scale is True
        assert config.clip_range == (0.1, 0.9)
        assert config.penalty_invalid == -1.0

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.5,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.2,
            }
        )
        # Should not raise - sums to 1.0
        assert sum(config.weights.values()) == 1.0

    def test_invalid_weights_sum(self):
        """Test validation fails with incorrect sum."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ObjectiveConfig(
                weights={
                    "robust_accuracy": 0.5,
                    "clean_accuracy": 0.3,
                    "cross_site_auroc": 0.1,  # Sums to 0.9
                }
            )

    def test_weights_sum_tolerance(self):
        """Test tolerance in weight sum validation."""
        # Should accept values within 0.99-1.01
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.295,  # Sums to 0.995
            }
        )
        assert config is not None

    def test_weights_sum_low_tolerance_fail(self):
        """Test validation fails below tolerance."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ObjectiveConfig(
                weights={
                    "robust_accuracy": 0.3,
                    "clean_accuracy": 0.3,
                    "cross_site_auroc": 0.3,  # Sums to 0.9
                }
            )

    def test_weights_sum_high_tolerance_fail(self):
        """Test validation fails above tolerance."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ObjectiveConfig(
                weights={
                    "robust_accuracy": 0.4,
                    "clean_accuracy": 0.4,
                    "cross_site_auroc": 0.3,  # Sums to 1.1
                }
            )


class TestObjectiveFunction:
    """Test ObjectiveFunction abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that ObjectiveFunction cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ObjectiveFunction()

    def test_subclass_must_implement_call(self):
        """Test that subclass must implement __call__."""

        class IncompleteObjective(ObjectiveFunction):
            def get_intermediate_value(self, metrics, epoch):
                return 0.0

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteObjective()

    def test_subclass_must_implement_get_intermediate_value(self):
        """Test that subclass must implement get_intermediate_value."""

        class IncompleteObjective(ObjectiveFunction):
            def __call__(self, metrics):
                return 0.0

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteObjective()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that complete subclass can be instantiated."""

        class CompleteObjective(ObjectiveFunction):
            def __call__(self, metrics):
                return 0.5

            def get_intermediate_value(self, metrics, epoch):
                return 0.5

        obj = CompleteObjective()
        metrics = TrialMetrics()
        assert obj(metrics) == 0.5
        assert obj.get_intermediate_value(metrics, 0) == 0.5


class TestWeightedTriObjective:
    """Test WeightedTriObjective implementation."""

    def test_default_initialization(self):
        """Test default initialization with standard weights."""
        objective = WeightedTriObjective()
        assert objective.config.weights["robust_accuracy"] == 0.4
        assert objective.config.weights["clean_accuracy"] == 0.3
        assert objective.config.weights["cross_site_auroc"] == 0.3

    def test_custom_weights_initialization(self):
        """Test initialization with custom weights."""
        objective = WeightedTriObjective(
            robust_weight=0.5, clean_weight=0.3, auroc_weight=0.2
        )
        assert objective.config.weights["robust_accuracy"] == 0.5
        assert objective.config.weights["clean_accuracy"] == 0.3
        assert objective.config.weights["cross_site_auroc"] == 0.2

    def test_config_initialization(self):
        """Test initialization with ObjectiveConfig."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.6,
                "clean_accuracy": 0.2,
                "cross_site_auroc": 0.2,
            }
        )
        objective = WeightedTriObjective(config=config)
        assert objective.config.weights["robust_accuracy"] == 0.6

    def test_validate_weights_missing_keys(self):
        """Test validation fails with missing weight keys."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.5,
                "clean_accuracy": 0.5,
                # Missing cross_site_auroc
            }
        )
        with pytest.raises(ValueError, match="Missing required weight keys"):
            WeightedTriObjective(config=config)

    def test_call_with_perfect_metrics(self):
        """Test objective computation with perfect scores."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=1.0, clean_accuracy=1.0, cross_site_auroc=1.0
        )
        score = objective(metrics)
        assert score == 1.0

    def test_call_with_zero_metrics(self):
        """Test objective computation with zero scores."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=0.0, clean_accuracy=0.0, cross_site_auroc=0.0
        )
        score = objective(metrics)
        assert score == 0.0

    def test_call_with_weighted_average(self):
        """Test objective computation with weighted average."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        # Expected: 0.4*0.8 + 0.3*0.9 + 0.3*0.85 = 0.32 + 0.27 + 0.255 = 0.845
        score = objective(metrics)
        assert abs(score - 0.845) < 1e-6

    def test_call_with_invalid_metrics(self):
        """Test objective returns penalty for invalid metrics."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=float("nan"),
            clean_accuracy=0.9,
            cross_site_auroc=0.85,
        )
        score = objective(metrics)
        assert score == objective.config.penalty_invalid

    def test_call_with_clipping(self):
        """Test that values outside clip range are clipped."""
        config = ObjectiveConfig(clip_range=(0.2, 0.8))
        objective = WeightedTriObjective(config=config)
        metrics = TrialMetrics(
            robust_accuracy=0.1,  # Below min
            clean_accuracy=0.95,  # Above max
            cross_site_auroc=0.5,  # Within range
        )
        score = objective(metrics)
        # Expected: 0.4*0.2 + 0.3*0.8 + 0.3*0.5 = 0.08 + 0.24 + 0.15 = 0.47
        assert abs(score - 0.47) < 1e-6

    def test_get_intermediate_value_single(self):
        """Test intermediate value with single evaluation."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        intermediate = objective.get_intermediate_value(metrics, epoch=0)
        assert abs(intermediate - 0.845) < 1e-6

    def test_get_intermediate_value_running_best(self):
        """Test intermediate value tracks running best."""
        objective = WeightedTriObjective()

        # First evaluation
        metrics1 = TrialMetrics(
            robust_accuracy=0.7, clean_accuracy=0.8, cross_site_auroc=0.75
        )
        intermediate1 = objective.get_intermediate_value(metrics1, epoch=0)

        # Second evaluation (worse)
        metrics2 = TrialMetrics(
            robust_accuracy=0.6, clean_accuracy=0.7, cross_site_auroc=0.65
        )
        intermediate2 = objective.get_intermediate_value(metrics2, epoch=1)

        # Should return best (first one)
        assert intermediate2 == intermediate1

    def test_get_intermediate_value_improves(self):
        """Test intermediate value improves with better metrics."""
        objective = WeightedTriObjective()

        # First evaluation
        metrics1 = TrialMetrics(
            robust_accuracy=0.7, clean_accuracy=0.8, cross_site_auroc=0.75
        )
        intermediate1 = objective.get_intermediate_value(metrics1, epoch=0)

        # Second evaluation (better)
        metrics2 = TrialMetrics(
            robust_accuracy=0.85, clean_accuracy=0.92, cross_site_auroc=0.88
        )
        intermediate2 = objective.get_intermediate_value(metrics2, epoch=1)

        # Should return new best
        assert intermediate2 > intermediate1

    def test_history_tracking(self):
        """Test that history is tracked across evaluations."""
        objective = WeightedTriObjective()

        for epoch in range(5):
            metrics = TrialMetrics(
                robust_accuracy=0.7 + epoch * 0.05,
                clean_accuracy=0.8 + epoch * 0.02,
                cross_site_auroc=0.75 + epoch * 0.03,
            )
            objective.get_intermediate_value(metrics, epoch=epoch)

        assert len(objective._history) == 5

    def test_equal_weights(self):
        """Test with equal weights."""
        objective = WeightedTriObjective(
            robust_weight=1 / 3, clean_weight=1 / 3, auroc_weight=1 / 3
        )
        metrics = TrialMetrics(
            robust_accuracy=0.6, clean_accuracy=0.9, cross_site_auroc=0.75
        )
        score = objective(metrics)
        # Expected: (0.6 + 0.9 + 0.75) / 3 = 0.75
        assert abs(score - 0.75) < 1e-6

    def test_robust_only_weight(self):
        """Test with robust accuracy only."""
        objective = WeightedTriObjective(
            robust_weight=1.0, clean_weight=0.0, auroc_weight=0.0
        )
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        score = objective(metrics)
        assert score == 0.8

    def test_clean_only_weight(self):
        """Test with clean accuracy only."""
        objective = WeightedTriObjective(
            robust_weight=0.0, clean_weight=1.0, auroc_weight=0.0
        )
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        score = objective(metrics)
        assert score == 0.9

    def test_auroc_only_weight(self):
        """Test with AUROC only."""
        objective = WeightedTriObjective(
            robust_weight=0.0, clean_weight=0.0, auroc_weight=1.0
        )
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        score = objective(metrics)
        assert score == 0.85

    def test_reset_history(self):
        """Test resetting history."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        objective.get_intermediate_value(metrics, epoch=0)
        assert len(objective._history) == 1

        objective.reset_history()
        assert len(objective._history) == 0

    def test_weights_property(self):
        """Test weights property returns copy."""
        objective = WeightedTriObjective()
        weights = objective.weights
        weights["robust_accuracy"] = 0.9
        # Original should be unchanged
        assert objective.config.weights["robust_accuracy"] == 0.4

    def test_get_component_contributions(self):
        """Test getting individual component contributions."""
        objective = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        contributions = objective.get_component_contributions(metrics)

        assert abs(contributions["robust_contribution"] - 0.32) < 1e-6
        assert abs(contributions["clean_contribution"] - 0.27) < 1e-6
        assert abs(contributions["auroc_contribution"] - 0.255) < 1e-6


class TestAdaptiveWeightedObjective:
    """Test AdaptiveWeightedObjective implementation."""

    def test_default_initialization(self):
        """Test default initialization."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(base_weights=base_weights)
        assert objective.strategy == "linear"
        assert objective.target_iterations == 50
        assert objective.iteration == 0

    def test_custom_initialization(self):
        """Test custom initialization."""
        base_weights = {
            "robust_accuracy": 0.5,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.2,
        }
        objective = AdaptiveWeightedObjective(
            base_weights=base_weights,
            adaptation_strategy="cyclic",
            target_iterations=100,
        )
        assert objective.strategy == "cyclic"
        assert objective.target_iterations == 100

    def test_call_with_valid_metrics(self):
        """Test objective computation."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(base_weights=base_weights)
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        score = objective(metrics)
        # Expected: 0.4*0.8 + 0.3*0.9 + 0.3*0.85 = 0.845
        assert abs(score - 0.845) < 1e-6

    def test_call_with_invalid_metrics(self):
        """Test returns 0.0 for invalid metrics."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(base_weights=base_weights)
        metrics = TrialMetrics(
            robust_accuracy=float("nan"),
            clean_accuracy=0.9,
            cross_site_auroc=0.85,
        )
        score = objective(metrics)
        assert score == 0.0

    def test_get_intermediate_value(self):
        """Test intermediate value computation."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(base_weights=base_weights)
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        intermediate = objective.get_intermediate_value(metrics, epoch=5)
        assert abs(intermediate - 0.845) < 1e-6

    def test_update_weights_linear(self):
        """Test linear weight update strategy."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(
            base_weights=base_weights, target_iterations=100
        )

        # Update at halfway point
        objective.update_weights(50)
        assert objective.iteration == 50
        # Weights should be normalized to sum to 1
        assert abs(sum(objective.current_weights.values()) - 1.0) < 1e-6

    def test_update_weights_cyclic(self):
        """Test cyclic weight update strategy."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(
            base_weights=base_weights,
            adaptation_strategy="cyclic",
            target_iterations=10,
        )

        # Test each cycle position
        for i in range(3):
            objective.update_weights(i)
            # One weight should be 0.5, others 0.25
            weights_list = list(objective.current_weights.values())
            assert 0.5 in weights_list
            assert weights_list.count(0.25) == 2

    def test_weights_normalization(self):
        """Test that weights are normalized to sum to 1."""
        base_weights = {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
        objective = AdaptiveWeightedObjective(
            base_weights=base_weights, target_iterations=100
        )

        for iteration in [0, 25, 50, 75, 100]:
            objective.update_weights(iteration)
            total = sum(objective.current_weights.values())
            assert abs(total - 1.0) < 1e-6


class TestMultiObjectiveEvaluator:
    """Test MultiObjectiveEvaluator implementation."""

    def test_default_initialization(self):
        """Test default initialization."""
        evaluator = MultiObjectiveEvaluator()
        assert evaluator.objectives == [
            "robust_accuracy",
            "clean_accuracy",
            "cross_site_auroc",
        ]
        assert len(evaluator.reference_point) == 3
        assert np.all(evaluator.reference_point == 0.0)

    def test_custom_initialization(self):
        """Test custom initialization."""
        objectives = ["robust_accuracy", "clean_accuracy"]
        evaluator = MultiObjectiveEvaluator(objectives=objectives)
        assert evaluator.objectives == objectives
        assert len(evaluator.reference_point) == 2

    def test_extract_objectives(self):
        """Test extracting objectives from metrics."""
        evaluator = MultiObjectiveEvaluator()
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        objectives = evaluator.extract_objectives(metrics)
        assert np.allclose(objectives, [0.8, 0.9, 0.85])

    def test_dominates_true(self):
        """Test domination check when a dominates b."""
        evaluator = MultiObjectiveEvaluator()
        a = np.array([0.9, 0.9, 0.9])
        b = np.array([0.8, 0.8, 0.8])
        assert evaluator.dominates(a, b) is True

    def test_dominates_false_worse(self):
        """Test domination check when a is worse."""
        evaluator = MultiObjectiveEvaluator()
        a = np.array([0.7, 0.7, 0.7])
        b = np.array([0.8, 0.8, 0.8])
        assert evaluator.dominates(a, b) is False

    def test_dominates_false_equal(self):
        """Test domination check with equal solutions."""
        evaluator = MultiObjectiveEvaluator()
        a = np.array([0.8, 0.8, 0.8])
        b = np.array([0.8, 0.8, 0.8])
        assert evaluator.dominates(a, b) is False

    def test_dominates_partial(self):
        """Test domination with partial ordering."""
        evaluator = MultiObjectiveEvaluator()
        a = np.array([0.9, 0.7, 0.8])
        b = np.array([0.8, 0.9, 0.7])
        # Neither dominates
        assert evaluator.dominates(a, b) is False
        assert evaluator.dominates(b, a) is False

    def test_update_pareto_front_first_solution(self):
        """Test adding first solution to Pareto front."""
        evaluator = MultiObjectiveEvaluator()
        solution = np.array([0.8, 0.9, 0.85])
        added = evaluator.update_pareto_front(solution)
        assert added is True
        assert len(evaluator._pareto_front) == 1

    def test_update_pareto_front_non_dominated(self):
        """Test adding non-dominated solution."""
        evaluator = MultiObjectiveEvaluator()
        solution1 = np.array([0.9, 0.7, 0.8])
        solution2 = np.array([0.8, 0.9, 0.7])

        evaluator.update_pareto_front(solution1)
        added = evaluator.update_pareto_front(solution2)

        assert added is True
        assert len(evaluator._pareto_front) == 2

    def test_update_pareto_front_dominated(self):
        """Test that dominated solution is not added."""
        evaluator = MultiObjectiveEvaluator()
        solution1 = np.array([0.9, 0.9, 0.9])
        solution2 = np.array([0.8, 0.8, 0.8])

        evaluator.update_pareto_front(solution1)
        added = evaluator.update_pareto_front(solution2)

        assert added is False
        assert len(evaluator._pareto_front) == 1

    def test_update_pareto_front_dominates_existing(self):
        """Test that new solution removes dominated ones."""
        evaluator = MultiObjectiveEvaluator()
        solution1 = np.array([0.8, 0.8, 0.8])
        solution2 = np.array([0.9, 0.9, 0.9])

        evaluator.update_pareto_front(solution1)
        evaluator.update_pareto_front(solution2)

        assert len(evaluator._pareto_front) == 1
        assert np.allclose(evaluator._pareto_front[0], solution2)

    def test_get_pareto_front_empty(self):
        """Test getting empty Pareto front."""
        evaluator = MultiObjectiveEvaluator()
        front = evaluator.get_pareto_front()
        assert front.size == 0

    def test_get_pareto_front_with_solutions(self):
        """Test getting Pareto front with solutions."""
        evaluator = MultiObjectiveEvaluator()
        solution1 = np.array([0.9, 0.7, 0.8])
        solution2 = np.array([0.8, 0.9, 0.7])

        evaluator.update_pareto_front(solution1)
        evaluator.update_pareto_front(solution2)

        front = evaluator.get_pareto_front()
        assert front.shape == (2, 3)

    def test_compute_hypervolume_empty(self):
        """Test hypervolume computation with empty front."""
        evaluator = MultiObjectiveEvaluator()
        hypervolume = evaluator.compute_hypervolume()
        assert hypervolume == 0.0

    def test_compute_hypervolume_2d(self):
        """Test 2D hypervolume computation."""
        objectives = ["robust_accuracy", "clean_accuracy"]
        evaluator = MultiObjectiveEvaluator(objectives=objectives)

        solution1 = np.array([0.8, 0.7])
        solution2 = np.array([0.7, 0.8])

        evaluator.update_pareto_front(solution1)
        evaluator.update_pareto_front(solution2)

        hypervolume = evaluator.compute_hypervolume()
        assert hypervolume > 0.0

    def test_compute_hypervolume_3d(self):
        """Test 3D hypervolume computation (approximate)."""
        evaluator = MultiObjectiveEvaluator()

        solution1 = np.array([0.9, 0.8, 0.85])
        solution2 = np.array([0.85, 0.9, 0.8])

        evaluator.update_pareto_front(solution1)
        evaluator.update_pareto_front(solution2)

        hypervolume = evaluator.compute_hypervolume()
        assert hypervolume >= 0.0

    def test_2d_hypervolume_single_point(self):
        """Test 2D hypervolume with single point."""
        objectives = ["robust_accuracy", "clean_accuracy"]
        evaluator = MultiObjectiveEvaluator(objectives=objectives)

        solution = np.array([0.8, 0.9])
        evaluator.update_pareto_front(solution)

        hypervolume = evaluator.compute_hypervolume()
        # Should be 0.8 * 0.9 = 0.72
        assert abs(hypervolume - 0.72) < 0.01

    def test_extract_objectives_missing_attribute(self):
        """Test extracting objectives with missing attribute."""
        evaluator = MultiObjectiveEvaluator(
            objectives=["robust_accuracy", "nonexistent_metric"]
        )
        metrics = TrialMetrics(robust_accuracy=0.8)
        objectives = evaluator.extract_objectives(metrics)
        assert objectives[0] == 0.8
        assert objectives[1] == 0.0  # Default for missing
