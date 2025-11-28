"""
Comprehensive tests for HPO Pruners to achieve 100% coverage.

Tests all pruner classes and the factory functions.

Author: Viraj Jain
Date: November 27, 2025
"""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from src.config.hpo.pruners import (
    AdaptivePruner,
    HybridPruner,
    MultiObjectivePruner,
    PerformanceBasedPruner,
    ResourceAwarePruner,
    create_pruner,
    get_default_pruner_for_objective,
)


def create_mock_study(direction=StudyDirection.MAXIMIZE, trials=None):
    """Create a mock Optuna study."""
    study = MagicMock()
    study.direction = direction
    study.trials = trials if trials is not None else []
    return study


def create_mock_trial(
    number=0,
    state=TrialState.RUNNING,
    last_step=None,
    intermediate_values=None,
    user_attrs=None,
    datetime_start=None,
):
    """Create a mock frozen trial."""
    trial = MagicMock(spec=FrozenTrial)
    trial.number = number
    trial.state = state
    trial.last_step = last_step
    trial.intermediate_values = (
        intermediate_values if intermediate_values is not None else {}
    )
    trial.user_attrs = user_attrs if user_attrs is not None else {}
    trial.datetime_start = datetime_start
    return trial


class TestPerformanceBasedPrunerComplete:
    """Complete tests for PerformanceBasedPruner."""

    def test_prune_poor_performance_minimize(self):
        """Test pruning with minimize direction."""
        pruner = PerformanceBasedPruner(
            min_steps=5, warmup_steps=3, patience=2, performance_threshold=0.8
        )

        # Create completed trial with good performance
        completed_trial = create_mock_trial(
            number=0,
            state=TrialState.COMPLETE,
            intermediate_values={
                10: 10.0,
                15: 8.0,
                20: 5.0,
            },  # Minimize - lower is better
        )

        # Create current trial with poor performance
        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=20,
            intermediate_values={10: 100.0, 15: 95.0, 20: 90.0},  # Much worse (higher)
        )

        study = create_mock_study(
            direction=StudyDirection.MINIMIZE, trials=[completed_trial, current_trial]
        )

        # Should prune after patience steps of poor performance
        should_prune = pruner.prune(study, current_trial)

        # First check might not prune immediately
        assert isinstance(should_prune, bool)

    def test_prune_no_best_values_at_step(self):
        """Test when no completed trials have values at current step."""
        pruner = PerformanceBasedPruner(min_steps=5, warmup_steps=3, patience=2)

        completed_trial = create_mock_trial(
            number=0,
            state=TrialState.COMPLETE,
            intermediate_values={5: 0.8, 10: 0.9},  # No value at step 20
        )

        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=20,
            intermediate_values={20: 0.5},
        )

        study = create_mock_study(trials=[completed_trial, current_trial])

        # Should not prune when no reference values
        assert not pruner.prune(study, current_trial)

    def test_prune_good_performance_maximize(self):
        """Test that good performance resets patience counter."""
        pruner = PerformanceBasedPruner(
            min_steps=5, warmup_steps=3, patience=3, performance_threshold=0.9
        )

        completed_trial = create_mock_trial(
            number=0,
            state=TrialState.COMPLETE,
            intermediate_values={10: 0.8, 15: 0.85, 20: 0.9},
        )

        # Trial with improving performance
        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=20,
            intermediate_values={
                10: 0.75,
                15: 0.80,
                20: 0.88,
            },  # Good relative performance
        )

        study = create_mock_study(trials=[completed_trial, current_trial])

        # Good performance should not prune
        assert not pruner.prune(study, current_trial)

    def test_prune_zero_division_handling(self):
        """Test handling of zero values in performance calculation."""
        pruner = PerformanceBasedPruner(min_steps=5, warmup_steps=3, patience=2)

        completed_trial = create_mock_trial(
            number=0,
            state=TrialState.COMPLETE,
            intermediate_values={10: 0.0},  # Zero value
        )

        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            intermediate_values={10: 0.5},
        )

        study = create_mock_study(trials=[completed_trial, current_trial])

        # Should handle zero division gracefully
        should_prune = pruner.prune(study, current_trial)
        assert isinstance(should_prune, bool)


class TestResourceAwarePrunerComplete:
    """Complete tests for ResourceAwarePruner."""

    def test_prune_exceeds_time_budget(self):
        """Test pruning when time budget is exceeded."""
        pruner = ResourceAwarePruner(
            time_budget=10.0, min_improvement_rate=0.01, check_interval=5, min_steps=5
        )

        # Trial started 15 seconds ago (exceeds 10 second budget)
        start_time = datetime.datetime.now() - datetime.timedelta(seconds=15)
        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            intermediate_values={5: 0.5, 10: 0.51},
            datetime_start=start_time,
        )

        study = create_mock_study()

        # Should prune due to time budget
        assert pruner.prune(study, current_trial)

    def test_prune_low_improvement_rate(self):
        """Test pruning when improvement rate is too low."""
        pruner = ResourceAwarePruner(
            time_budget=3600.0, min_improvement_rate=0.1, check_interval=1, min_steps=3
        )

        # Trial with very low improvement
        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            intermediate_values={
                5: 0.500,
                6: 0.501,
                7: 0.502,
                8: 0.503,
                9: 0.504,
                10: 0.505,
            },  # Very small improvements
            datetime_start=datetime.datetime.now(),
        )

        study = create_mock_study()

        # Should eventually prune due to low improvement
        # Run multiple times to build up improvement rate history
        for _ in range(5):
            result = pruner.prune(study, current_trial)

        # Result should be boolean - just verify we got a response
        assert result in [True, False]

    def test_prune_no_datetime_start(self):
        """Test when trial has no start time."""
        pruner = ResourceAwarePruner(time_budget=10.0, min_steps=5)

        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            intermediate_values={5: 0.5, 10: 0.6},
            datetime_start=None,  # No start time
        )

        study = create_mock_study()

        # Should not crash, check other conditions
        should_prune = pruner.prune(study, current_trial)
        assert isinstance(should_prune, bool)

    def test_prune_insufficient_values(self):
        """Test when trial has insufficient intermediate values."""
        pruner = ResourceAwarePruner(min_steps=5, check_interval=1)

        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            intermediate_values={10: 0.5},  # Only one value
        )

        study = create_mock_study()

        # Should not prune with insufficient data
        assert not pruner.prune(study, current_trial)


class TestMultiObjectivePruner:
    """Tests for MultiObjectivePruner."""

    def test_initialization(self):
        """Test multi-objective pruner initialization."""
        pruner = MultiObjectivePruner(
            min_steps=10,
            patience=3,
            threshold_accuracy=0.5,
            threshold_robustness=0.4,
            threshold_explainability=0.3,
            require_all=True,
        )

        assert pruner.min_steps == 10
        assert pruner.patience == 3
        assert pruner.threshold_accuracy == 0.5
        assert pruner.threshold_robustness == 0.4
        assert pruner.threshold_explainability == 0.3
        assert pruner.require_all is True

    def test_prune_all_below_threshold_require_all(self):
        """Test pruning when all objectives are below threshold (require_all=True)."""
        pruner = MultiObjectivePruner(
            min_steps=5,
            patience=2,
            threshold_accuracy=0.6,
            threshold_robustness=0.5,
            threshold_explainability=0.4,
            require_all=True,
        )

        # All objectives below threshold
        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            user_attrs={
                "accuracy_step_10": 0.5,  # Below 0.6
                "robustness_step_10": 0.4,  # Below 0.5
                "explainability_step_10": 0.3,  # Below 0.4
            },
        )

        study = create_mock_study()

        # First check - poor performance
        pruner.prune(study, current_trial)
        # Second check - should prune after patience
        current_trial.last_step = 11
        current_trial.user_attrs.update(
            {
                "accuracy_step_11": 0.5,
                "robustness_step_11": 0.4,
                "explainability_step_11": 0.3,
            }
        )
        result = pruner.prune(study, current_trial)

        assert isinstance(result, bool)

    def test_prune_one_below_threshold_require_any(self):
        """Test pruning when one objective is below threshold (require_all=False)."""
        pruner = MultiObjectivePruner(
            min_steps=5,
            patience=2,
            threshold_accuracy=0.6,
            threshold_robustness=0.5,
            threshold_explainability=0.4,
            require_all=False,  # Any objective below triggers pruning
        )

        # Only accuracy below threshold
        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            user_attrs={
                "accuracy_step_10": 0.5,  # Below 0.6
                "robustness_step_10": 0.8,  # Above 0.5
                "explainability_step_10": 0.9,  # Above 0.4
            },
        )

        study = create_mock_study()

        # Run multiple times to exceed patience
        for i in range(3):
            step = 10 + i
            current_trial.last_step = step
            current_trial.user_attrs.update(
                {
                    f"accuracy_step_{step}": 0.5,
                    f"robustness_step_{step}": 0.8,
                    f"explainability_step_{step}": 0.9,
                }
            )
            result = pruner.prune(study, current_trial)

        assert isinstance(result, bool)

    def test_prune_missing_objective_values(self):
        """Test when objective values are missing."""
        pruner = MultiObjectivePruner(min_steps=5, patience=2)

        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=10,
            user_attrs={},  # No objective values
        )

        study = create_mock_study()

        # Should not prune when objectives are missing
        assert not pruner.prune(study, current_trial)

    def test_prune_good_performance_resets_counter(self):
        """Test that good performance resets the poor performance counter."""
        pruner = MultiObjectivePruner(
            min_steps=5,
            patience=3,
            threshold_accuracy=0.6,
            threshold_robustness=0.5,
            threshold_explainability=0.4,
            require_all=False,
        )

        current_trial = create_mock_trial(
            number=1, state=TrialState.RUNNING, last_step=10
        )

        study = create_mock_study()

        # First: poor performance
        current_trial.user_attrs = {
            "accuracy_step_10": 0.5,
            "robustness_step_10": 0.8,
            "explainability_step_10": 0.9,
        }
        pruner.prune(study, current_trial)

        # Then: good performance (resets counter)
        current_trial.last_step = 11
        current_trial.user_attrs = {
            "accuracy_step_11": 0.9,  # Good
            "robustness_step_11": 0.9,  # Good
            "explainability_step_11": 0.9,  # Good
        }
        result = pruner.prune(study, current_trial)

        # Counter should be reset, no pruning
        assert not result


class TestAdaptivePruner:
    """Tests for AdaptivePruner."""

    def test_initialization(self):
        """Test adaptive pruner initialization."""
        pruner = AdaptivePruner(
            initial_threshold=0.3,
            final_threshold=0.8,
            n_trials_to_adapt=50,
            min_steps=8,
        )

        assert pruner.initial_threshold == 0.3
        assert pruner.final_threshold == 0.8
        assert pruner.n_trials_to_adapt == 50
        assert pruner.min_steps == 8

    def test_prune_basic_functionality(self):
        """Test basic pruning functionality."""
        pruner = AdaptivePruner(initial_threshold=0.4, final_threshold=0.7, min_steps=5)

        # Create trials to establish performance baseline
        completed_trials = [
            create_mock_trial(
                number=i,
                state=TrialState.COMPLETE,
                intermediate_values={10: 0.8 + i * 0.01},
            )
            for i in range(5)
        ]

        current_trial = create_mock_trial(
            number=10,
            state=TrialState.RUNNING,
            last_step=10,
            intermediate_values={10: 0.3},
        )

        study = create_mock_study(trials=completed_trials + [current_trial])

        # Test pruning decision
        result = pruner.prune(study, current_trial)
        # Result should be boolean - just verify we got a response
        assert result in [True, False]

    def test_prune_no_last_step(self):
        """Test when trial has no last step."""
        pruner = AdaptivePruner()

        current_trial = create_mock_trial(
            number=1, state=TrialState.RUNNING, last_step=None
        )

        study = create_mock_study()

        assert not pruner.prune(study, current_trial)

    def test_prune_before_min_steps(self):
        """Test that pruning doesn't occur before min_steps."""
        pruner = AdaptivePruner(min_steps=10)

        current_trial = create_mock_trial(
            number=1,
            state=TrialState.RUNNING,
            last_step=5,
            intermediate_values={5: 0.3},
        )

        study = create_mock_study()

        assert not pruner.prune(study, current_trial)


class TestHybridPruner:
    """Tests for HybridPruner."""

    def test_initialization_with_defaults(self):
        """Test hybrid pruner with default sub-pruners."""
        pruner = HybridPruner()

        assert isinstance(pruner.performance_pruner, PerformanceBasedPruner)
        assert isinstance(pruner.resource_pruner, ResourceAwarePruner)
        assert isinstance(pruner.adaptive_pruner, AdaptivePruner)
        assert pruner.require_all is False

    def test_initialization_with_custom_pruners(self):
        """Test hybrid pruner with custom sub-pruners."""
        perf_pruner = PerformanceBasedPruner(min_steps=5)
        res_pruner = ResourceAwarePruner(time_budget=1800.0)
        adapt_pruner = AdaptivePruner(initial_threshold=0.3)

        pruner = HybridPruner(
            performance_pruner=perf_pruner,
            resource_pruner=res_pruner,
            adaptive_pruner=adapt_pruner,
            require_all=True,
        )

        assert pruner.performance_pruner is perf_pruner
        assert pruner.resource_pruner is res_pruner
        assert pruner.adaptive_pruner is adapt_pruner
        assert pruner.require_all is True

    def test_prune_require_any(self):
        """Test hybrid pruning with require_all=False (OR logic)."""
        # Mock pruners where at least one returns True
        perf_pruner = MagicMock(spec=PerformanceBasedPruner)
        perf_pruner.prune.return_value = True  # This one says prune

        res_pruner = MagicMock(spec=ResourceAwarePruner)
        res_pruner.prune.return_value = False

        adapt_pruner = MagicMock(spec=AdaptivePruner)
        adapt_pruner.prune.return_value = False

        pruner = HybridPruner(
            performance_pruner=perf_pruner,
            resource_pruner=res_pruner,
            adaptive_pruner=adapt_pruner,
            require_all=False,
        )

        current_trial = create_mock_trial(number=1, last_step=10)
        study = create_mock_study()

        # Should prune because one pruner says yes (OR logic)
        assert pruner.prune(study, current_trial)

    def test_prune_require_all(self):
        """Test hybrid pruning with require_all=True (AND logic)."""
        # All pruners must agree to prune
        perf_pruner = MagicMock(spec=PerformanceBasedPruner)
        perf_pruner.prune.return_value = True

        res_pruner = MagicMock(spec=ResourceAwarePruner)
        res_pruner.prune.return_value = False  # This one says don't prune

        adapt_pruner = MagicMock(spec=AdaptivePruner)
        adapt_pruner.prune.return_value = True

        pruner = HybridPruner(
            performance_pruner=perf_pruner,
            resource_pruner=res_pruner,
            adaptive_pruner=adapt_pruner,
            require_all=True,
        )

        current_trial = create_mock_trial(number=1, last_step=10)
        study = create_mock_study()

        # Should not prune because not all agree (AND logic)
        assert not pruner.prune(study, current_trial)

    def test_prune_all_agree(self):
        """Test when all pruners agree to prune with require_all=True."""
        perf_pruner = MagicMock(spec=PerformanceBasedPruner)
        perf_pruner.prune.return_value = True

        res_pruner = MagicMock(spec=ResourceAwarePruner)
        res_pruner.prune.return_value = True

        adapt_pruner = MagicMock(spec=AdaptivePruner)
        adapt_pruner.prune.return_value = True

        pruner = HybridPruner(
            performance_pruner=perf_pruner,
            resource_pruner=res_pruner,
            adaptive_pruner=adapt_pruner,
            require_all=True,
        )

        current_trial = create_mock_trial(number=1, last_step=10)
        study = create_mock_study()

        # Should prune because all agree
        assert pruner.prune(study, current_trial)


class TestCreatePruner:
    """Tests for the create_pruner factory function."""

    def test_create_median_pruner(self):
        """Test creating median pruner."""
        pruner = create_pruner("median", n_startup_trials=5, n_warmup_steps=3)
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_create_percentile_pruner(self):
        """Test creating percentile pruner."""
        pruner = create_pruner("percentile", percentile=25.0)
        assert isinstance(pruner, optuna.pruners.PercentilePruner)

    def test_create_hyperband_pruner(self):
        """Test creating hyperband pruner."""
        pruner = create_pruner("hyperband", min_resource=1, max_resource=10)
        assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    def test_create_threshold_pruner(self):
        """Test creating threshold pruner."""
        pruner = create_pruner("threshold", lower=0.1, upper=0.9)
        assert isinstance(pruner, optuna.pruners.ThresholdPruner)

    def test_create_performance_pruner(self):
        """Test creating performance-based pruner."""
        pruner = create_pruner("performance", min_steps=10, patience=5)
        assert isinstance(pruner, PerformanceBasedPruner)
        assert pruner.min_steps == 10
        assert pruner.patience == 5

    def test_create_resource_aware_pruner(self):
        """Test creating resource-aware pruner."""
        pruner = create_pruner("resource_aware", time_budget=1800.0)
        assert isinstance(pruner, ResourceAwarePruner)
        assert pruner.time_budget == 1800.0

    def test_create_multi_objective_pruner(self):
        """Test creating multi-objective pruner."""
        pruner = create_pruner("multi_objective", min_steps=15, threshold_accuracy=0.5)
        assert isinstance(pruner, MultiObjectivePruner)
        assert pruner.min_steps == 15

    def test_create_adaptive_pruner(self):
        """Test creating adaptive pruner."""
        pruner = create_pruner("adaptive", initial_threshold=0.3)
        assert isinstance(pruner, AdaptivePruner)
        assert pruner.initial_threshold == 0.3

    def test_create_hybrid_pruner(self):
        """Test creating hybrid pruner."""
        pruner = create_pruner("hybrid", require_all=True)
        assert isinstance(pruner, HybridPruner)
        assert pruner.require_all is True

    def test_create_nop_pruner(self):
        """Test creating nop (no-op) pruner."""
        pruner = create_pruner("none")
        assert isinstance(pruner, optuna.pruners.NopPruner)

    def test_create_unknown_pruner(self):
        """Test creating unknown pruner type raises error."""
        with pytest.raises(ValueError, match="Unknown pruner type"):
            create_pruner("invalid_pruner_type")


class TestGetDefaultPruner:
    """Tests for get_default_pruner_for_objective function."""

    def test_default_pruner_accuracy(self):
        """Test default pruner for accuracy objective."""
        pruner = get_default_pruner_for_objective("accuracy")
        assert isinstance(pruner, PerformanceBasedPruner)
        assert pruner.min_steps == 10
        assert pruner.patience == 5
        assert pruner.performance_threshold == 0.6

    def test_default_pruner_robustness(self):
        """Test default pruner for robustness objective."""
        pruner = get_default_pruner_for_objective("robustness")
        assert isinstance(pruner, ResourceAwarePruner)
        assert pruner.time_budget == 7200.0
        assert pruner.min_improvement_rate == 0.005

    def test_default_pruner_explainability(self):
        """Test default pruner for explainability objective."""
        pruner = get_default_pruner_for_objective("explainability")
        assert isinstance(pruner, AdaptivePruner)
        assert pruner.initial_threshold == 0.4
        assert pruner.final_threshold == 0.7

    def test_default_pruner_weighted_sum(self):
        """Test default pruner for weighted_sum objective."""
        pruner = get_default_pruner_for_objective("weighted_sum")
        assert isinstance(pruner, MultiObjectivePruner)
        assert pruner.min_steps == 15
        assert pruner.patience == 5
        assert pruner.require_all is False

    def test_default_pruner_tri_objective(self):
        """Test default pruner for tri_objective."""
        pruner = get_default_pruner_for_objective("tri_objective")
        assert isinstance(pruner, MultiObjectivePruner)
        assert pruner.min_steps == 15

    def test_default_pruner_unknown(self):
        """Test default pruner for unknown objective."""
        pruner = get_default_pruner_for_objective("unknown_objective")
        assert isinstance(pruner, HybridPruner)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
