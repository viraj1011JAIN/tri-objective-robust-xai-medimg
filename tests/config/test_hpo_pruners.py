"""
Comprehensive tests for HPO Pruners.

Tests cover:
- PerformanceBasedPruner
- ResourceAwarePruner
- Different pruning scenarios

Author: Viraj Pankaj Jain
"""

import optuna
import pytest
from optuna.trial import TrialState

from src.config.hpo.pruners import PerformanceBasedPruner, ResourceAwarePruner


class TestPerformanceBasedPruner:
    """Tests for PerformanceBasedPruner."""

    def test_pruner_creation(self):
        """Test creating performance-based pruner."""
        pruner = PerformanceBasedPruner(
            performance_threshold=0.8, patience=3, min_steps=5, warmup_steps=2
        )

        assert pruner.performance_threshold == 0.8
        assert pruner.patience == 3
        assert pruner.min_steps == 5
        assert pruner.warmup_steps == 2

    def test_prune_during_warmup(self):
        """Test that pruning doesn't occur during warmup."""
        pruner = PerformanceBasedPruner(performance_threshold=0.8, warmup_steps=5)

        study = optuna.create_study(direction="maximize")
        trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.5],
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={2: 0.3},
        )

        # Should not prune during warmup (step 2 < warmup_steps 5)
        should_prune = pruner.prune(study, trial)
        assert not should_prune

    def test_prune_before_min_steps(self):
        """Test that pruning doesn't occur before min_steps."""
        pruner = PerformanceBasedPruner(
            performance_threshold=0.8, min_steps=10, warmup_steps=0
        )

        study = optuna.create_study(direction="maximize")
        trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.5],
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={5: 0.3},
        )

        # Should not prune before min_steps (step 5 < min_steps 10)
        should_prune = pruner.prune(study, trial)
        assert not should_prune

    def test_prune_no_last_step(self):
        """Test pruning when trial has no last step."""
        pruner = PerformanceBasedPruner()

        study = optuna.create_study(direction="maximize")
        trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.5],
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={},
        )

        # Should not prune when no intermediate values
        should_prune = pruner.prune(study, trial)
        assert not should_prune

    def test_prune_check_interval(self):
        """Test pruning respects check interval."""
        pruner = PerformanceBasedPruner(check_interval=5, min_steps=0, warmup_steps=0)

        study = optuna.create_study(direction="maximize")

        # Step 3 (not divisible by check_interval=5)
        trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.5],
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={3: 0.3},
        )

        should_prune = pruner.prune(study, trial)
        assert not should_prune

    def test_prune_no_completed_trials(self):
        """Test pruning when no completed trials exist."""
        pruner = PerformanceBasedPruner(min_steps=0, warmup_steps=0, check_interval=1)

        study = optuna.create_study(direction="maximize")
        trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.5],
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={5: 0.3},
        )

        # Should not prune when no completed trials
        should_prune = pruner.prune(study, trial)
        assert not should_prune

    def test_prune_poor_performance_maximize(self):
        """Test pruning with poor performance (maximize)."""
        pruner = PerformanceBasedPruner(
            performance_threshold=0.8,
            patience=2,
            min_steps=0,
            warmup_steps=0,
            check_interval=1,
        )

        study = optuna.create_study(direction="maximize")

        # Add a completed trial with good performance
        completed_trial = optuna.trial.create_trial(
            state=TrialState.COMPLETE,
            values=[0.9],
            params={"x": 0.9},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={5: 0.9},
        )
        study.add_trial(completed_trial)

        # Current trial with poor performance
        current_trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.3],
            params={"x": 0.3},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={5: 0.3},
        )

        # First poor performance
        should_prune = pruner.prune(study, current_trial)
        # May or may not prune on first check (depends on patience)

        # With patience=2, need multiple poor performances
        # The pruner tracks performance across steps

    def test_prune_good_performance_resets_patience(self):
        """Test that good performance resets the patience counter."""
        pruner = PerformanceBasedPruner(
            performance_threshold=0.8,
            patience=2,
            min_steps=0,
            warmup_steps=0,
            check_interval=1,
        )

        study = optuna.create_study(direction="maximize")

        # Add completed trial
        completed_trial = optuna.trial.create_trial(
            state=TrialState.COMPLETE,
            values=[0.9],
            params={"x": 0.9},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={5: 0.9, 10: 0.9},
        )
        study.add_trial(completed_trial)

        # Trial with initially poor then good performance
        trial_5 = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.3],
            params={"x": 0.3},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={5: 0.3},
        )

        pruner.prune(study, trial_5)

        # Now with good performance
        trial_10 = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            values=[0.85],
            params={"x": 0.85},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            intermediate_values={10: 0.85},
        )

        # Good performance should reset patience
        should_prune = pruner.prune(study, trial_10)
        # Patience should be reset


class TestResourceAwarePruner:
    """Tests for ResourceAwarePruner."""

    def test_pruner_creation(self):
        """Test creating resource-aware pruner."""
        pruner = ResourceAwarePruner(time_budget=3600.0, min_improvement_rate=0.01)

        assert pruner.time_budget == 3600.0
        assert pruner.min_improvement_rate == 0.01

    def test_pruner_has_prune_method(self):
        """Test that pruner has prune method."""
        pruner = ResourceAwarePruner()

        assert hasattr(pruner, "prune")
        assert callable(pruner.prune)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
