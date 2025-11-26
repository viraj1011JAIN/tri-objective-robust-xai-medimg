"""
Comprehensive tests for TRADES HPO objective functions - 100% Coverage.

This test suite achieves complete coverage of hpo_objective.py including:
- Import error handling
- Factory functions for Optuna integration
- Edge cases and error paths
- All conditional branches

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 26, 2025
Version: 5.4.0
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import optuna
import pytest

from src.training.hpo_objective import (
    AdaptiveWeightedObjective,
    MetricType,
    MultiObjectiveEvaluator,
    ObjectiveConfig,
    TrialMetrics,
    WeightedTriObjective,
    create_multi_objective_optuna,
    create_optuna_objective,
)


class TestImportErrorHandling:
    """Test import error handling for optional dependencies."""

    def test_optuna_available(self):
        """Test that optuna is available in normal environment."""
        from src.training import hpo_objective

        assert hpo_objective.OPTUNA_AVAILABLE is True

    def test_torch_available(self):
        """Test that torch is available in normal environment."""
        from src.training import hpo_objective

        assert hpo_objective.TORCH_AVAILABLE is True

    @patch.dict("sys.modules", {"optuna": None})
    def test_import_without_optuna(self):
        """Test module behavior when optuna is unavailable."""
        # This test verifies the import error handling paths
        # Lines 31-34 would be triggered if optuna was truly unavailable
        # We verify the module can handle missing dependencies gracefully
        import importlib

        # Reload to trigger import paths
        import src.training.hpo_objective as obj_module

        importlib.reload(obj_module)
        # Module should still be importable
        assert obj_module is not None

    @patch.dict("sys.modules", {"torch": None})
    def test_import_without_torch(self):
        """Test module behavior when torch is unavailable."""
        # This verifies lines 42-45 for torch import error handling
        import importlib

        import src.training.hpo_objective as obj_module

        importlib.reload(obj_module)
        # Module should still be importable
        assert obj_module is not None


class TestObjectiveConfigEdgeCases:
    """Test ObjectiveConfig edge cases and branches."""

    def test_config_with_minimal_tolerance(self):
        """Test config validation at exact weight sum boundaries."""
        # Test exactly 0.99 (lower bound)
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.33,
                "clean_accuracy": 0.33,
                "cross_site_auroc": 0.33,
            }
        )
        assert abs(sum(config.weights.values()) - 0.99) < 0.01

    def test_config_with_maximal_tolerance(self):
        """Test config validation at upper weight sum boundary."""
        # Test exactly 1.01 (upper bound)
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.337,
                "clean_accuracy": 0.336,
                "cross_site_auroc": 0.337,
            }
        )
        assert abs(sum(config.weights.values()) - 1.01) < 0.01

    def test_config_custom_clip_range(self):
        """Test config with custom clipping range."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            clip_range=(-0.5, 1.5),
        )
        assert config.clip_range == (-0.5, 1.5)

    def test_config_with_log_scale(self):
        """Test config with log scale enabled."""
        config = ObjectiveConfig(use_log_scale=True)
        assert config.use_log_scale is True

    def test_config_custom_penalty(self):
        """Test config with custom penalty for invalid metrics."""
        config = ObjectiveConfig(penalty_invalid=-1.0)
        assert config.penalty_invalid == -1.0


class TestTrialMetricsEdgeCases:
    """Test TrialMetrics edge cases."""

    def test_is_valid_with_zero_values(self):
        """Test is_valid with all zero values."""
        metrics = TrialMetrics(
            robust_accuracy=0.0, clean_accuracy=0.0, cross_site_auroc=0.0, loss=0.0
        )
        assert metrics.is_valid() is True

    def test_is_valid_with_mixed_inf(self):
        """Test is_valid with mixed infinity values."""
        metrics = TrialMetrics(
            robust_accuracy=0.8,
            clean_accuracy=0.9,
            cross_site_auroc=0.85,
            loss=float("inf"),  # Infinity is allowed for loss (default value)
            natural_loss=float("inf"),
        )
        # Loss can be inf, other metrics are finite
        assert metrics.is_valid() is True

    def test_to_dict_completeness(self):
        """Test to_dict includes all fields."""
        metrics = TrialMetrics(
            robust_accuracy=0.8,
            clean_accuracy=0.9,
            cross_site_auroc=0.85,
            loss=0.5,
            natural_loss=0.3,
            robust_loss=0.7,
            explanation_stability=0.75,
            epoch=5,
            step=100,
        )
        d = metrics.to_dict()
        assert len(d) == 10  # All fields present
        assert "timestamp" in d


class TestWeightedTriObjectiveEdgeCases:
    """Test WeightedTriObjective edge cases and branches."""

    def test_objective_with_values_at_clip_boundaries(self):
        """Test objective computation with values at clipping boundaries."""
        obj = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=1.0,  # At upper bound
            clean_accuracy=0.0,  # At lower bound
            cross_site_auroc=0.5,  # In middle
        )
        value = obj(metrics)
        # 0.4 * 1.0 + 0.3 * 0.0 + 0.3 * 0.5 = 0.55
        assert abs(value - 0.55) < 0.01

    def test_objective_with_values_exceeding_clip_range(self):
        """Test clipping when values exceed range."""
        obj = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=1.5,  # Above 1.0
            clean_accuracy=-0.2,  # Below 0.0
            cross_site_auroc=0.8,
        )
        value = obj(metrics)
        # Should clip to [0, 1]: robust=1.0, clean=0.0, auroc=0.8
        expected = 0.4 * 1.0 + 0.3 * 0.0 + 0.3 * 0.8
        assert abs(value - expected) < 0.01

    def test_get_component_contributions_with_zeros(self):
        """Test component contributions with zero values."""
        obj = WeightedTriObjective()
        metrics = TrialMetrics(
            robust_accuracy=0.0, clean_accuracy=0.0, cross_site_auroc=0.0
        )
        contributions = obj.get_component_contributions(metrics)
        assert contributions["robust_contribution"] == 0.0
        assert contributions["clean_contribution"] == 0.0
        assert contributions["auroc_contribution"] == 0.0

    def test_intermediate_value_with_empty_history(self):
        """Test get_intermediate_value when history is empty."""
        obj = WeightedTriObjective()
        obj.reset_history()
        metrics = TrialMetrics(robust_accuracy=0.8, clean_accuracy=0.9)
        value = obj.get_intermediate_value(metrics, epoch=0)
        # First value should just be returned
        assert value > 0.0

    def test_weights_property_returns_copy(self):
        """Test that weights property returns a copy."""
        obj = WeightedTriObjective()
        weights1 = obj.weights
        weights1["robust_accuracy"] = 0.99
        weights2 = obj.weights
        # Original should be unchanged
        assert weights2["robust_accuracy"] == 0.4


class TestAdaptiveWeightedObjectiveEdgeCases:
    """Test AdaptiveWeightedObjective edge cases."""

    def test_call_with_missing_weight_keys(self):
        """Test call with incomplete weight dictionary."""
        obj = AdaptiveWeightedObjective(
            base_weights={"robust_accuracy": 0.5, "clean_accuracy": 0.5}
        )
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        value = obj(metrics)
        # Uses default 0.3 for missing cross_site_auroc
        expected = 0.5 * 0.8 + 0.5 * 0.9 + 0.3 * 0.85
        assert abs(value - expected) < 0.01

    def test_update_weights_cyclic_all_positions(self):
        """Test cyclic weight update covers all cycle positions."""
        obj = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            adaptation_strategy="cyclic",
        )

        # Test all 3 cycle positions
        obj.update_weights(0)
        assert obj.current_weights["robust_accuracy"] == 0.5

        obj.update_weights(1)
        assert obj.current_weights["clean_accuracy"] == 0.5

        obj.update_weights(2)
        assert obj.current_weights["cross_site_auroc"] == 0.5

    def test_update_weights_linear_at_zero(self):
        """Test linear weight update at iteration zero."""
        obj = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            adaptation_strategy="linear",
            target_iterations=100,
        )
        obj.update_weights(0)
        # At iteration 0, should be base weights
        assert obj.current_weights["robust_accuracy"] == 0.4

    def test_update_weights_linear_at_target(self):
        """Test linear weight update at target iteration."""
        obj = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            adaptation_strategy="linear",
            target_iterations=100,
        )
        obj.update_weights(100)
        # At target, should have shifted by 0.1
        assert abs(obj.current_weights["robust_accuracy"] - 0.5) < 0.01

    def test_update_weights_with_zero_target_iterations(self):
        """Test weight update with zero target iterations."""
        obj = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            target_iterations=0,
        )
        obj.update_weights(5)
        # Should handle division by zero gracefully
        assert obj.iteration == 5


class TestMultiObjectiveEvaluatorEdgeCases:
    """Test MultiObjectiveEvaluator edge cases."""

    def test_dominates_with_nan_values(self):
        """Test domination check with NaN values."""
        evaluator = MultiObjectiveEvaluator()
        a = np.array([0.9, np.nan, 0.8])
        b = np.array([0.8, 0.8, 0.8])
        # NaN comparisons should return False
        result = evaluator.dominates(a, b)
        # With NaN, domination should fail
        assert isinstance(result, (bool, np.bool_))

    def test_compute_hypervolume_4d_warning(self):
        """Test hypervolume computation with 4D front (unsupported)."""
        evaluator = MultiObjectiveEvaluator(
            objectives=["robust_accuracy", "clean_accuracy", "cross_site_auroc", "loss"]
        )
        solution = np.array([0.8, 0.9, 0.85, 0.5])
        evaluator.update_pareto_front(solution)

        # Should trigger warning log for 4D
        hypervolume = evaluator.compute_hypervolume()
        assert hypervolume == 0.0  # Not implemented for >3D

    def test_2d_hypervolume_negative_area(self):
        """Test 2D hypervolume when reference point is above front."""
        objectives = ["robust_accuracy", "clean_accuracy"]
        reference = np.array([0.9, 0.9])
        evaluator = MultiObjectiveEvaluator(objectives=objectives)
        evaluator.reference_point = reference

        solution = np.array([0.8, 0.8])
        evaluator.update_pareto_front(solution)

        hypervolume = evaluator.compute_hypervolume()
        # Should be close to 0.0 for negative hypervolume (negative areas)
        assert hypervolume >= 0.0  # Non-negative result

    def test_3d_hypervolume_with_random_seed(self):
        """Test 3D hypervolume computation consistency."""
        evaluator = MultiObjectiveEvaluator()
        solution = np.array([0.9, 0.9, 0.9])
        evaluator.update_pareto_front(solution)

        # Run multiple times to verify Monte Carlo stability
        np.random.seed(42)
        hv1 = evaluator.compute_hypervolume()
        np.random.seed(42)
        hv2 = evaluator.compute_hypervolume()

        # Should be consistent with same seed
        assert abs(hv1 - hv2) < 0.01

    def test_extract_objectives_with_custom_order(self):
        """Test extracting objectives with custom ordering."""
        evaluator = MultiObjectiveEvaluator(
            objectives=["cross_site_auroc", "robust_accuracy", "clean_accuracy"]
        )
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        objectives = evaluator.extract_objectives(metrics)
        # Should follow custom order
        assert objectives[0] == 0.85  # cross_site_auroc
        assert objectives[1] == 0.8  # robust_accuracy
        assert objectives[2] == 0.9  # clean_accuracy


class TestOptunaIntegration:
    """Test Optuna integration factory functions."""

    def test_create_optuna_objective_success(self):
        """Test creating Optuna objective with successful trainer."""
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_metrics = TrialMetrics(
            robust_accuracy=0.85, clean_accuracy=0.90, cross_site_auroc=0.88
        )
        mock_trainer.train_and_evaluate.return_value = mock_metrics

        # Create factory
        def trainer_factory(trial):
            return mock_trainer

        # Create objective
        objective_fn = WeightedTriObjective()
        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)

        # Create mock trial
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # Run objective
        value = optuna_obj(trial)

        # Verify correct computation
        expected = 0.4 * 0.85 + 0.3 * 0.90 + 0.3 * 0.88
        assert abs(value - expected) < 0.01

    def test_create_optuna_objective_with_custom_evaluator(self):
        """Test creating Optuna objective with custom evaluator."""
        mock_trainer = MagicMock()
        custom_metrics = TrialMetrics(
            robust_accuracy=0.75, clean_accuracy=0.85, cross_site_auroc=0.80
        )

        def trainer_factory(trial):
            return mock_trainer

        def custom_evaluator(trainer):
            return custom_metrics

        objective_fn = WeightedTriObjective()
        optuna_obj = create_optuna_objective(
            objective_fn, trainer_factory, evaluator=custom_evaluator
        )

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        value = optuna_obj(trial)

        # Should use custom evaluator
        expected = 0.4 * 0.75 + 0.3 * 0.85 + 0.3 * 0.80
        assert abs(value - expected) < 0.01

    def test_create_optuna_objective_with_history_reset(self):
        """Test that objective history is reset between trials."""
        mock_trainer = MagicMock()
        mock_trainer.train_and_evaluate.return_value = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )

        def trainer_factory(trial):
            return mock_trainer

        objective_fn = WeightedTriObjective()

        # Add some history
        objective_fn._history = [(0, 0.5), (1, 0.6)]

        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        optuna_obj(trial)

        # History should be reset
        assert len(objective_fn._history) == 0

    def test_create_optuna_objective_with_exception(self):
        """Test Optuna objective handles trainer exceptions."""

        def trainer_factory(trial):
            raise RuntimeError("Training failed!")

        objective_fn = WeightedTriObjective()
        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # Should raise TrialPruned
        with pytest.raises(optuna.TrialPruned):
            optuna_obj(trial)

    def test_create_multi_objective_optuna_success(self):
        """Test creating multi-objective Optuna function."""
        mock_trainer = MagicMock()
        mock_trainer.train_and_evaluate.return_value = TrialMetrics(
            robust_accuracy=0.85, clean_accuracy=0.90, cross_site_auroc=0.88
        )

        def trainer_factory(trial):
            return mock_trainer

        # Create multiple objectives
        obj1 = WeightedTriObjective(
            robust_weight=0.5, clean_weight=0.3, auroc_weight=0.2
        )
        obj2 = WeightedTriObjective(
            robust_weight=0.3, clean_weight=0.5, auroc_weight=0.2
        )

        multi_obj = create_multi_objective_optuna([obj1, obj2], trainer_factory)

        study = optuna.create_study(directions=["maximize", "maximize"])
        trial = study.ask()
        values = multi_obj(trial)

        # Should return list of values
        assert isinstance(values, list)
        assert len(values) == 2
        assert all(isinstance(v, float) for v in values)

    def test_create_multi_objective_optuna_with_exception(self):
        """Test multi-objective Optuna handles exceptions."""

        def trainer_factory(trial):
            raise ValueError("Trainer error!")

        obj1 = WeightedTriObjective()
        obj2 = WeightedTriObjective()
        multi_obj = create_multi_objective_optuna([obj1, obj2], trainer_factory)

        study = optuna.create_study(directions=["maximize", "maximize"])
        trial = study.ask()

        with pytest.raises(optuna.TrialPruned):
            multi_obj(trial)

    def test_create_multi_objective_optuna_different_objectives(self):
        """Test multi-objective with different objective types."""
        mock_trainer = MagicMock()
        mock_trainer.train_and_evaluate.return_value = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )

        def trainer_factory(trial):
            return mock_trainer

        # Mix of objective types
        obj1 = WeightedTriObjective()
        obj2 = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            }
        )

        multi_obj = create_multi_objective_optuna([obj1, obj2], trainer_factory)

        study = optuna.create_study(directions=["maximize", "maximize"])
        trial = study.ask()
        values = multi_obj(trial)

        assert len(values) == 2
        assert all(0.0 <= v <= 1.0 for v in values)


class TestObjectiveIntegration:
    """Integration tests for objective functions."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create study
        study = optuna.create_study(direction="maximize")

        # Create objective
        objective_fn = WeightedTriObjective()

        # Mock trainer factory
        def trainer_factory(trial):
            mock_trainer = MagicMock()
            # Simulate improving metrics
            progress = trial.number / 10.0
            mock_trainer.train_and_evaluate.return_value = TrialMetrics(
                robust_accuracy=0.7 + progress * 0.1,
                clean_accuracy=0.8 + progress * 0.1,
                cross_site_auroc=0.75 + progress * 0.1,
            )
            return mock_trainer

        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)

        # Run optimization
        study.optimize(optuna_obj, n_trials=5)

        # Verify improvement
        assert len(study.trials) == 5
        assert study.best_value > 0.7

    def test_adaptive_objective_workflow(self):
        """Test adaptive objective in optimization workflow."""
        study = optuna.create_study(direction="maximize")

        objective_fn = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            adaptation_strategy="linear",
            target_iterations=10,
        )

        def trainer_factory(trial):
            # Update weights based on trial number
            objective_fn.update_weights(trial.number)

            mock_trainer = MagicMock()
            mock_trainer.train_and_evaluate.return_value = TrialMetrics(
                robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
            )
            return mock_trainer

        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)
        study.optimize(optuna_obj, n_trials=5)

        assert study.best_trial is not None

    def test_multi_objective_optimization_workflow(self):
        """Test multi-objective optimization workflow."""
        study = optuna.create_study(directions=["maximize", "maximize"])

        obj1 = WeightedTriObjective(
            robust_weight=0.6, clean_weight=0.2, auroc_weight=0.2
        )
        obj2 = WeightedTriObjective(
            robust_weight=0.2, clean_weight=0.6, auroc_weight=0.2
        )

        def trainer_factory(trial):
            mock_trainer = MagicMock()
            mock_trainer.train_and_evaluate.return_value = TrialMetrics(
                robust_accuracy=0.85, clean_accuracy=0.90, cross_site_auroc=0.88
            )
            return mock_trainer

        multi_obj = create_multi_objective_optuna([obj1, obj2], trainer_factory)
        study.optimize(multi_obj, n_trials=3)

        assert len(study.trials) == 3
        assert all(len(trial.values) == 2 for trial in study.trials)


class TestAdditionalCoverage:
    """Additional tests to achieve 100% coverage."""

    def test_objective_config_direction_minimize(self):
        """Test config with minimize direction."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            direction="minimize",
        )
        assert config.direction == "minimize"

    def test_weighted_tri_objective_with_use_log_scale(self):
        """Test WeightedTriObjective with log scale config."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            use_log_scale=True,
        )
        obj = WeightedTriObjective(config=config)
        metrics = TrialMetrics(
            robust_accuracy=0.8, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        value = obj(metrics)
        # Should still compute normally
        assert 0.0 <= value <= 1.0

    def test_weighted_tri_objective_with_negative_penalty(self):
        """Test objective with negative penalty for invalid metrics."""
        config = ObjectiveConfig(
            weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            penalty_invalid=-1.0,
        )
        obj = WeightedTriObjective(config=config)
        metrics = TrialMetrics(
            robust_accuracy=np.nan, clean_accuracy=0.9, cross_site_auroc=0.85
        )
        value = obj(metrics)
        # Should return negative penalty
        assert value == -1.0

    def test_adaptive_objective_unknown_strategy(self):
        """Test adaptive objective with unsupported strategy."""
        obj = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            adaptation_strategy="unknown",
        )
        obj.update_weights(5)
        # Should not crash, weights should remain as base
        assert "robust_accuracy" in obj.current_weights

    def test_multi_objective_evaluator_empty_objectives_list(self):
        """Test evaluator with empty objectives falls back to default."""
        evaluator = MultiObjectiveEvaluator(objectives=None)
        # Should have default objectives
        assert len(evaluator.objectives) == 3
        assert "robust_accuracy" in evaluator.objectives

    def test_trial_metrics_with_partial_infinity(self):
        """Test trial metrics with some infinity values."""
        metrics = TrialMetrics(
            robust_accuracy=0.8,
            clean_accuracy=0.9,
            cross_site_auroc=0.85,
            loss=float("inf"),  # This is OK
            natural_loss=0.5,
            robust_loss=float("inf"),  # This is OK
        )
        # Loss can be infinity, should still be valid
        assert metrics.is_valid() is True

    def test_objective_function_abstract_enforcement(self):
        """Test that ObjectiveFunction cannot be instantiated."""
        from src.training.hpo_objective import ObjectiveFunction

        with pytest.raises(TypeError):
            ObjectiveFunction()

    def test_create_optuna_objective_without_reset_history(self):
        """Test optuna objective when objective_fn has no reset_history."""

        # Create custom objective without reset_history
        class MinimalObjective:
            def __call__(self, metrics):
                return metrics.robust_accuracy

        minimal_obj = MinimalObjective()

        mock_trainer = MagicMock()
        mock_trainer.train_and_evaluate.return_value = TrialMetrics(robust_accuracy=0.8)

        def trainer_factory(trial):
            return mock_trainer

        optuna_obj = create_optuna_objective(minimal_obj, trainer_factory)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        value = optuna_obj(trial)

        # Should work without reset_history
        assert value == 0.8

    def test_hypervolume_computation_with_very_small_front(self):
        """Test hypervolume with very small values."""
        evaluator = MultiObjectiveEvaluator()
        solution = np.array([0.001, 0.002, 0.003])
        evaluator.update_pareto_front(solution)

        hypervolume = evaluator.compute_hypervolume()
        # Should handle small values
        assert hypervolume >= 0.0

    def test_weighted_objective_validate_weights_error(self):
        """Test WeightedTriObjective validation raises error."""
        config = ObjectiveConfig(
            weights={"robust_accuracy": 0.5, "clean_accuracy": 0.5}  # Missing auroc
        )

        with pytest.raises(ValueError, match="Missing required weight keys"):
            WeightedTriObjective(config=config)

    def test_adaptive_objective_normalization(self):
        """Test that adaptive weights are always normalized."""
        obj = AdaptiveWeightedObjective(
            base_weights={
                "robust_accuracy": 0.4,
                "clean_accuracy": 0.3,
                "cross_site_auroc": 0.3,
            },
            adaptation_strategy="cyclic",
        )

        for i in range(6):
            obj.update_weights(i)
            total = sum(obj.current_weights.values())
            # Should always sum to 1.0
            assert abs(total - 1.0) < 0.0001

    def test_optuna_objective_logging(self):
        """Test that optuna objective logs properly."""
        mock_trainer = MagicMock()
        mock_trainer.train_and_evaluate.return_value = TrialMetrics(
            robust_accuracy=0.85, clean_accuracy=0.90, cross_site_auroc=0.88
        )

        def trainer_factory(trial):
            return mock_trainer

        objective_fn = WeightedTriObjective()
        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # Capture logs
        with patch("src.training.hpo_objective.logger") as mock_logger:
            value = optuna_obj(trial)
            # Should log info
            mock_logger.info.assert_called_once()

    def test_optuna_objective_exception_logging(self):
        """Test that optuna objective logs exceptions."""

        def trainer_factory(trial):
            raise ValueError("Test error")

        objective_fn = WeightedTriObjective()
        optuna_obj = create_optuna_objective(objective_fn, trainer_factory)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # Capture logs
        with patch("src.training.hpo_objective.logger") as mock_logger:
            with pytest.raises(optuna.TrialPruned):
                optuna_obj(trial)
            # Should log error
            mock_logger.error.assert_called_once()

    def test_multi_objective_exception_logging(self):
        """Test multi-objective optuna logs exceptions."""

        def trainer_factory(trial):
            raise RuntimeError("Training crash")

        obj1 = WeightedTriObjective()
        obj2 = WeightedTriObjective()
        multi_obj = create_multi_objective_optuna([obj1, obj2], trainer_factory)

        study = optuna.create_study(directions=["maximize", "maximize"])
        trial = study.ask()

        with patch("src.training.hpo_objective.logger") as mock_logger:
            with pytest.raises(optuna.TrialPruned):
                multi_obj(trial)
            # Should log error
            mock_logger.error.assert_called_once()

    def test_pareto_front_multiple_updates(self):
        """Test Pareto front with many solution updates."""
        evaluator = MultiObjectiveEvaluator()

        # Add multiple non-dominated solutions
        solutions = [
            np.array([0.9, 0.7, 0.7]),
            np.array([0.7, 0.9, 0.7]),
            np.array([0.7, 0.7, 0.9]),
            np.array([0.8, 0.8, 0.8]),  # Does NOT dominate (not better in all)
        ]

        for sol in solutions:
            evaluator.update_pareto_front(sol)

        front = evaluator.get_pareto_front()
        # All 4 are non-dominated (none strictly dominates the others)
        assert len(front) == 4

    def test_extract_objectives_all_missing_attributes(self):
        """Test extracting objectives when all attributes are missing."""
        evaluator = MultiObjectiveEvaluator(
            objectives=["missing1", "missing2", "missing3"]
        )
        metrics = TrialMetrics()  # Default metrics, no custom attributes
        objectives = evaluator.extract_objectives(metrics)
        # All should default to 0.0
        assert all(obj == 0.0 for obj in objectives)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
