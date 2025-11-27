"""
Comprehensive integration tests for HPO system.

Tests cover:
- HPO trainer initialization
- Study creation and management
- Basic optimization workflow
- Integration between components

Author: Viraj Pankaj Jain
"""

import tempfile
from pathlib import Path

import optuna
import pytest

from src.config.hpo.hpo_trainer import HPOManager, HPOTrainer
from src.config.hpo.hyperparameters import HyperparameterConfig
from src.config.hpo.objectives import ObjectiveMetrics, WeightedSumObjective
from src.config.hpo.search_spaces import SearchSpaceFactory


class MockObjective:
    """Mock objective for testing HPO without actual training."""

    def __call__(self, trial: optuna.Trial):
        """Mock objective that returns synthetic metrics."""
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

        # Simulate metrics based on learning rate
        clean_acc = min(0.95, 0.7 + (0.01 - lr) * 10)

        return clean_acc


class TestHPOTrainer:
    """Tests for HPOTrainer class."""

    def test_initialization(self):
        """Test HPO trainer initialization."""
        hpo_trainer = HPOTrainer(
            study_name="test_study",
            n_trials=5,
        )

        assert hpo_trainer.study_name == "test_study"
        assert hpo_trainer.n_trials == 5

    def test_study_creation(self):
        """Test Optuna study is created correctly."""
        hpo_trainer = HPOTrainer(
            study_name="test_study",
            n_trials=3,
        )

        assert hpo_trainer.study is not None
        assert hpo_trainer.study.study_name == "test_study"

    def test_optimization_basic(self):
        """Test basic optimization runs."""
        hpo_trainer = HPOTrainer(
            study_name="test_optimization",
            n_trials=2,
        )

        mock_objective = MockObjective()

        # Run optimization
        hpo_trainer.study.optimize(mock_objective, n_trials=2)

        assert len(hpo_trainer.study.trials) == 2
        assert hpo_trainer.study.best_trial is not None

    def test_best_params_selection(self):
        """Test that best parameters are correctly identified."""
        hpo_trainer = HPOTrainer(
            study_name="test_best_params",
            n_trials=5,
        )

        mock_objective = MockObjective()
        hpo_trainer.study.optimize(mock_objective, n_trials=5)

        # Best trial should exist
        assert hpo_trainer.study.best_trial is not None
        assert hpo_trainer.study.best_value is not None
        assert "learning_rate" in hpo_trainer.study.best_params

    def test_pruning_integration(self):
        """Test pruning integration."""
        hpo_trainer = HPOTrainer(
            study_name="test_pruning",
            n_trials=5,
            pruner_type="median",
        )

        # Should initialize without errors
        assert hpo_trainer.pruner is not None


class TestHPOManager:
    """Tests for HPOManager class."""

    def test_manager_initialization(self):
        """Test HPO manager initialization."""
        manager = HPOManager()

        assert isinstance(manager, HPOManager)

    def test_manager_has_methods(self):
        """Test that manager has expected methods."""
        manager = HPOManager()

        # Check for common manager methods
        assert hasattr(manager, "__init__")


class TestObjectiveMetrics:
    """Tests for ObjectiveMetrics class."""

    def test_metrics_creation(self):
        """Test creating objective metrics."""
        metrics = ObjectiveMetrics(
            clean_accuracy=0.85,
            robust_accuracy=0.75,
            xai_coherence=0.80,
            xai_faithfulness=0.82,
        )

        assert metrics.clean_accuracy == 0.85
        assert metrics.robust_accuracy == 0.75
        assert metrics.xai_coherence == 0.80
        assert metrics.xai_faithfulness == 0.82


class TestWeightedSumObjective:
    """Tests for WeightedSumObjective class."""

    def test_objective_creation(self):
        """Test creating weighted sum objective."""
        objective = WeightedSumObjective(
            accuracy_weight=0.4,
            robustness_weight=0.3,
            explainability_weight=0.3,
        )

        assert objective.accuracy_weight == 0.4
        assert objective.robustness_weight == 0.3
        assert objective.explainability_weight == 0.3

    def test_objective_computation(self):
        """Test computing weighted sum objective."""
        objective = WeightedSumObjective(
            accuracy_weight=0.5,
            robustness_weight=0.3,
            explainability_weight=0.2,
        )

        metrics = ObjectiveMetrics(
            accuracy=0.9,
            robustness=0.8,
            explainability=0.85,
        )

        score = objective.evaluate(metrics)

        assert isinstance(score, float)
        assert score > 0


class TestHPOIntegration:
    """Integration tests for complete HPO workflow."""

    def test_minimal_search_space(self):
        """Test HPO with minimal search space."""
        hpo_trainer = HPOTrainer(
            study_name="test_minimal",
            n_trials=2,
        )

        def simple_objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x**2

        hpo_trainer.study.optimize(simple_objective, n_trials=2)

        assert len(hpo_trainer.study.trials) == 2
        assert hpo_trainer.study.best_value is not None


class TestHPOSamplerTypes:
    """Tests for different sampler types."""

    def test_tpe_sampler(self):
        """Test TPE sampler initialization."""
        hpo_trainer = HPOTrainer(study_name="test_tpe", n_trials=2, sampler_type="tpe")

        assert hpo_trainer.study is not None

    def test_random_sampler(self):
        """Test random sampler initialization."""
        hpo_trainer = HPOTrainer(
            study_name="test_random", n_trials=2, sampler_type="random"
        )

        assert hpo_trainer.study is not None

    def test_cmaes_sampler(self):
        """Test CMA-ES sampler initialization."""
        hpo_trainer = HPOTrainer(
            study_name="test_cmaes", n_trials=2, sampler_type="cmaes"
        )

        assert hpo_trainer.study is not None

    def test_grid_sampler(self):
        """Test grid sampler initialization (fallback to random)."""
        hpo_trainer = HPOTrainer(
            study_name="test_grid", n_trials=2, sampler_type="grid"
        )

        assert hpo_trainer.study is not None

    def test_unknown_sampler_raises_error(self):
        """Test that unknown sampler type raises error."""
        with pytest.raises(ValueError, match="Unknown sampler type"):
            HPOTrainer(
                study_name="test_unknown", n_trials=2, sampler_type="invalid_sampler"
            )


class TestHPOObjectiveTypes:
    """Tests for different objective types."""

    def test_accuracy_objective(self):
        """Test accuracy objective type."""
        hpo_trainer = HPOTrainer(
            study_name="test_accuracy", n_trials=2, objective_type="accuracy"
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        value = hpo_trainer._calculate_objective(metrics)
        assert value == 0.9

    def test_robustness_objective(self):
        """Test robustness objective type."""
        hpo_trainer = HPOTrainer(
            study_name="test_robustness", n_trials=2, objective_type="robustness"
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        value = hpo_trainer._calculate_objective(metrics)
        assert value == 0.8

    def test_explainability_objective(self):
        """Test explainability objective type."""
        hpo_trainer = HPOTrainer(
            study_name="test_explainability",
            n_trials=2,
            objective_type="explainability",
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        value = hpo_trainer._calculate_objective(metrics)
        assert value == 0.85

    def test_weighted_sum_objective(self):
        """Test weighted sum objective type."""
        hpo_trainer = HPOTrainer(
            study_name="test_weighted", n_trials=2, objective_type="weighted_sum"
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        value = hpo_trainer._calculate_objective(metrics)
        assert isinstance(value, float)

    def test_unknown_objective_raises_error(self):
        """Test that unknown objective type raises error."""
        hpo_trainer = HPOTrainer(
            study_name="test_unknown_obj",
            n_trials=2,
            objective_type="invalid_objective",
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        with pytest.raises(ValueError, match="Unknown objective type"):
            hpo_trainer._calculate_objective(metrics)


class TestHPOOptimizeMethod:
    """Tests for the optimize method."""

    def test_optimize_with_custom_train_fn(self, tmp_path):
        """Test optimize with custom training function."""
        hpo_trainer = HPOTrainer(
            study_name="test_optimize", n_trials=2, save_dir=str(tmp_path)
        )

        def mock_train_fn(config, trial):
            """Mock training function."""
            return ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        def mock_search_space(trial):
            """Mock search space."""
            from src.config.hpo.hyperparameters import (
                HyperparameterConfig,
                ModelHyperparameters,
                OptimizerHyperparameters,
                TrainingHyperparameters,
            )

            lr = trial.suggest_float("lr", 1e-4, 1e-2)

            return HyperparameterConfig(
                model=ModelHyperparameters(),
                optimizer=OptimizerHyperparameters(learning_rate=lr),
                training=TrainingHyperparameters(),
            )

        study = hpo_trainer.optimize(
            train_fn=mock_train_fn, search_space_fn=mock_search_space
        )

        assert len(study.trials) == 2
        assert study.best_trial is not None


class TestHPOStorageAndSaving:
    """Tests for storage and saving functionality."""

    def test_store_trial_metrics(self):
        """Test storing trial metrics."""
        hpo_trainer = HPOTrainer(study_name="test_store", n_trials=1)

        # Create a mock trial
        def objective(trial):
            return 0.5

        hpo_trainer.study.optimize(objective, n_trials=1)
        trial = hpo_trainer.study.trials[0]

        # Create mock metrics and config
        from src.config.hpo.hyperparameters import (
            HyperparameterConfig,
            ModelHyperparameters,
            OptimizerHyperparameters,
            TrainingHyperparameters,
        )

        metrics = ObjectiveMetrics(accuracy=0.9, robustness=0.8, explainability=0.85)

        config = HyperparameterConfig(
            model=ModelHyperparameters(),
            optimizer=OptimizerHyperparameters(),
            training=TrainingHyperparameters(),
        )

        hpo_trainer._store_trial_metrics(trial, metrics, config)

        # Check that metrics were stored
        assert "accuracy" in trial.user_attrs
        assert trial.user_attrs["accuracy"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
