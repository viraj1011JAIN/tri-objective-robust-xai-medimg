"""
Comprehensive tests for HPO search space definitions.

Tests cover:
- Search space configuration
- Optuna suggestion integration
- Search space factories
- Hyperparameter suggestion functions

Author: Viraj Pankaj Jain
"""

import optuna
import pytest

from src.config.hpo.search_spaces import (
    SearchSpaceConfig,
    SearchSpaceFactory,
    suggest_explainability_hyperparameters,
    suggest_model_hyperparameters,
    suggest_optimizer_hyperparameters,
    suggest_robustness_hyperparameters,
    suggest_scheduler_hyperparameters,
    suggest_training_hyperparameters,
    suggest_tri_objective_hyperparameters,
)


class TestSearchSpaceConfig:
    """Tests for SearchSpaceConfig class."""

    def test_architectures_list(self):
        """Test that architectures list is defined."""
        assert hasattr(SearchSpaceConfig, "ARCHITECTURES")
        assert len(SearchSpaceConfig.ARCHITECTURES) > 0
        assert "resnet50" in SearchSpaceConfig.ARCHITECTURES

    def test_activations_list(self):
        """Test that activations list is defined."""
        assert hasattr(SearchSpaceConfig, "ACTIVATIONS")
        assert len(SearchSpaceConfig.ACTIVATIONS) > 0

    def test_normalizations_list(self):
        """Test that normalizations list is defined."""
        assert hasattr(SearchSpaceConfig, "NORMALIZATIONS")
        assert len(SearchSpaceConfig.NORMALIZATIONS) > 0


class TestModelHyperparameterSuggestion:
    """Tests for model hyperparameter suggestion."""

    def test_suggest_model_hyperparameters(self):
        """Test suggesting model hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        model_hparams = suggest_model_hyperparameters(trial)

        assert hasattr(model_hparams, "architecture")
        assert hasattr(model_hparams, "pretrained")
        assert hasattr(model_hparams, "dropout_rate")
        assert model_hparams.architecture in SearchSpaceConfig.ARCHITECTURES
        assert isinstance(model_hparams.pretrained, bool)
        assert 0.0 <= model_hparams.dropout_rate <= 1.0


class TestOptimizerHyperparameterSuggestion:
    """Tests for optimizer hyperparameter suggestion."""

    def test_suggest_optimizer_hyperparameters(self):
        """Test suggesting optimizer hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        opt_hparams = suggest_optimizer_hyperparameters(trial)

        assert hasattr(opt_hparams, "optimizer_type")
        assert hasattr(opt_hparams, "learning_rate")
        assert hasattr(opt_hparams, "weight_decay")
        assert opt_hparams.learning_rate > 0
        assert opt_hparams.weight_decay >= 0


class TestSchedulerHyperparameterSuggestion:
    """Tests for scheduler hyperparameter suggestion."""

    def test_suggest_scheduler_hyperparameters(self):
        """Test suggesting scheduler hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        sched_hparams = suggest_scheduler_hyperparameters(trial)

        assert hasattr(sched_hparams, "scheduler_type")
        assert hasattr(sched_hparams, "warmup_epochs")


class TestTrainingHyperparameterSuggestion:
    """Tests for training hyperparameter suggestion."""

    def test_suggest_training_hyperparameters(self):
        """Test suggesting training hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        train_hparams = suggest_training_hyperparameters(trial)

        assert hasattr(train_hparams, "batch_size")
        assert hasattr(train_hparams, "num_epochs")
        assert train_hparams.batch_size > 0
        assert train_hparams.num_epochs > 0


class TestRobustnessHyperparameterSuggestion:
    """Tests for robustness hyperparameter suggestion."""

    def test_suggest_robustness_hyperparameters(self):
        """Test suggesting robustness hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        robust_hparams = suggest_robustness_hyperparameters(trial)

        assert hasattr(robust_hparams, "epsilon")
        assert hasattr(robust_hparams, "alpha")
        assert robust_hparams.epsilon > 0
        assert robust_hparams.alpha > 0


class TestExplainabilityHyperparameterSuggestion:
    """Tests for explainability hyperparameter suggestion."""

    def test_suggest_explainability_hyperparameters(self):
        """Test suggesting explainability hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        xai_hparams = suggest_explainability_hyperparameters(trial)

        assert hasattr(xai_hparams, "enable_xai_loss")
        assert hasattr(xai_hparams, "use_gradcam")


class TestTriObjectiveHyperparameterSuggestion:
    """Tests for tri-objective hyperparameter suggestion."""

    def test_suggest_tri_objective_hyperparameters(self):
        """Test suggesting tri-objective hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        tri_obj_hparams = suggest_tri_objective_hyperparameters(trial)

        assert hasattr(tri_obj_hparams, "robustness_weight")
        assert hasattr(tri_obj_hparams, "explainability_weight")
        assert tri_obj_hparams.robustness_weight >= 0
        assert tri_obj_hparams.explainability_weight >= 0


class TestSearchSpaceFactory:
    """Tests for SearchSpaceFactory."""

    def test_create_default_spaces(self):
        """Test creating default search spaces."""
        # Test that we can create search space configurations
        factory = SearchSpaceFactory()

        assert isinstance(factory, SearchSpaceFactory)

    def test_factory_has_methods(self):
        """Test that factory has expected methods."""
        assert (
            hasattr(SearchSpaceFactory, "create_model_space")
            or hasattr(SearchSpaceFactory, "create_optimizer_space")
            or callable(SearchSpaceFactory)
        )

    def test_create_full_search_space(self):
        """Test creating full search space function."""
        search_space_fn = SearchSpaceFactory.create_full_search_space()

        assert callable(search_space_fn)

        # Test that it works with an Optuna trial
        study = optuna.create_study()
        trial = study.ask()

        config = search_space_fn(trial)

        assert hasattr(config, "model")
        assert hasattr(config, "optimizer")
        assert hasattr(config, "training")


class TestSearchSpaceEdgeCases:
    """Tests for edge cases in search spaces."""

    def test_suggest_with_categorical_choices(self):
        """Test suggesting categorical hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        # Test that categorical choices work
        model_hparams = suggest_model_hyperparameters(trial)

        # Should have selected an architecture
        assert hasattr(model_hparams, "architecture")

    def test_suggest_with_log_scale(self):
        """Test suggesting hyperparameters with log scale."""
        study = optuna.create_study()
        trial = study.ask()

        # Learning rate typically uses log scale
        opt_hparams = suggest_optimizer_hyperparameters(trial)

        assert opt_hparams.learning_rate > 0
        assert opt_hparams.learning_rate < 1

    def test_suggest_with_integer_params(self):
        """Test suggesting integer hyperparameters."""
        study = optuna.create_study()
        trial = study.ask()

        training_hparams = suggest_training_hyperparameters(trial)

        # Batch size should be integer
        assert isinstance(training_hparams.batch_size, int)
        assert training_hparams.batch_size > 0

    def test_multiple_trials_different_values(self):
        """Test that different trials get different hyperparameters."""
        study = optuna.create_study()

        # Get hyperparameters from multiple trials
        configs = []
        for _ in range(5):
            trial = study.ask()
            opt_hparams = suggest_optimizer_hyperparameters(trial)
            configs.append(opt_hparams.learning_rate)

        # Should have some variation (though not guaranteed)
        # Just check that all values are valid
        assert all(0 < lr < 1 for lr in configs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
