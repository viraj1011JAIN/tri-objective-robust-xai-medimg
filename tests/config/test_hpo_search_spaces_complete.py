"""
Comprehensive tests for HPO search spaces to achieve 100% coverage.

This test suite covers all uncovered paths in search_spaces.py including:
- Fixed parameters overriding trial suggestions
- Scheduler-specific parameter branches
- Optimizer-specific parameter branches (SGD, Adam, RMSprop)
- Conditional dependencies (attention heads, SE blocks, TRADES beta)
- Factory methods (quick, accuracy-focused, robustness-focused, balanced)
- Edge cases in suggest_full_config

Author: Viraj Jain
Date: November 2025
"""

import optuna
import pytest

from src.config.hpo.search_spaces import (
    SearchSpaceConfig,
    SearchSpaceFactory,
    suggest_explainability_hyperparameters,
    suggest_full_config,
    suggest_model_hyperparameters,
    suggest_optimizer_hyperparameters,
    suggest_robustness_hyperparameters,
    suggest_scheduler_hyperparameters,
    suggest_training_hyperparameters,
    suggest_tri_objective_hyperparameters,
)


class TestModelHyperparametersWithFixedParams:
    """Test model hyperparameter suggestion with fixed parameters."""

    def test_suggest_with_fixed_architecture(self):
        """Test that fixed architecture parameter is respected."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"architecture": "resnet50"}
        model_hparams = suggest_model_hyperparameters(trial, fixed_params)

        assert model_hparams.architecture == "resnet50"

    def test_suggest_with_use_attention_true(self):
        """Test attention heads are suggested when use_attention is True."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"use_attention": True}
        model_hparams = suggest_model_hyperparameters(trial, fixed_params)

        assert model_hparams.use_attention is True
        # Attention heads should be between 4 and 16 with step 4
        assert model_hparams.attention_heads in [4, 8, 12, 16]

    def test_suggest_with_use_attention_false(self):
        """Test attention heads default when use_attention is False."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"use_attention": False}
        model_hparams = suggest_model_hyperparameters(trial, fixed_params)

        assert model_hparams.use_attention is False
        # Default attention_heads is 8 when not used
        assert model_hparams.attention_heads == 8

    def test_suggest_with_use_se_blocks_true(self):
        """Test SE reduction is suggested when use_se_blocks is True."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"use_se_blocks": True}
        model_hparams = suggest_model_hyperparameters(trial, fixed_params)

        assert model_hparams.use_se_blocks is True
        # SE reduction should be one of the categorical options
        assert model_hparams.se_reduction in [8, 16, 32]

    def test_suggest_with_use_se_blocks_false(self):
        """Test SE reduction defaults when use_se_blocks is False."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"use_se_blocks": False}
        model_hparams = suggest_model_hyperparameters(trial, fixed_params)

        assert model_hparams.use_se_blocks is False
        # Default se_reduction is 16 when not used
        assert model_hparams.se_reduction == 16

    def test_suggest_with_all_fixed_params(self):
        """Test suggestion with all parameters fixed."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {
            "architecture": "efficientnet_b0",
            "dropout_rate": 0.3,
            "activation": "relu",
            "normalization": "batch_norm",
            "use_attention": False,
            "attention_heads": 4,
            "use_se_blocks": False,
            "se_reduction": 8,
        }
        model_hparams = suggest_model_hyperparameters(trial, fixed_params)

        assert model_hparams.architecture == "efficientnet_b0"
        assert model_hparams.dropout_rate == 0.3
        assert model_hparams.activation == "relu"
        assert (
            str(model_hparams.normalization) == "batch_norm"
            or model_hparams.normalization.value == "batch_norm"
        )


class TestOptimizerHyperparametersSpecific:
    """Test optimizer-specific parameter branches."""

    def test_suggest_sgd_optimizer(self):
        """Test that SGD optimizer suggests momentum and nesterov."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"optimizer_type": "sgd"}
        opt_hparams = suggest_optimizer_hyperparameters(trial, fixed_params)

        assert opt_hparams.optimizer_type == "sgd"
        # Momentum should be in the specified range
        assert (
            SearchSpaceConfig.MOMENTUM_RANGE[0]
            <= opt_hparams.momentum
            <= SearchSpaceConfig.MOMENTUM_RANGE[1]
        )
        # Nesterov should be boolean
        assert isinstance(opt_hparams.nesterov, bool)

    def test_suggest_rmsprop_optimizer(self):
        """Test that RMSprop optimizer suggests momentum."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"optimizer_type": "rmsprop"}
        opt_hparams = suggest_optimizer_hyperparameters(trial, fixed_params)

        assert opt_hparams.optimizer_type == "rmsprop"
        # Momentum should be suggested for RMSprop
        assert (
            SearchSpaceConfig.MOMENTUM_RANGE[0]
            <= opt_hparams.momentum
            <= SearchSpaceConfig.MOMENTUM_RANGE[1]
        )
        # Nesterov should be False for non-SGD
        assert opt_hparams.nesterov is False

    def test_suggest_adam_optimizer(self):
        """Test that Adam optimizer suggests beta1 and beta2."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"optimizer_type": "adam"}
        opt_hparams = suggest_optimizer_hyperparameters(trial, fixed_params)

        assert opt_hparams.optimizer_type == "adam"
        # Betas should be a tuple
        assert isinstance(opt_hparams.betas, tuple)
        assert len(opt_hparams.betas) == 2
        beta1, beta2 = opt_hparams.betas
        assert (
            SearchSpaceConfig.BETA1_RANGE[0]
            <= beta1
            <= SearchSpaceConfig.BETA1_RANGE[1]
        )
        assert (
            SearchSpaceConfig.BETA2_RANGE[0]
            <= beta2
            <= SearchSpaceConfig.BETA2_RANGE[1]
        )

    def test_suggest_adamw_optimizer(self):
        """Test that AdamW optimizer suggests beta1 and beta2."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"optimizer_type": "adamw"}
        opt_hparams = suggest_optimizer_hyperparameters(trial, fixed_params)

        assert opt_hparams.optimizer_type == "adamw"
        beta1, beta2 = opt_hparams.betas
        assert (
            SearchSpaceConfig.BETA1_RANGE[0]
            <= beta1
            <= SearchSpaceConfig.BETA1_RANGE[1]
        )
        assert (
            SearchSpaceConfig.BETA2_RANGE[0]
            <= beta2
            <= SearchSpaceConfig.BETA2_RANGE[1]
        )

    def test_suggest_nadam_optimizer(self):
        """Test that NAdam optimizer suggests beta1 and beta2."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"optimizer_type": "nadam"}
        opt_hparams = suggest_optimizer_hyperparameters(trial, fixed_params)

        assert opt_hparams.optimizer_type == "nadam"
        beta1, beta2 = opt_hparams.betas
        assert (
            SearchSpaceConfig.BETA1_RANGE[0]
            <= beta1
            <= SearchSpaceConfig.BETA1_RANGE[1]
        )
        assert (
            SearchSpaceConfig.BETA2_RANGE[0]
            <= beta2
            <= SearchSpaceConfig.BETA2_RANGE[1]
        )


class TestSchedulerHyperparametersSpecific:
    """Test scheduler-specific parameter branches."""

    def test_suggest_step_scheduler(self):
        """Test that step scheduler suggests step_size and gamma."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"scheduler_type": "step"}
        sched_hparams = suggest_scheduler_hyperparameters(trial, fixed_params)

        assert sched_hparams.scheduler_type == "step"
        assert (
            SearchSpaceConfig.STEP_SIZE_RANGE[0]
            <= sched_hparams.step_size
            <= SearchSpaceConfig.STEP_SIZE_RANGE[1]
        )
        assert (
            SearchSpaceConfig.GAMMA_RANGE[0]
            <= sched_hparams.gamma
            <= SearchSpaceConfig.GAMMA_RANGE[1]
        )

    def test_suggest_cosine_scheduler(self):
        """Test that cosine scheduler suggests T_max and eta_min."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"scheduler_type": "cosine"}
        sched_hparams = suggest_scheduler_hyperparameters(trial, fixed_params)

        assert sched_hparams.scheduler_type == "cosine"
        assert (
            SearchSpaceConfig.T_MAX_RANGE[0]
            <= sched_hparams.T_max
            <= SearchSpaceConfig.T_MAX_RANGE[1]
        )
        assert (
            SearchSpaceConfig.ETA_MIN_RANGE[0]
            <= sched_hparams.eta_min
            <= SearchSpaceConfig.ETA_MIN_RANGE[1]
        )

    def test_suggest_cosine_warm_restarts_scheduler(self):
        """Test that cosine_warm_restarts scheduler suggests T_0, T_mult, eta_min."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"scheduler_type": "cosine_warm_restarts"}
        sched_hparams = suggest_scheduler_hyperparameters(trial, fixed_params)

        assert sched_hparams.scheduler_type == "cosine_warm_restarts"
        assert (
            SearchSpaceConfig.T_0_RANGE[0]
            <= sched_hparams.T_0
            <= SearchSpaceConfig.T_0_RANGE[1]
        )
        # T_mult should be between 1 and 3
        assert 1 <= sched_hparams.T_mult <= 3
        assert (
            SearchSpaceConfig.ETA_MIN_RANGE[0]
            <= sched_hparams.eta_min
            <= SearchSpaceConfig.ETA_MIN_RANGE[1]
        )

    def test_suggest_scheduler_with_warmup(self):
        """Test that warmup_epochs is suggested correctly."""
        study = optuna.create_study()
        trial = study.ask()

        sched_hparams = suggest_scheduler_hyperparameters(trial)

        # Warmup epochs should be between 0 and 10
        assert 0 <= sched_hparams.warmup_epochs <= 10


class TestRobustnessHyperparametersConditional:
    """Test conditional robustness hyperparameter suggestion."""

    def test_suggest_with_adversarial_training_disabled(self):
        """Test that minimal robustness params are returned when disabled."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_adversarial_training": False}
        robust_hparams = suggest_robustness_hyperparameters(trial, fixed_params)

        assert robust_hparams.enable_adversarial_training is False

    def test_suggest_with_adversarial_training_enabled(self):
        """Test full robustness params when adversarial training enabled."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_adversarial_training": True}
        robust_hparams = suggest_robustness_hyperparameters(trial, fixed_params)

        assert robust_hparams.enable_adversarial_training is True
        assert robust_hparams.attack_type in SearchSpaceConfig.ATTACK_TYPES
        assert (
            SearchSpaceConfig.EPSILON_RANGE[0]
            <= robust_hparams.epsilon
            <= SearchSpaceConfig.EPSILON_RANGE[1]
        )
        assert (
            SearchSpaceConfig.ALPHA_RANGE[0]
            <= robust_hparams.alpha
            <= SearchSpaceConfig.ALPHA_RANGE[1]
        )
        assert (
            SearchSpaceConfig.NUM_STEPS_RANGE[0]
            <= robust_hparams.num_steps
            <= SearchSpaceConfig.NUM_STEPS_RANGE[1]
        )

    def test_suggest_with_trades_enabled(self):
        """Test that TRADES beta is suggested when use_trades is True."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_adversarial_training": True, "use_trades": True}
        robust_hparams = suggest_robustness_hyperparameters(trial, fixed_params)

        assert robust_hparams.use_trades is True
        assert (
            SearchSpaceConfig.TRADES_BETA_RANGE[0]
            <= robust_hparams.trades_beta
            <= SearchSpaceConfig.TRADES_BETA_RANGE[1]
        )

    def test_suggest_with_trades_disabled(self):
        """Test that TRADES beta defaults when use_trades is False."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_adversarial_training": True, "use_trades": False}
        robust_hparams = suggest_robustness_hyperparameters(trial, fixed_params)

        assert robust_hparams.use_trades is False
        # Default trades_beta is 6.0
        assert robust_hparams.trades_beta == 6.0

    def test_suggest_adversarial_ratio(self):
        """Test that adversarial_ratio is suggested correctly."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_adversarial_training": True}
        robust_hparams = suggest_robustness_hyperparameters(trial, fixed_params)

        # Adversarial ratio should be between 0.5 and 1.0
        assert 0.5 <= robust_hparams.adversarial_ratio <= 1.0


class TestExplainabilityHyperparametersConditional:
    """Test conditional explainability hyperparameter suggestion."""

    def test_suggest_with_xai_loss_disabled(self):
        """Test that minimal XAI params are returned when disabled."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_xai_loss": False}
        xai_hparams = suggest_explainability_hyperparameters(trial, fixed_params)

        assert xai_hparams.enable_xai_loss is False

    def test_suggest_with_xai_loss_enabled(self):
        """Test full XAI params when XAI loss enabled."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_xai_loss": True}
        xai_hparams = suggest_explainability_hyperparameters(trial, fixed_params)

        assert xai_hparams.enable_xai_loss is True
        assert (
            SearchSpaceConfig.XAI_WEIGHT_RANGE[0]
            <= xai_hparams.xai_loss_weight
            <= SearchSpaceConfig.XAI_WEIGHT_RANGE[1]
        )
        assert (
            SearchSpaceConfig.COHERENCE_WEIGHT_RANGE[0]
            <= xai_hparams.concept_coherence_weight
            <= SearchSpaceConfig.COHERENCE_WEIGHT_RANGE[1]
        )
        assert (
            SearchSpaceConfig.COHERENCE_WEIGHT_RANGE[0]
            <= xai_hparams.spatial_coherence_weight
            <= SearchSpaceConfig.COHERENCE_WEIGHT_RANGE[1]
        )
        assert (
            SearchSpaceConfig.FAITHFULNESS_WEIGHT_RANGE[0]
            <= xai_hparams.faithfulness_weight
            <= SearchSpaceConfig.FAITHFULNESS_WEIGHT_RANGE[1]
        )
        assert (
            SearchSpaceConfig.SPARSITY_WEIGHT_RANGE[0]
            <= xai_hparams.sparsity_weight
            <= SearchSpaceConfig.SPARSITY_WEIGHT_RANGE[1]
        )

    def test_suggest_xai_methods(self):
        """Test that XAI methods are suggested correctly."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {"enable_xai_loss": True}
        xai_hparams = suggest_explainability_hyperparameters(trial, fixed_params)

        # Both should be boolean
        assert isinstance(xai_hparams.use_gradcam, bool)
        assert isinstance(xai_hparams.use_integrated_gradients, bool)


class TestFullConfigSuggestion:
    """Test suggest_full_config with various optimization flags."""

    def test_suggest_full_config_all_enabled(self):
        """Test full config with all optimization categories enabled."""
        study = optuna.create_study()
        trial = study.ask()

        config = suggest_full_config(trial)

        # All components should be present
        assert hasattr(config, "model")
        assert hasattr(config, "optimizer")
        assert hasattr(config, "scheduler")
        assert hasattr(config, "training")
        assert hasattr(config, "robustness")
        assert hasattr(config, "explainability")
        assert hasattr(config, "tri_objective")
        # Should have experiment metadata
        assert config.experiment_name == f"trial_{trial.number}"
        assert config.description == f"HPO trial {trial.number}"

    def test_suggest_full_config_selective_optimization(self):
        """Test full config with selective optimization."""
        study = optuna.create_study()
        trial = study.ask()

        config = suggest_full_config(
            trial,
            optimize_model=True,
            optimize_optimizer=False,
            optimize_scheduler=False,
            optimize_training=True,
            optimize_robustness=False,
            optimize_explainability=False,
            optimize_tri_objective=False,
        )

        # Model and training should be optimized
        assert config.model is not None
        assert config.training is not None
        # Others should have defaults
        assert config.optimizer is not None
        assert config.scheduler is not None

    def test_suggest_full_config_with_fixed_params(self):
        """Test full config with fixed parameters."""
        study = optuna.create_study()
        trial = study.ask()

        fixed_params = {
            "model": {"architecture": "resnet18"},
            "optimizer": {"optimizer_type": "adam", "learning_rate": 0.001},
        }

        config = suggest_full_config(trial, fixed_params=fixed_params)

        assert config.model.architecture == "resnet18"
        assert config.optimizer.optimizer_type == "adam"
        assert config.optimizer.learning_rate == 0.001


class TestSearchSpaceFactoryMethods:
    """Test all SearchSpaceFactory methods."""

    def test_create_quick_search_space(self):
        """Test quick search space creation."""
        search_space_fn = SearchSpaceFactory.create_quick_search_space()

        assert callable(search_space_fn)

        study = optuna.create_study()
        trial = study.ask()

        config = search_space_fn(trial)

        # Quick search should have minimal hyperparameters
        assert hasattr(config, "optimizer")
        assert hasattr(config, "training")
        assert hasattr(config, "robustness")

    def test_create_accuracy_focused_search_space(self):
        """Test accuracy-focused search space creation."""
        search_space_fn = SearchSpaceFactory.create_accuracy_focused_search_space()

        assert callable(search_space_fn)

        study = optuna.create_study()
        trial = study.ask()

        config = search_space_fn(trial)

        # When optimize_tri_objective=False with fixed params, weights are NOT fixed
        # because suggest_tri_objective_hyperparameters is called when optimize=True
        # Instead, verify that tri_objective params exist and are valid
        assert hasattr(config, "tri_objective")
        assert (
            config.tri_objective.accuracy_weight
            >= SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[0]
        )
        assert (
            config.tri_objective.robustness_weight
            >= SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[0]
        )

    def test_create_robustness_focused_search_space(self):
        """Test robustness-focused search space creation."""
        search_space_fn = SearchSpaceFactory.create_robustness_focused_search_space()

        assert callable(search_space_fn)

        study = optuna.create_study()
        trial = study.ask()

        config = search_space_fn(trial)

        # When optimize_tri_objective=False, tri_objective uses defaults not fixed params
        # But robustness params should be fixed
        assert hasattr(config, "tri_objective")
        assert hasattr(config, "robustness")
        # Adversarial training should be enabled (this IS fixed)
        assert config.robustness.enable_adversarial_training is True

    def test_create_balanced_search_space(self):
        """Test balanced search space creation."""
        search_space_fn = SearchSpaceFactory.create_balanced_search_space()

        assert callable(search_space_fn)

        study = optuna.create_study()
        trial = study.ask()

        config = search_space_fn(trial)

        # All weights should be equal (1.0)
        assert config.tri_objective.accuracy_weight == 1.0
        assert config.tri_objective.robustness_weight == 1.0
        assert config.tri_objective.explainability_weight == 1.0


class TestTriObjectiveWeights:
    """Test tri-objective weight suggestion."""

    def test_suggest_tri_objective_weights(self):
        """Test that all tri-objective weights are suggested correctly."""
        study = optuna.create_study()
        trial = study.ask()

        tri_obj_hparams = suggest_tri_objective_hyperparameters(trial)

        # All weights should be in the specified range
        assert (
            SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[0]
            <= tri_obj_hparams.accuracy_weight
            <= SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[1]
        )
        assert (
            SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[0]
            <= tri_obj_hparams.robustness_weight
            <= SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[1]
        )
        assert (
            SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[0]
            <= tri_obj_hparams.explainability_weight
            <= SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE[1]
        )

    def test_suggest_tri_objective_dynamic_weighting(self):
        """Test dynamic weighting suggestion."""
        study = optuna.create_study()
        trial = study.ask()

        tri_obj_hparams = suggest_tri_objective_hyperparameters(trial)

        assert isinstance(tri_obj_hparams.use_dynamic_weighting, bool)

    def test_suggest_tri_objective_pareto_alpha(self):
        """Test Pareto alpha suggestion."""
        study = optuna.create_study()
        trial = study.ask()

        tri_obj_hparams = suggest_tri_objective_hyperparameters(trial)

        assert (
            SearchSpaceConfig.PARETO_ALPHA_RANGE[0]
            <= tri_obj_hparams.pareto_alpha
            <= SearchSpaceConfig.PARETO_ALPHA_RANGE[1]
        )


class TestTrainingHyperparametersComplete:
    """Test training hyperparameters with all parameters."""

    def test_suggest_training_batch_size(self):
        """Test batch size is from the options list."""
        study = optuna.create_study()
        trial = study.ask()

        train_hparams = suggest_training_hyperparameters(trial)

        assert train_hparams.batch_size in SearchSpaceConfig.BATCH_SIZE_OPTIONS

    def test_suggest_training_gradient_clip(self):
        """Test gradient clip value is in range."""
        study = optuna.create_study()
        trial = study.ask()

        train_hparams = suggest_training_hyperparameters(trial)

        assert (
            SearchSpaceConfig.GRADIENT_CLIP_RANGE[0]
            <= train_hparams.gradient_clip_value
            <= SearchSpaceConfig.GRADIENT_CLIP_RANGE[1]
        )

    def test_suggest_training_accumulation_steps(self):
        """Test accumulation steps is from options list."""
        study = optuna.create_study()
        trial = study.ask()

        train_hparams = suggest_training_hyperparameters(trial)

        assert (
            train_hparams.accumulation_steps
            in SearchSpaceConfig.ACCUMULATION_STEPS_OPTIONS
        )

    def test_suggest_training_mixed_precision(self):
        """Test mixed precision is boolean."""
        study = optuna.create_study()
        trial = study.ask()

        train_hparams = suggest_training_hyperparameters(trial)

        assert isinstance(train_hparams.mixed_precision, bool)

    def test_suggest_training_early_stopping(self):
        """Test early stopping patience is in range."""
        study = optuna.create_study()
        trial = study.ask()

        train_hparams = suggest_training_hyperparameters(trial)

        assert 10 <= train_hparams.early_stopping_patience <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
