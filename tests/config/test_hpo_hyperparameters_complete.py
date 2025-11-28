"""
Comprehensive tests for hyperparameters.py to achieve 100% coverage.

This test suite covers all uncovered paths including:
- All validation error branches for every hyperparameter class
- Factory methods (create_baseline, create_high_accuracy, etc.)
- Configuration validation function
- __post_init__ dict-to-object conversions
- All edge cases and error conditions

Author: Viraj Jain
Date: November 2025
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.config.hpo.hyperparameters import (
    AttackType,
    ExplainabilityHyperparameters,
    HyperparameterConfig,
    ModelHyperparameters,
    NormalizationType,
    OptimizerHyperparameters,
    RobustnessHyperparameters,
    SchedulerHyperparameters,
    TrainingHyperparameters,
    TriObjectiveHyperparameters,
    validate_config,
)


class TestOptimizerHyperparametersValidation:
    """Test all validation branches in OptimizerHyperparameters."""

    def test_betas_length_validation(self):
        """Test that betas must have length 2."""
        with pytest.raises(ValueError, match="betas must have length 2"):
            OptimizerHyperparameters(betas=(0.9,))

        with pytest.raises(ValueError, match="betas must have length 2"):
            OptimizerHyperparameters(betas=(0.9, 0.999, 0.1))

    def test_betas_range_validation(self):
        """Test that betas values must be in [0, 1)."""
        with pytest.raises(ValueError, match="betas must be in"):
            OptimizerHyperparameters(betas=(-0.1, 0.999))

        with pytest.raises(ValueError, match="betas must be in"):
            OptimizerHyperparameters(betas=(0.9, 1.0))

        with pytest.raises(ValueError, match="betas must be in"):
            OptimizerHyperparameters(betas=(0.9, 1.1))

    def test_eps_validation(self):
        """Test that eps must be positive."""
        with pytest.raises(ValueError, match="eps must be positive"):
            OptimizerHyperparameters(eps=0.0)

        with pytest.raises(ValueError, match="eps must be positive"):
            OptimizerHyperparameters(eps=-1e-8)

    def test_momentum_edge_cases(self):
        """Test momentum validation edge cases."""
        with pytest.raises(ValueError, match="momentum must be in"):
            OptimizerHyperparameters(momentum=-0.1)

        with pytest.raises(ValueError, match="momentum must be in"):
            OptimizerHyperparameters(momentum=1.1)


class TestSchedulerHyperparametersValidation:
    """Test all validation branches in SchedulerHyperparameters."""

    def test_gamma_range_validation(self):
        """Test that gamma must be in (0, 1]."""
        with pytest.raises(ValueError, match="gamma must be in"):
            SchedulerHyperparameters(gamma=0.0)

        with pytest.raises(ValueError, match="gamma must be in"):
            SchedulerHyperparameters(gamma=-0.1)

        with pytest.raises(ValueError, match="gamma must be in"):
            SchedulerHyperparameters(gamma=1.1)

    def test_t_max_validation(self):
        """Test that T_max must be >= 1."""
        with pytest.raises(ValueError, match="T_max must be"):
            SchedulerHyperparameters(T_max=0)

        with pytest.raises(ValueError, match="T_max must be"):
            SchedulerHyperparameters(T_max=-1)

    def test_t_0_validation(self):
        """Test that T_0 must be >= 1."""
        with pytest.raises(ValueError, match="T_0 must be"):
            SchedulerHyperparameters(T_0=0)

        with pytest.raises(ValueError, match="T_0 must be"):
            SchedulerHyperparameters(T_0=-1)

    def test_t_mult_validation(self):
        """Test that T_mult must be >= 1."""
        with pytest.raises(ValueError, match="T_mult must be"):
            SchedulerHyperparameters(T_mult=0)

        with pytest.raises(ValueError, match="T_mult must be"):
            SchedulerHyperparameters(T_mult=-1)

    def test_eta_min_validation(self):
        """Test that eta_min must be non-negative."""
        with pytest.raises(ValueError, match="eta_min must be non-negative"):
            SchedulerHyperparameters(eta_min=-1e-6)

    def test_patience_validation(self):
        """Test that patience must be >= 1."""
        with pytest.raises(ValueError, match="patience must be"):
            SchedulerHyperparameters(patience=0)

        with pytest.raises(ValueError, match="patience must be"):
            SchedulerHyperparameters(patience=-1)

    def test_factor_validation(self):
        """Test that factor must be in (0, 1)."""
        with pytest.raises(ValueError, match="factor must be in"):
            SchedulerHyperparameters(factor=0.0)

        with pytest.raises(ValueError, match="factor must be in"):
            SchedulerHyperparameters(factor=1.0)

        with pytest.raises(ValueError, match="factor must be in"):
            SchedulerHyperparameters(factor=-0.1)

        with pytest.raises(ValueError, match="factor must be in"):
            SchedulerHyperparameters(factor=1.1)


class TestTrainingHyperparametersValidation:
    """Test all validation branches in TrainingHyperparameters."""

    def test_accumulation_steps_validation(self):
        """Test that accumulation_steps must be >= 1."""
        with pytest.raises(ValueError, match="accumulation_steps must be"):
            TrainingHyperparameters(accumulation_steps=0)

        with pytest.raises(ValueError, match="accumulation_steps must be"):
            TrainingHyperparameters(accumulation_steps=-1)

    def test_early_stopping_patience_validation(self):
        """Test that early_stopping_patience must be non-negative."""
        with pytest.raises(
            ValueError, match="early_stopping_patience must be non-negative"
        ):
            TrainingHyperparameters(early_stopping_patience=-1)

    def test_val_frequency_validation(self):
        """Test that val_frequency must be >= 1."""
        with pytest.raises(ValueError, match="val_frequency must be"):
            TrainingHyperparameters(val_frequency=0)

        with pytest.raises(ValueError, match="val_frequency must be"):
            TrainingHyperparameters(val_frequency=-1)

    def test_log_frequency_validation(self):
        """Test that log_frequency must be >= 1."""
        with pytest.raises(ValueError, match="log_frequency must be"):
            TrainingHyperparameters(log_frequency=0)

        with pytest.raises(ValueError, match="log_frequency must be"):
            TrainingHyperparameters(log_frequency=-1)

    def test_save_frequency_validation(self):
        """Test that save_frequency must be >= 1."""
        with pytest.raises(ValueError, match="save_frequency must be"):
            TrainingHyperparameters(save_frequency=0)

        with pytest.raises(ValueError, match="save_frequency must be"):
            TrainingHyperparameters(save_frequency=-1)

    def test_num_workers_validation(self):
        """Test that num_workers must be non-negative."""
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            TrainingHyperparameters(num_workers=-1)


class TestRobustnessHyperparametersValidation:
    """Test all validation branches in RobustnessHyperparameters."""

    def test_epsilon_range_validation(self):
        """Test that epsilon must be in [0, 1]."""
        with pytest.raises(ValueError, match="epsilon must be in"):
            RobustnessHyperparameters(epsilon=-0.1)

        with pytest.raises(ValueError, match="epsilon must be in"):
            RobustnessHyperparameters(epsilon=1.1)

    def test_alpha_validation(self):
        """Test that alpha must be positive."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            RobustnessHyperparameters(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            RobustnessHyperparameters(alpha=-0.1)

    def test_num_steps_validation(self):
        """Test that num_steps must be >= 1."""
        with pytest.raises(ValueError, match="num_steps must be"):
            RobustnessHyperparameters(num_steps=0)

        with pytest.raises(ValueError, match="num_steps must be"):
            RobustnessHyperparameters(num_steps=-1)

    def test_adversarial_ratio_range_validation(self):
        """Test that adversarial_ratio must be in [0, 1]."""
        with pytest.raises(ValueError, match="adversarial_ratio must be in"):
            RobustnessHyperparameters(adversarial_ratio=-0.1)

        with pytest.raises(ValueError, match="adversarial_ratio must be in"):
            RobustnessHyperparameters(adversarial_ratio=1.1)

    def test_trades_beta_validation(self):
        """Test that trades_beta must be non-negative."""
        with pytest.raises(ValueError, match="trades_beta must be non-negative"):
            RobustnessHyperparameters(trades_beta=-1.0)

    def test_mart_beta_validation(self):
        """Test that mart_beta must be non-negative."""
        with pytest.raises(ValueError, match="mart_beta must be non-negative"):
            RobustnessHyperparameters(mart_beta=-1.0)


class TestExplainabilityHyperparametersValidation:
    """Test all validation branches in ExplainabilityHyperparameters."""

    def test_xai_loss_weight_validation(self):
        """Test that xai_loss_weight must be non-negative."""
        with pytest.raises(ValueError, match="xai_loss_weight must be non-negative"):
            ExplainabilityHyperparameters(xai_loss_weight=-0.1)

    def test_ig_steps_validation(self):
        """Test that ig_steps must be >= 1."""
        with pytest.raises(ValueError, match="ig_steps must be"):
            ExplainabilityHyperparameters(ig_steps=0)

        with pytest.raises(ValueError, match="ig_steps must be"):
            ExplainabilityHyperparameters(ig_steps=-1)

    def test_lime_num_samples_validation(self):
        """Test that lime_num_samples must be >= 1."""
        with pytest.raises(ValueError, match="lime_num_samples must be"):
            ExplainabilityHyperparameters(lime_num_samples=0)

        with pytest.raises(ValueError, match="lime_num_samples must be"):
            ExplainabilityHyperparameters(lime_num_samples=-1)

    def test_shap_num_samples_validation(self):
        """Test that shap_num_samples must be >= 1."""
        with pytest.raises(ValueError, match="shap_num_samples must be"):
            ExplainabilityHyperparameters(shap_num_samples=0)

        with pytest.raises(ValueError, match="shap_num_samples must be"):
            ExplainabilityHyperparameters(shap_num_samples=-1)

    def test_concept_coherence_weight_validation(self):
        """Test that concept_coherence_weight must be non-negative."""
        with pytest.raises(
            ValueError, match="concept_coherence_weight must be non-negative"
        ):
            ExplainabilityHyperparameters(concept_coherence_weight=-0.1)

    def test_spatial_coherence_weight_validation(self):
        """Test that spatial_coherence_weight must be non-negative."""
        with pytest.raises(
            ValueError, match="spatial_coherence_weight must be non-negative"
        ):
            ExplainabilityHyperparameters(spatial_coherence_weight=-0.1)

    def test_faithfulness_weight_validation(self):
        """Test that faithfulness_weight must be non-negative."""
        with pytest.raises(
            ValueError, match="faithfulness_weight must be non-negative"
        ):
            ExplainabilityHyperparameters(faithfulness_weight=-0.1)

    def test_sparsity_weight_validation(self):
        """Test that sparsity_weight must be non-negative."""
        with pytest.raises(ValueError, match="sparsity_weight must be non-negative"):
            ExplainabilityHyperparameters(sparsity_weight=-0.1)


class TestTriObjectiveHyperparametersValidation:
    """Test all validation branches in TriObjectiveHyperparameters."""

    def test_accuracy_weight_validation(self):
        """Test that accuracy_weight must be non-negative."""
        with pytest.raises(ValueError, match="accuracy_weight must be non-negative"):
            TriObjectiveHyperparameters(accuracy_weight=-0.1)

    def test_robustness_weight_validation(self):
        """Test that robustness_weight must be non-negative."""
        with pytest.raises(ValueError, match="robustness_weight must be non-negative"):
            TriObjectiveHyperparameters(robustness_weight=-0.1)

    def test_explainability_weight_validation(self):
        """Test that explainability_weight must be non-negative."""
        with pytest.raises(
            ValueError, match="explainability_weight must be non-negative"
        ):
            TriObjectiveHyperparameters(explainability_weight=-0.1)

    def test_weight_adjustment_frequency_validation(self):
        """Test that weight_adjustment_frequency must be >= 1."""
        with pytest.raises(ValueError, match="weight_adjustment_frequency must be"):
            TriObjectiveHyperparameters(weight_adjustment_frequency=0)

        with pytest.raises(ValueError, match="weight_adjustment_frequency must be"):
            TriObjectiveHyperparameters(weight_adjustment_frequency=-1)

    def test_min_weight_validation(self):
        """Test that min_weight must be positive."""
        with pytest.raises(ValueError, match="min_weight must be positive"):
            TriObjectiveHyperparameters(min_weight=0.0)

        with pytest.raises(ValueError, match="min_weight must be positive"):
            TriObjectiveHyperparameters(min_weight=-0.1)

    def test_max_weight_validation(self):
        """Test that max_weight must be > min_weight."""
        with pytest.raises(ValueError, match="max_weight must be > min_weight"):
            TriObjectiveHyperparameters(min_weight=2.0, max_weight=2.0)

        with pytest.raises(ValueError, match="max_weight must be > min_weight"):
            TriObjectiveHyperparameters(min_weight=2.0, max_weight=1.0)

    def test_pareto_alpha_range_validation(self):
        """Test that pareto_alpha must be in [0, 1]."""
        with pytest.raises(ValueError, match="pareto_alpha must be in"):
            TriObjectiveHyperparameters(pareto_alpha=-0.1)

        with pytest.raises(ValueError, match="pareto_alpha must be in"):
            TriObjectiveHyperparameters(pareto_alpha=1.1)


class TestHyperparameterConfigPostInit:
    """Test __post_init__ dict-to-object conversions."""

    def test_model_dict_conversion(self):
        """Test that model dict is converted to ModelHyperparameters."""
        config = HyperparameterConfig(
            model={"architecture": "resnet34", "num_classes": 10}
        )

        assert isinstance(config.model, ModelHyperparameters)
        assert config.model.architecture == "resnet34"
        assert config.model.num_classes == 10

    def test_optimizer_dict_conversion(self):
        """Test that optimizer dict is converted to OptimizerHyperparameters."""
        config = HyperparameterConfig(
            optimizer={"learning_rate": 0.01, "weight_decay": 0.001}
        )

        assert isinstance(config.optimizer, OptimizerHyperparameters)
        assert config.optimizer.learning_rate == 0.01
        assert config.optimizer.weight_decay == 0.001

    def test_scheduler_dict_conversion(self):
        """Test that scheduler dict is converted to SchedulerHyperparameters."""
        config = HyperparameterConfig(scheduler={"step_size": 50, "gamma": 0.2})

        assert isinstance(config.scheduler, SchedulerHyperparameters)
        assert config.scheduler.step_size == 50
        assert config.scheduler.gamma == 0.2

    def test_training_dict_conversion(self):
        """Test that training dict is converted to TrainingHyperparameters."""
        config = HyperparameterConfig(training={"batch_size": 128, "num_epochs": 100})

        assert isinstance(config.training, TrainingHyperparameters)
        assert config.training.batch_size == 128
        assert config.training.num_epochs == 100

    def test_robustness_dict_conversion(self):
        """Test that robustness dict is converted to RobustnessHyperparameters."""
        config = HyperparameterConfig(robustness={"epsilon": 0.03, "num_steps": 20})

        assert isinstance(config.robustness, RobustnessHyperparameters)
        assert config.robustness.epsilon == 0.03
        assert config.robustness.num_steps == 20

    def test_explainability_dict_conversion(self):
        """Test that explainability dict is converted to ExplainabilityHyperparameters."""
        config = HyperparameterConfig(
            explainability={"xai_loss_weight": 0.2, "ig_steps": 100}
        )

        assert isinstance(config.explainability, ExplainabilityHyperparameters)
        assert config.explainability.xai_loss_weight == 0.2
        assert config.explainability.ig_steps == 100

    def test_tri_objective_dict_conversion(self):
        """Test that tri_objective dict is converted to TriObjectiveHyperparameters."""
        config = HyperparameterConfig(
            tri_objective={"accuracy_weight": 1.5, "robustness_weight": 0.5}
        )

        assert isinstance(config.tri_objective, TriObjectiveHyperparameters)
        assert config.tri_objective.accuracy_weight == 1.5
        assert config.tri_objective.robustness_weight == 0.5


class TestFactoryMethods:
    """Test all factory methods for HyperparameterConfig."""

    def test_create_baseline(self):
        """Test create_baseline factory method."""
        config = HyperparameterConfig.create_baseline()

        assert config.experiment_name == "baseline_config"
        assert "Baseline" in config.description
        assert isinstance(config.model, ModelHyperparameters)
        assert isinstance(config.optimizer, OptimizerHyperparameters)

    def test_create_high_accuracy(self):
        """Test create_high_accuracy factory method."""
        config = HyperparameterConfig.create_high_accuracy()

        assert config.experiment_name == "high_accuracy_config"
        assert "accuracy" in config.description
        assert config.model.architecture == "efficientnet_b4"
        assert config.model.dropout_rate == 0.3
        assert config.model.use_se_blocks is True
        assert config.optimizer.learning_rate == 0.0001
        assert config.optimizer.weight_decay == 0.05
        assert config.training.batch_size == 16
        assert config.training.num_epochs == 300
        assert config.tri_objective.accuracy_weight == 2.0
        assert config.tri_objective.robustness_weight == 0.5

    def test_create_high_robustness(self):
        """Test create_high_robustness factory method."""
        config = HyperparameterConfig.create_high_robustness()

        assert config.experiment_name == "high_robustness_config"
        assert "robustness" in config.description
        assert config.model.architecture == "resnet50"
        assert config.robustness.enable_adversarial_training is True
        assert config.robustness.attack_type == AttackType.PGD
        assert config.robustness.epsilon == 8.0 / 255.0
        assert config.robustness.num_steps == 20
        assert config.robustness.use_trades is True
        assert config.robustness.trades_beta == 8.0
        assert config.tri_objective.robustness_weight == 2.0

    def test_create_high_explainability(self):
        """Test create_high_explainability factory method."""
        config = HyperparameterConfig.create_high_explainability()

        assert config.experiment_name == "high_explainability_config"
        assert "explainability" in config.description
        assert config.model.architecture == "resnet34"
        assert config.model.use_attention is True
        assert config.model.attention_heads == 8
        assert config.explainability.enable_xai_loss is True
        assert config.explainability.xai_loss_weight == 0.2
        assert config.explainability.use_gradcam is True
        assert config.explainability.use_integrated_gradients is True
        assert config.explainability.concept_coherence_weight == 0.1
        assert config.tri_objective.explainability_weight == 2.0

    def test_create_balanced(self):
        """Test create_balanced factory method."""
        config = HyperparameterConfig.create_balanced()

        assert config.experiment_name == "balanced_config"
        assert "balanced" in config.description or "tri-objective" in config.description
        assert config.model.architecture == "resnet50"
        assert config.model.use_attention is True
        assert config.model.use_se_blocks is True
        assert config.robustness.enable_adversarial_training is True
        assert config.robustness.use_trades is True
        assert config.explainability.enable_xai_loss is True
        assert config.tri_objective.accuracy_weight == 1.0
        assert config.tri_objective.robustness_weight == 1.0
        assert config.tri_objective.explainability_weight == 1.0
        assert config.tri_objective.use_dynamic_weighting is True


class TestValidateConfigFunction:
    """Test validate_config function."""

    def test_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        config = HyperparameterConfig()

        is_valid, errors = validate_config(config)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_adversarial_training_epsilon_zero(self):
        """Test validate_config catches adversarial training with epsilon=0."""
        config = HyperparameterConfig(
            robustness=RobustnessHyperparameters(
                enable_adversarial_training=True, epsilon=0.0
            )
        )

        is_valid, errors = validate_config(config)

        assert is_valid is False
        assert len(errors) >= 1
        assert any(
            "Adversarial training" in error and "epsilon" in error for error in errors
        )

    def test_validate_config_xai_loss_weight_zero(self):
        """Test validate_config catches XAI loss enabled with weight=0."""
        config = HyperparameterConfig(
            explainability=ExplainabilityHyperparameters(
                enable_xai_loss=True, xai_loss_weight=0.0
            )
        )

        is_valid, errors = validate_config(config)

        assert is_valid is False
        assert len(errors) >= 1
        assert any("XAI loss" in error and "weight" in error for error in errors)

    def test_validate_config_weight_adjustment_exceeds_epochs(self):
        """Test validate_config catches weight adjustment frequency > epochs."""
        config = HyperparameterConfig(
            training=TrainingHyperparameters(num_epochs=100),
            tri_objective=TriObjectiveHyperparameters(
                use_dynamic_weighting=True, weight_adjustment_frequency=150
            ),
        )

        is_valid, errors = validate_config(config)

        assert is_valid is False
        assert len(errors) >= 1
        assert any(
            "Weight adjustment" in error and "epochs" in error for error in errors
        )

    def test_validate_config_serialization_error(self):
        """Test validate_config catches serialization errors."""
        # Create a config that might cause serialization issues
        config = HyperparameterConfig()

        # Patch to_dict to raise an exception
        original_to_dict = config.to_dict

        def broken_to_dict():
            raise RuntimeError("Serialization error")

        config.to_dict = broken_to_dict

        is_valid, errors = validate_config(config)

        assert is_valid is False
        assert len(errors) >= 1
        assert any("serialization" in error.lower() for error in errors)


class TestEnumStringConversions:
    """Test string-to-enum conversions in __post_init__."""

    def test_optimizer_type_string_conversion(self):
        """Test optimizer_type string is converted to enum."""
        opt = OptimizerHyperparameters(optimizer_type="sgd")

        from src.config.hpo.hyperparameters import OptimizerType

        assert opt.optimizer_type == OptimizerType.SGD

    def test_scheduler_type_string_conversion(self):
        """Test scheduler_type string is converted to enum."""
        sched = SchedulerHyperparameters(scheduler_type="step")

        from src.config.hpo.hyperparameters import SchedulerType

        assert sched.scheduler_type == SchedulerType.STEP

    def test_attack_type_string_conversion(self):
        """Test attack_type string is converted to enum."""
        robust = RobustnessHyperparameters(attack_type="fgsm")

        assert robust.attack_type == AttackType.FGSM


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
