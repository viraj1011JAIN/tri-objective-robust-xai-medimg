"""
Optuna Search Space Definitions for Hyperparameter Optimization.

This module defines search spaces for all hyperparameter categories using Optuna's
suggest methods. It provides flexible search space definitions that can be used
for different optimization strategies.

Author: Viraj Jain
Date: November 2025
"""

from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.trial import Trial

from .hyperparameters import (
    ActivationType,
    AttackType,
    ExplainabilityHyperparameters,
    HyperparameterConfig,
    ModelHyperparameters,
    NormalizationType,
    OptimizerHyperparameters,
    OptimizerType,
    RobustnessHyperparameters,
    SchedulerHyperparameters,
    SchedulerType,
    TrainingHyperparameters,
    TriObjectiveHyperparameters,
)


class SearchSpaceConfig:
    """
    Configuration for search space bounds and options.

    This class defines the ranges and choices for hyperparameter optimization.
    """

    # Model architecture search space
    ARCHITECTURES = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "densenet121",
        "densenet169",
        "densenet201",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
    ]

    ACTIVATIONS = [e.value for e in ActivationType]
    NORMALIZATIONS = [e.value for e in NormalizationType]

    # Optimizer search space
    OPTIMIZERS = [e.value for e in OptimizerType]
    LEARNING_RATE_RANGE = (1e-5, 1e-1)
    WEIGHT_DECAY_RANGE = (1e-6, 1e-1)
    MOMENTUM_RANGE = (0.8, 0.99)
    BETA1_RANGE = (0.8, 0.99)
    BETA2_RANGE = (0.9, 0.999)

    # Scheduler search space
    SCHEDULERS = [e.value for e in SchedulerType]
    STEP_SIZE_RANGE = (10, 50)
    GAMMA_RANGE = (0.05, 0.5)
    T_MAX_RANGE = (50, 300)
    T_0_RANGE = (5, 20)
    ETA_MIN_RANGE = (1e-7, 1e-5)

    # Training search space
    BATCH_SIZE_OPTIONS = [8, 16, 32, 64, 128]
    DROPOUT_RANGE = (0.0, 0.7)
    GRADIENT_CLIP_RANGE = (0.5, 5.0)
    ACCUMULATION_STEPS_OPTIONS = [1, 2, 4, 8]

    # Robustness search space
    ATTACK_TYPES = [e.value for e in AttackType]
    EPSILON_RANGE = (0.0, 16.0 / 255.0)
    ALPHA_RANGE = (0.5 / 255.0, 4.0 / 255.0)
    NUM_STEPS_RANGE = (5, 40)
    TRADES_BETA_RANGE = (1.0, 12.0)

    # Explainability search space
    XAI_WEIGHT_RANGE = (0.01, 0.5)
    COHERENCE_WEIGHT_RANGE = (0.01, 0.2)
    FAITHFULNESS_WEIGHT_RANGE = (0.05, 0.3)
    SPARSITY_WEIGHT_RANGE = (0.001, 0.05)

    # Tri-objective search space
    OBJECTIVE_WEIGHT_RANGE = (0.1, 2.0)
    PARETO_ALPHA_RANGE = (0.0, 1.0)


def suggest_model_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> ModelHyperparameters:
    """
    Suggest model architecture hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        ModelHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    architecture = fixed_params.get(
        "architecture",
        trial.suggest_categorical(
            "model_architecture", SearchSpaceConfig.ARCHITECTURES
        ),
    )

    dropout_rate = fixed_params.get(
        "dropout_rate",
        trial.suggest_float("model_dropout_rate", *SearchSpaceConfig.DROPOUT_RANGE),
    )

    activation = fixed_params.get(
        "activation",
        trial.suggest_categorical("model_activation", SearchSpaceConfig.ACTIVATIONS),
    )

    normalization = fixed_params.get(
        "normalization",
        trial.suggest_categorical(
            "model_normalization", SearchSpaceConfig.NORMALIZATIONS
        ),
    )

    use_attention = fixed_params.get(
        "use_attention", trial.suggest_categorical("model_use_attention", [True, False])
    )

    attention_heads = fixed_params.get(
        "attention_heads",
        (
            trial.suggest_int("model_attention_heads", 4, 16, step=4)
            if use_attention
            else 8
        ),
    )

    # Hidden dimensions
    num_hidden_layers = trial.suggest_int("model_num_hidden_layers", 1, 3)
    hidden_dims = []
    for i in range(num_hidden_layers):
        dim = trial.suggest_categorical(f"model_hidden_dim_{i}", [128, 256, 512, 1024])
        hidden_dims.append(dim)

    use_se_blocks = fixed_params.get(
        "use_se_blocks", trial.suggest_categorical("model_use_se_blocks", [True, False])
    )

    se_reduction = fixed_params.get(
        "se_reduction",
        (
            trial.suggest_categorical("model_se_reduction", [8, 16, 32])
            if use_se_blocks
            else 16
        ),
    )

    return ModelHyperparameters(
        architecture=architecture,
        dropout_rate=dropout_rate,
        activation=activation,
        normalization=normalization,
        use_attention=use_attention,
        attention_heads=attention_heads,
        hidden_dims=hidden_dims,
        use_se_blocks=use_se_blocks,
        se_reduction=se_reduction,
    )


def suggest_optimizer_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> OptimizerHyperparameters:
    """
    Suggest optimizer hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        OptimizerHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    optimizer_type = fixed_params.get(
        "optimizer_type",
        trial.suggest_categorical("optimizer_type", SearchSpaceConfig.OPTIMIZERS),
    )

    learning_rate = fixed_params.get(
        "learning_rate",
        trial.suggest_float(
            "optimizer_learning_rate", *SearchSpaceConfig.LEARNING_RATE_RANGE, log=True
        ),
    )

    weight_decay = fixed_params.get(
        "weight_decay",
        trial.suggest_float(
            "optimizer_weight_decay", *SearchSpaceConfig.WEIGHT_DECAY_RANGE, log=True
        ),
    )

    # Optimizer-specific parameters
    momentum = 0.9
    betas = (0.9, 0.999)

    if optimizer_type in ["sgd", "rmsprop"]:
        momentum = fixed_params.get(
            "momentum",
            trial.suggest_float(
                "optimizer_momentum", *SearchSpaceConfig.MOMENTUM_RANGE
            ),
        )

    if optimizer_type in ["adam", "adamw", "nadam"]:
        beta1 = trial.suggest_float("optimizer_beta1", *SearchSpaceConfig.BETA1_RANGE)
        beta2 = trial.suggest_float("optimizer_beta2", *SearchSpaceConfig.BETA2_RANGE)
        betas = (beta1, beta2)

    nesterov = (
        trial.suggest_categorical("optimizer_nesterov", [True, False])
        if optimizer_type == "sgd"
        else False
    )

    return OptimizerHyperparameters(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        betas=betas,
        nesterov=nesterov,
    )


def suggest_scheduler_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> SchedulerHyperparameters:
    """
    Suggest learning rate scheduler hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        SchedulerHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    scheduler_type = fixed_params.get(
        "scheduler_type",
        trial.suggest_categorical("scheduler_type", SearchSpaceConfig.SCHEDULERS),
    )

    # Scheduler-specific parameters
    step_size = 30
    gamma = 0.1
    T_max = 200
    T_0 = 10
    eta_min = 1e-6
    warmup_epochs = 5

    if scheduler_type == "step":
        step_size = trial.suggest_int(
            "scheduler_step_size", *SearchSpaceConfig.STEP_SIZE_RANGE
        )
        gamma = trial.suggest_float("scheduler_gamma", *SearchSpaceConfig.GAMMA_RANGE)

    elif scheduler_type == "cosine":
        T_max = trial.suggest_int("scheduler_T_max", *SearchSpaceConfig.T_MAX_RANGE)
        eta_min = trial.suggest_float(
            "scheduler_eta_min", *SearchSpaceConfig.ETA_MIN_RANGE, log=True
        )

    elif scheduler_type == "cosine_warm_restarts":
        T_0 = trial.suggest_int("scheduler_T_0", *SearchSpaceConfig.T_0_RANGE)
        T_mult = trial.suggest_int("scheduler_T_mult", 1, 3)
        eta_min = trial.suggest_float(
            "scheduler_eta_min", *SearchSpaceConfig.ETA_MIN_RANGE, log=True
        )

    warmup_epochs = trial.suggest_int("scheduler_warmup_epochs", 0, 10)

    return SchedulerHyperparameters(
        scheduler_type=scheduler_type,
        step_size=step_size,
        gamma=gamma,
        T_max=T_max,
        T_0=T_0,
        eta_min=eta_min,
        warmup_epochs=warmup_epochs,
    )


def suggest_training_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> TrainingHyperparameters:
    """
    Suggest training process hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        TrainingHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    batch_size = fixed_params.get(
        "batch_size",
        trial.suggest_categorical(
            "training_batch_size", SearchSpaceConfig.BATCH_SIZE_OPTIONS
        ),
    )

    gradient_clip_value = fixed_params.get(
        "gradient_clip_value",
        trial.suggest_float(
            "training_gradient_clip", *SearchSpaceConfig.GRADIENT_CLIP_RANGE
        ),
    )

    accumulation_steps = fixed_params.get(
        "accumulation_steps",
        trial.suggest_categorical(
            "training_accumulation_steps", SearchSpaceConfig.ACCUMULATION_STEPS_OPTIONS
        ),
    )

    mixed_precision = fixed_params.get(
        "mixed_precision",
        trial.suggest_categorical("training_mixed_precision", [True, False]),
    )

    early_stopping_patience = fixed_params.get(
        "early_stopping_patience",
        trial.suggest_int("training_early_stopping_patience", 10, 50),
    )

    return TrainingHyperparameters(
        batch_size=batch_size,
        gradient_clip_value=gradient_clip_value,
        accumulation_steps=accumulation_steps,
        mixed_precision=mixed_precision,
        early_stopping_patience=early_stopping_patience,
    )


def suggest_robustness_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> RobustnessHyperparameters:
    """
    Suggest adversarial robustness hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        RobustnessHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    enable_adversarial_training = fixed_params.get(
        "enable_adversarial_training",
        trial.suggest_categorical("robustness_enable_adv_training", [True, False]),
    )

    if not enable_adversarial_training:
        return RobustnessHyperparameters(enable_adversarial_training=False)

    attack_type = fixed_params.get(
        "attack_type",
        trial.suggest_categorical(
            "robustness_attack_type", SearchSpaceConfig.ATTACK_TYPES
        ),
    )

    epsilon = fixed_params.get(
        "epsilon",
        trial.suggest_float("robustness_epsilon", *SearchSpaceConfig.EPSILON_RANGE),
    )

    alpha = fixed_params.get(
        "alpha", trial.suggest_float("robustness_alpha", *SearchSpaceConfig.ALPHA_RANGE)
    )

    num_steps = fixed_params.get(
        "num_steps",
        trial.suggest_int("robustness_num_steps", *SearchSpaceConfig.NUM_STEPS_RANGE),
    )

    use_trades = fixed_params.get(
        "use_trades", trial.suggest_categorical("robustness_use_trades", [True, False])
    )

    trades_beta = 6.0
    if use_trades:
        trades_beta = trial.suggest_float(
            "robustness_trades_beta", *SearchSpaceConfig.TRADES_BETA_RANGE
        )

    adversarial_ratio = trial.suggest_float("robustness_adversarial_ratio", 0.5, 1.0)

    return RobustnessHyperparameters(
        enable_adversarial_training=enable_adversarial_training,
        attack_type=attack_type,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        use_trades=use_trades,
        trades_beta=trades_beta,
        adversarial_ratio=adversarial_ratio,
    )


def suggest_explainability_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> ExplainabilityHyperparameters:
    """
    Suggest explainability hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        ExplainabilityHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    enable_xai_loss = fixed_params.get(
        "enable_xai_loss",
        trial.suggest_categorical("xai_enable_xai_loss", [True, False]),
    )

    if not enable_xai_loss:
        return ExplainabilityHyperparameters(enable_xai_loss=False)

    xai_loss_weight = fixed_params.get(
        "xai_loss_weight",
        trial.suggest_float("xai_loss_weight", *SearchSpaceConfig.XAI_WEIGHT_RANGE),
    )

    concept_coherence_weight = trial.suggest_float(
        "xai_concept_coherence_weight", *SearchSpaceConfig.COHERENCE_WEIGHT_RANGE
    )

    spatial_coherence_weight = trial.suggest_float(
        "xai_spatial_coherence_weight", *SearchSpaceConfig.COHERENCE_WEIGHT_RANGE
    )

    faithfulness_weight = trial.suggest_float(
        "xai_faithfulness_weight", *SearchSpaceConfig.FAITHFULNESS_WEIGHT_RANGE
    )

    sparsity_weight = trial.suggest_float(
        "xai_sparsity_weight", *SearchSpaceConfig.SPARSITY_WEIGHT_RANGE
    )

    use_gradcam = trial.suggest_categorical("xai_use_gradcam", [True, False])
    use_integrated_gradients = trial.suggest_categorical("xai_use_ig", [True, False])

    return ExplainabilityHyperparameters(
        enable_xai_loss=enable_xai_loss,
        xai_loss_weight=xai_loss_weight,
        concept_coherence_weight=concept_coherence_weight,
        spatial_coherence_weight=spatial_coherence_weight,
        faithfulness_weight=faithfulness_weight,
        sparsity_weight=sparsity_weight,
        use_gradcam=use_gradcam,
        use_integrated_gradients=use_integrated_gradients,
    )


def suggest_tri_objective_hyperparameters(
    trial: Trial, fixed_params: Optional[Dict[str, Any]] = None
) -> TriObjectiveHyperparameters:
    """
    Suggest tri-objective optimization hyperparameters.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)

    Returns:
        TriObjectiveHyperparameters instance with suggested values
    """
    fixed_params = fixed_params or {}

    accuracy_weight = fixed_params.get(
        "accuracy_weight",
        trial.suggest_float(
            "tri_obj_accuracy_weight", *SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE
        ),
    )

    robustness_weight = fixed_params.get(
        "robustness_weight",
        trial.suggest_float(
            "tri_obj_robustness_weight", *SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE
        ),
    )

    explainability_weight = fixed_params.get(
        "explainability_weight",
        trial.suggest_float(
            "tri_obj_explainability_weight", *SearchSpaceConfig.OBJECTIVE_WEIGHT_RANGE
        ),
    )

    use_dynamic_weighting = fixed_params.get(
        "use_dynamic_weighting",
        trial.suggest_categorical("tri_obj_dynamic_weighting", [True, False]),
    )

    pareto_alpha = trial.suggest_float(
        "tri_obj_pareto_alpha", *SearchSpaceConfig.PARETO_ALPHA_RANGE
    )

    return TriObjectiveHyperparameters(
        accuracy_weight=accuracy_weight,
        robustness_weight=robustness_weight,
        explainability_weight=explainability_weight,
        use_dynamic_weighting=use_dynamic_weighting,
        pareto_alpha=pareto_alpha,
    )


def suggest_full_config(
    trial: Trial,
    fixed_params: Optional[Dict[str, Any]] = None,
    optimize_model: bool = True,
    optimize_optimizer: bool = True,
    optimize_scheduler: bool = True,
    optimize_training: bool = True,
    optimize_robustness: bool = True,
    optimize_explainability: bool = True,
    optimize_tri_objective: bool = True,
) -> HyperparameterConfig:
    """
    Suggest complete hyperparameter configuration.

    Args:
        trial: Optuna trial object
        fixed_params: Dictionary of parameters to fix (not optimize)
        optimize_model: Whether to optimize model hyperparameters
        optimize_optimizer: Whether to optimize optimizer hyperparameters
        optimize_scheduler: Whether to optimize scheduler hyperparameters
        optimize_training: Whether to optimize training hyperparameters
        optimize_robustness: Whether to optimize robustness hyperparameters
        optimize_explainability: Whether to optimize explainability hyperparameters
        optimize_tri_objective: Whether to optimize tri-objective hyperparameters

    Returns:
        HyperparameterConfig instance with suggested values
    """
    fixed_params = fixed_params or {}

    model = (
        suggest_model_hyperparameters(trial, fixed_params.get("model"))
        if optimize_model
        else ModelHyperparameters()
    )
    optimizer = (
        suggest_optimizer_hyperparameters(trial, fixed_params.get("optimizer"))
        if optimize_optimizer
        else OptimizerHyperparameters()
    )
    scheduler = (
        suggest_scheduler_hyperparameters(trial, fixed_params.get("scheduler"))
        if optimize_scheduler
        else SchedulerHyperparameters()
    )
    training = (
        suggest_training_hyperparameters(trial, fixed_params.get("training"))
        if optimize_training
        else TrainingHyperparameters()
    )
    robustness = (
        suggest_robustness_hyperparameters(trial, fixed_params.get("robustness"))
        if optimize_robustness
        else RobustnessHyperparameters()
    )
    explainability = (
        suggest_explainability_hyperparameters(
            trial, fixed_params.get("explainability")
        )
        if optimize_explainability
        else ExplainabilityHyperparameters()
    )
    tri_objective = (
        suggest_tri_objective_hyperparameters(trial, fixed_params.get("tri_objective"))
        if optimize_tri_objective
        else TriObjectiveHyperparameters()
    )

    return HyperparameterConfig(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        robustness=robustness,
        explainability=explainability,
        tri_objective=tri_objective,
        experiment_name=f"trial_{trial.number}",
        description=f"HPO trial {trial.number}",
    )


class SearchSpaceFactory:
    """
    Factory for creating different types of search spaces.
    """

    @staticmethod
    def create_quick_search_space() -> Callable[[Trial], HyperparameterConfig]:
        """
        Create a quick search space with fewer hyperparameters.

        Returns:
            Function that suggests hyperparameters for a trial
        """

        def suggest_fn(trial: Trial) -> HyperparameterConfig:
            return suggest_full_config(
                trial,
                optimize_model=False,
                optimize_scheduler=False,
                optimize_explainability=False,
            )

        return suggest_fn

    @staticmethod
    def create_full_search_space() -> Callable[[Trial], HyperparameterConfig]:
        """
        Create a comprehensive search space with all hyperparameters.

        Returns:
            Function that suggests hyperparameters for a trial
        """

        def suggest_fn(trial: Trial) -> HyperparameterConfig:
            return suggest_full_config(trial)

        return suggest_fn

    @staticmethod
    def create_accuracy_focused_search_space() -> (
        Callable[[Trial], HyperparameterConfig]
    ):
        """
        Create a search space focused on accuracy optimization.

        Returns:
            Function that suggests hyperparameters for a trial
        """

        def suggest_fn(trial: Trial) -> HyperparameterConfig:
            fixed_params = {
                "tri_objective": {
                    "accuracy_weight": 2.0,
                    "robustness_weight": 0.5,
                    "explainability_weight": 0.5,
                }
            }
            return suggest_full_config(
                trial, fixed_params=fixed_params, optimize_tri_objective=False
            )

        return suggest_fn

    @staticmethod
    def create_robustness_focused_search_space() -> (
        Callable[[Trial], HyperparameterConfig]
    ):
        """
        Create a search space focused on robustness optimization.

        Returns:
            Function that suggests hyperparameters for a trial
        """

        def suggest_fn(trial: Trial) -> HyperparameterConfig:
            fixed_params = {
                "robustness": {"enable_adversarial_training": True},
                "tri_objective": {
                    "accuracy_weight": 0.5,
                    "robustness_weight": 2.0,
                    "explainability_weight": 0.5,
                },
            }
            return suggest_full_config(
                trial, fixed_params=fixed_params, optimize_tri_objective=False
            )

        return suggest_fn

    @staticmethod
    def create_balanced_search_space() -> Callable[[Trial], HyperparameterConfig]:
        """
        Create a balanced search space for tri-objective optimization.

        Returns:
            Function that suggests hyperparameters for a trial
        """

        def suggest_fn(trial: Trial) -> HyperparameterConfig:
            fixed_params = {
                "tri_objective": {
                    "accuracy_weight": 1.0,
                    "robustness_weight": 1.0,
                    "explainability_weight": 1.0,
                }
            }
            return suggest_full_config(trial, fixed_params=fixed_params)

        return suggest_fn
