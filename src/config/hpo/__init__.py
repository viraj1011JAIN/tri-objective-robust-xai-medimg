"""
Hyperparameter Optimization Configuration Module.

This package provides comprehensive HPO infrastructure for the tri-objective
training framework, including:
- Hyperparameter configuration with dataclasses
- Search space definitions for Optuna
- Optimization objectives (weighted and Pareto)
- Pruning strategies for efficient optimization
- Main HPO trainer orchestration

Author: Viraj Jain
Date: November 2025
"""

from .hpo_trainer import HPOManager, HPOTrainer, create_hpo_trainer_from_config
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
    validate_config,
)
from .objectives import (  # ObjectiveDirection,  # Not in objectives.py - using OptimizationDirection; WeightedObjectiveConfig,  # Not needed - using WeightedSumObjective
    AccuracyObjective,
    AugmentedTchebycheffObjective,
    DynamicWeightAdjuster,
    ExplainabilityObjective,
    ObjectiveMetrics,
    ObjectiveType,
    OptimizationDirection,
    ParetoFrontTracker,
    PBIObjective,
    RobustnessObjective,
    SingleObjective,
    TchebycheffObjective,
    WeightedSumObjective,
    create_objective_function,
)
from .pruners import (
    AdaptivePruner,
    HybridPruner,
    MultiObjectivePruner,
    PerformanceBasedPruner,
    ResourceAwarePruner,
    create_pruner,
    get_default_pruner_for_objective,
)
from .search_spaces import (
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

__version__ = "1.0.0"

__all__ = [
    # Hyperparameters
    "HyperparameterConfig",
    "ModelHyperparameters",
    "OptimizerHyperparameters",
    "SchedulerHyperparameters",
    "TrainingHyperparameters",
    "RobustnessHyperparameters",
    "ExplainabilityHyperparameters",
    "TriObjectiveHyperparameters",
    "ActivationType",
    "OptimizerType",
    "SchedulerType",
    "NormalizationType",
    "AttackType",
    "validate_config",
    # Search Spaces
    "SearchSpaceConfig",
    "SearchSpaceFactory",
    "suggest_model_hyperparameters",
    "suggest_optimizer_hyperparameters",
    "suggest_scheduler_hyperparameters",
    "suggest_training_hyperparameters",
    "suggest_robustness_hyperparameters",
    "suggest_explainability_hyperparameters",
    "suggest_tri_objective_hyperparameters",
    "suggest_full_config",
    # Objectives
    "ObjectiveMetrics",
    "ObjectiveType",
    "OptimizationDirection",
    "SingleObjective",
    "AccuracyObjective",
    "RobustnessObjective",
    "ExplainabilityObjective",
    "WeightedSumObjective",
    "TchebycheffObjective",
    "AugmentedTchebycheffObjective",
    "PBIObjective",
    "DynamicWeightAdjuster",
    "ParetoFrontTracker",
    "create_objective_function",
    # Pruners
    "PerformanceBasedPruner",
    "ResourceAwarePruner",
    "MultiObjectivePruner",
    "AdaptivePruner",
    "HybridPruner",
    "create_pruner",
    "get_default_pruner_for_objective",
    # HPO Trainer
    "HPOTrainer",
    "HPOManager",
    "create_hpo_trainer_from_config",
]
