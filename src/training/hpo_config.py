"""
Hyperparameter Optimization Configuration for TRADES Training.

This module defines search spaces, pruning strategies, and optimization
configurations for the tri-objective robust XAI framework.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


class SearchSpaceType(Enum):
    """Types of hyperparameter search spaces."""

    CATEGORICAL = "categorical"
    FLOAT = "float"
    INT = "int"
    LOG_FLOAT = "log_float"
    LOG_INT = "log_int"


class PrunerType(Enum):
    """Available pruner types for early stopping."""

    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    THRESHOLD = "threshold"
    NONE = "none"


class SamplerType(Enum):
    """Available sampler types for hyperparameter search."""

    TPE = "tpe"
    CMA_ES = "cma_es"
    RANDOM = "random"
    GRID = "grid"
    NSGAII = "nsgaii"  # For multi-objective
    MOTPE = "motpe"  # For multi-objective


@dataclass(frozen=True)
class SearchSpace:
    """
    Definition of a single hyperparameter search space.

    Attributes:
        name: Hyperparameter name
        space_type: Type of search space
        low: Lower bound (for numeric types)
        high: Upper bound (for numeric types)
        choices: List of choices (for categorical type)
        step: Step size for discrete spaces
        log: Whether to use log scale
    """

    name: str
    space_type: SearchSpaceType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    step: Optional[float] = None
    log: bool = False

    def __post_init__(self) -> None:
        """Validate search space configuration."""
        if self.space_type == SearchSpaceType.CATEGORICAL:
            if self.choices is None or len(self.choices) == 0:
                raise ValueError(f"Categorical space '{self.name}' requires choices")
        elif self.space_type in (
            SearchSpaceType.FLOAT,
            SearchSpaceType.INT,
            SearchSpaceType.LOG_FLOAT,
            SearchSpaceType.LOG_INT,
        ):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Numeric space '{self.name}' requires low and high bounds"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"Invalid bounds for '{self.name}': low ({self.low}) >= high ({self.high})"
                )


@dataclass
class ObjectiveWeights:
    """
    Weights for multi-objective optimization.

    The tri-objective framework optimizes:
    1. Robust accuracy (adversarial performance)
    2. Clean accuracy (standard performance)
    3. Cross-site AUROC (generalization)

    Weights must sum to 1.0.
    """

    robust_accuracy: float = 0.4
    clean_accuracy: float = 0.3
    cross_site_auroc: float = 0.3

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0."""
        total = self.robust_accuracy + self.clean_accuracy + self.cross_site_auroc
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Objective weights must sum to 1.0, got {total:.4f}")

    def compute_weighted_score(
        self, robust_acc: float, clean_acc: float, cross_site_auroc: float
    ) -> float:
        """
        Compute weighted objective score.

        Args:
            robust_acc: Robust accuracy [0, 1]
            clean_acc: Clean accuracy [0, 1]
            cross_site_auroc: Cross-site AUROC [0, 1]

        Returns:
            Weighted composite score
        """
        return (
            self.robust_accuracy * robust_acc
            + self.clean_accuracy * clean_acc
            + self.cross_site_auroc * cross_site_auroc
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "robust_accuracy": self.robust_accuracy,
            "clean_accuracy": self.clean_accuracy,
            "cross_site_auroc": self.cross_site_auroc,
        }


@dataclass
class PrunerConfig:
    """
    Configuration for trial pruning (early stopping).

    Attributes:
        pruner_type: Type of pruner to use
        n_startup_trials: Number of trials before pruning starts
        n_warmup_steps: Number of steps before pruning in each trial
        interval_steps: Steps between pruning checks
        min_resource: Minimum resource for Hyperband
        max_resource: Maximum resource for Hyperband
        reduction_factor: Reduction factor for Hyperband
        percentile: Percentile threshold for PercentilePruner
    """

    pruner_type: PrunerType = PrunerType.MEDIAN
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    interval_steps: int = 1
    min_resource: int = 1
    max_resource: int = 100
    reduction_factor: int = 3
    percentile: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pruner_type": self.pruner_type.value,
            "n_startup_trials": self.n_startup_trials,
            "n_warmup_steps": self.n_warmup_steps,
            "interval_steps": self.interval_steps,
            "min_resource": self.min_resource,
            "max_resource": self.max_resource,
            "reduction_factor": self.reduction_factor,
            "percentile": self.percentile,
        }


@dataclass
class SamplerConfig:
    """
    Configuration for hyperparameter sampler.

    Attributes:
        sampler_type: Type of sampler to use
        seed: Random seed for reproducibility
        n_startup_trials: Trials before TPE starts
        multivariate: Whether to use multivariate TPE
        constant_liar: Use constant liar for parallel optimization
    """

    sampler_type: SamplerType = SamplerType.TPE
    seed: Optional[int] = 42
    n_startup_trials: int = 10
    multivariate: bool = True
    constant_liar: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sampler_type": self.sampler_type.value,
            "seed": self.seed,
            "n_startup_trials": self.n_startup_trials,
            "multivariate": self.multivariate,
            "constant_liar": self.constant_liar,
        }


@dataclass
class TRADESSearchSpace:
    """
    TRADES-specific search space for hyperparameter optimization.

    Search spaces defined per dissertation requirements:
    - β (beta): TRADES regularization strength ∈ [3.0, 10.0]
    - ε (epsilon): Perturbation budget ∈ {4/255, 6/255, 8/255}
    - Learning rate ∈ [1e-4, 1e-3]

    Additional spaces for comprehensive optimization.
    """

    # Core TRADES parameters
    beta: SearchSpace = field(
        default_factory=lambda: SearchSpace(
            name="beta",
            space_type=SearchSpaceType.FLOAT,
            low=3.0,
            high=10.0,
        )
    )

    epsilon: SearchSpace = field(
        default_factory=lambda: SearchSpace(
            name="epsilon",
            space_type=SearchSpaceType.CATEGORICAL,
            choices=(4 / 255, 6 / 255, 8 / 255),
        )
    )

    learning_rate: SearchSpace = field(
        default_factory=lambda: SearchSpace(
            name="learning_rate",
            space_type=SearchSpaceType.LOG_FLOAT,
            low=1e-4,
            high=1e-3,
            log=True,
        )
    )

    # Additional tunable parameters
    weight_decay: SearchSpace = field(
        default_factory=lambda: SearchSpace(
            name="weight_decay",
            space_type=SearchSpaceType.LOG_FLOAT,
            low=1e-5,
            high=1e-3,
            log=True,
        )
    )

    step_size: SearchSpace = field(
        default_factory=lambda: SearchSpace(
            name="step_size",
            space_type=SearchSpaceType.FLOAT,
            low=0.003,
            high=0.01,
        )
    )

    num_steps: SearchSpace = field(
        default_factory=lambda: SearchSpace(
            name="num_steps",
            space_type=SearchSpaceType.CATEGORICAL,
            choices=(7, 10, 15, 20),
        )
    )

    def get_core_spaces(self) -> List[SearchSpace]:
        """Get core TRADES search spaces (β, ε, lr)."""
        return [self.beta, self.epsilon, self.learning_rate]

    def get_all_spaces(self) -> List[SearchSpace]:
        """Get all search spaces including auxiliary parameters."""
        return [
            self.beta,
            self.epsilon,
            self.learning_rate,
            self.weight_decay,
            self.step_size,
            self.num_steps,
        ]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert search spaces to dictionary format."""
        spaces = {}
        for space in self.get_all_spaces():
            # Normalize log_float to float, log_int to int
            space_type = space.space_type.value
            if space_type == "log_float":
                space_type = "float"
            elif space_type == "log_int":
                space_type = "int"

            spaces[space.name] = {
                "type": space_type,
                "low": space.low,
                "high": space.high,
                "choices": list(space.choices) if space.choices else None,
                "log": space.log,
            }
        return spaces


@dataclass
class HPOConfig:
    """
    Complete HPO configuration for TRADES optimization.

    This configuration encapsulates all settings for running
    hyperparameter optimization studies.

    Attributes:
        study_name: Name of the Optuna study
        direction: Optimization direction ("maximize" or "minimize")
        n_trials: Number of trials to run
        timeout: Maximum time in seconds
        search_space: TRADES search space configuration
        objective_weights: Weights for multi-objective optimization
        pruner_config: Pruning configuration
        sampler_config: Sampler configuration
        n_jobs: Number of parallel jobs (-1 for all cores)
        storage_url: Optuna storage URL for persistence
        load_if_exists: Whether to load existing study
        output_dir: Directory for saving results
    """

    study_name: str = "trades_hpo_study"
    direction: str = "maximize"
    n_trials: int = 50
    timeout: Optional[int] = None
    search_space: TRADESSearchSpace = field(default_factory=TRADESSearchSpace)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    pruner_config: PrunerConfig = field(default_factory=PrunerConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    n_jobs: int = 1
    storage_url: Optional[str] = None
    load_if_exists: bool = True
    output_dir: Path = field(default_factory=lambda: Path("results/hpo"))

    # Training constraints
    min_epochs_per_trial: int = 10
    max_epochs_per_trial: int = 50
    early_stopping_patience: int = 5

    # Validation settings
    val_frequency: int = 1
    report_frequency: int = 1

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"Direction must be 'maximize' or 'minimize', got '{self.direction}'"
            )

    def get_storage_path(self) -> str:
        """Get storage URL for Optuna study persistence."""
        if self.storage_url:
            return self.storage_url
        db_path = self.output_dir / f"{self.study_name}.db"
        return f"sqlite:///{db_path}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "study_name": self.study_name,
            "direction": self.direction,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "search_space": self.search_space.to_dict(),
            "objective_weights": self.objective_weights.to_dict(),
            "pruner_config": self.pruner_config.to_dict(),
            "sampler_config": self.sampler_config.to_dict(),
            "n_jobs": self.n_jobs,
            "output_dir": str(self.output_dir),
            "min_epochs_per_trial": self.min_epochs_per_trial,
            "max_epochs_per_trial": self.max_epochs_per_trial,
            "early_stopping_patience": self.early_stopping_patience,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save configuration to YAML file.

        Args:
            path: Optional path override

        Returns:
            Path where configuration was saved
        """
        if path is None:
            path = self.output_dir / f"{self.study_name}_config.yaml"

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        logger.info(f"Saved HPO configuration to {path}")
        return path

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HPOConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            HPOConfig instance
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        search_space = TRADESSearchSpace()  # Use defaults
        objective_weights = ObjectiveWeights(**config_dict.pop("objective_weights", {}))
        pruner_config = PrunerConfig(
            pruner_type=PrunerType(
                config_dict.pop("pruner_config", {}).get("pruner_type", "median")
            ),
            **{
                k: v
                for k, v in config_dict.pop("pruner_config", {}).items()
                if k != "pruner_type"
            },
        )
        sampler_config = SamplerConfig(
            sampler_type=SamplerType(
                config_dict.pop("sampler_config", {}).get("sampler_type", "tpe")
            ),
            **{
                k: v
                for k, v in config_dict.pop("sampler_config", {}).items()
                if k != "sampler_type"
            },
        )

        # Remove search_space from dict as we use defaults
        config_dict.pop("search_space", None)

        return cls(
            search_space=search_space,
            objective_weights=objective_weights,
            pruner_config=pruner_config,
            sampler_config=sampler_config,
            **config_dict,
        )


def create_default_hpo_config(
    study_name: str = "trades_hpo_study",
    n_trials: int = 50,
    output_dir: Union[str, Path] = "results/hpo",
    **kwargs,
) -> HPOConfig:
    """
    Factory function to create HPO configuration with defaults.

    Args:
        study_name: Name of the study
        n_trials: Number of trials
        output_dir: Output directory
        **kwargs: Additional configuration overrides

    Returns:
        HPOConfig instance
    """
    return HPOConfig(
        study_name=study_name, n_trials=n_trials, output_dir=Path(output_dir), **kwargs
    )


def create_extended_search_space() -> TRADESSearchSpace:
    """
    Create extended search space with additional parameters.

    This includes parameters for more comprehensive optimization
    beyond the core TRADES hyperparameters.
    """
    return TRADESSearchSpace(
        beta=SearchSpace(
            name="beta",
            space_type=SearchSpaceType.FLOAT,
            low=1.0,
            high=15.0,
        ),
        epsilon=SearchSpace(
            name="epsilon",
            space_type=SearchSpaceType.CATEGORICAL,
            choices=(2 / 255, 4 / 255, 6 / 255, 8 / 255, 12 / 255),
        ),
        learning_rate=SearchSpace(
            name="learning_rate",
            space_type=SearchSpaceType.LOG_FLOAT,
            low=1e-5,
            high=1e-2,
            log=True,
        ),
        weight_decay=SearchSpace(
            name="weight_decay",
            space_type=SearchSpaceType.LOG_FLOAT,
            low=1e-6,
            high=1e-2,
            log=True,
        ),
        step_size=SearchSpace(
            name="step_size",
            space_type=SearchSpaceType.FLOAT,
            low=0.001,
            high=0.02,
        ),
        num_steps=SearchSpace(
            name="num_steps",
            space_type=SearchSpaceType.CATEGORICAL,
            choices=(5, 7, 10, 15, 20, 30),
        ),
    )
