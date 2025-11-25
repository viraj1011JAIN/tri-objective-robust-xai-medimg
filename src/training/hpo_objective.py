"""
Objective Function for TRADES Hyperparameter Optimization.

This module implements the tri-objective function that combines:
1. Robust accuracy (adversarial performance)
2. Clean accuracy (standard performance)
3. Cross-site AUROC (generalization capability)

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

try:
    import optuna
    from optuna.trial import FrozenTrial, Trial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any
    FrozenTrial = Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = Any
    DataLoader = Any

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked during optimization."""

    ROBUST_ACCURACY = "robust_accuracy"
    CLEAN_ACCURACY = "clean_accuracy"
    CROSS_SITE_AUROC = "cross_site_auroc"
    LOSS = "loss"
    TRADES_LOSS = "trades_loss"
    NATURAL_LOSS = "natural_loss"
    ROBUST_LOSS = "robust_loss"
    EXPLANATION_STABILITY = "explanation_stability"


@dataclass
class TrialMetrics:
    """
    Container for all metrics collected during a trial.

    Attributes:
        robust_accuracy: Accuracy under adversarial attack
        clean_accuracy: Standard accuracy on clean data
        cross_site_auroc: AUROC for cross-site generalization
        loss: Total training loss
        natural_loss: Loss on natural examples
        robust_loss: Loss on adversarial examples
        explanation_stability: XAI explanation stability score
        epoch: Current epoch
        step: Current training step
        timestamp: Metric collection timestamp
    """

    robust_accuracy: float = 0.0
    clean_accuracy: float = 0.0
    cross_site_auroc: float = 0.0
    loss: float = float("inf")
    natural_loss: float = float("inf")
    robust_loss: float = float("inf")
    explanation_stability: float = 0.0
    epoch: int = 0
    step: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "robust_accuracy": self.robust_accuracy,
            "clean_accuracy": self.clean_accuracy,
            "cross_site_auroc": self.cross_site_auroc,
            "loss": self.loss,
            "natural_loss": self.natural_loss,
            "robust_loss": self.robust_loss,
            "explanation_stability": self.explanation_stability,
            "epoch": self.epoch,
            "step": self.step,
            "timestamp": self.timestamp,
        }

    def is_valid(self) -> bool:
        """Check if metrics are valid (no NaN or Inf)."""
        values = [
            self.robust_accuracy,
            self.clean_accuracy,
            self.cross_site_auroc,
            self.loss,
        ]
        return all(np.isfinite(v) for v in values if v != float("inf"))


@dataclass
class ObjectiveConfig:
    """
    Configuration for objective function computation.

    Attributes:
        weights: Dictionary of metric weights
        direction: Optimization direction ("maximize" or "minimize")
        use_log_scale: Whether to use log scale for loss terms
        clip_range: Range to clip metrics to avoid extreme values
        penalty_invalid: Penalty value for invalid metrics
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "robust_accuracy": 0.4,
            "clean_accuracy": 0.3,
            "cross_site_auroc": 0.3,
        }
    )
    direction: str = "maximize"
    use_log_scale: bool = False
    clip_range: Tuple[float, float] = (0.0, 1.0)
    penalty_invalid: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        total_weight = sum(self.weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight:.4f}")


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.

    Provides interface for computing optimization objectives
    from trial metrics.
    """

    @abstractmethod
    def __call__(self, metrics: TrialMetrics) -> float:
        """
        Compute objective value from metrics.

        Args:
            metrics: Trial metrics

        Returns:
            Objective value
        """
        pass

    @abstractmethod
    def get_intermediate_value(
        self,
        metrics: TrialMetrics,
        epoch: int,
    ) -> float:
        """
        Get intermediate value for pruning decisions.

        Args:
            metrics: Current trial metrics
            epoch: Current epoch

        Returns:
            Intermediate objective value
        """
        pass


class WeightedTriObjective(ObjectiveFunction):
    """
    Weighted tri-objective function for TRADES HPO.

    Computes a weighted combination of:
    - Robust accuracy (adversarial performance)
    - Clean accuracy (standard performance)
    - Cross-site AUROC (generalization)

    Default weights: 0.4 × robust + 0.3 × clean + 0.3 × AUROC
    """

    def __init__(
        self,
        config: Optional[ObjectiveConfig] = None,
        robust_weight: float = 0.4,
        clean_weight: float = 0.3,
        auroc_weight: float = 0.3,
    ) -> None:
        """
        Initialize weighted tri-objective.

        Args:
            config: Optional objective configuration
            robust_weight: Weight for robust accuracy
            clean_weight: Weight for clean accuracy
            auroc_weight: Weight for cross-site AUROC
        """
        if config is not None:
            self.config = config
        else:
            self.config = ObjectiveConfig(
                weights={
                    "robust_accuracy": robust_weight,
                    "clean_accuracy": clean_weight,
                    "cross_site_auroc": auroc_weight,
                }
            )

        self._validate_weights()

        # Cache for intermediate computations
        self._history: List[Tuple[int, float]] = []

    def _validate_weights(self) -> None:
        """Validate that weights are properly configured."""
        required_keys = {"robust_accuracy", "clean_accuracy", "cross_site_auroc"}
        if not required_keys.issubset(self.config.weights.keys()):
            missing = required_keys - set(self.config.weights.keys())
            raise ValueError(f"Missing required weight keys: {missing}")

    def __call__(self, metrics: TrialMetrics) -> float:
        """
        Compute weighted objective value.

        Args:
            metrics: Trial metrics containing accuracy values

        Returns:
            Weighted objective score in [0, 1]
        """
        if not metrics.is_valid():
            logger.warning("Invalid metrics detected, returning penalty value")
            return self.config.penalty_invalid

        # Clip values to valid range
        robust = np.clip(
            metrics.robust_accuracy,
            self.config.clip_range[0],
            self.config.clip_range[1],
        )
        clean = np.clip(
            metrics.clean_accuracy, self.config.clip_range[0], self.config.clip_range[1]
        )
        auroc = np.clip(
            metrics.cross_site_auroc,
            self.config.clip_range[0],
            self.config.clip_range[1],
        )

        # Compute weighted combination
        objective = (
            self.config.weights["robust_accuracy"] * robust
            + self.config.weights["clean_accuracy"] * clean
            + self.config.weights["cross_site_auroc"] * auroc
        )

        return float(objective)

    def get_intermediate_value(
        self,
        metrics: TrialMetrics,
        epoch: int,
    ) -> float:
        """
        Get intermediate value for pruning.

        Uses running best to handle noisy intermediate values.

        Args:
            metrics: Current metrics
            epoch: Current epoch

        Returns:
            Intermediate objective value
        """
        current_value = self(metrics)
        self._history.append((epoch, current_value))

        # Return running best for stability
        return max(v for _, v in self._history)

    def reset_history(self) -> None:
        """Reset intermediate value history for new trial."""
        self._history.clear()

    @property
    def weights(self) -> Dict[str, float]:
        """Get current weight configuration."""
        return self.config.weights.copy()

    def get_component_contributions(self, metrics: TrialMetrics) -> Dict[str, float]:
        """
        Get individual component contributions to objective.

        Useful for analysis and understanding trade-offs.

        Args:
            metrics: Trial metrics

        Returns:
            Dictionary of component contributions
        """
        return {
            "robust_contribution": (
                self.config.weights["robust_accuracy"] * metrics.robust_accuracy
            ),
            "clean_contribution": (
                self.config.weights["clean_accuracy"] * metrics.clean_accuracy
            ),
            "auroc_contribution": (
                self.config.weights["cross_site_auroc"] * metrics.cross_site_auroc
            ),
        }


class AdaptiveWeightedObjective(ObjectiveFunction):
    """
    Adaptive objective that adjusts weights during optimization.

    Useful for finding Pareto-optimal solutions by varying
    the weight emphasis during the study.
    """

    def __init__(
        self,
        base_weights: Dict[str, float],
        adaptation_strategy: str = "linear",
        target_iterations: int = 50,
    ) -> None:
        """
        Initialize adaptive objective.

        Args:
            base_weights: Initial weight configuration
            adaptation_strategy: How to adapt weights
            target_iterations: Expected number of iterations
        """
        self.base_weights = base_weights.copy()
        self.current_weights = base_weights.copy()
        self.strategy = adaptation_strategy
        self.target_iterations = target_iterations
        self.iteration = 0

    def __call__(self, metrics: TrialMetrics) -> float:
        """Compute objective with current weights."""
        if not metrics.is_valid():
            return 0.0

        return (
            self.current_weights.get("robust_accuracy", 0.4) * metrics.robust_accuracy
            + self.current_weights.get("clean_accuracy", 0.3) * metrics.clean_accuracy
            + self.current_weights.get("cross_site_auroc", 0.3)
            * metrics.cross_site_auroc
        )

    def get_intermediate_value(
        self,
        metrics: TrialMetrics,
        epoch: int,
    ) -> float:
        """Get intermediate value."""
        return self(metrics)

    def update_weights(self, iteration: int) -> None:
        """
        Update weights based on iteration progress.

        Args:
            iteration: Current iteration number
        """
        self.iteration = iteration
        progress = iteration / max(self.target_iterations, 1)

        if self.strategy == "linear":
            # Linearly shift emphasis toward robustness
            self.current_weights = {
                "robust_accuracy": self.base_weights["robust_accuracy"]
                + 0.1 * progress,
                "clean_accuracy": self.base_weights["clean_accuracy"] - 0.05 * progress,
                "cross_site_auroc": self.base_weights["cross_site_auroc"]
                - 0.05 * progress,
            }
        elif self.strategy == "cyclic":
            # Cycle through different weight emphases
            cycle_pos = iteration % 3
            emphasis = ["robust_accuracy", "clean_accuracy", "cross_site_auroc"][
                cycle_pos
            ]
            self.current_weights = {
                k: 0.5 if k == emphasis else 0.25 for k in self.base_weights
            }

        # Normalize to sum to 1
        total = sum(self.current_weights.values())
        self.current_weights = {k: v / total for k, v in self.current_weights.items()}


class MultiObjectiveEvaluator:
    """
    Evaluator for multi-objective optimization scenarios.

    Supports Pareto frontier analysis and hypervolume computation
    for true multi-objective optimization.
    """

    def __init__(
        self,
        objectives: List[str] = None,
        reference_point: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize multi-objective evaluator.

        Args:
            objectives: List of objective names
            reference_point: Reference point for hypervolume
        """
        self.objectives = objectives or [
            "robust_accuracy",
            "clean_accuracy",
            "cross_site_auroc",
        ]
        self.reference_point = reference_point or np.zeros(len(self.objectives))
        self._pareto_front: List[np.ndarray] = []

    def extract_objectives(self, metrics: TrialMetrics) -> np.ndarray:
        """
        Extract objective values from metrics.

        Args:
            metrics: Trial metrics

        Returns:
            Array of objective values
        """
        return np.array([getattr(metrics, obj, 0.0) for obj in self.objectives])

    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Check if solution a dominates solution b.

        A dominates B if A is at least as good in all objectives
        and strictly better in at least one.

        Args:
            a: First solution
            b: Second solution

        Returns:
            True if a dominates b
        """
        return all(a >= b) and any(a > b)

    def update_pareto_front(self, solution: np.ndarray) -> bool:
        """
        Update Pareto front with new solution.

        Args:
            solution: New solution to consider

        Returns:
            True if solution was added to front
        """
        # Check if new solution is dominated by any existing
        for existing in self._pareto_front:
            if self.dominates(existing, solution):
                return False

        # Remove solutions dominated by new solution
        self._pareto_front = [
            existing
            for existing in self._pareto_front
            if not self.dominates(solution, existing)
        ]

        self._pareto_front.append(solution.copy())
        return True

    def get_pareto_front(self) -> np.ndarray:
        """Get current Pareto front."""
        if not self._pareto_front:
            return np.array([])
        return np.array(self._pareto_front)

    def compute_hypervolume(self) -> float:
        """
        Compute hypervolume indicator.

        Higher hypervolume indicates better Pareto front quality.

        Returns:
            Hypervolume value
        """
        if len(self._pareto_front) == 0:
            return 0.0

        front = self.get_pareto_front()

        # Simple 2D/3D hypervolume computation
        # For production, use a proper hypervolume library
        if front.shape[1] == 2:
            return self._compute_2d_hypervolume(front)
        elif front.shape[1] == 3:
            return self._compute_3d_hypervolume_approx(front)

        logger.warning(f"Hypervolume computation not implemented for {front.shape[1]}D")
        return 0.0

    def _compute_2d_hypervolume(self, front: np.ndarray) -> float:
        """Compute 2D hypervolume using sweep line."""
        # Sort by first objective (descending)
        sorted_front = front[front[:, 0].argsort()[::-1]]

        hypervolume = 0.0
        prev_y = self.reference_point[1]

        for point in sorted_front:
            x_diff = point[0] - self.reference_point[0]
            y_diff = point[1] - prev_y
            hypervolume += x_diff * y_diff
            prev_y = point[1]

        return max(0.0, hypervolume)

    def _compute_3d_hypervolume_approx(self, front: np.ndarray) -> float:
        """Approximate 3D hypervolume using Monte Carlo."""
        n_samples = 10000

        # Determine bounding box
        mins = np.minimum(front.min(axis=0), self.reference_point)
        maxs = front.max(axis=0)

        # Sample random points
        samples = np.random.uniform(mins, maxs, size=(n_samples, 3))

        # Count dominated points
        dominated = 0
        for sample in samples:
            if any(self.dominates(point, sample) for point in front):
                dominated += 1

        # Estimate hypervolume
        box_volume = np.prod(maxs - mins)
        return box_volume * dominated / n_samples


def create_optuna_objective(
    objective_fn: ObjectiveFunction,
    trainer_factory: Callable[[Trial], Any],
    evaluator: Optional[Callable[[Any], TrialMetrics]] = None,
) -> Callable[[Trial], float]:
    """
    Create Optuna-compatible objective function.

    Factory function that wraps our objective function
    for use with Optuna's optimization framework.

    Args:
        objective_fn: Our objective function
        trainer_factory: Factory to create trainer from trial
        evaluator: Optional custom evaluator function

    Returns:
        Optuna objective function
    """

    def optuna_objective(trial: Trial) -> float:
        """Optuna objective function wrapper."""
        try:
            # Create trainer with trial hyperparameters
            trainer = trainer_factory(trial)

            # Run training and get metrics
            if evaluator is not None:
                metrics = evaluator(trainer)
            else:
                metrics = trainer.train_and_evaluate()

            # Reset objective history for new trial
            if hasattr(objective_fn, "reset_history"):
                objective_fn.reset_history()

            # Compute objective value
            objective_value = objective_fn(metrics)

            logger.info(
                f"Trial {trial.number}: objective={objective_value:.4f}, "
                f"robust={metrics.robust_accuracy:.4f}, "
                f"clean={metrics.clean_accuracy:.4f}, "
                f"auroc={metrics.cross_site_auroc:.4f}"
            )

            return objective_value

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned(f"Trial failed: {e}")

    return optuna_objective


def create_multi_objective_optuna(
    objectives: List[ObjectiveFunction],
    trainer_factory: Callable[[Trial], Any],
) -> Callable[[Trial], List[float]]:
    """
    Create multi-objective Optuna function.

    For use with Optuna's multi-objective samplers
    (NSGA-II, MOTPE).

    Args:
        objectives: List of objective functions
        trainer_factory: Factory to create trainer

    Returns:
        Multi-objective Optuna function
    """

    def multi_objective(trial: Trial) -> List[float]:
        """Multi-objective function for Optuna."""
        try:
            trainer = trainer_factory(trial)
            metrics = trainer.train_and_evaluate()

            return [obj(metrics) for obj in objectives]

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned(f"Trial failed: {e}")

    return multi_objective
