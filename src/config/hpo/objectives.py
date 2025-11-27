"""
Optimization Objective Functions for Hyperparameter Optimization.

This module defines various objective functions for multi-objective optimization,
including single objectives (accuracy, robustness, explainability) and combined
objectives using different strategies (weighted sum, Pareto, scalarization).

Author: Viraj Jain
Date: November 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.trial import Trial


class ObjectiveType(str, Enum):
    """Types of optimization objectives."""

    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"
    WEIGHTED_SUM = "weighted_sum"
    PARETO = "pareto"
    TCHEBYCHEFF = "tchebycheff"
    AUGMENTED_TCHEBYCHEFF = "augmented_tchebycheff"
    PBI = "pbi"  # Penalty-based Boundary Intersection


class OptimizationDirection(str, Enum):
    """Optimization direction for objectives."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class ObjectiveMetrics:
    """
    Container for objective metrics.

    Attributes:
        accuracy: Classification accuracy (0-1)
        robustness: Robustness score (0-1)
        explainability: Explainability score (0-1)
        loss: Training loss
        val_loss: Validation loss
        robust_accuracy: Accuracy under adversarial attack
        clean_accuracy: Accuracy on clean examples
        xai_coherence: XAI coherence score
        xai_faithfulness: XAI faithfulness score
        training_time: Training time in seconds
        inference_time: Inference time in seconds
    """

    accuracy: float = 0.0
    robustness: float = 0.0
    explainability: float = 0.0
    loss: float = float("inf")
    val_loss: float = float("inf")
    robust_accuracy: float = 0.0
    clean_accuracy: float = 0.0
    xai_coherence: float = 0.0
    xai_faithfulness: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "robustness": self.robustness,
            "explainability": self.explainability,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "robust_accuracy": self.robust_accuracy,
            "clean_accuracy": self.clean_accuracy,
            "xai_coherence": self.xai_coherence,
            "xai_faithfulness": self.xai_faithfulness,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
        }


class SingleObjective:
    """
    Single objective function for optimization.

    This class defines a single objective that can be optimized independently.
    """

    def __init__(
        self,
        objective_type: ObjectiveType,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        name: Optional[str] = None,
    ):
        """
        Initialize single objective.

        Args:
            objective_type: Type of objective
            direction: Optimization direction
            name: Optional name for the objective
        """
        self.objective_type = objective_type
        self.direction = direction
        self.name = name or objective_type.value

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate objective value from metrics.

        Args:
            metrics: Objective metrics

        Returns:
            Objective value
        """
        if self.objective_type == ObjectiveType.ACCURACY:
            return metrics.accuracy
        elif self.objective_type == ObjectiveType.ROBUSTNESS:
            return metrics.robustness
        elif self.objective_type == ObjectiveType.EXPLAINABILITY:
            return metrics.explainability
        else:
            raise ValueError(f"Invalid objective type: {self.objective_type}")

    def is_better(self, value1: float, value2: float) -> bool:
        """
        Check if value1 is better than value2 according to optimization direction.

        Args:
            value1: First value
            value2: Second value

        Returns:
            True if value1 is better than value2
        """
        if self.direction == OptimizationDirection.MAXIMIZE:
            return value1 > value2
        else:
            return value1 < value2


class AccuracyObjective(SingleObjective):
    """Objective for maximizing classification accuracy."""

    def __init__(self):
        super().__init__(
            ObjectiveType.ACCURACY, OptimizationDirection.MAXIMIZE, "accuracy"
        )

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate accuracy objective.

        Args:
            metrics: Objective metrics

        Returns:
            Accuracy score
        """
        return metrics.accuracy


class RobustnessObjective(SingleObjective):
    """Objective for maximizing adversarial robustness."""

    def __init__(self, robustness_metric: str = "robust_accuracy"):
        """
        Initialize robustness objective.

        Args:
            robustness_metric: Which robustness metric to use
        """
        super().__init__(
            ObjectiveType.ROBUSTNESS, OptimizationDirection.MAXIMIZE, "robustness"
        )
        self.robustness_metric = robustness_metric

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate robustness objective.

        Args:
            metrics: Objective metrics

        Returns:
            Robustness score
        """
        if self.robustness_metric == "robust_accuracy":
            return metrics.robust_accuracy
        elif self.robustness_metric == "robustness":
            return metrics.robustness
        else:
            # Combine multiple robustness metrics
            return 0.5 * metrics.robust_accuracy + 0.5 * metrics.robustness


class ExplainabilityObjective(SingleObjective):
    """Objective for maximizing explainability."""

    def __init__(self, coherence_weight: float = 0.5, faithfulness_weight: float = 0.5):
        """
        Initialize explainability objective.

        Args:
            coherence_weight: Weight for coherence metric
            faithfulness_weight: Weight for faithfulness metric
        """
        super().__init__(
            ObjectiveType.EXPLAINABILITY,
            OptimizationDirection.MAXIMIZE,
            "explainability",
        )
        self.coherence_weight = coherence_weight
        self.faithfulness_weight = faithfulness_weight

        # Normalize weights
        total = coherence_weight + faithfulness_weight
        self.coherence_weight /= total
        self.faithfulness_weight /= total

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate explainability objective.

        Args:
            metrics: Objective metrics

        Returns:
            Explainability score
        """
        return (
            self.coherence_weight * metrics.xai_coherence
            + self.faithfulness_weight * metrics.xai_faithfulness
        )


class WeightedSumObjective:
    """
    Weighted sum multi-objective optimization.

    This combines multiple objectives using a weighted sum approach.
    """

    def __init__(
        self,
        accuracy_weight: float = 1.0,
        robustness_weight: float = 1.0,
        explainability_weight: float = 1.0,
        normalize_weights: bool = True,
    ):
        """
        Initialize weighted sum objective.

        Args:
            accuracy_weight: Weight for accuracy objective
            robustness_weight: Weight for robustness objective
            explainability_weight: Weight for explainability objective
            normalize_weights: Whether to normalize weights to sum to 1
        """
        self.accuracy_weight = accuracy_weight
        self.robustness_weight = robustness_weight
        self.explainability_weight = explainability_weight

        if normalize_weights:
            total = accuracy_weight + robustness_weight + explainability_weight
            self.accuracy_weight /= total
            self.robustness_weight /= total
            self.explainability_weight /= total

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate weighted sum objective.

        Args:
            metrics: Objective metrics

        Returns:
            Weighted sum of objectives
        """
        return (
            self.accuracy_weight * metrics.accuracy
            + self.robustness_weight * metrics.robustness
            + self.explainability_weight * metrics.explainability
        )

    def get_weights(self) -> Tuple[float, float, float]:
        """Get current weights."""
        return (
            self.accuracy_weight,
            self.robustness_weight,
            self.explainability_weight,
        )

    def set_weights(
        self,
        accuracy_weight: float,
        robustness_weight: float,
        explainability_weight: float,
        normalize: bool = True,
    ):
        """
        Set new weights.

        Args:
            accuracy_weight: Weight for accuracy objective
            robustness_weight: Weight for robustness objective
            explainability_weight: Weight for explainability objective
            normalize: Whether to normalize weights
        """
        self.accuracy_weight = accuracy_weight
        self.robustness_weight = robustness_weight
        self.explainability_weight = explainability_weight

        if normalize:
            total = accuracy_weight + robustness_weight + explainability_weight
            if total > 0:
                self.accuracy_weight /= total
                self.robustness_weight /= total
                self.explainability_weight /= total


class TchebycheffObjective:
    """
    Tchebycheff scalarization for multi-objective optimization.

    This uses the Tchebycheff approach to convert multi-objective to single-objective.
    """

    def __init__(
        self,
        reference_point: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize Tchebycheff objective.

        Args:
            reference_point: Reference point (ideal point) for objectives
            weights: Weight vector for objectives
        """
        self.reference_point = (
            reference_point
            if reference_point is not None
            else np.array([1.0, 1.0, 1.0])
        )
        self.weights = weights if weights is not None else np.array([1.0, 1.0, 1.0])
        self.weights = self.weights / np.sum(self.weights)  # Normalize

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate Tchebycheff objective.

        Args:
            metrics: Objective metrics

        Returns:
            Tchebycheff scalarized value (to minimize)
        """
        objectives = np.array(
            [metrics.accuracy, metrics.robustness, metrics.explainability]
        )
        weighted_diff = self.weights * np.abs(self.reference_point - objectives)
        return np.max(weighted_diff)

    def update_reference_point(self, new_point: np.ndarray):
        """Update reference point."""
        self.reference_point = new_point


class AugmentedTchebycheffObjective:
    """
    Augmented Tchebycheff scalarization.

    This adds a sum term to the standard Tchebycheff to ensure Pareto optimality.
    """

    def __init__(
        self,
        reference_point: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        augmentation_factor: float = 0.05,
    ):
        """
        Initialize augmented Tchebycheff objective.

        Args:
            reference_point: Reference point (ideal point) for objectives
            weights: Weight vector for objectives
            augmentation_factor: Augmentation term coefficient
        """
        self.reference_point = (
            reference_point
            if reference_point is not None
            else np.array([1.0, 1.0, 1.0])
        )
        self.weights = weights if weights is not None else np.array([1.0, 1.0, 1.0])
        self.weights = self.weights / np.sum(self.weights)
        self.augmentation_factor = augmentation_factor

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate augmented Tchebycheff objective.

        Args:
            metrics: Objective metrics

        Returns:
            Augmented Tchebycheff value (to minimize)
        """
        objectives = np.array(
            [metrics.accuracy, metrics.robustness, metrics.explainability]
        )
        weighted_diff = self.weights * np.abs(self.reference_point - objectives)
        tchebycheff_term = np.max(weighted_diff)
        augmentation_term = self.augmentation_factor * np.sum(weighted_diff)
        return tchebycheff_term + augmentation_term


class PBIObjective:
    """
    Penalty-based Boundary Intersection (PBI) scalarization.

    This decomposes objectives into distance along and perpendicular to weight vector.
    """

    def __init__(
        self, weights: Optional[np.ndarray] = None, penalty_parameter: float = 5.0
    ):
        """
        Initialize PBI objective.

        Args:
            weights: Weight vector for objectives
            penalty_parameter: Penalty parameter for perpendicular distance
        """
        self.weights = weights if weights is not None else np.array([1.0, 1.0, 1.0])
        self.weights = self.weights / np.linalg.norm(
            self.weights
        )  # Normalize to unit vector
        self.penalty_parameter = penalty_parameter

    def evaluate(self, metrics: ObjectiveMetrics) -> float:
        """
        Evaluate PBI objective.

        Args:
            metrics: Objective metrics

        Returns:
            PBI scalarized value (to minimize)
        """
        objectives = np.array(
            [metrics.accuracy, metrics.robustness, metrics.explainability]
        )

        # Distance along weight vector (d1)
        d1 = np.dot(objectives, self.weights)

        # Distance perpendicular to weight vector (d2)
        projection = d1 * self.weights
        perpendicular = objectives - projection
        d2 = np.linalg.norm(perpendicular)

        # PBI value (to minimize, so negate d1)
        return -d1 + self.penalty_parameter * d2


class DynamicWeightAdjuster:
    """
    Dynamic weight adjustment for multi-objective optimization.

    This adjusts objective weights during optimization based on progress.
    """

    def __init__(
        self,
        initial_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        adjustment_strategy: str = "performance_based",
        adjustment_rate: float = 0.1,
    ):
        """
        Initialize dynamic weight adjuster.

        Args:
            initial_weights: Initial weights for objectives
            adjustment_strategy: Strategy for adjusting weights
            adjustment_rate: Rate of weight adjustment
        """
        self.current_weights = np.array(initial_weights)
        self.current_weights = self.current_weights / np.sum(self.current_weights)
        self.adjustment_strategy = adjustment_strategy
        self.adjustment_rate = adjustment_rate
        self.history: List[ObjectiveMetrics] = []

    def update_weights(self, metrics: ObjectiveMetrics) -> Tuple[float, float, float]:
        """
        Update weights based on recent performance.

        Args:
            metrics: Recent objective metrics

        Returns:
            Updated weights
        """
        self.history.append(metrics)

        if len(self.history) < 2:
            return tuple(self.current_weights)

        if self.adjustment_strategy == "performance_based":
            self._adjust_performance_based()
        elif self.adjustment_strategy == "gradient_based":
            self._adjust_gradient_based()
        elif self.adjustment_strategy == "variance_based":
            self._adjust_variance_based()

        return tuple(self.current_weights)

    def _adjust_performance_based(self):
        """Adjust weights based on relative performance of objectives."""
        recent_metrics = self.history[-10:]  # Look at last 10 evaluations

        # Calculate average performance for each objective
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_robustness = np.mean([m.robustness for m in recent_metrics])
        avg_explainability = np.mean([m.explainability for m in recent_metrics])

        # Increase weight for underperforming objectives
        performances = np.array([avg_accuracy, avg_robustness, avg_explainability])
        normalized_perf = performances / np.sum(performances)

        # Inverse proportional adjustment
        adjustment = (1.0 - normalized_perf) * self.adjustment_rate
        self.current_weights = self.current_weights + adjustment
        self.current_weights = np.maximum(self.current_weights, 0.1)  # Min weight
        self.current_weights = self.current_weights / np.sum(self.current_weights)

    def _adjust_gradient_based(self):
        """Adjust weights based on improvement gradients."""
        if len(self.history) < 5:
            return

        recent = self.history[-5:]

        # Calculate improvement rates
        acc_gradient = (recent[-1].accuracy - recent[0].accuracy) / len(recent)
        rob_gradient = (recent[-1].robustness - recent[0].robustness) / len(recent)
        xai_gradient = (recent[-1].explainability - recent[0].explainability) / len(
            recent
        )

        gradients = np.array([acc_gradient, rob_gradient, xai_gradient])

        # Increase weight for objectives with slower improvement
        if np.sum(np.abs(gradients)) > 0:
            normalized_gradients = gradients / np.sum(np.abs(gradients))
            adjustment = (1.0 - np.abs(normalized_gradients)) * self.adjustment_rate
            self.current_weights = self.current_weights + adjustment
            self.current_weights = np.maximum(self.current_weights, 0.1)
            self.current_weights = self.current_weights / np.sum(self.current_weights)

    def _adjust_variance_based(self):
        """Adjust weights based on variance in objective values."""
        if len(self.history) < 10:
            return

        recent = self.history[-20:]

        # Calculate variance for each objective
        acc_variance = np.var([m.accuracy for m in recent])
        rob_variance = np.var([m.robustness for m in recent])
        xai_variance = np.var([m.explainability for m in recent])

        variances = np.array([acc_variance, rob_variance, xai_variance])

        # Increase weight for objectives with high variance (unstable)
        if np.sum(variances) > 0:
            normalized_variances = variances / np.sum(variances)
            adjustment = normalized_variances * self.adjustment_rate
            self.current_weights = self.current_weights + adjustment
            self.current_weights = np.maximum(self.current_weights, 0.1)
            self.current_weights = self.current_weights / np.sum(self.current_weights)


class ParetoFrontTracker:
    """
    Tracker for Pareto-optimal solutions in multi-objective optimization.
    """

    def __init__(self):
        """Initialize Pareto front tracker."""
        self.pareto_front: List[Tuple[ObjectiveMetrics, Dict[str, Any]]] = []

    def update(self, metrics: ObjectiveMetrics, config: Dict[str, Any]):
        """
        Update Pareto front with new solution.

        Args:
            metrics: Objective metrics for the solution
            config: Configuration corresponding to the metrics
        """
        objectives = np.array(
            [metrics.accuracy, metrics.robustness, metrics.explainability]
        )

        # Check if solution is dominated
        is_dominated = False
        to_remove = []

        for i, (pareto_metrics, _) in enumerate(self.pareto_front):
            pareto_objectives = np.array(
                [
                    pareto_metrics.accuracy,
                    pareto_metrics.robustness,
                    pareto_metrics.explainability,
                ]
            )

            # Check dominance (maximization)
            if np.all(pareto_objectives >= objectives) and np.any(
                pareto_objectives > objectives
            ):
                is_dominated = True
                break
            elif np.all(objectives >= pareto_objectives) and np.any(
                objectives > pareto_objectives
            ):
                to_remove.append(i)

        # Remove dominated solutions
        for i in sorted(to_remove, reverse=True):
            self.pareto_front.pop(i)

        # Add new solution if not dominated
        if not is_dominated:
            self.pareto_front.append((metrics, config))

    def get_pareto_front(self) -> List[Tuple[ObjectiveMetrics, Dict[str, Any]]]:
        """Get current Pareto front."""
        return self.pareto_front

    def get_best_by_objective(
        self, objective: str
    ) -> Tuple[ObjectiveMetrics, Dict[str, Any]]:
        """
        Get best solution for a specific objective.

        Args:
            objective: Objective name ('accuracy', 'robustness', or 'explainability')

        Returns:
            Best solution for the objective
        """
        if not self.pareto_front:
            raise ValueError("Pareto front is empty")

        if objective == "accuracy":
            return max(self.pareto_front, key=lambda x: x[0].accuracy)
        elif objective == "robustness":
            return max(self.pareto_front, key=lambda x: x[0].robustness)
        elif objective == "explainability":
            return max(self.pareto_front, key=lambda x: x[0].explainability)
        else:
            raise ValueError(f"Unknown objective: {objective}")


def create_objective_function(
    objective_type: str, **kwargs
) -> Callable[[ObjectiveMetrics], float]:
    """
    Factory function to create objective functions.

    Args:
        objective_type: Type of objective function
        **kwargs: Additional arguments for objective function

    Returns:
        Objective function
    """
    if objective_type == "accuracy":
        obj = AccuracyObjective()
    elif objective_type == "robustness":
        obj = RobustnessObjective(**kwargs)
    elif objective_type == "explainability":
        obj = ExplainabilityObjective(**kwargs)
    elif objective_type == "weighted_sum":
        obj = WeightedSumObjective(**kwargs)
    elif objective_type == "tchebycheff":
        obj = TchebycheffObjective(**kwargs)
    elif objective_type == "augmented_tchebycheff":
        obj = AugmentedTchebycheffObjective(**kwargs)
    elif objective_type == "pbi":
        obj = PBIObjective(**kwargs)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    return obj.evaluate
