"""
Custom Pruning Strategies for Optuna Hyperparameter Optimization.

This module implements sophisticated pruning strategies to efficiently terminate
unpromising trials during hyperparameter optimization, including performance-based,
resource-aware, and multi-objective pruning.

Author: Viraj Jain
Date: November 2025
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState


class PerformanceBasedPruner(BasePruner):
    """
    Performance-based pruner that considers multiple metrics.

    This pruner terminates trials that show poor performance across multiple
    objectives early in training.
    """

    def __init__(
        self,
        min_steps: int = 10,
        warmup_steps: int = 5,
        patience: int = 3,
        performance_threshold: float = 0.5,
        check_interval: int = 1,
    ):
        """
        Initialize performance-based pruner.

        Args:
            min_steps: Minimum steps before pruning can occur
            warmup_steps: Steps to wait before starting performance checks
            patience: Number of steps with poor performance before pruning
            performance_threshold: Minimum relative performance to continue
            check_interval: Interval for checking pruning conditions
        """
        self.min_steps = min_steps
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.performance_threshold = performance_threshold
        self.check_interval = check_interval
        self._trial_records: Dict[int, List[float]] = {}

    def prune(self, study: "optuna.Study", trial: FrozenTrial) -> bool:
        """
        Determine whether to prune the trial.

        Args:
            study: Optuna study object
            trial: Trial to potentially prune

        Returns:
            True if trial should be pruned
        """
        step = trial.last_step

        if step is None:
            return False

        # Don't prune during warmup
        if step < self.warmup_steps:
            return False

        # Don't prune before minimum steps
        if step < self.min_steps:
            return False

        # Check interval
        if step % self.check_interval != 0:
            return False

        current_value = trial.intermediate_values[step]

        # Get completed trials
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if len(completed_trials) == 0:
            return False

        # Calculate performance relative to best completed trials
        best_values = []
        for completed_trial in completed_trials:
            if step in completed_trial.intermediate_values:
                best_values.append(completed_trial.intermediate_values[step])

        if not best_values:
            return False

        # Determine if performance is acceptable
        if study.direction == StudyDirection.MAXIMIZE:
            best_at_step = max(best_values)
            relative_performance = (
                current_value / best_at_step if best_at_step > 0 else 0
            )
        else:
            best_at_step = min(best_values)
            relative_performance = (
                best_at_step / current_value if current_value > 0 else 0
            )

        # Track poor performance
        trial_id = trial.number
        if trial_id not in self._trial_records:
            self._trial_records[trial_id] = []

        if relative_performance < self.performance_threshold:
            self._trial_records[trial_id].append(relative_performance)
        else:
            self._trial_records[trial_id] = []  # Reset on good performance

        # Prune if consistently poor
        return len(self._trial_records[trial_id]) >= self.patience


class ResourceAwarePruner(BasePruner):
    """
    Resource-aware pruner that considers computational cost.

    This pruner balances performance improvement with resource usage,
    terminating trials that consume too many resources for minimal gain.
    """

    def __init__(
        self,
        time_budget: float = 3600.0,  # 1 hour
        min_improvement_rate: float = 0.01,
        check_interval: int = 5,
        min_steps: int = 10,
    ):
        """
        Initialize resource-aware pruner.

        Args:
            time_budget: Total time budget in seconds
            min_improvement_rate: Minimum improvement rate to continue
            check_interval: Interval for checking pruning conditions
            min_steps: Minimum steps before pruning
        """
        self.time_budget = time_budget
        self.min_improvement_rate = min_improvement_rate
        self.check_interval = check_interval
        self.min_steps = min_steps
        self._study_start_time: Optional[float] = None
        self._trial_improvement_rates: Dict[int, List[float]] = {}

    def prune(self, study: "optuna.Study", trial: FrozenTrial) -> bool:
        """
        Determine whether to prune based on resource usage.

        Args:
            study: Optuna study object
            trial: Trial to potentially prune

        Returns:
            True if trial should be pruned
        """
        step = trial.last_step

        if step is None or step < self.min_steps:
            return False

        if step % self.check_interval != 0:
            return False

        # Check time budget
        if trial.datetime_start is not None:
            import datetime

            current_time = datetime.datetime.now()
            elapsed_time = (current_time - trial.datetime_start).total_seconds()

            if elapsed_time > self.time_budget:
                return True

        # Check improvement rate
        if len(trial.intermediate_values) < 2:
            return False

        steps = sorted(trial.intermediate_values.keys())
        values = [trial.intermediate_values[s] for s in steps]

        # Calculate improvement rate
        if len(values) >= 2:
            recent_improvement = abs(values[-1] - values[-2]) / (abs(values[-2]) + 1e-8)

            trial_id = trial.number
            if trial_id not in self._trial_improvement_rates:
                self._trial_improvement_rates[trial_id] = []

            self._trial_improvement_rates[trial_id].append(recent_improvement)

            # Prune if improvement rate is consistently low
            if len(self._trial_improvement_rates[trial_id]) >= 3:
                avg_improvement = np.mean(self._trial_improvement_rates[trial_id][-3:])
                return avg_improvement < self.min_improvement_rate

        return False


class MultiObjectivePruner(BasePruner):
    """
    Multi-objective pruner for tri-objective optimization.

    This pruner considers all three objectives (accuracy, robustness, explainability)
    when deciding whether to terminate a trial.
    """

    def __init__(
        self,
        min_steps: int = 15,
        patience: int = 5,
        threshold_accuracy: float = 0.4,
        threshold_robustness: float = 0.3,
        threshold_explainability: float = 0.3,
        require_all: bool = False,
    ):
        """
        Initialize multi-objective pruner.

        Args:
            min_steps: Minimum steps before pruning
            patience: Steps with poor performance before pruning
            threshold_accuracy: Minimum accuracy threshold
            threshold_robustness: Minimum robustness threshold
            threshold_explainability: Minimum explainability threshold
            require_all: If True, all objectives must be below threshold to prune
        """
        self.min_steps = min_steps
        self.patience = patience
        self.threshold_accuracy = threshold_accuracy
        self.threshold_robustness = threshold_robustness
        self.threshold_explainability = threshold_explainability
        self.require_all = require_all
        self._poor_performance_counts: Dict[int, int] = {}

    def prune(self, study: "optuna.Study", trial: FrozenTrial) -> bool:
        """
        Determine whether to prune based on multi-objective performance.

        Args:
            study: Optuna study object
            trial: Trial to potentially prune

        Returns:
            True if trial should be pruned
        """
        step = trial.last_step

        if step is None or step < self.min_steps:
            return False

        # Get objective values from user attributes
        accuracy = trial.user_attrs.get(f"accuracy_step_{step}")
        robustness = trial.user_attrs.get(f"robustness_step_{step}")
        explainability = trial.user_attrs.get(f"explainability_step_{step}")

        if accuracy is None or robustness is None or explainability is None:
            return False

        # Check thresholds
        below_accuracy = accuracy < self.threshold_accuracy
        below_robustness = robustness < self.threshold_robustness
        below_explainability = explainability < self.threshold_explainability

        # Determine if performance is poor
        if self.require_all:
            is_poor = below_accuracy and below_robustness and below_explainability
        else:
            is_poor = below_accuracy or below_robustness or below_explainability

        # Track poor performance
        trial_id = trial.number
        if trial_id not in self._poor_performance_counts:
            self._poor_performance_counts[trial_id] = 0

        if is_poor:
            self._poor_performance_counts[trial_id] += 1
        else:
            self._poor_performance_counts[trial_id] = 0

        return self._poor_performance_counts[trial_id] >= self.patience


class AdaptivePruner(BasePruner):
    """
    Adaptive pruner that adjusts thresholds based on study progress.

    This pruner becomes more aggressive as the study progresses and more
    information becomes available.
    """

    def __init__(
        self,
        initial_threshold: float = 0.3,
        final_threshold: float = 0.7,
        n_trials_to_adapt: int = 50,
        min_steps: int = 10,
        check_interval: int = 2,
    ):
        """
        Initialize adaptive pruner.

        Args:
            initial_threshold: Starting performance threshold
            final_threshold: Final performance threshold
            n_trials_to_adapt: Number of trials over which to adapt
            min_steps: Minimum steps before pruning
            check_interval: Interval for checking pruning conditions
        """
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.n_trials_to_adapt = n_trials_to_adapt
        self.min_steps = min_steps
        self.check_interval = check_interval

    def prune(self, study: "optuna.Study", trial: FrozenTrial) -> bool:
        """
        Determine whether to prune with adaptive threshold.

        Args:
            study: Optuna study object
            trial: Trial to potentially prune

        Returns:
            True if trial should be pruned
        """
        step = trial.last_step

        if step is None or step < self.min_steps:
            return False

        if step % self.check_interval != 0:
            return False

        # Calculate current threshold based on study progress
        n_trials = len(study.trials)
        progress = min(n_trials / self.n_trials_to_adapt, 1.0)
        current_threshold = (
            self.initial_threshold
            + (self.final_threshold - self.initial_threshold) * progress
        )

        # Get current value
        if step not in trial.intermediate_values:
            return False

        current_value = trial.intermediate_values[step]

        # Get reference values from completed trials
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if len(completed_trials) < 3:
            return False

        # Calculate quantile of completed trials at this step
        values_at_step = []
        for t in completed_trials:
            if step in t.intermediate_values:
                values_at_step.append(t.intermediate_values[step])

        if not values_at_step:
            return False

        if study.direction == StudyDirection.MAXIMIZE:
            threshold_value = np.quantile(values_at_step, 1.0 - current_threshold)
            return current_value < threshold_value
        else:
            threshold_value = np.quantile(values_at_step, current_threshold)
            return current_value > threshold_value


class HybridPruner(BasePruner):
    """
    Hybrid pruner combining multiple pruning strategies.

    This pruner uses a combination of performance-based, resource-aware,
    and adaptive pruning strategies.
    """

    def __init__(
        self,
        performance_pruner: Optional[PerformanceBasedPruner] = None,
        resource_pruner: Optional[ResourceAwarePruner] = None,
        adaptive_pruner: Optional[AdaptivePruner] = None,
        require_all: bool = False,
    ):
        """
        Initialize hybrid pruner.

        Args:
            performance_pruner: Performance-based pruner
            resource_pruner: Resource-aware pruner
            adaptive_pruner: Adaptive pruner
            require_all: If True, all pruners must agree to prune
        """
        self.performance_pruner = performance_pruner or PerformanceBasedPruner()
        self.resource_pruner = resource_pruner or ResourceAwarePruner()
        self.adaptive_pruner = adaptive_pruner or AdaptivePruner()
        self.require_all = require_all

    def prune(self, study: "optuna.Study", trial: FrozenTrial) -> bool:
        """
        Determine whether to prune using hybrid strategy.

        Args:
            study: Optuna study object
            trial: Trial to potentially prune

        Returns:
            True if trial should be pruned
        """
        performance_prune = self.performance_pruner.prune(study, trial)
        resource_prune = self.resource_pruner.prune(study, trial)
        adaptive_prune = self.adaptive_pruner.prune(study, trial)

        if self.require_all:
            return performance_prune and resource_prune and adaptive_prune
        else:
            return performance_prune or resource_prune or adaptive_prune


def create_pruner(pruner_type: str, **kwargs) -> BasePruner:
    """
    Factory function to create pruners.

    Args:
        pruner_type: Type of pruner to create
        **kwargs: Additional arguments for pruner

    Returns:
        Pruner instance
    """
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(**kwargs)
    elif pruner_type == "percentile":
        return optuna.pruners.PercentilePruner(**kwargs)
    elif pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner(**kwargs)
    elif pruner_type == "threshold":
        return optuna.pruners.ThresholdPruner(**kwargs)
    elif pruner_type == "performance":
        return PerformanceBasedPruner(**kwargs)
    elif pruner_type == "resource_aware":
        return ResourceAwarePruner(**kwargs)
    elif pruner_type == "multi_objective":
        return MultiObjectivePruner(**kwargs)
    elif pruner_type == "adaptive":
        return AdaptivePruner(**kwargs)
    elif pruner_type == "hybrid":
        return HybridPruner(**kwargs)
    elif pruner_type == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")


def get_default_pruner_for_objective(objective_type: str) -> BasePruner:
    """
    Get default pruner for a given objective type.

    Args:
        objective_type: Type of optimization objective

    Returns:
        Appropriate pruner for the objective
    """
    if objective_type == "accuracy":
        return PerformanceBasedPruner(
            min_steps=10, patience=5, performance_threshold=0.6
        )
    elif objective_type == "robustness":
        return ResourceAwarePruner(
            time_budget=7200.0, min_improvement_rate=0.005  # 2 hours for robustness
        )
    elif objective_type == "explainability":
        return AdaptivePruner(initial_threshold=0.4, final_threshold=0.7)
    elif objective_type in ["weighted_sum", "tri_objective"]:
        return MultiObjectivePruner(min_steps=15, patience=5, require_all=False)
    else:
        return HybridPruner()
