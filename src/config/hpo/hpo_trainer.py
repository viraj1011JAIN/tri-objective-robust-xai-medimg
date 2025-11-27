"""
Main HPO Orchestration Engine for Hyperparameter Optimization.

This module provides the main orchestration engine for conducting hyperparameter
optimization experiments, including study management, trial execution, result
tracking, and visualization.

Author: Viraj Jain
Date: November 2025
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, Trial

from .hyperparameters import HyperparameterConfig
from .objectives import (
    DynamicWeightAdjuster,
    ObjectiveMetrics,
    ParetoFrontTracker,
    WeightedSumObjective,
)
from .pruners import HybridPruner, create_pruner
from .search_spaces import SearchSpaceFactory, suggest_full_config

logger = logging.getLogger(__name__)


class HPOTrainer:
    """
    Main hyperparameter optimization trainer.

    This class orchestrates the entire HPO process, including study creation,
    trial execution, result tracking, and visualization.
    """

    def __init__(
        self,
        study_name: str,
        storage: Optional[str] = None,
        save_dir: Path = Path("hpo_results"),
        objective_type: str = "weighted_sum",
        sampler_type: str = "tpe",
        pruner_type: str = "hybrid",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize HPO trainer.

        Args:
            study_name: Name of the study
            storage: Database URL for study storage (None for in-memory)
            save_dir: Directory to save results
            objective_type: Type of optimization objective
            sampler_type: Type of sampler to use
            pruner_type: Type of pruner to use
            n_trials: Number of trials to run
            timeout: Maximum time for optimization in seconds
            n_jobs: Number of parallel jobs
            device: Device for training (cuda/cpu)
        """
        self.study_name = study_name
        self.storage = storage
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.objective_type = objective_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.device = device

        # Create sampler
        self.sampler = self._create_sampler(sampler_type)

        # Create pruner
        self.pruner = create_pruner(pruner_type)

        # Initialize tracking
        self.pareto_tracker = ParetoFrontTracker()
        self.weight_adjuster = DynamicWeightAdjuster()
        self.trial_history: List[Dict[str, Any]] = []

        # Create study
        self.study = self._create_study()

        logger.info(f"Initialized HPO trainer for study: {study_name}")
        logger.info(f"Sampler: {sampler_type}, Pruner: {pruner_type}")
        logger.info(f"Device: {device}, Parallel jobs: {n_jobs}")

    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        """
        Create sampler based on type.

        Args:
            sampler_type: Type of sampler

        Returns:
            Sampler instance
        """
        if sampler_type == "tpe":
            return TPESampler(
                n_startup_trials=10, n_ei_candidates=24, multivariate=True, seed=42
            )
        elif sampler_type == "random":
            return RandomSampler(seed=42)
        elif sampler_type == "cmaes":
            return CmaEsSampler(seed=42)
        elif sampler_type == "grid":
            # Grid sampler requires search space
            return RandomSampler(seed=42)  # Fallback to random
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    def _create_study(self) -> Study:
        """
        Create or load Optuna study.

        Returns:
            Study instance
        """
        direction = StudyDirection.MAXIMIZE

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=direction,
            load_if_exists=True,
        )

        return study

    def optimize(
        self,
        train_fn: Callable[[HyperparameterConfig, Trial], ObjectiveMetrics],
        search_space_fn: Optional[Callable[[Trial], HyperparameterConfig]] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Study:
        """
        Run hyperparameter optimization.

        Args:
            train_fn: Function that trains model and returns metrics
            search_space_fn: Function that defines search space
            callbacks: List of callback functions

        Returns:
            Completed study
        """
        if search_space_fn is None:
            search_space_fn = SearchSpaceFactory.create_full_search_space()

        def objective(trial: Trial) -> float:
            """Objective function for Optuna."""
            try:
                # Suggest hyperparameters
                config = search_space_fn(trial)

                # Train and evaluate
                metrics = train_fn(config, trial)

                # Store metrics in trial
                self._store_trial_metrics(trial, metrics, config)

                # Update Pareto front
                self.pareto_tracker.update(metrics, config.to_dict())

                # Update dynamic weights if applicable
                if hasattr(self, "weight_adjuster"):
                    updated_weights = self.weight_adjuster.update_weights(metrics)
                    trial.set_user_attr("dynamic_weights", updated_weights)

                # Calculate objective value
                objective_value = self._calculate_objective(metrics)

                return objective_value

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                raise optuna.TrialPruned()

        # Run optimization
        logger.info(f"Starting optimization with {self.n_trials} trials")
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
            show_progress_bar=True,
        )

        # Save results
        self._save_results()

        logger.info("Optimization complete")
        self._log_best_trial()

        return self.study

    def _calculate_objective(self, metrics: ObjectiveMetrics) -> float:
        """
        Calculate objective value from metrics.

        Args:
            metrics: Objective metrics

        Returns:
            Objective value
        """
        if self.objective_type == "accuracy":
            return metrics.accuracy
        elif self.objective_type == "robustness":
            return metrics.robustness
        elif self.objective_type == "explainability":
            return metrics.explainability
        elif self.objective_type == "weighted_sum":
            objective = WeightedSumObjective()
            return objective.evaluate(metrics)
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

    def _store_trial_metrics(
        self, trial: Trial, metrics: ObjectiveMetrics, config: HyperparameterConfig
    ):
        """
        Store trial metrics and configuration.

        Args:
            trial: Optuna trial
            metrics: Objective metrics
            config: Hyperparameter configuration
        """
        # Store all metrics as user attributes
        for key, value in metrics.to_dict().items():
            trial.set_user_attr(key, value)

        # Store configuration
        trial.set_user_attr("config", config.to_dict())

        # Add to history
        self.trial_history.append(
            {
                "trial_number": trial.number,
                "metrics": metrics.to_dict(),
                "config": config.to_dict(),
                "datetime": datetime.now().isoformat(),
            }
        )

    def _save_results(self):
        """Save optimization results."""
        # Save study
        study_path = self.save_dir / f"{self.study_name}_study.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(self.study, f)
        logger.info(f"Saved study to {study_path}")

        # Save best trial
        best_trial_path = self.save_dir / f"{self.study_name}_best_trial.json"
        with open(best_trial_path, "w") as f:
            json.dump(self._get_best_trial_info(), f, indent=2)
        logger.info(f"Saved best trial to {best_trial_path}")

        # Save Pareto front
        pareto_path = self.save_dir / f"{self.study_name}_pareto_front.json"
        pareto_data = []
        for metrics, config in self.pareto_tracker.get_pareto_front():
            pareto_data.append({"metrics": metrics.to_dict(), "config": config})
        with open(pareto_path, "w") as f:
            json.dump(pareto_data, f, indent=2)
        logger.info(f"Saved Pareto front to {pareto_path}")

        # Save trial history
        history_path = self.save_dir / f"{self.study_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.trial_history, f, indent=2)
        logger.info(f"Saved trial history to {history_path}")

    def _get_best_trial_info(self) -> Dict[str, Any]:
        """Get information about best trial."""
        best_trial = self.study.best_trial

        return {
            "trial_number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
            "datetime_start": (
                best_trial.datetime_start.isoformat()
                if best_trial.datetime_start
                else None
            ),
            "datetime_complete": (
                best_trial.datetime_complete.isoformat()
                if best_trial.datetime_complete
                else None
            ),
            "duration": (
                best_trial.duration.total_seconds() if best_trial.duration else None
            ),
        }

    def _log_best_trial(self):
        """Log information about best trial."""
        best_trial = self.study.best_trial

        logger.info("=" * 80)
        logger.info("BEST TRIAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"Trial number: {best_trial.number}")
        logger.info(f"Objective value: {best_trial.value:.4f}")

        if "accuracy" in best_trial.user_attrs:
            logger.info(f"Accuracy: {best_trial.user_attrs['accuracy']:.4f}")
        if "robustness" in best_trial.user_attrs:
            logger.info(f"Robustness: {best_trial.user_attrs['robustness']:.4f}")
        if "explainability" in best_trial.user_attrs:
            logger.info(
                f"Explainability: {best_trial.user_attrs['explainability']:.4f}"
            )

        logger.info("\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)

    def get_best_config(self) -> HyperparameterConfig:
        """
        Get best hyperparameter configuration.

        Returns:
            Best configuration found
        """
        best_trial = self.study.best_trial
        config_dict = best_trial.user_attrs.get("config")

        if config_dict is None:
            raise ValueError("Best trial does not have stored configuration")

        return HyperparameterConfig.from_dict(config_dict)

    def get_pareto_front(self) -> List[Tuple[ObjectiveMetrics, HyperparameterConfig]]:
        """
        Get Pareto-optimal solutions.

        Returns:
            List of Pareto-optimal solutions
        """
        pareto_solutions = []
        for metrics, config_dict in self.pareto_tracker.get_pareto_front():
            config = HyperparameterConfig.from_dict(config_dict)
            pareto_solutions.append((metrics, config))

        return pareto_solutions

    def get_trial_dataframe(self):
        """
        Get trial results as pandas DataFrame.

        Returns:
            DataFrame with trial results
        """
        try:
            import pandas as pd

            return self.study.trials_dataframe()
        except ImportError:
            logger.warning("pandas not available, returning None")
            return None

    def plot_optimization_history(self, save_path: Optional[Path] = None):
        """
        Plot optimization history.

        Args:
            save_path: Path to save plot
        """
        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self.study)

            if save_path:
                fig.write_html(str(save_path))
                logger.info(f"Saved optimization history plot to {save_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("Visualization dependencies not available")

    def plot_param_importances(self, save_path: Optional[Path] = None):
        """
        Plot parameter importances.

        Args:
            save_path: Path to save plot
        """
        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self.study)

            if save_path:
                fig.write_html(str(save_path))
                logger.info(f"Saved parameter importances plot to {save_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("Visualization dependencies not available")

    def plot_pareto_front(self, save_path: Optional[Path] = None):
        """
        Plot Pareto front for multi-objective optimization.

        Args:
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            pareto_front = self.pareto_tracker.get_pareto_front()

            if not pareto_front:
                logger.warning("Pareto front is empty, cannot plot")
                return

            # Extract objectives
            accuracies = [m.accuracy for m, _ in pareto_front]
            robustnesses = [m.robustness for m, _ in pareto_front]
            explainabilities = [m.explainability for m, _ in pareto_front]

            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

            scatter = ax.scatter(
                accuracies,
                robustnesses,
                explainabilities,
                c=range(len(accuracies)),
                cmap="viridis",
                s=100,
                alpha=0.7,
            )

            ax.set_xlabel("Accuracy", fontsize=12)
            ax.set_ylabel("Robustness", fontsize=12)
            ax.set_zlabel("Explainability", fontsize=12)
            ax.set_title("Pareto Front - Tri-Objective Optimization", fontsize=14)

            plt.colorbar(scatter, label="Solution Index")

            if save_path:
                plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
                logger.info(f"Saved Pareto front plot to {save_path}")
            else:
                plt.show()

            plt.close()
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

    def export_best_config(self, output_path: Path):
        """
        Export best configuration to file.

        Args:
            output_path: Path to save configuration
        """
        best_config = self.get_best_config()

        if output_path.suffix == ".yaml":
            best_config.to_yaml(output_path)
        elif output_path.suffix == ".json":
            best_config.to_json(output_path)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Exported best configuration to {output_path}")

    def resume_optimization(
        self,
        train_fn: Callable[[HyperparameterConfig, Trial], ObjectiveMetrics],
        additional_trials: int,
        search_space_fn: Optional[Callable[[Trial], HyperparameterConfig]] = None,
    ) -> Study:
        """
        Resume optimization from previous study.

        Args:
            train_fn: Training function
            additional_trials: Number of additional trials to run
            search_space_fn: Search space function

        Returns:
            Updated study
        """
        logger.info(f"Resuming optimization with {additional_trials} additional trials")
        original_n_trials = self.n_trials
        self.n_trials = additional_trials

        result = self.optimize(train_fn, search_space_fn)

        self.n_trials = original_n_trials + additional_trials
        return result


class HPOManager:
    """
    Manager for multiple HPO studies.

    This class manages multiple HPO studies for different objectives or configurations.
    """

    def __init__(self, base_dir: Path = Path("hpo_experiments")):
        """
        Initialize HPO manager.

        Args:
            base_dir: Base directory for all experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.trainers: Dict[str, HPOTrainer] = {}

        logger.info(f"Initialized HPO manager at {base_dir}")

    def create_trainer(self, study_name: str, **kwargs) -> HPOTrainer:
        """
        Create new HPO trainer.

        Args:
            study_name: Name of the study
            **kwargs: Additional arguments for HPOTrainer

        Returns:
            HPOTrainer instance
        """
        save_dir = self.base_dir / study_name
        trainer = HPOTrainer(study_name=study_name, save_dir=save_dir, **kwargs)
        self.trainers[study_name] = trainer
        return trainer

    def get_trainer(self, study_name: str) -> Optional[HPOTrainer]:
        """Get existing trainer."""
        return self.trainers.get(study_name)

    def compare_studies(
        self, study_names: List[str], metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Compare results across multiple studies.

        Args:
            study_names: Names of studies to compare
            metric: Metric to compare

        Returns:
            Comparison results
        """
        results = {}

        for name in study_names:
            trainer = self.trainers.get(name)
            if trainer is None:
                logger.warning(f"Study {name} not found")
                continue

            best_trial = trainer.study.best_trial
            results[name] = {
                "best_value": best_trial.value,
                "metric_value": best_trial.user_attrs.get(metric, None),
                "n_trials": len(trainer.study.trials),
                "best_trial_number": best_trial.number,
            }

        return results

    def export_comparison(self, study_names: List[str], output_path: Path):
        """
        Export study comparison to file.

        Args:
            study_names: Names of studies to compare
            output_path: Path to save comparison
        """
        comparison = self.compare_studies(study_names)

        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Exported comparison to {output_path}")


def create_hpo_trainer_from_config(
    config_path: Path, study_name: Optional[str] = None
) -> HPOTrainer:
    """
    Create HPO trainer from configuration file.

    Args:
        config_path: Path to configuration file
        study_name: Optional study name override

    Returns:
        HPOTrainer instance
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if study_name is None:
        study_name = config.get("study_name", "hpo_study")

    return HPOTrainer(
        study_name=study_name,
        storage=config.get("storage"),
        save_dir=Path(config.get("save_dir", "hpo_results")),
        objective_type=config.get("objective_type", "weighted_sum"),
        sampler_type=config.get("sampler_type", "tpe"),
        pruner_type=config.get("pruner_type", "hybrid"),
        n_trials=config.get("n_trials", 100),
        timeout=config.get("timeout"),
        n_jobs=config.get("n_jobs", 1),
    )
