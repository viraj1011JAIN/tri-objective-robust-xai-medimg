"""
HPO Trainer for TRADES Hyperparameter Optimization.

This module implements the training loop with Optuna integration,
handling trial suggestions, intermediate value reporting, and pruning.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.0
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import optuna
    from optuna.pruners import BasePruner, MedianPruner
    from optuna.samplers import BaseSampler, TPESampler
    from optuna.storages import RDBStorage
    from optuna.trial import FrozenTrial, Trial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any
    FrozenTrial = Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = Any
    DataLoader = Any

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .hpo_config import (
    HPOConfig,
    PrunerConfig,
    SamplerConfig,
    SearchSpace,
    SearchSpaceType,
)
from .hpo_objective import ObjectiveFunction, TrialMetrics, WeightedTriObjective

logger = logging.getLogger(__name__)


class HPOTrainer:
    """
    Base trainer for hyperparameter optimization with Optuna.

    Handles:
    - Trial hyperparameter suggestion
    - Training loop execution
    - Intermediate value reporting for pruning
    - Metric collection and logging
    """

    def __init__(
        self,
        config: HPOConfig,
        objective_fn: ObjectiveFunction,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize HPO trainer.

        Args:
            config: HPO configuration
            objective_fn: Objective function for optimization
            model_factory: Factory to create model from hyperparameters
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            device: Device to train on
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        self.config = config
        self.objective_fn = objective_fn
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Study management
        self.study: Optional[optuna.Study] = None
        self.current_trial: Optional[Trial] = None

        # Metric tracking
        self.trial_metrics_history: List[TrialMetrics] = []
        self.best_trial_metrics: Optional[TrialMetrics] = None

        logger.info(f"Initialized HPO trainer on device: {self.device}")

    def create_study(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """
        Create or load Optuna study.

        Args:
            study_name: Name for the study
            storage: Storage URL (e.g., sqlite:///hpo.db)
            load_if_exists: Load existing study if found

        Returns:
            Optuna study object
        """
        study_name = study_name or self.config.study_name

        # Create pruner
        pruner = self._create_pruner()

        # Create sampler
        sampler = self._create_sampler()

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.config.direction,
            load_if_exists=load_if_exists,
        )

        logger.info(
            f"Created study '{study_name}' with {sampler.__class__.__name__} "
            f"sampler and {pruner.__class__.__name__} pruner"
        )

        return self.study

    def _create_pruner(self) -> BasePruner:
        """Create Optuna pruner from configuration."""
        pruner_config = self.config.pruner_config

        if pruner_config.pruner_type.value == "median":
            return MedianPruner(
                n_startup_trials=pruner_config.n_startup_trials,
                n_warmup_steps=pruner_config.n_warmup_steps,
            )
        elif pruner_config.pruner_type.value == "percentile":
            from optuna.pruners import PercentilePruner

            return PercentilePruner(
                percentile=pruner_config.percentile or 25.0,
                n_startup_trials=pruner_config.n_startup_trials,
                n_warmup_steps=pruner_config.n_warmup_steps,
            )
        elif pruner_config.pruner_type.value == "hyperband":
            from optuna.pruners import HyperbandPruner

            return HyperbandPruner(
                min_resource=pruner_config.min_resource or 1,
                max_resource=pruner_config.max_resource or self.config.n_trials,
                reduction_factor=pruner_config.reduction_factor or 3,
            )
        else:
            from optuna.pruners import NopPruner

            return NopPruner()

    def _create_sampler(self) -> BaseSampler:
        """Create Optuna sampler from configuration."""
        sampler_config = self.config.sampler_config

        if sampler_config.sampler_type.value == "tpe":
            return TPESampler(
                n_startup_trials=sampler_config.n_startup_trials,
                seed=sampler_config.seed,
            )
        elif sampler_config.sampler_type.value == "cma_es":
            from optuna.samplers import CmaEsSampler

            return CmaEsSampler(
                n_startup_trials=sampler_config.n_startup_trials,
                seed=sampler_config.seed,
            )
        elif sampler_config.sampler_type.value == "random":
            from optuna.samplers import RandomSampler

            return RandomSampler(seed=sampler_config.seed)
        elif sampler_config.sampler_type.value == "nsgaii":
            from optuna.samplers import NSGAIISampler

            return NSGAIISampler(
                population_size=sampler_config.population_size or 50,
                seed=sampler_config.seed,
            )
        else:
            return TPESampler(seed=sampler_config.seed)

    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        hyperparams = {}

        for param_name, search_space in self.config.search_space.to_dict().items():
            space_type = search_space["type"]

            if space_type == "float":
                hyperparams[param_name] = trial.suggest_float(
                    param_name,
                    search_space["low"],
                    search_space["high"],
                    log=search_space.get("log", False),
                )
            elif space_type == "int":
                hyperparams[param_name] = trial.suggest_int(
                    param_name,
                    search_space["low"],
                    search_space["high"],
                    log=search_space.get("log", False),
                )
            elif space_type == "categorical":
                hyperparams[param_name] = trial.suggest_categorical(
                    param_name,
                    search_space["choices"],
                )

        return hyperparams

    def train_and_evaluate(self, trial: Trial) -> TrialMetrics:
        """
        Train model and evaluate metrics for trial.

        This is the main method that should be overridden by subclasses
        to implement specific training logic.

        Args:
            trial: Optuna trial object

        Returns:
            Trial metrics
        """
        raise NotImplementedError("Subclasses must implement train_and_evaluate")

    def run_study(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> optuna.Study:
        """
        Run HPO study.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            callbacks: Optional callbacks for study

        Returns:
            Completed study
        """
        if self.study is None:
            raise ValueError("Study not created. Call create_study() first.")

        n_trials = n_trials or self.config.n_trials

        logger.info(f"Starting HPO study with {n_trials} trials")

        # Create objective wrapper
        def objective(trial: Trial) -> float:
            self.current_trial = trial
            metrics = self.train_and_evaluate(trial)
            self.trial_metrics_history.append(metrics)

            # Update best metrics
            objective_value = self.objective_fn(metrics)
            if self.best_trial_metrics is None or objective_value > self.objective_fn(
                self.best_trial_metrics
            ):
                self.best_trial_metrics = metrics

            return objective_value

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True,
        )

        logger.info(
            f"Study completed. Best value: {self.study.best_value:.4f}, "
            f"Best params: {self.study.best_params}"
        )

        return self.study

    def report_intermediate_value(
        self,
        metrics: TrialMetrics,
        epoch: int,
    ) -> None:
        """
        Report intermediate value for pruning.

        Args:
            metrics: Current metrics
            epoch: Current epoch
        """
        if self.current_trial is None:
            return

        intermediate_value = self.objective_fn.get_intermediate_value(metrics, epoch)
        self.current_trial.report(intermediate_value, epoch)

        # Check if should prune
        if self.current_trial.should_prune():
            raise optuna.TrialPruned()

    def save_checkpoint(
        self,
        model: nn.Module,
        trial: Trial,
        metrics: TrialMetrics,
        checkpoint_dir: Path,
    ) -> Path:
        """
        Save model checkpoint for trial.

        Args:
            model: Model to save
            trial: Current trial
            metrics: Current metrics
            checkpoint_dir: Directory to save checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"trial_{trial.number}_checkpoint.pt"

        torch.save(
            {
                "trial_number": trial.number,
                "trial_params": trial.params,
                "model_state_dict": model.state_dict(),
                "metrics": metrics.to_dict(),
                "timestamp": time.time(),
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path


class TRADESHPOTrainer(HPOTrainer):
    """
    TRADES-specific HPO trainer.

    Implements TRADES adversarial training with hyperparameter
    optimization for β, ε, and learning rate.
    """

    def __init__(
        self,
        config: HPOConfig,
        objective_fn: ObjectiveFunction,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        attack_fn: Optional[Callable] = None,
        n_epochs: int = 10,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize TRADES HPO trainer.

        Args:
            config: HPO configuration
            objective_fn: Objective function
            model_factory: Factory to create model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            device: Device to train on
            attack_fn: Optional custom attack function
            n_epochs: Number of epochs per trial
            checkpoint_dir: Directory to save checkpoints
        """
        super().__init__(
            config,
            objective_fn,
            model_factory,
            train_loader,
            val_loader,
            test_loader,
            device,
        )

        self.attack_fn = attack_fn or self._default_pgd_attack
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints/hpo")

    def _default_pgd_attack(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        alpha: float = 2 / 255,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Default PGD attack for adversarial examples.

        Args:
            model: Model to attack
            x: Input batch
            y: Target labels
            epsilon: Perturbation budget
            alpha: Step size
            num_steps: Number of attack steps

        Returns:
            Adversarial examples
        """
        model.eval()
        x_adv = x.clone().detach()

        for _ in range(num_steps):
            x_adv.requires_grad = True

            with torch.enable_grad():
                logits = model(x_adv)
                loss = nn.CrossEntropyLoss()(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()

            # Project to epsilon ball
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv.detach()

    def trades_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        beta: float,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute TRADES loss.

        Args:
            model: Model
            x: Input batch
            y: Target labels
            beta: TRADES beta parameter
            epsilon: Perturbation budget

        Returns:
            Tuple of (total_loss, natural_loss, robust_loss)
        """
        model.eval()

        # Generate adversarial examples
        x_adv = self._default_pgd_attack(model, x, y, epsilon)

        model.train()

        # Natural loss
        logits_natural = model(x)
        natural_loss = nn.CrossEntropyLoss()(logits_natural, y)

        # Robust loss (KL divergence)
        logits_adv = model(x_adv)
        log_prob_natural = nn.functional.log_softmax(logits_natural, dim=1)
        prob_adv = nn.functional.softmax(logits_adv, dim=1)

        robust_loss = nn.functional.kl_div(
            log_prob_natural,
            prob_adv,
            reduction="batchmean",
        )

        total_loss = natural_loss + beta * robust_loss

        return total_loss, natural_loss, robust_loss

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        beta: float,
        epsilon: float,
    ) -> Tuple[float, float, float]:
        """
        Train one epoch with TRADES.

        Args:
            model: Model to train
            optimizer: Optimizer
            beta: TRADES beta parameter
            epsilon: Perturbation budget

        Returns:
            Tuple of (avg_loss, avg_natural_loss, avg_robust_loss)
        """
        model.train()

        total_loss = 0.0
        total_natural_loss = 0.0
        total_robust_loss = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()

            loss, natural_loss, robust_loss = self.trades_loss(
                model, x, y, beta, epsilon
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_natural_loss += natural_loss.item()
            total_robust_loss += robust_loss.item()
            n_batches += 1

        return (
            total_loss / n_batches,
            total_natural_loss / n_batches,
            total_robust_loss / n_batches,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        epsilon: float,
    ) -> Tuple[float, float]:
        """
        Evaluate model on clean and adversarial data.

        Args:
            model: Model to evaluate
            loader: Data loader
            epsilon: Perturbation budget for attacks

        Returns:
            Tuple of (clean_accuracy, robust_accuracy)
        """
        model.eval()

        clean_correct = 0
        robust_correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            # Clean accuracy
            logits_clean = model(x)
            pred_clean = logits_clean.argmax(dim=1)
            clean_correct += (pred_clean == y).sum().item()

            # Robust accuracy
            x_adv = self._default_pgd_attack(model, x, y, epsilon)
            logits_adv = model(x_adv)
            pred_adv = logits_adv.argmax(dim=1)
            robust_correct += (pred_adv == y).sum().item()

            total += y.size(0)

        clean_acc = clean_correct / total
        robust_acc = robust_correct / total

        return clean_acc, robust_acc

    def compute_cross_site_auroc(
        self,
        model: nn.Module,
    ) -> float:
        """
        Compute cross-site AUROC for generalization.

        This should be implemented based on your specific
        cross-site evaluation setup. Placeholder returns 0.8.

        Args:
            model: Model to evaluate

        Returns:
            Cross-site AUROC score
        """
        # TODO: Implement actual cross-site AUROC computation
        # This requires your specific cross-site test data
        logger.warning("Using placeholder AUROC value. Implement actual computation.")
        return 0.8

    def train_and_evaluate(self, trial: Trial) -> TrialMetrics:
        """
        Train and evaluate for one trial.

        Args:
            trial: Optuna trial object

        Returns:
            Trial metrics
        """
        # Suggest hyperparameters
        hyperparams = self.suggest_hyperparameters(trial)

        beta = hyperparams["beta"]
        epsilon = hyperparams["epsilon"]
        learning_rate = hyperparams["learning_rate"]

        logger.info(
            f"Trial {trial.number}: beta={beta:.2f}, "
            f"epsilon={epsilon:.4f}, lr={learning_rate:.5f}"
        )

        # Create model
        model = self.model_factory(hyperparams)
        model = model.to(self.device)

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(self.n_epochs):
            # Train epoch
            avg_loss, avg_natural_loss, avg_robust_loss = self.train_epoch(
                model, optimizer, beta, epsilon
            )

            # Evaluate on validation set
            clean_acc, robust_acc = self.evaluate(model, self.val_loader, epsilon)

            # Compute cross-site AUROC (placeholder)
            auroc = self.compute_cross_site_auroc(model)

            # Create metrics
            metrics = TrialMetrics(
                robust_accuracy=robust_acc,
                clean_accuracy=clean_acc,
                cross_site_auroc=auroc,
                loss=avg_loss,
                natural_loss=avg_natural_loss,
                robust_loss=avg_robust_loss,
                epoch=epoch,
            )

            # Report intermediate value for pruning
            try:
                self.report_intermediate_value(metrics, epoch)
            except optuna.TrialPruned:
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                raise

            logger.info(
                f"Trial {trial.number} Epoch {epoch}: "
                f"loss={avg_loss:.4f}, clean_acc={clean_acc:.4f}, "
                f"robust_acc={robust_acc:.4f}, auroc={auroc:.4f}"
            )

        # Final evaluation on test set if available
        if self.test_loader is not None:
            clean_acc, robust_acc = self.evaluate(model, self.test_loader, epsilon)
            auroc = self.compute_cross_site_auroc(model)

            metrics = TrialMetrics(
                robust_accuracy=robust_acc,
                clean_accuracy=clean_acc,
                cross_site_auroc=auroc,
                loss=avg_loss,
                natural_loss=avg_natural_loss,
                robust_loss=avg_robust_loss,
                epoch=self.n_epochs,
            )

        # Save checkpoint
        self.save_checkpoint(model, trial, metrics, self.checkpoint_dir)

        return metrics


def create_trainer_factory(
    hpo_trainer: TRADESHPOTrainer,
) -> Callable[[Trial], TRADESHPOTrainer]:
    """
    Create trainer factory for use with objective functions.

    Args:
        hpo_trainer: Base HPO trainer instance

    Returns:
        Factory function that returns trainer configured for trial
    """

    def factory(trial: Trial) -> TRADESHPOTrainer:
        """Factory to create trainer from trial."""
        hpo_trainer.current_trial = trial
        return hpo_trainer

    return factory
