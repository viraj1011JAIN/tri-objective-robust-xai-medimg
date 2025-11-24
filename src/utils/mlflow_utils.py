"""MLflow utilities for consistent experiment tracking and run management.

This module provides standardized MLflow configuration for the tri-objective
robust XAI project, ensuring reproducible experiment tracking across all
training pipelines (CIFAR-10 debug, NIH CXR, ISIC, etc.).

Key features:
- Local file-based tracking with configurable URI
- Consistent experiment and run naming conventions
- Automatic experiment creation
- Type hints and docstrings for maintainability

Typical usage:
    >>> from src.utils.mlflow_utils import init_mlflow, build_experiment_and_run_name
    >>>
    >>> exp_name, run_name = build_experiment_and_run_name(
    ...     dataset="NIH-CXR",
    ...     model="resnet50",
    ...     objective="tri-objective",
    ...     extra_tag="pgd-eps-0.03"
    ... )
    >>>
    >>> run = init_mlflow(exp_name, run_name)
    >>> mlflow.log_param("learning_rate", 0.001)
    >>> mlflow.log_metric("accuracy", 0.95)
    >>> mlflow.end_run()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow


def init_mlflow(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> mlflow.ActiveRun:
    """Initialize MLflow for this project with a consistent configuration.

    Sets up MLflow tracking with a local file-based backend by default,
    creates the experiment if it doesn't exist, and starts a new run.

    Args:
        experiment_name: Name of the MLflow experiment. Should follow the
            convention <dataset>__<objective>, e.g., "NIH-CXR__tri-objective".
        run_name: Optional name for this specific run. Should follow the
            convention <model>[__<extra_tag>], e.g., "resnet50__pgd-eps-0.03".
            If None, MLflow generates a random name.
        tracking_uri: MLflow tracking URI. If None, defaults to a local
            file-based store at ./mlruns (resolved to absolute path).

    Returns:
        An active MLflow run object. The caller is responsible for ending
        the run with mlflow.end_run() or using it as a context manager.

    Example:
        >>> run = init_mlflow("CIFAR10-debug__baseline", "SimpleDebugNet__seed-42")
        >>> mlflow.log_param("epochs", 10)
        >>> mlflow.log_metric("loss", 0.5)
        >>> mlflow.end_run()

    Notes:
        - The default tracking directory (./mlruns) is created automatically
          by MLflow if it doesn't exist.
        - Experiments are created on first use; subsequent calls reuse the
          existing experiment.
        - On Windows, the path is converted to POSIX format for MLflow
          compatibility.
    """
    if tracking_uri is None:
        tracking_dir = Path("mlruns").resolve()
        tracking_uri = f"file:{tracking_dir.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=run_name)
    return run


def build_experiment_and_run_name(
    dataset: str,
    model: str,
    objective: str,
    extra_tag: Optional[str] = None,
) -> tuple[str, str]:
    """Build standardized experiment and run names following project conventions.

    This function enforces a consistent naming scheme across all experiments:
    - Experiment name: <dataset>__<objective>
    - Run name: <model>[__<extra_tag>]

    The naming convention enables:
    - Easy filtering and grouping in the MLflow UI
    - Systematic comparison of objectives across datasets
    - Clear identification of model variants and hyperparameters

    Args:
        dataset: Dataset identifier, e.g., "NIH-CXR", "ISIC-2020", "CIFAR10-debug".
        model: Model architecture name, e.g., "resnet50", "efficientnet-b0",
            "SimpleDebugNet".
        objective: Training objective, e.g., "baseline", "adversarial",
            "tri-objective".
        extra_tag: Optional additional tag for run identification, e.g.,
            "seed-42", "pgd-eps-0.03", "lr-1e-4". Used to distinguish runs
            with different hyperparameters or random seeds.

    Returns:
        A tuple of (experiment_name, run_name) strings.

    Examples:
        >>> build_experiment_and_run_name("NIH-CXR", "resnet50", "tri-objective")
        ('NIH-CXR__tri-objective', 'resnet50')

        >>> build_experiment_and_run_name(
        ...     "ISIC-2020", "efficientnet-b0", "adversarial", "pgd-eps-0.03"
        ... )
        ('ISIC-2020__adversarial', 'efficientnet-b0__pgd-eps-0.03')

        >>> build_experiment_and_run_name(
        ...     "CIFAR10-debug", "SimpleDebugNet", "baseline", "seed-42"
        ... )
        ('CIFAR10-debug__baseline', 'SimpleDebugNet__seed-42')

    Notes:
        - Use double underscores (__) as separators to avoid conflicts with
          single underscores in dataset/model names.
        - Keep names concise but descriptive for clarity in the MLflow UI.
        - For ablation studies, use extra_tag to denote the variant, e.g.,
          "no-xai-loss", "lambda-rob-0.5".
    """
    experiment_name = f"{dataset}__{objective}"
    run_name_parts = [model]
    if extra_tag:
        run_name_parts.append(extra_tag)
    run_name = "__".join(run_name_parts)
    return experiment_name, run_name


# Backwards compatibility aliases for tests
setup_mlflow = init_mlflow


__all__ = [
    "init_mlflow",
    "build_experiment_and_run_name",
    # Aliases
    "setup_mlflow",
]
