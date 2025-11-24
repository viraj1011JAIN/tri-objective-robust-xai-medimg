"""
Utilities package for tri-objective robust XAI project.

This module exports commonly used utility functions across the codebase.
"""

# Reproducibility utilities
from src.utils.reproducibility import (
    ReproducibilityState,
    get_reproducibility_state,
    log_reproducibility_to_mlflow,
    make_torch_generator,
    quick_determinism_check,
    reproducibility_header,
    seed_worker,
    set_global_seed,
    summarise_reproducibility_state,
)

# Backwards compatibility aliases for tests
set_seed = set_global_seed
get_seed_worker = seed_worker
set_deterministic = set_global_seed  # Same function, different name

# Configuration utilities
from src.utils.config import (
    ExperimentConfig,
    get_config_hash,
    load_experiment_config,
    save_resolved_config,
)

# Backwards compatibility aliases for tests
load_config = load_experiment_config
merge_configs = load_experiment_config  # Same underlying mechanism
validate_config = load_experiment_config  # Validation happens in loading

# MLflow utilities
from src.utils.mlflow_utils import build_experiment_and_run_name, init_mlflow

# Backwards compatibility aliases for tests
setup_mlflow = init_mlflow
log_params = None  # Use mlflow.log_param directly
log_metrics = None  # Use mlflow.log_metric directly

__all__ = [
    # Reproducibility
    "ReproducibilityState",
    "get_reproducibility_state",
    "log_reproducibility_to_mlflow",
    "make_torch_generator",
    "quick_determinism_check",
    "reproducibility_header",
    "seed_worker",
    "set_global_seed",
    "summarise_reproducibility_state",
    # Aliases
    "set_seed",
    "get_seed_worker",
    "set_deterministic",
    # Config
    "ExperimentConfig",
    "get_config_hash",
    "load_experiment_config",
    "save_resolved_config",
    # Aliases
    "load_config",
    "merge_configs",
    "validate_config",
    # MLflow
    "build_experiment_and_run_name",
    "init_mlflow",
    # Aliases
    "setup_mlflow",
]
