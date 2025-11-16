"""
Configuration management utilities.

This module handles:
- Loading multiple YAML config files (base, dataset, model, experiment).
- Deep-merging them in a predictable order.
- Expanding environment variables in YAML values (e.g. "${DATA_ROOT}/nih_cxr").
- Validating the final configuration against a schema (Pydantic models).
- Computing a stable hash of the resolved configuration for provenance.
- Optional saving of the resolved config for experiment reproducibility.

Typical usage in a training script:

    from src.utils.config import (
        load_experiment_config,
        save_resolved_config,
        get_config_hash,
    )
    from src.utils.reproducibility import set_global_seed

    cfg = load_experiment_config(
        "configs/base.yaml",
        "configs/datasets/nih_cxr_debug.yaml",
        "configs/models/resnet50.yaml",
        "configs/experiments/nih_triobj_debug.yaml",
    )

    # Access fields:
    cfg.training.max_epochs
    cfg.dataset.batch_size
    cfg.model.name
    cfg.reproducibility.seed

    # Optional: save fully-resolved config & hash for provenance
    save_resolved_config(cfg, "results/configs/last_run.yaml")
    print("Config hash:", get_config_hash(cfg))
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------------------
# Pydantic models defining the config schema
# ---------------------------------------------------------------------------


class ReproducibilityConfig(BaseModel):
    """
    Configuration of all reproducibility-related knobs.

    This is intentionally slightly more detailed than simply a "seed", so that
    you can explicitly state in the config whether you want deterministic
    algorithms, cuDNN benchmarking, TF32, etc.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(42, ge=0)
    deterministic_cudnn: bool = Field(
        True,
        description="If True, sets torch.backends.cudnn.deterministic = True.",
    )
    benchmark_cudnn: bool = Field(
        False,
        description="If True, enables cuDNN benchmarking (better perf, less reproducible).",
    )
    enable_tf32: bool = Field(
        False,
        description="If True, allows TF32 matrix-multiply on Ampere+ GPUs.",
    )
    use_deterministic_algorithms: bool = Field(
        False,
        description="If True, calls torch.use_deterministic_algorithms(True) where available.",
    )
    dataloader_seed_offset: int = Field(
        0,
        ge=0,
        description="Optional offset added when seeding DataLoader workers/generator.",
    )


class DatasetConfig(BaseModel):
    """
    Configuration for dataset loading and DataLoader behaviour.

    extra='allow' permits you to add dataset-specific fields in YAML
    without breaking the schema, e.g. img_size, augmentation flags, etc.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Short dataset name, for logging/MLflow tags.")
    root: str = Field(..., description="Root folder for dataset files.")
    batch_size: int = Field(64, gt=0)
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = Field(True)
    train_subset: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional train subset size for debug runs.",
    )
    test_subset: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional test subset size for debug runs.",
    )


class ModelConfig(BaseModel):
    """
    Configuration for the model architecture and checkpoints.

    extra='allow' so you can attach arbitrary model hyperparameters in YAML
    without changing the schema (e.g. depth, width, dropout, etc.).
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Model architecture name, e.g. ResNet50.")
    num_classes: int = Field(..., gt=0)
    pretrained: bool = Field(False)
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional path to start from a pretrained checkpoint.",
    )


class TrainingConfig(BaseModel):
    """
    Configuration for the training loop and optimiser.

    extra='allow' allows you to extend this later with scheduler, warmup,
    tri-objective weights, etc., without having to adjust the schema.
    """

    model_config = ConfigDict(extra="allow")

    max_epochs: int = Field(..., gt=0)
    device: str = Field(
        "cuda",
        pattern=r"^(cpu|cuda(:[0-9]+)?)$",
        description="Device string, usually 'cuda', 'cuda:0' or 'cpu'.",
    )
    eval_every_n_epochs: int = Field(1, gt=0)
    log_every_n_steps: int = Field(50, gt=0)
    gradient_clip_val: float = Field(0.0, ge=0.0)
    learning_rate: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)


class ExperimentMeta(BaseModel):
    """
    Human-visible metadata for the experiment.

    This is what you would typically surface in MLflow / W&B / logs.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Human readable experiment name.")
    description: Optional[str] = Field(
        default=None,
        description="Longer free-text description (optional).",
    )
    project_name: str = Field(
        "tri-objective-robust-xai-medimg",
        description="Logical project grouping for MLflow or wandb.",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Freeform metadata tags, e.g. dataset=..., kind=baseline.",
    )


class ExperimentConfig(BaseModel):
    """
    Final validated configuration object.

    Top-level fields mirror the YAML structure:
    - experiment: ExperimentMeta
    - dataset: DatasetConfig
    - model: ModelConfig
    - training: TrainingConfig
    - reproducibility: ReproducibilityConfig

    Additional field:
    - yaml_stack: ordered list of YAML paths used to construct this config.
      This is filled in by load_experiment_config and is extremely helpful for
      provenance (e.g. logging to MLflow).
    """

    model_config = ConfigDict(extra="ignore")

    experiment: ExperimentMeta
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    reproducibility: ReproducibilityConfig

    yaml_stack: List[str] = Field(
        default_factory=list,
        description="YAML files (in order) used to build this configuration.",
    )


# ---------------------------------------------------------------------------
# YAML loading, env expansion and deep merging
# ---------------------------------------------------------------------------


def _load_yaml_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a single YAML file into a dictionary.

    Raises:
        FileNotFoundError if the file does not exist.
        ValueError if the top-level YAML is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping/object.")
    return data


def _deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge 'new' into 'base', returning a new dictionary.

    Values in 'new' take precedence over 'base'. Nested dicts are merged
    recursively, non-dict values overwrite directly.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables and user (~) in strings.
    Normalizes all paths to use forward slashes for cross-platform consistency.

    This lets you write things like:

        dataset:
          root: "${DATA_ROOT}/nih_cxr"

    and have it resolved at load time.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_expand_env_vars(v) for v in obj)
    if isinstance(obj, str):
        expanded = os.path.expandvars(obj)
        expanded = os.path.expanduser(expanded)
        expanded = str(Path(expanded).as_posix())
        return expanded
    return obj


def _normalize_paths_in_obj(obj: Any) -> Any:
    """
    Recursively normalize all path-like strings to use forward slashes.
    This ensures deterministic config hashing across platforms.

    Args:
        obj: Object to normalize (dict, list, tuple, or str)

    Returns:
        Object with normalized paths
    """
    if isinstance(obj, dict):
        return {k: _normalize_paths_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_paths_in_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_paths_in_obj(v) for v in obj)
    if isinstance(obj, str):
        if "/" in obj or "\\" in obj:
            return str(Path(obj).as_posix())
        return obj
    return obj


def _flatten_for_hash(
    obj: Any,
    prefix: str = "",
    out: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Flatten a nested dict-like object into a dotted-key dictionary of strings.

    Used for building a stable hash of the configuration.
    Excludes yaml_stack field to ensure hash stability across different file paths.
    """
    if out is None:
        out = {}

    if isinstance(obj, BaseModel):
        obj = obj.model_dump()

    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            if k == "yaml_stack":
                continue
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_for_hash(obj[k], new_prefix, out)
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]"
            _flatten_for_hash(v, new_prefix, out)
    else:
        normalized_val = _normalize_paths_in_obj(obj)
        out[prefix] = repr(normalized_val)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_experiment_config(*paths: str | Path) -> ExperimentConfig:
    """
    Load, merge and validate an experiment configuration from multiple YAML files.

    The order of paths matters: later files override earlier ones. A common pattern is:

        cfg = load_experiment_config(
            "configs/base.yaml",
            "configs/datasets/nih_cxr_debug.yaml",
            "configs/models/resnet50.yaml",
            "configs/experiments/nih_triobj_debug.yaml",
        )
    """
    merged: Dict[str, Any] = {}
    resolved_paths: List[Path] = [Path(p) for p in paths]

    for p in resolved_paths:
        chunk = _load_yaml_file(p)
        merged = _deep_merge(merged, chunk)

    merged = _expand_env_vars(merged)

    try:
        cfg = ExperimentConfig.model_validate(merged)
    except ValidationError as e:
        message_lines = ["Invalid experiment configuration:"]
        for err in e.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            message_lines.append(f"  - {loc}: {err.get('msg')}")
        message = "\n".join(message_lines)
        raise ValueError(message) from e

    cfg.yaml_stack = [str(p) for p in resolved_paths]
    return cfg


def save_resolved_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """
    Save the fully resolved configuration to a single YAML file.

    This is useful for logging the exact config used in a given run for
    full reproducibility.
    """
    path = Path(path)
    payload = {
        "experiment": cfg.experiment.model_dump(),
        "dataset": cfg.dataset.model_dump(),
        "model": cfg.model.model_dump(),
        "training": cfg.training.model_dump(),
        "reproducibility": cfg.reproducibility.model_dump(),
        "yaml_stack": cfg.yaml_stack,
        "config_hash": get_config_hash(cfg),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def get_config_hash(cfg: ExperimentConfig, algo: str = "sha256") -> str:
    """
    Compute a stable hash of the resolved configuration.

    Extremely useful for:
    - tagging MLflow runs with a config_hash
    - checking whether two runs used *exactly* the same configuration
    - reporting in the dissertation that each experimental condition
      is uniquely identified by a hash over all hyperparameters.
    """
    hasher = hashlib.new(algo)
    flat = _flatten_for_hash(cfg)
    for key in sorted(flat.keys()):
        line = f"{key}={flat[key]}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()
