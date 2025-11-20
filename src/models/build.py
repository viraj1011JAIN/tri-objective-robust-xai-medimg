"""
Model factory for instantiating architectures from configuration.

This module provides a centralized factory for building models in the
Tri-Objective Robust XAI pipeline. It handles:

- Architecture selection (ResNet, EfficientNet, ViT, etc.)
- Configuration validation and default values
- Model registry integration
- Logging and metadata tracking

The factory pattern enables:
    - Easy model switching via config files
    - Consistent initialization across experiments
    - Clean separation between model definition and instantiation

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

import torch.nn as nn

from .base_model import BaseModel
from .efficientnet import EfficientNetB0Classifier
from .resnet import ResNet50Classifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Model registry: maps architecture names to model classes
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "resnet50": ResNet50Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,  # Add when implemented
    # "vit_b16": ViTB16Classifier,  # Add when implemented
}

# Default configurations for each architecture
DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "pretrained": True,
        "in_channels": 3,
        "dropout": 0.0,
        "global_pool": "avg",
    },
    "efficientnet_b0": {
        "pretrained": True,
        "in_channels": 3,
        "dropout": 0.2,
    },
    "vit_b16": {
        "pretrained": True,
        "in_channels": 3,
        "dropout": 0.1,
        "patch_size": 16,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_architecture(
    architecture: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Resolve the effective architecture key from either ``architecture``
    or ``name`` (treated as aliases).

    Parameters
    ----------
    architecture:
        Canonical architecture name (e.g. "resnet50").
    name:
        Optional alias (e.g. from config model.name).

    Returns
    -------
    str
        Lowercased architecture key.

    Raises
    ------
    ValueError
        If neither ``architecture`` nor ``name`` is provided.
    """
    if architecture is None and name is None:
        raise ValueError("Either 'architecture' or 'name' must be provided.")

    arch = architecture if architecture is not None else name
    if arch is None:
        raise ValueError("Architecture name cannot be None")
    return arch.lower().strip()


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_model(
    architecture: Optional[str] = None,
    num_classes: int = 1,
    config: Optional[Dict[str, Any]] = None,
    *,
    name: Optional[str] = None,
    pretrained: Optional[bool] = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Build a model from architecture/name and configuration.

    This is the main entry point for model instantiation in the pipeline.
    It handles validation, default values, and logging.

    Parameters
    ----------
    architecture:
        Architecture name. Must be in MODEL_REGISTRY.
        Examples: "resnet50", "efficientnet_b0", "vit_b16".
        Optional if ``name`` is provided.
    num_classes:
        Number of output classes for the task. Must be positive.
    config:
        Optional model-specific configuration dict. Keys depend on the
        architecture. If None or missing keys, defaults are used from
        DEFAULT_MODEL_CONFIGS. Values in ``config`` override defaults.
    name:
        Optional alias for the architecture, typically coming from a
        higher-level config object (e.g. ExperimentConfig.model.name).
        If ``architecture`` is not provided, this value is used.
    pretrained:
        Optional override for the ``pretrained`` flag. If provided, this
        takes precedence over both DEFAULT_MODEL_CONFIGS and ``config``.
    **kwargs:
        Additional model-specific parameters. These override both
        DEFAULT_MODEL_CONFIGS and values in ``config``.

    Returns
    -------
    BaseModel
        Instantiated model ready for training/inference.

    Raises
    ------
    ValueError
        If architecture is not registered or num_classes is invalid, or
        if the supplied configuration does not match the model's
        ``__init__`` signature.

    Examples
    --------
    Basic usage with defaults:

    >>> model = build_model("resnet50", num_classes=7)
    >>> model.num_classes
    7

    Using ``name`` instead of ``architecture``:

    >>> model = build_model(name="resnet50", num_classes=7, pretrained=False)

    Custom configuration:

    >>> config = {"pretrained": False, "in_channels": 1, "dropout": 0.3}
    >>> model = build_model("resnet50", num_classes=2, config=config)
    """
    # Resolve architecture from either 'architecture' or 'name'
    architecture_key = _resolve_architecture(architecture=architecture, name=name)

    # Validate architecture
    if architecture_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown architecture: '{architecture_key}'. "
            f"Available architectures: {available}"
        )

    # Validate num_classes
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    # Get model class
    model_class = MODEL_REGISTRY[architecture_key]

    # Merge default config with user config and explicit kwargs
    merged_config: Dict[str, Any] = DEFAULT_MODEL_CONFIGS.get(
        architecture_key, {}
    ).copy()

    if config is not None:
        merged_config.update(config)

    if pretrained is not None:
        merged_config["pretrained"] = pretrained

    if kwargs:
        merged_config.update(kwargs)

    # Log model creation
    logger.info(
        f"Building {architecture_key} with num_classes={num_classes}, "
        f"config={merged_config}"
    )

    # Instantiate model
    try:
        model = model_class(num_classes=num_classes, **merged_config)
    except TypeError as e:
        # Wrap in a clearer error for config debugging
        raise ValueError(
            f"Failed to instantiate {architecture_key}: {e}. "
            f"Check that config keys match the model's __init__ parameters."
        ) from e

    # Log model info (if implemented by BaseModel)
    if hasattr(model, "get_model_info"):
        try:
            info = model.get_model_info()
            logger.info(
                f"Created {info.get('architecture', architecture_key)}: "
                f"{info.get('total_params', 0):,} total params, "
                f"{info.get('trainable_params', 0):,} trainable params"
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("get_model_info() raised; continuing without detailed stats")

    return model


def build_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """
    Build model from a complete experiment configuration dictionary.

    This is a convenience wrapper around :func:`build_model` that extracts
    the necessary fields from a nested config structure.

    Expected config structure (supports both styles):

        {
            "model": {
                "architecture": "resnet50",  # OR "name": "resnet50"
                "num_classes": 7,
                "pretrained": True,
                "dropout": 0.2,
                ...
            }
        }

    Parameters
    ----------
    config:
        Experiment configuration containing a "model" key.

    Returns
    -------
    BaseModel
        Instantiated model.

    Raises
    ------
    KeyError
        If required keys are missing from config.
    ValueError
        If architecture is invalid or num_classes is invalid.
    """
    if "model" not in config:
        raise KeyError(
            "Config must contain a 'model' key. "
            "Example: {'model': {'architecture': 'resnet50', 'num_classes': 7}}"
        )

    model_config = config["model"]

    architecture = model_config.get("architecture") or model_config.get("name")
    if architecture is None:
        raise KeyError("model config must contain either 'architecture' or 'name' key")

    if "num_classes" not in model_config:
        raise KeyError("model config must contain 'num_classes' key")

    num_classes = model_config["num_classes"]

    # Extract other model-specific params (excluding architecture/name and num_classes)
    model_params = {
        k: v
        for k, v in model_config.items()
        if k not in ["architecture", "name", "num_classes"]
    }

    return build_model(
        architecture=architecture,
        num_classes=num_classes,
        config=model_params,
    )


def list_available_architectures() -> List[str]:
    """
    List all registered model architectures.

    Returns
    -------
    list[str]
        Sorted list of available architecture names.

    Examples
    --------
    >>> architectures = list_available_architectures()
    >>> "resnet50" in architectures
    True
    """
    return sorted(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Register a new model architecture.

    This allows external code to add custom architectures to the factory.

    Parameters
    ----------
    name:
        Architecture name (will be converted to lowercase).
    model_class:
        Model class that inherits from BaseModel.

    Raises
    ------
    ValueError
        If model_class does not inherit from BaseModel.

    Examples
    --------
    >>> class CustomCNN(BaseModel):
    ...     # implementation
    ...     pass
    >>> register_model("custom_cnn", CustomCNN)
    >>> "custom_cnn" in list_available_architectures()
    True
    """
    if not issubclass(model_class, BaseModel):
        raise ValueError(
            f"Model class must inherit from BaseModel, got {model_class.__name__}"
        )

    name_lower = name.lower()
    if name_lower in MODEL_REGISTRY:
        logger.warning(f"Overwriting existing architecture: {name_lower}")

    MODEL_REGISTRY[name_lower] = model_class
    logger.info(f"Registered architecture: {name_lower}")


def get_default_config(architecture: str) -> Dict[str, Any]:
    """
    Get default configuration for an architecture.

    Parameters
    ----------
    architecture:
        Architecture name.

    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary.

    Raises
    ------
    ValueError
        If architecture is not registered.

    Examples
    --------
    >>> config = get_default_config("resnet50")
    >>> config["pretrained"]
    True
    >>> config["in_channels"]
    3
    """
    architecture_lower = architecture.lower()
    if architecture_lower not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown architecture: '{architecture}'. " f"Available: {available}"
        )

    return DEFAULT_MODEL_CONFIGS.get(architecture_lower, {}).copy()


def build_pooling(cfg):
    """Build pooling layer."""
    # Fix None-safe access
    pool_type = (cfg.global_pool or "avg").lower()

    if pool_type == "avg":
        return nn.AdaptiveAvgPool2d(1)
    elif pool_type == "max":
        return nn.AdaptiveMaxPool2d(1)
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")


# Convenience function for common use case
def build_classifier(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs: Any,
) -> BaseModel:
    """
    Quick builder for classifiers with common parameters.

    This is a simplified interface for the most common use case.

    Parameters
    ----------
    architecture:
        Architecture name (e.g., "resnet50").
    num_classes:
        Number of output classes.
    pretrained:
        Whether to use pretrained weights.
    **kwargs:
        Additional model-specific parameters.

    Returns
    -------
    BaseModel
        Instantiated model.

    Examples
    --------
    >>> model = build_classifier("resnet50", num_classes=7, pretrained=True)
    >>> model = build_classifier("resnet50", num_classes=2,
    ...                          pretrained=True, in_channels=1, dropout=0.3)
    """
    config: Dict[str, Any] = {"pretrained": pretrained}
    config.update(kwargs)
    return build_model(
        architecture=architecture, num_classes=num_classes, config=config
    )


__all__ = [
    "build_model",
    "build_model_from_config",
    "build_classifier",
    "list_available_architectures",
    "register_model",
    "get_default_config",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_CONFIGS",
]
