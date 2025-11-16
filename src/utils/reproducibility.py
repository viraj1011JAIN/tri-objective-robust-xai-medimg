"""
Reproducibility utilities for deterministic experiments.
Handles seed setting, environment configuration, and reproducibility state tracking.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReproducibilityState:
    """Snapshot of key reproducibility-related settings and environment."""

    seed: int
    deterministic: bool
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: Tuple[str, ...]
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    extra: Dict[str, Any]


def _get_cuda_device_names() -> Tuple[str, ...]:
    """Return a tuple of CUDA device names (or empty tuple if no CUDA)."""
    if not torch.cuda.is_available():
        return tuple()

    names: list[str] = []
    for idx in range(torch.cuda.device_count()):
        names.append(torch.cuda.get_device_name(idx))
    return tuple(names)


def get_reproducibility_state(seed: int, deterministic: bool) -> ReproducibilityState:
    """Capture current reproducibility state."""
    cuda_available = torch.cuda.is_available()
    # Device count is environment-specific; we don't care about both branches for coverage.
    cuda_device_count = (
        torch.cuda.device_count() if cuda_available else 0
    )  # pragma: no branch

    return ReproducibilityState(
        seed=seed,
        deterministic=deterministic,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=_get_cuda_device_names(),
        cudnn_deterministic=getattr(torch.backends.cudnn, "deterministic", False),
        cudnn_benchmark=getattr(torch.backends.cudnn, "benchmark", False),
        extra={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "env_PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "env_CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    )


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    *,
    logger: Optional[logging.Logger] = None,
) -> ReproducibilityState:
    """Set random seeds for Python, NumPy and PyTorch (CPU + CUDA)."""
    log = logger or LOGGER

    # 0) Python hash seed (affects iteration order over dicts etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 1) Python + NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 2) PyTorch CPU
    torch.manual_seed(seed)

    # 3) PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4) Deterministic algorithms (optional, environment-specific)
    if hasattr(torch, "use_deterministic_algorithms"):  # pragma: no branch
        torch.use_deterministic_algorithms(deterministic)

    # 5) cuDNN flags (also environment-specific)
    if hasattr(torch.backends, "cudnn"):  # pragma: no branch
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    # 6) CUBLAS workspace config (optional but recommended for CUDA)
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    state = get_reproducibility_state(seed=seed, deterministic=deterministic)
    log.info("Reproducibility state: %s", json.dumps(asdict(state), indent=2))
    return state


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducible workers.

    Usage:

        loader = DataLoader(
            dataset,
            batch_size=...,
            num_workers=...,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    Follows the pattern recommended in the PyTorch docs.
    """
    # torch.initial_seed() is different for each worker when used with
    # DataLoader(generator=...), so we mod by 2**32 to map to valid seeds.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator with a given seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def quick_determinism_check(
    seed: int = 12345,
    device: Optional[torch.device] = None,
) -> bool:
    """Run a tiny deterministic check.

    Returns:
        True if two forward passes with the same seed produce identical tensors.
    """
    if device is None:
        # The actual choice of device is environment-specific; we don't require
        # branch coverage over CPU vs CUDA here.
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # pragma: no branch

    set_global_seed(seed, deterministic=True)

    x1 = torch.randn(4, 4, device=device)
    set_global_seed(seed, deterministic=True)
    x2 = torch.randn(4, 4, device=device)

    return torch.allclose(x1, x2)


def summarise_reproducibility_state(state: ReproducibilityState) -> str:
    """Pretty-print a summary string for logs / README / thesis appendix."""
    lines: list[str] = [
        f"Seed: {state.seed}",
        f"Deterministic: {state.deterministic}",
        f"Python: {state.python_version}",
        f"PyTorch: {state.torch_version}",
        f"CUDA available: {state.cuda_available}",
        f"CUDA devices: {state.cuda_device_count}",
    ]
    if state.cuda_device_names:
        for idx, name in enumerate(state.cuda_device_names):
            lines.append(f"  - GPU {idx}: {name}")
    lines.append(f"cuDNN deterministic: {state.cudnn_deterministic}")
    lines.append(f"cuDNN benchmark: {state.cudnn_benchmark}")
    return "\n".join(lines)


def reproducibility_header(seed: int, deterministic: bool) -> str:
    """Short header suitable for logging / MLflow tags."""
    device = "cuda" if torch.cuda.is_available() else "cpu"  # pragma: no branch
    return (
        f"seed={seed} | deterministic={deterministic} | "
        f"device={device} | cuda={torch.cuda.is_available()}"
    )


def log_reproducibility_to_mlflow(state: ReproducibilityState) -> None:
    """Best-effort logging of reproducibility info to MLflow, if available."""
    mlflow_mod = sys.modules.get("mlflow")
    if mlflow_mod is None:
        LOGGER.debug("MLflow not available; skipping reproducibility logging.")
        return

    mlflow = mlflow_mod

    mlflow.log_params(
        {
            "seed": state.seed,
            "deterministic": state.deterministic,
            "python_version": state.python_version,
            "torch_version": state.torch_version,
            "cuda_available": state.cuda_available,
            "cuda_device_count": state.cuda_device_count,
            "cudnn_deterministic": state.cudnn_deterministic,
            "cudnn_benchmark": state.cudnn_benchmark,
        }
    )

    if state.cuda_device_names:
        for idx, name in enumerate(state.cuda_device_names):
            mlflow.log_param(f"cuda_device_{idx}", name)
