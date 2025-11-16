"""
Fast smoke tests for overall project setup.

These are meant to fail early if the environment or repo layout is broken,
before running heavier unit / integration tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _project_root() -> Path:
    # tests/ lives directly under the project root
    return Path(__file__).resolve().parents[1]


def test_expected_top_level_directories_exist() -> None:
    """Check that the core repo layout is present."""
    root = _project_root()
    for dirname in ["src", "tests", "configs", "data", "docs"]:
        path = root / dirname
        assert path.exists(), f"Expected {dirname}/ to exist at {path}"
        assert path.is_dir(), f"{dirname}/ should be a directory"


def test_expected_config_subdirectories_exist() -> None:
    """
    Check that main config folders exist.

    We only enforce directory structure here; specific XAI configs like
    configs/xai/base.yaml are optional at this phase.
    """
    root = _project_root()
    cfg_root = root / "configs"

    for dirname in ["attacks", "datasets", "experiments", "hpo", "models", "xai"]:
        path = cfg_root / dirname
        assert path.exists(), f"Expected configs/{dirname}/ to exist"
        assert path.is_dir(), f"configs/{dirname} should be a directory"

    # Optional XAI base config – do not fail if it doesn't exist yet
    xai_base = cfg_root / "xai" / "base.yaml"
    if xai_base.exists():
        assert xai_base.is_file(), "configs/xai/base.yaml should be a regular file"
    else:
        pytest.skip(
            "configs/xai/base.yaml not present yet – skipping optional XAI config check"
        )


def test_data_directory_layout_is_valid() -> None:
    """
    Validate that the top-level data directory has the basic structure.

    We do *not* require datasets to be downloaded yet, only the folders.
    """
    root = _project_root()
    data_root = root / "data"
    assert data_root.exists() and data_root.is_dir(), "data/ directory must exist"

    for sub in ["raw", "processed"]:
        subdir = data_root / sub
        assert subdir.exists() and subdir.is_dir(), f"data/{sub}/ directory must exist"

    gitignore = data_root / ".gitignore"
    assert (
        gitignore.exists()
    ), "data/.gitignore should exist to keep large files out of Git"


def test_core_project_imports() -> None:
    """
    Smoke test for core imports.

    If any of these fail, something is wrong with the installation or src layout.
    """
    import importlib

    modules = [
        "src",
        "src.attacks",
        "src.datasets",
        "src.eval",
        "src.losses",
        "src.models",
        "src.train",
        "src.utils",
        "src.utils.config",
        "src.utils.reproducibility",
        "src.utils.mlflow_utils",
        "src.xai",
        "src.cli",
    ]

    for name in modules:
        module = importlib.import_module(name)
        # Use the module so flake8 does not treat this as an unused import.
        assert module is not None


def test_cuda_flag_is_consistent() -> None:
    """
    Ensure CUDA availability flag is sensible.

    This NEVER fails on CPU-only machines. It only enforces consistency
    when CUDA is reported as available.
    """
    is_available = torch.cuda.is_available()
    assert isinstance(
        is_available, bool
    ), "torch.cuda.is_available() must return a bool"

    if is_available:
        assert (
            torch.cuda.device_count() >= 1
        ), "CUDA is reported available but no devices are visible"
