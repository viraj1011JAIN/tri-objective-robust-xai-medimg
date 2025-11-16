"""
Tests for src.utils.reproducibility

These are written to give real assurance that seeding works
and to exercise all relevant branches for 100% coverage.
"""

import sys

import numpy as np
import torch

from src.utils import reproducibility as repro


def test_set_global_seed_python_numpy_torch_consistent(monkeypatch):
    """Setting the same seed twice should give identical RNG outputs.

    We force torch.cuda.is_available() to False so the behaviour is
    deterministic regardless of whether CUDA is actually installed.
    """
    seed = 1234

    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False, raising=False)

    def run_once():
        repro.set_global_seed(seed, deterministic=True)
        return (
            __import__("random").random(),
            float(np.random.rand()),
            float(torch.rand(1)),
        )

    r1 = run_once()
    r2 = run_once()
    assert r1 == r2
    # PYTHONHASHSEED should be fixed as well
    assert __import__("os").environ.get("PYTHONHASHSEED") == str(seed)


def test_set_global_seed_gpu_branch_sets_cuda_seeds(monkeypatch):
    """Exercise the CUDA branch without requiring real GPU hardware."""
    calls = {}

    def fake_is_available():
        return True

    def fake_manual_seed(seed):
        calls["manual_seed"] = seed

    def fake_manual_seed_all(seed):
        calls["manual_seed_all"] = seed

    monkeypatch.setattr(
        repro.torch.cuda, "is_available", fake_is_available, raising=False
    )
    monkeypatch.setattr(
        repro.torch.cuda, "manual_seed", fake_manual_seed, raising=False
    )
    monkeypatch.setattr(
        repro.torch.cuda, "manual_seed_all", fake_manual_seed_all, raising=False
    )

    state = repro.set_global_seed(321, deterministic=True)
    assert calls["manual_seed"] == 321
    assert calls["manual_seed_all"] == 321
    assert state.seed == 321


def test_seed_worker_reproducible_across_invocations():
    """seed_worker should give deterministic Python + NumPy RNG state
    when called with the same global seed + worker id.
    """
    from random import random as py_random

    def run_worker(seed: int):
        repro.set_global_seed(seed, deterministic=True)
        repro.seed_worker(worker_id=0)
        return (py_random(), float(np.random.rand()))

    r1 = run_worker(999)
    r2 = run_worker(999)
    assert r1 == r2


def test_make_torch_generator_produces_deterministic_sequence():
    g1 = repro.make_torch_generator(7)
    g2 = repro.make_torch_generator(7)

    t1 = torch.rand(3, generator=g1)
    t2 = torch.rand(3, generator=g2)

    assert torch.allclose(t1, t2)


def test_quick_determinism_check_returns_true(monkeypatch):
    """Default path: device is None, we force CPU."""
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False, raising=False)
    assert repro.quick_determinism_check(seed=2025)


def test_quick_determinism_check_with_explicit_cpu_device():
    """Explicit device path: the function should still be deterministic."""
    cpu_device = torch.device("cpu")
    assert repro.quick_determinism_check(seed=7, device=cpu_device)


def test_get_cuda_device_names_fake_cpu(monkeypatch):
    """Force the CPU path in _get_cuda_device_names."""
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False, raising=False)
    names = repro._get_cuda_device_names()
    assert names == ()


def test_get_cuda_device_names_fake_gpu(monkeypatch):
    """Force the GPU path in _get_cuda_device_names without real hardware."""

    def fake_is_available():
        return True

    def fake_device_count():
        return 2

    def fake_get_device_name(idx: int) -> str:
        return f"FakeGPU-{idx}"

    monkeypatch.setattr(
        repro.torch.cuda, "is_available", fake_is_available, raising=False
    )
    monkeypatch.setattr(
        repro.torch.cuda, "device_count", fake_device_count, raising=False
    )
    monkeypatch.setattr(
        repro.torch.cuda, "get_device_name", fake_get_device_name, raising=False
    )

    names = repro._get_cuda_device_names()
    assert names == ("FakeGPU-0", "FakeGPU-1")


def test_get_reproducibility_state_fields_basic(monkeypatch):
    """CPU-only branch in get_reproducibility_state."""
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(repro.torch.cuda, "device_count", lambda: 0, raising=False)

    state = repro.get_reproducibility_state(seed=42, deterministic=True)
    assert state.seed == 42
    assert state.deterministic is True
    assert isinstance(state.python_version, str)
    assert isinstance(state.torch_version, str)
    assert state.cuda_available is False
    assert state.cuda_device_count == 0
    assert isinstance(state.extra, dict)


def test_get_reproducibility_state_with_fake_gpu(monkeypatch):
    """CUDA-available branch in get_reproducibility_state."""
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(repro.torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(
        repro,
        "_get_cuda_device_names",
        lambda: ("FakeGPU-0", "FakeGPU-1"),
        raising=False,
    )

    state = repro.get_reproducibility_state(seed=99, deterministic=False)
    assert state.cuda_available is True
    assert state.cuda_device_count == 2
    assert state.cuda_device_names == ("FakeGPU-0", "FakeGPU-1")


def test_summarise_reproducibility_state_contains_key_info(monkeypatch):
    """Summary for a CPU-only state (no device names)."""
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(repro.torch.cuda, "device_count", lambda: 0, raising=False)
    monkeypatch.setattr(repro, "_get_cuda_device_names", lambda: tuple(), raising=False)

    state = repro.get_reproducibility_state(seed=1, deterministic=False)
    summary = repro.summarise_reproducibility_state(state)
    assert "Seed: 1" in summary
    assert "Deterministic: False" in summary
    assert "PyTorch:" in summary
    # No GPU lines when there are no device names
    assert "GPU 0:" not in summary


def test_summarise_reproducibility_state_includes_gpu_names():
    """If cuda_device_names is non-empty, GPU lines should be printed."""
    state = repro.ReproducibilityState(
        seed=7,
        deterministic=True,
        python_version="3.x",
        torch_version="2.x",
        cuda_available=True,
        cuda_device_count=2,
        cuda_device_names=("GPU0", "GPU1"),
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        extra={},
    )
    summary = repro.summarise_reproducibility_state(state)
    assert "GPU 0: GPU0" in summary
    assert "GPU 1: GPU1" in summary


def test_reproducibility_header_includes_seed_and_device_cpu(monkeypatch):
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False, raising=False)
    header = repro.reproducibility_header(seed=123, deterministic=True)
    assert "seed=123" in header
    assert "deterministic=True" in header
    assert "device=cpu" in header
    assert "cuda=False" in header


def test_reproducibility_header_includes_seed_and_device_cuda(monkeypatch):
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: True, raising=False)
    header = repro.reproducibility_header(seed=5, deterministic=False)
    assert "seed=5" in header
    assert "deterministic=False" in header
    assert "device=cuda" in header
    assert "cuda=True" in header


def test_log_reproducibility_to_mlflow_with_dummy_module(monkeypatch):
    """Simulate mlflow module so we can test logging path, including per-GPU params."""
    logged_params = {}

    class DummyMlflow:
        def log_params(self, params):
            logged_params.update(params)

        # log_param may be called per-GPU
        def log_param(self, key, value):
            logged_params[key] = value

    dummy_mlflow = DummyMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)

    state = repro.ReproducibilityState(
        seed=17,
        deterministic=True,
        python_version="3.x",
        torch_version="2.x",
        cuda_available=True,
        cuda_device_count=2,
        cuda_device_names=("FakeGPU-0", "FakeGPU-1"),
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        extra={},
    )

    repro.log_reproducibility_to_mlflow(state)

    assert logged_params["seed"] == 17
    assert "python_version" in logged_params
    assert logged_params["cuda_device_0"] == "FakeGPU-0"
    assert logged_params["cuda_device_1"] == "FakeGPU-1"


def test_log_reproducibility_to_mlflow_with_dummy_module_no_device_names(monkeypatch):
    """When there are no cuda_device_names, per-GPU params should not be logged."""
    logged_params = {}

    class DummyMlflow:
        def log_params(self, params):
            logged_params.update(params)

        def log_param(self, key, value):
            logged_params[key] = value

    dummy_mlflow = DummyMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)

    state = repro.ReproducibilityState(
        seed=21,
        deterministic=False,
        python_version="3.x",
        torch_version="2.x",
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_names=(),
        cudnn_deterministic=False,
        cudnn_benchmark=False,
        extra={},
    )

    repro.log_reproducibility_to_mlflow(state)

    assert logged_params["seed"] == 21
    assert "cuda_device_0" not in logged_params


def test_log_reproducibility_to_mlflow_without_mlflow_module(monkeypatch):
    """If mlflow is not importable / not in sys.modules, the function should be a no-op."""
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    state = repro.ReproducibilityState(
        seed=0,
        deterministic=False,
        python_version="3.x",
        torch_version="2.x",
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_names=(),
        cudnn_deterministic=False,
        cudnn_benchmark=False,
        extra={},
    )

    # Should not raise
    repro.log_reproducibility_to_mlflow(state)
