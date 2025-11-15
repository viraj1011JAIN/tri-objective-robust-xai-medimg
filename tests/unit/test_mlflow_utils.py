from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils import mlflow_utils


class DummyRun:
    def __init__(self, run_name: str | None) -> None:
        self.run_name = run_name
        # Minimal stub to look like an mlflow ActiveRun
        self.info = SimpleNamespace(run_id="dummy-run-id")


def test_build_experiment_and_run_name_without_extra_tag() -> None:
    exp_name, run_name = mlflow_utils.build_experiment_and_run_name(
        dataset="NIH-CXR",
        model="resnet50",
        objective="tri-objective",
        extra_tag=None,
    )

    assert exp_name == "NIH-CXR__tri-objective"
    assert run_name == "resnet50"


def test_build_experiment_and_run_name_with_extra_tag() -> None:
    exp_name, run_name = mlflow_utils.build_experiment_and_run_name(
        dataset="ISIC-2020",
        model="efficientnet-b0",
        objective="adversarial",
        extra_tag="pgd-eps-0.03",
    )

    assert exp_name == "ISIC-2020__adversarial"
    assert run_name == "efficientnet-b0__pgd-eps-0.03"


def test_init_mlflow_default_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    set_tracking_calls: list[str] = []
    experiments: list[str] = []

    def fake_set_tracking_uri(uri: str) -> None:
        set_tracking_calls.append(uri)

    def fake_set_experiment(name: str) -> None:
        experiments.append(name)

    def fake_start_run(run_name: str | None = None) -> DummyRun:
        return DummyRun(run_name)

    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(mlflow_utils.mlflow, "set_experiment", fake_set_experiment)
    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", fake_start_run)

    run = mlflow_utils.init_mlflow(
        experiment_name="CIFAR10-debug__baseline",
        run_name="SimpleCIFARNet__seed-42",
        tracking_uri=None,
    )

    # We get our dummy run back
    assert isinstance(run, DummyRun)
    assert run.run_name == "SimpleCIFARNet__seed-42"

    # Exactly one call to set_tracking_uri with default "file:..." path
    assert len(set_tracking_calls) == 1
    uri = set_tracking_calls[0]
    assert uri.startswith("file:")
    # Normalize for Windows vs POSIX
    normalized = uri.replace("\\", "/")
    assert "mlruns" in normalized

    # Experiment name must be set correctly
    assert experiments == ["CIFAR10-debug__baseline"]


def test_init_mlflow_custom_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    set_tracking_calls: list[str] = []

    def fake_set_tracking_uri(uri: str) -> None:
        set_tracking_calls.append(uri)

    def fake_set_experiment(name: str) -> None:
        # no-op in this test
        pass

    def fake_start_run(run_name: str | None = None) -> DummyRun:
        return DummyRun(run_name)

    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(mlflow_utils.mlflow, "set_experiment", fake_set_experiment)
    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", fake_start_run)

    custom_uri = "sqlite:///mlflow.db"

    run = mlflow_utils.init_mlflow(
        experiment_name="TEST-EXP",
        run_name="test-run",
        tracking_uri=custom_uri,
    )

    assert isinstance(run, DummyRun)
    assert run.run_name == "test-run"

    # Ensure our explicit URI is respected (no override to file:./mlruns)
    assert set_tracking_calls == [custom_uri]
