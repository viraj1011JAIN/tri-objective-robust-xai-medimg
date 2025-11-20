"""
Test suite for src/training/train_baseline.py.

Covers:
- setup_logging
- create_dataloaders
- main() in both config / no-config modes
- MLflow metric logging branches

Author: Viraj Pankaj Jain
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.train_baseline import create_dataloaders, main, setup_logging

# ---------------------------------------------------------------------------
# Fixtures & simple helpers
# ---------------------------------------------------------------------------


class DummyModel(nn.Module):
    """Very small CNN-ish model for smoke tests."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


class DummyTrainer:
    """Trainer stub used to avoid heavy training in tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.best_epoch = 1
        self.best_val_loss = 0.123

    def fit(self) -> Dict[str, Any]:
        # Non-empty val_loss -> triggers 'True' branch in main()
        return {"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.8]}


class DummyTrainerNoVal:
    """Trainer stub that simulates empty validation history."""

    def __init__(self, *args, **kwargs) -> None:
        self.best_epoch = 0
        self.best_val_loss = 1.234

    def fit(self) -> Dict[str, Any]:
        # Empty list -> 'False' branch in 'if history.get("val_loss")'
        return {"train_loss": [1.0], "val_loss": []}


class DummySubCfg:
    """Mimic a pydantic sub-config with model_dump()."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._fields)


class DummyExperimentCfg:
    """Full experiment config object for load_experiment_config."""

    def __init__(self, dataset_name: str = "isic2018", num_classes: int = 7) -> None:
        self.experiment = DummySubCfg(name="test_experiment")
        self.model = DummySubCfg(
            name="resnet18", num_classes=num_classes, pretrained=False
        )
        self.dataset = DummySubCfg(name=dataset_name, batch_size=16)
        self.training = DummySubCfg(
            max_epochs=2,
            learning_rate=1e-4,
            weight_decay=1e-6,
            device="cpu",
            early_stopping_patience=3,
        )


@pytest.fixture
def temp_dirs():
    """Temporary directories for logs, checkpoints, and results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "log_dir": base / "logs",
            "checkpoint_dir": base / "checkpoints",
            "results_dir": base / "results",
        }


@pytest.fixture
def base_args(temp_dirs):
    """Base argparse.Namespace used for main() tests."""
    return argparse.Namespace(
        config=None,
        seed=42,
        device="cpu",
        checkpoint_dir=str(temp_dirs["checkpoint_dir"]),
        results_dir=str(temp_dirs["results_dir"]),
        log_dir=str(temp_dirs["log_dir"]),
    )


# ---------------------------------------------------------------------------
# setup_logging tests
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_creates_log_directory_and_file(self, temp_dirs):
        log_dir = temp_dirs["log_dir"]
        assert not log_dir.exists()

        setup_logging(log_dir)

        # Directory is created and we can log into train.log
        assert log_dir.exists()
        logger = logging.getLogger("test_logger")
        logger.info("Test message")

        log_file = log_dir / "train.log"
        assert log_file.exists()

    def test_logging_captures_messages(self, temp_dirs, caplog):
        log_dir = temp_dirs["log_dir"]
        setup_logging(log_dir)

        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("another_test_logger")
            logger.info("Hello from logging test")

        assert any("Hello from logging test" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# create_dataloaders tests
# ---------------------------------------------------------------------------


class TestCreateDataloaders:
    def test_isic2018_dataset(self):
        train_loader, val_loader, num_classes = create_dataloaders(
            batch_size=16, dataset="isic2018"
        )

        assert num_classes == 7
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert len(train_loader.dataset) == int(256 * 0.8)
        assert len(val_loader.dataset) == int(256 * 0.2)

    def test_nih_chestxray14_dataset(self):
        train_loader, val_loader, num_classes = create_dataloaders(
            batch_size=32, dataset="nih_chestxray14"
        )

        assert num_classes == 14
        assert len(train_loader.dataset) == int(512 * 0.8)
        assert len(val_loader.dataset) == int(512 * 0.2)

    def test_chest_xray_alias(self):
        _, _, num_classes = create_dataloaders(batch_size=8, dataset="chest_xray")
        assert num_classes == 14

    def test_case_insensitive_names(self):
        _, _, num_classes = create_dataloaders(batch_size=8, dataset="ISIC2018")
        assert num_classes == 7

        _, _, num_classes = create_dataloaders(batch_size=8, dataset="NIH_ChestXray14")
        assert num_classes == 14

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_dataloaders(batch_size=16, dataset="unknown_dataset")

    def test_data_shapes(self):
        train_loader, _, num_classes = create_dataloaders(
            batch_size=4, dataset="isic2018"
        )
        images, labels = next(iter(train_loader))

        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        assert labels.max().item() < num_classes


# ---------------------------------------------------------------------------
# main() tests (no-config and with-config)
# ---------------------------------------------------------------------------


class TestMain:
    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_without_config_uses_defaults(
        self,
        mock_set_global_seed,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """Main with no config should use the default baseline settings."""
        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()

        # Simulate MLflow context manager
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        mock_set_global_seed.assert_called_once_with(42)
        mock_mlflow.set_experiment.assert_called_once_with("baseline")
        mock_build_model.assert_called_once()

        # Results file should exist
        results_dir = Path(temp_dirs["results_dir"])
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1

        with result_files[0].open() as f:
            results = json.load(f)

        assert results["seed"] == 42
        assert results["model"] == "resnet50"
        assert results["dataset"] == "isic2018"
        assert "best_epoch" in results
        assert "best_val_loss" in results
        assert "history" in results
        assert results["history"]["train_loss"]
        assert results["history"]["val_loss"]

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.load_experiment_config")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_with_config_uses_experiment_name(
        self,
        mock_set_global_seed,
        mock_load_experiment_config,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """Main with config should take experiment name from config object."""
        base_args.config = "fake_config.yaml"

        mock_load_experiment_config.return_value = DummyExperimentCfg()
        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        mock_load_experiment_config.assert_called_once_with("fake_config.yaml")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.load_experiment_config")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_chest_xray_dataset_uses_14_classes(
        self,
        mock_set_global_seed,
        mock_load_experiment_config,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """Config with NIH ChestX-ray dataset should use 14 classes."""
        base_args.config = "config.yaml"
        mock_load_experiment_config.return_value = DummyExperimentCfg(
            dataset_name="nih_chestxray14", num_classes=14
        )

        mock_build_model.return_value = DummyModel(num_classes=14)
        mock_trainer_cls.return_value = DummyTrainer()

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        # Ensure build_model was called with the correct num_classes
        _, kwargs = mock_build_model.call_args
        assert kwargs["num_classes"] == 14

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_logs_metrics_with_and_without_val_loss(
        self,
        mock_set_global_seed,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """
        Exercise both branches of:

            if history.get("val_loss"):
                mlflow.log_metric("final_val_loss", ...)

        by running main() once with non-empty val_loss and once with empty.
        """
        mock_build_model.return_value = DummyModel(num_classes=7)

        # --- Run 1: non-empty val_loss (True branch) --------------------
        mock_trainer_cls.return_value = DummyTrainer()
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        metric_names_true = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        assert "best_val_loss" in metric_names_true
        assert "best_epoch" in metric_names_true
        assert "final_val_loss" in metric_names_true

        # Reset call history
        mock_mlflow.log_metric.reset_mock()

        # --- Run 2: empty val_loss (False branch) ----------------------
        mock_trainer_cls.return_value = DummyTrainerNoVal()

        main(base_args)

        metric_names_false = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        assert "best_val_loss" in metric_names_false
        assert "best_epoch" in metric_names_false
        assert "final_val_loss" not in metric_names_false

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_mlflow_run_name_includes_seed(
        self,
        mock_set_global_seed,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """MLflow run name should include the seed value."""
        base_args.seed = 99

        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        mock_mlflow.start_run.assert_called_once_with(run_name="seed_99")
