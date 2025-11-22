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
            eval_every_n_epochs=1,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
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
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_mlflow_import_available(self):
        """Test that MLflow import is handled correctly."""
        # MLflow should be imported successfully in the module
        from src.training import train_baseline

        assert train_baseline.mlflow is not None

    def test_mlflow_import_exception_handler(self):
        """Test the except ImportError block for mlflow."""
        import builtins
        import importlib
        import sys

        # Save original states
        original_import = builtins.__import__
        original_mlflow = sys.modules.get("mlflow")
        original_tb = sys.modules.get("src.training.train_baseline")

        def mock_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("Mocked mlflow import failure")
            return original_import(name, *args, **kwargs)

        try:
            # Clean up existing imports
            for key in list(sys.modules.keys()):
                if "src.training.train_baseline" in key:
                    del sys.modules[key]
            if "mlflow" in sys.modules:
                del sys.modules["mlflow"]

            # Mock import to raise error
            builtins.__import__ = mock_import

            # Import should trigger except block
            import src.training.train_baseline as tb

            # MLflow should be None after ImportError
            assert tb.mlflow is None

        finally:
            # Restore everything
            builtins.__import__ = original_import
            if original_mlflow:
                sys.modules["mlflow"] = original_mlflow
            if original_tb:
                sys.modules["src.training.train_baseline"] = original_tb

            # Reload with real mlflow
            if "src.training.train_baseline" in sys.modules:
                importlib.reload(sys.modules["src.training.train_baseline"])


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

        assert any(
            "Hello from logging test" in r.message for r in caplog.records
        )


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
        """Test chest_xray alias maps to NIH ChestX-ray14."""
        _, _, num_classes = create_dataloaders(
            batch_size=8, dataset="chest_xray"
        )
        assert num_classes == 14

    def test_case_insensitive_names(self):
        _, _, num_classes = create_dataloaders(
            batch_size=8, dataset="ISIC2018"
        )
        assert num_classes == 7

        _, _, num_classes = create_dataloaders(
            batch_size=8, dataset="NIH_ChestXray14"
        )
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

        metric_names_true = [
            c.args[0] for c in mock_mlflow.log_metric.call_args_list
        ]
        assert "best_val_loss" in metric_names_true
        assert "best_epoch" in metric_names_true
        assert "final_val_loss" in metric_names_true

        # Reset call history
        mock_mlflow.log_metric.reset_mock()

        # --- Run 2: empty val_loss (False branch) ----------------------
        mock_trainer_cls.return_value = DummyTrainerNoVal()

        main(base_args)

        metric_names_false = [
            c.args[0] for c in mock_mlflow.log_metric.call_args_list
        ]
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


# ---------------------------------------------------------------------------
# _cfg_from_experiment_object tests
# ---------------------------------------------------------------------------


class TestCfgFromExperimentObject:
    def test_converts_pydantic_config_to_dict(self):
        """Test that _cfg_from_experiment_object converts config correctly."""
        from src.training.train_baseline import _cfg_from_experiment_object

        cfg_obj = DummyExperimentCfg(dataset_name="isic2018", num_classes=7)
        result = _cfg_from_experiment_object(cfg_obj, device_fallback="cuda")

        assert "experiment" in result
        assert "model" in result
        assert "dataset" in result
        assert "training" in result

        assert result["experiment"]["name"] == "test_experiment"
        assert result["model"]["name"] == "resnet18"
        assert result["model"]["num_classes"] == 7
        assert result["dataset"]["name"] == "isic2018"
        assert result["training"]["device"] == "cpu"  # From config

    def test_uses_device_fallback_when_not_in_config(self):
        """Test that device_fallback is used when device not in training."""
        from src.training.train_baseline import _cfg_from_experiment_object

        # Create config without device in training
        cfg_obj = DummyExperimentCfg()
        # Remove device from training config
        cfg_obj.training = DummySubCfg(
            max_epochs=2,
            learning_rate=1e-4,
            weight_decay=1e-6,
            early_stopping_patience=3,
            # No device key
        )

        result = _cfg_from_experiment_object(cfg_obj, device_fallback="cuda")

        # Should use fallback
        assert result["training"]["device"] == "cuda"

    def test_preserves_all_config_fields(self):
        """Test that all fields from config are preserved."""
        from src.training.train_baseline import _cfg_from_experiment_object

        cfg_obj = DummyExperimentCfg()
        result = _cfg_from_experiment_object(cfg_obj, device_fallback="cpu")

        # Check all fields are preserved
        assert result["model"]["pretrained"] is False
        assert result["training"]["max_epochs"] == 2
        assert result["training"]["learning_rate"] == 1e-4
        assert result["training"]["weight_decay"] == 1e-6
        assert result["training"]["early_stopping_patience"] == 3


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestCreateDataloadersEdgeCases:
    def test_isic_alias_variants(self):
        """Test all ISIC alias variants."""
        for alias in ["isic", "isic_2018", "ISIC", "ISIC_2018"]:
            _, _, num_classes = create_dataloaders(batch_size=8, dataset=alias)
            assert num_classes == 7

    def test_nih_alias_variants(self):
        """Test all NIH ChestX-ray alias variants."""
        for alias in [
            "nih_chestxray14",
            "chest_x_ray",
            "chestxray14",
            "NIH_CHESTXRAY14",
        ]:
            _, _, num_classes = create_dataloaders(batch_size=8, dataset=alias)
            assert num_classes == 14

    def test_dataloader_deterministic_split(self):
        """Test that dataloaders have deterministic splits."""
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=16, dataset="isic2018"
        )

        # Verify consistent split sizes
        assert len(train_loader.dataset) == 204
        assert len(val_loader.dataset) == 51


class TestMainIntegration:
    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.load_experiment_config")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_creates_checkpoint_directory_with_seed(
        self,
        mock_set_global_seed,
        mock_load_experiment_config,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """Test that checkpoint directory is created with seed suffix."""
        base_args.config = "config.yaml"
        base_args.seed = 123

        mock_load_experiment_config.return_value = DummyExperimentCfg()
        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        # Check checkpoint directory was created with seed
        checkpoint_dir = Path(temp_dirs["checkpoint_dir"]) / "seed_123"
        assert checkpoint_dir.exists()

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_logs_all_hyperparameters(
        self,
        mock_set_global_seed,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
    ):
        """Test that all hyperparameters are logged to MLflow."""
        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        # Check that log_params was called with correct params
        params_call = mock_mlflow.log_params.call_args
        params = params_call[0][0]

        assert params["seed"] == 42
        assert params["model"] == "resnet50"
        assert params["dataset"] == "isic2018"
        assert params["lr"] == 1e-3
        assert params["weight_decay"] == 1e-5
        assert params["max_epochs"] == 10

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_saves_results_json_with_correct_name(
        self,
        mock_set_global_seed,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
        temp_dirs,
    ):
        """Test that results JSON is saved with correct filename."""
        base_args.seed = 999

        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        # Check filename format: {model}_{dataset}_seed{seed}.json
        results_file = (
            Path(temp_dirs["results_dir"]) / "resnet50_isic2018_seed999.json"
        )
        assert results_file.exists()

        with results_file.open() as f:
            results = json.load(f)

        assert results["seed"] == 999
        assert results["model"] == "resnet50"
        assert results["dataset"] == "isic2018"

    @patch("src.training.train_baseline.BaselineTrainer")
    @patch("src.training.train_baseline.build_model")
    @patch("src.training.train_baseline.mlflow")
    @patch("src.training.train_baseline.load_experiment_config")
    @patch("src.training.train_baseline.set_global_seed")
    def test_main_with_config_device_override(
        self,
        mock_set_global_seed,
        mock_load_experiment_config,
        mock_mlflow,
        mock_build_model,
        mock_trainer_cls,
        base_args,
    ):
        """Test device from args.device is used as fallback."""
        base_args.config = "config.yaml"
        base_args.device = "cuda"

        # Config has device="cpu" in training
        mock_load_experiment_config.return_value = DummyExperimentCfg()
        mock_build_model.return_value = DummyModel(num_classes=7)
        mock_trainer_cls.return_value = DummyTrainer()

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        main(base_args)

        # Trainer should be called with device from config
        trainer_call = mock_trainer_cls.call_args
        assert trainer_call[1]["device"].type == "cpu"  # From config, not args
