from __future__ import annotations

"""
Tests for src.models.model_registry.ModelRegistry and ModelRecord.

These tests are intentionally integration-leaning: they exercise
both the in-memory registry behaviour and the on-disk JSON index
plus checkpoint files, using a temporary directory for isolation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.model_registry import ModelRecord, ModelRegistry

# ---------------------------------------------------------------------------
# Tiny test model
# ---------------------------------------------------------------------------


class TinyModel(BaseModel):
    """
    Minimal BaseModel subclass for registry tests.

    This model is deliberately small and fast to construct / save, while
    still exposing a couple of parameters so that state-dict loading can
    be meaningfully verified.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = False) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out = self.linear(x)
        if return_features:
            return out, {"features": out}
        return out

    def get_feature_maps(self, x: torch.Tensor, layer_names: List[str] | None = None):
        logits = self.linear(x)
        return {"features": logits}

    def get_classifier(self) -> nn.Module:
        return self.linear

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_layer_output_shapes(
        self, input_size: tuple[int, int] = (2, 2)
    ) -> Dict[str, Any]:
        # Not essential for these tests; provided for completeness
        return {"features": (self.num_classes,) + input_size}

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "TinyModel",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_root(tmp_path: Path) -> Path:
    """Temporary root directory for the model registry."""
    root = tmp_path / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def registry(registry_root: Path) -> ModelRegistry:
    """Fresh ModelRegistry instance pointing at an isolated directory."""
    return ModelRegistry(root_dir=registry_root)


@pytest.fixture
def tiny_model() -> TinyModel:
    """Fresh TinyModel instance on CPU."""
    torch.manual_seed(1234)
    return TinyModel(num_classes=3, pretrained=False)


# ---------------------------------------------------------------------------
# ModelRecord tests
# ---------------------------------------------------------------------------


class TestModelRecord:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """ModelRecord should survive a to_dict / from_dict roundtrip."""
        record = ModelRecord(
            model_key="resnet50_baseline",
            version=3,
            architecture="ResNet50Classifier",
            checkpoint_path="resnet50_baseline_v3.pt",
            tag="best",
            config={"lr": 1e-3, "scheduler": {"type": "cosine", "T_max": 50}},
            metrics={"val_acc": 0.9123, "val_loss": 0.23},
            model_info={"depth": 50, "in_channels": 3},
        )

        data = record.to_dict()
        restored = ModelRecord.from_dict(data)

        assert restored.model_key == record.model_key
        assert restored.version == record.version
        assert restored.architecture == record.architecture
        assert restored.checkpoint_path == record.checkpoint_path
        assert restored.tag == record.tag
        assert restored.config == record.config
        assert restored.metrics == record.metrics
        assert restored.model_info == record.model_info

    def test_to_dict_serialises_tensors(self) -> None:
        """to_dict must convert non-JSON types (e.g. tensors) into primitives."""
        record = ModelRecord(
            model_key="tiny",
            version=1,
            architecture="TinyModel",
            checkpoint_path="tiny_v1.pt",
            metrics={"val_acc": torch.tensor(0.9)},
        )

        data = record.to_dict()
        assert isinstance(
            data["metrics"]["val_acc"], (float, int)
        ), "Scalar tensor should be converted to a Python number"


# ---------------------------------------------------------------------------
# ModelRegistry basic behaviour
# ---------------------------------------------------------------------------


class TestModelRegistryBasics:
    def test_save_single_version_creates_entry_and_file(
        self, registry: ModelRegistry, tiny_model: TinyModel, registry_root: Path
    ) -> None:
        record = registry.save_model(
            tiny_model,
            model_key="tiny_baseline",
            tag="epoch10",
            config={"lr": 1e-3, "weight_decay": 1e-4},
            metrics={"val_acc": 0.85},
            epoch=10,
            step=1234,
        )

        # Record content
        assert isinstance(record, ModelRecord)
        assert record.model_key == "tiny_baseline"
        assert record.version == 1
        assert record.architecture == "TinyModel"
        assert record.metrics["val_acc"] == 0.85

        # Checkpoint file exists
        ckpt_path = registry_root / record.checkpoint_path
        assert ckpt_path.is_file()

        # Index JSON exists and has expected structure
        assert registry.index_path.is_file()
        with registry.index_path.open("r", encoding="utf8") as f:
            payload = json.load(f)

        assert "models" in payload
        assert "tiny_baseline" in payload["models"]
        assert len(payload["models"]["tiny_baseline"]) == 1

        # Registry API reflects the same
        assert registry.models == ["tiny_baseline"]
        versions = registry.list_versions("tiny_baseline")
        assert len(versions) == 1
        assert versions[0].version == 1

    def test_versioning_increments_and_get_latest(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        # First version
        rec1 = registry.save_model(
            tiny_model,
            model_key="tiny",
            tag="epoch5",
            metrics={"val_acc": 0.80},
            epoch=5,
        )
        # Second version
        rec2 = registry.save_model(
            tiny_model,
            model_key="tiny",
            tag="epoch10",
            metrics={"val_acc": 0.90},
            epoch=10,
        )

        assert rec1.version == 1
        assert rec2.version == 2

        versions = registry.list_versions("tiny")
        assert [r.version for r in versions] == [1, 2]

        latest = registry.get_latest("tiny")
        assert latest is not None
        assert latest.version == 2
        assert latest.tag == "epoch10"
        assert latest.metrics["val_acc"] == 0.90

    def test_save_without_model_key_uses_architecture(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        record = registry.save_model(tiny_model)

        # By default, model_key is derived from architecture
        assert record.model_key == record.architecture
        assert record.architecture == "TinyModel"

        assert record.model_key in registry.models


# ---------------------------------------------------------------------------
# ModelRegistry + checkpoint I/O
# ---------------------------------------------------------------------------


class TestModelRegistryCheckpointIO:
    def test_load_checkpoint_restores_model_and_optimizer_and_scheduler(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        # Original model + optimiser + scheduler
        model_orig = tiny_model
        optimizer_orig = torch.optim.SGD(model_orig.parameters(), lr=0.01, momentum=0.9)
        scheduler_orig = torch.optim.StepLR(optimizer_orig, step_size=5, gamma=0.1)

        # One dummy optimisation step so optimizer/scheduler have non-trivial state
        x = torch.randn(2, 4)
        y = torch.randint(0, model_orig.num_classes, (2,))
        criterion = nn.CrossEntropyLoss()
        optimizer_orig.zero_grad()
        logits = model_orig(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer_orig.step()
        scheduler_orig.step()

        record = registry.save_model(
            model_orig,
            optimizer=optimizer_orig,
            scheduler=scheduler_orig,
            model_key="tiny_io",
            metrics={"val_loss": 0.5},
            epoch=3,
            step=42,
            extra_state={"note": "test-run"},
        )

        # New model + optimiser + scheduler initialised differently
        model_new = TinyModel(num_classes=3, pretrained=False)
        # Force obviously different parameters
        for p in model_new.parameters():
            p.data.zero_()

        optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.9, momentum=0.0)
        scheduler_new = torch.optim.StepLR(optimizer_new, step_size=1, gamma=0.5)

        # Load checkpoint into new objects
        ckpt = registry.load_checkpoint(
            record,
            model=model_new,
            optimizer=optimizer_new,
            scheduler=scheduler_new,
            map_location="cpu",
            strict=True,
        )

        # Model weights should now match original
        sd_orig = model_orig.state_dict()
        sd_new = model_new.state_dict()
        assert set(sd_orig.keys()) == set(sd_new.keys())
        for key in sd_orig.keys():
            assert torch.allclose(sd_orig[key], sd_new[key])

        # Optimizer state should be non-empty
        opt_state = optimizer_new.state_dict()
        assert opt_state["state"], "optimizer state should have been restored"

        # Scheduler state should also be non-empty
        sched_state = scheduler_new.state_dict()
        assert sched_state["state"], "scheduler state should have been restored"

        # Check that training-position metadata travelled through
        assert ckpt["epoch"] == 3
        assert ckpt["step"] == 42
        assert ckpt["metrics"]["val_loss"] == 0.5
        assert ckpt["extra_state"]["note"] == "test-run"

    def test_load_checkpoint_by_model_key_uses_latest(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        # Save two versions under the same key
        registry.save_model(
            tiny_model, model_key="tiny_key", metrics={"val_acc": 0.7}, epoch=1
        )
        registry.save_model(
            tiny_model, model_key="tiny_key", metrics={"val_acc": 0.9}, epoch=2
        )

        # Load by string key (should pick latest version)
        model_new = TinyModel(num_classes=3, pretrained=False)
        ckpt = registry.load_checkpoint("tiny_key", model=model_new, map_location="cpu")

        assert ckpt["version"] == 2
        assert ckpt["metrics"]["val_acc"] == 0.9

    def test_load_checkpoint_unknown_model_key_raises(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        with pytest.raises(KeyError):
            _ = registry.load_checkpoint("does_not_exist", model=tiny_model)

    def test_missing_checkpoint_file_raises(
        self, registry: ModelRegistry, tiny_model: TinyModel, registry_root: Path
    ) -> None:
        # Save once, then delete the underlying file
        record = registry.save_model(tiny_model, model_key="tiny_missing")
        ckpt_path = registry_root / record.checkpoint_path
        assert ckpt_path.is_file()
        ckpt_path.unlink()

        with pytest.raises(FileNotFoundError):
            _ = registry.load_checkpoint(record, model=tiny_model)


# ---------------------------------------------------------------------------
# Index persistence / reload behaviour
# ---------------------------------------------------------------------------


class TestModelRegistryPersistence:
    def test_index_persists_across_registry_instances(
        self, registry_root: Path, tiny_model: TinyModel
    ) -> None:
        # First instance writes some entries
        registry1 = ModelRegistry(root_dir=registry_root)
        registry1.save_model(tiny_model, model_key="tiny", metrics={"val_acc": 0.8})
        registry1.save_model(tiny_model, model_key="tiny", metrics={"val_acc": 0.9})

        assert "tiny" in registry1.models
        assert len(registry1.list_versions("tiny")) == 2

        # New instance should load existing index from disk
        registry2 = ModelRegistry(root_dir=registry_root)
        assert "tiny" in registry2.models
        versions = registry2.list_versions("tiny")
        assert len(versions) == 2
        assert [v.version for v in versions] == [1, 2]
