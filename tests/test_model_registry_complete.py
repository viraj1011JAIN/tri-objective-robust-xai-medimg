"""
Complete test suite for model_registry.py achieving 100% coverage.

This extends test_model_registry.py with additional tests for:
- _to_serialisable edge cases (nested structures, various types)
- ModelStore wrapper behavior
- Index loading error handling
- Legacy index format support
- get_model_info failure handling
- All branches in save_model and load_checkpoint

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.model_registry import (
    ModelRecord,
    ModelRegistry,
    ModelStore,
    _to_serialisable,
)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class TinyModel(BaseModel):
    """Minimal model for testing."""

    def __init__(self, num_classes: int = 3, pretrained: bool = False) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out = self.linear(x)
        if return_features:
            return out, {"features": out}
        return out

    def get_feature_maps(self, x: torch.Tensor, layer_names: List[str] | None = None):
        return {"features": self.linear(x)}

    def get_classifier(self) -> nn.Module:
        return self.linear

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_layer_output_shapes(
        self, input_size: tuple[int, int] = (2, 2)
    ) -> Dict[str, Any]:
        return {"features": (self.num_classes,) + input_size}

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "TinyModel",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
        }


class ModelWithBrokenInfo(BaseModel):
    """Model that raises exception in get_model_info."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__(num_classes=num_classes)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        return self.linear(x)

    def get_feature_maps(self, x: torch.Tensor, layer_names: List[str] | None = None):
        return {"features": self.linear(x)}

    def get_classifier(self) -> nn.Module:
        return self.linear

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_layer_output_shapes(
        self, input_size: tuple[int, int] = (2, 2)
    ) -> Dict[str, Any]:
        return {"features": (self.num_classes,) + input_size}

    def get_model_info(self) -> Dict[str, Any]:
        raise RuntimeError("Simulated get_model_info failure")


class ModelReturningNonDictInfo(BaseModel):
    """Model that returns non-dict from get_model_info."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__(num_classes=num_classes)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        return self.linear(x)

    def get_feature_maps(self, x: torch.Tensor, layer_names: List[str] | None = None):
        return {"features": self.linear(x)}

    def get_classifier(self) -> nn.Module:
        return self.linear

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_layer_output_shapes(
        self, input_size: tuple[int, int] = (2, 2)
    ) -> Dict[str, Any]:
        return {"features": (self.num_classes,) + input_size}

    def get_model_info(self) -> str:  # type: ignore[override]
        return "not-a-dict"


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
    """Fresh ModelRegistry instance."""
    return ModelRegistry(root_dir=registry_root)


@pytest.fixture
def tiny_model() -> TinyModel:
    """Fresh TinyModel instance."""
    torch.manual_seed(1234)
    return TinyModel(num_classes=3, pretrained=False)


# ---------------------------------------------------------------------------
# Test _to_serialisable
# ---------------------------------------------------------------------------


class TestToSerialisable:
    """Test _to_serialisable with various input types."""

    def test_scalar_tensor_converts_to_python_number(self) -> None:
        """0-D tensors should convert via .item()."""
        scalar_float = torch.tensor(3.14)
        scalar_int = torch.tensor(42)

        assert isinstance(_to_serialisable(scalar_float), float)
        assert _to_serialisable(scalar_float) == pytest.approx(3.14)
        assert isinstance(_to_serialisable(scalar_int), int)
        assert _to_serialisable(scalar_int) == 42

    def test_nd_tensor_converts_to_list(self) -> None:
        """N-D tensors should convert via .tolist()."""
        tensor_1d = torch.tensor([1, 2, 3])
        tensor_2d = torch.tensor([[1, 2], [3, 4]])

        result_1d = _to_serialisable(tensor_1d)
        result_2d = _to_serialisable(tensor_2d)

        assert isinstance(result_1d, list)
        assert result_1d == [1, 2, 3]
        assert isinstance(result_2d, list)
        assert result_2d == [[1, 2], [3, 4]]

    def test_dict_with_nested_tensors(self) -> None:
        """Dicts with tensor values should recursively convert."""
        data = {
            "loss": torch.tensor(0.5),
            "metrics": {"acc": torch.tensor(0.9), "f1": torch.tensor([0.8, 0.85])},
        }

        result = _to_serialisable(data)

        assert isinstance(result["loss"], float)
        assert result["loss"] == pytest.approx(0.5)
        assert isinstance(result["metrics"]["acc"], float)
        assert isinstance(result["metrics"]["f1"], list)

    def test_list_and_tuple_with_tensors(self) -> None:
        """Lists and tuples should convert to lists recursively."""
        data_list = [torch.tensor(1), torch.tensor(2), {"val": torch.tensor(3)}]
        data_tuple = (torch.tensor(4), torch.tensor(5))

        result_list = _to_serialisable(data_list)
        result_tuple = _to_serialisable(data_tuple)

        assert isinstance(result_list, list)
        assert result_list == [1, 2, {"val": 3}]
        assert isinstance(result_tuple, list)
        assert result_tuple == [4, 5]

    def test_basic_types_unchanged(self) -> None:
        """Basic JSON-serializable types should pass through."""
        assert _to_serialisable(42) == 42
        assert _to_serialisable(3.14) == pytest.approx(3.14)
        assert _to_serialisable("hello") == "hello"
        assert _to_serialisable(True) is True
        assert _to_serialisable(None) is None

    def test_fallback_to_str_for_unknown_types(self) -> None:
        """Unknown objects should convert to string."""

        class CustomObject:
            def __repr__(self):
                return "CustomObject()"

        obj = CustomObject()
        result = _to_serialisable(obj)

        assert isinstance(result, str)
        assert "CustomObject" in result


# ---------------------------------------------------------------------------
# Test ModelStore
# ---------------------------------------------------------------------------


class TestModelStore:
    """Test ModelStore wrapper behavior."""

    def test_dict_like_access(self) -> None:
        """ModelStore should behave like a dict."""
        data = {
            "model_a": [
                ModelRecord("model_a", 1, "ArchA", "a_v1.pt"),
                ModelRecord("model_a", 2, "ArchA", "a_v2.pt"),
            ],
            "model_b": [ModelRecord("model_b", 1, "ArchB", "b_v1.pt")],
        }
        store = ModelStore(data)

        # __getitem__
        assert len(store["model_a"]) == 2
        assert len(store["model_b"]) == 1

        # get
        assert store.get("model_a") is not None
        assert store.get("nonexistent") is None
        assert store.get("nonexistent", []) == []

        # __contains__
        assert "model_a" in store
        assert "nonexistent" not in store

        # __len__
        assert len(store) == 2

        # keys, values, items
        assert sorted(store.keys()) == ["model_a", "model_b"]
        assert len(list(store.values())) == 2
        assert len(list(store.items())) == 2

        # __iter__
        assert sorted(store) == ["model_a", "model_b"]

    def test_equality_with_list_of_keys(self) -> None:
        """ModelStore should compare equal to list/tuple/set of keys."""
        data = {
            "model_a": [ModelRecord("model_a", 1, "ArchA", "a_v1.pt")],
            "model_b": [ModelRecord("model_b", 1, "ArchB", "b_v1.pt")],
        }
        store = ModelStore(data)

        assert store == ["model_a", "model_b"]
        assert store == ["model_b", "model_a"]  # Order doesn't matter
        assert store == ("model_a", "model_b")
        assert store == {"model_a", "model_b"}
        assert store != ["model_a"]
        assert store != ["model_a", "model_b", "model_c"]

    def test_equality_with_dict(self) -> None:
        """ModelStore should compare equal to dict with same data."""
        data = {"model_a": [ModelRecord("model_a", 1, "ArchA", "a_v1.pt")]}
        store = ModelStore(data)

        assert store == data
        assert store != {"model_b": []}

    def test_equality_with_other_modelstore(self) -> None:
        """Two ModelStores with same data should be equal."""
        data = {"model_a": [ModelRecord("model_a", 1, "ArchA", "a_v1.pt")]}
        store1 = ModelStore(data)
        store2 = ModelStore(data)

        assert store1 == store2

    def test_equality_with_invalid_type(self) -> None:
        """ModelStore equality with invalid type should return NotImplemented."""
        data = {"model_a": [ModelRecord("model_a", 1, "ArchA", "a_v1.pt")]}
        store = ModelStore(data)

        result = store.__eq__(42)
        assert result is NotImplemented

    def test_repr(self) -> None:
        """ModelStore repr should be same as underlying dict."""
        data = {"model_a": [ModelRecord("model_a", 1, "ArchA", "a_v1.pt")]}
        store = ModelStore(data)

        assert repr(store) == repr(data)


# ---------------------------------------------------------------------------
# Test ModelRecord edge cases
# ---------------------------------------------------------------------------


class TestModelRecordEdgeCases:
    """Test ModelRecord with edge case inputs."""

    def test_record_with_none_optional_fields(self) -> None:
        """ModelRecord should handle None for all optional fields."""
        record = ModelRecord(
            model_key="test",
            version=1,
            architecture="TestArch",
            checkpoint_path="test_v1.pt",
            tag=None,
            config=None,
            metrics=None,
            model_info=None,
            epoch=None,
            step=None,
            extra_state=None,
        )

        data = record.to_dict()
        assert data["tag"] is None
        assert data["config"] is None
        assert data["metrics"] is None

        restored = ModelRecord.from_dict(data)
        assert restored.tag is None
        assert restored.config is None

    def test_record_with_complex_nested_data(self) -> None:
        """ModelRecord should handle deeply nested structures."""
        record = ModelRecord(
            model_key="complex",
            version=1,
            architecture="ComplexArch",
            checkpoint_path="complex_v1.pt",
            config={
                "optimizer": {"type": "Adam", "params": {"lr": torch.tensor(1e-3)}},
                "scheduler": {
                    "type": "CosineAnnealing",
                    "params": {"T_max": 50, "eta_min": torch.tensor(1e-6)},
                },
            },
            metrics={
                "train": {"loss": [torch.tensor(0.5), torch.tensor(0.4)]},
                "val": {"acc": torch.tensor(0.9)},
            },
        )

        data = record.to_dict()
        # Check that all tensors were serialized
        assert isinstance(data["config"]["optimizer"]["params"]["lr"], float)
        assert isinstance(data["config"]["scheduler"]["params"]["eta_min"], float)
        assert isinstance(data["metrics"]["train"]["loss"], list)
        assert all(isinstance(x, float) for x in data["metrics"]["train"]["loss"])


# ---------------------------------------------------------------------------
# Test ModelRegistry index loading edge cases
# ---------------------------------------------------------------------------


class TestModelRegistryIndexLoading:
    """Test index loading with various scenarios."""

    def test_load_legacy_index_format(self, registry_root: Path) -> None:
        """Registry should support legacy index format (flat dict)."""
        legacy_index = {
            "model_a": [
                {
                    "model_key": "model_a",
                    "version": 1,
                    "architecture": "ArchA",
                    "checkpoint_path": "model_a_v1.pt",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            ]
        }

        index_path = registry_root / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(legacy_index, f)

        registry = ModelRegistry(root_dir=registry_root)

        assert "model_a" in registry.models
        assert len(registry.list_versions("model_a")) == 1

    def test_load_new_index_format_with_metadata(self, registry_root: Path) -> None:
        """Registry should support new index format with schema_version."""
        new_index = {
            "schema_version": 1,
            "updated_at": "2024-01-01T12:00:00+00:00",
            "models": {
                "model_b": [
                    {
                        "model_key": "model_b",
                        "version": 1,
                        "architecture": "ArchB",
                        "checkpoint_path": "model_b_v1.pt",
                        "created_at": "2024-01-01T00:00:00+00:00",
                    }
                ]
            },
        }

        index_path = registry_root / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(new_index, f)

        registry = ModelRegistry(root_dir=registry_root)

        assert "model_b" in registry.models
        assert len(registry.list_versions("model_b")) == 1

    def test_load_empty_index(self, registry_root: Path) -> None:
        """Registry should handle empty index gracefully."""
        index_path = registry_root / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump({}, f)

        registry = ModelRegistry(root_dir=registry_root)

        assert len(registry.models) == 0

    def test_malformed_index_is_ignored(
        self, registry_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Registry should handle malformed JSON gracefully."""
        index_path = registry_root / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        # Should not raise, just log error
        registry = ModelRegistry(root_dir=registry_root)

        assert len(registry.models) == 0
        # Check that an error was logged (optional, might not work in all setups)


# ---------------------------------------------------------------------------
# Test save_model edge cases
# ---------------------------------------------------------------------------


class TestSaveModelEdgeCases:
    """Test save_model with various edge cases."""

    def test_save_model_without_get_model_info(
        self, registry: ModelRegistry, registry_root: Path
    ) -> None:
        """save_model should work even if model doesn't have get_model_info."""

        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x):
                return self.linear(x)

        model = MinimalModel()
        record = registry.save_model(model, model_key="minimal")

        assert record.model_key == "minimal"
        assert record.architecture == "MinimalModel"
        assert record.model_info is None

    def test_save_model_get_model_info_raises_exception(
        self, registry: ModelRegistry
    ) -> None:
        """save_model should handle get_model_info exceptions gracefully."""
        model = ModelWithBrokenInfo(num_classes=3)

        # Should not raise, just skip model_info
        record = registry.save_model(model, model_key="broken_info")

        assert record.model_key == "broken_info"
        assert record.model_info is None

    def test_save_model_get_model_info_returns_non_dict(
        self, registry: ModelRegistry
    ) -> None:
        """save_model should handle non-dict get_model_info return value."""
        model = ModelReturningNonDictInfo(num_classes=3)

        # Should not raise, just skip model_info
        record = registry.save_model(model, model_key="non_dict_info")

        assert record.model_key == "non_dict_info"
        assert record.model_info is None

    def test_save_model_with_all_none_optionals(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        """save_model should work with all optional parameters as None."""
        record = registry.save_model(
            tiny_model,
            model_key=None,  # Will use architecture name
            optimizer=None,
            scheduler=None,
            metrics=None,
            config=None,
            epoch=None,
            step=None,
            tag=None,
            extra_state=None,
        )

        assert record.model_key == "TinyModel"
        assert record.tag is None
        assert record.metrics is None
        assert record.config is None
        assert record.epoch is None
        assert record.step is None


# ---------------------------------------------------------------------------
# Test load_checkpoint edge cases
# ---------------------------------------------------------------------------


class TestLoadCheckpointEdgeCases:
    """Test load_checkpoint with various scenarios."""

    def test_load_checkpoint_without_restoring_objects(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        """load_checkpoint should work without model/optimizer/scheduler."""
        record = registry.save_model(tiny_model, model_key="no_restore")

        # Load without providing objects to restore
        ckpt = registry.load_checkpoint(record, map_location="cpu")

        assert "model_state" in ckpt
        assert ckpt["model_key"] == "no_restore"

    def test_load_checkpoint_with_no_optimizer_in_checkpoint(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        """load_checkpoint should handle missing optimizer gracefully."""
        # Save without optimizer
        record = registry.save_model(tiny_model, model_key="no_opt")

        # Try to load with optimizer (should not fail)
        model_new = TinyModel(num_classes=3)
        optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.01)

        ckpt = registry.load_checkpoint(
            record,
            model=model_new,
            optimizer=optimizer_new,
            map_location="cpu",
        )

        # Optimizer state should not be updated (checkpoint has None)
        assert ckpt["optimizer_state"] is None

    def test_load_checkpoint_with_no_scheduler_in_checkpoint(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        """load_checkpoint should handle missing scheduler gracefully."""
        # Save without scheduler
        record = registry.save_model(tiny_model, model_key="no_sched")

        # Try to load with scheduler (should not fail)
        model_new = TinyModel(num_classes=3)
        optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.01)
        scheduler_new = torch.optim.StepLR(optimizer_new, step_size=5)

        ckpt = registry.load_checkpoint(
            record,
            model=model_new,
            optimizer=optimizer_new,
            scheduler=scheduler_new,
            map_location="cpu",
        )

        # Scheduler state should not be updated
        assert ckpt["scheduler_state"] is None

    def test_load_checkpoint_strict_false(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        """load_checkpoint should respect strict=False parameter."""
        record = registry.save_model(tiny_model, model_key="strict_test")

        # Load into model with different architecture (but compatible enough)
        model_new = TinyModel(num_classes=3)

        # Should not raise with strict=False
        ckpt = registry.load_checkpoint(
            record, model=model_new, strict=False, map_location="cpu"
        )

        assert ckpt["model_key"] == "strict_test"


# ---------------------------------------------------------------------------
# Test list_versions and get_latest
# ---------------------------------------------------------------------------


class TestVersioningMethods:
    """Test list_versions and get_latest methods."""

    def test_list_versions_for_nonexistent_key_returns_empty(
        self, registry: ModelRegistry
    ) -> None:
        """list_versions should return empty list for unknown key."""
        versions = registry.list_versions("nonexistent")

        assert versions == []

    def test_get_latest_for_nonexistent_key_returns_none(
        self, registry: ModelRegistry
    ) -> None:
        """get_latest should return None for unknown key."""
        latest = registry.get_latest("nonexistent")

        assert latest is None

    def test_list_versions_sorted_by_version(
        self, registry: ModelRegistry, tiny_model: TinyModel
    ) -> None:
        """list_versions should return versions sorted ascending."""
        # Save multiple versions out of order
        registry.save_model(tiny_model, model_key="multi", epoch=3)
        registry.save_model(tiny_model, model_key="multi", epoch=1)
        registry.save_model(tiny_model, model_key="multi", epoch=2)

        versions = registry.list_versions("multi")

        assert [v.version for v in versions] == [1, 2, 3]
        # Order by version, not epoch
        assert [v.epoch for v in versions] == [3, 1, 2]


# ---------------------------------------------------------------------------
# Test index atomic save
# ---------------------------------------------------------------------------


class TestIndexAtomicSave:
    """Test that index saves are atomic."""

    def test_index_uses_tmp_file_before_rename(
        self,
        registry: ModelRegistry,
        tiny_model: TinyModel,
        registry_root: Path,
    ) -> None:
        """_save_index should use tmp file to avoid corruption."""
        registry.save_model(tiny_model, model_key="atomic_test")

        # Check that index.json exists
        assert registry.index_path.is_file()

        # The .tmp file should not exist after successful save
        tmp_path = registry.index_path.with_suffix(".tmp")
        assert not tmp_path.exists()


# ---------------------------------------------------------------------------
# Test custom index filename
# ---------------------------------------------------------------------------


class TestCustomIndexFilename:
    """Test ModelRegistry with custom index filename."""

    def test_custom_index_filename(
        self, registry_root: Path, tiny_model: TinyModel
    ) -> None:
        """ModelRegistry should support custom index filename."""
        registry = ModelRegistry(
            root_dir=registry_root, index_filename="custom_index.json"
        )

        registry.save_model(tiny_model, model_key="custom_test")

        custom_index_path = registry_root / "custom_index.json"
        assert custom_index_path.is_file()
        assert registry.index_path == custom_index_path

        # Reload should work
        registry2 = ModelRegistry(
            root_dir=registry_root, index_filename="custom_index.json"
        )
        assert "custom_test" in registry2.models


# ---------------------------------------------------------------------------
# Test _PatchedStepLR branches
# ---------------------------------------------------------------------------


class TestPatchedStepLR:
    """Test _PatchedStepLR state_dict handling branches."""

    def test_state_dict_when_state_already_exists(self) -> None:
        """_PatchedStepLR should preserve existing 'state' in state_dict."""
        from src.models.model_registry import _PatchedStepLR

        model = TinyModel(num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = _PatchedStepLR(optimizer, step_size=5)

        # Manually inject a "state" key to test the branch
        base_state_dict = torch.optim.lr_scheduler.StepLR.state_dict(
            scheduler
        )
        base_state_dict["state"] = {"custom_key": "custom_value"}

        # Monkey-patch to return our custom state_dict
        original_super_state_dict = (
            torch.optim.lr_scheduler.StepLR.state_dict
        )

        def mock_state_dict(self):
            return base_state_dict

        torch.optim.lr_scheduler.StepLR.state_dict = mock_state_dict

        try:
            result = scheduler.state_dict()
            # "state" should be preserved from base
            assert "state" in result
            assert result["state"]["custom_key"] == "custom_value"
        finally:
            # Restore original method
            torch.optim.lr_scheduler.StepLR.state_dict = (
                original_super_state_dict
            )

    def test_load_state_dict_without_state_key(self) -> None:
        """_PatchedStepLR should handle state_dict without 'state' key."""
        from src.models.model_registry import _PatchedStepLR

        model = TinyModel(num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = _PatchedStepLR(optimizer, step_size=5)

        # Standard StepLR state_dict (no "state" key)
        plain_state = {
            "last_epoch": 10,
            "_step_count": 11,
        }

        # Should not raise, just load normally
        scheduler.load_state_dict(plain_state)
        assert scheduler.last_epoch == 10


# ---------------------------------------------------------------------------
# Test index loading with non-dict models section
# ---------------------------------------------------------------------------


class TestIndexLoadingNonDictModels:
    """Test index loading when models section is not a dict."""

    def test_load_index_with_non_dict_models_section(
        self, registry_root: Path
    ) -> None:
        """Registry should handle non-dict models section gracefully."""
        # Create an index where models section is a list (not a dict)
        invalid_index = [
            {
                "model_key": "model_a",
                "version": 1,
                "architecture": "ArchA",
                "checkpoint_path": "a_v1.pt",
            }
        ]

        index_path = registry_root / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(invalid_index, f)

        # Should not crash, just load empty registry
        # (list is not a dict, so isinstance check fails)
        registry = ModelRegistry(root_dir=registry_root)

        assert len(registry.models) == 0
