"""
Tests for src.utils.config

These are written to properly exercise:
- YAML loading + error paths
- Deep merge
- Env var expansion
- Pydantic validation
- Config hash stability
- Saving resolved configs
- Edge cases for 100% coverage
"""

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ExperimentConfig,
    ExperimentMeta,
    _deep_merge,
    _expand_env_vars,
    _flatten_for_hash,
    _load_yaml_file,
    _normalize_paths_in_obj,
    get_config_hash,
    load_experiment_config,
    save_resolved_config,
)


def test_load_yaml_file_success_and_type_check(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("a: 1\nb: 2\n", encoding="utf-8")

    data = _load_yaml_file(cfg_path)
    assert data == {"a": 1, "b": 2}


def test_load_yaml_file_missing_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        _load_yaml_file(missing)


def test_load_yaml_file_non_mapping_raises(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _load_yaml_file(cfg_path)


def test_load_yaml_file_empty_file(tmp_path: Path):
    """Test loading an empty YAML file returns empty dict."""
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("", encoding="utf-8")

    data = _load_yaml_file(cfg_path)
    assert data == {}


def test_deep_merge_nested_dicts():
    base = {
        "a": 1,
        "nested": {"x": 1, "y": 2},
        "overwrite_me": 1,
    }
    new = {
        "nested": {"y": 3, "z": 4},
        "overwrite_me": 42,
        "new_key": "value",
    }

    merged = _deep_merge(base, new)
    assert merged["a"] == 1
    assert merged["nested"] == {"x": 1, "y": 3, "z": 4}
    assert merged["overwrite_me"] == 42
    assert merged["new_key"] == "value"


def test_expand_env_vars_and_user_expansion(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    obj = {
        "root": "${DATA_ROOT}/data",
        "home_path": "~/somewhere",
        "list": ["${DATA_ROOT}/x", 1, True],
        "tuple": ("${DATA_ROOT}/y", 2),
    }

    expanded = _expand_env_vars(obj)

    # Use as_posix() for cross-platform path comparison
    assert Path(expanded["root"]).as_posix() == Path(tmp_path / "data").as_posix()
    # "~" should expand to a real home dir
    assert str(Path(expanded["home_path"]).parent) == str(Path.home())
    assert Path(expanded["list"][0]).as_posix() == Path(tmp_path / "x").as_posix()
    assert Path(expanded["tuple"][0]).as_posix() == Path(tmp_path / "y").as_posix()


def test_expand_env_vars_recursively(monkeypatch) -> None:
    """Cross-platform check that nested lists/tuples get expanded."""
    monkeypatch.setenv("DATA_ROOT", "/data")
    obj = {
        "root": "${DATA_ROOT}/ds",
        "list": ["${DATA_ROOT}/a", "~"],
        "tuple": ("${DATA_ROOT}/b",),
    }

    expanded = _expand_env_vars(obj)

    # Use as_posix() for cross-platform comparison
    assert Path(expanded["root"]).as_posix() == Path("/data/ds").as_posix()
    assert Path(expanded["list"][0]).as_posix() == Path("/data/a").as_posix()
    assert Path(expanded["tuple"][0]).as_posix() == Path("/data/b").as_posix()
    # "~" should be expanded, not left literally as "~"
    assert expanded["list"][1] != "~"


def test_expand_env_vars_non_string_types():
    """Test that non-string types are passed through unchanged."""
    obj = {
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
    }

    expanded = _expand_env_vars(obj)
    assert expanded == obj


def test_normalize_paths_in_obj_with_backslashes():
    """Test path normalization with Windows-style backslashes."""
    obj = {
        "path": r"C:\Users\test\data",
        "nested": {"path": r"D:\data\folder"},
        "list": [r"E:\path\one", r"F:\path\two"],
        "non_path": "regular_string",
    }

    normalized = _normalize_paths_in_obj(obj)

    # All paths should be normalized to forward slashes
    assert "/" in normalized["path"]
    assert "\\" not in normalized["path"]
    assert "/" in normalized["nested"]["path"]
    assert normalized["non_path"] == "regular_string"


def test_normalize_paths_in_obj_with_tuple_paths():
    """Tuple branch in _normalize_paths_in_obj should also be normalised."""
    obj = (
        r"C:\Users\test\data",
        "/tmp/somewhere",
        "plain_string",
    )

    normalized = _normalize_paths_in_obj(obj)

    # Still a tuple
    assert isinstance(normalized, tuple)
    # Path-like entries use forward slashes, non-path strings are unchanged
    assert "/" in normalized[0] and "\\" not in normalized[0]
    assert "/" in normalized[1] and "\\" not in normalized[1]
    assert normalized[2] == "plain_string"


def test_normalize_paths_in_obj_non_path_strings():
    """Test that strings without path separators are unchanged."""
    obj = {
        "name": "experiment_name",
        "value": "some_value",
        "number": 42,
    }

    normalized = _normalize_paths_in_obj(obj)
    assert normalized == obj


def test_flatten_for_hash_excludes_yaml_stack():
    """Test that yaml_stack is excluded from hash computation."""
    obj = {
        "experiment": {"name": "test"},
        "yaml_stack": ["/path/one.yaml", "/path/two.yaml"],
        "other": {"value": 123},
    }

    flat = _flatten_for_hash(obj)

    # yaml_stack should not appear in flattened dict
    assert not any("yaml_stack" in key for key in flat.keys())
    # Other fields should be present
    assert "experiment.name" in flat
    assert "other.value" in flat


def test_flatten_for_hash_with_lists_and_tuples():
    """Test flattening with complex nested structures."""
    obj = {
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "nested": {"inner_list": ["a", "b"]},
    }

    flat = _flatten_for_hash(obj)

    assert "list[0]" in flat
    assert "list[1]" in flat
    assert "tuple[0]" in flat
    assert "nested.inner_list[0]" in flat


def test_flatten_for_hash_with_pydantic_model():
    """Test flattening with Pydantic models."""
    config = ExperimentMeta(
        name="test_exp",
        project_name="test_proj",
        tags={"env": "dev"},
    )

    flat = _flatten_for_hash(config)

    assert "name" in flat
    assert "project_name" in flat
    assert "tags.env" in flat


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_load_experiment_config_success_and_merge(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    base = {
        "experiment": {"name": "base-exp"},
        "dataset": {
            "name": "nih_cxr",
            "root": "${DATA_ROOT}/nih",
            "batch_size": 8,
        },
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {"seed": 123},
    }

    override = {
        "experiment": {"name": "override-exp", "tags": {"kind": "test"}},
        "dataset": {"batch_size": 16, "num_workers": 2},
    }

    base_path = tmp_path / "base.yaml"
    override_path = tmp_path / "override.yaml"
    _write_yaml(base_path, base)
    _write_yaml(override_path, override)

    cfg = load_experiment_config(base_path, override_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment.name == "override-exp"
    assert cfg.experiment.tags["kind"] == "test"
    assert cfg.dataset.name == "nih_cxr"

    # Use as_posix() for cross-platform path comparison
    assert Path(cfg.dataset.root).as_posix() == Path(tmp_path / "nih").as_posix()

    assert cfg.dataset.batch_size == 16
    assert cfg.dataset.num_workers == 2
    assert cfg.model.name == "resnet"
    assert cfg.training.max_epochs == 5
    assert cfg.training.device == "cpu"
    assert cfg.reproducibility.seed == 123
    assert cfg.yaml_stack == [str(base_path), str(override_path)]


def test_load_experiment_config_single_file(tmp_path: Path):
    """Test loading config from a single file."""
    config = {
        "experiment": {"name": "single-file-exp"},
        "dataset": {"name": "test", "root": "/tmp", "batch_size": 8},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {"seed": 42},
    }

    cfg_path = tmp_path / "single.yaml"
    _write_yaml(cfg_path, config)

    cfg = load_experiment_config(cfg_path)
    assert cfg.experiment.name == "single-file-exp"
    assert len(cfg.yaml_stack) == 1


def test_load_experiment_config_validation_error(tmp_path: Path):
    # Missing required 'experiment' and 'model' sections
    bad = {
        "dataset": {"name": "nih", "root": "/tmp", "batch_size": 8},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {"seed": 1},
    }
    bad_path = tmp_path / "bad.yaml"
    _write_yaml(bad_path, bad)

    with pytest.raises(ValueError) as excinfo:
        load_experiment_config(bad_path)

    msg = str(excinfo.value)
    assert "experiment" in msg
    assert "model" in msg


def test_load_experiment_config_invalid_device(tmp_path: Path):
    """Test validation error for invalid device string."""
    bad = {
        "experiment": {"name": "test"},
        "dataset": {"name": "test", "root": "/tmp", "batch_size": 8},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "invalid_device"},
        "reproducibility": {"seed": 1},
    }
    bad_path = tmp_path / "bad_device.yaml"
    _write_yaml(bad_path, bad)

    with pytest.raises(ValueError) as excinfo:
        load_experiment_config(bad_path)

    msg = str(excinfo.value)
    assert "device" in msg.lower()


def test_get_config_hash_stable_and_changing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    base = {
        "experiment": {"name": "exp"},
        "dataset": {"name": "nih", "root": "${DATA_ROOT}/nih", "batch_size": 8},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {"seed": 1},
    }
    p1 = tmp_path / "cfg1.yaml"
    p2 = tmp_path / "cfg2.yaml"
    _write_yaml(p1, base)
    _write_yaml(p2, base)

    cfg1 = load_experiment_config(p1)
    cfg2 = load_experiment_config(p2)

    h1 = get_config_hash(cfg1)
    h2 = get_config_hash(cfg2)
    assert h1 == h2, f"Hashes should be identical for same config: {h1} != {h2}"

    # Now change a single hyperparameter and hash should differ
    base["training"]["learning_rate"] = 1e-4
    p3 = tmp_path / "cfg3.yaml"
    _write_yaml(p3, base)
    cfg3 = load_experiment_config(p3)
    h3 = get_config_hash(cfg3)

    assert h3 != h1, f"Hash should differ after config change: {h3} == {h1}"


def test_get_config_hash_different_algorithms(tmp_path: Path):
    """Test hash computation with different algorithms."""
    config = {
        "experiment": {"name": "test"},
        "dataset": {"name": "test", "root": "/tmp", "batch_size": 8},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {"seed": 42},
    }

    cfg_path = tmp_path / "test.yaml"
    _write_yaml(cfg_path, config)
    cfg = load_experiment_config(cfg_path)

    hash_sha256 = get_config_hash(cfg, algo="sha256")
    hash_md5 = get_config_hash(cfg, algo="md5")

    assert hash_sha256 != hash_md5
    assert len(hash_sha256) == 64  # SHA256 produces 64 hex chars
    assert len(hash_md5) == 32  # MD5 produces 32 hex chars


def test_save_resolved_config_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    base = {
        "experiment": {"name": "exp", "project_name": "proj"},
        "dataset": {"name": "nih", "root": "${DATA_ROOT}/nih", "batch_size": 4},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 3, "device": "cpu"},
        "reproducibility": {"seed": 7},
    }
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, base)

    cfg = load_experiment_config(cfg_path)
    out_path = tmp_path / "resolved" / "resolved.yaml"

    save_resolved_config(cfg, out_path)
    assert out_path.is_file()

    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert loaded["experiment"]["name"] == "exp"
    assert loaded["dataset"]["batch_size"] == 4
    assert "config_hash" in loaded
    # Hash in the file should match recomputed hash
    assert loaded["config_hash"] == get_config_hash(cfg)


def test_save_resolved_config_creates_parent_dirs(tmp_path: Path):
    """Test that save_resolved_config creates parent directories."""
    config = {
        "experiment": {"name": "test"},
        "dataset": {"name": "test", "root": "/tmp", "batch_size": 8},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {"seed": 42},
    }

    cfg_path = tmp_path / "test.yaml"
    _write_yaml(cfg_path, config)
    cfg = load_experiment_config(cfg_path)

    # Save to deeply nested path
    out_path = tmp_path / "deeply" / "nested" / "path" / "config.yaml"
    save_resolved_config(cfg, out_path)

    assert out_path.is_file()
    assert out_path.parent.exists()


def test_pydantic_models_with_extra_fields(tmp_path: Path):
    """Test that extra fields in YAML are handled correctly."""
    config = {
        "experiment": {
            "name": "test",
            "custom_field": "custom_value",  # Extra field (allowed)
        },
        "dataset": {
            "name": "test",
            "root": "/tmp",
            "batch_size": 8,
            "custom_augmentation": True,  # Extra field (allowed)
        },
        "model": {
            "name": "resnet",
            "num_classes": 2,
            "dropout": 0.5,  # Extra field (allowed)
        },
        "training": {
            "max_epochs": 5,
            "device": "cpu",
            "warmup_epochs": 2,  # Extra field (allowed)
        },
        "reproducibility": {
            "seed": 42,
            # No extra fields allowed here (extra="forbid")
        },
    }

    cfg_path = tmp_path / "extra_fields.yaml"
    _write_yaml(cfg_path, config)

    cfg = load_experiment_config(cfg_path)

    # Check that extra fields are accessible
    assert cfg.experiment.model_dump()["custom_field"] == "custom_value"
    assert cfg.dataset.model_dump()["custom_augmentation"] is True
    assert cfg.model.model_dump()["dropout"] == 0.5
    assert cfg.training.model_dump()["warmup_epochs"] == 2


def test_reproducibility_config_forbids_extra_fields(tmp_path: Path):
    """Test that ReproducibilityConfig rejects extra fields."""
    config = {
        "experiment": {"name": "test"},
        "dataset": {"name": "test", "root": "/tmp", "batch_size": 8},
        "model": {"name": "resnet", "num_classes": 2},
        "training": {"max_epochs": 5, "device": "cpu"},
        "reproducibility": {
            "seed": 42,
            "invalid_field": "should_fail",  # This should be rejected
        },
    }

    cfg_path = tmp_path / "bad_repro.yaml"
    _write_yaml(cfg_path, config)

    with pytest.raises(ValueError) as excinfo:
        load_experiment_config(cfg_path)

    msg = str(excinfo.value)
    assert "reproducibility" in msg.lower()


def test_config_with_cuda_device_format(tmp_path: Path):
    """Test valid CUDA device formats."""
    valid_devices = ["cpu", "cuda", "cuda:0", "cuda:1"]

    for device in valid_devices:
        config = {
            "experiment": {"name": "test"},
            "dataset": {"name": "test", "root": "/tmp", "batch_size": 8},
            "model": {"name": "resnet", "num_classes": 2},
            "training": {"max_epochs": 5, "device": device},
            "reproducibility": {"seed": 42},
        }

        cfg_path = tmp_path / f"device_{device.replace(':', '_')}.yaml"
        _write_yaml(cfg_path, config)

        cfg = load_experiment_config(cfg_path)
        assert cfg.training.device == device
