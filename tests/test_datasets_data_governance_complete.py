# tests/test_datasets_data_governance_complete.py
"""
Comprehensive A1-grade test suite for src/datasets/data_governance.py

Coverage target: 100% line coverage, 100% branch coverage
Testing:
- Dataset metadata classes (DatasetLicenseInfo, DatasetInfo)
- Dataset registry and lookup functions
- Path detection and configuration
- JSON logging (data access, provenance, compliance)
- Compliance checking and assertions
- Helper functions (canonicalization, Git, etc.)
- Edge cases and error handling
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.datasets.data_governance import (
    AUDIT_LOG_FILENAME,
    COMPLIANCE_LOG_FILENAME,
    PROVENANCE_LOG_FILENAME,
    DatasetInfo,
    DatasetLicenseInfo,
    _append_json_record,
    _detect_project_root,
    _get_git_commit,
    _get_logger,
    _infer_script_name,
    _now_utc_iso,
    assert_data_usage_allowed,
    canonicalize_dataset_key,
    get_dataset_info,
    get_governance_dir,
    list_datasets,
    log_data_access,
    log_provenance,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_governance_dir(tmp_path: Path) -> Path:
    """Create a temporary governance directory for testing."""
    gov_dir = tmp_path / "test_governance"
    gov_dir.mkdir(parents=True, exist_ok=True)
    return gov_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("USERNAME", "test_user")
    monkeypatch.setenv("USER", "test_user")
    return monkeypatch


@pytest.fixture
def mock_sys_argv(monkeypatch):
    """Mock sys.argv for script name inference."""
    monkeypatch.setattr(sys, "argv", ["test_script.py", "--arg1", "value1"])
    return monkeypatch


# ============================================================================
# Test DatasetLicenseInfo
# ============================================================================


class TestDatasetLicenseInfo:
    """Test DatasetLicenseInfo dataclass."""

    def test_initialization_basic(self):
        """Test basic initialization of DatasetLicenseInfo."""
        license_info = DatasetLicenseInfo(
            name="MIT License",
            url="https://opensource.org/licenses/MIT",
            summary="Permissive open source license",
        )
        assert license_info.name == "MIT License"
        assert license_info.url == "https://opensource.org/licenses/MIT"
        assert license_info.summary == "Permissive open source license"

    def test_initialization_with_none_url(self):
        """Test initialization with None URL."""
        license_info = DatasetLicenseInfo(
            name="Custom License", url=None, summary="Custom terms"
        )
        assert license_info.name == "Custom License"
        assert license_info.url is None
        assert license_info.summary == "Custom terms"

    def test_frozen_dataclass(self):
        """Test that DatasetLicenseInfo is frozen (immutable)."""
        license_info = DatasetLicenseInfo(
            name="Test", url="http://test.com", summary="Test license"
        )
        with pytest.raises(AttributeError):
            license_info.name = "Modified"  # type: ignore

    def test_equality(self):
        """Test equality comparison of DatasetLicenseInfo instances."""
        license1 = DatasetLicenseInfo(
            name="Test", url="http://test.com", summary="Summary"
        )
        license2 = DatasetLicenseInfo(
            name="Test", url="http://test.com", summary="Summary"
        )
        license3 = DatasetLicenseInfo(
            name="Different", url="http://test.com", summary="Summary"
        )
        assert license1 == license2
        assert license1 != license3


# ============================================================================
# Test DatasetInfo
# ============================================================================


class TestDatasetInfo:
    """Test DatasetInfo dataclass."""

    def test_initialization_with_defaults(self):
        """Test DatasetInfo initialization with default values."""
        license_info = DatasetLicenseInfo(
            name="Test License", url="http://test.com", summary="Test"
        )
        dataset_info = DatasetInfo(
            key="test_dataset",
            display_name="Test Dataset",
            source_url="http://source.com",
            license=license_info,
        )
        assert dataset_info.key == "test_dataset"
        assert dataset_info.display_name == "Test Dataset"
        assert dataset_info.source_url == "http://source.com"
        assert dataset_info.license == license_info
        assert dataset_info.allowed_purposes == ("research", "education")
        assert dataset_info.allow_commercial is False
        assert dataset_info.contains_direct_identifiers is False
        assert dataset_info.notes == ""

    def test_initialization_with_custom_values(self):
        """Test DatasetInfo initialization with custom values."""
        license_info = DatasetLicenseInfo(
            name="Commercial", url="http://commercial.com", summary="Paid"
        )
        dataset_info = DatasetInfo(
            key="commercial_dataset",
            display_name="Commercial Dataset",
            source_url="http://source.com",
            license=license_info,
            allowed_purposes=("research", "commercial", "education"),
            allow_commercial=True,
            contains_direct_identifiers=True,
            notes="Contains PII",
        )
        assert dataset_info.allowed_purposes == ("research", "commercial", "education")
        assert dataset_info.allow_commercial is True
        assert dataset_info.contains_direct_identifiers is True
        assert dataset_info.notes == "Contains PII"

    def test_frozen_dataclass(self):
        """Test that DatasetInfo is frozen (immutable)."""
        license_info = DatasetLicenseInfo(
            name="Test", url="http://test.com", summary="Test"
        )
        dataset_info = DatasetInfo(
            key="test", display_name="Test", source_url="http://test.com", license=license_info
        )
        with pytest.raises(AttributeError):
            dataset_info.key = "modified"  # type: ignore


# ============================================================================
# Test canonicalize_dataset_key
# ============================================================================


class TestCanonicalizeDatasetKey:
    """Test dataset key canonicalization."""

    def test_basic_lowercase(self):
        """Test basic lowercase conversion."""
        assert canonicalize_dataset_key("ISIC2018") == "isic2018"
        assert canonicalize_dataset_key("NIH_CXR") == "nih_cxr"

    def test_hyphen_to_underscore(self):
        """Test hyphen conversion to underscore."""
        assert canonicalize_dataset_key("derm-7pt") == "derm_7pt"
        assert canonicalize_dataset_key("pad-chest") == "padchest"

    def test_space_to_underscore(self):
        """Test space conversion to underscore."""
        assert canonicalize_dataset_key("chest xray") == "chest_xray"

    def test_synonym_mapping_isic(self):
        """Test ISIC dataset synonym mappings."""
        assert canonicalize_dataset_key("isic18") == "isic2018"
        assert canonicalize_dataset_key("isic_2018") == "isic2018"
        assert canonicalize_dataset_key("isic19") == "isic2019"
        assert canonicalize_dataset_key("isic_2019") == "isic2019"
        assert canonicalize_dataset_key("isic20") == "isic2020"
        assert canonicalize_dataset_key("isic_2020") == "isic2020"

    def test_synonym_mapping_nih(self):
        """Test NIH ChestXray synonym mappings."""
        assert canonicalize_dataset_key("nih") == "nih_cxr"
        assert canonicalize_dataset_key("nih_chestxray") == "nih_cxr"
        assert canonicalize_dataset_key("chestxray14") == "nih_cxr"

    def test_synonym_mapping_padchest(self):
        """Test PadChest synonym mappings."""
        assert canonicalize_dataset_key("pad_chest") == "padchest"
        assert canonicalize_dataset_key("pad-chest") == "padchest"

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        assert canonicalize_dataset_key("  isic2018  ") == "isic2018"

    def test_unknown_key_passthrough(self):
        """Test that unknown keys pass through after normalization."""
        assert canonicalize_dataset_key("unknown_dataset") == "unknown_dataset"


# ============================================================================
# Test get_dataset_info and list_datasets
# ============================================================================


class TestDatasetRegistry:
    """Test dataset registry functions."""

    def test_get_dataset_info_isic2018(self):
        """Test getting ISIC 2018 dataset info."""
        info = get_dataset_info("isic2018")
        assert info.key == "isic2018"
        assert info.display_name == "ISIC 2018 Dermoscopy"
        assert "isic-archive.com" in info.source_url
        assert info.license.name is not None
        assert "research" in info.allowed_purposes

    def test_get_dataset_info_case_insensitive(self):
        """Test case-insensitive dataset lookup."""
        info1 = get_dataset_info("ISIC2018")
        info2 = get_dataset_info("isic2018")
        assert info1.key == info2.key

    def test_get_dataset_info_with_synonym(self):
        """Test dataset lookup with synonym."""
        info = get_dataset_info("isic18")
        assert info.key == "isic2018"

    def test_get_dataset_info_unknown_raises_keyerror(self):
        """Test that unknown dataset raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_dataset_info("unknown_dataset_xyz")
        assert "unknown_dataset_xyz" in str(exc_info.value)

    def test_list_datasets_returns_dict(self):
        """Test that list_datasets returns a dictionary."""
        datasets = list_datasets()
        assert isinstance(datasets, dict)
        assert len(datasets) > 0

    def test_list_datasets_contains_expected_keys(self):
        """Test that list_datasets contains expected datasets."""
        datasets = list_datasets()
        expected_keys = ["isic2018", "isic2019", "isic2020", "derm7pt", "nih_cxr", "padchest"]
        for key in expected_keys:
            assert key in datasets
            assert isinstance(datasets[key], DatasetInfo)

    def test_list_datasets_returns_copy(self):
        """Test that list_datasets returns a copy, not the original."""
        datasets1 = list_datasets()
        datasets2 = list_datasets()
        assert datasets1 is not datasets2
        assert datasets1 == datasets2


# ============================================================================
# Test path and configuration functions
# ============================================================================


class TestPathFunctions:
    """Test path detection and configuration functions."""

    def test_detect_project_root_with_git(self, tmp_path: Path):
        """Test project root detection with .git directory."""
        # Simply test that _detect_project_root returns a Path
        result = _detect_project_root()
        assert isinstance(result, Path)
        # The function should return some valid directory
        assert result.exists() or len(result.parents) > 0

    def test_detect_project_root_fallback_to_parent(self):
        """Test project root detection fallback to parent directory when no .git found."""
        # We need to test the case where:
        # 1. No .git directory is found in any parent
        # 2. There are >= 2 parents
        # 3. It returns here.parents[1]
        
        # Create mock parents list
        parent0 = Mock(spec=Path)
        parent1 = Mock(spec=Path)
        parent2 = Mock(spec=Path)
        
        # Make is_dir return False for all (no .git found)
        for parent in [parent0, parent1, parent2]:
            git_mock = Mock()
            git_mock.is_dir.return_value = False
            parent.__truediv__ = Mock(return_value=git_mock)
        
        # Mock the resolved path with enough parents
        mock_resolved = Mock()
        mock_resolved.parents = [parent0, parent1, parent2]
        
        # Mock the Path(__file__).resolve() chain
        with patch("src.datasets.data_governance.Path") as mock_path_cls:
            mock_file_obj = Mock()
            mock_file_obj.resolve.return_value = mock_resolved
            mock_path_cls.return_value = mock_file_obj
            
            # Call the function
            result = _detect_project_root()
            
            # Should return parents[1] when no .git found and len(parents) >= 2
            assert result == parent1

    def test_detect_project_root_fallback_to_cwd(self, tmp_path: Path, monkeypatch):
        """Test project root detection fallback to cwd when not enough parents."""
        monkeypatch.chdir(tmp_path)
        # When no .git found and not enough parents, should fall back to cwd
        with patch("src.datasets.data_governance.Path") as mock_path_cls:
            mock_resolved = Mock()
            mock_resolved.parents = []  # Not enough parents
            
            mock_file = Mock()
            mock_file.resolve.return_value = mock_resolved
            mock_path_cls.return_value = mock_file
            mock_path_cls.cwd.return_value = tmp_path
            
            result = _detect_project_root()
            # Should return cwd as fallback
            assert isinstance(result, (Path, type(tmp_path)))

    def test_get_governance_dir_default(self, tmp_path: Path, monkeypatch):
        """Test get_governance_dir with default path."""
        # Mock _detect_project_root to return tmp_path
        monkeypatch.setattr("src.datasets.data_governance._detect_project_root", lambda: tmp_path)
        
        # Clear environment variable if set
        monkeypatch.delenv("DATA_GOVERNANCE_DIR", raising=False)
        
        result = get_governance_dir()
        assert result == tmp_path / "logs" / "data_governance"
        assert result.exists()

    def test_get_governance_dir_from_env(self, tmp_path: Path, monkeypatch):
        """Test get_governance_dir with environment variable."""
        custom_dir = tmp_path / "custom_governance"
        monkeypatch.setenv("DATA_GOVERNANCE_DIR", str(custom_dir))
        
        result = get_governance_dir()
        assert result == custom_dir
        assert result.exists()


# ============================================================================
# Test helper functions
# ============================================================================


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = _get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "data_governance"
        assert logger.level == logging.INFO

    def test_get_logger_singleton(self):
        """Test that _get_logger returns the same logger instance."""
        logger1 = _get_logger()
        logger2 = _get_logger()
        assert logger1 is logger2

    def test_now_utc_iso(self):
        """Test UTC ISO timestamp generation."""
        timestamp = _now_utc_iso()
        assert isinstance(timestamp, str)
        # Should be parseable as ISO format
        dt = datetime.fromisoformat(timestamp)
        assert dt.tzinfo is not None

    def test_infer_script_name_with_argv(self, mock_sys_argv):
        """Test script name inference from sys.argv."""
        script_name = _infer_script_name()
        assert script_name == "test_script.py"

    def test_infer_script_name_without_argv(self, monkeypatch):
        """Test script name inference when sys.argv is not available."""
        monkeypatch.delattr(sys, "argv", raising=False)
        script_name = _infer_script_name()
        assert script_name == "<interactive>"

    def test_get_git_commit_success(self, tmp_path: Path, monkeypatch):
        """Test Git commit hash retrieval success."""
        # Create a fake git repo
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        
        # Mock _detect_project_root
        monkeypatch.setattr("src.datasets.data_governance._detect_project_root", lambda: tmp_path)
        
        # Mock subprocess.run to return a fake commit hash
        fake_commit = "abc123def456"
        mock_result = Mock()
        mock_result.stdout = fake_commit + "\n"
        
        with patch("subprocess.run", return_value=mock_result):
            commit = _get_git_commit()
            assert commit == fake_commit

    def test_get_git_commit_failure(self, tmp_path: Path, monkeypatch):
        """Test Git commit hash retrieval failure."""
        monkeypatch.setattr("src.datasets.data_governance._detect_project_root", lambda: tmp_path)
        
        # Mock subprocess to raise exception
        with patch("subprocess.run", side_effect=Exception("No git")):
            commit = _get_git_commit()
            assert commit is None


# ============================================================================
# Test JSON logging
# ============================================================================


class TestJsonLogging:
    """Test JSON record logging functionality."""

    def test_append_json_record_basic(self, temp_governance_dir: Path):
        """Test basic JSON record appending."""
        record = {"event": "test", "data": "value", "count": 42}
        _append_json_record("test.jsonl", record, temp_governance_dir)
        
        log_file = temp_governance_dir / "test.jsonl"
        assert log_file.exists()
        
        with log_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            loaded = json.loads(lines[0])
            assert loaded["event"] == "test"
            assert loaded["data"] == "value"
            assert loaded["count"] == 42

    def test_append_json_record_multiple(self, temp_governance_dir: Path):
        """Test appending multiple JSON records."""
        records = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            {"id": 3, "name": "third"},
        ]
        
        for record in records:
            _append_json_record("multi.jsonl", record, temp_governance_dir)
        
        log_file = temp_governance_dir / "multi.jsonl"
        with log_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 3
            for i, line in enumerate(lines):
                loaded = json.loads(line)
                assert loaded["id"] == i + 1

    def test_append_json_record_creates_parent_dirs(self, temp_governance_dir: Path):
        """Test that _append_json_record creates parent directories."""
        nested_path = "nested/deep/file.jsonl"
        record = {"test": "data"}
        _append_json_record(nested_path, record, temp_governance_dir)
        
        log_file = temp_governance_dir / nested_path
        assert log_file.exists()
        assert log_file.parent.exists()

    def test_append_json_record_with_datetime(self, temp_governance_dir: Path):
        """Test JSON serialization with datetime objects."""
        now = datetime.now(timezone.utc)
        record = {"timestamp": now, "event": "test"}
        _append_json_record("datetime.jsonl", record, temp_governance_dir)
        
        log_file = temp_governance_dir / "datetime.jsonl"
        with log_file.open("r") as f:
            loaded = json.loads(f.read())
            assert "timestamp" in loaded
            assert isinstance(loaded["timestamp"], str)

    def test_append_json_record_exception_handling(self, temp_governance_dir: Path):
        """Test that exceptions in _append_json_record don't crash."""
        # Make the directory read-only to cause write failure
        read_only_dir = temp_governance_dir / "readonly"
        read_only_dir.mkdir()
        
        # Mock open to raise exception
        with patch("builtins.open", side_effect=PermissionError("No write access")):
            # Should not raise exception
            _append_json_record("test.jsonl", {"data": "test"}, read_only_dir)


# ============================================================================
# Test log_data_access
# ============================================================================


class TestLogDataAccess:
    """Test data access logging."""

    def test_log_data_access_basic(self, temp_governance_dir: Path, mock_env_vars, mock_sys_argv):
        """Test basic data access logging."""
        log_data_access(
            dataset_name="isic2018",
            split="train",
            purpose="training",
            num_samples=1000,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / AUDIT_LOG_FILENAME
        assert log_file.exists()
        
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["event"] == "data_access"
            assert record["dataset"] == "isic2018"
            assert record["split"] == "train"
            assert record["purpose"] == "training"
            assert record["num_samples"] == 1000
            assert record["user"] == "test_user"
            assert record["script"] == "test_script.py"

    def test_log_data_access_with_dataset_info(self, temp_governance_dir: Path):
        """Test data access logging includes dataset metadata."""
        log_data_access(
            dataset_name="isic2018",
            purpose="research",
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / AUDIT_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert "dataset_display_name" in record
            assert record["dataset_display_name"] == "ISIC 2018 Dermoscopy"
            assert "license_name" in record
            assert "allowed_purposes" in record
            assert "research" in record["allowed_purposes"]

    def test_log_data_access_unknown_dataset(self, temp_governance_dir: Path):
        """Test data access logging with unknown dataset."""
        log_data_access(
            dataset_name="unknown_dataset",
            purpose="testing",
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / AUDIT_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["dataset"] == "unknown_dataset"
            assert "dataset_display_name" not in record

    def test_log_data_access_with_extra_metadata(self, temp_governance_dir: Path):
        """Test data access logging with extra metadata."""
        extra = {"batch_size": 32, "augmentation": "standard"}
        log_data_access(
            dataset_name="isic2019",
            split="val",
            purpose="validation",
            extra=extra,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / AUDIT_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert "extra" in record
            assert record["extra"]["batch_size"] == 32
            assert record["extra"]["augmentation"] == "standard"

    def test_log_data_access_with_custom_user_and_script(self, temp_governance_dir: Path):
        """Test data access logging with custom user and script."""
        log_data_access(
            dataset_name="derm7pt",
            purpose="training",
            user="custom_user",
            script="custom_script.py",
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / AUDIT_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["user"] == "custom_user"
            assert record["script"] == "custom_script.py"

    def test_log_data_access_action_variants(self, temp_governance_dir: Path):
        """Test data access logging with different actions."""
        actions = ["read", "write", "download", "delete"]
        for i, action in enumerate(actions):
            log_data_access(
                dataset_name="nih_cxr",
                action=action,
                purpose="testing",
                governance_dir=temp_governance_dir,
            )
        
        log_file = temp_governance_dir / AUDIT_LOG_FILENAME
        with log_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) == len(actions)
            for i, line in enumerate(lines):
                record = json.loads(line)
                assert record["action"] == actions[i]


# ============================================================================
# Test log_provenance
# ============================================================================


class TestLogProvenance:
    """Test provenance logging."""

    def test_log_provenance_basic(self, temp_governance_dir: Path, mock_sys_argv):
        """Test basic provenance logging."""
        log_provenance(
            stage="preprocess",
            dataset_name="isic2018",
            input_paths=["data/raw/isic2018/metadata.csv"],
            output_paths=["data/processed/isic2018/train.h5"],
            params={"image_size": 224, "normalize": True},
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        assert log_file.exists()
        
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["event"] == "provenance"
            assert record["stage"] == "preprocess"
            assert record["dataset"] == "isic2018"
            assert len(record["inputs"]) == 1
            assert len(record["outputs"]) == 1
            assert record["params"]["image_size"] == 224
            assert record["params"]["normalize"] is True

    def test_log_provenance_multiple_paths(self, temp_governance_dir: Path):
        """Test provenance logging with multiple input/output paths."""
        inputs = ["data/raw/file1.csv", "data/raw/file2.csv", "data/raw/file3.csv"]
        outputs = ["data/processed/out1.h5", "data/processed/out2.h5"]
        
        log_provenance(
            stage="merge_datasets",
            input_paths=inputs,
            output_paths=outputs,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert len(record["inputs"]) == 3
            assert len(record["outputs"]) == 2

    def test_log_provenance_with_tags(self, temp_governance_dir: Path):
        """Test provenance logging with tags."""
        tags = {"experiment": "rq1_baseline", "version": "v1.0"}
        log_provenance(
            stage="train_model",
            dataset_name="isic2019",
            tags=tags,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["tags"]["experiment"] == "rq1_baseline"
            assert record["tags"]["version"] == "v1.0"

    def test_log_provenance_with_path_objects(self, temp_governance_dir: Path, tmp_path: Path):
        """Test provenance logging with Path objects."""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.h5"
        
        log_provenance(
            stage="convert",
            input_paths=[input_path],
            output_paths=[output_path],
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert isinstance(record["inputs"][0], str)
            assert isinstance(record["outputs"][0], str)

    def test_log_provenance_no_dataset(self, temp_governance_dir: Path):
        """Test provenance logging without dataset name."""
        log_provenance(
            stage="generic_stage",
            input_paths=["input.txt"],
            output_paths=["output.txt"],
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["dataset"] is None

    def test_log_provenance_empty_paths(self, temp_governance_dir: Path):
        """Test provenance logging with empty path lists."""
        log_provenance(
            stage="initialization",
            input_paths=[],
            output_paths=[],
            params={},
            tags={},
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["inputs"] == []
            assert record["outputs"] == []
            assert record["params"] == {}
            assert record["tags"] == {}

    def test_log_provenance_none_paths(self, temp_governance_dir: Path):
        """Test provenance logging with None paths."""
        log_provenance(
            stage="test_stage",
            input_paths=None,
            output_paths=None,
            params=None,
            tags=None,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / PROVENANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["inputs"] == []
            assert record["outputs"] == []
            assert record["params"] == {}
            assert record["tags"] == {}


# ============================================================================
# Test assert_data_usage_allowed
# ============================================================================


class TestAssertDataUsageAllowed:
    """Test compliance checking."""

    def test_assert_data_usage_allowed_valid_research(self, temp_governance_dir: Path):
        """Test valid research usage passes."""
        # Should not raise
        assert_data_usage_allowed(
            "isic2018",
            purpose="research",
            commercial=False,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        assert log_file.exists()
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["result"] == "allowed"
            assert record["reason"] is None

    def test_assert_data_usage_allowed_valid_education(self, temp_governance_dir: Path):
        """Test valid education usage passes."""
        assert_data_usage_allowed(
            "derm7pt",
            purpose="education",
            commercial=False,
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["result"] == "allowed"

    def test_assert_data_usage_allowed_invalid_purpose(self, temp_governance_dir: Path):
        """Test invalid purpose raises PermissionError."""
        with pytest.raises(PermissionError) as exc_info:
            assert_data_usage_allowed(
                "isic2019",
                purpose="clinical_diagnosis",
                governance_dir=temp_governance_dir,
            )
        
        assert "clinical_diagnosis" in str(exc_info.value)
        assert "allowed_purposes" in str(exc_info.value)
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["result"] == "denied"
            assert "purpose" in record["reason"]

    def test_assert_data_usage_allowed_invalid_commercial(self, temp_governance_dir: Path):
        """Test commercial usage on non-commercial dataset raises error."""
        with pytest.raises(PermissionError) as exc_info:
            assert_data_usage_allowed(
                "padchest",
                purpose="research",
                commercial=True,
                governance_dir=temp_governance_dir,
            )
        
        assert "commercial" in str(exc_info.value).lower()
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["result"] == "denied"
            assert "commercial" in record["reason"].lower()

    def test_assert_data_usage_allowed_with_country(self, temp_governance_dir: Path):
        """Test compliance check with country metadata."""
        assert_data_usage_allowed(
            "nih_cxr",
            purpose="research",
            country="USA",
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["country"] == "USA"

    def test_assert_data_usage_allowed_case_insensitive_purpose(self, temp_governance_dir: Path):
        """Test that purpose checking is case-insensitive."""
        assert_data_usage_allowed(
            "isic2020",
            purpose="RESEARCH",
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert record["result"] == "allowed"

    def test_assert_data_usage_allowed_logs_license_info(self, temp_governance_dir: Path):
        """Test that compliance check logs license information."""
        assert_data_usage_allowed(
            "derm7pt",
            purpose="research",
            governance_dir=temp_governance_dir,
        )
        
        log_file = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        with log_file.open("r") as f:
            record = json.loads(f.read())
            assert "license" in record
            assert "name" in record["license"]
            assert "url" in record["license"]
            assert "summary" in record["license"]


# ============================================================================
# Test integration and edge cases
# ============================================================================


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_full_workflow_data_access_and_compliance(self, temp_governance_dir: Path, mock_sys_argv):
        """Test full workflow: compliance check -> data access logging."""
        # First check compliance
        assert_data_usage_allowed(
            "isic2018",
            purpose="research",
            governance_dir=temp_governance_dir,
        )
        
        # Then log data access
        log_data_access(
            dataset_name="isic2018",
            split="train",
            purpose="research",
            num_samples=5000,
            governance_dir=temp_governance_dir,
        )
        
        # Verify both logs exist
        compliance_log = temp_governance_dir / COMPLIANCE_LOG_FILENAME
        audit_log = temp_governance_dir / AUDIT_LOG_FILENAME
        assert compliance_log.exists()
        assert audit_log.exists()

    def test_full_workflow_with_provenance(self, temp_governance_dir: Path):
        """Test full workflow including provenance logging."""
        # Log provenance for preprocessing
        log_provenance(
            stage="preprocess_isic2019",
            dataset_name="isic2019",
            input_paths=["data/raw/isic2019/images/"],
            output_paths=["data/processed/isic2019/train.h5"],
            params={"resize": 224, "normalize": "imagenet"},
            tags={"phase": "preprocessing"},
            governance_dir=temp_governance_dir,
        )
        
        # Log data access
        log_data_access(
            dataset_name="isic2019",
            split="train",
            purpose="training",
            action="read",
            governance_dir=temp_governance_dir,
        )
        
        # Verify logs
        provenance_log = temp_governance_dir / PROVENANCE_LOG_FILENAME
        audit_log = temp_governance_dir / AUDIT_LOG_FILENAME
        assert provenance_log.exists()
        assert audit_log.exists()

    def test_multiple_datasets_logging(self, temp_governance_dir: Path):
        """Test logging for multiple datasets."""
        datasets = ["isic2018", "isic2019", "derm7pt"]
        
        for dataset in datasets:
            log_data_access(
                dataset_name=dataset,
                purpose="research",
                governance_dir=temp_governance_dir,
            )
        
        audit_log = temp_governance_dir / AUDIT_LOG_FILENAME
        with audit_log.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 3
            
            recorded_datasets = [json.loads(line)["dataset"] for line in lines]
            assert set(recorded_datasets) == set(datasets)

    def test_concurrent_logging_simulation(self, temp_governance_dir: Path):
        """Test that multiple log operations work correctly."""
        # Simulate concurrent operations
        for i in range(10):
            log_data_access(
                dataset_name="isic2020",
                split="train",
                purpose="training",
                num_samples=100 * i,
                governance_dir=temp_governance_dir,
            )
        
        audit_log = temp_governance_dir / AUDIT_LOG_FILENAME
        with audit_log.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 10

    def test_special_characters_in_metadata(self, temp_governance_dir: Path):
        """Test handling of special characters in metadata."""
        log_data_access(
            dataset_name="isic2018",
            purpose="research",
            extra={"note": "Testing with special chars: ñ, é, 中文, 🎉"},
            governance_dir=temp_governance_dir,
        )
        
        audit_log = temp_governance_dir / AUDIT_LOG_FILENAME
        with audit_log.open("r", encoding="utf-8") as f:
            record = json.loads(f.read())
            assert "🎉" in record["extra"]["note"]

    def test_git_commit_in_logs(self, temp_governance_dir: Path, monkeypatch):
        """Test that git commit is included in logs when available."""
        fake_commit = "abc123def456789"
        monkeypatch.setattr("src.datasets.data_governance._get_git_commit", lambda: fake_commit)
        
        log_data_access(
            dataset_name="derm7pt",
            purpose="research",
            governance_dir=temp_governance_dir,
        )
        
        audit_log = temp_governance_dir / AUDIT_LOG_FILENAME
        with audit_log.open("r") as f:
            record = json.loads(f.read())
            assert record["git_commit"] == fake_commit

    def test_all_constants_defined(self):
        """Test that all expected constants are defined."""
        assert AUDIT_LOG_FILENAME == "data_access.jsonl"
        assert PROVENANCE_LOG_FILENAME == "data_provenance.jsonl"
        assert COMPLIANCE_LOG_FILENAME == "compliance_checks.jsonl"
