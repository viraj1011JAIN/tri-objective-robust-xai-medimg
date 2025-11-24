# tests/test_data_ready_check_complete.py
"""
Comprehensive A1-grade test suite for src/data/ready_check.py

Coverage target: 100% line coverage, 100% branch coverage
Testing:
- DatasetReadyReport dataclass
- Logger setup with various verbosity levels
- run_ready_check with all scenarios (missing metadata, missing columns, missing files, etc.)
- save_report JSON serialization
- CLI argument parsing
- main() function with various CLI options
- Edge cases and error handling
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.ready_check import (
    REQUIRED_COLUMNS,
    DatasetReadyReport,
    _setup_logger,
    main,
    parse_args,
    run_ready_check,
    save_report,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset directory."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


@pytest.fixture
def valid_metadata_csv(temp_dataset_dir: Path) -> Path:
    """Create a valid metadata.csv with all required columns and image files."""
    metadata_path = temp_dataset_dir / "metadata.csv"
    
    # Create some fake image files
    images_dir = temp_dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    image_files = []
    for i in range(5):
        img_path = images_dir / f"image_{i}.jpg"
        img_path.write_bytes(b"fake_image_data")
        image_files.append(f"images/image_{i}.jpg")
    
    # Create metadata
    df = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(5)],
        "image_path": image_files,
        "label": ["melanoma", "nevus", "melanoma", "nevus", "melanoma"],
        "split": ["train", "train", "val", "val", "test"],
    })
    df.to_csv(metadata_path, index=False)
    
    return metadata_path


@pytest.fixture
def metadata_with_missing_files(temp_dataset_dir: Path) -> Path:
    """Create metadata.csv with references to non-existent files."""
    metadata_path = temp_dataset_dir / "metadata.csv"
    
    # Create only some image files
    images_dir = temp_dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Only create first 2 files, leave 3 missing
    for i in range(2):
        img_path = images_dir / f"image_{i}.jpg"
        img_path.write_bytes(b"fake_image_data")
    
    # Metadata references 5 files but only 2 exist
    df = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(5)],
        "image_path": [f"images/image_{i}.jpg" for i in range(5)],
        "label": ["melanoma"] * 5,
        "split": ["train"] * 5,
    })
    df.to_csv(metadata_path, index=False)
    
    return metadata_path


@pytest.fixture
def metadata_missing_columns(temp_dataset_dir: Path) -> Path:
    """Create metadata.csv missing required columns."""
    metadata_path = temp_dataset_dir / "metadata.csv"
    
    # Missing 'label' and 'split' columns
    df = pd.DataFrame({
        "image_id": ["img_0", "img_1"],
        "image_path": ["images/image_0.jpg", "images/image_1.jpg"],
    })
    df.to_csv(metadata_path, index=False)
    
    return metadata_path


@pytest.fixture
def metadata_no_train_split(temp_dataset_dir: Path) -> Path:
    """Create metadata.csv without a train split."""
    metadata_path = temp_dataset_dir / "metadata.csv"
    
    images_dir = temp_dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        img_path = images_dir / f"image_{i}.jpg"
        img_path.write_bytes(b"fake_image_data")
    
    # Only val and test splits, no train
    df = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(3)],
        "image_path": [f"images/image_{i}.jpg" for i in range(3)],
        "label": ["melanoma"] * 3,
        "split": ["val", "val", "test"],
    })
    df.to_csv(metadata_path, index=False)
    
    return metadata_path


@pytest.fixture
def empty_metadata(temp_dataset_dir: Path) -> Path:
    """Create an empty metadata.csv (header only, no data rows)."""
    metadata_path = temp_dataset_dir / "metadata.csv"
    
    df = pd.DataFrame({
        "image_id": [],
        "image_path": [],
        "label": [],
        "split": [],
    })
    df.to_csv(metadata_path, index=False)
    
    return metadata_path


# ============================================================================
# Test DatasetReadyReport
# ============================================================================


class TestDatasetReadyReport:
    """Test DatasetReadyReport dataclass."""

    def test_initialization(self):
        """Test basic initialization of DatasetReadyReport."""
        report = DatasetReadyReport(
            dataset="test_dataset",
            root="/path/to/data",
            metadata_path="/path/to/data/metadata.csv",
            exists_metadata=True,
            required_columns=["col1", "col2"],
            missing_columns=[],
            num_rows=100,
            num_missing_files=0,
            num_bad_rows=0,
            unique_labels=["class_a", "class_b"],
            splits=["train", "val", "test"],
            status="ok",
            warnings=[],
            errors=[],
        )
        
        assert report.dataset == "test_dataset"
        assert report.status == "ok"
        assert report.num_rows == 100
        assert len(report.unique_labels) == 2
        assert len(report.splits) == 3

    def test_report_with_errors(self):
        """Test DatasetReadyReport with error status."""
        report = DatasetReadyReport(
            dataset="bad_dataset",
            root="/path/to/data",
            metadata_path="/path/to/data/metadata.csv",
            exists_metadata=False,
            required_columns=REQUIRED_COLUMNS,
            missing_columns=list(REQUIRED_COLUMNS),
            num_rows=0,
            num_missing_files=0,
            num_bad_rows=0,
            unique_labels=[],
            splits=[],
            status="error",
            warnings=[],
            errors=["metadata file not found"],
        )
        
        assert report.status == "error"
        assert len(report.errors) == 1
        assert not report.exists_metadata

    def test_report_with_warnings(self):
        """Test DatasetReadyReport with warning status."""
        report = DatasetReadyReport(
            dataset="warning_dataset",
            root="/path/to/data",
            metadata_path="/path/to/data/metadata.csv",
            exists_metadata=True,
            required_columns=REQUIRED_COLUMNS,
            missing_columns=[],
            num_rows=50,
            num_missing_files=5,
            num_bad_rows=5,
            unique_labels=["class_a"],
            splits=["train"],
            status="warning",
            warnings=["5 image files referenced in metadata do not exist under root."],
            errors=[],
        )
        
        assert report.status == "warning"
        assert report.num_missing_files == 5
        assert len(report.warnings) == 1


# ============================================================================
# Test _setup_logger
# ============================================================================


class TestSetupLogger:
    """Test logger setup function."""

    def test_setup_logger_default_verbosity(self):
        """Test logger setup with default verbosity (INFO level)."""
        logger = _setup_logger(verbosity=1)
        assert logger.name == "data_ready"
        assert logger.level == logging.INFO

    def test_setup_logger_quiet(self):
        """Test logger setup with verbosity 0 (WARNING level)."""
        logger = _setup_logger(verbosity=0)
        assert logger.level == logging.WARNING

    def test_setup_logger_verbose(self):
        """Test logger setup with verbosity > 1 (DEBUG level)."""
        logger = _setup_logger(verbosity=2)
        assert logger.level == logging.DEBUG

    def test_setup_logger_idempotent(self):
        """Test that calling _setup_logger multiple times doesn't add multiple handlers."""
        logger1 = _setup_logger()
        initial_handler_count = len(logger1.handlers)
        
        logger2 = _setup_logger()
        assert logger1 is logger2
        assert len(logger2.handlers) == initial_handler_count


# ============================================================================
# Test run_ready_check - Success scenarios
# ============================================================================


class TestRunReadyCheckSuccess:
    """Test run_ready_check with successful scenarios."""

    def test_run_ready_check_valid_dataset(self, temp_dataset_dir: Path, valid_metadata_csv: Path):
        """Test run_ready_check with a completely valid dataset."""
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.dataset == "test_dataset"
        assert report.status == "ok"
        assert report.exists_metadata is True
        assert report.num_rows == 5
        assert report.num_missing_files == 0
        assert report.num_bad_rows == 0
        assert len(report.unique_labels) == 2
        assert "melanoma" in report.unique_labels
        assert "nevus" in report.unique_labels
        assert len(report.splits) == 3
        assert "train" in report.splits
        assert len(report.errors) == 0
        assert len(report.missing_columns) == 0

    def test_run_ready_check_with_custom_logger(self, temp_dataset_dir: Path, valid_metadata_csv: Path):
        """Test run_ready_check with a custom logger."""
        custom_logger = logging.getLogger("custom_test_logger")
        custom_logger.setLevel(logging.DEBUG)
        
        report = run_ready_check(
            "test_dataset",
            temp_dataset_dir,
            logger=custom_logger
        )
        
        assert report.status == "ok"

    def test_run_ready_check_with_custom_metadata_name(self, temp_dataset_dir: Path):
        """Test run_ready_check with custom metadata filename."""
        # Create metadata with custom name
        custom_metadata = temp_dataset_dir / "custom_meta.csv"
        images_dir = temp_dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        img_path = images_dir / "image_0.jpg"
        img_path.write_bytes(b"fake_image_data")
        
        df = pd.DataFrame({
            "image_id": ["img_0"],
            "image_path": ["images/image_0.jpg"],
            "label": ["melanoma"],
            "split": ["train"],
        })
        df.to_csv(custom_metadata, index=False)
        
        report = run_ready_check(
            "test_dataset",
            temp_dataset_dir,
            metadata_name="custom_meta.csv"
        )
        
        assert report.status == "ok"
        assert "custom_meta.csv" in report.metadata_path


# ============================================================================
# Test run_ready_check - Error scenarios
# ============================================================================


class TestRunReadyCheckErrors:
    """Test run_ready_check with error scenarios."""

    def test_run_ready_check_missing_metadata_file(self, temp_dataset_dir: Path):
        """Test run_ready_check when metadata.csv doesn't exist."""
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "error"
        assert report.exists_metadata is False
        assert len(report.errors) == 1
        assert "metadata file not found" in report.errors[0]
        assert report.num_rows == 0

    def test_run_ready_check_missing_required_columns(
        self, temp_dataset_dir: Path, metadata_missing_columns: Path
    ):
        """Test run_ready_check when required columns are missing."""
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "error"
        assert len(report.missing_columns) == 2
        assert "label" in report.missing_columns
        assert "split" in report.missing_columns
        assert len(report.errors) == 1
        assert "missing required columns" in report.errors[0]

    def test_run_ready_check_corrupted_csv(self, temp_dataset_dir: Path):
        """Test run_ready_check when CSV is corrupted/unreadable."""
        # Create a corrupted CSV
        metadata_path = temp_dataset_dir / "metadata.csv"
        metadata_path.write_bytes(b"\xff\xfe\x00\x00corrupted binary data")
        
        # Mock pd.read_csv to raise an exception
        with patch("src.data.ready_check.pd.read_csv", side_effect=Exception("CSV parse error")):
            report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "error"
        assert len(report.errors) == 1
        assert "failed to read metadata CSV" in report.errors[0]
        assert report.num_rows == 0


# ============================================================================
# Test run_ready_check - Warning scenarios
# ============================================================================


class TestRunReadyCheckWarnings:
    """Test run_ready_check with warning scenarios."""

    def test_run_ready_check_missing_image_files(
        self, temp_dataset_dir: Path, metadata_with_missing_files: Path
    ):
        """Test run_ready_check when some image files don't exist."""
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "warning"
        assert report.num_missing_files == 3
        assert report.num_bad_rows == 3
        assert len(report.warnings) == 1
        assert "3 image files" in report.warnings[0]
        assert "do not exist" in report.warnings[0]

    def test_run_ready_check_no_train_split(
        self, temp_dataset_dir: Path, metadata_no_train_split: Path
    ):
        """Test run_ready_check when 'train' split is missing."""
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "warning"
        assert len(report.warnings) == 1
        assert "no 'train' split detected" in report.warnings[0]
        assert "train" not in [s.lower() for s in report.splits]

    def test_run_ready_check_train_split_case_insensitive(self, temp_dataset_dir: Path):
        """Test that train split detection is case-insensitive."""
        metadata_path = temp_dataset_dir / "metadata.csv"
        images_dir = temp_dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        img_path = images_dir / "image_0.jpg"
        img_path.write_bytes(b"fake_image_data")
        
        # Use TRAIN instead of train
        df = pd.DataFrame({
            "image_id": ["img_0"],
            "image_path": ["images/image_0.jpg"],
            "label": ["melanoma"],
            "split": ["TRAIN"],
        })
        df.to_csv(metadata_path, index=False)
        
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        # Should NOT have warning about missing train split
        assert report.status == "ok"
        assert not any("train" in w.lower() for w in report.warnings)


# ============================================================================
# Test run_ready_check - Edge cases
# ============================================================================


class TestRunReadyCheckEdgeCases:
    """Test run_ready_check with edge cases."""

    def test_run_ready_check_empty_metadata(self, temp_dataset_dir: Path, empty_metadata: Path):
        """Test run_ready_check with empty metadata (no data rows)."""
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        # Empty metadata should not trigger the deeper checks
        assert report.num_rows == 0
        assert report.unique_labels == []
        assert report.splits == []
        # Should have OK status since columns exist, just no data
        assert report.status == "ok"

    def test_run_ready_check_absolute_image_paths(self, temp_dataset_dir: Path):
        """Test run_ready_check with absolute image paths."""
        metadata_path = temp_dataset_dir / "metadata.csv"
        
        # Create image file
        img_path = temp_dataset_dir / "image_0.jpg"
        img_path.write_bytes(b"fake_image_data")
        
        # Use absolute path in metadata
        df = pd.DataFrame({
            "image_id": ["img_0"],
            "image_path": [str(img_path)],  # Absolute path
            "label": ["melanoma"],
            "split": ["train"],
        })
        df.to_csv(metadata_path, index=False)
        
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "ok"
        assert report.num_missing_files == 0

    def test_run_ready_check_relative_paths_resolution(self, temp_dataset_dir: Path):
        """Test that relative paths are correctly resolved from root."""
        metadata_path = temp_dataset_dir / "metadata.csv"
        
        # Create nested directory structure
        nested_dir = temp_dataset_dir / "data" / "images"
        nested_dir.mkdir(parents=True)
        
        img_path = nested_dir / "image_0.jpg"
        img_path.write_bytes(b"fake_image_data")
        
        # Use relative path
        df = pd.DataFrame({
            "image_id": ["img_0"],
            "image_path": ["data/images/image_0.jpg"],
            "label": ["melanoma"],
            "split": ["train"],
        })
        df.to_csv(metadata_path, index=False)
        
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "ok"
        assert report.num_missing_files == 0

    def test_run_ready_check_expanduser_tilde(self, temp_dataset_dir: Path, monkeypatch):
        """Test that paths with ~ are expanded correctly."""
        # Mock expanduser to return our temp dir
        original_expanduser = Path.expanduser
        
        def mock_expanduser(self):
            if str(self).startswith("~"):
                return temp_dataset_dir
            return original_expanduser(self)
        
        with patch.object(Path, "expanduser", mock_expanduser):
            # Create valid metadata
            metadata_path = temp_dataset_dir / "metadata.csv"
            images_dir = temp_dataset_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            img_path = images_dir / "image_0.jpg"
            img_path.write_bytes(b"fake_image_data")
            
            df = pd.DataFrame({
                "image_id": ["img_0"],
                "image_path": ["images/image_0.jpg"],
                "label": ["melanoma"],
                "split": ["train"],
            })
            df.to_csv(metadata_path, index=False)
            
            report = run_ready_check("test_dataset", Path("~/test_data"))
            
            assert report.status == "ok"

    def test_run_ready_check_multiple_labels(self, temp_dataset_dir: Path):
        """Test run_ready_check with many unique labels."""
        metadata_path = temp_dataset_dir / "metadata.csv"
        images_dir = temp_dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        labels = [f"class_{i}" for i in range(10)]
        image_paths = []
        
        for i in range(10):
            img_path = images_dir / f"image_{i}.jpg"
            img_path.write_bytes(b"fake_image_data")
            image_paths.append(f"images/image_{i}.jpg")
        
        df = pd.DataFrame({
            "image_id": [f"img_{i}" for i in range(10)],
            "image_path": image_paths,
            "label": labels,
            "split": ["train"] * 10,
        })
        df.to_csv(metadata_path, index=False)
        
        report = run_ready_check("test_dataset", temp_dataset_dir)
        
        assert report.status == "ok"
        assert len(report.unique_labels) == 10
        assert report.unique_labels == sorted(labels)


# ============================================================================
# Test save_report
# ============================================================================


class TestSaveReport:
    """Test save_report function."""

    def test_save_report_basic(self, tmp_path: Path):
        """Test basic save_report functionality."""
        report = DatasetReadyReport(
            dataset="test_dataset",
            root="/path/to/data",
            metadata_path="/path/to/data/metadata.csv",
            exists_metadata=True,
            required_columns=REQUIRED_COLUMNS,
            missing_columns=[],
            num_rows=100,
            num_missing_files=0,
            num_bad_rows=0,
            unique_labels=["class_a", "class_b"],
            splits=["train", "val", "test"],
            status="ok",
            warnings=[],
            errors=[],
        )
        
        output_path = tmp_path / "report.json"
        save_report(report, output_path)
        
        assert output_path.exists()
        
        with output_path.open("r") as f:
            loaded_data = json.load(f)
        
        assert loaded_data["dataset"] == "test_dataset"
        assert loaded_data["status"] == "ok"
        assert loaded_data["num_rows"] == 100

    def test_save_report_creates_parent_dirs(self, tmp_path: Path):
        """Test that save_report creates parent directories."""
        report = DatasetReadyReport(
            dataset="test",
            root="/data",
            metadata_path="/data/metadata.csv",
            exists_metadata=True,
            required_columns=REQUIRED_COLUMNS,
            missing_columns=[],
            num_rows=0,
            num_missing_files=0,
            num_bad_rows=0,
            unique_labels=[],
            splits=[],
            status="ok",
            warnings=[],
            errors=[],
        )
        
        output_path = tmp_path / "nested" / "deep" / "report.json"
        save_report(report, output_path)
        
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_report_with_warnings_and_errors(self, tmp_path: Path):
        """Test save_report with warnings and errors."""
        report = DatasetReadyReport(
            dataset="problem_dataset",
            root="/data",
            metadata_path="/data/metadata.csv",
            exists_metadata=True,
            required_columns=REQUIRED_COLUMNS,
            missing_columns=["label"],
            num_rows=50,
            num_missing_files=10,
            num_bad_rows=10,
            unique_labels=[],
            splits=[],
            status="error",
            warnings=["warning 1", "warning 2"],
            errors=["error 1", "error 2"],
        )
        
        output_path = tmp_path / "report_with_issues.json"
        save_report(report, output_path)
        
        with output_path.open("r") as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data["warnings"]) == 2
        assert len(loaded_data["errors"]) == 2
        assert loaded_data["status"] == "error"


# ============================================================================
# Test parse_args
# ============================================================================


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_parse_args_minimal(self):
        """Test parse_args with minimal required arguments."""
        test_args = ["--dataset", "isic2020", "--root", "/path/to/data"]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            args = parse_args()
        
        assert args.dataset == "isic2020"
        assert args.root == "/path/to/data"
        assert args.metadata_name == "metadata.csv"
        assert args.output_json is None
        assert args.fail_on_error is False
        assert args.verbose == 1

    def test_parse_args_all_options(self):
        """Test parse_args with all optional arguments."""
        test_args = [
            "--dataset", "derm7pt",
            "--root", "/data/derm",
            "--metadata-name", "custom_meta.csv",
            "--output-json", "results/report.json",
            "--fail-on-error",
            "-vv"
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            args = parse_args()
        
        assert args.dataset == "derm7pt"
        assert args.root == "/data/derm"
        assert args.metadata_name == "custom_meta.csv"
        assert args.output_json == "results/report.json"
        assert args.fail_on_error is True
        assert args.verbose == 3  # -vv means 2 additional verbosity levels

    def test_parse_args_single_verbose(self):
        """Test parse_args with single -v flag."""
        test_args = ["--dataset", "test", "--root", "/data", "-v"]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            args = parse_args()
        
        assert args.verbose == 2  # Default 1 + 1


# ============================================================================
# Test main
# ============================================================================


class TestMain:
    """Test main CLI function."""

    def test_main_success_no_output(self, temp_dataset_dir: Path, valid_metadata_csv: Path):
        """Test main function with successful check and no JSON output."""
        test_args = [
            "--dataset", "test_dataset",
            "--root", str(temp_dataset_dir),
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            # Should not raise
            main()

    def test_main_with_json_output(self, temp_dataset_dir: Path, valid_metadata_csv: Path, tmp_path: Path):
        """Test main function with JSON output."""
        output_json = tmp_path / "output_report.json"
        
        test_args = [
            "--dataset", "test_dataset",
            "--root", str(temp_dataset_dir),
            "--output-json", str(output_json),
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            main()
        
        assert output_json.exists()
        
        with output_json.open("r") as f:
            data = json.load(f)
        
        assert data["dataset"] == "test_dataset"
        assert data["status"] == "ok"

    def test_main_fail_on_error_with_error_status(self, temp_dataset_dir: Path):
        """Test main function with --fail-on-error flag and error status."""
        test_args = [
            "--dataset", "bad_dataset",
            "--root", str(temp_dataset_dir),
            "--fail-on-error",
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
        
        assert exc_info.value.code == 1

    def test_main_fail_on_error_with_warning_status(
        self, temp_dataset_dir: Path, metadata_with_missing_files: Path
    ):
        """Test main function with --fail-on-error and warning status (should not fail)."""
        test_args = [
            "--dataset", "warning_dataset",
            "--root", str(temp_dataset_dir),
            "--fail-on-error",
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            # Should not raise because status is 'warning', not 'error'
            main()

    def test_main_with_custom_metadata_name(self, temp_dataset_dir: Path):
        """Test main function with custom metadata filename."""
        # Create custom metadata
        custom_meta = temp_dataset_dir / "custom.csv"
        images_dir = temp_dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        img_path = images_dir / "image_0.jpg"
        img_path.write_bytes(b"fake_image_data")
        
        df = pd.DataFrame({
            "image_id": ["img_0"],
            "image_path": ["images/image_0.jpg"],
            "label": ["melanoma"],
            "split": ["train"],
        })
        df.to_csv(custom_meta, index=False)
        
        test_args = [
            "--dataset", "custom_dataset",
            "--root", str(temp_dataset_dir),
            "--metadata-name", "custom.csv",
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            main()

    def test_main_verbose_logging(self, temp_dataset_dir: Path, valid_metadata_csv: Path):
        """Test main function with increased verbosity."""
        test_args = [
            "--dataset", "test_dataset",
            "--root", str(temp_dataset_dir),
            "-vvv",  # Very verbose
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            main()


# ============================================================================
# Test REQUIRED_COLUMNS constant
# ============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_required_columns_constant(self):
        """Test that REQUIRED_COLUMNS is defined correctly."""
        assert isinstance(REQUIRED_COLUMNS, list)
        assert len(REQUIRED_COLUMNS) == 4
        assert "image_id" in REQUIRED_COLUMNS
        assert "image_path" in REQUIRED_COLUMNS
        assert "label" in REQUIRED_COLUMNS
        assert "split" in REQUIRED_COLUMNS


# ============================================================================
# Test integration scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_full_workflow_valid_dataset(self, temp_dataset_dir: Path, valid_metadata_csv: Path, tmp_path: Path):
        """Test complete workflow: check dataset -> save report -> verify JSON."""
        # Run check
        report = run_ready_check("integration_test", temp_dataset_dir)
        
        # Verify in-memory report
        assert report.status == "ok"
        assert report.num_rows == 5
        
        # Save to JSON
        output_path = tmp_path / "integration_report.json"
        save_report(report, output_path)
        
        # Load and verify JSON
        with output_path.open("r") as f:
            data = json.load(f)
        
        assert data["dataset"] == "integration_test"
        assert data["status"] == "ok"
        assert data["num_rows"] == 5
        assert len(data["unique_labels"]) == 2

    def test_full_workflow_dataset_with_issues(
        self, temp_dataset_dir: Path, metadata_with_missing_files: Path, tmp_path: Path
    ):
        """Test complete workflow with a dataset that has issues."""
        report = run_ready_check("problematic_dataset", temp_dataset_dir)
        
        assert report.status == "warning"
        assert report.num_missing_files > 0
        
        output_path = tmp_path / "problem_report.json"
        save_report(report, output_path)
        
        with output_path.open("r") as f:
            data = json.load(f)
        
        assert data["status"] == "warning"
        assert len(data["warnings"]) > 0

    def test_end_to_end_cli_workflow(
        self, temp_dataset_dir: Path, valid_metadata_csv: Path, tmp_path: Path
    ):
        """Test end-to-end CLI workflow."""
        output_json = tmp_path / "cli_report.json"
        
        test_args = [
            "--dataset", "cli_test",
            "--root", str(temp_dataset_dir),
            "--output-json", str(output_json),
            "-v",
        ]
        
        with patch("sys.argv", ["ready_check.py"] + test_args):
            main()
        
        assert output_json.exists()
        
        with output_json.open("r") as f:
            data = json.load(f)
        
        assert data["dataset"] == "cli_test"
        assert data["status"] == "ok"
