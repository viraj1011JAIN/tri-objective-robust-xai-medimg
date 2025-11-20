# tests/test_data_ready.py
"""
Unit tests for src/data/ready_check.py

Covers:
- Successful ready check on a tiny synthetic dataset
- Missing metadata.csv -> status=error
- Missing image files -> status=warning
- CLI main() behaviour with --fail-on-error for both error and ok paths
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest
from PIL import Image

rc = pytest.importorskip("src.data.ready_check")


def _make_dummy_image(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path, format="JPEG")


def _write_metadata(root: Path, rows: List[dict]) -> Path:
    csv_path = root / "metadata.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_ready_check_success(tmp_path: Path) -> None:
    root = tmp_path / "dataset_ok"
    root.mkdir(parents=True, exist_ok=True)

    img_dir = root / "images"
    img1 = img_dir / "img1.jpg"
    img2 = img_dir / "img2.jpg"
    _make_dummy_image(img1)
    _make_dummy_image(img2)

    rows = [
        {
            "image_id": "1",
            "image_path": "images/img1.jpg",
            "label": "A",
            "split": "train",
        },
        {
            "image_id": "2",
            "image_path": "images/img2.jpg",
            "label": "B",
            "split": "val",
        },
    ]
    _write_metadata(root, rows)

    report = rc.run_ready_check(dataset="dummy", root=root)

    assert report.status == "ok"
    assert report.exists_metadata is True
    assert report.num_rows == 2
    assert report.num_missing_files == 0
    assert report.missing_columns == []


def test_ready_check_missing_metadata(tmp_path: Path) -> None:
    root = tmp_path / "dataset_missing_meta"
    root.mkdir(parents=True, exist_ok=True)

    report = rc.run_ready_check(dataset="dummy", root=root)

    assert report.status == "error"
    assert report.exists_metadata is False
    assert any("metadata" in e.lower() for e in report.errors)


def test_ready_check_missing_files(tmp_path: Path) -> None:
    """
    When files referenced in metadata are missing, the current implementation
    reports status='warning' (not 'error') and sets num_missing_files > 0.
    """
    root = tmp_path / "dataset_missing_files"
    root.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "image_id": "1",
            "image_path": "images/missing1.jpg",
            "label": "A",
            "split": "train",
        },
        {
            "image_id": "2",
            "image_path": "images/missing2.jpg",
            "label": "B",
            "split": "val",
        },
    ]
    _write_metadata(root, rows)

    report = rc.run_ready_check(dataset="dummy", root=root)

    assert report.num_missing_files == 2
    assert report.status == "warning"


def test_ready_check_main_fail_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    main() should exit with code 1 when status == error and --fail-on-error is True.
    Here we trigger an error by omitting metadata.csv entirely.
    """
    root = tmp_path / "dataset_cli_fail"
    root.mkdir(parents=True, exist_ok=True)
    output_json = tmp_path / "report.json"

    def _fake_parse_args():
        return SimpleNamespace(
            dataset="dummy_cli",
            root=str(root),
            metadata_name="metadata.csv",
            output_json=str(output_json),
            fail_on_error=True,
            verbose=0,
        )

    monkeypatch.setattr(rc, "parse_args", _fake_parse_args)

    with pytest.raises(SystemExit) as excinfo:
        rc.main()

    assert excinfo.value.code == 1
    assert output_json.exists()

    with output_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "dummy_cli"
    assert data["status"] == "error"


def test_ready_check_main_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    main() should complete without raising when dataset is OK, even if fail_on_error=True.
    """
    root = tmp_path / "dataset_cli_ok"
    root.mkdir(parents=True, exist_ok=True)

    img = root / "img.jpg"
    _make_dummy_image(img)

    _write_metadata(
        root,
        [{"image_id": "1", "image_path": "img.jpg", "label": "A", "split": "train"}],
    )

    output_json = tmp_path / "report_ok.json"

    def _fake_parse_args():
        return SimpleNamespace(
            dataset="dummy_ok",
            root=str(root),
            metadata_name="metadata.csv",
            output_json=str(output_json),
            fail_on_error=True,
            verbose=0,
        )

    monkeypatch.setattr(rc, "parse_args", _fake_parse_args)

    rc.main()

    assert output_json.exists()
    with output_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "dummy_ok"
    assert data["status"] == "ok"
    assert data["num_missing_files"] == 0
