# tests/test_preprocess_data.py
"""
Unit tests for scripts/data/preprocess_data.py

Covers:
- Synthetic metadata + images
- preprocess_dataset() writing processed JPEGs and returning a DataFrame
- Optional HDF5 output
- main() writing metadata_processed.csv and preprocess_log.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.data import preprocess_data as pp  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_image(path: Path, size: int = 64) -> None:
    """Create a small RGB dummy image at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path, format="JPEG")


def _build_tiny_metadata(tmp_root: Path) -> Path:
    """
    Create a minimal metadata.csv with:
    - 2 train samples
    - 1 val sample
    - 1 test sample
    All pointing to images under tmp_root/raw_images/.
    """
    raw_img_dir = tmp_root / "raw_images"
    raw_img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, split in enumerate(["train", "train", "val", "test"]):
        img_id = f"img_{idx:03d}"
        rel_path = f"raw_images/{img_id}.jpg"
        _make_dummy_image(raw_img_dir / f"{img_id}.jpg")
        rows.append(
            {
                "image_id": img_id,
                "image_path": rel_path,
                "label": "class_0" if idx < 3 else "class_1",
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_root / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _call_preprocess_dataset(
    cfg: pp.PreprocessConfig, logger
) -> Tuple[pd.DataFrame, int]:
    """
    Call preprocess_dataset in a way that is robust to signature changes.

    Handles all of these shapes:

    - def preprocess_dataset(cfg)
    - def preprocess_dataset(cfg, *, logger)
    - def preprocess_dataset(cfg, raw_meta)
    - def preprocess_dataset(cfg, raw_meta, logger)
    - def preprocess_dataset(cfg, raw_meta, *, logger)

    We infer raw_meta from cfg.metadata_csv when needed.
    """
    import inspect

    sig = inspect.signature(pp.preprocess_dataset)
    param_names = list(sig.parameters.keys())

    args = [cfg]
    kwargs = {}

    needs_raw_meta = "raw_meta" in param_names
    needs_logger = "logger" in param_names

    if needs_raw_meta:
        # Load raw metadata from the CSV path in the config
        raw_meta = pd.read_csv(cfg.metadata_csv)
        args.append(raw_meta)

    if needs_logger:
        # Works for positional-or-keyword and keyword-only "logger"
        kwargs["logger"] = logger

    return pp.preprocess_dataset(*args, **kwargs)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_preprocess_dataset_creates_jpegs_and_metadata(tmp_path: Path) -> None:
    """
    Smoke test: preprocess_dataset() on a tiny synthetic dataset.

    - processed_df has the expected number of rows
    - JPEGs exist under output_dir/images/<split>/
    - paths inside processed_df are relative to output_dir
    """
    root = tmp_path / "root"
    root.mkdir()
    metadata_csv = _build_tiny_metadata(root)

    output_dir = tmp_path / "processed"
    logger = pp.setup_logger(verbosity=0)

    cfg = pp.PreprocessConfig(
        dataset="dummy_small",
        root=root,
        output_dir=output_dir,
        metadata_csv=metadata_csv,
        splits=("train", "val"),  # ignore test split on purpose
        image_size=32,
        normalize="zero_one",
        to_hdf5=False,
        max_samples=None,
        num_workers=0,
    )

    processed_df, num_failed = _call_preprocess_dataset(cfg, logger)

    # 3 rows for train+val
    assert len(processed_df) == 3
    assert num_failed == 0

    # All processed paths must live under output_dir/images/<split>/
    for _, row in processed_df.iterrows():
        rel_path = Path(row["image_path"])
        abs_path = output_dir / rel_path
        assert abs_path.exists()
        assert "images" in rel_path.parts

    # Check one sample's image size
    sample_path = output_dir / processed_df.iloc[0]["image_path"]
    img = Image.open(sample_path)
    assert img.size == (32, 32)


@pytest.mark.parametrize("to_hdf5", [False, True])
def test_preprocess_dataset_hdf5_optional(tmp_path: Path, to_hdf5: bool) -> None:
    """
    Test that:
    - when to_hdf5=False, no dataset.h5 is written
    - when to_hdf5=True, dataset.h5 exists and has the correct shape/attrs
    """
    # If h5py is not installed, skip the to_hdf5=True case
    if to_hdf5:
        pytest.importorskip("h5py")

    root = tmp_path / "root"
    root.mkdir()
    metadata_csv = _build_tiny_metadata(root)

    output_dir = tmp_path / "processed"
    logger = pp.setup_logger(verbosity=0)

    cfg = pp.PreprocessConfig(
        dataset="dummy_hdf5",
        root=root,
        output_dir=output_dir,
        metadata_csv=metadata_csv,
        splits=("train", "val"),
        image_size=32,
        normalize="zero_one",
        to_hdf5=to_hdf5,
        max_samples=None,
        num_workers=0,
    )

    processed_df, num_failed = _call_preprocess_dataset(cfg, logger)
    assert len(processed_df) == 3
    assert num_failed == 0

    h5_path = output_dir / "dataset.h5"

    if not to_hdf5:
        assert not h5_path.exists()
        return

    import h5py  # type: ignore

    assert h5_path.exists()

    with h5py.File(h5_path, "r") as h5f:
        imgs = h5f["images"]
        labels = h5f["labels"]
        splits = h5f["splits"]
        ids = h5f["image_ids"]

        assert imgs.shape[0] == len(processed_df)
        assert imgs.shape[1:] == (3, 32, 32)

        assert labels.shape[0] == len(processed_df)
        assert splits.shape[0] == len(processed_df)
        assert ids.shape[0] == len(processed_df)

        # Attributes should contain mapping metadata
        assert "label_to_idx" in h5f.attrs
        assert "labels" in h5f.attrs
        assert h5f.attrs["image_size"] == 32
        assert h5f.attrs["normalize"] == "zero_one"
        assert h5f.attrs["dataset"] == "dummy_hdf5"


def test_preprocess_pipeline_log_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Higher-level test: simulate (a subset of) main() logic and check that
    preprocess_log.json and metadata_processed.csv are written.
    """
    root = tmp_path / "root"
    root.mkdir()
    _ = _build_tiny_metadata(root)

    # Monkeypatch DATASET_CONFIGS to inject a synthetic dataset for this test
    fake_cfg = pp.DatasetConfig(
        name="test_dataset",
        env_var="TEST_DATASET_ROOT",
        default_subdir="test_dataset",
        default_splits=("train", "val", "test"),
    )
    monkeypatch.setitem(pp.DATASET_CONFIGS, "test_dataset", fake_cfg)

    # And ensure resolve_dataset_root will see our temporary root
    monkeypatch.setenv("TEST_DATASET_ROOT", str(root))

    # Build CLI-like args and call main() via monkeypatched parse_args
    def _fake_parse_args():
        class _Args:
            dataset = "test_dataset"
            root = None
            splits = ["train", "val"]
            image_size = 32
            normalize = "zero_one"
            to_hdf5 = False
            output_dir = str(tmp_path / "processed")
            max_samples = None
            num_workers = 0
            verbose = 0

        return _Args()

    monkeypatch.setattr(pp, "parse_args", _fake_parse_args)

    # Run main()
    pp.main()

    processed_dir = tmp_path / "processed"
    meta_processed = processed_dir / "metadata_processed.csv"
    log_path = processed_dir / "preprocess_log.json"

    assert meta_processed.exists(), "metadata_processed.csv should be created by main()"
    assert log_path.exists(), "preprocess_log.json should be created by main()"

    # Light check of log contents
    with log_path.open("r", encoding="utf-8") as f:
        log_data = json.load(f)

    assert log_data["dataset"] == "test_dataset"
    assert log_data["image_size"] == 32
    assert log_data["normalize"] == "zero_one"
    assert log_data["num_processed_rows"] == 3
