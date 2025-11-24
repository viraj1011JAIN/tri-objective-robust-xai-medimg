# scripts/data/preprocess_data.py
"""
Unified preprocessing pipeline for medical imaging datasets.

Features
--------
- Supports ISIC 2018/2019/2020, Derm7pt, NIH CXR, PadChest.
- Uses metadata.csv with columns: image_id, image_path, label, split.
- Resizes images to a standard size (default 224 x 224).
- Normalises intensities:
    * zero_one: float32 in [0, 1]
    * imagenet: standard ImageNet mean/std normalisation
    * none: raw uint8 values
- Saves processed data as:
    * JPEG images under <output_dir>/images/<split>/<image_id>.jpg
    * metadata_processed.csv that mirrors metadata.csv but points to processed images
    * Optionally: a single HDF5 file (dataset.h5) with images, labels and splits
- Logs a full preprocessing summary to JSON (preprocess_log.json).
- Integrates with src.datasets.data_governance for:
    * Compliance checks (assert_data_usage_allowed)
    * Data access logging (log_data_access)
    * Provenance logging (log_provenance)

Typical usage
-------------
Direct:
    python -m scripts.data.preprocess_data \
        --dataset isic2020 \
        --root/content/drive/MyDrive/data/isic_2020 \
        --output-dir data/processed/isic2020 \
        --image-size 224 \
        --to-hdf5

Via DVC (see dvc.yaml stages).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.datasets import data_governance as gov

# Optional HDF5 support
try:
    import h5py  # type: ignore

    H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment detail
    h5py = None
    H5PY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str
    env_var: str
    default_subdir: str
    default_splits: Tuple[str, ...]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "isic2018": DatasetConfig(
        name="isic2018",
        env_var="ISIC2018_ROOT",
        default_subdir="isic_2018",
        default_splits=("train", "val", "test"),
    ),
    "isic2019": DatasetConfig(
        name="isic2019",
        env_var="ISIC2019_ROOT",
        default_subdir="isic_2019",
        default_splits=("train", "val", "test"),
    ),
    "isic2020": DatasetConfig(
        name="isic2020",
        env_var="ISIC2020_ROOT",
        default_subdir="isic_2020",
        default_splits=("train", "test"),
    ),
    "derm7pt": DatasetConfig(
        name="derm7pt",
        env_var="DERM7PT_ROOT",
        default_subdir="derm7pt",
        default_splits=("train", "val", "test"),
    ),
    "nih_cxr": DatasetConfig(
        name="nih_cxr",
        env_var="NIH_CXR_ROOT",
        default_subdir="nih_cxr",
        default_splits=("train", "val", "test"),
    ),
    "padchest": DatasetConfig(
        name="padchest",
        env_var="PADCHEST_ROOT",
        default_subdir="padchest",
        default_splits=("train", "val", "test"),
    ),
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logger(verbosity: int = 1) -> logging.Logger:
    logger = logging.getLogger("preprocess_data")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    if verbosity <= 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    return logger


# ---------------------------------------------------------------------------
# Dataclasses for run config and logging
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    dataset: str
    root: Path
    output_dir: Path
    metadata_csv: Path
    splits: Sequence[str]
    image_size: int
    normalize: str
    to_hdf5: bool
    max_samples: Optional[int]
    num_workers: int


@dataclass
class RunLog:
    dataset: str
    version: str
    root: str
    output_dir: str
    metadata_csv: str
    image_size: int
    normalize: str
    to_hdf5: bool
    max_samples: Optional[int]
    num_workers: int
    num_input_rows: int
    num_processed_rows: int
    num_failed: int
    splits_present: List[str]
    unique_labels: List[str]
    run_time_sec: float
    timestamp_utc: str
    git_commit: Optional[str]
    dvc_lock_sha1: Optional[str]


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def resolve_dataset_root(
    cfg: DatasetConfig, logger: logging.Logger, override_root: Optional[str] = None
) -> Path:
    """
    Resolve dataset root using, in order of priority:
    1) Explicit CLI --root (override_root)
    2) Environment variable (e.g. ISIC2018_ROOT)
    3) Common fallback paths (/content/drive/MyDrive/data, C:/Users/.../data, ./data)
    """
    if override_root is not None:
        root = Path(override_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"--root was provided as {root}, but this path does not exist."
            )
        logger.info("Using explicit --root=%s", root)
        return root

    env_path = os.environ.get(cfg.env_var)
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            logger.info("Using %s=%s", cfg.env_var, p)
            return p
        logger.warning(
            "Environment variable %s is set to %s but path does not exist.",
            cfg.env_var,
            p,
        )

    candidates = [
        Path("/content/drive/MyDrive/data") / cfg.default_subdir,  # Samsung SSD T7 (primary)
        Path("/content/drive/MyDrive/data") / cfg.default_subdir,  # Legacy external drive
        Path("/content/drive/MyDrive/data") / cfg.default_subdir,
        Path("C:/Users/Dissertation/data") / cfg.default_subdir,
        Path("C:/Users/Viraj Jain/data") / cfg.default_subdir,
        Path.home() / "data" / cfg.default_subdir,
        Path("data") / cfg.default_subdir,
    ]

    for c in candidates:
        if c.exists():
            logger.info("Resolved root for %s: %s", cfg.name, c)
            return c

    raise FileNotFoundError(
        f"Could not resolve root for dataset {cfg.name}. "
        f"Set {cfg.env_var}, pass --root, or create {cfg.default_subdir} "
        f"under a known data directory."
    )


def load_metadata(metadata_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load metadata.csv and validate required columns.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_csv}")

    logger.info("Loading metadata from %s", metadata_csv)
    meta = pd.read_csv(metadata_csv)

    required_cols = ["image_id", "image_path", "label", "split"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise ValueError(
            f"metadata.csv is missing required columns: {missing}. "
            f"Expected at least {required_cols}"
        )

    logger.info(
        "Loaded metadata: %d rows, splits=%s, labels=%s",
        len(meta),
        sorted(meta["split"].astype(str).unique()),
        sorted(meta["label"].astype(str).unique()),
    )
    return meta


def normalise_image(arr: np.ndarray, mode: str = "zero_one") -> np.ndarray:
    """
    Normalise HWC uint8 image to float32 according to the selected mode.

    Returns an array of shape HWC, dtype float32.
    """
    if mode == "none":
        return arr.astype(np.float32)

    arr = arr.astype(np.float32) / 255.0

    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

    return arr


def resolve_image_path(root: Path, image_path: str) -> Path:
    """
    Resolve an image path that may be relative to the dataset root or absolute.

    Handles Windows-style paths and common variants.
    """
    p = Path(str(image_path)).expanduser()

    # Absolute path that exists
    if p.is_absolute() and p.exists():
        return p

    # Relative to root
    candidate = root / p
    if candidate.exists():
        return candidate

    # Clean Windows-style paths / leading ".\\"
    cleaned = str(image_path).replace("\\\\", "\\").replace("\\", os.sep)
    if cleaned.startswith(f".{os.sep}"):
        cleaned = cleaned[2:]
    candidate2 = root / cleaned
    if candidate2.exists():
        return candidate2

    # Fallback to basename only
    basename = Path(image_path).name
    candidate3 = root / basename
    if candidate3.exists():
        return candidate3

    raise FileNotFoundError(
        f"Could not resolve image path '{image_path}' under root '{root}'"
    )


def open_and_preprocess_image(
    img_path: Path, image_size: int, normalize: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image from disk, convert to RGB, resize and normalise.

    Returns
    -------
    arr_uint8 : np.ndarray
        HWC uint8 image in [0, 255] (for saving as JPEG/PNG).
    arr_chw : np.ndarray
        CHW float32 image after normalisation.
    """
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        arr_uint8 = np.asarray(img, dtype=np.uint8)  # HWC

    arr_norm = normalise_image(arr_uint8, mode=normalize)  # HWC float32
    arr_chw = np.transpose(arr_norm, (2, 0, 1))  # CHW
    return arr_uint8, arr_chw.astype(np.float32)


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        return out.strip()
    except Exception:
        return None


def get_dvc_lock_sha1(repo_root: Path) -> Optional[str]:
    """
    Compute a SHA1 of dvc.lock if it exists. This is not DVC's internal hash,
    but serves as a simple provenance fingerprint.
    """
    lock_path = repo_root / "dvc.lock"
    if not lock_path.exists():
        return None

    import hashlib

    sha1 = hashlib.sha1()
    with lock_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


# ---------------------------------------------------------------------------
# Preprocessing core
# ---------------------------------------------------------------------------


def preprocess_dataset(
    cfg: PreprocessConfig,
    raw_meta: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, int]:
    """
    Main driver: iterates over metadata rows, writes processed images and
    optionally appends to a HDF5 file.

    Parameters
    ----------
    cfg:
        PreprocessConfig with dataset-specific settings.
    raw_meta:
        Full metadata DataFrame loaded from metadata.csv.
    logger:
        Logger instance.

    Returns
    -------
    processed_df:
        New metadata DataFrame with updated image_path pointing to processed images.
    num_failed:
        Number of samples that could not be processed.
    """
    # Filter by requested splits
    meta = raw_meta.copy()
    if cfg.splits:
        meta = meta[meta["split"].isin(cfg.splits)].copy()

    if cfg.max_samples is not None and cfg.max_samples > 0:
        meta = meta.iloc[: cfg.max_samples].copy()

    num_rows = len(meta)
    if num_rows == 0:
        logger.warning(
            "No rows in metadata for requested splits -> nothing to preprocess."
        )
        return meta, 0

    logger.info("Preparing to preprocess %d samples", num_rows)

    # Prepare output directories
    images_root = cfg.output_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    # HDF5 initialisation (resizable datasets)
    h5_file = None
    img_dset = None
    label_dset = None
    id_dset = None
    split_dset = None
    label_to_idx: Dict[str, int] = {}

    if cfg.to_hdf5:
        if not H5PY_AVAILABLE:
            raise RuntimeError(
                "h5py is not installed but --to-hdf5 was requested. "
                "Install h5py or run without --to-hdf5."
            )

        labels_sorted = sorted(meta["label"].astype(str).unique().tolist())
        label_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}

        h5_path = cfg.output_dir / "dataset.h5"
        logger.info("Creating resizable HDF5 file at %s", h5_path)

        h5_file = h5py.File(h5_path, "w")  # type: ignore
        img_dset = h5_file.create_dataset(
            "images",
            shape=(0, 3, cfg.image_size, cfg.image_size),
            maxshape=(None, 3, cfg.image_size, cfg.image_size),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(1, 3, cfg.image_size, cfg.image_size),
        )
        label_dset = h5_file.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype="int64",
        )
        id_dset = h5_file.create_dataset(
            "image_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),  # type: ignore
        )
        split_dset = h5_file.create_dataset(
            "splits",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),  # type: ignore
        )

        # store mapping as JSON for clarity
        h5_file.attrs["label_to_idx"] = json.dumps(label_to_idx)
        h5_file.attrs["labels"] = json.dumps(labels_sorted)
        h5_file.attrs["image_size"] = int(cfg.image_size)
        h5_file.attrs["normalize"] = cfg.normalize
        h5_file.attrs["dataset"] = cfg.dataset

    processed_records: List[Dict[str, object]] = []
    num_failed = 0

    for _, row in meta.iterrows():
        image_id = str(row["image_id"])
        raw_image_path = str(row["image_path"])
        label = row["label"]
        split = str(row["split"])

        try:
            abs_path = resolve_image_path(cfg.root, raw_image_path)
            arr_uint8, arr_chw = open_and_preprocess_image(
                abs_path, cfg.image_size, cfg.normalize
            )

            # Save processed JPEG
            split_dir = images_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_filename = f"{image_id}.jpg"
            out_path = split_dir / out_filename

            out_img = Image.fromarray(arr_uint8)
            out_img.save(out_path, format="JPEG", quality=95)

            # Append to HDF5, if enabled
            if cfg.to_hdf5 and h5_file is not None and img_dset is not None:
                assert (
                    label_dset is not None
                    and id_dset is not None
                    and split_dset is not None
                )
                idx = img_dset.shape[0]
                img_dset.resize(idx + 1, axis=0)
                img_dset[idx] = arr_chw  # type: ignore[index]

                label_idx = label_to_idx[str(label)]
                label_dset.resize(idx + 1, axis=0)
                label_dset[idx] = int(label_idx)  # type: ignore[index]

                id_dset.resize(idx + 1, axis=0)
                id_dset[idx] = image_id  # type: ignore[index]

                split_dset.resize(idx + 1, axis=0)
                split_dset[idx] = split  # type: ignore[index]

            # Record processed metadata (relative path from processed root)
            rel_processed_path = Path("images") / split / out_filename
            processed_records.append(
                {
                    "image_id": image_id,
                    "image_path": str(rel_processed_path).replace("\\", "/"),
                    "label": label,
                    "split": split,
                    "original_image_path": raw_image_path,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            num_failed += 1
            logger.warning(
                "Failed to process image_id=%s, image_path=%s: %s",
                image_id,
                raw_image_path,
                exc,
            )
            continue

    if h5_file is not None:
        h5_file.close()

    processed_df = pd.DataFrame(processed_records)
    logger.info(
        "Successfully processed %d/%d images (failed=%d)",
        len(processed_df),
        num_rows,
        num_failed,
    )

    return processed_df, num_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified preprocessing pipeline for medical imaging datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset name to preprocess.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Root directory of the RAW dataset. If omitted, will be resolved "
            "from environment variables and common paths."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to preprocess (default: dataset-specific defaults).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output image size (square). Default: 224",
    )
    parser.add_argument(
        "--normalize",
        "--normalization",
        dest="normalize",
        choices=["none", "zero_one", "imagenet"],
        default="zero_one",
        help="Intensity normalisation strategy.",
    )
    parser.add_argument(
        "--to-hdf5",
        action="store_true",
        help="Also write a dataset.h5 file with float32 images and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory. Default: data/processed/<dataset>",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (for debugging / smoke tests).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Reserved for potential multiprocessing; currently unused "
            "but logged for reproducibility."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (can be repeated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)

    ds_cfg = DATASET_CONFIGS[args.dataset]

    # ------------------------------------------------------------------
    # 1) Compliance gate: ensure usage is research / non-commercial
    # ------------------------------------------------------------------
    try:
        gov.assert_data_usage_allowed(
            ds_cfg.name,
            purpose="research",
            commercial=False,
        )
    except PermissionError as exc:
        logger.error("Data usage not allowed for dataset=%s: %s", ds_cfg.name, exc)
        raise
    except Exception:
        # Governance should not break the pipeline if misconfigured
        logger.debug(
            "Compliance check failed unexpectedly for dataset=%s",
            ds_cfg.name,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # 2) Resolve paths and configuration
    # ------------------------------------------------------------------
    root = resolve_dataset_root(ds_cfg, logger, override_root=args.root)

    if args.splits is None:
        splits: Sequence[str] = list(ds_cfg.default_splits)
    else:
        splits = args.splits

    base_out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (Path("data/processed") / ds_cfg.name)
    )
    base_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = root / "metadata.csv"

    logger.info("=" * 70)
    logger.info("PREPROCESSING DATASET")
    logger.info("Dataset   : %s", ds_cfg.name)
    logger.info("Root      : %s", root)
    logger.info("Metadata  : %s", metadata_csv)
    logger.info("Splits    : %s", splits)
    logger.info("Image size: %d", args.image_size)
    logger.info("Normalise : %s", args.normalize)
    logger.info("To HDF5   : %s", bool(args.to_hdf5))
    logger.info("Output dir: %s", base_out_dir)
    logger.info("Max smpls : %s", args.max_samples)
    logger.info("Num workers (unused): %d", args.num_workers)
    logger.info("=" * 70)

    # Load raw metadata once
    raw_meta = load_metadata(metadata_csv, logger)

    # Log read access (dataset-level)
    try:
        gov.log_data_access(
            dataset_name=ds_cfg.name,
            split="*",
            action="read",
            purpose="preprocessing",
            num_samples=int(len(raw_meta)),
            extra={
                "stage": "preprocess",
                "image_size": args.image_size,
                "normalize": args.normalize,
                "to_hdf5": bool(args.to_hdf5),
            },
        )
    except Exception:
        logger.debug(
            "Failed to log data access (read) for dataset=%s",
            ds_cfg.name,
            exc_info=True,
        )

    cfg = PreprocessConfig(
        dataset=ds_cfg.name,
        root=root,
        output_dir=base_out_dir,
        metadata_csv=metadata_csv,
        splits=splits,
        image_size=args.image_size,
        normalize=args.normalize,
        to_hdf5=bool(args.to_hdf5),
        max_samples=args.max_samples,
        num_workers=int(args.num_workers),
    )

    # ------------------------------------------------------------------
    # 3) Actual preprocessing
    # ------------------------------------------------------------------
    overall_start = time.time()
    processed_df, num_failed = preprocess_dataset(cfg, raw_meta, logger)
    overall_elapsed = time.time() - overall_start

    # Save processed metadata CSV
    meta_out_path = base_out_dir / "metadata_processed.csv"
    processed_df.to_csv(meta_out_path, index=False)
    logger.info("Saved processed metadata CSV to %s", meta_out_path)

    # ------------------------------------------------------------------
    # 4) Build run log
    # ------------------------------------------------------------------
    git_commit = get_git_commit()
    repo_root = Path(__file__).resolve().parents[2]
    dvc_sha1 = get_dvc_lock_sha1(repo_root)

    log = RunLog(
        dataset=cfg.dataset,
        version="1.0",
        root=str(cfg.root),
        output_dir=str(cfg.output_dir),
        metadata_csv=str(cfg.metadata_csv),
        image_size=cfg.image_size,
        normalize=cfg.normalize,
        to_hdf5=cfg.to_hdf5,
        max_samples=cfg.max_samples,
        num_workers=cfg.num_workers,
        num_input_rows=int(len(raw_meta)),
        num_processed_rows=int(len(processed_df)),
        num_failed=int(num_failed),
        splits_present=(
            sorted(processed_df["split"].astype(str).unique().tolist())
            if not processed_df.empty
            else []
        ),
        unique_labels=(
            sorted(processed_df["label"].astype(str).unique().tolist())
            if not processed_df.empty
            else []
        ),
        run_time_sec=float(overall_elapsed),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        dvc_lock_sha1=dvc_sha1,
    )

    log_path = base_out_dir / "preprocess_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(log), f, indent=2)

    # ------------------------------------------------------------------
    # 5) Governance: per-split write logs
    # ------------------------------------------------------------------
    try:
        if not processed_df.empty:
            split_counts = processed_df["split"].astype(str).value_counts().to_dict()
            for split, count in split_counts.items():
                gov.log_data_access(
                    dataset_name=cfg.dataset,
                    split=str(split),
                    action="write",
                    purpose="preprocessing",
                    num_samples=int(count),
                    extra={
                        "format": "hdf5+jpeg" if cfg.to_hdf5 else "jpeg_only",
                        "image_size": cfg.image_size,
                        "normalize": cfg.normalize,
                    },
                )
    except Exception:
        logger.debug(
            "Failed to log data access (write) for dataset=%s",
            cfg.dataset,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # 6) Governance: provenance record
    # ------------------------------------------------------------------
    try:
        output_paths: List[Path] = [meta_out_path, log_path]
        if cfg.to_hdf5:
            h5_path = cfg.output_dir / "dataset.h5"
            if h5_path.exists():
                output_paths.append(h5_path)

        gov.log_provenance(
            stage=f"preprocess_{cfg.dataset}",
            dataset_name=cfg.dataset,
            input_paths=[metadata_csv],
            output_paths=output_paths,
            params={
                "image_size": cfg.image_size,
                "normalize": cfg.normalize,
                "to_hdf5": cfg.to_hdf5,
                "splits": list(cfg.splits),
                "max_samples": cfg.max_samples,
            },
            tags={"script": "preprocess_data"},
        )
    except Exception:
        logger.debug(
            "Failed to log provenance for dataset=%s",
            cfg.dataset,
            exc_info=True,
        )

    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("Log written to %s", log_path)
    logger.info("Total elapsed: %.1f s", overall_elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
