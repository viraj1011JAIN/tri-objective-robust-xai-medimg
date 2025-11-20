# scripts/data/build_derm7pt_metadata.py
"""
Build a unified metadata.csv for the Derm7pt dataset.

Output (aligned with the rest of your pipeline)
-----------------------------------------------
Creates a CSV with at least these columns:

- image_id        : unique image identifier (string)
- image_path      : path to the image *relative* to the dataset root
- label           : binary target label
                    * "target" for malignant / melanoma-like lesions
                    * "UNK"   for all other lesions
- split           : "train", "val", or "test"

Plus useful extra columns when available:
- finding_labels  : textual diagnosis / multiclass label (for concept bank / TCAV)
- patient_id      : patient ID (if present)
- lesion_id       : lesion ID (if present)
- image_path_rel  : same as image_path (kept for consistency)
- all original columns from the source CSV

Splitting strategy
------------------
- Prefer patient-level split if a "patient_id"-like column exists.
- Else, prefer lesion-level split if a "lesion_id"-like column exists.
- Else, fall back to sample-wise random split.

Usage
-----
From shell:

    python -m scripts.data.build_derm7pt_metadata --root F:/data/derm7pt

From notebook:

    from scripts.data.build_derm7pt_metadata import build_derm7pt_metadata
    df = build_derm7pt_metadata(root="F:/data/derm7pt")

This script is designed to be robust to different Derm7pt CSV variants, including
the Kaggle-style one with columns:
    ['filepath', 'target_melanoma', 'label_multiclass', 'diagnosis']
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logger(verbosity: int = 1) -> logging.Logger:
    logger = logging.getLogger("derm7pt_metadata")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
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
# Core helpers
# ---------------------------------------------------------------------------


def _find_derm7pt_csv(
    root: Path, logger: logging.Logger, explicit: Optional[Path] = None
) -> Path:
    """
    Try to locate the Derm7pt metadata CSV.

    Priority:
    1. explicit path (if provided)
    2. common Derm7pt file names in root
    3. first CSV file in root (with warning)
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit CSV path does not exist: {explicit}")
        logger.info(f"Using explicit Derm7pt CSV: {explicit}")
        return explicit

    # Known candidate names (non-exhaustive but robust)
    candidate_names = [
        "derm7pt_metadata.csv",
        "derm7pt_meta.csv",
        "derm7pt.csv",
        "meta.csv",
        "metadata_derm7pt.csv",
        "train.csv",  # common for Kaggle variants
    ]
    for name in candidate_names:
        cand = root / name
        if cand.exists():
            logger.info(f"Using Derm7pt CSV: {cand}")
            return cand

    # Fallback: first CSV in root
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in Derm7pt root: {root}. "
            f"Pass --csv-path explicitly if the file is elsewhere."
        )

    logger.warning(
        "No obvious Derm7pt CSV found; using first CSV in root: %s", csv_files[0]
    )
    return csv_files[0]


def _infer_image_column(df: pd.DataFrame) -> str:
    """
    Infer which column contains the image identifier / path.

    Updated to support Kaggle-style Derm7pt CSV with 'filepath' column.
    """
    candidates = [
        "filepath",  # your actual column name
        "image_id",
        "image",
        "image_name",
        "filename",
        "file_name",
        "file",
        "ImageID",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer image column. Tried: "
        f"{candidates}. Available columns: {list(df.columns)}"
    )


def _infer_group_column(df: pd.DataFrame) -> Optional[str]:
    """
    Decide whether to split by patient, lesion, or sample-wise.

    Returns:
        group_col name or None if no suitable column found.
    """
    patient_candidates = ["patient_id", "patient", "patientID", "patient_id_int"]
    lesion_candidates = ["lesion_id", "lesion", "lesionID", "lesion_id_int"]

    for col in patient_candidates:
        if col in df.columns:
            return col

    for col in lesion_candidates:
        if col in df.columns:
            return col

    return None  # fall back to sample-wise split


def _make_binary_label(row: pd.Series) -> str:
    """
    Map Derm7pt diagnosis into binary label:

    - "target" for malignant / melanoma-like lesions
    - "UNK"   for everything else

    Priority:
    1. 'target_melanoma' column if present (0/1 or boolean-like).
    2. diagnosis_group / diagnosis heuristic (for other Derm7pt variants).
    """
    # 1) Use target_melanoma when available
    if "target_melanoma" in row.index:
        val = row["target_melanoma"]
        if pd.notna(val):
            # Accept 0/1, True/False, or string equivalents
            s = str(val).strip().lower()
            if s in {"1", "true", "t", "yes", "y"}:
                return "target"
            if s in {"0", "false", "f", "no", "n"}:
                return "UNK"
            # If it's numeric, use > 0 as malignant
            try:
                if float(s) > 0:
                    return "target"
                return "UNK"
            except ValueError:
                # If weird string, fall through to heuristic
                pass

    # 2) Fallback: use diagnosis_group / diagnosis heuristic
    diag_group = str(row.get("diagnosis_group", "")).lower()
    diag = str(row.get("diagnosis", "")).lower()

    if "malignant" in diag_group:
        return "target"

    malignant_keywords = [
        "melanoma",
        "bcc",
        "basal cell carcinoma",
        "scc",
        "squamous cell carcinoma",
        "invasive carcinoma",
        "in situ carcinoma",
        "lentigo maligna",
    ]
    for kw in malignant_keywords:
        if kw in diag:
            return "target"

    return "UNK"


def _infer_image_base_dir(
    root: Path, image_names: List[str], logger: logging.Logger, max_checks: int = 50
) -> Path:
    """
    Try to infer which directory under root actually contains the images.

    We test a subset of filenames under a few likely directories:
    - root / "images"
    - root / "img"
    - root / "derm7pt"
    - root (direct)

    Returns:
        Path to base directory (may fall back to root with a warning).
    """
    if not image_names:
        logger.warning(
            "No image names found while inferring base directory. "
            "Using dataset root as image base directory."
        )
        return root

    sample_names = image_names[: max_checks or len(image_names)]

    candidates = [
        root / "images",
        root / "img",
        root / "derm7pt",
        root,
    ]

    for base in candidates:
        if not base.exists():
            continue
        hits = 0
        for name in sample_names:
            if (base / name).exists():
                hits += 1
        if hits > 0:
            logger.info(
                "Using image base directory: %s (found %d/%d sample files)",
                base,
                hits,
                len(sample_names),
            )
            return base

    logger.warning(
        "Could not confirm any image directory on disk; "
        "using dataset root as base image directory. "
        "Check image_path/image_path_rel manually if needed."
    )
    return root


def _assign_splits_grouped(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    """
    Assign splits train/val/test using group-wise splitting
    (e.g., by patient_id or lesion_id).

    Returns:
        A pd.Series of split labels aligned with df index.
    """
    groups = df[group_col].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])

    def _map_group(g: str) -> str:
        if g in train_groups:
            return "train"
        if g in val_groups:
            return "val"
        return "test"

    return df[group_col].astype(str).map(_map_group)


def _assign_splits_rowwise(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    """
    Assign splits train/val/test sample-wise (no grouping).
    """
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))

    split_arr = np.empty(n, dtype=object)
    split_arr[idx[:n_train]] = "train"
    split_arr[idx[n_train : n_train + n_val]] = "val"
    split_arr[idx[n_train + n_val :]] = "test"
    return pd.Series(split_arr, index=df.index)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


@dataclass
class Derm7ptMetadataSummary:
    total_rows: int
    split_counts: Dict[str, int]
    binary_label_counts: Dict[str, int]
    unique_groups: Optional[int]
    group_column: Optional[str]
    csv_path: str
    root: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


def build_derm7pt_metadata(
    root: str | Path,
    output_csv: Optional[str | Path] = None,
    summary_json: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    verbosity: int = 1,
) -> pd.DataFrame:
    """
    Build Derm7pt metadata.csv and metadata_summary.json.

    Parameters
    ----------
    root:
        Dataset root directory (where images and original CSV reside).
    output_csv:
        Where to write metadata.csv (default: <root>/metadata.csv).
    summary_json:
        Where to write summary JSON (default: <root>/metadata_summary.json).
    csv_path:
        Optional explicit path to the original Derm7pt CSV.
    train_ratio, val_ratio, test_ratio:
        Split ratios (must sum approximately to 1.0).
    seed:
        Random seed for reproducible splitting.
    verbosity:
        Logging verbosity level.

    Returns
    -------
    df : pd.DataFrame
        Final metadata DataFrame.
    """
    logger = _setup_logger(verbosity)
    root = Path(root).expanduser().resolve()

    if output_csv is None:
        output_csv = root / "metadata.csv"
    else:
        output_csv = Path(output_csv).expanduser().resolve()

    if summary_json is None:
        summary_json = root / "metadata_summary.json"
    else:
        summary_json = Path(summary_json).expanduser().resolve()

    csv_path_resolved = (
        Path(csv_path).expanduser().resolve() if csv_path is not None else None
    )

    logger.info("=" * 78)
    logger.info("Building Derm7pt metadata")
    logger.info("Root          : %s", root)
    logger.info("Output CSV    : %s", output_csv)
    logger.info("Summary JSON  : %s", summary_json)
    logger.info("=" * 78)

    # ------------------------------------------------------------------ #
    # Load source CSV
    # ------------------------------------------------------------------ #
    derm_csv = _find_derm7pt_csv(root, logger, csv_path_resolved)
    df = pd.read_csv(derm_csv)
    logger.info(
        "Loaded Derm7pt CSV with %d rows and %d columns.", len(df), len(df.columns)
    )

    # ------------------------------------------------------------------ #
    # Image identifiers and paths
    # ------------------------------------------------------------------ #
    img_col = _infer_image_column(df)

    if img_col == "filepath":
        # Kaggle-style: 'filepath' already contains a relative path from root.
        df["image_path_rel"] = df["filepath"].astype(str)
        # image_id as the file stem (without directories / extension)
        df["image_id"] = df["filepath"].astype(str).apply(lambda p: Path(p).stem)
    else:
        # Classic variant: image column contains just a filename, we must find base dir
        df["image_id"] = df[img_col].astype(str)
        image_names = df["image_id"].tolist()
        base_dir = _infer_image_base_dir(root, image_names, logger)

        if base_dir == root:
            df["image_path_rel"] = df["image_id"].astype(str)
        else:
            rel_prefix = base_dir.relative_to(root)
            df["image_path_rel"] = df["image_id"].apply(
                lambda x: str(rel_prefix / str(x))
            )

    # For consistency with the rest of your pipeline:
    df["image_path"] = df["image_path_rel"]

    # ------------------------------------------------------------------ #
    # Diagnosis / label fields
    # ------------------------------------------------------------------ #
    # finding_labels: prioritise diagnosis, else label_multiclass, else empty string
    if "diagnosis" in df.columns and df["diagnosis"].notna().any():
        df["finding_labels"] = df["diagnosis"].fillna("").astype(str)
    elif "label_multiclass" in df.columns:
        df["finding_labels"] = df["label_multiclass"].fillna("").astype(str)
    else:
        df["finding_labels"] = ""

    # Binary label
    df["label"] = df.apply(_make_binary_label, axis=1)

    # ------------------------------------------------------------------ #
    # Splits: train / val / test
    # ------------------------------------------------------------------ #
    group_col = _infer_group_column(df)
    if group_col is not None:
        logger.info("Using group-wise splitting by column: %s", group_col)
        df["split"] = _assign_splits_grouped(
            df=df,
            group_col=group_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        logger.info("No patient/lesion column found; using sample-wise splitting.")
        df["split"] = _assign_splits_rowwise(
            df=df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    # For convenience, standardize patient_id / lesion_id columns if present
    if group_col is not None and "patient_id" not in df.columns:
        # If our chosen group_col isn't literally "patient_id", still expose it
        df["patient_id"] = df[group_col].astype(str)

    # ------------------------------------------------------------------ #
    # Summary & sanity logs
    # ------------------------------------------------------------------ #
    split_counts = df["split"].value_counts().to_dict()
    split_counts = {k: int(v) for k, v in split_counts.items()}

    binary_counts = df["label"].value_counts().to_dict()
    binary_counts = {k: int(v) for k, v in binary_counts.items()}

    unique_groups = None
    if group_col is not None:
        unique_groups = int(df[group_col].nunique())

    logger.info("Total rows          : %d", len(df))
    logger.info("Split counts        : %s", split_counts)
    logger.info("Binary label counts : %s", binary_counts)
    if unique_groups is not None:
        logger.info("Unique groups (%s)  : %d", group_col, unique_groups)

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("metadata.csv written to: %s", output_csv)

    summary = Derm7ptMetadataSummary(
        total_rows=int(len(df)),
        split_counts=split_counts,
        binary_label_counts=binary_counts,
        unique_groups=unique_groups,
        group_column=group_col,
        csv_path=str(derm_csv),
        root=str(root),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
    )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    logger.info("Summary JSON written to: %s", summary_json)
    logger.info("Derm7pt metadata building COMPLETE")
    logger.info("=" * 78)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified Derm7pt metadata.csv for the robust-XAI pipeline."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Derm7pt dataset root directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to write metadata.csv (default: <root>/metadata.csv).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Path to write metadata_summary.json "
        "(default: <root>/metadata_summary.json).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional explicit path to the original Derm7pt CSV.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility.",
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
    args = _parse_args()
    build_derm7pt_metadata(
        root=args.root,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        csv_path=args.csv_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        verbosity=args.verbose,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
