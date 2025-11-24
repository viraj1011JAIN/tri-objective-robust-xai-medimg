# scripts/data/build_padchest_metadata.py
"""
Build unified metadata.csv for the PadChest chest X-ray dataset.

This script:

- Reads the official PadChest CSV (e.g. PADCHEST_chest_x_ray_images_labels_160K.csv)
- Normalises columns into the tri-objective-robust-xai-medimg format:
    * image_id      : string ID for each image
    * image_path    : relative path from ROOT to the image file
    * label         : binary label ("target" / "UNK") for high-level tasks
    * split         : "train" / "val" / "test" (patient-level split)
    * finding_labels: normalised multi-label string
    * patient_id    : patient identifier
    * view_position : view / projection (if available)
- Preserves all original PadChest columns for analysis.
- Writes:
    * <root>/metadata.csv
    * <root>/metadata_summary.json (optional)

Typical CLI usage (from repo root):

    python -m scripts.data.build_padchest_metadata --root/content/drive/MyDrive/data/padchest

You can also call `build_padchest_metadata(...)` directly from a notebook.

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
from datetime import datetime, timezone  # Add missing import
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# =============================================================================
# Logging
# =============================================================================


def setup_logger(name: str = "padchest_metadata", verbosity: int = 1) -> logging.Logger:
    """Create a console logger with adjustable verbosity."""
    logger = logging.getLogger(name)
    # Avoid duplicate handlers if re-imported
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


# =============================================================================
# Core helpers
# =============================================================================


def _find_padchest_csv(root: Path, logger: logging.Logger) -> Path:
    """
    Try to locate the main PadChest CSV under the given root.

    We first look for a CSV whose name contains both 'PADCHEST' and 'LABELS'.
    If not found, we fall back to any *.csv in the root directory.

    Raises:
        FileNotFoundError if no suitable CSV is found.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    candidates: List[Path] = []

    # Most common official name
    for pattern in [
        "PADCHEST*labels*160K*.csv",
        "PADCHEST*LABELS*.csv",
        "*.csv",
    ]:
        found = list(root.glob(pattern))
        candidates.extend(found)

    # De-duplicate
    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # Prefer ones that look like the official PadChest CSV
    preferred: List[Path] = [
        c
        for c in unique_candidates
        if "PADCHEST" in c.name.upper() and "LABEL" in c.name.upper()
    ]

    if preferred:
        csv_path = preferred[0]
        logger.info(f"Using PadChest CSV (preferred): {csv_path}")
        return csv_path

    if unique_candidates:
        csv_path = unique_candidates[0]
        logger.warning(
            f"No obvious PadChest CSV found; using first CSV in root: {csv_path}"
        )
        return csv_path

    raise FileNotFoundError(
        f"No CSV files found under {root}. Please pass --csv-path explicitly."
    )


def _parse_labels(value) -> List[str]:
    """
    Robustly parse the 'Labels' field from PadChest into a list of strings.

    Handles:
    - Python list-like strings: "['metal', 'foreign body']"
    - Single strings
    - Strings with separators ('|', ';', ',')
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    # Try list-like repr using ast.literal_eval
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip() for v in parsed if str(v).strip()]
        # Single item (string / number)
        return [str(parsed).strip()]
    except Exception:
        pass

    # Fallback: try common delimiters
    for sep in ["|", ";", ","]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            return [p for p in parts if p]

    return [text]


_NORMAL_SYNONYMS = {
    "normal",
    "unchanged",
    "no finding",
    "no findings",
    "unremarkable",
    "clear",
}


def _infer_binary_label(labels: List[str]) -> str:
    """
    Convert the PadChest multi-label list into a high-level binary label.

    Rule-of-thumb (aligned with NIH-style logic):

    - If labels is empty  -> "UNK"
    - If *all* labels are "normal-like" -> "UNK"
    - Otherwise -> "target"
    """
    if not labels:
        return "UNK"

    labels_lower = [lbl.lower().strip() for lbl in labels if lbl.strip()]
    if labels_lower and all(lbl in _NORMAL_SYNONYMS for lbl in labels_lower):
        return "UNK"

    return "target"


def _make_patient_splits(
    patient_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    logger: logging.Logger,
) -> Dict[str, str]:
    """
    Create patient-level train/val/test splits.

    Returns:
        dict patient_id -> split
    """
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if ratios.sum() <= 0.0:
        raise ValueError("Sum of train/val/test ratios must be > 0.")

    ratios = ratios / ratios.sum()
    train_ratio, val_ratio, test_ratio = ratios.tolist()

    unique_pids = sorted(set(patient_ids))
    n_patients = len(unique_pids)

    rng = random.Random(seed)
    rng.shuffle(unique_pids)

    n_train = int(round(train_ratio * n_patients))
    n_val = int(round(val_ratio * n_patients))

    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train : n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val :])

    logger.info(
        f"Patient-level split: {n_patients} patients -> "
        f"{len(train_pids)} train, {len(val_pids)} val, {len(test_pids)} test "
        f"(ratios approx: {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f})"
    )

    mapping: Dict[str, str] = {}
    for pid in unique_pids:
        if pid in train_pids:
            mapping[pid] = "train"
        elif pid in val_pids:
            mapping[pid] = "val"
        else:
            mapping[pid] = "test"

    return mapping


# =============================================================================
# Main builder
# =============================================================================


def build_padchest_metadata(
    root: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    summary_json: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    verbosity: int = 1,
) -> pd.DataFrame:
    """
    Core function to build PadChest metadata.

    Args:
        root: Path to PadChest root directory (where images + CSV live).
        output_csv: Target metadata.csv path (default: <root>/metadata.csv).
        summary_json: Optional summary JSON path
                      (default: <root>/metadata_summary.json).
        csv_path: Path to the original PadChest CSV. If None, auto-detect.
        train_ratio, val_ratio, test_ratio: patient-level split ratios.
        seed: RNG seed for split reproducibility.
        verbosity: logger verbosity (0=warn, 1=info, 2+=debug).

    Returns:
        The final metadata DataFrame.
    """
    logger = setup_logger("padchest_metadata", verbosity)

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_path}")

    if csv_path is None:
        padchest_csv = _find_padchest_csv(root_path, logger)
    else:
        padchest_csv = Path(csv_path).expanduser().resolve()
        if not padchest_csv.exists():
            raise FileNotFoundError(f"PadChest CSV does not exist: {padchest_csv}")

    if output_csv is None:
        output_csv_path = root_path / "metadata.csv"
    else:
        output_csv_path = Path(output_csv).expanduser().resolve()

    if summary_json is None:
        summary_json_path = root_path / "metadata_summary.json"
    else:
        summary_json_path = Path(summary_json).expanduser().resolve()

    logger.info("=" * 80)
    logger.info("Building PadChest metadata")
    logger.info(f"Root          : {root_path}")
    logger.info(f"PadChest CSV  : {padchest_csv}")
    logger.info(f"Output CSV    : {output_csv_path}")
    logger.info(f"Summary JSON  : {summary_json_path}")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Load raw CSV
    # -------------------------------------------------------------------------
    df_raw = pd.read_csv(padchest_csv)
    logger.info(
        f"Loaded PadChest CSV with {len(df_raw)} rows and {len(df_raw.columns)} columns."
    )

    required_cols = {"ImageID", "ImageDir", "Labels", "PatientID"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"PadChest CSV is missing required columns: {missing}. "
            f"Expected at least: {sorted(required_cols)}"
        )

    # -------------------------------------------------------------------------
    # Core columns: image_id, image_path, patient_id, finding_labels, label
    # -------------------------------------------------------------------------
    image_ids = df_raw["ImageID"].astype(str).str.strip()
    image_dirs = df_raw["ImageDir"].astype(str).str.strip().str.replace("\\", "/")

    # Relative path from root to the image file
    image_path_rel = (image_dirs + "/" + image_ids).str.replace("//", "/")

    # Parse labels + binary label
    labels_list = df_raw["Labels"].apply(_parse_labels)
    finding_labels_str = labels_list.apply(lambda xs: "|".join(xs) if xs else "")

    binary_labels = labels_list.apply(_infer_binary_label)

    # Patient & view position
    patient_ids = df_raw["PatientID"].astype(str).str.strip()

    if "ViewPosition_DICOM" in df_raw.columns:
        view_position = df_raw["ViewPosition_DICOM"].astype(str).str.strip()
    elif "Projection" in df_raw.columns:
        view_position = df_raw["Projection"].astype(str).str.strip()
    else:
        view_position = pd.Series([""] * len(df_raw), index=df_raw.index)

    # -------------------------------------------------------------------------
    # Patient-level splits
    # -------------------------------------------------------------------------
    pid_to_split = _make_patient_splits(
        patient_ids=patient_ids.tolist(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        logger=logger,
    )
    splits = patient_ids.map(pid_to_split)

    # -------------------------------------------------------------------------
    # Assemble final metadata DataFrame
    # -------------------------------------------------------------------------
    df = df_raw.copy()

    # Insert the unified columns at the front for easy access
    df.insert(0, "image_id", image_ids)
    df.insert(1, "image_path", image_path_rel)
    df.insert(2, "label", binary_labels)
    df.insert(3, "split", splits)

    # Additional convenience columns
    df["finding_labels"] = finding_labels_str
    df["patient_id"] = patient_ids
    df["view_position"] = view_position
    df["image_path_rel"] = image_path_rel

    # -------------------------------------------------------------------------
    # Basic sanity / stats
    # -------------------------------------------------------------------------
    split_counts = df["split"].value_counts().to_dict()
    label_counts = df["label"].value_counts().to_dict()

    logger.info(f"Total rows          : {len(df)}")
    logger.info(f"Split counts        : {split_counts}")
    logger.info(f"Binary label counts : {label_counts}")
    logger.info(f"Unique patients     : {df['patient_id'].nunique()}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"metadata.csv written to: {output_csv_path}")

    summary = {
        "dataset": "padchest",
        "root": str(root_path),
        "source_csv": str(padchest_csv),
        "output_csv": str(output_csv_path),
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "splits": {k: int(v) for k, v in split_counts.items()},
        "labels": {k: int(v) for k, v in label_counts.items()},
        "num_unique_patients": int(df["patient_id"].nunique()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
    }

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON written to: {summary_json_path}")
    logger.info("PadChest metadata building COMPLETE")
    logger.info("=" * 80)

    return df


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    CLI interface – mirrors the NIH script arguments so you can call it
    in the same way from Jupyter or PowerShell.
    """
    parser = argparse.ArgumentParser(
        description="Build metadata.csv + summary JSON for the PadChest dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="PadChest root directory (where images and the original CSV live).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output metadata CSV path (default: <root>/metadata.csv).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Output summary JSON path (default: <root>/metadata_summary.json).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Explicit path to the original PadChest CSV. "
            "If omitted, the script tries to auto-detect it."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of patients assigned to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of patients assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of patients assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient-level splitting.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (can be repeated).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    build_padchest_metadata(
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
