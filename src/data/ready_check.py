# src/data/ready_check.py
"""
Dataset readiness / validation checks.

This small utility is responsible for **data validation before training**:

- Check that metadata.csv exists and has the required columns.
- Count rows, splits and labels.
- Verify that image files referenced by metadata actually exist.
- Produce a machine-readable JSON report that can be stored with the
  experiment or used by CI.

It is intentionally independent from the heavy training code so it can be
run as a lightweight pre-flight check or wired into DVC stages.

Example (CLI)
-------------

    python -m src.data.ready_check \\
        --dataset isic2020 \\
        --root/content/drive/MyDrive/data/isic_2020 \\
        --output-json results/data_ready/isic2020_ready.json \\
        --fail-on-error

Example (Python)
----------------

    from pathlib import Path
    from src.data.ready_check import run_ready_check

    report = run_ready_check("isic2020", Path("/content/drive/MyDrive/data/isic_2020"))
    print(report.status, report.num_missing_files)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetReadyReport:
    dataset: str
    root: str
    metadata_path: str
    exists_metadata: bool
    required_columns: Sequence[str]
    missing_columns: List[str]
    num_rows: int
    num_missing_files: int
    num_bad_rows: int
    unique_labels: List[str]
    splits: List[str]
    status: str  # "ok", "warning", "error"
    warnings: List[str]
    errors: List[str]


REQUIRED_COLUMNS: list[str] = ["image_id", "image_path", "label", "split"]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _setup_logger(verbosity: int = 1) -> logging.Logger:
    logger = logging.getLogger("data_ready")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    if verbosity <= 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    return logger


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run_ready_check(
    dataset: str,
    root: Path,
    metadata_name: str = "metadata.csv",
    logger: logging.Logger | None = None,
) -> DatasetReadyReport:
    """
    Run validation checks on a dataset folder.

    Parameters
    ----------
    dataset:
        Logical dataset name (e.g. "isic2020").
    root:
        Directory containing metadata.csv and images.
    metadata_name:
        File name of the metadata CSV (default: "metadata.csv").
    logger:
        Optional logger; if None, a default logger is created.

    Returns
    -------
    DatasetReadyReport
        Structured result with counts, warnings and status.
    """
    if logger is None:
        logger = _setup_logger()

    root = Path(root).expanduser().resolve()
    metadata_path = root / metadata_name

    warnings: list[str] = []
    errors: list[str] = []
    missing_columns: list[str] = []
    num_rows = 0
    num_missing_files = 0
    num_bad_rows = 0
    unique_labels: list[str] = []
    splits: list[str] = []
    exists_metadata = metadata_path.exists()

    logger.info("====================================================================")
    logger.info("[data_ready] Dataset    : %s", dataset)
    logger.info("[data_ready] Root       : %s", root)
    logger.info("[data_ready] Metadata   : %s", metadata_path)
    logger.info("====================================================================")

    if not exists_metadata:
        msg = f"metadata file not found at {metadata_path}"
        logger.error(msg)
        errors.append(msg)
        status = "error"
        return DatasetReadyReport(
            dataset=dataset,
            root=str(root),
            metadata_path=str(metadata_path),
            exists_metadata=False,
            required_columns=list(REQUIRED_COLUMNS),
            missing_columns=missing_columns,
            num_rows=num_rows,
            num_missing_files=num_missing_files,
            num_bad_rows=num_bad_rows,
            unique_labels=unique_labels,
            splits=splits,
            status=status,
            warnings=warnings,
            errors=errors,
        )

    # Load metadata
    try:
        df = pd.read_csv(metadata_path)
    except Exception as exc:  # pragma: no cover - extremely rare
        msg = f"failed to read metadata CSV: {exc}"
        logger.error(msg)
        errors.append(msg)
        return DatasetReadyReport(
            dataset=dataset,
            root=str(root),
            metadata_path=str(metadata_path),
            exists_metadata=True,
            required_columns=list(REQUIRED_COLUMNS),
            missing_columns=list(REQUIRED_COLUMNS),
            num_rows=0,
            num_missing_files=0,
            num_bad_rows=0,
            unique_labels=[],
            splits=[],
            status="error",
            warnings=warnings,
            errors=errors,
        )

    num_rows = int(len(df))
    logger.info("[data_ready] Loaded %d metadata rows.", num_rows)

    # Column check
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        msg = f"metadata is missing required columns: {missing_columns}"
        logger.error(msg)
        errors.append(msg)

    # Only proceed with deeper checks if core columns exist
    if not missing_columns and num_rows > 0:
        df["image_path"] = df["image_path"].astype(str)
        df["label"] = df["label"].astype(str)
        df["split"] = df["split"].astype(str)

        unique_labels = sorted(df["label"].unique().tolist())
        splits = sorted(df["split"].unique().tolist())
        logger.info("[data_ready] Unique labels : %s", unique_labels)
        logger.info("[data_ready] Splits        : %s", splits)

        # Simple heuristic: require at least a train split
        if "train" not in {s.lower() for s in splits}:
            msg = "no 'train' split detected in metadata.splits"
            logger.warning(msg)
            warnings.append(msg)

        # Check that image files exist
        missing_files_indices: list[int] = []
        for idx, rel_path in enumerate(df["image_path"]):
            p = Path(rel_path)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                num_missing_files += 1
                missing_files_indices.append(idx)

        if num_missing_files > 0:
            msg = f"{num_missing_files} image files referenced in metadata do not exist under root."
            logger.warning(msg)
            warnings.append(msg)

        num_bad_rows = num_missing_files

    # Decide status
    if errors:
        status = "error"
    elif num_missing_files > 0 or warnings:
        status = "warning"
    else:
        status = "ok"

    logger.info("[data_ready] Status            : %s", status.upper())
    logger.info("[data_ready] Missing files     : %d", num_missing_files)
    logger.info("[data_ready] Missing columns   : %s", missing_columns)
    logger.info("====================================================================")

    return DatasetReadyReport(
        dataset=dataset,
        root=str(root),
        metadata_path=str(metadata_path),
        exists_metadata=exists_metadata,
        required_columns=list(REQUIRED_COLUMNS),
        missing_columns=missing_columns,
        num_rows=num_rows,
        num_missing_files=num_missing_files,
        num_bad_rows=num_bad_rows,
        unique_labels=unique_labels,
        splits=splits,
        status=status,
        warnings=warnings,
        errors=errors,
    )


def save_report(report: DatasetReadyReport, output_path: Path) -> None:
    """
    Serialize a DatasetReadyReport to JSON.

    Parameters
    ----------
    report:
        Result from run_ready_check().
    output_path:
        Path where JSON will be written (parent dirs are created).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset readiness / validation checks."
    )
    parser.add_argument(
        "--dataset", required=True, help="Logical dataset name (e.g. isic2020)."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder containing metadata.csv and images.",
    )
    parser.add_argument(
        "--metadata-name",
        default="metadata.csv",
        help="Name of the metadata CSV file (default: metadata.csv).",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        default=None,
        help="Optional path to write a JSON report. If omitted, only logs are printed.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero status code if the final status is 'error'.",
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
    logger = _setup_logger(args.verbose)

    root = Path(args.root)
    report = run_ready_check(
        dataset=args.dataset, root=root, metadata_name=args.metadata_name, logger=logger
    )

    if args.output_json is not None:
        save_report(report, Path(args.output_json))
        logger.info("[data_ready] JSON report written to %s", args.output_json)

    if args.fail_on_error and report.status == "error":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
