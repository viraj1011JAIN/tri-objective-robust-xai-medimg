# scripts/data/build_concept_bank.py
"""
Build a concept bank from a metadata CSV for any supported dataset.

This script is intentionally simple and robust. It is designed to be used
in the data pipeline (e.g., via DVC) *after* preprocessing has produced a
metadata CSV such as:

    -/content/drive/MyDrive/data/isic_2020/metadata.csv
    - data/processed/isic2020/metadata_processed.csv

Expected columns in the CSV:
    - image_id  (string)
    - label     (string / categorical)
    - split     (optional but recommended; e.g., train / val / test)

Output:
    A JSON "concept bank" with:
        - one concept per unique label
        - dataset-specific high-level groups (e.g. melanoma_like vs benign_like)
        - label_to_idx mapping
        - overall label counts
        - split-wise label counts

Typical usage (single dataset):
    python -m scripts.data.build_concept_bank \
        --dataset isic2020 \
        --metadata-csv data/processed/isic2020/metadata_processed.csv \
        --output-path data/concepts/isic2020_concept_bank.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = (
    "isic2018",
    "isic2019",
    "isic2020",
    "derm7pt",
    "nih_cxr",
    "padchest",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Concept:
    """
    A single concept: usually all images that share a label.
    """

    name: str  # e.g. "label::MEL", "label::UNK"
    type: str  # e.g. "label_based", "group"
    description: str
    label: Optional[str]
    image_ids: List[str]


@dataclass
class ConceptBank:
    """
    Concept bank for a single dataset.
    """

    dataset: str
    source_metadata: str
    created_at_utc: str
    concepts: List[Concept]
    groups: Dict[str, List[str]]  # group_name -> list of concept names
    label_to_idx: Dict[str, int]
    overall_label_counts: Dict[str, int]
    split_label_counts: Dict[str, Dict[str, int]]  # split -> {label: count}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(verbosity: int = 1) -> logging.Logger:
    logger = logging.getLogger("build_concept_bank")
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
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concept bank JSON from a metadata CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name (used for dataset-specific grouping rules).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="Path to metadata CSV file (raw or processed).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSON path for the concept bank.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase logging verbosity (can be repeated).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_metadata(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    logger.info("Loading metadata from %s", path)
    df = pd.read_csv(path)

    # Required columns
    required_cols = {"image_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV missing required columns {missing}. "
            f"Required: {required_cols}"
        )

    if "split" not in df.columns:
        logger.warning(
            "Column 'split' not found in metadata; "
            "split-wise label counts will be empty."
        )

    logger.info(
        "Loaded %d rows, labels=%s, splits=%s",
        len(df),
        sorted(df["label"].astype(str).unique().tolist()),
        (
            sorted(df["split"].astype(str).unique().tolist())
            if "split" in df.columns
            else []
        ),
    )
    return df


def build_label_concepts(df: pd.DataFrame) -> List[Concept]:
    """
    Build one concept per unique label.
    """
    concepts: List[Concept] = []

    labels = sorted(df["label"].astype(str).unique().tolist())
    for label in labels:
        ids = df.loc[df["label"].astype(str) == label, "image_id"].astype(str).tolist()
        concepts.append(
            Concept(
                name=f"label::{label}",
                type="label_based",
                description=f"All samples with label == '{label}'.",
                label=label,
                image_ids=ids,
            )
        )

    return concepts


def build_isic_like_groups(concepts: List[Concept]) -> Dict[str, List[str]]:
    """
    For ISIC-style skin lesion datasets (ISIC 2018/2019/2020, Derm7pt),
    construct coarse-grained groups useful for TCAV:

    - melanoma_like: labels containing 'mel' (case-insensitive) OR 'target'
    - benign_like  : everything else
    """
    melanoma_concepts: List[str] = []
    benign_concepts: List[str] = []

    for c in concepts:
        if c.label is None:
            continue
        label_lower = c.label.lower()
        if "mel" in label_lower or c.label == "target":
            melanoma_concepts.append(c.name)
        else:
            benign_concepts.append(c.name)

    groups: Dict[str, List[str]] = {}
    if melanoma_concepts:
        groups["melanoma_like"] = melanoma_concepts
    if benign_concepts:
        groups["benign_like"] = benign_concepts

    return groups


def build_cxr_groups(concepts: List[Concept]) -> Dict[str, List[str]]:
    """
    For chest X-ray datasets (NIH CXR, PadChest), build simple super concepts:

    - normal_like   : labels suggesting "no findings" or "normal"
    - pathology_like: everything else

    This is heuristic and purely string-based, but it is transparent and
    easy to extend later if you harmonise label names.
    """
    normal_concepts: List[str] = []
    pathology_concepts: List[str] = []

    normal_keywords = ("normal", "no finding", "no_finding", "healthy", "control")

    for c in concepts:
        if c.label is None:
            continue
        label_lower = c.label.lower()
        if any(kw in label_lower for kw in normal_keywords):
            normal_concepts.append(c.name)
        else:
            pathology_concepts.append(c.name)

    groups: Dict[str, List[str]] = {}
    if normal_concepts:
        groups["normal_like"] = normal_concepts
    if pathology_concepts:
        groups["pathology_like"] = pathology_concepts

    return groups


def build_generic_groups(concepts: List[Concept]) -> Dict[str, List[str]]:
    """
    Placeholder for datasets where we do not yet define high-level groups.
    """
    return {}


def build_groups_for_dataset(
    dataset: str, concepts: List[Concept]
) -> Dict[str, List[str]]:
    """
    Dispatch to dataset-specific grouping rules.
    """
    if dataset.startswith("isic") or dataset == "derm7pt":
        return build_isic_like_groups(concepts)
    if dataset in ("nih_cxr", "padchest"):
        return build_cxr_groups(concepts)
    return build_generic_groups(concepts)


def compute_label_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Compute overall label counts and split-wise label counts.

    Returns a dict:
        {
            "overall_label_counts": {label: count, ...},
            "split_label_counts": {
                split: {label: count, ...},
                ...
            }
        }
    """
    # Overall
    overall_series = df["label"].astype(str).value_counts()
    overall_label_counts = {str(k): int(v) for k, v in overall_series.items()}

    split_label_counts: Dict[str, Dict[str, int]] = {}

    if "split" in df.columns:
        # Ensure we include all labels for each split
        labels = sorted(df["label"].astype(str).unique().tolist())
        for split in sorted(df["split"].astype(str).unique().tolist()):
            df_split = df[df["split"].astype(str) == split]
            vc = df_split["label"].astype(str).value_counts()
            split_label_counts[split] = {
                label: int(vc.get(label, 0)) for label in labels
            }

    return {
        "overall_label_counts": overall_label_counts,
        "split_label_counts": split_label_counts,
    }


def build_concept_bank(
    dataset: str, meta: pd.DataFrame, source_path: Path
) -> ConceptBank:
    """
    High-level constructor: takes a metadata DataFrame and builds the
    ConceptBank dataclass.
    """
    # 1) Per-label concepts
    concepts = build_label_concepts(meta)

    # 2) Groups (dataset-specific)
    groups = build_groups_for_dataset(dataset, concepts)

    # 3) Label index mapping
    labels_sorted = sorted(meta["label"].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

    # 4) Label statistics
    stats = compute_label_stats(meta)

    cb = ConceptBank(
        dataset=dataset,
        source_metadata=str(source_path),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        concepts=concepts,
        groups=groups,
        label_to_idx=label_to_idx,
        overall_label_counts=stats["overall_label_counts"],
        split_label_counts=stats["split_label_counts"],
    )
    return cb


def save_concept_bank(
    cb: ConceptBank, output_path: Path, logger: logging.Logger
) -> None:
    """
    Serialize the concept bank to JSON.
    """
    obj = {
        "dataset": cb.dataset,
        "source_metadata": cb.source_metadata,
        "created_at_utc": cb.created_at_utc,
        "label_to_idx": cb.label_to_idx,
        "overall_label_counts": cb.overall_label_counts,
        "split_label_counts": cb.split_label_counts,
        "concepts": [asdict(c) for c in cb.concepts],
        "groups": cb.groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    logger.info("Concept bank written to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logger = setup_logger(args.verbose)

    dataset = args.dataset
    meta_path = Path(args.metadata_csv).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    logger.info("Building concept bank")
    logger.info("  Dataset      : %s", dataset)
    logger.info("  Metadata CSV : %s", meta_path)
    logger.info("  Output path  : %s", output_path)

    meta = load_metadata(meta_path, logger)
    cb = build_concept_bank(dataset, meta, meta_path)
    save_concept_bank(cb, output_path, logger)

    logger.info("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
