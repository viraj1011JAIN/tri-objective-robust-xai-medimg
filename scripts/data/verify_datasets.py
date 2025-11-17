#!/usr/bin/env python
"""
Verify presence and basic integrity of all external datasets.

- Uses DATA_ROOT (env var) or --data-root to locate the data vault.
- Checks that each dataset directory exists and is non-empty.
- Counts total files / images / metadata files.
- Computes a manifest hash (based on relative path + file size) so that
  future runs can detect changes without hashing all bytes.

Results are saved to data/governance/dataset_checksums.json
for reproducibility and dissertation reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".dcm"}
META_EXTENSIONS = {".csv", ".json", ".xlsx", ".tsv"}


@dataclass
class DatasetSummary:
    name: str
    root: str
    exists: bool
    num_files: int
    num_images: int
    num_meta: int
    manifest_hash: str | None
    notes: str = ""


def _iter_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path)
    return files


def _compute_manifest_hash(root: Path, files: List[Path]) -> str:
    """
    Build a SHA256 hash over (relative path, file size).

    This is fast even for 158GB because it hashes metadata, not file contents.
    """
    entries: List[str] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        entries.append(f"{rel}:{size}")
    entries.sort()

    hasher = hashlib.sha256()
    for line in entries:
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def summarise_dataset(name: str, root: Path) -> DatasetSummary:
    if not root.exists():
        return DatasetSummary(
            name=name,
            root=str(root),
            exists=False,
            num_files=0,
            num_images=0,
            num_meta=0,
            manifest_hash=None,
            notes="directory does not exist",
        )

    files = _iter_files(root)
    num_files = len(files)
    num_images = sum(p.suffix.lower() in IMAGE_EXTENSIONS for p in files)
    num_meta = sum(p.suffix.lower() in META_EXTENSIONS for p in files)
    manifest_hash = _compute_manifest_hash(root, files)

    return DatasetSummary(
        name=name,
        root=str(root),
        exists=True,
        num_files=num_files,
        num_images=num_images,
        num_meta=num_meta,
        manifest_hash=manifest_hash,
    )


def verify_all_datasets(data_root: Path) -> Dict[str, DatasetSummary]:
    """
    Expected layout under data_root:

        derm7pt/
        isic_2018/
        isic_2019/
        isic_2020/
        nih_cxr/
        padchest/
    """
    datasets = {
        "derm7pt": "derm7pt",
        "isic_2018": "isic_2018",
        "isic_2019": "isic_2019",
        "isic_2020": "isic_2020",
        "nih_cxr": "nih_cxr",
        "padchest": "padchest",
    }

    summaries: Dict[str, DatasetSummary] = {}
    for name, subdir in datasets.items():
        root = data_root / subdir
        summaries[name] = summarise_dataset(name, root)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify external datasets and write a checksum manifest."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override DATA_ROOT (default: use DATA_ROOT env var).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/governance/dataset_checksums.json",
        help="Output JSON path (relative to repo root).",
    )
    args = parser.parse_args()

    if args.data_root is not None:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        env_root = os.environ.get("DATA_ROOT")
        if not env_root:
            raise SystemExit(
                "No data root provided. Set DATA_ROOT or pass --data-root PATH."
            )
        data_root = Path(env_root).expanduser().resolve()

    print(f"[verify_datasets] Using data root: {data_root}")

    summaries = verify_all_datasets(data_root)

    print("\nDataset overview:")
    for name, summary in summaries.items():
        status = "OK" if summary.exists and summary.num_files > 0 else "MISSING"
        print(
            f"- {name:9s} | status={status:7s} | "
            f"files={summary.num_files:7d} | "
            f"images={summary.num_images:7d} | "
            f"meta={summary.num_meta:5d}"
        )

    # Save manifest under data/governance/
    project_root = Path(__file__).resolve().parents[2]
    out_path = (project_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "datasets": {name: asdict(summary) for name, summary in summaries.items()},
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote manifest to {out_path}")


if __name__ == "__main__":
    main()
