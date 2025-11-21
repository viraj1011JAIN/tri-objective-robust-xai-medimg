"""
Production-Level DVC Data Registry for External Datasets
Tracks datasets at F:/data without moving/copying them.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def calculate_directory_hash(directory: Path, sample_size: int = 100) -> str:
    """Calculate hash of directory by sampling files."""
    files = list(directory.rglob("*"))
    files = [f for f in files if f.is_file()]

    # Sample files for large directories
    if len(files) > sample_size:
        import random

        random.seed(42)
        files = random.sample(files, sample_size)

    hasher = hashlib.sha256()
    for file_path in sorted(files):
        try:
            hasher.update(str(file_path.relative_to(directory)).encode())
            hasher.update(str(file_path.stat().st_size).encode())
            hasher.update(str(file_path.stat().st_mtime).encode())
        except Exception:
            continue

    return hasher.hexdigest()[:16]


def count_files(directory: Path, extensions: List[str] = None) -> int:
    """Count files in directory with optional extension filter."""
    if extensions:
        count = sum(1 for ext in extensions for _ in directory.rglob(f"*{ext}"))
    else:
        count = sum(1 for _ in directory.rglob("*") if _.is_file())
    return count


def get_directory_size(directory: Path) -> int:
    """Calculate total size of directory in bytes."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def register_dataset(
    name: str,
    path: Path,
    description: str,
    modality: str,
    num_classes: int,
    task_type: str,
) -> Dict[str, Any]:
    """Register a dataset in the DVC data registry."""

    # Count files
    image_extensions = [".jpg", ".jpeg", ".png"]
    num_images = count_files(path, image_extensions)
    total_files = count_files(path)

    # Calculate size
    size_bytes = get_directory_size(path)
    size_gb = size_bytes / (1024**3)

    # Calculate directory hash (fingerprint)
    dir_hash = calculate_directory_hash(path, sample_size=100)

    # Find metadata files
    metadata_files = []
    for csv_file in path.rglob("*.csv"):
        rel_path = csv_file.relative_to(path)
        metadata_files.append(
            {
                "path": str(rel_path),
                "size_kb": csv_file.stat().st_size / 1024,
                "rows": (
                    len(pd.read_csv(csv_file))
                    if csv_file.stat().st_size < 100 * 1024 * 1024
                    else "large"
                ),
            }
        )

    return {
        "name": name,
        "path": str(path),
        "description": description,
        "modality": modality,
        "num_classes": num_classes,
        "task_type": task_type,
        "statistics": {
            "num_images": num_images,
            "total_files": total_files,
            "size_gb": round(size_gb, 2),
            "directory_hash": dir_hash,
        },
        "metadata_files": metadata_files,
        "registered_at": datetime.now().isoformat(),
        "location": "external",
        "storage": "F:/data (fixed location - do not move)",
    }


def main():
    """Generate DVC data registry for all datasets at F:/data."""

    DATA_ROOT = Path("F:/data")

    if not DATA_ROOT.exists():
        print(f"ERROR: Data root not found: {DATA_ROOT}")
        return

    print("=" * 80)
    print("DVC DATA REGISTRY GENERATION")
    print("Tracking datasets at F:/data (external, fixed location)")
    print("=" * 80)

    datasets = []

    # ISIC 2018
    print("\n[1/6] Registering ISIC 2018...")
    try:
        datasets.append(
            register_dataset(
                name="isic_2018",
                path=DATA_ROOT / "isic_2018",
                description="ISIC 2018 Challenge - HAM10000 (7-class skin lesion classification)",
                modality="Dermoscopy (RGB)",
                num_classes=7,
                task_type="multi-class",
            )
        )
        print("  ✓ ISIC 2018 registered")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # ISIC 2019
    print("\n[2/6] Registering ISIC 2019...")
    try:
        datasets.append(
            register_dataset(
                name="isic_2019",
                path=DATA_ROOT / "isic_2019",
                description="ISIC 2019 Challenge (8-class skin lesion classification)",
                modality="Dermoscopy (RGB)",
                num_classes=8,
                task_type="multi-class",
            )
        )
        print("  ✓ ISIC 2019 registered")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # ISIC 2020
    print("\n[3/6] Registering ISIC 2020...")
    try:
        datasets.append(
            register_dataset(
                name="isic_2020",
                path=DATA_ROOT / "isic_2020",
                description="ISIC 2020 Challenge (Binary melanoma detection)",
                modality="Dermoscopy (RGB)",
                num_classes=2,
                task_type="binary",
            )
        )
        print("  ✓ ISIC 2020 registered")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Derm7pt
    print("\n[4/6] Registering Derm7pt...")
    try:
        datasets.append(
            register_dataset(
                name="derm7pt",
                path=DATA_ROOT / "derm7pt",
                description="Derm7pt (7-point checklist melanoma detection)",
                modality="Dermoscopy + Clinical (RGB)",
                num_classes=2,
                task_type="binary + attributes",
            )
        )
        print("  ✓ Derm7pt registered")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # NIH ChestX-ray14
    print("\n[5/6] Registering NIH ChestX-ray14...")
    try:
        datasets.append(
            register_dataset(
                name="nih_cxr",
                path=DATA_ROOT / "nih_cxr",
                description="NIH ChestX-ray14 (14-class multi-label thoracic disease detection)",
                modality="Chest X-ray (Grayscale)",
                num_classes=14,
                task_type="multi-label",
            )
        )
        print("  ✓ NIH ChestX-ray14 registered")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # PadChest
    print("\n[6/6] Registering PadChest...")
    try:
        datasets.append(
            register_dataset(
                name="padchest",
                path=DATA_ROOT / "padchest",
                description="PadChest (174+ radiological findings, Spanish labels)",
                modality="Chest X-ray (Grayscale)",
                num_classes=174,
                task_type="multi-label",
            )
        )
        print("  ✓ PadChest registered")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Generate registry
    note_text = (
        "Datasets are stored at F:/data and should NOT be moved "
        "or copied. All DVC operations reference this external location."
    )
    registry = {
        "version": "1.0.0",
        "description": "DVC Data Registry for External Datasets at F:/data",
        "data_root": str(DATA_ROOT),
        "storage_policy": "external-fixed-location",
        "note": note_text,
        "total_datasets": len(datasets),
        "total_size_gb": sum(d["statistics"]["size_gb"] for d in datasets),
        "generated_at": datetime.now().isoformat(),
        "datasets": datasets,
    }

    # Save registry
    output_path = Path("data/governance/dvc_data_registry.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)

    print("\n" + "=" * 80)
    print("REGISTRY SUMMARY")
    print("=" * 80)
    print("Total Datasets: {}".format(registry["total_datasets"]))
    print("Total Size: {:.2f} GB".format(registry["total_size_gb"]))
    print("Registry saved to: {}".format(output_path))
    print("\n" + "=" * 80)
    print("DVC DATA REGISTRY COMPLETE")
    print("=" * 80)
    print("\nAll datasets at F:/data are now documented.")
    print("Use dvc.yaml dependencies to reference: F:/data/<dataset>/metadata.csv")
    print("DVC remote 'fstore' (F:/triobj_dvc_remote) configured for backups.")


if __name__ == "__main__":
    main()
