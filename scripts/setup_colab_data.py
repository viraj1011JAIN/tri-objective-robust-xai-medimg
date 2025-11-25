"""
Setup script to create minimal test datasets for Colab testing.

This creates a small mock ISIC 2018 dataset for quick HPO validation.
For full HPO, you need to upload the real datasets or mount Google Drive.

Author: Viraj Pankaj Jain
Date: November 24, 2025
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def create_mock_isic2018(
    root_dir: str = "data/processed/isic2018", num_samples: int = 100
):
    """
    Create a minimal mock ISIC 2018 dataset for testing.

    Args:
        root_dir: Root directory for dataset
        num_samples: Number of samples per split (train/val/test)
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Create image directories
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)

    # ISIC 2018 classes
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    # Create metadata
    metadata = []

    for split in ["train", "val", "test"]:
        n = num_samples if split == "train" else num_samples // 2

        for i in range(n):
            image_id = f"MOCK_ISIC_{split}_{i:04d}"
            image_path = f"images/{split}/{image_id}.jpg"
            label = np.random.choice(classes)

            # Create dummy 224x224 RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(root / image_path)

            metadata.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "label": label,
                    "split": split,
                    "original_image_path": f"mock/{image_id}.jpg",
                }
            )

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(root / "metadata_processed.csv", index=False)

    print(f"✅ Created mock ISIC 2018 dataset:")
    print(f"   Location: {root.absolute()}")
    print(f"   Train: {num_samples} samples")
    print(f"   Val: {num_samples // 2} samples")
    print(f"   Test: {num_samples // 2} samples")
    print(f"   Classes: {classes}")
    print(f"   Total: {len(metadata)} samples")

    return root


def create_mock_derm7pt(
    root_dir: str = "data/processed/derm7pt", num_samples: int = 80
):
    """Create a minimal mock Derm7pt dataset."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Create image directories
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)

    # Binary classification (benign/malignant)
    classes = [0, 1]

    metadata = []

    for split in ["train", "val", "test"]:
        n = num_samples if split == "train" else num_samples // 2

        for i in range(n):
            image_id = f"MOCK_DERM_{split}_{i:04d}"
            image_path = f"images/{split}/{image_id}.jpg"
            label = np.random.choice(classes)

            # Create dummy 224x224 RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(root / image_path)

            metadata.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "label": label,
                    "split": split,
                }
            )

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(root / "metadata_processed.csv", index=False)

    print(f"✅ Created mock Derm7pt dataset:")
    print(f"   Location: {root.absolute()}")
    print(f"   Total: {len(metadata)} samples")


def create_mock_nih_cxr(
    root_dir: str = "data/processed/nih_cxr", num_samples: int = 80
):
    """Create a minimal mock NIH ChestX-ray dataset."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Create image directories
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)

    # Multi-label classes
    all_labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    metadata = []

    for split in ["train", "val", "test"]:
        n = num_samples if split == "train" else num_samples // 2

        for i in range(n):
            image_id = f"MOCK_CXR_{split}_{i:04d}"
            image_path = f"images/{split}/{image_id}.jpg"

            # Random multi-label (1-3 labels)
            num_labels = np.random.randint(1, 4)
            labels = "|".join(np.random.choice(all_labels, num_labels, replace=False))

            # Create dummy 224x224 grayscale->RGB image
            gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img_array = np.stack([gray, gray, gray], axis=-1)
            img = Image.fromarray(img_array)
            img.save(root / image_path)

            metadata.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "labels": labels,
                    "split": split,
                }
            )

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(root / "metadata_processed.csv", index=False)

    print(f"✅ Created mock NIH CXR dataset:")
    print(f"   Location: {root.absolute()}")
    print(f"   Total: {len(metadata)} samples")


def main():
    """Create all mock datasets."""
    print("=" * 80)
    print("CREATING MOCK DATASETS FOR COLAB TESTING")
    print("=" * 80)
    print()
    print("⚠️  WARNING: These are MOCK datasets for infrastructure testing only!")
    print("   For dissertation results, use REAL medical imaging datasets.")
    print()

    # Create mock datasets
    create_mock_isic2018(num_samples=100)
    print()
    create_mock_derm7pt(num_samples=80)
    print()
    create_mock_nih_cxr(num_samples=80)

    print()
    print("=" * 80)
    print("MOCK DATASETS CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Test with:")
    print("  python scripts/run_hpo_medical.py --dataset isic2018 --quick-test")
    print()
    print("⚠️  Remember: Use REAL datasets for actual dissertation work!")


if __name__ == "__main__":
    main()
