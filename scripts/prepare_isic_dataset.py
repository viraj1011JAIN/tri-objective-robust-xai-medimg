#!/usr/bin/env python
"""
ISIC 2018 Dataset Preparation Script
=====================================

This script processes the raw ISIC 2018 Challenge data and creates:
1. A unified metadata_processed.csv with train/val/test splits
2. Proper directory structure for the framework's ISICDataset class

Expected input structure:
    data/raw/isic2018/
        ISIC2018_Task3_Training_Input/     # Training images
        ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv
        ISIC2018_Task3_Test_Input/         # Test images (no labels)

Output structure:
    data/processed/isic2018/
        metadata_processed.csv
        images/                            # Symlinks or copies of images

The 7 classes in ISIC 2018:
    MEL  - Melanoma (0)
    NV   - Melanocytic nevus (1)
    BCC  - Basal cell carcinoma (2)
    AKIEC - Actinic keratosis (3)
    BKL  - Benign keratosis (4)
    DF   - Dermatofibroma (5)
    VASC - Vascular lesion (6)
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Class mapping for ISIC 2018 Task 3
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """
    Load and parse the ISIC 2018 ground truth CSV.

    Args:
        gt_path: Path to ISIC2018_Task3_Training_GroundTruth.csv

    Returns:
        DataFrame with columns: image_id, image_path, label, label_name
    """
    df = pd.read_csv(gt_path)

    # Extract image ID from filepath column
    df["image_id"] = df["filepath"].apply(
        lambda x: Path(x).stem if pd.notna(x) else None
    )

    # Determine label from one-hot encoded columns
    label_cols = CLASS_NAMES

    def get_label(row):
        for idx, col in enumerate(label_cols):
            if col in row and row[col] == 1.0:
                return col
        return None

    df["label"] = df.apply(get_label, axis=1)
    df["label_idx"] = df["label"].map(CLASS_TO_IDX)

    # Keep only necessary columns
    result = df[["image_id", "label", "label_idx"]].copy()
    result = result.dropna(subset=["label"])

    return result


def create_stratified_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create stratified train/val/test splits.

    Args:
        df: DataFrame with image_id and label columns
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with additional 'split' column
    """
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_ratio / (1 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    # Assign splits
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Combine
    result = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return result


def prepare_isic_dataset(
    raw_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_images: bool = False,
    random_state: int = 42,
) -> None:
    """
    Prepare the ISIC 2018 dataset for training.

    Args:
        raw_dir: Path to data/raw/isic2018/
        output_dir: Path to data/processed/isic2018/
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        copy_images: If True, copy images; otherwise create relative paths
        random_state: Random seed
    """
    print("=" * 60)
    print("ISIC 2018 Dataset Preparation")
    print("=" * 60)

    # Validate input paths
    train_images_dir = raw_dir / "ISIC2018_Task3_Training_Input"
    gt_csv = (
        raw_dir
        / "ISIC2018_Task3_Training_GroundTruth"
        / "ISIC2018_Task3_Training_GroundTruth.csv"
    )

    if not train_images_dir.exists():
        raise FileNotFoundError(
            f"Training images directory not found: {train_images_dir}"
        )
    if not gt_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_csv}")

    print(f"\n[1/4] Loading ground truth from: {gt_csv}")
    df = load_ground_truth(gt_csv)
    print(f"      Loaded {len(df)} labeled images")

    # Show class distribution
    print("\n      Class distribution:")
    for label in CLASS_NAMES:
        count = (df["label"] == label).sum()
        print(f"        {label}: {count:5d} ({100*count/len(df):5.1f}%)")

    print(
        f"\n[2/4] Creating stratified splits (val={val_ratio:.0%}, test={test_ratio:.0%})"
    )
    df = create_stratified_splits(df, val_ratio, test_ratio, random_state)

    # Show split distribution
    print("\n      Split distribution:")
    for split in ["train", "val", "test"]:
        count = (df["split"] == split).sum()
        print(f"        {split:5s}: {count:5d} ({100*count/len(df):5.1f}%)")

    print(f"\n[3/4] Setting up output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create image paths (relative to output_dir for portability)
    images_dir = output_dir / "images"

    if copy_images:
        print("      Copying images...")
        images_dir.mkdir(exist_ok=True)

        for idx, row in df.iterrows():
            src = train_images_dir / f"{row['image_id']}.jpg"
            dst = images_dir / f"{row['image_id']}.jpg"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        df["image_path"] = df["image_id"].apply(lambda x: f"images/{x}.jpg")
    else:
        # Use relative path back to raw data
        rel_path = train_images_dir.resolve()
        df["image_path"] = df["image_id"].apply(lambda x: str(rel_path / f"{x}.jpg"))

    print(f"\n[4/4] Saving metadata to: {output_dir / 'metadata_processed.csv'}")

    # Reorder columns for clarity
    df = df[["image_id", "image_path", "label", "label_idx", "split"]]
    df.to_csv(output_dir / "metadata_processed.csv", index=False)

    # Verify the output
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check a few image paths exist
    sample_paths = df["image_path"].head(5).tolist()
    all_exist = True
    for p in sample_paths:
        path = Path(p) if Path(p).is_absolute() else output_dir / p
        if not path.exists():
            print(f"  [WARN] Image not found: {path}")
            all_exist = False

    if all_exist:
        print("  [OK] Sample image paths verified")

    print(f"\n  Output files:")
    print(f"    - {output_dir / 'metadata_processed.csv'}")
    if copy_images:
        print(f"    - {images_dir}/ ({len(list(images_dir.glob('*.jpg')))} images)")

    print("\n" + "=" * 60)
    print("DONE! Dataset is ready for training.")
    print("=" * 60)

    # Print usage example
    print(
        """
USAGE IN NOTEBOOK:
------------------
# Local usage
from src.datasets.isic import ISICDataset
from src.datasets.base_dataset import Split

dataset = ISICDataset(
    root='data/processed/isic2018',
    split=Split.TRAIN,
    transform=train_transform
)

# Google Colab usage (after uploading to Drive)
dataset = ISICDataset(
    root='/content/drive/MyDrive/dissertation/data/processed/isic2018',
    split=Split.TRAIN,
    transform=train_transform
)
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ISIC 2018 dataset for training"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/isic2018"),
        help="Path to raw ISIC 2018 data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/isic2018"),
        help="Path to output processed data directory",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output directory (default: use relative paths)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    prepare_isic_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_images=args.copy_images,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
