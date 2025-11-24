#!/usr/bin/env python
"""
Create metadata.csv for ISIC2020 dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_isic2020_metadata():
    print("=" * 70)
    print("CREATING ISIC2020 METADATA")
    print("=" * 70)

    root = Path(r"F:\data\isic_2020")

    # --- STEP 1: CSVs --------------------------------------------------------
    print("\n📋 STEP 1: Analyzing existing CSV files")
    print("-" * 60)
    csv_files = list(root.glob("*.csv"))
    if csv_files:
        print(f"Found CSV files: {[f.name for f in csv_files]}")
    else:
        print("No CSV files found in root directory")

    # Use train.csv as main labels file (you already checked this in the nb)
    train_df = pd.read_csv(root / "train.csv")
    print(f"Loaded train.csv: {len(train_df)} rows")
    print(f"Columns: {list(train_df.columns)}")

    # --- STEP 2: image dirs --------------------------------------------------
    print("\n🖼️ STEP 2: Finding image files")
    print("-" * 60)
    image_dirs = []
    for item in root.rglob("*"):
        if item.is_dir():
            jpg_count = len(list(item.glob("*.jpg")))
            png_count = len(list(item.glob("*.png")))
            if jpg_count > 0 or png_count > 0:
                print(f"Found: {item.relative_to(root)}")
                print(f"   JPG: {jpg_count}, PNG: {png_count}")
                image_dirs.append(item)
    if not image_dirs:
        raise RuntimeError(
            "No image directories found under /content/drive/MyDrive/data\\isic_2020"
        )

    # --- STEP 3: identify id/target cols ------------------------------------
    print("\n🏷️ STEP 3: Identifying label format")
    print("-" * 60)
    id_col = None
    for col in ["image_name", "isic_id", "image", "image_id"]:
        if col in train_df.columns:
            id_col = col
            break

    target_col = None
    for col in ["target", "benign_malignant", "diagnosis", "label"]:
        if col in train_df.columns:
            target_col = col
            break

    print(f"Image ID column: {id_col}")
    print(f"Target column: {target_col}")

    # --- STEP 4: build image index ------------------------------------------
    print("\n🖼️ STEP 5: Building image index")
    print("-" * 60)
    image_index = {}
    for img_dir in image_dirs:
        for ext in ("*.jpg", "*.png"):
            for img_file in img_dir.glob(ext):
                img_id = img_file.stem
                rel_path = img_file.relative_to(root)
                image_index[img_id] = str(rel_path)
    print(f"Indexed {len(image_index)} images")

    # --- STEP 5: create metadata --------------------------------------------
    print("\n📝 STEP 6: Creating metadata DataFrame")
    print("-" * 60)
    records = []
    missing = 0

    for _, row in train_df.iterrows():
        img_id = str(row[id_col])

        # resolve path
        base = img_id.replace(".jpg", "").replace(".png", "")
        if base in image_index:
            img_path = image_index[base]
        else:
            missing += 1
            continue

        # binary label
        if target_col:
            target = row[target_col]
            label = "target" if int(target) == 1 else "UNK"
        else:
            label = "UNK"

        records.append(
            {
                "image_id": base,
                "image_path": img_path,
                "label": label,
                "split": "train",  # updated below
            }
        )

    if missing > 0:
        print(f"⚠️ {missing} images not found in directories")

    metadata = pd.DataFrame(records)
    print(f"Created metadata with {len(metadata)} samples")

    # --- STEP 6: train/test split -------------------------------------------
    print("\n✂️ STEP 7: Creating train/test splits")
    print("-" * 60)
    train_idx, test_idx = train_test_split(
        metadata.index,
        test_size=0.1,
        stratify=metadata["label"],
        random_state=42,
    )
    metadata.loc[test_idx, "split"] = "test"
    print("Split distribution:")
    print(metadata["split"].value_counts())

    # --- STEP 7: save --------------------------------------------------------
    print("\n💾 STEP 8: Saving metadata.csv")
    print("-" * 60)
    output_path = root / "metadata.csv"
    metadata.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
    print(f"   Total samples: {len(metadata)}")


def main() -> None:
    create_isic2020_metadata()


if __name__ == "__main__":
    main()
