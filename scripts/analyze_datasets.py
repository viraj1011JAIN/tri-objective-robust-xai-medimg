"""
Production-Level Dataset Analysis Script for Phase 2.1
Analyzes all medical imaging datasets at F:/data
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path("F:/data")


def analyze_image_statistics(
    image_paths: List[Path], sample_size: int = 100
) -> Dict[str, Any]:
    """Analyze image statistics from a sample."""
    sample_paths = (
        np.random.choice(image_paths, min(sample_size, len(image_paths)), replace=False)
        if len(image_paths) > sample_size
        else image_paths
    )

    widths, heights, channels, file_sizes = [], [], [], []

    for img_path in sample_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                # Determine channels
                if img.mode == "L":
                    channels.append(1)
                elif img.mode == "RGB":
                    channels.append(3)
                elif img.mode == "RGBA":
                    channels.append(4)
                else:
                    channels.append(len(img.getbands()))

                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return {
        "width": {
            "min": min(widths) if widths else 0,
            "max": max(widths) if widths else 0,
            "mean": np.mean(widths) if widths else 0,
            "std": np.std(widths) if widths else 0,
        },
        "height": {
            "min": min(heights) if heights else 0,
            "max": max(heights) if heights else 0,
            "mean": np.mean(heights) if heights else 0,
            "std": np.std(heights) if heights else 0,
        },
        "channels": dict(Counter(channels)),
        "file_size_kb": {
            "min": min(file_sizes) if file_sizes else 0,
            "max": max(file_sizes) if file_sizes else 0,
            "mean": np.mean(file_sizes) if file_sizes else 0,
            "std": np.std(file_sizes) if file_sizes else 0,
        },
        "samples_analyzed": len(widths),
    }


def analyze_isic_2018() -> Dict[str, Any]:
    """Analyze ISIC 2018 dataset (HAM10000)."""
    print("\n" + "=" * 80)
    print("ANALYZING ISIC 2018 (HAM10000 - Task 3: Lesion Diagnosis)")
    print("=" * 80)

    dataset_path = DATA_ROOT / "isic_2018"

    # Load ground truth CSVs
    train_gt = pd.read_csv(
        dataset_path
        / "ISIC2018_Task3_Training_GroundTruth"
        / "ISIC2018_Task3_Training_GroundTruth.csv"
    )
    val_gt = pd.read_csv(
        dataset_path
        / "ISIC2018_Task3_Validation_GroundTruth"
        / "ISIC2018_Task3_Validation_GroundTruth.csv"
    )

    # Class names
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    class_names_full = {
        "MEL": "Melanoma",
        "NV": "Melanocytic Nevus",
        "BCC": "Basal Cell Carcinoma",
        "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
        "BKL": "Benign Keratosis",
        "DF": "Dermatofibroma",
        "VASC": "Vascular Lesion",
    }

    # Count samples
    train_images_dir = dataset_path / "ISIC2018_Task3_Training_Input"
    val_images_dir = dataset_path / "ISIC2018_Task3_Validation_Input"

    train_images = list(train_images_dir.glob("*.jpg"))
    val_images = list(val_images_dir.glob("*.jpg"))

    # Class distribution
    train_dist = {cls: int(train_gt[cls].sum()) for cls in classes}
    val_dist = {cls: int(val_gt[cls].sum()) for cls in classes}

    # Image statistics
    train_img_stats = analyze_image_statistics(train_images, sample_size=200)
    val_img_stats = analyze_image_statistics(val_images, sample_size=100)

    analysis = {
        "dataset_name": "ISIC 2018 (HAM10000)",
        "task": "Skin Lesion Classification (7 classes)",
        "modality": "Dermoscopy (RGB)",
        "num_classes": 7,
        "class_names": class_names_full,
        "task_type": "Multi-class Classification",
        "splits": {
            "train": {
                "num_samples": len(train_images),
                "num_csv_entries": len(train_gt),
                "class_distribution": train_dist,
                "class_percentages": {
                    cls: f"{(count/len(train_gt)*100):.2f}%"
                    for cls, count in train_dist.items()
                },
                "image_statistics": train_img_stats,
            },
            "validation": {
                "num_samples": len(val_images),
                "num_csv_entries": len(val_gt),
                "class_distribution": val_dist,
                "class_percentages": {
                    cls: f"{(count/len(val_gt)*100):.2f}%"
                    for cls, count in val_dist.items()
                },
                "image_statistics": val_img_stats,
            },
        },
        "total_samples": len(train_images) + len(val_images),
        "class_balance": "Highly imbalanced (NV dominant: 67%)",
        "data_format": {
            "images": "JPEG",
            "labels": "CSV (one-hot encoded)",
        },
        "image_naming": "ISIC_XXXXXXX.jpg",
        "metadata_files": [
            "ISIC2018_Task3_Training_GroundTruth.csv",
            "ISIC2018_Task3_Validation_GroundTruth.csv",
        ],
        "license": "CC0 1.0 Universal (Public Domain)",
        "source": "https://challenge.isic-archive.com/data/#2018",
    }

    return analysis


def analyze_isic_2019() -> Dict[str, Any]:
    """Analyze ISIC 2019 dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING ISIC 2019 (8-Class Skin Lesion Classification)")
    print("=" * 80)

    dataset_path = DATA_ROOT / "isic_2019"

    # Load metadata
    train_csv = pd.read_csv(dataset_path / "train.csv")

    # Count images
    train_images_dir = dataset_path / "train-image"
    train_images = list(train_images_dir.glob("*.jpg"))

    # Get class distribution
    class_distribution = train_csv["target"].value_counts().to_dict()

    # Image statistics
    img_stats = analyze_image_statistics(train_images, sample_size=200)

    analysis = {
        "dataset_name": "ISIC 2019",
        "task": "Skin Lesion Classification (8 classes)",
        "modality": "Dermoscopy (RGB)",
        "num_classes": 8,
        "class_names": {
            "MEL": "Melanoma",
            "NV": "Melanocytic Nevus",
            "BCC": "Basal Cell Carcinoma",
            "AK": "Actinic Keratosis",
            "BKL": "Benign Keratosis",
            "DF": "Dermatofibroma",
            "VASC": "Vascular Lesion",
            "SCC": "Squamous Cell Carcinoma",
        },
        "task_type": "Multi-class Classification",
        "splits": {
            "train": {
                "num_samples": len(train_images),
                "num_csv_entries": len(train_csv),
                "class_distribution": class_distribution,
                "image_statistics": img_stats,
            }
        },
        "total_samples": len(train_images),
        "class_balance": "Imbalanced (NV dominant)",
        "data_format": {
            "images": "JPEG",
            "labels": "CSV (integer labels)",
        },
        "image_naming": "ISIC_XXXXXXXX.jpg",
        "metadata_files": ["train.csv", "metadata.csv"],
        "license": "CC0 1.0 Universal (Public Domain)",
        "source": "https://challenge.isic-archive.com/data/#2019",
    }

    return analysis


def analyze_isic_2020() -> Dict[str, Any]:
    """Analyze ISIC 2020 dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING ISIC 2020 (Binary: Benign vs Malignant)")
    print("=" * 80)

    dataset_path = DATA_ROOT / "isic_2020"

    # Load metadata
    train_csv = pd.read_csv(dataset_path / "train.csv")

    # Count images
    train_images_dir = dataset_path / "train-image"
    train_images = list(train_images_dir.glob("*.jpg"))

    # Get class distribution
    class_distribution = train_csv["target"].value_counts().to_dict()

    # Image statistics
    img_stats = analyze_image_statistics(train_images, sample_size=200)

    analysis = {
        "dataset_name": "ISIC 2020",
        "task": "Melanoma Detection (Binary Classification)",
        "modality": "Dermoscopy (RGB)",
        "num_classes": 2,
        "class_names": {"0": "Benign", "1": "Malignant (Melanoma)"},
        "task_type": "Binary Classification",
        "splits": {
            "train": {
                "num_samples": len(train_images),
                "num_csv_entries": len(train_csv),
                "class_distribution": class_distribution,
                "class_percentages": {
                    cls: f"{(count/len(train_csv)*100):.2f}%"
                    for cls, count in class_distribution.items()
                },
                "image_statistics": img_stats,
            }
        },
        "total_samples": len(train_images),
        "class_balance": "Highly imbalanced (~98% benign, ~2% malignant)",
        "data_format": {
            "images": "JPEG",
            "labels": "CSV (binary 0/1)",
        },
        "image_naming": "ISIC_XXXXXXXX.jpg",
        "metadata_files": ["train.csv", "train-metadata.csv"],
        "license": "CC0 1.0 Universal (Public Domain)",
        "source": "https://challenge.isic-archive.com/data/#2020",
    }

    return analysis


def analyze_derm7pt() -> Dict[str, Any]:
    """Analyze Derm7pt dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING DERM7PT (7-Point Checklist)")
    print("=" * 80)

    dataset_path = DATA_ROOT / "derm7pt"

    # Load metadata
    meta_csv = pd.read_csv(dataset_path / "meta" / "meta.csv")

    # Count images
    images_dir = dataset_path / "images"
    images = list(images_dir.glob("**/*.jpg"))  # Recursive search

    # Get diagnosis distribution
    diagnosis_dist = meta_csv["diagnosis"].value_counts().to_dict()

    # Get 7-point criteria columns
    criteria = [
        "pigment_network",
        "streaks",
        "pigmentation",
        "regression_structures",
        "dots_and_globules",
        "blue_whitish_veil",
        "vascular_structures",
    ]

    # Image statistics
    img_stats = analyze_image_statistics(images[:200], sample_size=200)

    analysis = {
        "dataset_name": "Derm7pt",
        "task": "Melanoma Detection with 7-Point Checklist",
        "modality": "Dermoscopy (RGB) + Clinical (RGB)",
        "num_classes": 2,
        "class_names": {"benign": "Benign", "malignant": "Malignant (Melanoma)"},
        "task_type": "Binary Classification + Attribute Prediction",
        "seven_point_criteria": criteria,
        "splits": {
            "total": {
                "num_samples": len(images),
                "num_csv_entries": len(meta_csv),
                "diagnosis_distribution": diagnosis_dist,
                "image_statistics": img_stats,
            }
        },
        "total_samples": len(images),
        "unique_patients": int(meta_csv["case_num"].nunique())
        if "case_num" in meta_csv.columns
        else "Unknown",
        "image_types": ["dermoscopy", "clinical"],
        "class_balance": "Check diagnosis column",
        "data_format": {
            "images": "JPEG (multiple per case)",
            "labels": "CSV (diagnosis + 7-point checklist attributes)",
        },
        "metadata_files": ["meta.csv", "train_indexes.csv", "valid_indexes.csv"],
        "license": "Academic Use",
        "source": "http://derm.cs.sfu.ca/Welcome.html",
    }

    return analysis


def analyze_nih_cxr() -> Dict[str, Any]:
    """Analyze NIH ChestX-ray14 dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING NIH ChestX-ray14 (Multi-label Chest X-ray)")
    print("=" * 80)

    dataset_path = DATA_ROOT / "nih_cxr"

    # Load metadata
    data_entry_csv = pd.read_csv(dataset_path / "Data_Entry_2017.csv")

    # Count images
    image_dirs = [
        dataset_path / f"images_{str(i).zfill(3)}" for i in range(1, 13)
    ]
    total_images = sum(len(list(d.glob("*.png"))) for d in image_dirs if d.exists())

    # Parse labels (pipe-separated multi-label)
    labels = data_entry_csv["Finding Labels"].str.split("|", expand=False)
    all_labels = [label for sublist in labels for label in sublist]
    label_distribution = Counter(all_labels)

    # Get unique diseases (excluding "No Finding")
    diseases = sorted([d for d in label_distribution.keys() if d != "No Finding"])

    # Image statistics (sample from first directory)
    sample_images = list((dataset_path / "images_001").glob("*.png"))[:100]
    img_stats = analyze_image_statistics(sample_images, sample_size=100)

    analysis = {
        "dataset_name": "NIH ChestX-ray14",
        "task": "Thoracic Disease Detection (14 classes, Multi-label)",
        "modality": "Chest X-ray (Grayscale)",
        "num_classes": 14,
        "class_names": diseases
        + ["No Finding"],  # 14 diseases + 1 no-finding class
        "diseases": diseases,
        "task_type": "Multi-label Classification",
        "splits": {
            "total": {
                "num_samples": total_images,
                "num_csv_entries": len(data_entry_csv),
                "label_distribution": dict(label_distribution),
                "image_statistics": img_stats,
            }
        },
        "total_samples": total_images,
        "unique_patients": int(data_entry_csv["Patient ID"].nunique()),
        "class_balance": "Highly imbalanced ('No Finding' dominant)",
        "data_format": {
            "images": "PNG (1024x1024 originally)",
            "labels": "CSV (pipe-separated multi-label strings)",
        },
        "image_naming": "XXXXXXXX_XXX.png",
        "metadata_files": ["Data_Entry_2017.csv", "BBox_List_2017.csv"],
        "image_directories": 12,
        "bbox_available": True,
        "license": "CC0 1.0 Universal (Public Domain)",
        "source": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
    }

    return analysis


def analyze_padchest() -> Dict[str, Any]:
    """Analyze PadChest dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING PADCHEST (Spanish Chest X-ray Dataset)")
    print("=" * 80)

    dataset_path = DATA_ROOT / "padchest"

    # Count images
    images_dir = dataset_path / "images"
    images = list(images_dir.glob("*.png"))

    # Load metadata if available
    metadata_path = dataset_path / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        num_entries = len(metadata)
    else:
        metadata = None
        num_entries = 0

    # Image statistics
    img_stats = analyze_image_statistics(images[:100], sample_size=100)

    analysis = {
        "dataset_name": "PadChest",
        "task": "Thoracic Disease Detection (Multi-label, Spanish labels)",
        "modality": "Chest X-ray (Grayscale)",
        "num_classes": "174+ radiological findings",
        "task_type": "Multi-label Classification",
        "splits": {
            "total": {
                "num_samples": len(images),
                "num_csv_entries": num_entries,
                "image_statistics": img_stats,
            }
        },
        "total_samples": len(images),
        "language": "Spanish (labels in Spanish)",
        "data_format": {
            "images": "PNG",
            "labels": "CSV (complex multi-label with Spanish terms)",
        },
        "metadata_files": ["metadata.csv", "train.csv", "val.csv"]
        if metadata_path.exists()
        else [],
        "license": "Academic/Research Use",
        "source": "https://bimcv.cipf.es/bimcv-projects/padchest/",
        "notes": "Requires preprocessing for label translation",
    }

    return analysis


def generate_summary_table(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate summary comparison table."""
    summary = []
    for analysis in analyses:
        summary.append(
            {
                "Dataset": analysis["dataset_name"],
                "Modality": analysis["modality"],
                "Task": analysis["task"],
                "Num Classes": analysis["num_classes"],
                "Task Type": analysis["task_type"],
                "Total Samples": analysis["total_samples"],
                "Class Balance": analysis.get("class_balance", "Unknown"),
                "Image Format": analysis["data_format"]["images"],
                "Label Format": analysis["data_format"]["labels"],
            }
        )
    return pd.DataFrame(summary)


def save_analysis_report(
    analyses: List[Dict[str, Any]], output_path: Path
) -> None:
    """Save comprehensive analysis report."""
    report = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "data_root": str(DATA_ROOT),
        "datasets": analyses,
        "summary_statistics": {
            "total_datasets": len(analyses),
            "total_samples": sum(a["total_samples"] for a in analyses),
            "modalities": list(set(a["modality"] for a in analyses)),
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[SUCCESS] Analysis report saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("PHASE 2.1: COMPREHENSIVE DATASET ANALYSIS")
    print("Production-Level Analysis of Medical Imaging Datasets")
    print("=" * 80)

    # Verify data root exists
    if not DATA_ROOT.exists():
        print(f"[ERROR] Data root not found: {DATA_ROOT}")
        return

    # Analyze each dataset
    analyses = []

    try:
        analyses.append(analyze_isic_2018())
    except Exception as e:
        print(f"[ERROR] Error analyzing ISIC 2018: {e}")

    try:
        analyses.append(analyze_isic_2019())
    except Exception as e:
        print(f"[ERROR] Error analyzing ISIC 2019: {e}")

    try:
        analyses.append(analyze_isic_2020())
    except Exception as e:
        print(f"[ERROR] Error analyzing ISIC 2020: {e}")

    try:
        analyses.append(analyze_derm7pt())
    except Exception as e:
        print(f"[ERROR] Error analyzing Derm7pt: {e}")

    try:
        analyses.append(analyze_nih_cxr())
    except Exception as e:
        print(f"[ERROR] Error analyzing NIH CXR: {e}")

    try:
        analyses.append(analyze_padchest())
    except Exception as e:
        print(f"[ERROR] Error analyzing PadChest: {e}")

    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    summary_df = generate_summary_table(analyses)
    print(summary_df.to_string(index=False))

    # Save report
    output_dir = Path(__file__).parent.parent / "docs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_2_1_dataset_analysis.json"

    save_analysis_report(analyses, output_path)

    # Print individual analyses
    print("\n" + "=" * 80)
    print("DETAILED ANALYSES")
    print("=" * 80)
    for analysis in analyses:
        print(f"\n{analysis['dataset_name']}:")
        print(json.dumps(analysis, indent=2, default=str))

    print("\n" + "=" * 80)
    print("[SUCCESS] PHASE 2.1 DATASET ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total Datasets Analyzed: {len(analyses)}")
    print(
        f"Total Samples: {sum(a['total_samples'] for a in analyses):,}"
    )
    print(f"Report Location: {output_path}")


if __name__ == "__main__":
    main()
