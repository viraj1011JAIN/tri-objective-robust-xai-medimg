#!/usr/bin/env python
"""
Production-Grade Data Validation and Statistics for Medical Imaging Datasets.

This script performs comprehensive validation and statistical analysis for
medical imaging datasets (ISIC, Derm7pt, ChestXRay), including:
- Missing file detection
- Corrupted image detection
- Image format and size validation
- Label distribution analysis
- Class imbalance detection
- Cross-site distribution analysis
- Metadata CSV quality checks
- Publication-quality visualization generation
- Detailed JSON and Markdown report generation

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Course: MSc Computing Science - Dissertation
Date: November 2025

Usage:
    python scripts/data/validate_data.py \
        --dataset isic2018 \
        --root data/raw/ISIC2018 \
        --csv-path data/raw/ISIC2018/metadata.csv \
        --splits train val test \
        --output-dir results/data_validation \
        --generate-plots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.datasets.base_dataset import BaseMedicalDataset
from src.datasets.chest_xray import ChestXRayDataset
from src.datasets.derm7pt import Derm7ptDataset
from src.datasets.isic import ISICDataset

plt.switch_backend("Agg")


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def build_dataset(
    dataset_key: str,
    root: Path,
    split: str,
    csv_path: Optional[Path] = None,
) -> BaseMedicalDataset:
    key = dataset_key.lower().strip()
    csv = csv_path or (root / "metadata.csv")

    logger.info("Building dataset: %s, split: %s", dataset_key, split)
    logger.info("Root: %s", root)
    logger.info("CSV: %s", csv)

    if key in {
        "isic",
        "isic2018",
        "isic_2018",
        "isic2019",
        "isic_2019",
        "isic2020",
        "isic_2020",
    }:
        return ISICDataset(root=root, split=split, csv_path=csv)

    if key in {"derm7pt", "derm", "derm_7pt"}:
        return Derm7ptDataset(root=root, split=split, csv_path=csv)

    if key in {"chest_xray", "cxr", "nih_cxr", "padchest", "nih", "chest"}:
        return ChestXRayDataset(root=root, split=split, csv_path=csv, transforms=None)

    raise ValueError(
        f"Unknown dataset key: '{dataset_key}'. "
        "Supported: isic2018, isic2019, isic2020, derm7pt, nih_cxr, padchest"
    )


def validate_images_comprehensive(
    dataset: BaseMedicalDataset,
    max_images: Optional[int] = None,
    skip_missing: bool = True,
) -> Dict[str, Any]:
    logger.info(
        "Starting comprehensive image validation (max_images=%s)...", str(max_images)
    )

    widths: List[int] = []
    heights: List[int] = []
    channels: List[int] = []
    modes: Counter = Counter()
    extensions: Counter = Counter()
    corrupted: List[Dict[str, str]] = []

    if skip_missing:
        validation_result = dataset.validate(strict=False)
        missing_files = set(validation_result.get("missing_files", []))
        logger.info("Skipping %d known missing files", len(missing_files))
    else:
        missing_files = set()

    samples = dataset.samples
    total_samples = len(samples)
    scan_limit = min(max_images, total_samples) if max_images else total_samples

    logger.info("Scanning %d images out of %d total samples", scan_limit, total_samples)

    for idx in tqdm(range(scan_limit), desc="Validating images", unit="img"):
        sample = samples[idx]
        image_path = sample.image_path

        if str(image_path) in missing_files:
            continue

        if hasattr(dataset, "_resolve_image_path"):
            try:
                path = dataset._resolve_image_path(image_path)
            except Exception:
                path = Path(image_path)
        else:
            path = Path(image_path)

        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                w, h = img.size
                mode = img.mode
                c = len(img.getbands())

                widths.append(int(w))
                heights.append(int(h))
                channels.append(int(c))
                modes[mode] += 1

                ext = path.suffix.lower() or "<no_ext>"
                extensions[ext] += 1

        except (OSError, UnidentifiedImageError, ValueError) as e:
            corrupted.append(
                {
                    "path": str(path),
                    "error": f"{type(e).__name__}: {str(e)}",
                }
            )
            logger.warning("Corrupted image: %s - %s", path, e)

    num_scanned = len(widths)
    num_corrupted = len(corrupted)
    success_rate = (num_scanned / scan_limit * 100) if scan_limit > 0 else 0.0

    if widths:
        w_arr = np.array(widths, dtype=np.float64)
        h_arr = np.array(heights, dtype=np.float64)

        stats: Dict[str, Any] = {
            "num_images_scanned": num_scanned,
            "num_corrupted": num_corrupted,
            "success_rate": float(success_rate),
            "width": {
                "min": int(w_arr.min()),
                "max": int(w_arr.max()),
                "mean": float(w_arr.mean()),
                "std": float(w_arr.std()),
                "median": float(np.median(w_arr)),
                "q25": float(np.percentile(w_arr, 25)),
                "q75": float(np.percentile(w_arr, 75)),
            },
            "height": {
                "min": int(h_arr.min()),
                "max": int(h_arr.max()),
                "mean": float(h_arr.mean()),
                "std": float(h_arr.std()),
                "median": float(np.median(h_arr)),
                "q25": float(np.percentile(h_arr, 25)),
                "q75": float(np.percentile(h_arr, 75)),
            },
            "channels": {
                "unique": sorted(set(channels)),
                "counts": dict(Counter(channels)),
            },
            "modes": dict(modes),
            "extensions": dict(extensions),
            "corrupted_files": corrupted,
            "sizes": list(zip(widths, heights)),
        }
    else:
        stats = {
            "num_images_scanned": 0,
            "num_corrupted": num_corrupted,
            "success_rate": 0.0,
            "width": None,
            "height": None,
            "channels": None,
            "modes": {},
            "extensions": {},
            "corrupted_files": corrupted,
            "sizes": [],
        }

    logger.info(
        "Image validation complete: %d scanned, %d corrupted",
        num_scanned,
        num_corrupted,
    )
    return stats


def compute_label_statistics(
    dataset: BaseMedicalDataset,
    imbalance_threshold: float = 5.0,
) -> Dict[str, Any]:
    logger.info("Computing label statistics and imbalance metrics...")

    stats = dataset.compute_class_statistics()

    class_names = stats.get("class_names", [])
    class_counts = np.array(stats.get("class_counts", []), dtype=np.float64)

    if len(class_counts) == 0:
        logger.warning("No class counts found!")
        return {
            **stats,
            "imbalance_ratio": 0.0,
            "is_imbalanced": False,
            "most_frequent_class": None,
            "least_frequent_class": None,
        }

    max_idx = int(class_counts.argmax())
    min_idx = int(class_counts.argmin())
    max_count = float(class_counts[max_idx])
    min_count = float(class_counts[min_idx])

    imbalance_ratio = max_count / max(min_count, 1.0)
    is_imbalanced = imbalance_ratio >= imbalance_threshold

    enhanced_stats = {
        **stats,
        "imbalance_ratio": float(imbalance_ratio),
        "is_imbalanced": bool(is_imbalanced),
        "most_frequent_class": class_names[max_idx],
        "least_frequent_class": class_names[min_idx],
        "max_count": float(max_count),
        "min_count": float(min_count),
        "total_samples": int(class_counts.sum()),
    }

    logger.info(
        "Imbalance ratio: %.2f %s",
        imbalance_ratio,
        "(IMBALANCED)" if is_imbalanced else "(OK)",
    )
    logger.info("Most frequent: %s (%.0f)", class_names[max_idx], max_count)
    logger.info("Least frequent: %s (%.0f)", class_names[min_idx], min_count)

    return enhanced_stats


def analyze_csv_metadata(csv_path: Path) -> Dict[str, Any]:
    logger.info("Analyzing CSV metadata: %s", csv_path)

    if not csv_path.exists():
        logger.warning("CSV file not found: %s", csv_path)
        return {"error": "CSV file not found"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        return {"error": f"Failed to read CSV: {e}"}

    num_rows = len(df)
    num_cols = len(df.columns)

    missing_counts = df.isna().sum().to_dict()
    missing_fractions = {col: count / num_rows for col, count in missing_counts.items()}
    rows_with_missing = int(df.isna().any(axis=1).sum())

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    unique_counts = {col: int(df[col].nunique()) for col in df.columns}

    site_candidates = ["dataset", "source_dataset", "site", "hospital", "center"]
    site_column = next((col for col in site_candidates if col in df.columns), None)

    analysis: Dict[str, Any] = {
        "num_rows": num_rows,
        "num_columns": num_cols,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_data": {
            "counts": missing_counts,
            "fractions": missing_fractions,
            "rows_with_any_missing": rows_with_missing,
            "fraction_rows_with_missing": (
                rows_with_missing / num_rows if num_rows > 0 else 0.0
            ),
        },
        "unique_values": unique_counts,
        "site_column": site_column,
    }

    logger.info("CSV analysis complete: %d rows, %d columns", num_rows, num_cols)
    logger.info(
        "Rows with missing data: %d (%.1f%%)",
        rows_with_missing,
        (rows_with_missing / num_rows * 100) if num_rows > 0 else 0.0,
    )

    return analysis


def compute_cross_site_distribution(
    dataset: BaseMedicalDataset,
    csv_path: Path,
) -> Optional[Dict[str, Any]]:
    logger.info("Computing cross-site distribution...")

    if isinstance(dataset, ChestXRayDataset):
        site_counts: Counter = Counter()
        for sample in dataset.samples:
            meta = sample.meta or {}
            site = meta.get("dataset", "Unknown")
            site_counts[str(site)] += 1

        if site_counts:
            logger.info(
                "Found %d sites from dataset metadata",
                len(site_counts),
            )
            return {
                "method": "dataset_metadata",
                "num_sites": len(site_counts),
                "by_site": dict(site_counts),
            }

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        site_candidates = ["dataset", "source_dataset", "site", "hospital", "center"]
        site_col = next((col for col in site_candidates if col in df.columns), None)

        if site_col:
            site_counts_df = df[site_col].value_counts().to_dict()
            logger.info(
                "Found %d sites from CSV column '%s'",
                len(site_counts_df),
                site_col,
            )
            return {
                "method": "csv_column",
                "site_column": site_col,
                "num_sites": len(site_counts_df),
                "by_site": site_counts_df,
            }
    except Exception as e:
        logger.warning("Could not compute cross-site distribution from CSV: %s", e)

    return None


def ensure_plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_class_distribution(
    label_stats: Dict[str, Any],
    output_dir: Path,
    dataset_key: str,
    split: str,
) -> Optional[Path]:
    class_names = label_stats.get("class_names", [])
    class_counts = label_stats.get("class_counts", [])

    if not class_names or not class_counts:
        logger.warning("No class data to plot")
        return None

    try:
        fig, ax = plt.subplots(
            figsize=(max(8, len(class_names) * 0.6), 6),
        )

        x = np.arange(len(class_names))
        counts = np.array(class_counts)

        bars = ax.bar(x, counts, alpha=0.8, edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            f"{dataset_key} - {split} - Class Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.tight_layout()

        plots_dir = ensure_plots_dir(output_dir)
        plot_path = plots_dir / f"{dataset_key}_{split}_class_distribution.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved class distribution plot: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.error("Failed to generate class distribution plot: %s", e)
        return None


def plot_image_sizes(
    image_stats: Dict[str, Any],
    output_dir: Path,
    dataset_key: str,
    split: str,
) -> Optional[Path]:
    sizes = image_stats.get("sizes", [])

    if not sizes:
        logger.warning("No image size data to plot")
        return None

    try:
        widths, heights = zip(*sizes)
        w_arr = np.array(widths)
        h_arr = np.array(heights)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(
            w_arr,
            h_arr,
            s=10,
            alpha=0.5,
            edgecolors="none",
        )
        ax1.set_xlabel("Width (pixels)", fontsize=12)
        ax1.set_ylabel("Height (pixels)", fontsize=12)
        ax1.set_title(
            f"{dataset_key} - {split} - Image Dimensions",
            fontsize=13,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)

        stats_text = (
            f"Width: {w_arr.min()}-{w_arr.max()} "
            f"(mean={w_arr.mean():.1f}, std={w_arr.std():.1f})\n"
            f"Height: {h_arr.min()}-{h_arr.max()} "
            f"(mean={h_arr.mean():.1f}, std={h_arr.std():.1f})\n"
            f"N = {len(sizes)}"
        )
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.5,
            },
            fontsize=9,
        )

        ax2.hist(
            w_arr,
            bins=50,
            alpha=0.6,
            edgecolor="black",
            label="Width",
        )
        ax2.hist(
            h_arr,
            bins=50,
            alpha=0.6,
            edgecolor="black",
            label="Height",
        )
        ax2.set_xlabel("Pixels", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Size Distribution", fontsize=13, fontweight="bold")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        plots_dir = ensure_plots_dir(output_dir)
        plot_path = plots_dir / f"{dataset_key}_{split}_image_sizes.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved image sizes plot: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.error("Failed to generate image sizes plot: %s", e)
        return None


def plot_cross_site_distribution(
    cross_site: Optional[Dict[str, Any]],
    output_dir: Path,
    dataset_key: str,
    split: str,
) -> Optional[Path]:
    if not cross_site:
        return None

    by_site = cross_site.get("by_site", {})
    if not by_site:
        return None

    try:
        sites = list(by_site.keys())
        counts = np.array(list(by_site.values()))

        fig, ax = plt.subplots(figsize=(max(8, len(sites) * 0.7), 6))

        x = np.arange(len(sites))
        bars = ax.bar(x, counts, alpha=0.8, edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=45, ha="right")
        ax.set_ylabel("Samples", fontsize=12)
        ax.set_title(
            f"{dataset_key} - {split} - Cross-Site Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.tight_layout()

        plots_dir = ensure_plots_dir(output_dir)
        plot_path = plots_dir / f"{dataset_key}_{split}_cross_site.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved cross-site plot: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.error("Failed to generate cross-site plot: %s", e)
        return None


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append(f"# Data Validation Report: {report_data['dataset']}")
    lines.append("")
    lines.append(f"**Generated:** {report_data['timestamp']}")
    lines.append("**Script:** scripts/data/validate_data.py")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- **Dataset:** {report_data['dataset']}")
    lines.append(f"- **Root Directory:** {report_data['root']}")
    lines.append(f"- **Metadata CSV:** {report_data['csv_path']}")
    lines.append(
        "- **Splits Analyzed:** " f"{', '.join(sorted(report_data['splits'].keys()))}"
    )
    lines.append("")

    if "csv_analysis" in report_data and "error" not in report_data["csv_analysis"]:
        csv_analysis = report_data["csv_analysis"]
        lines.append("## 2. Metadata CSV Analysis")
        lines.append("")
        lines.append(f"- **Total Rows:** {csv_analysis['num_rows']:,}")
        lines.append(f"- **Total Columns:** {csv_analysis['num_columns']}")
        lines.append(
            "- **Rows with Missing Data:** "
            f"{csv_analysis['missing_data']['rows_with_any_missing']:,} "
            f"({csv_analysis['missing_data']['fraction_rows_with_missing']*100:.1f}%)"
        )
        lines.append("")

        lines.append("### 2.1 Missing Data Per Column")
        lines.append("")
        lines.append("| Column | Missing Count | Missing Fraction |")
        lines.append("|--------|-------------:|-----------------:|")
        for col, count in csv_analysis["missing_data"]["counts"].items():
            frac = csv_analysis["missing_data"]["fractions"][col]
            if count > 0:
                lines.append(f"| {col} | {count:,} | {frac*100:.2f}% |")
        lines.append("")

    for split_name in sorted(report_data["splits"].keys()):
        split_data = report_data["splits"][split_name]

        lines.append(f"## 3. Split: {split_name}")
        lines.append("")

        if "error" in split_data:
            lines.append(f"ERROR: {split_data['error']}")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        lines.append("### 3.1 Summary")
        lines.append("")
        lines.append(f"- **Total Samples:** {split_data['num_samples']:,}")
        lines.append(
            "- **Missing Files:** "
            f"{split_data['validation']['num_missing_files']} "
            f"{'OK' if split_data['validation']['num_missing_files'] == 0 else 'ISSUES'}"
        )
        lines.append(
            "- **Corrupted Images:** "
            f"{split_data['image_stats']['num_corrupted']} "
            f"{'OK' if split_data['image_stats']['num_corrupted'] == 0 else 'ISSUES'}"
        )
        lines.append(
            "- **Image Scan Success Rate:** "
            f"{split_data['image_stats']['success_rate']:.1f}%"
        )
        lines.append("")

        label_stats = split_data["label_stats"]
        lines.append("### 3.2 Class Distribution")
        lines.append("")
        lines.append(
            f"- **Number of Classes:** {label_stats.get('num_classes', 'N/A')}"
        )
        lines.append(
            "- **Total Labeled Samples:** "
            f"{label_stats.get('total_samples', 'N/A'):,}"
        )
        lines.append(
            "- **Most Frequent Class:** "
            f"{label_stats['most_frequent_class']} "
            f"({label_stats['max_count']:.0f} samples)"
        )
        lines.append(
            "- **Least Frequent Class:** "
            f"{label_stats['least_frequent_class']} "
            f"({label_stats['min_count']:.0f} samples)"
        )
        lines.append(
            "- **Imbalance Ratio:** "
            f"{label_stats['imbalance_ratio']:.2f} "
            f"{'IMBALANCED' if label_stats['is_imbalanced'] else 'OK'}"
        )
        lines.append("")

        if "class_names" in label_stats and "class_counts" in label_stats:
            lines.append("| Class | Count | Positive Rate | Weight |")
            lines.append("|-------|------:|--------------:|-------:|")
            pos_rates = label_stats.get(
                "positive_rates",
                [0.0] * len(label_stats["class_names"]),
            )
            weights = label_stats.get(
                "class_weights",
                [0.0] * len(label_stats["class_names"]),
            )
            for i, name in enumerate(label_stats["class_names"]):
                count = label_stats["class_counts"][i]
                pos_rate = pos_rates[i]
                weight = weights[i]
                lines.append(
                    f"| {name} | {int(count):,} | {pos_rate:.4f} | {weight:.4f} |"
                )
            lines.append("")

        img_stats = split_data["image_stats"]
        if img_stats.get("width"):
            lines.append("### 3.3 Image Size Statistics")
            lines.append("")

            w = img_stats["width"]
            h = img_stats["height"]

            lines.append("**Width (pixels):**")
            lines.append(
                f"- Min: {w['min']}, Max: {w['max']}, Mean: {w['mean']:.1f}, "
                f"Std: {w['std']:.1f}"
            )
            lines.append(
                f"- Median: {w['median']:.1f}, Q25: {w['q25']:.1f}, "
                f"Q75: {w['q75']:.1f}"
            )
            lines.append("")

            lines.append("**Height (pixels):**")
            lines.append(
                f"- Min: {h['min']}, Max: {h['max']}, Mean: {h['mean']:.1f}, "
                f"Std: {h['std']:.1f}"
            )
            lines.append(
                f"- Median: {h['median']:.1f}, Q25: {h['q25']:.1f}, "
                f"Q75: {h['q75']:.1f}"
            )
            lines.append("")

            if img_stats.get("channels"):
                lines.append(f"**Channels:** {img_stats['channels']['unique']}")
                lines.append("")

            if img_stats.get("extensions"):
                lines.append(
                    f"**File Extensions:** " f"{list(img_stats['extensions'].keys())}"
                )
                lines.append("")

        if split_data.get("cross_site"):
            cross_site = split_data["cross_site"]
            lines.append("### 3.4 Cross-Site Distribution")
            lines.append("")
            lines.append(f"- **Number of Sites:** {cross_site['num_sites']}")
            lines.append(f"- **Detection Method:** {cross_site['method']}")
            lines.append("")
            lines.append("| Site | Samples |")
            lines.append("|------|--------:|")
            for site, count in cross_site["by_site"].items():
                lines.append(f"| {site} | {int(count):,} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## 4. Visualizations")
    lines.append("")
    lines.append("Generated plots are saved in plots/ subdirectory:")
    lines.append("")
    for split_name in sorted(report_data["splits"].keys()):
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append(
            f"- Class distribution: "
            f"{report_data['dataset']}_{split_name}_class_distribution.png"
        )
        lines.append(
            f"- Image sizes: " f"{report_data['dataset']}_{split_name}_image_sizes.png"
        )
        if report_data["splits"][split_name].get("cross_site"):
            lines.append(
                f"- Cross-site distribution: "
                f"{report_data['dataset']}_{split_name}_cross_site.png"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**End of Report**")

    return "\n".join(lines)


def validate_split(
    dataset_key: str,
    dataset: BaseMedicalDataset,
    split: str,
    csv_path: Path,
    output_dir: Path,
    max_images: Optional[int],
    imbalance_threshold: float,
    generate_plots: bool,
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Validating %s - %s", dataset_key, split)
    logger.info("=" * 60)

    validation_result = dataset.validate(strict=False)
    logger.info("Missing files: %d", validation_result["num_missing_files"])

    image_stats = validate_images_comprehensive(
        dataset=dataset,
        max_images=max_images,
        skip_missing=True,
    )

    label_stats = compute_label_statistics(
        dataset=dataset,
        imbalance_threshold=imbalance_threshold,
    )

    cross_site = compute_cross_site_distribution(dataset, csv_path)

    if generate_plots:
        logger.info("Generating plots...")
        plot_class_distribution(label_stats, output_dir, dataset_key, split)
        plot_image_sizes(image_stats, output_dir, dataset_key, split)
        plot_cross_site_distribution(cross_site, output_dir, dataset_key, split)

    return {
        "split": split,
        "num_samples": len(dataset),
        "validation": validation_result,
        "image_stats": image_stats,
        "label_stats": label_stats,
        "cross_site": cross_site,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-Grade Data Validation for Medical Imaging Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset key: isic2018, isic2019, isic2020, derm7pt, nih_cxr, padchest",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root directory containing images",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to metadata CSV (default: <root>/metadata.csv)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to validate (default: train val test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_validation"),
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to scan per split (None = all)",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=5.0,
        help="Class imbalance ratio threshold (default: 5.0)",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots (PNG files)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    root = args.root.expanduser().resolve()
    csv_path = (args.csv_path or (root / "metadata.csv")).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(
        "PRODUCTION-GRADE DATA VALIDATION - Medical Imaging Datasets",
    )
    logger.info("=" * 70)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Root: %s", root)
    logger.info("CSV: %s", csv_path)
    logger.info("Splits: %s", ", ".join(args.splits))
    logger.info("Output: %s", output_dir)
    logger.info("=" * 70)

    report: Dict[str, Any] = {
        "dataset": args.dataset,
        "root": str(root),
        "csv_path": str(csv_path),
        "timestamp": datetime.now().isoformat(),
        "splits": {},
        "csv_analysis": {},
    }

    try:
        report["csv_analysis"] = analyze_csv_metadata(csv_path)
    except Exception as e:
        logger.error("Failed to analyze CSV metadata: %s", e)
        report["csv_analysis"] = {"error": str(e)}

    for split in args.splits:
        try:
            dataset = build_dataset(
                dataset_key=args.dataset,
                root=root,
                split=split,
                csv_path=csv_path,
            )

            split_result = validate_split(
                dataset_key=args.dataset,
                dataset=dataset,
                split=split,
                csv_path=csv_path,
                output_dir=output_dir,
                max_images=args.max_images,
                imbalance_threshold=args.imbalance_threshold,
                generate_plots=args.generate_plots,
            )

            report["splits"][split] = split_result
        except Exception as e:
            logger.error("Failed to validate split '%s': %s", split, e, exc_info=True)
            report["splits"][split] = {
                "error": str(e),
            }

    json_path = output_dir / f"{args.dataset}_validation_report.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report saved: %s", json_path)
    except Exception as e:
        logger.error("Failed to save JSON report: %s", e)

    md_path = output_dir / f"{args.dataset}_validation_report.md"
    try:
        markdown = generate_markdown_report(report)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Markdown report saved: %s", md_path)
    except Exception as e:
        logger.error("Failed to save Markdown report: %s", e)

    logger.info("=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Results directory: %s", output_dir)
    logger.info("  - JSON: %s", json_path.name)
    logger.info("  - Markdown: %s", md_path.name)
    if args.generate_plots:
        logger.info("  - Plots: plots/")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
