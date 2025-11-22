"""
Aggregate Baseline Evaluation Results for Phase 3.5.

This script aggregates evaluation results across multiple datasets:
- ISIC 2018 (test set, same-site)
- ISIC 2019 (cross-site)
- ISIC 2020 (cross-site)
- Derm7pt (cross-site)

Generates:
- Consolidated CSV with all metrics
- Summary table (results/metrics/rq1_robustness/baseline.csv)
- Comparison plots

Usage:
    python scripts/evaluation/aggregate_baseline_results.py \\
        --results-dir results/evaluation \\
        --output-dir results/metrics/rq1_robustness

Phase 3.5: Baseline Evaluation - Dermoscopy
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_evaluation_results(results_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """
    Load evaluation results for a dataset.

    Parameters
    ----------
    results_dir : Path
        Base results directory
    dataset_name : str
        Dataset name

    Returns
    -------
    results : dict
        Evaluation results or empty dict if not found
    """
    json_path = results_dir / f"baseline_{dataset_name}" / "evaluation_results.json"

    if not json_path.exists():
        logger.warning(f"Results not found for {dataset_name}: {json_path}")
        return {}

    with open(json_path, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded results for {dataset_name}")
    return results


def extract_summary_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from evaluation results.

    Parameters
    ----------
    results : dict
        Full evaluation results

    Returns
    -------
    summary : dict
        Summary metrics
    """
    if not results:
        return {}

    summary = {}

    # Classification metrics
    classification = results.get("classification", {})
    summary["accuracy"] = classification.get("accuracy", np.nan)
    summary["auroc_macro"] = classification.get("auroc_macro", np.nan)
    summary["auroc_weighted"] = classification.get("auroc_weighted", np.nan)
    summary["f1_macro"] = classification.get("f1_macro", np.nan)
    summary["f1_weighted"] = classification.get("f1_weighted", np.nan)
    summary["mcc"] = classification.get("mcc", np.nan)

    # Calibration metrics
    calibration = results.get("calibration", {})
    summary["ece"] = calibration.get("ece", np.nan)
    summary["mce"] = calibration.get("mce", np.nan)
    summary["brier_score"] = calibration.get("brier_score", np.nan)

    # Bootstrap CI
    bootstrap_ci = results.get("bootstrap_ci", {})
    for metric in ["accuracy", "auroc_macro", "f1_macro", "mcc"]:
        ci = bootstrap_ci.get(f"{metric}_ci", (np.nan, np.nan))
        summary[f"{metric}_ci_lower"] = ci[0]
        summary[f"{metric}_ci_upper"] = ci[1]

    # Metadata
    metadata = results.get("metadata", {})
    summary["num_samples"] = metadata.get("num_samples", 0)
    summary["num_classes"] = metadata.get("num_classes", 0)

    return summary


def create_summary_table(
    results_dict: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Create summary table from all results.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping dataset names to evaluation results

    Returns
    -------
    df : pd.DataFrame
        Summary table
    """
    rows = []

    for dataset_name, results in results_dict.items():
        if not results:
            continue

        summary = extract_summary_metrics(results)
        summary["dataset"] = dataset_name

        # Determine if cross-site
        summary["evaluation_type"] = (
            "same-site" if dataset_name == "isic2018" else "cross-site"
        )

        rows.append(summary)

    df = pd.DataFrame(rows)

    # Reorder columns
    if not df.empty:
        first_cols = ["dataset", "evaluation_type", "num_samples", "num_classes"]
        metric_cols = [
            "accuracy",
            "auroc_macro",
            "auroc_weighted",
            "f1_macro",
            "f1_weighted",
            "mcc",
            "ece",
            "mce",
            "brier_score",
        ]
        ci_cols = [col for col in df.columns if "_ci_" in col]
        other_cols = [
            col
            for col in df.columns
            if col not in first_cols + metric_cols + ci_cols
        ]
        df = df[first_cols + metric_cols + ci_cols + other_cols]

    return df


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str = None,
) -> None:
    """
    Plot metric comparison across datasets with error bars.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table
    metric : str
        Metric name to plot
    output_path : Path
        Path to save figure
    title : str, optional
        Plot title
    """
    if df.empty or metric not in df.columns:
        logger.warning(f"Cannot plot {metric}: data not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    datasets = df["dataset"].values
    values = df[metric].values
    ci_lower = df.get(f"{metric}_ci_lower", np.full_like(values, np.nan))
    ci_upper = df.get(f"{metric}_ci_upper", np.full_like(values, np.nan))

    # Compute error bars
    yerr_lower = values - ci_lower
    yerr_upper = ci_upper - values
    yerr = np.array([yerr_lower, yerr_upper])

    # Colors based on evaluation type
    colors = [
        "steelblue" if t == "same-site" else "coral"
        for t in df["evaluation_type"]
    ]

    # Bar plot
    x_pos = np.arange(len(datasets))
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor="black")

    # Error bars (only if CI available)
    if not np.all(np.isnan(yerr)):
        ax.errorbar(
            x_pos,
            values,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=2,
        )

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(metric.upper().replace("_", " "), fontsize=12)
    ax.set_title(
        title or f"{metric.upper().replace('_', ' ')} Comparison", fontsize=14
    )
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="Same-Site"),
        Patch(facecolor="coral", alpha=0.7, label="Cross-Site"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved plot to {output_path}")


def plot_all_metrics_heatmap(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot heatmap of all metrics across datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table
    output_path : Path
        Path to save figure
    """
    if df.empty:
        return

    # Select numeric metric columns
    metric_cols = [
        "accuracy",
        "auroc_macro",
        "f1_macro",
        "mcc",
        "ece",
        "mce",
        "brier_score",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    if not metric_cols:
        return

    # Create matrix
    data = df[metric_cols].values
    datasets = df["dataset"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap
    im = ax.imshow(data.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Ticks and labels
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticklabels([m.upper().replace("_", " ") for m in metric_cols])

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(metric_cols)):
            value = data[i, j]
            if not np.isnan(value):
                text = ax.text(
                    i,
                    j,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Metric Value", rotation=270, labelpad=20)

    ax.set_title("Baseline Evaluation Metrics Heatmap", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved heatmap to {output_path}")


def generate_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate LaTeX table for dissertation.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table
    output_path : Path
        Path to save LaTeX file
    """
    if df.empty:
        return

    # Select columns for LaTeX table
    columns = [
        "dataset",
        "evaluation_type",
        "num_samples",
        "accuracy",
        "auroc_macro",
        "f1_macro",
        "mcc",
        "ece",
    ]
    columns = [c for c in columns if c in df.columns]

    df_latex = df[columns].copy()

    # Format metrics with CI
    for metric in ["accuracy", "auroc_macro", "f1_macro", "mcc"]:
        if metric in df_latex.columns and f"{metric}_ci_lower" in df.columns:
            df_latex[metric] = df.apply(
                lambda row: f"{row[metric]:.3f} "
                f"[{row[f'{metric}_ci_lower']:.3f}, "
                f"{row[f'{metric}_ci_upper']:.3f}]",
                axis=1,
            )

    # Generate LaTeX
    latex_str = df_latex.to_latex(
        index=False,
        caption="Baseline Evaluation Results on Dermoscopy Datasets",
        label="tab:baseline_evaluation",
        float_format="%.3f",
    )

    with open(output_path, "w") as f:
        f.write(latex_str)

    logger.info(f"Saved LaTeX table to {output_path}")


def main(args: argparse.Namespace) -> None:
    """Main aggregation function."""
    setup_logging()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Phase 3.5: Aggregate Baseline Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Datasets to aggregate
    datasets = ["isic2018", "isic2019", "isic2020", "derm7pt"]

    # Load all results
    results_dict = {}
    for dataset in datasets:
        results = load_evaluation_results(results_dir, dataset)
        if results:
            results_dict[dataset] = results

    if not results_dict:
        logger.error("No evaluation results found!")
        logger.error(f"Expected results in: {results_dir}/baseline_*/evaluation_results.json")
        logger.error(
            "⚠️  Run evaluation first with scripts/evaluation/evaluate_baseline.py"
        )
        return

    # Create summary table
    logger.info("Creating summary table...")
    df_summary = create_summary_table(results_dict)

    if df_summary.empty:
        logger.error("Failed to create summary table")
        return

    # Save summary CSV
    csv_path = output_dir / "baseline.csv"
    df_summary.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")

    # Save detailed CSV
    csv_path_detailed = output_dir / "baseline_detailed.csv"
    df_summary.to_csv(csv_path_detailed, index=False)
    logger.info(f"Saved detailed results to {csv_path_detailed}")

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating comparison plots...")

    for metric in ["accuracy", "auroc_macro", "f1_macro", "mcc", "ece"]:
        plot_metric_comparison(
            df_summary,
            metric,
            plots_dir / f"{metric}_comparison.png",
            title=f"Baseline {metric.upper().replace('_', ' ')} Comparison",
        )

    # Heatmap
    plot_all_metrics_heatmap(
        df_summary,
        plots_dir / "metrics_heatmap.png",
    )

    # LaTeX table
    latex_path = output_dir / "baseline_table.tex"
    generate_latex_table(df_summary, latex_path)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE EVALUATION SUMMARY")
    logger.info("=" * 80)
    print("\n" + df_summary.to_string(index=False))
    logger.info("=" * 80)

    logger.info(f"\n✅ Aggregation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate baseline evaluation results"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/evaluation",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/metrics/rq1_robustness",
        help="Directory to save aggregated results",
    )

    args = parser.parse_args()
    main(args)
