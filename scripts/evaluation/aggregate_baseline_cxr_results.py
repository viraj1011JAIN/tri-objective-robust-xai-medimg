"""
Aggregate baseline chest X-ray evaluation results across datasets.

Consolidates evaluation results from NIH ChestX-ray14 (same-site) and
PadChest (cross-site) evaluations, computes cross-site AUROC drop,
generates comparison visualizations, and exports to CSV.

Usage:
    python scripts/evaluation/aggregate_baseline_cxr_results.py \
        --results-dir results/evaluation \
        --output-dir results/metrics/rq1_robustness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_evaluation_results(
    results_dir: Path, dataset_names: List[str]
) -> Dict[str, Dict]:
    """
    Load evaluation results from JSON files.

    Parameters
    ----------
    results_dir : Path
        Directory containing evaluation subdirectories
    dataset_names : List[str]
        List of dataset names to load results for

    Returns
    -------
    dict
        Dictionary mapping dataset names to their evaluation results
    """
    results = {}

    for dataset_name in dataset_names:
        # Try multiple possible directory names
        possible_dirs = [
            results_dir / f"baseline_{dataset_name}",
            results_dir / dataset_name,
        ]

        results_file = None
        for dir_path in possible_dirs:
            candidate = dir_path / "results.json"
            if candidate.exists():
                results_file = candidate
                break

        if results_file is None:
            print(f"⚠️  Warning: Results not found for {dataset_name}")
            continue

        with open(results_file, "r") as f:
            results[dataset_name] = json.load(f)

        print(f"✓ Loaded results for {dataset_name}")

    return results


def extract_summary_metrics(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Extract summary metrics from evaluation results.

    Parameters
    ----------
    results : dict
        Dictionary mapping dataset names to evaluation results

    Returns
    -------
    pd.DataFrame
        DataFrame with summary metrics for each dataset
    """
    summary_data = []

    for dataset_name, metrics in results.items():
        row = {
            "dataset": dataset_name,
            "evaluation_type": (
                "same-site" if dataset_name == "nih_chestxray14" else "cross-site"
            ),
            # AUROC metrics
            "auroc_macro": metrics.get("auroc_macro", np.nan),
            "auroc_macro_ci_lower": metrics.get("auroc_macro_ci_lower", np.nan),
            "auroc_macro_ci_upper": metrics.get("auroc_macro_ci_upper", np.nan),
            "auroc_micro": metrics.get("auroc_micro", np.nan),
            "auroc_micro_ci_lower": metrics.get("auroc_micro_ci_lower", np.nan),
            "auroc_micro_ci_upper": metrics.get("auroc_micro_ci_upper", np.nan),
            "auroc_weighted": metrics.get("auroc_weighted", np.nan),
            # Multi-label metrics
            "hamming_loss": metrics.get("hamming_loss", np.nan),
            "hamming_loss_ci_lower": metrics.get("hamming_loss_ci_lower", np.nan),
            "hamming_loss_ci_upper": metrics.get("hamming_loss_ci_upper", np.nan),
            "subset_accuracy": metrics.get("subset_accuracy", np.nan),
            "precision_macro": metrics.get("precision_macro", np.nan),
            "recall_macro": metrics.get("recall_macro", np.nan),
            "f1_macro": metrics.get("f1_macro", np.nan),
            # Calibration metrics
            "ece_macro": metrics.get("ece_macro", np.nan),
            "mce_macro": metrics.get("mce_macro", np.nan),
            "brier_score_macro": metrics.get("brier_score_macro", np.nan),
            # Ranking metrics
            "coverage_error": metrics.get("coverage_error", np.nan),
            "ranking_loss": metrics.get("ranking_loss", np.nan),
            "label_ranking_avg_precision": metrics.get(
                "label_ranking_avg_precision", np.nan
            ),
        }

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def compute_cross_site_auroc_drop(summary_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute AUROC drop from same-site to cross-site evaluation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with evaluation results

    Returns
    -------
    dict
        Dictionary with AUROC drop metrics
    """
    # Get same-site (NIH) results
    nih_row = summary_df[summary_df["dataset"] == "nih_chestxray14"]
    if nih_row.empty:
        print("⚠️  Warning: NIH ChestX-ray14 results not found")
        return {}

    nih_auroc_macro = nih_row["auroc_macro"].values[0]
    nih_auroc_micro = nih_auroc_micro = nih_row["auroc_micro"].values[0]

    # Get cross-site (PadChest) results
    padchest_row = summary_df[summary_df["dataset"] == "padchest"]
    if padchest_row.empty:
        print("⚠️  Warning: PadChest results not found")
        return {
            "same_site_auroc_macro": nih_auroc_macro,
            "same_site_auroc_micro": nih_auroc_micro,
        }

    padchest_auroc_macro = padchest_row["auroc_macro"].values[0]
    padchest_auroc_micro = padchest_row["auroc_micro"].values[0]

    # Compute absolute and relative drop
    auroc_macro_drop_abs = nih_auroc_macro - padchest_auroc_macro
    auroc_macro_drop_rel = (auroc_macro_drop_abs / nih_auroc_macro) * 100

    auroc_micro_drop_abs = nih_auroc_micro - padchest_auroc_micro
    auroc_micro_drop_rel = (auroc_micro_drop_abs / nih_auroc_micro) * 100

    return {
        "same_site_auroc_macro": nih_auroc_macro,
        "cross_site_auroc_macro": padchest_auroc_macro,
        "auroc_macro_drop_absolute": auroc_macro_drop_abs,
        "auroc_macro_drop_relative_percent": auroc_macro_drop_rel,
        "same_site_auroc_micro": nih_auroc_micro,
        "cross_site_auroc_micro": padchest_auroc_micro,
        "auroc_micro_drop_absolute": auroc_micro_drop_abs,
        "auroc_micro_drop_relative_percent": auroc_micro_drop_rel,
    }


def create_summary_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comprehensive summary table with all metrics.

    Parameters
    ----------
    all_results : dict
        Dictionary mapping dataset names to evaluation results

    Returns
    -------
    pd.DataFrame
        Summary table with all metrics
    """
    summary_df = extract_summary_metrics(all_results)

    # Sort by evaluation type (same-site first)
    summary_df = summary_df.sort_values(
        by=["evaluation_type", "dataset"], ascending=[False, True]
    )

    return summary_df


def plot_metric_comparison(
    summary_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Plot comparison of key metrics across datasets.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with evaluation results
    output_dir : Path
        Directory to save plots
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Metrics to compare
    metrics_to_plot = [
        ("auroc_macro", "Macro AUROC"),
        ("auroc_micro", "Micro AUROC"),
        ("hamming_loss", "Hamming Loss"),
        ("subset_accuracy", "Subset Accuracy"),
        ("f1_macro", "Macro F1"),
        ("ece_macro", "ECE (Macro)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Color mapping for evaluation types
    color_map = {"same-site": "#2ecc71", "cross-site": "#e74c3c"}

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Extract data
        datasets = summary_df["dataset"].tolist()
        values = summary_df[metric_key].tolist()
        colors = [color_map[et] for et in summary_df["evaluation_type"]]

        # Get CI bounds if available
        ci_lower_key = f"{metric_key}_ci_lower"
        ci_upper_key = f"{metric_key}_ci_upper"

        if ci_lower_key in summary_df.columns and ci_upper_key in summary_df.columns:
            ci_lower = summary_df[ci_lower_key].tolist()
            ci_upper = summary_df[ci_upper_key].tolist()
            errors = [
                [values[i] - ci_lower[i], ci_upper[i] - values[i]]
                for i in range(len(values))
            ]
            errors = np.array(errors).T
        else:
            errors = None

        # Create bar plot
        x_pos = np.arange(len(datasets))
        ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor="black")

        # Add error bars if available
        if errors is not None:
            ax.errorbar(
                x_pos,
                values,
                yerr=errors,
                fmt="none",
                ecolor="black",
                capsize=5,
                alpha=0.8,
            )

        # Customize subplot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (x, v) in enumerate(zip(x_pos, values)):
            if not np.isnan(v):
                ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map["same-site"], label="Same-site (NIH)"),
        Patch(facecolor=color_map["cross-site"], label="Cross-site (PadChest)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = plots_dir / "metric_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved metric comparison plot: {save_path}")


def plot_auroc_drop_visualization(
    auroc_drop: Dict[str, float], output_dir: Path
) -> None:
    """
    Visualize AUROC drop from same-site to cross-site evaluation.

    Parameters
    ----------
    auroc_drop : dict
        Dictionary with AUROC drop metrics
    output_dir : Path
        Directory to save plots
    """
    if not auroc_drop:
        print("⚠️  Skipping AUROC drop visualization (insufficient data)")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Macro AUROC comparison
    ax1 = axes[0]
    datasets = ["NIH\n(same-site)", "PadChest\n(cross-site)"]
    auroc_values = [
        auroc_drop["same_site_auroc_macro"],
        auroc_drop["cross_site_auroc_macro"],
    ]
    colors = ["#2ecc71", "#e74c3c"]

    ax1.bar(datasets, auroc_values, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Macro AUROC", fontsize=12)
    ax1.set_title("Macro AUROC: Same-Site vs Cross-Site", fontsize=13, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(auroc_values):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Add drop annotation
    drop_abs = auroc_drop["auroc_macro_drop_absolute"]
    drop_rel = auroc_drop["auroc_macro_drop_relative_percent"]
    ax1.text(
        0.5,
        0.1,
        f"Drop: {drop_abs:.3f} ({drop_rel:.1f}%)",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        transform=ax1.transAxes,
    )

    # Micro AUROC comparison
    ax2 = axes[1]
    auroc_values = [
        auroc_drop["same_site_auroc_micro"],
        auroc_drop["cross_site_auroc_micro"],
    ]

    ax2.bar(datasets, auroc_values, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Micro AUROC", fontsize=12)
    ax2.set_title("Micro AUROC: Same-Site vs Cross-Site", fontsize=13, fontweight="bold")
    ax2.set_ylim([0, 1])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(auroc_values):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Add drop annotation
    drop_abs = auroc_drop["auroc_micro_drop_absolute"]
    drop_rel = auroc_drop["auroc_micro_drop_relative_percent"]
    ax2.text(
        0.5,
        0.1,
        f"Drop: {drop_abs:.3f} ({drop_rel:.1f}%)",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        transform=ax2.transAxes,
    )

    plt.tight_layout()
    save_path = plots_dir / "auroc_drop_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved AUROC drop visualization: {save_path}")


def generate_latex_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate LaTeX table for dissertation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with evaluation results
    output_dir : Path
        Directory to save LaTeX file
    """
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Baseline Chest X-Ray Evaluation Results}")
    latex_lines.append("\\label{tab:baseline_cxr_results}")
    latex_lines.append("\\begin{tabular}{lccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append(
        "Dataset & Type & Macro AUROC & Micro AUROC & Hamming Loss & ECE \\\\"
    )
    latex_lines.append("\\midrule")

    for _, row in summary_df.iterrows():
        dataset = row["dataset"].replace("_", " ").title()
        eval_type = row["evaluation_type"]

        auroc_macro = row["auroc_macro"]
        auroc_macro_ci_lower = row.get("auroc_macro_ci_lower", np.nan)
        auroc_macro_ci_upper = row.get("auroc_macro_ci_upper", np.nan)

        auroc_micro = row["auroc_micro"]
        hamming = row["hamming_loss"]
        ece = row["ece_macro"]

        # Format with CI if available
        if not np.isnan(auroc_macro_ci_lower) and not np.isnan(auroc_macro_ci_upper):
            auroc_macro_str = f"{auroc_macro:.3f} [{auroc_macro_ci_lower:.3f}, {auroc_macro_ci_upper:.3f}]"
        else:
            auroc_macro_str = f"{auroc_macro:.3f}"

        latex_lines.append(
            f"{dataset} & {eval_type} & {auroc_macro_str} & {auroc_micro:.3f} & "
            f"{hamming:.3f} & {ece:.3f} \\\\"
        )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_file = output_dir / "baseline_cxr_table.tex"
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"✓ Saved LaTeX table: {latex_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate baseline chest X-ray evaluation results"
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
        help="Output directory for aggregated results",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Aggregating Baseline Chest X-Ray Evaluation Results")
    print("=" * 60)

    # Load results
    dataset_names = ["nih_chestxray14", "padchest"]
    all_results = load_evaluation_results(results_dir, dataset_names)

    if not all_results:
        print("\n⚠️  No evaluation results found!")
        print("Please run evaluate_baseline_cxr.py first.")
        return

    # Create summary table
    summary_df = create_summary_table(all_results)
    print(f"\n✓ Created summary table with {len(summary_df)} datasets")

    # Compute cross-site AUROC drop
    auroc_drop = compute_cross_site_auroc_drop(summary_df)
    if auroc_drop:
        print("\nCross-Site AUROC Drop:")
        print(f"  Macro AUROC drop: {auroc_drop['auroc_macro_drop_absolute']:.3f} "
              f"({auroc_drop['auroc_macro_drop_relative_percent']:.1f}%)")
        print(f"  Micro AUROC drop: {auroc_drop['auroc_micro_drop_absolute']:.3f} "
              f"({auroc_drop['auroc_micro_drop_relative_percent']:.1f}%)")

    # Save summary CSV
    summary_file = output_dir / "baseline_cxr.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Saved summary CSV: {summary_file}")

    # Save detailed CSV (with per-class metrics if available)
    detailed_file = output_dir / "baseline_cxr_detailed.csv"
    summary_df.to_csv(detailed_file, index=False)
    print(f"✓ Saved detailed CSV: {detailed_file}")

    # Save AUROC drop metrics
    if auroc_drop:
        auroc_drop_file = output_dir / "baseline_cxr_auroc_drop.json"
        with open(auroc_drop_file, "w") as f:
            json.dump(auroc_drop, f, indent=2)
        print(f"✓ Saved AUROC drop metrics: {auroc_drop_file}")

    # Generate plots
    print("\nGenerating plots...")
    plot_metric_comparison(summary_df, output_dir)
    plot_auroc_drop_visualization(auroc_drop, output_dir)

    # Generate LaTeX table
    generate_latex_table(summary_df, output_dir)

    print(f"\n{'=' * 60}")
    print("Aggregation Complete")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - baseline_cxr.csv (summary)")
    print(f"  - baseline_cxr_detailed.csv (full metrics)")
    print(f"  - baseline_cxr_auroc_drop.json (cross-site drop)")
    print(f"  - baseline_cxr_table.tex (LaTeX table)")
    print(f"  - plots/ (visualizations)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
