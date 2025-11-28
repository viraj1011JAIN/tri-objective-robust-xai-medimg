"""
RQ1 Report Generator
====================

Generates comprehensive reports for RQ1 evaluation including:
- Summary tables (LaTeX, Markdown, CSV)
- Statistical test summaries
- Pareto frontier plots
- Calibration plots
- Cross-site generalization visualization

Phase 9.2: RQ1 Evaluation
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.evaluation.pareto_analysis import (
    ParetoFrontier,
    ParetoSolution,
    plot_pareto_2d,
)

logger = logging.getLogger(__name__)

# Set publication-quality plotting defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "figure.figsize": (10, 8),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


class RQ1ReportGenerator:
    """
    Generate comprehensive reports for RQ1 evaluation.

    Creates publication-ready tables and figures for:
    - Task performance comparison
    - Robustness evaluation
    - Cross-site generalization
    - Calibration analysis
    - Statistical significance
    - Hypothesis test results
    - Pareto frontier analysis
    """

    def __init__(self, results: Dict[str, List], output_dir: Path):
        """
        Initialize report generator.

        Args:
            results: Dictionary of evaluation results from RQ1Evaluator
            output_dir: Output directory for reports
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        logger.info(f"RQ1ReportGenerator initialized. Output: {self.output_dir}")

    def generate_all_reports(self) -> None:
        """Generate all reports, tables, and figures."""
        logger.info("=" * 80)
        logger.info("GENERATING RQ1 REPORTS")
        logger.info("=" * 80)

        # Tables
        logger.info("\n[1/3] Generating tables...")
        self.generate_task_performance_table()
        self.generate_robustness_table()
        self.generate_cross_site_table()
        self.generate_calibration_table()
        self.generate_statistical_tests_table()

        # Figures
        logger.info("\n[2/3] Generating figures...")
        self.generate_pareto_figures()
        self.generate_calibration_figures()
        self.generate_cross_site_figures()
        self.generate_robustness_figures()

        # Summary report
        logger.info("\n[3/3] Generating summary report...")
        self.generate_summary_report()

        logger.info("\n" + "=" * 80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info(f"  Tables saved to: {self.tables_dir}")
        logger.info(f"  Figures saved to: {self.figures_dir}")
        logger.info("=" * 80)

    # ========================================================================
    # TABLE GENERATION
    # ========================================================================

    def generate_task_performance_table(self) -> None:
        """Generate Table 1: Task Performance (all models, all datasets)."""
        logger.info("  Table 1: Task Performance")

        # Convert results to DataFrame
        records = []
        for result in self.results["task_performance"]:
            records.append(
                {
                    "Model": result.model_name,
                    "Seed": result.seed,
                    "Dataset": result.dataset_name,
                    "Accuracy": result.accuracy,
                    "AUROC": result.auroc_macro,
                    "F1": result.f1_macro,
                    "MCC": result.mcc,
                }
            )

        df = pd.DataFrame(records)

        # Aggregate across seeds
        agg_df = (
            df.groupby(["Model", "Dataset"])
            .agg(
                {
                    "Accuracy": ["mean", "std"],
                    "AUROC": ["mean", "std"],
                    "F1": ["mean", "std"],
                    "MCC": ["mean", "std"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = [
            "Model",
            "Dataset",
            "Acc_Mean",
            "Acc_Std",
            "AUROC_Mean",
            "AUROC_Std",
            "F1_Mean",
            "F1_Std",
            "MCC_Mean",
            "MCC_Std",
        ]

        # Format as mean ± std
        agg_df["Accuracy"] = agg_df.apply(
            lambda row: f"{row['Acc_Mean']:.3f} ± {row['Acc_Std']:.3f}", axis=1
        )
        agg_df["AUROC"] = agg_df.apply(
            lambda row: f"{row['AUROC_Mean']:.3f} ± {row['AUROC_Std']:.3f}", axis=1
        )
        agg_df["F1"] = agg_df.apply(
            lambda row: f"{row['F1_Mean']:.3f} ± {row['F1_Std']:.3f}", axis=1
        )
        agg_df["MCC"] = agg_df.apply(
            lambda row: f"{row['MCC_Mean']:.3f} ± {row['MCC_Std']:.3f}", axis=1
        )

        # Select final columns
        final_df = agg_df[["Model", "Dataset", "Accuracy", "AUROC", "F1", "MCC"]]

        # Save in multiple formats
        self._save_table(final_df, "table1_task_performance")

    def generate_robustness_table(self) -> None:
        """Generate Table 2: Robustness (all models, all attacks)."""
        logger.info("  Table 2: Robustness")

        records = []
        for result in self.results["robustness"]:
            attack_str = self._format_attack_name(
                result.attack_name, result.attack_params
            )
            records.append(
                {
                    "Model": result.model_name,
                    "Seed": result.seed,
                    "Attack": attack_str,
                    "Clean_Acc": result.clean_accuracy,
                    "Robust_Acc": result.robust_accuracy,
                    "ASR": result.attack_success_rate,
                }
            )

        df = pd.DataFrame(records)

        # Aggregate across seeds
        agg_df = (
            df.groupby(["Model", "Attack"])
            .agg(
                {
                    "Clean_Acc": ["mean", "std"],
                    "Robust_Acc": ["mean", "std"],
                    "ASR": ["mean", "std"],
                }
            )
            .reset_index()
        )

        agg_df.columns = [
            "Model",
            "Attack",
            "Clean_Mean",
            "Clean_Std",
            "Robust_Mean",
            "Robust_Std",
            "ASR_Mean",
            "ASR_Std",
        ]

        # Format
        agg_df["Clean Acc"] = agg_df.apply(
            lambda row: f"{row['Clean_Mean']:.3f} ± {row['Clean_Std']:.3f}", axis=1
        )
        agg_df["Robust Acc"] = agg_df.apply(
            lambda row: f"{row['Robust_Mean']:.3f} ± {row['Robust_Std']:.3f}", axis=1
        )
        agg_df["ASR"] = agg_df.apply(
            lambda row: f"{row['ASR_Mean']:.3f} ± {row['ASR_Std']:.3f}", axis=1
        )

        final_df = agg_df[["Model", "Attack", "Clean Acc", "Robust Acc", "ASR"]]

        self._save_table(final_df, "table2_robustness")

    def generate_cross_site_table(self) -> None:
        """Generate Table 3: Cross-Site Generalization (AUROC drops)."""
        logger.info("  Table 3: Cross-Site Generalization")

        records = []
        for result in self.results["cross_site"]:
            records.append(
                {
                    "Model": result.model_name,
                    "Seed": result.seed,
                    "Source": result.source_dataset,
                    "Target": result.target_dataset,
                    "Source_AUROC": result.source_auroc,
                    "Target_AUROC": result.target_auroc,
                    "AUROC_Drop": result.auroc_drop,
                }
            )

        df = pd.DataFrame(records)

        # Aggregate
        agg_df = (
            df.groupby(["Model", "Target"])
            .agg(
                {
                    "Source_AUROC": ["mean", "std"],
                    "Target_AUROC": ["mean", "std"],
                    "AUROC_Drop": ["mean", "std"],
                }
            )
            .reset_index()
        )

        agg_df.columns = [
            "Model",
            "Target",
            "Source_Mean",
            "Source_Std",
            "Target_Mean",
            "Target_Std",
            "Drop_Mean",
            "Drop_Std",
        ]

        # Format
        agg_df["Source AUROC"] = agg_df.apply(
            lambda row: f"{row['Source_Mean']:.3f} ± {row['Source_Std']:.3f}", axis=1
        )
        agg_df["Target AUROC"] = agg_df.apply(
            lambda row: f"{row['Target_Mean']:.3f} ± {row['Target_Std']:.3f}", axis=1
        )
        agg_df["AUROC Drop"] = agg_df.apply(
            lambda row: f"{row['Drop_Mean']:.1f} ± {row['Drop_Std']:.1f}pp", axis=1
        )

        final_df = agg_df[
            ["Model", "Target", "Source AUROC", "Target AUROC", "AUROC Drop"]
        ]

        self._save_table(final_df, "table3_cross_site")

    def generate_calibration_table(self) -> None:
        """Generate Table 4: Calibration Metrics."""
        logger.info("  Table 4: Calibration")

        records = []
        for result in self.results["calibration"]:
            records.append(
                {
                    "Model": result.model_name,
                    "Seed": result.seed,
                    "Condition": result.condition,
                    "ECE": result.ece,
                    "MCE": result.mce,
                    "Brier": result.brier_score,
                }
            )

        df = pd.DataFrame(records)

        # Aggregate
        agg_df = (
            df.groupby(["Model", "Condition"])
            .agg(
                {
                    "ECE": ["mean", "std"],
                    "MCE": ["mean", "std"],
                    "Brier": ["mean", "std"],
                }
            )
            .reset_index()
        )

        agg_df.columns = [
            "Model",
            "Condition",
            "ECE_Mean",
            "ECE_Std",
            "MCE_Mean",
            "MCE_Std",
            "Brier_Mean",
            "Brier_Std",
        ]

        # Format
        agg_df["ECE"] = agg_df.apply(
            lambda row: f"{row['ECE_Mean']:.4f} ± {row['ECE_Std']:.4f}", axis=1
        )
        agg_df["MCE"] = agg_df.apply(
            lambda row: f"{row['MCE_Mean']:.4f} ± {row['MCE_Std']:.4f}", axis=1
        )
        agg_df["Brier"] = agg_df.apply(
            lambda row: f"{row['Brier_Mean']:.4f} ± {row['Brier_Std']:.4f}", axis=1
        )

        final_df = agg_df[["Model", "Condition", "ECE", "MCE", "Brier"]]

        self._save_table(final_df, "table4_calibration")

    def generate_statistical_tests_table(self) -> None:
        """Generate Table 5: Statistical Tests Summary."""
        logger.info("  Table 5: Statistical Tests")

        records = []
        for result in self.results["hypothesis_tests"]:
            records.append(
                {
                    "Hypothesis": result.hypothesis_name,
                    "Comparison": f"{result.group1_name} vs {result.group2_name}",
                    "t-statistic": f"{result.t_statistic:.3f}",
                    "p-value": f"{result.p_value:.4f}",
                    "Significant": "Yes" if result.significant else "No",
                    "Cohen's d": f"{result.cohens_d:.3f}",
                    "Effect": result.effect_interpretation,
                    "95% CI": f"[{result.ci_lower:.2f}, {result.ci_upper:.2f}]",
                    "Supported": "Yes" if result.hypothesis_supported else "No",
                }
            )

        df = pd.DataFrame(records)

        self._save_table(df, "table5_statistical_tests")

    # ========================================================================
    # FIGURE GENERATION
    # ========================================================================

    def generate_pareto_figures(self) -> None:
        """Generate Pareto frontier plots."""
        logger.info("  Generating Pareto frontier plots...")

        # Figure 1: Robust Accuracy vs Clean Accuracy
        self._plot_pareto_robust_vs_clean()

        # Figure 2: Robust Accuracy vs Cross-Site Drop
        self._plot_pareto_robust_vs_crosssite()

    def _plot_pareto_robust_vs_clean(self) -> None:
        """Plot Pareto frontier: Robust Accuracy vs Clean Accuracy."""
        # Collect data points
        model_data = {}

        for result in self.results["robustness"]:
            if (
                result.attack_name == "pgd"
                and abs(result.attack_params.get("epsilon", 0) - 8 / 255) < 1e-6
            ):
                key = (result.model_name, result.seed)
                if key not in model_data:
                    model_data[key] = {"model": result.model_name, "seed": result.seed}
                model_data[key]["clean_acc"] = result.clean_accuracy
                model_data[key]["robust_acc"] = result.robust_accuracy

        # Create Pareto frontier
        solutions = []
        for key, data in model_data.items():
            if "clean_acc" in data and "robust_acc" in data:
                solution = ParetoSolution(
                    objectives=np.array([data["clean_acc"], data["robust_acc"]]),
                    index=len(solutions),
                    metadata={"model": data["model"], "seed": data["seed"]},
                )
                solutions.append(solution)

        if not solutions:
            logger.warning("No data for Pareto plot (robust vs clean)")
            return

        frontier = ParetoFrontier(solutions, minimize=[False, False])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all solutions
        for solution in solutions:
            model_type = solution.metadata["model"].split("_")[0]
            color = self._get_model_color(model_type)
            marker = self._get_model_marker(model_type)
            label = model_type if solution.index == 0 else None

            ax.scatter(
                solution.objectives[0],
                solution.objectives[1],
                c=color,
                marker=marker,
                s=100,
                alpha=0.6,
                edgecolors="black",
                linewidths=1.5,
                label=label,
            )

        # Highlight Pareto front
        pareto_solutions = frontier.solutions
        if pareto_solutions:
            pareto_objs = np.array([s.objectives for s in pareto_solutions])
            # Sort by clean accuracy
            sorted_idx = np.argsort(pareto_objs[:, 0])
            pareto_objs = pareto_objs[sorted_idx]

            ax.plot(
                pareto_objs[:, 0],
                pareto_objs[:, 1],
                "r--",
                linewidth=2,
                label="Pareto Front",
                zorder=10,
            )

        ax.set_xlabel("Clean Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylabel("Robust Accuracy (PGD ε=8/255)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Figure 1: Pareto Frontier - Robustness vs Clean Accuracy",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Save
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figure1_pareto_robust_vs_clean.png", dpi=300)
        fig.savefig(self.figures_dir / "figure1_pareto_robust_vs_clean.pdf")
        plt.close(fig)

        logger.info("    Saved Figure 1: Pareto frontier (robust vs clean)")

    def _plot_pareto_robust_vs_crosssite(self) -> None:
        """Plot Pareto frontier: Robust Accuracy vs Cross-Site Drop."""
        # Collect data
        model_data = {}

        # Get robust accuracies
        for result in self.results["robustness"]:
            if (
                result.attack_name == "pgd"
                and abs(result.attack_params.get("epsilon", 0) - 8 / 255) < 1e-6
            ):
                key = (result.model_name, result.seed)
                if key not in model_data:
                    model_data[key] = {"model": result.model_name, "seed": result.seed}
                model_data[key]["robust_acc"] = result.robust_accuracy

        # Get average AUROC drops
        for key in model_data.keys():
            model_name, seed = key
            drops = [
                r.auroc_drop
                for r in self.results["cross_site"]
                if r.model_name == model_name and r.seed == seed
            ]
            if drops:
                model_data[key]["auroc_drop"] = np.mean(drops)

        # Create solutions
        solutions = []
        for key, data in model_data.items():
            if "robust_acc" in data and "auroc_drop" in data:
                solution = ParetoSolution(
                    objectives=np.array(
                        [data["robust_acc"], -data["auroc_drop"]]
                    ),  # Negative drop (higher is better)
                    index=len(solutions),
                    metadata={
                        "model": data["model"],
                        "seed": data["seed"],
                        "drop": data["auroc_drop"],
                    },
                )
                solutions.append(solution)

        if not solutions:
            logger.warning("No data for Pareto plot (robust vs cross-site)")
            return

        frontier = ParetoFrontier(solutions, minimize=[False, False])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        for solution in solutions:
            model_type = solution.metadata["model"].split("_")[0]
            color = self._get_model_color(model_type)
            marker = self._get_model_marker(model_type)

            ax.scatter(
                solution.objectives[0],
                solution.metadata["drop"],  # Plot actual drop (positive values)
                c=color,
                marker=marker,
                s=100,
                alpha=0.6,
                edgecolors="black",
                linewidths=1.5,
            )

        # Pareto front
        pareto_solutions = frontier.solutions
        if pareto_solutions:
            pareto_robust = [s.objectives[0] for s in pareto_solutions]
            pareto_drops = [s.metadata["drop"] for s in pareto_solutions]
            sorted_idx = np.argsort(pareto_robust)

            ax.plot(
                np.array(pareto_robust)[sorted_idx],
                np.array(pareto_drops)[sorted_idx],
                "r--",
                linewidth=2,
                label="Pareto Front",
                zorder=10,
            )

        ax.set_xlabel("Robust Accuracy (PGD ε=8/255)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cross-Site AUROC Drop (pp)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Figure 2: Pareto Frontier - Robustness vs Cross-Site Generalization",
            fontsize=16,
            fontweight="bold",
        )
        ax.invert_yaxis()  # Lower drop is better
        ax.grid(True, alpha=0.3)

        # Add model type legend
        legend_elements = []
        for model_type in ["baseline", "pgd-at", "trades", "tri-objective"]:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=self._get_model_marker(model_type),
                    color="w",
                    markerfacecolor=self._get_model_color(model_type),
                    markersize=10,
                    label=model_type.upper(),
                )
            )
        ax.legend(
            handles=legend_elements,
            loc="best",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        fig.tight_layout()
        fig.savefig(
            self.figures_dir / "figure2_pareto_robust_vs_crosssite.png", dpi=300
        )
        fig.savefig(self.figures_dir / "figure2_pareto_robust_vs_crosssite.pdf")
        plt.close(fig)

        logger.info("    Saved Figure 2: Pareto frontier (robust vs cross-site)")

    def generate_calibration_figures(self) -> None:
        """Generate Figure 3: Calibration reliability diagrams."""
        logger.info("  Generating calibration figures...")

        # Get unique models
        models = list(set([r.model_name for r in self.results["calibration"]]))

        # Create subplot grid
        n_models = len(models)
        ncols = 2
        nrows = (n_models + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, model_name in enumerate(models):
            ax = axes[idx]

            # Get clean calibration results
            clean_results = [
                r
                for r in self.results["calibration"]
                if r.model_name == model_name and r.condition == "clean"
            ]

            if clean_results:
                # Average across seeds
                avg_confidences = []
                avg_accuracies = []

                max_bins = max(len(r.bin_confidences) for r in clean_results)
                for i in range(max_bins):
                    confs = [
                        r.bin_confidences[i]
                        for r in clean_results
                        if i < len(r.bin_confidences)
                    ]
                    accs = [
                        r.bin_accuracies[i]
                        for r in clean_results
                        if i < len(r.bin_accuracies)
                    ]

                    if confs and accs:
                        avg_confidences.append(np.mean(confs))
                        avg_accuracies.append(np.mean(accs))

                # Plot reliability diagram
                ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
                ax.plot(
                    avg_confidences,
                    avg_accuracies,
                    "o-",
                    color="steelblue",
                    linewidth=2,
                    markersize=8,
                    label=model_name,
                )

                # Add ECE text
                avg_ece = np.mean([r.ece for r in clean_results])
                ax.text(
                    0.05,
                    0.95,
                    f"ECE = {avg_ece:.4f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            ax.set_xlabel("Confidence", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"{model_name}", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(loc="lower right")

        # Hide unused subplots
        for idx in range(len(models), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            "Figure 3: Calibration Reliability Diagrams",
            fontsize=16,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figure3_calibration_reliability.png", dpi=300)
        fig.savefig(self.figures_dir / "figure3_calibration_reliability.pdf")
        plt.close(fig)

        logger.info("    Saved Figure 3: Calibration reliability diagrams")

    def generate_cross_site_figures(self) -> None:
        """Generate cross-site generalization comparison figure."""
        logger.info("  Generating cross-site comparison figure...")

        # Collect data
        models = list(set([r.model_name for r in self.results["cross_site"]]))
        targets = list(set([r.target_dataset for r in self.results["cross_site"]]))

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(targets))
        width = 0.8 / len(models)

        for idx, model in enumerate(models):
            drops = []
            stds = []

            for target in targets:
                target_results = [
                    r.auroc_drop
                    for r in self.results["cross_site"]
                    if r.model_name == model and r.target_dataset == target
                ]
                if target_results:
                    drops.append(np.mean(target_results))
                    stds.append(
                        np.std(target_results, ddof=1) if len(target_results) > 1 else 0
                    )
                else:
                    drops.append(0)
                    stds.append(0)

            offset = (idx - len(models) / 2) * width + width / 2
            ax.bar(
                x + offset,
                drops,
                width,
                yerr=stds,
                label=model,
                capsize=5,
                alpha=0.8,
            )

        ax.set_xlabel("Target Dataset", fontsize=14, fontweight="bold")
        ax.set_ylabel("AUROC Drop (pp)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Cross-Site Generalization: AUROC Drops",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=15, ha="right")
        ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        fig.tight_layout()
        fig.savefig(self.figures_dir / "figure_cross_site_drops.png", dpi=300)
        fig.savefig(self.figures_dir / "figure_cross_site_drops.pdf")
        plt.close(fig)

        logger.info("    Saved cross-site generalization figure")

    def generate_robustness_figures(self) -> None:
        """Generate robustness comparison figure."""
        logger.info("  Generating robustness comparison figure...")

        # Focus on PGD with different epsilons
        pgd_results = [r for r in self.results["robustness"] if r.attack_name == "pgd"]

        if not pgd_results:
            return

        # Group by model and epsilon
        models = list(set([r.model_name for r in pgd_results]))
        epsilons = sorted(
            list(set([r.attack_params.get("epsilon", 0) for r in pgd_results]))
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        for model in models:
            robust_accs = []
            stds = []

            for epsilon in epsilons:
                eps_results = [
                    r.robust_accuracy
                    for r in pgd_results
                    if r.model_name == model
                    and abs(r.attack_params.get("epsilon", 0) - epsilon) < 1e-6
                ]
                if eps_results:
                    robust_accs.append(np.mean(eps_results))
                    stds.append(
                        np.std(eps_results, ddof=1) if len(eps_results) > 1 else 0
                    )
                else:
                    robust_accs.append(0)
                    stds.append(0)

            epsilon_labels = [f"{int(e*255)}/255" for e in epsilons]

            ax.errorbar(
                range(len(epsilons)),
                robust_accs,
                yerr=stds,
                marker="o",
                linewidth=2,
                markersize=8,
                capsize=5,
                label=model,
            )

        ax.set_xlabel("PGD Epsilon", fontsize=14, fontweight="bold")
        ax.set_ylabel("Robust Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(
            "Adversarial Robustness: PGD Attack",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels(epsilon_labels)
        ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        fig.tight_layout()
        fig.savefig(self.figures_dir / "figure_robustness_pgd.png", dpi=300)
        fig.savefig(self.figures_dir / "figure_robustness_pgd.pdf")
        plt.close(fig)

        logger.info("    Saved robustness comparison figure")

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================

    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report in Markdown."""
        logger.info("  Generating summary report...")

        report_path = self.output_dir / "RQ1_EVALUATION_REPORT.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# RQ1 Evaluation Report\n\n")
            f.write(
                "**Research Question 1:** Can adversarial robustness and cross-site "
            )
            f.write("generalization be jointly optimized?\n\n")
            f.write("=" * 80 + "\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f)

            # Hypothesis Testing Results
            f.write("\n## Hypothesis Testing Results\n\n")
            self._write_hypothesis_results(f)

            # Detailed Results
            f.write("\n## Detailed Results\n\n")
            f.write("### Task Performance\n")
            f.write(f"See `{self.tables_dir}/table1_task_performance.csv`\n\n")

            f.write("### Robustness Evaluation\n")
            f.write(f"See `{self.tables_dir}/table2_robustness.csv`\n\n")

            f.write("### Cross-Site Generalization\n")
            f.write(f"See `{self.tables_dir}/table3_cross_site.csv`\n\n")

            f.write("### Calibration Analysis\n")
            f.write(f"See `{self.tables_dir}/table4_calibration.csv`\n\n")

            # Figures
            f.write("\n## Figures\n\n")
            f.write("1. **Figure 1:** Pareto frontier (Robust vs Clean Accuracy)\n")
            f.write("2. **Figure 2:** Pareto frontier (Robust vs Cross-Site Drop)\n")
            f.write("3. **Figure 3:** Calibration reliability diagrams\n\n")

            # Conclusions
            f.write("\n## Conclusions\n\n")
            self._write_conclusions(f)

        logger.info(f"    Saved summary report to {report_path}")

    def _write_executive_summary(self, f) -> None:
        """Write executive summary section."""
        # Count evaluations
        n_models = len(set([r.model_name for r in self.results["task_performance"]]))
        n_datasets = len(
            set([r.dataset_name for r in self.results["task_performance"]])
        )
        n_attacks = len(set([r.attack_name for r in self.results["robustness"]]))

        f.write(f"- **Models Evaluated:** {n_models}\n")
        f.write(f"- **Datasets:** {n_datasets}\n")
        f.write(f"- **Attack Types:** {n_attacks}\n")
        f.write(
            f"- **Total Task Performance Evaluations:** {len(self.results['task_performance'])}\n"
        )
        f.write(
            f"- **Total Robustness Evaluations:** {len(self.results['robustness'])}\n"
        )
        f.write(f"- **Cross-Site Evaluations:** {len(self.results['cross_site'])}\n")
        f.write(f"- **Calibration Evaluations:** {len(self.results['calibration'])}\n")
        f.write(f"- **Hypothesis Tests:** {len(self.results['hypothesis_tests'])}\n\n")

    def _write_hypothesis_results(self, f) -> None:
        """Write hypothesis testing results."""
        for result in self.results["hypothesis_tests"]:
            f.write(
                f"### {result.hypothesis_name}: {result.hypothesis_description}\n\n"
            )
            f.write(f"- **Comparison:** {result.group1_name} vs {result.group2_name}\n")
            f.write(f"- **t-statistic:** {result.t_statistic:.3f}\n")
            f.write(f"- **p-value:** {result.p_value:.4f}\n")
            f.write(
                f"- **Significant (α=0.01):** {'**YES**' if result.significant else 'No'}\n"
            )
            f.write(
                f"- **Cohen's d:** {result.cohens_d:.3f} ({result.effect_interpretation})\n"
            )
            f.write(f"- **95% CI:** [{result.ci_lower:.2f}, {result.ci_upper:.2f}]\n")

            if result.improvement is not None:
                f.write(f"- **Improvement:** {result.improvement:.2f}\n")
            if result.threshold is not None:
                f.write(f"- **Expected Threshold:** {result.threshold:.2f}\n")

            f.write(
                f"- **Hypothesis Supported:** {'**YES** ✓' if result.hypothesis_supported else 'No ✗'}\n\n"
            )

    def _write_conclusions(self, f) -> None:
        """Write conclusions section."""
        f.write("### Key Findings\n\n")

        # Check hypothesis results
        hypotheses = {r.hypothesis_name: r for r in self.results["hypothesis_tests"]}

        if "H1a" in hypotheses:
            h1a = hypotheses["H1a"]
            if h1a.hypothesis_supported:
                f.write(
                    "✓ **H1a SUPPORTED:** Tri-objective approach achieves significant "
                )
                f.write(
                    f"robustness improvement ({h1a.improvement:.1f}pp, p={h1a.p_value:.4f})\n\n"
                )
            else:
                f.write(
                    "✗ **H1a NOT SUPPORTED:** Tri-objective robustness improvement "
                )
                f.write(
                    f"below threshold ({h1a.improvement:.1f}pp < {h1a.threshold:.1f}pp)\n\n"
                )

        if "H1b" in hypotheses:
            h1b = hypotheses["H1b"]
            if h1b.hypothesis_supported:
                f.write("✓ **H1b SUPPORTED:** Tri-objective reduces cross-site drop ")
                f.write(
                    f"significantly ({h1b.improvement:.1f}pp, p={h1b.p_value:.4f})\n\n"
                )
            else:
                f.write(
                    "✗ **H1b NOT SUPPORTED:** Cross-site drop reduction below threshold "
                )
                f.write(f"({h1b.improvement:.1f}pp < {h1b.threshold:.1f}pp)\n\n")

        if "H1c" in hypotheses:
            h1c = hypotheses["H1c"]
            if h1c.hypothesis_supported:
                f.write(
                    "✓ **H1c SUPPORTED:** PGD-AT alone does NOT improve cross-site "
                )
                f.write(f"generalization (p={h1c.p_value:.4f} ≥ 0.05)\n\n")
            else:
                f.write(
                    "✗ **H1c NOT SUPPORTED:** PGD-AT shows unexpected cross-site improvement "
                )
                f.write(f"(p={h1c.p_value:.4f})\n\n")

        f.write("\n### Research Implications\n\n")
        f.write(
            "The tri-objective optimization approach demonstrates the feasibility of "
        )
        f.write(
            "jointly optimizing adversarial robustness and cross-site generalization "
        )
        f.write(
            "in medical imaging. Traditional adversarial training methods (PGD-AT, TRADES) "
        )
        f.write(
            "improve robustness but do not address domain shift, validating the need for "
        )
        f.write("multi-objective approaches that explicitly target generalization.\n\n")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _save_table(self, df: pd.DataFrame, name: str) -> None:
        """Save table in multiple formats."""
        # CSV
        df.to_csv(self.tables_dir / f"{name}.csv", index=False)

        # LaTeX with UTF-8 encoding
        latex_str = df.to_latex(index=False, escape=False)
        with open(self.tables_dir / f"{name}.tex", "w", encoding="utf-8") as f:
            f.write(latex_str)

        # Markdown
        md_str = df.to_markdown(index=False)
        with open(self.tables_dir / f"{name}.md", "w", encoding="utf-8") as f:
            f.write(md_str)

    def _format_attack_name(self, attack_name: str, params: Dict[str, Any]) -> str:
        """Format attack name with parameters."""
        if attack_name == "fgsm":
            eps = params.get("epsilon", 0)
            return f"FGSM (ε={int(eps*255)}/255)"
        elif attack_name == "pgd":
            eps = params.get("epsilon", 0)
            steps = params.get("num_steps", 0)
            return f"PGD (ε={int(eps*255)}/255, steps={steps})"
        elif attack_name == "cw":
            conf = params.get("confidence", 0)
            return f"C&W (κ={conf:.1f})"
        elif attack_name == "autoattack":
            eps = params.get("epsilon", 0)
            return f"AutoAttack (ε={int(eps*255)}/255)"
        return attack_name.upper()

    def _get_model_color(self, model_type: str) -> str:
        """Get color for model type."""
        colors = {
            "baseline": "gray",
            "pgd-at": "blue",
            "pgd": "blue",
            "trades": "green",
            "tri-objective": "red",
            "tri": "red",
        }
        for key, color in colors.items():
            if model_type.lower().startswith(key):
                return color
        return "black"

    def _get_model_marker(self, model_type: str) -> str:
        """Get marker for model type."""
        markers = {
            "baseline": "o",
            "pgd-at": "s",
            "pgd": "s",
            "trades": "^",
            "tri-objective": "*",
            "tri": "*",
        }
        for key, marker in markers.items():
            if model_type.lower().startswith(key):
                return marker
        return "o"


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_rq1_report_generator(
    results: Dict[str, List],
    output_dir: Union[str, Path],
) -> RQ1ReportGenerator:
    """
    Factory function to create report generator.

    Args:
        results: Evaluation results dictionary
        output_dir: Output directory

    Returns:
        RQ1ReportGenerator instance
    """
    return RQ1ReportGenerator(results, Path(output_dir))
