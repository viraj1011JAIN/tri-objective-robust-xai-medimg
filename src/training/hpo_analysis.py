"""
HPO Analysis and Visualization Tools.

This module provides comprehensive analysis of hyperparameter optimization
results, including visualization, importance analysis, and trade-off exploration.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.study import Study
    from optuna.trial import FrozenTrial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Study = Any
    FrozenTrial = Any

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .hpo_config import HPOConfig
from .hpo_objective import TrialMetrics

logger = logging.getLogger(__name__)


class HPOAnalyzer:
    """
    Comprehensive analyzer for HPO study results.

    Provides:
    - Trial history visualization
    - Parameter importance analysis
    - Trade-off analysis between objectives
    - Convergence analysis
    - Best trial identification
    """

    def __init__(
        self,
        study: Study,
        config: Optional[HPOConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize HPO analyzer.

        Args:
            study: Completed Optuna study
            config: Optional HPO configuration
            output_dir: Directory to save outputs
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.study = study
        self.config = config
        self.output_dir = (
            Path(output_dir) if output_dir else Path("results/hpo_analysis")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        self.trials_df = self._create_trials_dataframe()
        self.best_trial = study.best_trial

        logger.info(f"Initialized HPO analyzer with {len(study.trials)} trials")

    def _create_trials_dataframe(self) -> pd.DataFrame:
        """
        Create pandas DataFrame from study trials.

        Returns:
            DataFrame with trial data
        """
        trials_data = []

        for trial in self.study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            row = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "duration": trial.duration.total_seconds() if trial.duration else None,
            }

            # Add parameters
            row.update(trial.params)

            # Add user attributes (metrics)
            row.update(trial.user_attrs)

            trials_data.append(row)

        df = pd.DataFrame(trials_data)

        if not df.empty:
            df = df.sort_values("trial_number")

        return df

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary statistics for HPO study.

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "study_name": self.study.study_name,
            "n_trials": len(self.study.trials),
            "n_complete": len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
            ),
            "n_pruned": len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]
            ),
            "n_failed": len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.FAIL
                ]
            ),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.best_trial.number if self.best_trial else None,
        }

        # Add parameter statistics
        if not self.trials_df.empty:
            param_stats = {}
            for param in self.study.best_params.keys():
                if param in self.trials_df.columns:
                    param_stats[param] = {
                        "mean": float(self.trials_df[param].mean()),
                        "std": float(self.trials_df[param].std()),
                        "min": float(self.trials_df[param].min()),
                        "max": float(self.trials_df[param].max()),
                        "best": self.study.best_params[param],
                    }
            summary["param_statistics"] = param_stats

        # Save to file
        import json

        summary_path = self.output_dir / "hpo_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary report to {summary_path}")

        return summary

    def plot_optimization_history(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot optimization history showing objective value over trials.

        Args:
            save: Whether to save plot
            show: Whether to display plot

        Returns:
            Matplotlib figure if matplotlib available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plot.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all trials
        ax.scatter(
            self.trials_df["trial_number"],
            self.trials_df["value"],
            alpha=0.6,
            label="Trial value",
        )

        # Plot running best
        running_best = self.trials_df["value"].cummax()
        ax.plot(
            self.trials_df["trial_number"],
            running_best,
            color="red",
            linewidth=2,
            label="Best value",
        )

        # Highlight best trial
        ax.scatter(
            [self.best_trial.number],
            [self.best_trial.value],
            color="gold",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Best trial",
            zorder=5,
        )

        ax.set_xlabel("Trial Number", fontsize=12)
        ax.set_ylabel("Objective Value", fontsize=12)
        ax.set_title("Optimization History", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "optimization_history.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved optimization history to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_parameter_importance(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot parameter importance using fANOVA.

        Args:
            save: Whether to save plot
            show: Whether to display plot

        Returns:
            Matplotlib figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plot.")
            return None

        try:
            # Compute parameter importance
            importance = optuna.importance.get_param_importances(self.study)

            # Create bar plot
            fig, ax = plt.subplots(figsize=(10, 6))

            params = list(importance.keys())
            values = list(importance.values())

            bars = ax.barh(params, values, color="steelblue")

            # Color code by importance
            colors = plt.cm.RdYlGn(np.array(values) / max(values))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel("Importance", fontsize=12)
            ax.set_ylabel("Parameter", fontsize=12)
            ax.set_title(
                "Parameter Importance (fANOVA)", fontsize=14, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()

            if save:
                save_path = self.output_dir / "parameter_importance.png"
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved parameter importance to {save_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

            return fig

        except Exception as e:
            logger.error(f"Failed to compute parameter importance: {e}")
            return None

    def plot_parameter_relationships(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot relationships between parameters and objective.

        Args:
            save: Whether to save plot
            show: Whether to display plot

        Returns:
            Matplotlib figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plot.")
            return None

        params = list(self.study.best_params.keys())
        n_params = len(params)

        fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))

        if n_params == 1:
            axes = [axes]

        for ax, param in zip(axes, params):
            # Scatter plot of parameter vs objective
            scatter = ax.scatter(
                self.trials_df[param],
                self.trials_df["value"],
                c=self.trials_df["trial_number"],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )

            # Highlight best trial
            best_param_value = self.best_trial.params[param]
            ax.scatter(
                [best_param_value],
                [self.best_trial.value],
                color="red",
                s=200,
                marker="*",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

            ax.set_xlabel(param, fontsize=12)
            ax.set_ylabel("Objective Value", fontsize=12)
            ax.set_title(f"{param} vs Objective", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label="Trial Number")

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "parameter_relationships.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved parameter relationships to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_trade_offs(
        self,
        metrics: List[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot trade-offs between different objectives.

        Args:
            metrics: List of metric names to plot
            save: Whether to save plot
            show: Whether to display plot

        Returns:
            Matplotlib figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plot.")
            return None

        metrics = metrics or ["robust_accuracy", "clean_accuracy", "cross_site_auroc"]

        # Filter metrics that exist in dataframe
        available_metrics = [m for m in metrics if m in self.trials_df.columns]

        if len(available_metrics) < 2:
            logger.warning(f"Not enough metrics available for trade-off plot")
            return None

        # Create pairwise plots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(
            n_metrics - 1,
            n_metrics - 1,
            figsize=(4 * (n_metrics - 1), 4 * (n_metrics - 1)),
        )

        if n_metrics == 2:
            axes = np.array([[axes]])
        elif n_metrics == 3:
            axes = axes.reshape(2, 2)

        for i in range(n_metrics - 1):
            for j in range(n_metrics - 1):
                if j > i:
                    axes[i, j].axis("off")
                    continue

                ax = axes[i, j] if n_metrics > 2 else axes[0, 0]

                metric_x = available_metrics[j]
                metric_y = available_metrics[i + 1]

                # Scatter plot
                scatter = ax.scatter(
                    self.trials_df[metric_x],
                    self.trials_df[metric_y],
                    c=self.trials_df["value"],
                    cmap="RdYlGn",
                    alpha=0.6,
                    s=50,
                )

                # Highlight best trial
                if (
                    metric_x in self.best_trial.user_attrs
                    and metric_y in self.best_trial.user_attrs
                ):
                    ax.scatter(
                        [self.best_trial.user_attrs[metric_x]],
                        [self.best_trial.user_attrs[metric_y]],
                        color="red",
                        s=200,
                        marker="*",
                        edgecolors="black",
                        linewidth=2,
                        zorder=5,
                    )

                ax.set_xlabel(metric_x.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel(metric_y.replace("_", " ").title(), fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add colorbar
                plt.colorbar(scatter, ax=ax, label="Objective")

        plt.suptitle("Objective Trade-offs", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / "objective_tradeoffs.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved objective trade-offs to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_convergence(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot convergence of optimization over time.

        Args:
            save: Whether to save plot
            show: Whether to display plot

        Returns:
            Matplotlib figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plot.")
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Cumulative best value
        ax1 = axes[0]
        running_best = self.trials_df["value"].cummax()

        ax1.plot(
            self.trials_df["trial_number"], running_best, linewidth=2, color="steelblue"
        )
        ax1.fill_between(
            self.trials_df["trial_number"],
            running_best.min(),
            running_best,
            alpha=0.3,
            color="steelblue",
        )

        ax1.set_xlabel("Trial Number", fontsize=12)
        ax1.set_ylabel("Best Objective Value", fontsize=12)
        ax1.set_title(
            "Convergence: Best Value Over Time", fontsize=12, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Plot 2: Improvement rate
        ax2 = axes[1]
        improvements = running_best.diff().fillna(0)

        ax2.bar(
            self.trials_df["trial_number"],
            improvements,
            color=["green" if x > 0 else "gray" for x in improvements],
            alpha=0.6,
        )

        ax2.set_xlabel("Trial Number", fontsize=12)
        ax2.set_ylabel("Improvement", fontsize=12)
        ax2.set_title("Improvement per Trial", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "convergence_analysis.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved convergence analysis to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_interactive_plots(
        self,
        save: bool = True,
    ) -> List[Path]:
        """
        Create interactive Plotly visualizations.

        Args:
            save: Whether to save HTML files

        Returns:
            List of saved file paths
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping interactive plots.")
            return []

        saved_files = []

        # 1. Interactive optimization history
        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                x=self.trials_df["trial_number"],
                y=self.trials_df["value"],
                mode="markers",
                name="Trial Value",
                marker=dict(size=8, opacity=0.6),
                hovertemplate="<b>Trial %{x}</b><br>Value: %{y:.4f}<extra></extra>",
            )
        )

        running_best = self.trials_df["value"].cummax()
        fig1.add_trace(
            go.Scatter(
                x=self.trials_df["trial_number"],
                y=running_best,
                mode="lines",
                name="Best Value",
                line=dict(color="red", width=2),
                hovertemplate="<b>Trial %{x}</b><br>Best: %{y:.4f}<extra></extra>",
            )
        )

        fig1.update_layout(
            title="Interactive Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Objective Value",
            hovermode="closest",
        )

        if save:
            path1 = self.output_dir / "interactive_optimization_history.html"
            fig1.write_html(str(path1))
            saved_files.append(path1)
            logger.info(f"Saved interactive optimization history to {path1}")

        # 2. Interactive parallel coordinates
        params = list(self.study.best_params.keys())
        dimensions = []

        for param in params:
            dimensions.append(dict(label=param, values=self.trials_df[param]))

        dimensions.append(dict(label="Objective", values=self.trials_df["value"]))

        fig2 = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=self.trials_df["value"], colorscale="RdYlGn", showscale=True
                ),
                dimensions=dimensions,
            )
        )

        fig2.update_layout(title="Interactive Parallel Coordinates Plot")

        if save:
            path2 = self.output_dir / "interactive_parallel_coordinates.html"
            fig2.write_html(str(path2))
            saved_files.append(path2)
            logger.info(f"Saved interactive parallel coordinates to {path2}")

        return saved_files

    def export_results(
        self,
        format: str = "csv",
    ) -> Path:
        """
        Export trial results to file.

        Args:
            format: Export format ("csv", "json", "excel")

        Returns:
            Path to exported file
        """
        if format == "csv":
            export_path = self.output_dir / "hpo_trials.csv"
            self.trials_df.to_csv(export_path, index=False)
        elif format == "json":
            export_path = self.output_dir / "hpo_trials.json"
            self.trials_df.to_json(export_path, orient="records", indent=2)
        elif format == "excel":
            export_path = self.output_dir / "hpo_trials.xlsx"
            self.trials_df.to_excel(export_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported trial results to {export_path}")
        return export_path

    def generate_full_report(
        self,
        include_interactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate complete analysis report with all visualizations.

        Args:
            include_interactive: Whether to generate interactive plots

        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Generating full HPO analysis report...")

        report = {
            "summary": self.generate_summary_report(),
            "plots": {},
            "exports": {},
        }

        # Generate static plots
        report["plots"]["optimization_history"] = self.plot_optimization_history()
        report["plots"]["parameter_importance"] = self.plot_parameter_importance()
        report["plots"]["parameter_relationships"] = self.plot_parameter_relationships()
        report["plots"]["trade_offs"] = self.plot_trade_offs()
        report["plots"]["convergence"] = self.plot_convergence()

        # Generate interactive plots
        if include_interactive:
            report["plots"]["interactive"] = self.create_interactive_plots()

        # Export data
        report["exports"]["csv"] = self.export_results("csv")
        report["exports"]["json"] = self.export_results("json")

        logger.info(f"Full report generated in {self.output_dir}")

        return report


def analyze_study(
    study: Study,
    config: Optional[HPOConfig] = None,
    output_dir: Optional[Path] = None,
) -> HPOAnalyzer:
    """
    Convenience function to analyze HPO study.

    Args:
        study: Completed Optuna study
        config: Optional HPO configuration
        output_dir: Directory to save outputs

    Returns:
        HPO analyzer instance
    """
    analyzer = HPOAnalyzer(study, config, output_dir)
    analyzer.generate_full_report()
    return analyzer
