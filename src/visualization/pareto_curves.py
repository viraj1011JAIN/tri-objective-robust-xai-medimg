"""
================================================================================
Pareto Curve Visualization - Phase 5.3
================================================================================
Publication-quality Pareto frontier plots for adversarial training analysis.

Features:
    - 2D/3D Pareto frontier plots
    - Knee point highlighting
    - Multi-method comparison
    - Trade-off curves
    - Customizable styling for publications

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 2025
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Publication-quality style
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


class ParetoVisualizer:
    """Visualize Pareto frontiers for multi-objective optimization."""

    def __init__(self, style: str = "seaborn-v0_8-paper"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("seaborn-v0_8-darkgrid")

        self.colors = sns.color_palette("husl", 8)

    def plot_2d_pareto(
        self,
        points: Dict[str, np.ndarray],
        pareto_points: Optional[Dict[str, np.ndarray]] = None,
        knee_point: Optional[Tuple[str, np.ndarray]] = None,
        objectives: List[str] = ["Clean Accuracy", "Robust Accuracy"],
        title: str = "Clean-Robust Accuracy Trade-off",
        save_path: Optional[Path] = None,
        show_grid: bool = True,
        show_legend: bool = True,
    ) -> plt.Figure:
        """
        Plot 2D Pareto frontier.

        Args:
            points: Dictionary mapping method_name -> points [N, 2]
            pareto_points: Optional Pareto-optimal points per method
            knee_point: Optional (method_name, knee_point_coords)
            objectives: List of objective names [x_axis, y_axis]
            title: Plot title
            save_path: Optional path to save figure
            show_grid: Show grid lines
            show_legend: Show legend

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points
        for i, (method, pts) in enumerate(points.items()):
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=[self.colors[i]],
                marker="o",
                s=100,
                alpha=0.6,
                label=method,
                edgecolors="black",
                linewidths=0.5,
            )

        # Plot Pareto frontier
        if pareto_points is not None:
            for i, (method, pts) in enumerate(pareto_points.items()):
                # Sort by x-coordinate
                sorted_indices = np.argsort(pts[:, 0])
                sorted_pts = pts[sorted_indices]

                ax.plot(
                    sorted_pts[:, 0],
                    sorted_pts[:, 1],
                    c=self.colors[i],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"{method} (Pareto)",
                )

        # Highlight knee point
        if knee_point is not None:
            method_name, knee_coords = knee_point
            ax.scatter(
                knee_coords[0],
                knee_coords[1],
                c="red",
                marker="*",
                s=500,
                label="Knee Point",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
            )

            # Annotate knee point
            ax.annotate(
                f"Knee\n({knee_coords[0]:.3f}, {knee_coords[1]:.3f})",
                xy=(knee_coords[0], knee_coords[1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3,rad=0", color="red"
                ),
            )

        # Styling
        ax.set_xlabel(objectives[0], fontweight="bold")
        ax.set_ylabel(objectives[1], fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        if show_legend:
            ax.legend(loc="best", frameon=True, shadow=True, fancybox=True)

        # Set equal aspect ratio for fair comparison
        ax.set_aspect("auto")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved 2D Pareto plot to {save_path}")

        return fig

    def plot_3d_pareto(
        self,
        points: Dict[str, np.ndarray],
        pareto_points: Optional[Dict[str, np.ndarray]] = None,
        objectives: List[str] = ["Clean Acc", "Robust Acc", "ECE"],
        title: str = "3D Pareto Frontier",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot 3D Pareto frontier.

        Args:
            points: Dictionary mapping method_name -> points [N, 3]
            pareto_points: Optional Pareto-optimal points per method
            objectives: List of objective names [x, y, z]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot all points
        for i, (method, pts) in enumerate(points.items()):
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[self.colors[i]],
                marker="o",
                s=100,
                alpha=0.6,
                label=method,
                edgecolors="black",
                linewidths=0.5,
            )

        # Plot Pareto surface
        if pareto_points is not None:
            for i, (method, pts) in enumerate(pareto_points.items()):
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c=[self.colors[i]],
                    marker="^",
                    s=200,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{method} (Pareto)",
                )

        # Styling
        ax.set_xlabel(objectives[0], fontweight="bold", labelpad=10)
        ax.set_ylabel(objectives[1], fontweight="bold", labelpad=10)
        ax.set_zlabel(objectives[2], fontweight="bold", labelpad=10)
        ax.set_title(title, fontweight="bold", pad=20)

        ax.legend(loc="best", frameon=True, shadow=True)

        # Rotate for better view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved 3D Pareto plot to {save_path}")

        return fig

    def plot_tradeoff_curves(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        x_metric: str = "clean_accuracy",
        y_metric: str = "robust_accuracy",
        parameter_key: str = "beta",
        title: str = "Clean-Robust Trade-off Curves",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot trade-off curves for different parameter values.

        Args:
            results: Nested dict {method: {param_value: {metric: value}}}
            x_metric: X-axis metric
            y_metric: Y-axis metric
            parameter_key: Parameter name (e.g., 'beta', 'epsilon')
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (method, param_results) in enumerate(results.items()):
            # Extract parameter values and metrics
            params = []
            x_vals = []
            y_vals = []

            for param_val, metrics in param_results.items():
                params.append(float(param_val))
                x_vals.append(metrics[x_metric])
                y_vals.append(metrics[y_metric])

            # Sort by parameter value
            sorted_indices = np.argsort(params)
            params = np.array(params)[sorted_indices]
            x_vals = np.array(x_vals)[sorted_indices]
            y_vals = np.array(y_vals)[sorted_indices]

            # Plot curve
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=8,
                linewidth=2,
                label=method,
                c=self.colors[i],
            )

            # Annotate parameter values
            for j, (x, y, p) in enumerate(zip(x_vals, y_vals, params)):
                if j % 2 == 0:  # Annotate every other point
                    ax.annotate(
                        f"{parameter_key}={p:.1f}",
                        xy=(x, y),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

        # Styling
        ax.set_xlabel(x_metric.replace("_", " ").title(), fontweight="bold")
        ax.set_ylabel(y_metric.replace("_", " ").title(), fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="best", frameon=True, shadow=True, fancybox=True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved trade-off curves to {save_path}")

        return fig

    def plot_comparison_bar(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Method Comparison",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot bar chart comparing methods across metrics.

        Args:
            results: Dictionary {method: {metric: value}}
            metrics: List of metrics to compare
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        methods = list(results.keys())
        num_methods = len(methods)
        num_metrics = len(metrics)

        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

        if num_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            values = [results[method][metric] for method in methods]

            bars = ax.bar(
                methods,
                values,
                color=self.colors[:num_methods],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.8,
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_ylabel("Score", fontweight="bold")
            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
            ax.set_ylim(0, 1.0)

        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved comparison bar chart to {save_path}")

        return fig


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Pareto Visualization")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load results
    with open(args.results, "r") as f:
        data = json.load(f)

    # Initialize visualizer
    visualizer = ParetoVisualizer()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example: Plot 2D Pareto frontier
    points = {
        "TRADES": np.array([[0.85, 0.70], [0.84, 0.72], [0.83, 0.74]]),
        "PGD-AT": np.array([[0.82, 0.65], [0.81, 0.68], [0.80, 0.70]]),
    }

    visualizer.plot_2d_pareto(points=points, save_path=output_dir / "pareto_2d.png")

    logging.info(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
