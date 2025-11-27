#!/usr/bin/env python3
"""
Real-Time Tri-Objective Training Monitor

This module provides comprehensive real-time monitoring for tri-objective training runs,
displaying loss components, accuracy metrics, XAI quality (SSIM, TCAV), and GPU utilization.

Features:
- Real-time plotting with matplotlib (3x3 subplot grid)
- MLflow metrics integration
- Anomaly detection (loss spikes, NaN values, overfitting)
- GPU monitoring (memory, utilization)
- Target lines for each metric
- Alert system for training issues
- Command-line interface

Author: Dissertation Project
Date: November 2025
"""

import argparse
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Use TkAgg backend for interactive plots
matplotlib.use("TkAgg")

# Optional GPU monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. GPU monitoring disabled.")


@dataclass
class MetricConfig:
    """Configuration for a single metric."""

    name: str
    display_name: str
    target: Optional[float] = None
    lower_better: bool = True
    anomaly_threshold: float = 3.0  # Standard deviations
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class AlertConfig:
    """Configuration for alert thresholds."""

    loss_spike_threshold: float = 2.0  # Multiplier over recent average
    nan_check: bool = True
    overfitting_gap: float = 0.15  # Train-val accuracy gap
    gpu_memory_threshold: float = 0.95  # 95% utilization
    min_samples: int = 10  # Minimum samples for anomaly detection


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    steps: List[int] = field(default_factory=list)

    # Loss components
    total_loss: List[float] = field(default_factory=list)
    ce_loss: List[float] = field(default_factory=list)
    robust_loss: List[float] = field(default_factory=list)
    xai_loss: List[float] = field(default_factory=list)

    # Accuracies
    train_acc_clean: List[float] = field(default_factory=list)
    train_acc_adv: List[float] = field(default_factory=list)
    val_acc_clean: List[float] = field(default_factory=list)
    val_acc_adv: List[float] = field(default_factory=list)

    # XAI metrics
    ssim: List[float] = field(default_factory=list)
    tcav: List[float] = field(default_factory=list)

    # GPU metrics
    gpu_memory: List[float] = field(default_factory=list)
    gpu_util: List[float] = field(default_factory=list)

    # Timestamps
    timestamps: List[float] = field(default_factory=list)


class GPUMonitor:
    """Monitor GPU utilization and memory usage."""

    def __init__(self, device_id: int = 0):
        """
        Initialize GPU monitor.

        Args:
            device_id: GPU device ID to monitor
        """
        self.device_id = device_id
        self.available = PYNVML_AVAILABLE

        if self.available:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                warnings.warn(f"Failed to initialize GPU monitoring: {e}")
                self.available = False

    def get_metrics(self) -> Tuple[float, float]:
        """
        Get current GPU metrics.

        Returns:
            Tuple of (memory_used_fraction, utilization_fraction)
        """
        if not self.available:
            return 0.0, 0.0

        try:
            # Memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = mem_info.used / mem_info.total

            # GPU utilization
            util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            utilization = util_info.gpu / 100.0

            return memory_used, utilization
        except Exception as e:
            warnings.warn(f"Failed to get GPU metrics: {e}")
            return 0.0, 0.0

    def cleanup(self):
        """Cleanup GPU monitoring resources."""
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class AnomalyDetector:
    """Detect anomalies in training metrics."""

    def __init__(self, config: AlertConfig):
        """
        Initialize anomaly detector.

        Args:
            config: Alert configuration
        """
        self.config = config
        self.metric_windows = defaultdict(lambda: deque(maxlen=50))

    def check_loss_spike(self, metric_name: str, value: float) -> Optional[str]:
        """
        Check for loss spikes.

        Args:
            metric_name: Name of the loss metric
            value: Current value

        Returns:
            Alert message if spike detected, None otherwise
        """
        window = self.metric_windows[metric_name]

        if len(window) < self.config.min_samples:
            window.append(value)
            return None

        recent_mean = np.mean(list(window))
        recent_std = np.std(list(window))

        if (
            recent_std > 0
            and (value - recent_mean) > self.config.loss_spike_threshold * recent_std
        ):
            return f"SPIKE: {metric_name} = {value:.4f} (mean={recent_mean:.4f}, std={recent_std:.4f})"

        window.append(value)
        return None

    def check_nan(self, metric_name: str, value: float) -> Optional[str]:
        """
        Check for NaN values.

        Args:
            metric_name: Name of the metric
            value: Current value

        Returns:
            Alert message if NaN detected, None otherwise
        """
        if self.config.nan_check and (np.isnan(value) or np.isinf(value)):
            return f"NaN/INF: {metric_name} = {value}"
        return None

    def check_overfitting(self, train_acc: float, val_acc: float) -> Optional[str]:
        """
        Check for overfitting.

        Args:
            train_acc: Training accuracy
            val_acc: Validation accuracy

        Returns:
            Alert message if overfitting detected, None otherwise
        """
        gap = train_acc - val_acc
        if gap > self.config.overfitting_gap:
            return (
                f"OVERFITTING: Train={train_acc:.3f}, Val={val_acc:.3f}, Gap={gap:.3f}"
            )
        return None

    def check_gpu_memory(self, memory_used: float) -> Optional[str]:
        """
        Check GPU memory usage.

        Args:
            memory_used: Fraction of GPU memory used (0-1)

        Returns:
            Alert message if threshold exceeded, None otherwise
        """
        if memory_used > self.config.gpu_memory_threshold:
            return f"GPU MEMORY: {memory_used*100:.1f}% used (threshold={self.config.gpu_memory_threshold*100:.1f}%)"
        return None


class TriObjectiveMonitor:
    """
    Real-time monitor for tri-objective training.

    Displays comprehensive metrics including loss components, accuracies,
    XAI quality metrics, and GPU utilization with anomaly detection.
    """

    # Metric configurations
    METRICS = [
        MetricConfig("total_loss", "Total Loss", target=0.5, lower_better=True),
        MetricConfig("ce_loss", "CE Loss", target=0.3, lower_better=True),
        MetricConfig("robust_loss", "Robust Loss", target=0.2, lower_better=True),
        MetricConfig("xai_loss", "XAI Loss", target=0.1, lower_better=True),
        MetricConfig(
            "train_acc_clean",
            "Train Acc (Clean)",
            target=0.95,
            lower_better=False,
            min_value=0,
            max_value=1,
        ),
        MetricConfig(
            "val_acc_clean",
            "Val Acc (Clean)",
            target=0.90,
            lower_better=False,
            min_value=0,
            max_value=1,
        ),
        MetricConfig(
            "train_acc_adv",
            "Train Acc (Adv)",
            target=0.85,
            lower_better=False,
            min_value=0,
            max_value=1,
        ),
        MetricConfig(
            "val_acc_adv",
            "Val Acc (Adv)",
            target=0.80,
            lower_better=False,
            min_value=0,
            max_value=1,
        ),
        MetricConfig(
            "ssim", "SSIM", target=0.90, lower_better=False, min_value=0, max_value=1
        ),
    ]

    def __init__(
        self,
        run_id: str,
        refresh_interval: int = 5,
        window_size: int = 100,
        tracking_uri: Optional[str] = None,
        alert_config: Optional[AlertConfig] = None,
    ):
        """
        Initialize tri-objective monitor.

        Args:
            run_id: MLflow run ID to monitor
            refresh_interval: Refresh interval in seconds
            window_size: Number of recent points to display
            tracking_uri: MLflow tracking URI
            alert_config: Alert configuration
        """
        self.run_id = run_id
        self.refresh_interval = refresh_interval
        self.window_size = window_size
        self.tracking_uri = tracking_uri or "file:./mlruns"

        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

        # Verify run exists
        try:
            self.run = self.client.get_run(run_id)
        except Exception as e:
            raise ValueError(f"Run {run_id} not found: {e}")

        # Initialize components
        self.metrics = TrainingMetrics()
        self.gpu_monitor = GPUMonitor()
        self.anomaly_detector = AnomalyDetector(alert_config or AlertConfig())
        self.alerts = deque(maxlen=5)  # Keep last 5 alerts

        # Setup plot
        self.fig = None
        self.axes = None
        self.lines = {}
        self.target_lines = {}
        self._setup_plot()

        print(f"Monitoring run: {run_id}")
        print(f"Run name: {self.run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"Refresh interval: {refresh_interval}s")
        print(f"Tracking URI: {self.tracking_uri}")
        print("-" * 80)

    def _setup_plot(self):
        """Setup matplotlib figure and subplots."""
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.canvas.manager.set_window_title(
            f"Tri-Objective Training Monitor - {self.run_id[:8]}"
        )

        # Create 3x3 grid
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # Create subplots
        self.axes = {
            "total_loss": self.fig.add_subplot(gs[0, 0]),
            "loss_components": self.fig.add_subplot(gs[0, 1]),
            "train_acc": self.fig.add_subplot(gs[0, 2]),
            "val_acc": self.fig.add_subplot(gs[1, 0]),
            "acc_comparison": self.fig.add_subplot(gs[1, 1]),
            "xai_metrics": self.fig.add_subplot(gs[1, 2]),
            "gpu_memory": self.fig.add_subplot(gs[2, 0]),
            "gpu_util": self.fig.add_subplot(gs[2, 1]),
            "alerts": self.fig.add_subplot(gs[2, 2]),
        }

        # Configure subplots
        self._configure_subplot(self.axes["total_loss"], "Total Loss", "Step", "Loss")
        self._configure_subplot(
            self.axes["loss_components"], "Loss Components", "Step", "Loss"
        )
        self._configure_subplot(
            self.axes["train_acc"], "Training Accuracy", "Step", "Accuracy"
        )
        self._configure_subplot(
            self.axes["val_acc"], "Validation Accuracy", "Step", "Accuracy"
        )
        self._configure_subplot(
            self.axes["acc_comparison"], "Train vs Val", "Step", "Accuracy"
        )
        self._configure_subplot(
            self.axes["xai_metrics"], "XAI Quality", "Step", "Score"
        )
        self._configure_subplot(
            self.axes["gpu_memory"], "GPU Memory", "Step", "Usage (%)"
        )
        self._configure_subplot(
            self.axes["gpu_util"], "GPU Utilization", "Step", "Util (%)"
        )
        self._configure_subplot(self.axes["alerts"], "Alerts", "", "")
        self.axes["alerts"].axis("off")

        # Add title
        self.fig.suptitle(
            f"Tri-Objective Training Monitor - Run: {self.run_id[:8]}",
            fontsize=16,
            fontweight="bold",
        )

    def _configure_subplot(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """
        Configure a subplot.

        Args:
            ax: Matplotlib axes
            title: Subplot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    def _fetch_metrics(self) -> bool:
        """
        Fetch latest metrics from MLflow.

        Returns:
            True if new metrics were fetched, False otherwise
        """
        try:
            # Get metric history
            metric_names = [
                "train/total_loss",
                "train/ce_loss",
                "train/robust_loss",
                "train/xai_loss",
                "train/acc_clean",
                "train/acc_adv",
                "val/acc_clean",
                "val/acc_adv",
                "xai/ssim",
                "xai/tcav_score",
            ]

            # Fetch all metrics
            new_data = False
            for metric_name in metric_names:
                try:
                    history = self.client.get_metric_history(self.run_id, metric_name)
                    if not history:
                        continue

                    # Get latest point
                    latest = history[-1]
                    step = latest.step
                    value = latest.value
                    timestamp = latest.timestamp / 1000.0  # Convert to seconds

                    # Check if this is new data
                    if not self.metrics.steps or step > self.metrics.steps[-1]:
                        new_data = True

                        # Only add if it's a new step
                        if step not in self.metrics.steps:
                            self.metrics.steps.append(step)
                            self.metrics.timestamps.append(timestamp)

                            # Initialize all metrics for this step
                            for attr in [
                                "total_loss",
                                "ce_loss",
                                "robust_loss",
                                "xai_loss",
                                "train_acc_clean",
                                "train_acc_adv",
                                "val_acc_clean",
                                "val_acc_adv",
                                "ssim",
                                "tcav",
                            ]:
                                getattr(self.metrics, attr).append(np.nan)

                        # Update the specific metric
                        idx = self.metrics.steps.index(step)
                        if "total_loss" in metric_name:
                            self.metrics.total_loss[idx] = value
                        elif "ce_loss" in metric_name:
                            self.metrics.ce_loss[idx] = value
                        elif "robust_loss" in metric_name:
                            self.metrics.robust_loss[idx] = value
                        elif "xai_loss" in metric_name:
                            self.metrics.xai_loss[idx] = value
                        elif "train/acc_clean" in metric_name:
                            self.metrics.train_acc_clean[idx] = value
                        elif "train/acc_adv" in metric_name:
                            self.metrics.train_acc_adv[idx] = value
                        elif "val/acc_clean" in metric_name:
                            self.metrics.val_acc_clean[idx] = value
                        elif "val/acc_adv" in metric_name:
                            self.metrics.val_acc_adv[idx] = value
                        elif "ssim" in metric_name:
                            self.metrics.ssim[idx] = value
                        elif "tcav" in metric_name:
                            self.metrics.tcav[idx] = value

                except Exception as e:
                    # Metric might not exist yet
                    continue

            # Get GPU metrics
            if self.gpu_monitor.available:
                memory_used, utilization = self.gpu_monitor.get_metrics()
                if self.metrics.steps:
                    # Update GPU metrics for latest step
                    if len(self.metrics.gpu_memory) < len(self.metrics.steps):
                        self.metrics.gpu_memory.append(memory_used)
                        self.metrics.gpu_util.append(utilization)
                    else:
                        self.metrics.gpu_memory[-1] = memory_used
                        self.metrics.gpu_util[-1] = utilization

            return new_data

        except Exception as e:
            warnings.warn(f"Failed to fetch metrics: {e}")
            return False

    def _detect_anomalies(self):
        """Detect and report anomalies in recent metrics."""
        if not self.metrics.steps or len(self.metrics.steps) < 2:
            return

        # Get latest values
        idx = -1

        # Check loss spikes
        for loss_name in ["total_loss", "ce_loss", "robust_loss", "xai_loss"]:
            values = getattr(self.metrics, loss_name)
            if values and not np.isnan(values[idx]):
                alert = self.anomaly_detector.check_loss_spike(loss_name, values[idx])
                if alert:
                    self.alerts.append(alert)

        # Check NaN values
        for metric_name in ["total_loss", "train_acc_clean", "val_acc_clean"]:
            values = getattr(self.metrics, metric_name)
            if values:
                alert = self.anomaly_detector.check_nan(metric_name, values[idx])
                if alert:
                    self.alerts.append(alert)

        # Check overfitting
        if (
            self.metrics.train_acc_clean
            and self.metrics.val_acc_clean
            and not np.isnan(self.metrics.train_acc_clean[idx])
            and not np.isnan(self.metrics.val_acc_clean[idx])
        ):
            alert = self.anomaly_detector.check_overfitting(
                self.metrics.train_acc_clean[idx], self.metrics.val_acc_clean[idx]
            )
            if alert:
                self.alerts.append(alert)

        # Check GPU memory
        if self.metrics.gpu_memory and self.metrics.gpu_memory[idx] > 0:
            alert = self.anomaly_detector.check_gpu_memory(self.metrics.gpu_memory[idx])
            if alert:
                self.alerts.append(alert)

    def _update_plot(self, frame):
        """
        Update plot with latest data.

        Args:
            frame: Frame number (from FuncAnimation)
        """
        # Fetch new metrics
        new_data = self._fetch_metrics()

        if not new_data or not self.metrics.steps:
            return

        # Detect anomalies
        self._detect_anomalies()

        # Get window of data to display
        window_start = max(0, len(self.metrics.steps) - self.window_size)
        steps = self.metrics.steps[window_start:]

        # Update total loss
        self._plot_metric(
            self.axes["total_loss"],
            steps,
            self.metrics.total_loss[window_start:],
            "Total Loss",
            color="red",
            target=0.5,
        )

        # Update loss components
        ax = self.axes["loss_components"]
        ax.clear()
        self._configure_subplot(ax, "Loss Components", "Step", "Loss")
        if self.metrics.ce_loss:
            ax.plot(
                steps,
                self.metrics.ce_loss[window_start:],
                "b-",
                label="CE",
                linewidth=2,
            )
        if self.metrics.robust_loss:
            ax.plot(
                steps,
                self.metrics.robust_loss[window_start:],
                "r-",
                label="Robust",
                linewidth=2,
            )
        if self.metrics.xai_loss:
            ax.plot(
                steps,
                self.metrics.xai_loss[window_start:],
                "g-",
                label="XAI",
                linewidth=2,
            )
        ax.legend(fontsize=8)

        # Update training accuracy
        ax = self.axes["train_acc"]
        ax.clear()
        self._configure_subplot(ax, "Training Accuracy", "Step", "Accuracy")
        if self.metrics.train_acc_clean:
            ax.plot(
                steps,
                self.metrics.train_acc_clean[window_start:],
                "b-",
                label="Clean",
                linewidth=2,
            )
        if self.metrics.train_acc_adv:
            ax.plot(
                steps,
                self.metrics.train_acc_adv[window_start:],
                "r-",
                label="Adv",
                linewidth=2,
            )
        ax.axhline(y=0.95, color="g", linestyle="--", alpha=0.5, label="Target")
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)

        # Update validation accuracy
        ax = self.axes["val_acc"]
        ax.clear()
        self._configure_subplot(ax, "Validation Accuracy", "Step", "Accuracy")
        if self.metrics.val_acc_clean:
            ax.plot(
                steps,
                self.metrics.val_acc_clean[window_start:],
                "b-",
                label="Clean",
                linewidth=2,
            )
        if self.metrics.val_acc_adv:
            ax.plot(
                steps,
                self.metrics.val_acc_adv[window_start:],
                "r-",
                label="Adv",
                linewidth=2,
            )
        ax.axhline(y=0.90, color="g", linestyle="--", alpha=0.5, label="Target")
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)

        # Update accuracy comparison
        ax = self.axes["acc_comparison"]
        ax.clear()
        self._configure_subplot(ax, "Train vs Val (Clean)", "Step", "Accuracy")
        if self.metrics.train_acc_clean:
            ax.plot(
                steps,
                self.metrics.train_acc_clean[window_start:],
                "b-",
                label="Train",
                linewidth=2,
            )
        if self.metrics.val_acc_clean:
            ax.plot(
                steps,
                self.metrics.val_acc_clean[window_start:],
                "r-",
                label="Val",
                linewidth=2,
            )
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)

        # Update XAI metrics
        ax = self.axes["xai_metrics"]
        ax.clear()
        self._configure_subplot(ax, "XAI Quality", "Step", "Score")
        if self.metrics.ssim:
            ax.plot(
                steps, self.metrics.ssim[window_start:], "b-", label="SSIM", linewidth=2
            )
        if self.metrics.tcav:
            ax.plot(
                steps, self.metrics.tcav[window_start:], "g-", label="TCAV", linewidth=2
            )
        ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="Target")
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)

        # Update GPU memory
        if self.metrics.gpu_memory and any(x > 0 for x in self.metrics.gpu_memory):
            self._plot_metric(
                self.axes["gpu_memory"],
                steps,
                [x * 100 for x in self.metrics.gpu_memory[window_start:]],
                "GPU Memory",
                color="orange",
                ylabel="Usage (%)",
                target=95,
            )

        # Update GPU utilization
        if self.metrics.gpu_util and any(x > 0 for x in self.metrics.gpu_util):
            self._plot_metric(
                self.axes["gpu_util"],
                steps,
                [x * 100 for x in self.metrics.gpu_util[window_start:]],
                "GPU Utilization",
                color="purple",
                ylabel="Util (%)",
            )

        # Update alerts
        ax = self.axes["alerts"]
        ax.clear()
        ax.axis("off")
        ax.set_title("Recent Alerts", fontsize=10, fontweight="bold")

        if self.alerts:
            alert_text = "\n".join(list(self.alerts)[-5:])
            ax.text(
                0.05,
                0.95,
                alert_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No alerts",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="green",
            )

        # Update figure title with latest stats
        if self.metrics.steps:
            latest_step = self.metrics.steps[-1]
            status_text = f"Step: {latest_step}"

            if self.metrics.total_loss and not np.isnan(self.metrics.total_loss[-1]):
                status_text += f" | Loss: {self.metrics.total_loss[-1]:.4f}"

            if self.metrics.val_acc_clean and not np.isnan(
                self.metrics.val_acc_clean[-1]
            ):
                status_text += f" | Val Acc: {self.metrics.val_acc_clean[-1]:.3f}"

            self.fig.suptitle(
                f"Tri-Objective Training Monitor - Run: {self.run_id[:8]}\n{status_text}",
                fontsize=16,
                fontweight="bold",
            )

    def _plot_metric(
        self,
        ax: plt.Axes,
        steps: List[int],
        values: List[float],
        title: str,
        color: str = "blue",
        ylabel: str = "Value",
        target: Optional[float] = None,
    ):
        """
        Plot a single metric.

        Args:
            ax: Matplotlib axes
            steps: Step numbers
            values: Metric values
            title: Plot title
            color: Line color
            ylabel: Y-axis label
            target: Target value line
        """
        ax.clear()
        self._configure_subplot(ax, title, "Step", ylabel)

        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        valid_steps = [s for s, m in zip(steps, valid_mask) if m]
        valid_values = [v for v, m in zip(values, valid_mask) if m]

        if valid_steps:
            ax.plot(valid_steps, valid_values, color=color, linewidth=2)

            if target is not None:
                ax.axhline(
                    y=target,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Target: {target}",
                )
                ax.legend(fontsize=8)

    def start(self):
        """Start the real-time monitoring loop."""
        try:
            # Create animation
            anim = FuncAnimation(
                self.fig,
                self._update_plot,
                interval=self.refresh_interval * 1000,
                cache_frame_data=False,
            )

            plt.tight_layout()
            plt.show()

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        self.gpu_monitor.cleanup()
        plt.close(self.fig)


def get_latest_run(tracking_uri: str) -> Optional[str]:
    """
    Get the latest active run.

    Args:
        tracking_uri: MLflow tracking URI

    Returns:
        Run ID of the latest active run, or None if no runs found
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Search for active runs
    runs = client.search_runs(
        experiment_ids=["0"],  # Default experiment
        filter_string="",
        order_by=["start_time DESC"],
        max_results=10,
    )

    # Find most recent run
    for run in runs:
        if run.info.status == "RUNNING":
            return run.info.run_id

    # If no running runs, return most recent
    if runs:
        return runs[0].info.run_id

    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time monitoring for tri-objective training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor specific run
  python monitor_training.py --run_id abc123def456

  # Monitor latest run
  python monitor_training.py --latest

  # Custom refresh interval and window
  python monitor_training.py --run_id abc123 --refresh 10 --window 200

  # Custom tracking URI
  python monitor_training.py --run_id abc123 --tracking_uri http://localhost:5000
        """,
    )

    parser.add_argument("--run_id", type=str, help="MLflow run ID to monitor")

    parser.add_argument("--latest", action="store_true", help="Monitor the latest run")

    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Number of recent points to display (default: 100)",
    )

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI (default: file:./mlruns)",
    )

    args = parser.parse_args()

    # Determine run ID
    if args.latest:
        print("Searching for latest run...")
        run_id = get_latest_run(args.tracking_uri)
        if not run_id:
            print("ERROR: No runs found")
            sys.exit(1)
        print(f"Found latest run: {run_id}")
    elif args.run_id:
        run_id = args.run_id
    else:
        print("ERROR: Must specify either --run_id or --latest")
        parser.print_help()
        sys.exit(1)

    # Create monitor
    try:
        monitor = TriObjectiveMonitor(
            run_id=run_id,
            refresh_interval=args.refresh,
            window_size=args.window,
            tracking_uri=args.tracking_uri,
        )

        print("\nStarting real-time monitoring...")
        print("Press Ctrl+C to stop\n")

        monitor.start()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
