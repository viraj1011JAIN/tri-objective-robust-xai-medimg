"""
Multi-label calibration metrics for chest X-ray classification.

Implements calibration assessment for multi-label medical image classification:
- Class-wise Expected Calibration Error (ECE)
- Class-wise Maximum Calibration Error (MCE)
- Multi-label Brier score
- Per-class reliability diagrams
- Confidence distribution analysis

Designed for NIH ChestX-ray14 and PadChest calibration evaluation in Phase 3.6.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve


def compute_multilabel_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) for multi-label classification.

    ECE is computed per-class and then averaged.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    n_bins : int
        Number of bins for calibration (default: 15)
    strategy : str
        Binning strategy: 'uniform' or 'quantile' (default: 'uniform')

    Returns
    -------
    dict
        Dictionary containing:
        - ece_macro: Macro-averaged ECE across all classes
        - ece_per_class: Per-class ECE scores
        - ece_weighted: Weighted ECE by class support
    """
    num_classes = y_true.shape[1]
    ece_per_class = []

    for i in range(num_classes):
        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]

        if len(np.unique(y_true_class)) < 2:
            ece_per_class.append(np.nan)
            continue

        # Bin predictions
        if strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        elif strategy == "quantile":
            bin_edges = np.percentile(
                y_prob_class, np.linspace(0, 100, n_bins + 1)
            )
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute ECE for this class
        ece = 0.0
        total_samples = len(y_true_class)

        for j in range(n_bins):
            # Get samples in this bin
            mask = (y_prob_class >= bin_edges[j]) & (y_prob_class < bin_edges[j + 1])
            if j == n_bins - 1:  # Include right edge in last bin
                mask = (y_prob_class >= bin_edges[j]) & (y_prob_class <= bin_edges[j + 1])

            if not np.any(mask):
                continue

            bin_true = y_true_class[mask]
            bin_prob = y_prob_class[mask]

            # Accuracy and confidence in this bin
            bin_acc = np.mean(bin_true)
            bin_conf = np.mean(bin_prob)
            bin_size = len(bin_true)

            # Weighted contribution to ECE
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

        ece_per_class.append(float(ece))

    # Compute macro and weighted averages
    ece_per_class_clean = [e for e in ece_per_class if not np.isnan(e)]
    ece_macro = np.mean(ece_per_class_clean) if ece_per_class_clean else np.nan

    # Weighted by class support (number of positive samples)
    support_per_class = np.sum(y_true, axis=0)
    weights = support_per_class / np.sum(support_per_class)
    ece_weighted = np.sum(
        [
            w * e
            for w, e in zip(weights, ece_per_class)
            if not np.isnan(e)
        ]
    )

    return {
        "ece_macro": float(ece_macro),
        "ece_weighted": float(ece_weighted),
        "ece_per_class": ece_per_class,
    }


def compute_multilabel_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Compute Maximum Calibration Error (MCE) for multi-label classification.

    MCE is the maximum calibration error across all bins and classes.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    n_bins : int
        Number of bins for calibration (default: 15)
    strategy : str
        Binning strategy: 'uniform' or 'quantile' (default: 'uniform')

    Returns
    -------
    dict
        Dictionary containing:
        - mce_macro: Maximum calibration error across all classes
        - mce_per_class: Per-class MCE scores
    """
    num_classes = y_true.shape[1]
    mce_per_class = []

    for i in range(num_classes):
        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]

        if len(np.unique(y_true_class)) < 2:
            mce_per_class.append(np.nan)
            continue

        # Bin predictions
        if strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        elif strategy == "quantile":
            bin_edges = np.percentile(
                y_prob_class, np.linspace(0, 100, n_bins + 1)
            )
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute MCE for this class
        max_error = 0.0

        for j in range(n_bins):
            # Get samples in this bin
            mask = (y_prob_class >= bin_edges[j]) & (y_prob_class < bin_edges[j + 1])
            if j == n_bins - 1:  # Include right edge in last bin
                mask = (y_prob_class >= bin_edges[j]) & (y_prob_class <= bin_edges[j + 1])

            if not np.any(mask):
                continue

            bin_true = y_true_class[mask]
            bin_prob = y_prob_class[mask]

            # Accuracy and confidence in this bin
            bin_acc = np.mean(bin_true)
            bin_conf = np.mean(bin_prob)

            # Update maximum error
            max_error = max(max_error, np.abs(bin_acc - bin_conf))

        mce_per_class.append(float(max_error))

    # Macro MCE (worst calibration error across all classes)
    mce_per_class_clean = [e for e in mce_per_class if not np.isnan(e)]
    mce_macro = np.max(mce_per_class_clean) if mce_per_class_clean else np.nan

    return {
        "mce_macro": float(mce_macro),
        "mce_per_class": mce_per_class,
    }


def compute_multilabel_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute Brier score for multi-label classification.

    Brier score measures the mean squared difference between predicted probabilities
    and true binary labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]

    Returns
    -------
    dict
        Dictionary containing:
        - brier_score_macro: Macro-averaged Brier score
        - brier_score_per_class: Per-class Brier scores
    """
    # Per-class Brier score
    brier_per_class = np.mean((y_prob - y_true) ** 2, axis=0)

    # Macro average
    brier_macro = np.mean(brier_per_class)

    return {
        "brier_score_macro": float(brier_macro),
        "brier_score_per_class": brier_per_class.tolist(),
    }


def compute_multilabel_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Compute all calibration metrics for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    n_bins : int
        Number of bins for calibration (default: 15)
    strategy : str
        Binning strategy: 'uniform' or 'quantile' (default: 'uniform')

    Returns
    -------
    dict
        Dictionary containing ECE, MCE, and Brier score metrics
    """
    ece_results = compute_multilabel_ece(y_true, y_prob, n_bins, strategy)
    mce_results = compute_multilabel_mce(y_true, y_prob, n_bins, strategy)
    brier_results = compute_multilabel_brier_score(y_true, y_prob)

    return {
        **ece_results,
        **mce_results,
        **brier_results,
    }


def plot_multilabel_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    n_bins: int = 15,
    save_path: Optional[str] = None,
    title: str = "Multi-Label Reliability Diagram",
) -> None:
    """
    Plot reliability diagrams for all classes in multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    class_names : List[str]
        List of class names
    n_bins : int
        Number of bins for calibration (default: 15)
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    """
    num_classes = y_true.shape[1]

    # Determine grid layout
    n_cols = min(4, num_classes)
    n_rows = (num_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), squeeze=False
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, name in enumerate(class_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]

        if len(np.unique(y_true_class)) < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue

        # Compute calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_class, y_prob_class, n_bins=n_bins, strategy="uniform"
            )

            # Plot reliability curve
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker="o",
                linewidth=2,
                label="Model",
            )

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

            # Customize subplot
            ax.set_xlabel("Predicted probability", fontsize=9)
            ax.set_ylabel("True probability", fontsize=9)
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)[:20]}",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_title(name, fontsize=10, fontweight="bold")

    # Hide unused subplots
    for i in range(num_classes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_multilabel_confidence_histogram(
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Multi-Label Confidence Distribution",
) -> None:
    """
    Plot confidence distribution histograms for all classes.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    class_names : List[str]
        List of class names
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    """
    num_classes = y_prob.shape[1]

    # Determine grid layout
    n_cols = min(4, num_classes)
    n_rows = (num_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), squeeze=False
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, name in enumerate(class_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        y_prob_class = y_prob[:, i]

        # Plot histogram
        ax.hist(y_prob_class, bins=50, edgecolor="black", alpha=0.7)

        # Customize subplot
        ax.set_xlabel("Predicted probability", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim([0, 1])

    # Hide unused subplots
    for i in range(num_classes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_class_wise_calibration_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute calibration curves for all classes.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    class_names : List[str]
        List of class names
    n_bins : int
        Number of bins for calibration

    Returns
    -------
    dict
        Dictionary mapping class names to calibration curve data:
        - fraction_of_positives: True frequencies
        - mean_predicted_value: Mean predicted probabilities
    """
    num_classes = y_true.shape[1]
    calibration_data = {}

    for i, name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_prob_class = y_prob[:, i]

        if len(np.unique(y_true_class)) < 2:
            calibration_data[name] = {
                "fraction_of_positives": None,
                "mean_predicted_value": None,
                "error": "Insufficient data",
            }
            continue

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_class, y_prob_class, n_bins=n_bins, strategy="uniform"
            )

            calibration_data[name] = {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
            }
        except Exception as e:
            calibration_data[name] = {
                "fraction_of_positives": None,
                "mean_predicted_value": None,
                "error": str(e),
            }

    return calibration_data
