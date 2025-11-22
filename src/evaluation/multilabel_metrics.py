"""
Multi-label evaluation metrics for chest X-ray classification.

Implements metrics specifically designed for multi-label medical image classification:
- Per-class AUROC (macro, micro, weighted)
- Hamming loss
- Subset accuracy
- Per-disease metrics (precision, recall, F1)
- Multi-label confusion matrix
- Coverage error, ranking loss
- Bootstrap confidence intervals for multi-label metrics

Designed for NIH ChestX-ray14 and PadChest evaluation in Phase 3.6.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    coverage_error,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_multilabel_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute AUROC metrics for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    class_names : List[str], optional
        List of class names for per-class reporting

    Returns
    -------
    dict
        Dictionary containing:
        - auroc_macro: Macro-averaged AUROC
        - auroc_micro: Micro-averaged AUROC
        - auroc_weighted: Weighted AUROC by class support
        - auroc_per_class: Per-class AUROC scores
        - class_names: List of class names (if provided)
    """
    num_classes = y_true.shape[1]

    # Macro AUROC (average of per-class AUROC)
    auroc_macro = roc_auc_score(y_true, y_prob, average="macro")

    # Micro AUROC (aggregate all classes)
    auroc_micro = roc_auc_score(y_true, y_prob, average="micro")

    # Weighted AUROC (weighted by class support)
    auroc_weighted = roc_auc_score(y_true, y_prob, average="weighted")

    # Per-class AUROC
    auroc_per_class = []
    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) < 2:
            # Skip classes with only one label in the dataset
            auroc_per_class.append(np.nan)
        else:
            auroc_per_class.append(roc_auc_score(y_true[:, i], y_prob[:, i]))

    results = {
        "auroc_macro": float(auroc_macro),
        "auroc_micro": float(auroc_micro),
        "auroc_weighted": float(auroc_weighted),
        "auroc_per_class": auroc_per_class,
    }

    if class_names is not None:
        results["class_names"] = class_names
        results["auroc_by_class"] = {
            name: score for name, score in zip(class_names, auroc_per_class)
        }

    return results


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-label classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_pred : np.ndarray
        Predicted binary labels (after thresholding), shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    class_names : List[str], optional
        List of class names for per-class reporting
    threshold : float
        Threshold for converting probabilities to binary predictions (default: 0.5)

    Returns
    -------
    dict
        Dictionary containing:
        - AUROC metrics (macro, micro, weighted, per-class)
        - Hamming loss
        - Subset accuracy
        - Per-class precision, recall, F1
        - Coverage error
        - Ranking loss
        - Label ranking average precision
    """
    # Ensure binary predictions if not already
    if y_pred is None or y_pred.shape != y_true.shape:
        y_pred = (y_prob >= threshold).astype(int)

    # AUROC metrics
    auroc_results = compute_multilabel_auroc(y_true, y_prob, class_names)

    # Hamming loss (fraction of incorrect labels)
    hamming = hamming_loss(y_true, y_pred)

    # Subset accuracy (exact match ratio)
    subset_acc = np.mean(np.all(y_true == y_pred, axis=1))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Ranking metrics
    coverage_err = coverage_error(y_true, y_prob)
    ranking_loss = label_ranking_loss(y_true, y_prob)
    lrap = label_ranking_average_precision_score(y_true, y_prob)

    results = {
        **auroc_results,
        "hamming_loss": float(hamming),
        "subset_accuracy": float(subset_acc),
        "coverage_error": float(coverage_err),
        "ranking_loss": float(ranking_loss),
        "label_ranking_avg_precision": float(lrap),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }

    if class_names is not None:
        results["per_class_metrics"] = {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auroc": auroc_results["auroc_per_class"][i],
            }
            for i, name in enumerate(class_names)
        }

    return results


def compute_multilabel_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-class confusion matrices for multi-label classification.

    For each class, computes a 2x2 confusion matrix:
    [[TN, FP],
     [FN, TP]]

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_pred : np.ndarray
        Predicted binary labels, shape [N, C]
    class_names : List[str], optional
        List of class names

    Returns
    -------
    dict
        Dictionary containing:
        - confusion_matrices: List of [2, 2] confusion matrices per class
        - class_names: List of class names (if provided)
        - per_class_cm: Dict mapping class names to confusion matrices
    """
    # Compute multi-label confusion matrix (returns [C, 2, 2] array)
    cm_array = multilabel_confusion_matrix(y_true, y_pred)

    results = {
        "confusion_matrices": cm_array.tolist(),
        "num_classes": cm_array.shape[0],
    }

    if class_names is not None:
        results["class_names"] = class_names
        results["per_class_cm"] = {
            name: cm_array[i].tolist() for i, name in enumerate(class_names)
        }

    return results


def plot_multilabel_auroc_per_class(
    auroc_scores: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Class AUROC",
) -> None:
    """
    Plot per-class AUROC scores as a horizontal bar chart.

    Parameters
    ----------
    auroc_scores : np.ndarray
        Per-class AUROC scores, shape [C]
    class_names : List[str]
        List of class names
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.4)))

    # Sort by AUROC score
    sorted_indices = np.argsort(auroc_scores)
    sorted_scores = auroc_scores[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]

    # Create horizontal bar chart
    colors = plt.cm.RdYlGn(sorted_scores)
    ax.barh(range(len(sorted_names)), sorted_scores, color=colors)

    # Customize plot
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, score in enumerate(sorted_scores):
        ax.text(score + 0.02, i, f"{score:.3f}", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_multilabel_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Multi-Label ROC Curves",
) -> None:
    """
    Plot ROC curves for all classes in multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    class_names : List[str]
        List of class names
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    """
    num_classes = y_true.shape[1]

    # Compute micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Compute macro-average ROC curve
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot per-class ROC curves
    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        # Interpolate for macro-average
        mean_tpr += np.interp(all_fpr, fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            alpha=0.3,
            linewidth=1,
            label=f"{class_names[i]} (AUC={roc_auc:.3f})",
        )

    # Plot micro-average
    ax.plot(
        fpr_micro,
        tpr_micro,
        color="deeppink",
        linewidth=2,
        linestyle=":",
        label=f"Micro-average (AUC={roc_auc_micro:.3f})",
    )

    # Plot macro-average
    mean_tpr /= num_classes
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    roc_auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr,
        mean_tpr,
        color="navy",
        linewidth=2,
        linestyle="--",
        label=f"Macro-average (AUC={roc_auc_macro:.3f})",
    )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)

    # Customize plot
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_bootstrap_ci_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for multi-label metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    metric_fn : callable
        Metric function that takes (y_true, y_prob, **kwargs) and returns a scalar
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    confidence_level : float
        Confidence level for CI (default: 0.95)
    random_state : int
        Random seed for reproducibility
    **metric_kwargs
        Additional keyword arguments for metric_fn

    Returns
    -------
    tuple
        (metric_value, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    n_samples = y_true.shape[0]

    # Compute original metric
    metric_value = metric_fn(y_true, y_prob, **metric_kwargs)

    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        # Compute metric on bootstrap sample
        try:
            score = metric_fn(y_true_boot, y_prob_boot, **metric_kwargs)
            bootstrap_scores.append(score)
        except Exception:
            # Skip failed bootstrap samples
            continue

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return float(metric_value), float(lower_bound), float(upper_bound)


def compute_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> np.ndarray:
    """
    Compute optimal per-class thresholds for multi-label classification.

    For each class, searches for the threshold that maximizes the specified metric.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities, shape [N, C]
    metric : str
        Metric to optimize ('f1', 'precision', 'recall', 'j_statistic')

    Returns
    -------
    np.ndarray
        Optimal thresholds per class, shape [C]
    """
    num_classes = y_true.shape[1]
    optimal_thresholds = np.zeros(num_classes)

    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) < 2:
            optimal_thresholds[i] = 0.5
            continue

        if metric == "j_statistic":
            # Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_prob[:, i])
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            optimal_thresholds[i] = thresholds[best_idx]
        else:
            # Optimize precision/recall/F1
            precision, recall, thresholds = precision_recall_curve(
                y_true[:, i], y_prob[:, i]
            )

            if metric == "f1":
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
            elif metric == "precision":
                best_idx = np.argmax(precision[:-1])
            elif metric == "recall":
                best_idx = np.argmax(recall[:-1])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            optimal_thresholds[i] = thresholds[best_idx]

    return optimal_thresholds


def plot_per_class_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Class Confusion Matrices",
) -> None:
    """
    Plot confusion matrices for each class in a grid layout.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_pred : np.ndarray
        Predicted binary labels, shape [N, C]
    class_names : List[str]
        List of class names
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    """
    cm_array = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)

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

        cm = cm_array[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

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
