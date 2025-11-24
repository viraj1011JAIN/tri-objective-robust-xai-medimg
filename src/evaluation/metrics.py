"""
Comprehensive classification and evaluation metrics for medical imaging.

This module provides production-grade evaluation metrics including:
- Classification metrics (Accuracy, AUROC, F1, MCC, Precision, Recall)
- Per-class metrics with macro/micro/weighted averaging
- Confusion matrix computation and visualization
- Bootstrap confidence intervals (95% CI)
- Support for both binary and multi-class classification

Phase 3.5: Baseline Evaluation - Dermoscopy
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities or logits, shape (N, num_classes) or (N,)
    labels : np.ndarray or torch.Tensor
        Ground truth labels, shape (N,)
    num_classes : int
        Number of classes
    class_names : list of str, optional
        Names of classes for per-class metrics

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - auroc_macro: Macro-averaged AUROC
        - auroc_weighted: Weighted AUROC
        - f1_macro: Macro-averaged F1 score
        - f1_weighted: Weighted F1 score
        - mcc: Matthews correlation coefficient
        - auroc_per_class: Per-class AUROC (if class_names provided)
    """
    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Handle shape: (N, num_classes) or (N,)
    if predictions.ndim == 1:
        # Binary classification with probability for positive class
        pred_probs = predictions
        pred_labels = (predictions > 0.5).astype(int)
    elif predictions.ndim == 2:
        # Multi-class: convert to probabilities if needed (softmax)
        if predictions.shape[1] > 1:
            # Apply softmax if not already probabilities
            if not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-3):
                pred_probs = np.exp(predictions) / np.exp(predictions).sum(
                    axis=1, keepdims=True
                )
            else:
                pred_probs = predictions
            pred_labels = pred_probs.argmax(axis=1)
        else:
            # Binary with logits/probs as (N, 1)
            pred_probs = predictions.squeeze()
            pred_labels = (pred_probs > 0.5).astype(int)
    else:
        raise ValueError(f"Invalid predictions shape: {predictions.shape}")

    metrics: Dict[str, Any] = {}

    # 1. Accuracy
    metrics["accuracy"] = float(accuracy_score(labels, pred_labels))

    # 2. AUROC (macro and weighted)
    if num_classes == 2:
        # Binary classification
        try:
            metrics["auroc_macro"] = float(roc_auc_score(labels, pred_probs))
            metrics["auroc_weighted"] = metrics["auroc_macro"]
        except ValueError:
            metrics["auroc_macro"] = float("nan")
            metrics["auroc_weighted"] = float("nan")
    else:
        # Multi-class
        try:
            metrics["auroc_macro"] = float(
                roc_auc_score(
                    labels, pred_probs, multi_class="ovr", average="macro"
                )
            )
            metrics["auroc_weighted"] = float(
                roc_auc_score(
                    labels, pred_probs, multi_class="ovr", average="weighted"
                )
            )
        except ValueError:
            metrics["auroc_macro"] = float("nan")
            metrics["auroc_weighted"] = float("nan")

    # 3. Per-class AUROC
    if class_names and num_classes > 1:
        for i, name in enumerate(class_names):
            try:
                binary_labels = (labels == i).astype(int)
                class_probs = pred_probs[:, i] if pred_probs.ndim == 2 else pred_probs
                metrics[f"auroc_{name}"] = float(
                    roc_auc_score(binary_labels, class_probs)
                )
            except ValueError:
                metrics[f"auroc_{name}"] = float("nan")

    # 4. F1 Score (macro and weighted)
    metrics["f1_macro"] = float(
        f1_score(labels, pred_labels, average="macro", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(labels, pred_labels, average="weighted", zero_division=0)
    )

    # 5. Matthews Correlation Coefficient
    try:
        metrics["mcc"] = float(matthews_corrcoef(labels, pred_labels))
    except ValueError:
        metrics["mcc"] = float("nan")

    return metrics


def compute_per_class_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1 score.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities, shape (N, num_classes) or predicted labels (N,)
    labels : np.ndarray or torch.Tensor
        Ground truth labels, shape (N,)
    class_names : list of str
        Names of classes

    Returns
    -------
    per_class_metrics : dict
        Dictionary with class names as keys, each containing:
        - precision: Precision for the class
        - recall: Recall for the class
        - f1: F1 score for the class
        - support: Number of samples in the class
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get predicted labels
    if predictions.ndim == 2:
        pred_labels = predictions.argmax(axis=1)
    else:
        pred_labels = predictions

    # Compute precision, recall, f1, support for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred_labels, average=None, zero_division=0
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return per_class_metrics


def compute_confusion_matrix(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities (N, num_classes) or labels (N,)
    labels : np.ndarray or torch.Tensor
        Ground truth labels, shape (N,)
    class_names : list of str, optional
        Names of classes for axis labels
    normalize : str, optional
        'true', 'pred', 'all' or None. Normalizes confusion matrix over
        true (rows), predicted (columns) or all the population.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix, shape (num_classes, num_classes)
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get predicted labels
    if predictions.ndim == 2:
        pred_labels = predictions.argmax(axis=1)
    else:
        pred_labels = predictions

    cm = confusion_matrix(labels, pred_labels, normalize=normalize)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list of str
        Names of classes
    title : str
        Plot title
    normalize : bool
        If True, normalize by row (true labels)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if normalize and cm.sum(axis=1, keepdims=True).max() > 0:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compute_bootstrap_ci(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for metrics.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities or labels
    labels : np.ndarray or torch.Tensor
        Ground truth labels
    num_classes : int
        Number of classes
    metric_fn : callable
        Function that computes metrics, signature: metric_fn(pred, labels, num_classes) -> dict
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    ci_metrics : dict
        Dictionary with keys: {metric_name: value, metric_name_ci: (lower, upper)}
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    rng = np.random.RandomState(random_seed)
    n_samples = len(labels)

    # Compute metrics on original data
    original_metrics = metric_fn(predictions, labels, num_classes)

    # Bootstrap resampling
    bootstrap_metrics: Dict[str, List[float]] = {
        key: [] for key in original_metrics.keys()
    }

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        pred_boot = predictions[indices]
        labels_boot = labels[indices]

        # Compute metrics on bootstrap sample
        try:
            boot_metrics = metric_fn(pred_boot, labels_boot, num_classes)
            for key, value in boot_metrics.items():
                if not np.isnan(value):
                    bootstrap_metrics[key].append(value)
        except Exception:
            continue

    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_metrics: Dict[str, Union[float, Tuple[float, float]]] = {}

    for key, values in bootstrap_metrics.items():
        if len(values) > 0:
            ci_metrics[key] = original_metrics[key]
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            ci_metrics[f"{key}_ci"] = (float(lower), float(upper))
        else:
            ci_metrics[key] = original_metrics[key]
            ci_metrics[f"{key}_ci"] = (float("nan"), float("nan"))

    return ci_metrics


def compute_roc_curve(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    class_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve for a specific class (one-vs-rest).

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities, shape (N, num_classes) or (N,)
    labels : np.ndarray or torch.Tensor
        Ground truth labels
    class_idx : int
        Class index for binary ROC (default=1 for positive class)

    Returns
    -------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    thresholds : np.ndarray
        Thresholds
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Binary labels for class_idx
    binary_labels = (labels == class_idx).astype(int)

    # Get probabilities for class_idx
    if predictions.ndim == 2:
        class_probs = predictions[:, class_idx]
    else:
        class_probs = predictions

    fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)
    return fpr, tpr, thresholds


def compute_pr_curve(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    class_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve for a specific class.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities
    labels : np.ndarray or torch.Tensor
        Ground truth labels
    class_idx : int
        Class index

    Returns
    -------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    thresholds : np.ndarray
        Thresholds
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    binary_labels = (labels == class_idx).astype(int)

    if predictions.ndim == 2:
        class_probs = predictions[:, class_idx]
    else:
        class_probs = predictions

    precision, recall, thresholds = precision_recall_curve(binary_labels, class_probs)
    return precision, recall, thresholds


def plot_roc_curves(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for all classes.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Predicted probabilities, shape (N, num_classes)
    labels : np.ndarray or torch.Tensor
        Ground truth labels
    class_names : list of str
        Names of classes
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(class_names):
        fpr, tpr, _ = compute_roc_curve(predictions, labels, class_idx=i)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
