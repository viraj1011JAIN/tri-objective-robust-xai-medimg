"""
Metrics Utilities
=================

Common metrics calculation for model evaluation.

Author: Viraj Pankaj Jain
Date: November 24, 2025
"""

from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Union[List, np.ndarray, None] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUROC)

    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Basic accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred) * 100)

    # Balanced accuracy (important for imbalanced datasets)
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred) * 100)

    # Precision, Recall, F1 (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision"] = float(precision * 100)
    metrics["recall"] = float(recall * 100)
    metrics["f1_macro"] = float(f1 * 100)

    # Cohen's Kappa (agreement measure)
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # AUROC (if probabilities provided)
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
            if n_classes == 2:
                # Binary classification
                metrics["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]) * 100)
            else:
                # Multi-class (one-vs-rest)
                metrics["auroc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                    * 100
                )
        except Exception:
            metrics["auroc"] = 0.0
    else:
        metrics["auroc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def calculate_robust_metrics(
    clean_acc: float,
    robust_acc: float,
) -> Dict[str, float]:
    """
    Calculate robustness-specific metrics.

    Args:
        clean_acc: Clean accuracy (%)
        robust_acc: Robust accuracy (%)

    Returns:
        Dictionary with robustness metrics
    """
    return {
        "clean_acc": float(clean_acc),
        "robust_acc": float(robust_acc),
        "robustness_gap": float(clean_acc - robust_acc),
        "robustness_ratio": float(robust_acc / clean_acc) if clean_acc > 0 else 0.0,
    }
