"""
Evaluation module for medical imaging models.

Provides comprehensive evaluation metrics including:
- Classification metrics (accuracy, AUROC, F1, MCC)
- Multi-label metrics (macro/micro AUROC, Hamming loss, subset accuracy)
- Calibration metrics (ECE, MCE, Brier score)
- Multi-label calibration (per-class ECE, MCE, Brier)
- Bootstrap confidence intervals
- Cross-site evaluation
- Confusion matrices and per-class metrics
"""

from src.evaluation.calibration import (
    calculate_ece,
    calculate_mce,
    evaluate_calibration,
    plot_confidence_histogram,
    plot_reliability_diagram,
)
from src.evaluation.metrics import (
    compute_bootstrap_ci,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)
from src.evaluation.multilabel_calibration import (
    compute_multilabel_brier_score,
    compute_multilabel_calibration_metrics,
    compute_multilabel_ece,
    compute_multilabel_mce,
    plot_multilabel_confidence_histogram,
    plot_multilabel_reliability_diagram,
)
from src.evaluation.multilabel_metrics import (
    compute_bootstrap_ci_multilabel,
    compute_multilabel_auroc,
    compute_multilabel_confusion_matrix,
    compute_multilabel_metrics,
    compute_optimal_thresholds,
    plot_multilabel_auroc_per_class,
    plot_multilabel_roc_curves,
    plot_per_class_confusion_matrices,
)

__all__ = [
    # Multi-class metrics
    "compute_classification_metrics",
    "compute_per_class_metrics",
    "compute_confusion_matrix",
    "compute_bootstrap_ci",
    # Multi-class calibration
    "calculate_ece",
    "calculate_mce",
    "evaluate_calibration",
    "plot_reliability_diagram",
    "plot_confidence_histogram",
    # Multi-label metrics
    "compute_multilabel_auroc",
    "compute_multilabel_metrics",
    "compute_multilabel_confusion_matrix",
    "compute_bootstrap_ci_multilabel",
    "compute_optimal_thresholds",
    "plot_multilabel_auroc_per_class",
    "plot_multilabel_roc_curves",
    "plot_per_class_confusion_matrices",
    # Multi-label calibration
    "compute_multilabel_ece",
    "compute_multilabel_mce",
    "compute_multilabel_brier_score",
    "compute_multilabel_calibration_metrics",
    "plot_multilabel_reliability_diagram",
    "plot_multilabel_confidence_histogram",
]
