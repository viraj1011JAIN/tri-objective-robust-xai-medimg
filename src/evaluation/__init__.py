"""
Evaluation module for medical imaging models.

Provides comprehensive evaluation metrics including:
- Classification metrics (accuracy, AUROC, F1, MCC)
- Multi-label metrics (macro/micro AUROC, Hamming loss, subset accuracy)
- Calibration metrics (ECE, MCE, Brier score)
- Multi-label calibration (per-class ECE, MCE, Brier)
- Statistical tests (paired t-test, Cohen's d, McNemar's, bootstrap CI)
- Pareto analysis (frontier, dominated solutions, knee points, hypervolume)
- Bootstrap confidence intervals
- Cross-site evaluation
- Confusion matrices and per-class metrics
- RQ1 evaluation pipeline (Phase 9.2)

Phase 9.1: Comprehensive Evaluation Infrastructure
Phase 9.2: RQ1 Evaluation Pipeline
Author: Viraj Jain
MSc Dissertation - University of Glasgow
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
    compute_pr_curve,
    compute_roc_curve,
    plot_confusion_matrix,
    plot_roc_curves,
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
from src.evaluation.pareto_analysis import (
    ParetoFrontier,
    ParetoSolution,
    analyze_tradeoffs,
    compute_hypervolume,
    compute_hypervolume_2d,
    compute_pareto_frontier,
    find_knee_point_angle,
    find_knee_point_curvature,
    find_knee_point_distance,
    find_knee_points,
    get_dominated_solutions,
    is_dominated,
    load_frontier,
    non_dominated_sort,
    plot_parallel_coordinates,
    plot_pareto_2d,
    plot_pareto_3d,
    save_frontier,
    select_best_solution,
)
from src.evaluation.rq1_evaluator import (
    CalibrationResults,
    CrossSiteResults,
    EvaluationConfig,
    HypothesisTestResults,
    ModelCheckpoint,
    RobustnessResults,
    RQ1Evaluator,
    TaskPerformanceResults,
    create_rq1_evaluator,
)
from src.evaluation.rq1_report_generator import (
    RQ1ReportGenerator,
    create_rq1_report_generator,
)
from src.evaluation.statistical_tests import (
    BootstrapResult,
    StatisticalTestResult,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_confidence_interval,
    bootstrap_metric_comparison,
    bootstrap_paired_difference,
    comprehensive_model_comparison,
    compute_cohens_d,
    compute_glass_delta,
    compute_hedges_g,
    generate_comparison_report,
    independent_t_test,
    interpret_effect_size,
    mann_whitney_u_test,
    mcnemars_test,
    paired_t_test,
    wilcoxon_signed_rank_test,
)

__all__ = [
    # Multi-class metrics
    "compute_classification_metrics",
    "compute_per_class_metrics",
    "compute_confusion_matrix",
    "compute_bootstrap_ci",
    "compute_roc_curve",
    "compute_pr_curve",
    "plot_confusion_matrix",
    "plot_roc_curves",
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
    # Statistical tests
    "StatisticalTestResult",
    "BootstrapResult",
    "compute_cohens_d",
    "compute_glass_delta",
    "compute_hedges_g",
    "interpret_effect_size",
    "paired_t_test",
    "independent_t_test",
    "mcnemars_test",
    "wilcoxon_signed_rank_test",
    "mann_whitney_u_test",
    "bootstrap_confidence_interval",
    "bootstrap_paired_difference",
    "bootstrap_metric_comparison",
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    "comprehensive_model_comparison",
    "generate_comparison_report",
    # Pareto analysis
    "ParetoSolution",
    "ParetoFrontier",
    "is_dominated",
    "compute_pareto_frontier",
    "get_dominated_solutions",
    "non_dominated_sort",
    "find_knee_point_angle",
    "find_knee_point_distance",
    "find_knee_point_curvature",
    "find_knee_points",
    "compute_hypervolume_2d",
    "compute_hypervolume",
    "plot_pareto_2d",
    "plot_pareto_3d",
    "plot_parallel_coordinates",
    "analyze_tradeoffs",
    "select_best_solution",
    "save_frontier",
    "load_frontier",
    # RQ1 evaluation (Phase 9.2)
    "ModelCheckpoint",
    "EvaluationConfig",
    "TaskPerformanceResults",
    "RobustnessResults",
    "CrossSiteResults",
    "CalibrationResults",
    "HypothesisTestResults",
    "RQ1Evaluator",
    "create_rq1_evaluator",
    "RQ1ReportGenerator",
    "create_rq1_report_generator",
]
