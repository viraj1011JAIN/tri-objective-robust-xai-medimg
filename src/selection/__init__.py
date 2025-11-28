"""
Selective Prediction Module for Phase 8.

This package implements world-class production-grade selective prediction by
combining confidence and stability gating with advanced optimizations:
- Cascading gate optimization (2-4× speedup)
- Parallel batch processing (ThreadPoolExecutor)
- Streaming API for memory efficiency
- Numerical stability guards
- Comprehensive performance metrics
- Type-safe Pydantic configuration

Modules
-------
- selective_predictor: Main gating logic with world-class optimizations
- selective_metrics: Comprehensive evaluation metrics (Phase 8.5)
- (confidence_scorer and stability_scorer are in src/validation/)

Key Components
--------------
1. **SelectivePredictor**: Combined gating with cascading optimization
2. **SelectionResult**: Dataclass for prediction decisions with metadata
3. **GatingStrategy**: Enum for different gating approaches
4. **SelectivePredictorConfig**: Pydantic config with validation
5. **SelectiveMetrics**: Comprehensive metrics container (Phase 8.5)
6. **RiskCoverageCurve**: Risk-coverage curve data structure

Metrics (Phase 8.5)
-------------------
- Coverage: Fraction of samples accepted
- Selective Accuracy: Accuracy on accepted samples
- Selective Risk: Error rate on accepted samples
- Risk on Rejected: Error rate on rejected samples
- AURC/E-AURC: Area Under Risk-Coverage curve
- Improvement: Selective accuracy - Overall accuracy
- ECE Post-Selection: Calibration on accepted samples

Research Question 3 (RQ3)
-------------------------
"Can multi-signal gating (confidence + stability) enable safe selective prediction?"

Target: ≥4pp improvement in accuracy at 90% coverage vs. confidence-only baseline

Integration
-----------
This module integrates with:
- src.validation.confidence_scorer: Confidence estimation
- src.validation.stability_scorer: Explanation stability scoring
- src.evaluation.metrics: Coverage-accuracy curves, selective risk

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 8.5 - Selective Prediction Evaluation Metrics
Date: November 28, 2025
Version: 8.5.0 (Production)
"""

from .selective_metrics import (  # Data classes; Core metric functions; Risk-coverage; Calibration; Main entry point; Comparison; Visualization; Utilities
    RiskCoverageCurve,
    SelectiveMetrics,
    compare_strategies,
    compute_aurc,
    compute_coverage,
    compute_ece_post_selection,
    compute_improvement,
    compute_metrics_at_coverage,
    compute_rejection_precision_recall,
    compute_rejection_quality,
    compute_risk_coverage_curve,
    compute_risk_on_rejected,
    compute_selective_accuracy,
    compute_selective_metrics,
    compute_selective_risk,
    find_threshold_for_coverage,
    plot_accuracy_coverage_curve,
    plot_risk_coverage_curve,
    plot_strategy_comparison,
    validate_hypothesis_h3a,
)
from .selective_predictor import (
    GatingStrategy,
    SelectionResult,
    SelectivePredictor,
    SelectivePredictorConfig,
)
from .selective_predictor import (
    compute_selective_metrics as compute_selective_metrics_legacy,
)

__all__ = [
    # Core classes (Phase 8.3)
    "SelectivePredictor",
    "SelectionResult",
    "GatingStrategy",
    "SelectivePredictorConfig",
    # Metrics data classes (Phase 8.5)
    "SelectiveMetrics",
    "RiskCoverageCurve",
    # Core metric functions (Phase 8.5)
    "compute_coverage",
    "compute_selective_accuracy",
    "compute_selective_risk",
    "compute_risk_on_rejected",
    "compute_improvement",
    "compute_rejection_quality",
    "compute_rejection_precision_recall",
    # Risk-coverage (Phase 8.5)
    "compute_risk_coverage_curve",
    "compute_aurc",
    # Calibration (Phase 8.5)
    "compute_ece_post_selection",
    # Main entry point (Phase 8.5)
    "compute_selective_metrics",
    "compute_selective_metrics_legacy",
    # Comparison (Phase 8.5)
    "compare_strategies",
    # Visualization (Phase 8.5)
    "plot_risk_coverage_curve",
    "plot_accuracy_coverage_curve",
    "plot_strategy_comparison",
    # Utilities (Phase 8.5)
    "find_threshold_for_coverage",
    "compute_metrics_at_coverage",
    "validate_hypothesis_h3a",
]

__version__ = "8.3.0"
