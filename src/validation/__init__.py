"""
Validation Module for Tri-Objective Robust XAI.

This module provides comprehensive validation capabilities for the tri-objective
adversarial training framework, including confidence scoring for selective prediction.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 8.1 - Confidence Scoring for Selective Prediction
"""

# Import confidence scoring (Phase 8.1)
from src.validation.confidence_scorer import (
    ConfidenceMethod,
    ConfidenceScore,
    ConfidenceScorer,
    EntropyScorer,
    MCDropoutScorer,
    SoftmaxMaxScorer,
    TemperatureScaledScorer,
    compute_confidence_metrics,
)

# Import stability scoring (Phase 8.2)
from src.validation.stability_scorer import (
    SSIMStabilityScorer,
    StabilityMethod,
    StabilityScore,
    StabilityScorer,
)

# Import threshold tuning (Phase 8.4)
from src.validation.threshold_tuner import (
    ThresholdConfig,
    ThresholdTuner,
    TuningObjective,
    TuningResult,
    compare_strategies,
    tune_thresholds_for_dataset,
)

# Conditional imports for Phase 7.7 components (may not exist yet)
try:
    from src.validation.tri_objective_validator import (
        BASELINE_VALUES,
        DEFAULT_TARGETS,
        ConvergenceAnalysis,
        ConvergenceAnalyzer,
        MultiSeedAggregator,
        ObjectiveType,
        TriObjectiveValidator,
        ValidationMetrics,
        ValidationResult,
        ValidationStatus,
        create_validator,
    )

    _HAS_VALIDATOR = True
except ImportError:
    _HAS_VALIDATOR = False

try:
    from src.validation.training_curves import (
        METRIC_COLORS,
        OBJECTIVE_COLORS,
        PUBLICATION_STYLE,
        SEED_COLORS,
        TrainingCurvePlotter,
        TrainingHistory,
        create_plotter,
    )

    _HAS_CURVES = True
except ImportError:
    _HAS_CURVES = False

# Build __all__ dynamically based on available modules
__all__ = [
    # Confidence scoring (Phase 8.1) - always available
    "ConfidenceMethod",
    "ConfidenceScore",
    "SoftmaxMaxScorer",
    "EntropyScorer",
    "MCDropoutScorer",
    "TemperatureScaledScorer",
    "ConfidenceScorer",
    "compute_confidence_metrics",
]

# Add validator components if available (Phase 7.7)
if _HAS_VALIDATOR:
    __all__.extend(
        [
            "TriObjectiveValidator",
            "ValidationMetrics",
            "ValidationResult",
            "ValidationStatus",
            "ObjectiveType",
            "ConvergenceAnalyzer",
            "ConvergenceAnalysis",
            "MultiSeedAggregator",
            "create_validator",
            "DEFAULT_TARGETS",
            "BASELINE_VALUES",
        ]
    )

# Add training curves if available (Phase 7.7)
if _HAS_CURVES:
    __all__.extend(
        [
            "TrainingCurvePlotter",
            "TrainingHistory",
            "create_plotter",
            "PUBLICATION_STYLE",
            "OBJECTIVE_COLORS",
            "METRIC_COLORS",
            "SEED_COLORS",
        ]
    )

__version__ = "1.0.0"
__author__ = "Viraj Pankaj Jain"
__phase__ = "7.7"
