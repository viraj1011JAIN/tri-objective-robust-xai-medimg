"""
Production-Grade Selective Predictor for Phase 8.3.

This module implements combined gating (confidence AND stability) for safe
selective prediction in medical imaging. The predictor can abstain from
uncertain predictions, enabling clinical deployment with human-in-the-loop fallback.

Mathematical Foundation
-----------------------
Combined Gating Decision:
    Accept if: (confidence > τ_conf) AND (stability > τ_stab)
    Reject otherwise

Where:
    - confidence: P(ŷ|x) from softmax or MC Dropout variance
    - stability: SSIM(H_clean, H_perturbed) ∈ [0, 1]
    - τ_conf: Confidence threshold (tuned on validation set)
    - τ_stab: Stability threshold (typically 0.75)

Key Metrics:
    - Coverage: Fraction of samples accepted
    - Selective Accuracy: Accuracy on accepted samples
    - Risk on Rejected: Error rate on rejected samples
    - AURC: Area Under Risk-Coverage curve

Clinical Motivation
--------------------
In medical imaging, false positives and false negatives have different costs:
    - False Negative (miss cancer): High clinical risk
    - False Positive (false alarm): Lower risk, but causes patient anxiety
    - Abstention: Safe fallback to human expert

Selective prediction enables the model to:
1. Make predictions when confident AND stable
2. Defer to human expert when uncertain OR unstable
3. Achieve higher accuracy on accepted cases

Research Integration
--------------------
This module implements the final component for RQ3:
    "Can multi-signal gating enable safe selective prediction?"

Target Performance:
    - Combined gating (confidence + stability): +4.2pp @ 90% coverage
    - Confidence-only baseline: +0.8pp @ 90% coverage
    - Improvement: 5.25× better selective gain

Key Classes
-----------
1. **GatingStrategy**: Enum for gating approaches
    - CONFIDENCE_ONLY: Traditional selective prediction
    - STABILITY_ONLY: Novel XAI-based selection
    - COMBINED: Full approach (confidence AND stability)

2. **SelectionResult**: Dataclass containing:
    - prediction: Model's predicted class
    - confidence: Confidence score [0, 1]
    - stability: Stability score [0, 1]
    - is_accepted: Boolean decision
    - rejection_reason: String explanation if rejected

3. **SelectivePredictor**: Main predictor with:
    - Batch-efficient processing
    - Configurable thresholds
    - Multiple gating strategies
    - Comprehensive metadata tracking

Usage Example
-------------
```python
from src.selection import SelectivePredictor, GatingStrategy
from src.validation import ConfidenceScorer, StabilityScorer

# Initialize scorers
confidence_scorer = ConfidenceScorer(model, method="mc_dropout")
stability_scorer = StabilityScorer(model, xai_method="gradcam")

# Initialize predictor
predictor = SelectivePredictor(
    confidence_scorer=confidence_scorer,
    stability_scorer=stability_scorer,
    confidence_threshold=0.85,
    stability_threshold=0.75,
    strategy=GatingStrategy.COMBINED
)

# Make selective predictions
results = predictor.predict_batch(images, return_all_metadata=True)

# Analyze results
accepted = [r for r in results if r.is_accepted]
rejected = [r for r in results if not r.is_accepted]

print(f"Coverage: {len(accepted) / len(results):.2%}")
print(f"Selective Accuracy: {sum(r.prediction == r.true_label for r in accepted) / len(accepted):.2%}")
```

Production Features
-------------------
1. **Batch Efficiency**: Vectorized operations for fast inference
2. **Flexible Configuration**: YAML-based threshold tuning
3. **Comprehensive Logging**: Detailed decision metadata
4. **Error Handling**: Graceful fallbacks for edge cases
5. **Memory Management**: GPU memory-efficient processing
6. **Reproducibility**: Deterministic behavior with fixed seeds

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 8.3 - Selective Predictor
Date: November 27, 2025
Version: 8.3.0 (Production)
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm

try:
    from pydantic import BaseModel, Field, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    warnings.warn("Pydantic not available. Configuration validation disabled.")

# Import confidence and stability scorers from validation module
from ..validation.confidence_scorer import ConfidenceMethod, ConfidenceScorer
from ..validation.stability_scorer import StabilityMethod, StabilityScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATACLASSES
# ============================================================================


if PYDANTIC_AVAILABLE:

    class SelectivePredictorConfig(BaseModel):
        """
        Type-safe configuration with validation.

        Attributes
        ----------
        confidence_threshold : float
            Minimum confidence for acceptance [0, 1].

        stability_threshold : float
            Minimum stability for acceptance [0, 1].

        strategy : str
            Gating strategy: 'confidence_only', 'stability_only', or 'combined'.

        device : str
            Computing device: 'cpu', 'cuda', or 'mps'.

        batch_size : int
            Batch size for processing (must be > 0).

        verbose : bool
            Enable progress bars and logging.

        num_workers : int
            Number of parallel workers for stability computation.

        enable_cascading : bool
            Enable fast accept/reject paths for extreme confidence values.

        fast_accept_threshold : float
            Confidence threshold for immediate acceptance (no stability check).

        fast_reject_threshold : float
            Confidence threshold for immediate rejection (no stability check).
        """

        confidence_threshold: float = Field(
            default=0.85,
            ge=0.0,
            le=1.0,
            description="Minimum confidence for acceptance",
        )
        stability_threshold: float = Field(
            default=0.75, ge=0.0, le=1.0, description="Minimum stability for acceptance"
        )
        strategy: str = Field(default="combined", description="Gating strategy")
        device: str = Field(default="cuda", description="Computing device")
        batch_size: int = Field(
            default=32, gt=0, description="Batch size for processing"
        )
        verbose: bool = Field(default=True, description="Enable logging")
        num_workers: int = Field(
            default=4, ge=1, description="Parallel workers for stability computation"
        )
        enable_cascading: bool = Field(
            default=True, description="Enable cascading gate optimization"
        )
        fast_accept_threshold: float = Field(
            default=0.98, ge=0.0, le=1.0, description="Threshold for fast acceptance"
        )
        fast_reject_threshold: float = Field(
            default=0.50, ge=0.0, le=1.0, description="Threshold for fast rejection"
        )

        @validator("device")
        def validate_device(cls, v):
            """Validate device availability."""
            if v not in ["cpu", "cuda", "mps"]:
                raise ValueError(
                    f"Invalid device: {v}. Must be 'cpu', 'cuda', or 'mps'."
                )
            if v == "cuda" and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return v

        @validator("strategy")
        def validate_strategy(cls, v):
            """Validate gating strategy."""
            valid_strategies = ["confidence_only", "stability_only", "combined"]
            if v not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy: {v}. Must be one of {valid_strategies}."
                )
            return v

        @validator("fast_accept_threshold")
        def validate_fast_accept(cls, v, values):
            """Ensure fast_accept > confidence_threshold."""
            if "confidence_threshold" in values and v <= values["confidence_threshold"]:
                raise ValueError(
                    f"fast_accept_threshold ({v}) must be > confidence_threshold ({values['confidence_threshold']})"
                )
            return v

        @validator("fast_reject_threshold")
        def validate_fast_reject(cls, v, values):
            """Ensure fast_reject < confidence_threshold."""
            if "confidence_threshold" in values and v >= values["confidence_threshold"]:
                raise ValueError(
                    f"fast_reject_threshold ({v}) must be < confidence_threshold ({values['confidence_threshold']})"
                )
            return v

        class Config:
            use_enum_values = True

else:
    # Fallback dict-based config if pydantic not available
    SelectivePredictorConfig = None


class GatingStrategy(Enum):
    """
    Gating strategies for selective prediction.

    Strategies
    ----------
    CONFIDENCE_ONLY : Only use confidence for gating
        Traditional selective prediction approach. Accept if confidence > τ_conf.

    STABILITY_ONLY : Only use stability for gating
        Novel XAI-based approach. Accept if stability > τ_stab.

    COMBINED : Use both confidence AND stability (recommended)
        Full approach for RQ3. Accept if (confidence > τ_conf) AND (stability > τ_stab).
        Provides best selective accuracy with safety guarantees.
    """

    CONFIDENCE_ONLY = "confidence_only"
    STABILITY_ONLY = "stability_only"
    COMBINED = "combined"


@dataclass
class SelectionResult:
    """
    Result of a selective prediction decision.

    This dataclass contains all information about a single prediction,
    including the decision, scores, and reasoning for clinical auditing.

    Attributes
    ----------
    prediction : int
        Predicted class index (argmax of logits).

    confidence : float
        Confidence score [0, 1]. Higher = more confident.

    stability : float
        Stability score [0, 1]. Higher = more stable explanation.

    is_accepted : bool
        Whether the prediction is accepted (True) or rejected (False).

    rejection_reason : Optional[str]
        Human-readable explanation if rejected. One of:
        - "low_confidence": confidence < τ_conf
        - "low_stability": stability < τ_stab
        - "both_low": Both confidence and stability below thresholds
        - "invalid_scores": NaN or Inf detected
        - None: Prediction accepted

    decision_strategy : Optional[str]
        Strategy used for decision. One of:
        - "FAST_ACCEPT": Ultra-confident, skipped stability check
        - "FAST_REJECT": Very uncertain, skipped stability check
        - "ROBUST_ACCEPT": Passed both confidence and stability
        - "ROBUST_REJECT": Failed one or both checks

    logits : Optional[np.ndarray]
        Raw model logits for all classes. Shape: (num_classes,)

    probabilities : Optional[np.ndarray]
        Softmax probabilities for all classes. Shape: (num_classes,)

    true_label : Optional[int]
        Ground truth label (if available). Used for evaluation.

    sample_id : Optional[str]
        Unique identifier for the sample (e.g., image filename).

    metadata : Dict[str, Any]
        Additional metadata (e.g., patient ID, acquisition details).
    """

    prediction: int
    confidence: float
    stability: float
    is_accepted: bool
    rejection_reason: Optional[str] = None
    decision_strategy: Optional[str] = None
    logits: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    true_label: Optional[int] = None
    sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Concise string representation for debugging."""
        status = "ACCEPTED" if self.is_accepted else "REJECTED"
        reason = f" ({self.rejection_reason})" if self.rejection_reason else ""
        strategy = f" [{self.decision_strategy}]" if self.decision_strategy else ""
        return (
            f"SelectionResult(pred={self.prediction}, "
            f"conf={self.confidence:.3f}, stab={self.stability:.3f}, "
            f"{status}{reason}{strategy})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON/YAML.
        """
        return {
            "prediction": int(self.prediction),
            "confidence": float(self.confidence),
            "stability": float(self.stability),
            "is_accepted": bool(self.is_accepted),
            "rejection_reason": self.rejection_reason,
            "decision_strategy": self.decision_strategy,
            "logits": self.logits.tolist() if self.logits is not None else None,
            "probabilities": (
                self.probabilities.tolist() if self.probabilities is not None else None
            ),
            "true_label": int(self.true_label) if self.true_label is not None else None,
            "sample_id": self.sample_id,
            "metadata": self.metadata,
        }


# ============================================================================
# MAIN SELECTIVE PREDICTOR CLASS
# ============================================================================


class SelectivePredictor:
    """
    Production-grade selective predictor with combined gating.

    This class implements the core logic for Phase 8.3: combined confidence
    and stability gating for safe selective prediction in medical imaging.

    Parameters
    ----------
    confidence_scorer : ConfidenceScorer
        Scorer for confidence estimation (from Phase 8.1).

    stability_scorer : StabilityScorer
        Scorer for explanation stability (from Phase 8.2).

    confidence_threshold : float, optional (default=0.85)
        Minimum confidence for acceptance. Tuned on validation set.

    stability_threshold : float, optional (default=0.75)
        Minimum stability for acceptance. Based on RQ2 target (75% stable).

    strategy : GatingStrategy, optional (default=COMBINED)
        Gating strategy to use. COMBINED is recommended for production.

    device : str or torch.device, optional (default="cuda" if available)
        Device for computation.

    batch_size : int, optional (default=32)
        Batch size for efficient processing.

    verbose : bool, optional (default=True)
        Whether to show progress bars and logging.

    Attributes
    ----------
    total_predictions : int
        Total number of predictions made.

    total_accepted : int
        Number of accepted predictions.

    total_rejected : int
        Number of rejected predictions.

    coverage : float
        Fraction of samples accepted (total_accepted / total_predictions).

    Examples
    --------
    Basic usage with default settings:

    >>> from src.selection import SelectivePredictor, GatingStrategy
    >>> predictor = SelectivePredictor(
    ...     confidence_scorer=confidence_scorer,
    ...     stability_scorer=stability_scorer,
    ...     confidence_threshold=0.85,
    ...     stability_threshold=0.75
    ... )
    >>> results = predictor.predict_batch(images, labels)
    >>> print(f"Coverage: {predictor.coverage:.2%}")

    Advanced usage with different strategies:

    >>> # Confidence-only (baseline)
    >>> predictor_conf = SelectivePredictor(
    ...     confidence_scorer=confidence_scorer,
    ...     stability_scorer=stability_scorer,
    ...     strategy=GatingStrategy.CONFIDENCE_ONLY
    ... )
    >>>
    >>> # Stability-only (ablation)
    >>> predictor_stab = SelectivePredictor(
    ...     confidence_scorer=confidence_scorer,
    ...     stability_scorer=stability_scorer,
    ...     strategy=GatingStrategy.STABILITY_ONLY
    ... )
    >>>
    >>> # Compare strategies
    >>> results_conf = predictor_conf.predict_batch(images, labels)
    >>> results_stab = predictor_stab.predict_batch(images, labels)
    >>> results_combined = predictor.predict_batch(images, labels)
    """

    def __init__(
        self,
        confidence_scorer: ConfidenceScorer,
        stability_scorer: StabilityScorer,
        confidence_threshold: float = 0.85,
        stability_threshold: float = 0.75,
        strategy: GatingStrategy = GatingStrategy.COMBINED,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32,
        verbose: bool = True,
        num_workers: int = 4,
        enable_cascading: bool = True,
        fast_accept_threshold: float = 0.98,
        fast_reject_threshold: float = 0.50,
    ):
        """Initialize selective predictor with scorers and thresholds."""
        # Store scorers
        self.confidence_scorer = confidence_scorer
        self.stability_scorer = stability_scorer

        # Validate thresholds
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        if not (0.0 <= stability_threshold <= 1.0):
            raise ValueError(
                f"stability_threshold must be in [0, 1], got {stability_threshold}"
            )

        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold

        # Validate strategy
        if not isinstance(strategy, GatingStrategy):
            try:
                strategy = GatingStrategy(strategy)
            except ValueError:
                valid_strategies = [s.value for s in GatingStrategy]
                raise ValueError(
                    f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
                )
        self.strategy = strategy

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Configuration
        self.batch_size = batch_size
        self.verbose = verbose
        self.num_workers = num_workers
        self.enable_cascading = enable_cascading
        self.fast_accept_threshold = fast_accept_threshold
        self.fast_reject_threshold = fast_reject_threshold

        # Validate cascading thresholds
        if enable_cascading:
            if not (
                fast_reject_threshold < confidence_threshold < fast_accept_threshold
            ):
                raise ValueError(
                    f"Cascading thresholds must satisfy: "
                    f"fast_reject ({fast_reject_threshold}) < conf ({confidence_threshold}) < "
                    f"fast_accept ({fast_accept_threshold})"
                )

        # Statistics tracking
        self.total_predictions = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.fast_accepts = 0
        self.fast_rejects = 0
        self.robust_accepts = 0
        self.robust_rejects = 0

        # Performance metrics
        self.metrics = {
            "total_inference_time": 0.0,
            "confidence_computation_time": 0.0,
            "stability_computation_time": 0.0,
            "gating_computation_time": 0.0,
            "avg_confidence_accepted": [],
            "avg_confidence_rejected": [],
            "avg_stability_accepted": [],
            "avg_stability_rejected": [],
            "rejection_reasons": defaultdict(int),
            "decision_strategies": defaultdict(int),
        }

        # Logging
        logger.info(f"Initialized SelectivePredictor with:")
        logger.info(f"  Strategy: {self.strategy.value}")
        logger.info(f"  Confidence threshold: {self.confidence_threshold:.3f}")
        logger.info(f"  Stability threshold: {self.stability_threshold:.3f}")
        logger.info(f"  Device: {self.device}")
        if enable_cascading:
            logger.info(
                f"  Cascading enabled: fast_accept={fast_accept_threshold:.3f}, fast_reject={fast_reject_threshold:.3f}"
            )
        if num_workers > 1:
            logger.info(f"  Parallel processing: {num_workers} workers")

    @classmethod
    def from_config(
        cls,
        config: Union[dict, "SelectivePredictorConfig"],
        confidence_scorer: ConfidenceScorer,
        stability_scorer: StabilityScorer,
    ) -> "SelectivePredictor":
        """
        Create predictor from configuration.

        Parameters
        ----------
        config : dict or SelectivePredictorConfig
            Configuration dictionary or Pydantic config object.

        confidence_scorer : ConfidenceScorer
            Scorer for confidence estimation.

        stability_scorer : StabilityScorer
            Scorer for explanation stability.

        Returns
        -------
        SelectivePredictor
            Configured predictor instance.

        Examples
        --------
        >>> from src.selection import SelectivePredictorConfig
        >>> config = SelectivePredictorConfig(
        ...     confidence_threshold=0.9,
        ...     stability_threshold=0.8,
        ...     enable_cascading=True
        ... )
        >>> predictor = SelectivePredictor.from_config(
        ...     config, confidence_scorer, stability_scorer
        ... )
        """
        if PYDANTIC_AVAILABLE and isinstance(config, SelectivePredictorConfig):
            config_dict = config.dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(
                f"config must be dict or SelectivePredictorConfig, got {type(config)}"
            )

        # Convert strategy string to enum
        strategy_str = config_dict.pop("strategy", "combined")
        strategy = GatingStrategy(strategy_str)

        return cls(
            confidence_scorer=confidence_scorer,
            stability_scorer=stability_scorer,
            strategy=strategy,
            **config_dict,
        )

    @contextmanager
    def _timer(self, operation: str):
        """
        Context manager for timing operations.

        Parameters
        ----------
        operation : str
            Name of the operation being timed.

        Yields
        ------
        None

        Examples
        --------
        >>> with self._timer("batch_prediction"):
        ...     results = process_batch(images)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            metric_key = f"{operation}_time"
            if metric_key in self.metrics:
                self.metrics[metric_key] += elapsed
            logger.debug(f"{operation} took {elapsed:.4f}s")

    @property
    def coverage(self) -> float:
        """
        Compute current coverage (fraction of accepted predictions).

        Returns
        -------
        float
            Coverage rate [0, 1]. Returns 0.0 if no predictions made yet.
        """
        if self.total_predictions == 0:
            return 0.0
        return self.total_accepted / self.total_predictions

    def _apply_gating_logic(
        self, confidence: float, stability: float, skip_stability: bool = False
    ) -> Tuple[bool, Optional[str], str]:
        """
        Apply gating logic based on selected strategy with numerical stability.

        Parameters
        ----------
        confidence : float
            Confidence score [0, 1].

        stability : float
            Stability score [0, 1].

        skip_stability : bool, optional (default=False)
            Whether to skip stability check (for cascading gates).

        Returns
        -------
        is_accepted : bool
            Whether the prediction is accepted.

        rejection_reason : Optional[str]
            Reason for rejection if not accepted, None otherwise.

        decision_strategy : str
            Strategy used for decision (FAST_ACCEPT, FAST_REJECT, ROBUST_ACCEPT, ROBUST_REJECT).
        """
        # Handle NaN/Inf (numerical stability)
        if not (np.isfinite(confidence) and np.isfinite(stability)):
            logger.warning(
                f"Non-finite values detected: conf={confidence}, stab={stability}"
            )
            self.metrics["rejection_reasons"]["invalid_scores"] += 1
            return False, "invalid_scores", "ROBUST_REJECT"

        # Clip to valid range (defensive programming)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        stability = float(np.clip(stability, 0.0, 1.0))

        # Cascading Gate: Fast paths for extreme confidence values
        if self.enable_cascading and not skip_stability:
            if confidence >= self.fast_accept_threshold:
                self.metrics["decision_strategies"]["FAST_ACCEPT"] += 1
                return True, None, "FAST_ACCEPT"
            elif confidence <= self.fast_reject_threshold:
                self.metrics["decision_strategies"]["FAST_REJECT"] += 1
                self.metrics["rejection_reasons"]["low_confidence"] += 1
                return False, "low_confidence", "FAST_REJECT"

        # Standard gating logic
        if self.strategy == GatingStrategy.CONFIDENCE_ONLY:
            # Accept if confidence > threshold
            if confidence >= self.confidence_threshold:
                self.metrics["decision_strategies"]["ROBUST_ACCEPT"] += 1
                return True, None, "ROBUST_ACCEPT"
            else:
                self.metrics["decision_strategies"]["ROBUST_REJECT"] += 1
                self.metrics["rejection_reasons"]["low_confidence"] += 1
                return False, "low_confidence", "ROBUST_REJECT"

        elif self.strategy == GatingStrategy.STABILITY_ONLY:
            # Accept if stability > threshold
            if stability >= self.stability_threshold:
                self.metrics["decision_strategies"]["ROBUST_ACCEPT"] += 1
                return True, None, "ROBUST_ACCEPT"
            else:
                self.metrics["decision_strategies"]["ROBUST_REJECT"] += 1
                self.metrics["rejection_reasons"]["low_stability"] += 1
                return False, "low_stability", "ROBUST_REJECT"

        elif self.strategy == GatingStrategy.COMBINED:
            # Accept if BOTH confidence AND stability > thresholds
            conf_ok = confidence >= self.confidence_threshold
            stab_ok = stability >= self.stability_threshold

            if conf_ok and stab_ok:
                self.metrics["decision_strategies"]["ROBUST_ACCEPT"] += 1
                return True, None, "ROBUST_ACCEPT"
            elif not conf_ok and not stab_ok:
                self.metrics["decision_strategies"]["ROBUST_REJECT"] += 1
                self.metrics["rejection_reasons"]["both_low"] += 1
                return False, "both_low", "ROBUST_REJECT"
            elif not conf_ok:
                self.metrics["decision_strategies"]["ROBUST_REJECT"] += 1
                self.metrics["rejection_reasons"]["low_confidence"] += 1
                return False, "low_confidence", "ROBUST_REJECT"
            else:  # not stab_ok
                self.metrics["decision_strategies"]["ROBUST_REJECT"] += 1
                self.metrics["rejection_reasons"]["low_stability"] += 1
                return False, "low_stability", "ROBUST_REJECT"

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict_single(
        self,
        image: Tensor,
        true_label: Optional[int] = None,
        sample_id: Optional[str] = None,
        return_all_metadata: bool = False,
    ) -> SelectionResult:
        """
        Make selective prediction on a single image.

        Parameters
        ----------
        image : Tensor
            Input image tensor, shape (C, H, W) or (1, C, H, W).

        true_label : int, optional
            Ground truth label (if available).

        sample_id : str, optional
            Unique identifier for this sample.

        return_all_metadata : bool, optional (default=False)
            Whether to return logits and probabilities in the result.

        Returns
        -------
        SelectionResult
            Result containing prediction, scores, and decision.

        Examples
        --------
        >>> from PIL import Image
        >>> from torchvision import transforms
        >>>
        >>> # Load and preprocess image
        >>> img = Image.open("lesion.jpg")
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ... ])
        >>> x = transform(img)
        >>>
        >>> # Make selective prediction
        >>> result = predictor.predict_single(x, sample_id="lesion_001")
        >>> print(result)
        >>> if result.is_accepted:
        ...     print(f"Predicted class: {result.prediction}")
        ... else:
        ...     print(f"Rejected: {result.rejection_reason}")
        """
        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(self.device)

        with self._timer("single_prediction"):
            # Compute confidence
            with self._timer("confidence_computation"):
                conf_score = self.confidence_scorer.compute_confidence(image)
                confidence = float(conf_score.confidence)

            # Get prediction and logits
            with torch.no_grad():
                logits = self.confidence_scorer.model(image)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()

            logits_np = logits.cpu().numpy().squeeze() if return_all_metadata else None
            probs_np = (
                probabilities.cpu().numpy().squeeze() if return_all_metadata else None
            )

            # Cascading Gate: Check if we can skip expensive stability computation
            skip_stability = False
            if self.enable_cascading:
                if confidence >= self.fast_accept_threshold:
                    # Ultra-confident: skip stability check
                    stability = 1.0  # Placeholder (not used in decision)
                    skip_stability = True
                    self.fast_accepts += 1
                elif confidence <= self.fast_reject_threshold:
                    # Very uncertain: skip stability check
                    stability = 0.0  # Placeholder (not used in decision)
                    skip_stability = True
                    self.fast_rejects += 1

            # Compute stability only if needed
            if not skip_stability:
                with self._timer("stability_computation"):
                    stab_score = self.stability_scorer.compute_single_stability(
                        image=image,
                        true_label=true_label if true_label is not None else prediction,
                    )
                    stability = float(stab_score.stability)

            # Apply gating logic
            with self._timer("gating_computation"):
                is_accepted, rejection_reason, decision_strategy = (
                    self._apply_gating_logic(
                        confidence, stability, skip_stability=skip_stability
                    )
                )

            # Update statistics
            self.total_predictions += 1
            if is_accepted:
                self.total_accepted += 1
                self.metrics["avg_confidence_accepted"].append(confidence)
                self.metrics["avg_stability_accepted"].append(stability)
                if decision_strategy == "ROBUST_ACCEPT":
                    self.robust_accepts += 1
            else:
                self.total_rejected += 1
                self.metrics["avg_confidence_rejected"].append(confidence)
                self.metrics["avg_stability_rejected"].append(stability)
                if decision_strategy == "ROBUST_REJECT":
                    self.robust_rejects += 1

            # Create result
            result = SelectionResult(
                prediction=prediction,
                confidence=confidence,
                stability=stability,
                is_accepted=is_accepted,
                rejection_reason=rejection_reason,
                decision_strategy=decision_strategy,
                logits=logits_np,
                probabilities=probs_np,
                true_label=true_label,
                sample_id=sample_id,
            )

        return result

    def predict_batch(
        self,
        images: Tensor,
        true_labels: Optional[Union[List[int], Tensor]] = None,
        sample_ids: Optional[List[str]] = None,
        return_all_metadata: bool = False,
    ) -> List[SelectionResult]:
        """
        Make selective predictions on a batch of images (batch-efficient).

        Parameters
        ----------
        images : Tensor
            Batch of images, shape (B, C, H, W).

        true_labels : List[int] or Tensor, optional
            Ground truth labels (if available).

        sample_ids : List[str], optional
            Unique identifiers for each sample.

        return_all_metadata : bool, optional (default=False)
            Whether to return logits and probabilities in results.

        Returns
        -------
        List[SelectionResult]
            List of results, one per image in the batch.

        Examples
        --------
        >>> # Batch processing
        >>> results = predictor.predict_batch(images, labels=test_labels)
        >>>
        >>> # Analyze results
        >>> accepted = [r for r in results if r.is_accepted]
        >>> rejected = [r for r in results if not r.is_accepted]
        >>>
        >>> print(f"Coverage: {len(accepted) / len(results):.2%}")
        >>> print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        >>>
        >>> # Compute selective accuracy
        >>> correct = sum(r.prediction == r.true_label for r in accepted if r.true_label is not None)
        >>> selective_acc = correct / len(accepted) if len(accepted) > 0 else 0.0
        >>> print(f"Selective Accuracy: {selective_acc:.2%}")
        """
        batch_size = images.size(0)

        # Validate inputs
        if true_labels is not None:
            if isinstance(true_labels, Tensor):
                true_labels = true_labels.cpu().tolist()
            if len(true_labels) != batch_size:
                raise ValueError(
                    f"Length mismatch: images={batch_size}, labels={len(true_labels)}"
                )

        if sample_ids is not None and len(sample_ids) != batch_size:
            raise ValueError(
                f"Length mismatch: images={batch_size}, sample_ids={len(sample_ids)}"
            )

        # Move to device
        images = images.to(self.device)

        with self._timer("batch_prediction"):
            # Compute confidence scores (batch-efficient)
            with self._timer("confidence_computation"):
                with torch.no_grad():
                    logits = self.confidence_scorer.model(images)
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    confidences = torch.max(probabilities, dim=1).values

            # Identify which samples need stability computation (cascading optimization)
            needs_stability = np.ones(batch_size, dtype=bool)
            stabilities = np.zeros(batch_size)

            if self.enable_cascading:
                confidences_np = confidences.cpu().numpy()

                # Fast accept: ultra-confident samples
                fast_accept_mask = confidences_np >= self.fast_accept_threshold
                needs_stability[fast_accept_mask] = False
                stabilities[fast_accept_mask] = 1.0  # Placeholder
                self.fast_accepts += fast_accept_mask.sum()

                # Fast reject: very uncertain samples
                fast_reject_mask = confidences_np <= self.fast_reject_threshold
                needs_stability[fast_reject_mask] = False
                stabilities[fast_reject_mask] = 0.0  # Placeholder
                self.fast_rejects += fast_reject_mask.sum()

                if (
                    self.verbose
                    and (fast_accept_mask.sum() + fast_reject_mask.sum()) > 0
                ):
                    logger.info(
                        f"Cascading optimization: Skipped {needs_stability.sum()} stability computations "
                        f"({fast_accept_mask.sum()} fast accepts, {fast_reject_mask.sum()} fast rejects)"
                    )

            # Compute stability scores only for samples in the "grey zone"
            stability_indices = np.where(needs_stability)[0]

            with self._timer("stability_computation"):
                if len(stability_indices) > 0:
                    if self.num_workers > 1:
                        # Parallel processing with ThreadPoolExecutor
                        with ThreadPoolExecutor(
                            max_workers=self.num_workers
                        ) as executor:
                            futures = []
                            for idx in stability_indices:
                                image = images[idx : idx + 1]
                                true_label = (
                                    true_labels[idx]
                                    if true_labels is not None
                                    else predictions[idx].item()
                                )

                                future = executor.submit(
                                    self.stability_scorer.compute_single_stability,
                                    image=image,
                                    true_label=true_label,
                                )
                                futures.append((idx, future))

                            # Collect results with progress bar
                            for idx, future in tqdm(
                                futures,
                                desc="Computing stability scores (parallel)",
                                disable=not self.verbose,
                            ):
                                stabilities[idx] = future.result().stability
                    else:
                        # Sequential processing (fallback)
                        for idx in tqdm(
                            stability_indices,
                            desc="Computing stability scores",
                            disable=not self.verbose,
                        ):
                            image = images[idx : idx + 1]
                            true_label = (
                                true_labels[idx]
                                if true_labels is not None
                                else predictions[idx].item()
                            )

                            stab_score = self.stability_scorer.compute_single_stability(
                                image=image, true_label=true_label
                            )
                            stabilities[idx] = stab_score.stability

            # Apply gating logic and create results
            results = []
            with self._timer("gating_computation"):
                for idx in range(batch_size):
                    prediction = int(predictions[idx].item())
                    confidence = float(confidences[idx].item())
                    stability = float(stabilities[idx])

                    skip_stability = not needs_stability[idx]
                    is_accepted, rejection_reason, decision_strategy = (
                        self._apply_gating_logic(
                            confidence, stability, skip_stability=skip_stability
                        )
                    )

                    # Update statistics
                    self.total_predictions += 1
                    if is_accepted:
                        self.total_accepted += 1
                        self.metrics["avg_confidence_accepted"].append(confidence)
                        self.metrics["avg_stability_accepted"].append(stability)
                        if decision_strategy == "ROBUST_ACCEPT":
                            self.robust_accepts += 1
                    else:
                        self.total_rejected += 1
                        self.metrics["avg_confidence_rejected"].append(confidence)
                        self.metrics["avg_stability_rejected"].append(stability)
                        if decision_strategy == "ROBUST_REJECT":
                            self.robust_rejects += 1

                    # Prepare metadata
                    logits_np = (
                        logits[idx].cpu().numpy() if return_all_metadata else None
                    )
                    probs_np = (
                        probabilities[idx].cpu().numpy()
                        if return_all_metadata
                        else None
                    )
                    true_label = true_labels[idx] if true_labels is not None else None
                    sample_id = sample_ids[idx] if sample_ids is not None else None

                    result = SelectionResult(
                        prediction=prediction,
                        confidence=confidence,
                        stability=stability,
                        is_accepted=is_accepted,
                        rejection_reason=rejection_reason,
                        decision_strategy=decision_strategy,
                        logits=logits_np,
                        probabilities=probs_np,
                        true_label=true_label,
                        sample_id=sample_id,
                    )
                    results.append(result)

        return results

    def predict_batch_streaming(
        self,
        images: Tensor,
        true_labels: Optional[Union[List[int], Tensor]] = None,
        sample_ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        return_all_metadata: bool = False,
    ) -> Generator[SelectionResult, None, None]:
        """
        Memory-efficient streaming prediction for large datasets.

        Yields results one at a time instead of accumulating in memory.
        Ideal for datasets >10K images where memory is constrained.

        Parameters
        ----------
        images : Tensor
            All images to process, shape (N, C, H, W).

        true_labels : List[int] or Tensor, optional
            Ground truth labels (if available).

        sample_ids : List[str], optional
            Unique identifiers for each sample.

        batch_size : int, optional
            Processing batch size. Defaults to self.batch_size.

        return_all_metadata : bool, optional (default=False)
            Whether to return logits and probabilities.

        Yields
        ------
        SelectionResult
            Individual prediction results, one at a time.

        Examples
        --------
        >>> # Process large dataset without loading all results in memory
        >>> accepted_count = 0
        >>> for result in predictor.predict_batch_streaming(large_dataset):
        ...     if result.is_accepted:
        ...         accepted_count += 1
        ...         save_to_disk(result)  # Save on-the-fly
        >>> print(f"Accepted: {accepted_count}")
        """
        if batch_size is None:
            batch_size = self.batch_size

        num_samples = images.size(0)

        # Convert labels to list if needed
        if true_labels is not None and isinstance(true_labels, Tensor):
            true_labels = true_labels.cpu().tolist()

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_images = images[start_idx:end_idx]

            # Slice labels and sample_ids for this batch
            batch_labels = None
            if true_labels is not None:
                batch_labels = true_labels[start_idx:end_idx]

            batch_ids = None
            if sample_ids is not None:
                batch_ids = sample_ids[start_idx:end_idx]

            # Process batch
            batch_results = self.predict_batch(
                batch_images,
                true_labels=batch_labels,
                sample_ids=batch_ids,
                return_all_metadata=return_all_metadata,
            )

            # Yield results one by one
            for result in batch_results:
                yield result

    def reset_statistics(self):
        """Reset prediction statistics counters."""
        self.total_predictions = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.fast_accepts = 0
        self.fast_rejects = 0
        self.robust_accepts = 0
        self.robust_rejects = 0

        # Reset metrics
        self.metrics["total_inference_time"] = 0.0
        self.metrics["confidence_computation_time"] = 0.0
        self.metrics["stability_computation_time"] = 0.0
        self.metrics["gating_computation_time"] = 0.0
        self.metrics["avg_confidence_accepted"] = []
        self.metrics["avg_confidence_rejected"] = []
        self.metrics["avg_stability_accepted"] = []
        self.metrics["avg_stability_rejected"] = []
        self.metrics["rejection_reasons"] = defaultdict(int)
        self.metrics["decision_strategies"] = defaultdict(int)

        logger.info("Statistics reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive prediction statistics and performance metrics.

        Returns
        -------
        dict
            Dictionary containing:
            - total_predictions: Total predictions made
            - total_accepted: Number accepted
            - total_rejected: Number rejected
            - coverage: Acceptance rate [0, 1]
            - confidence_threshold: Current confidence threshold
            - stability_threshold: Current stability threshold
            - strategy: Current gating strategy
            - fast_accepts: Number of fast accepts (cascading)
            - fast_rejects: Number of fast rejects (cascading)
            - robust_accepts: Number of robust accepts
            - robust_rejects: Number of robust rejects
            - avg_confidence_accepted: Mean confidence on accepted samples
            - avg_confidence_rejected: Mean confidence on rejected samples
            - avg_stability_accepted: Mean stability on accepted samples
            - avg_stability_rejected: Mean stability on rejected samples
            - confidence_gap: Difference in mean confidence (accepted - rejected)
            - stability_gap: Difference in mean stability (accepted - rejected)
            - rejection_breakdown: Count of each rejection reason
            - decision_breakdown: Count of each decision strategy
            - avg_inference_time: Average time per prediction
            - total_inference_time: Total inference time
            - cascading_speedup: Percentage of predictions that skipped stability
        """
        # Compute average scores
        avg_conf_accepted = (
            np.mean(self.metrics["avg_confidence_accepted"])
            if self.metrics["avg_confidence_accepted"]
            else 0.0
        )
        avg_conf_rejected = (
            np.mean(self.metrics["avg_confidence_rejected"])
            if self.metrics["avg_confidence_rejected"]
            else 0.0
        )
        avg_stab_accepted = (
            np.mean(self.metrics["avg_stability_accepted"])
            if self.metrics["avg_stability_accepted"]
            else 0.0
        )
        avg_stab_rejected = (
            np.mean(self.metrics["avg_stability_rejected"])
            if self.metrics["avg_stability_rejected"]
            else 0.0
        )

        # Compute gaps
        confidence_gap = avg_conf_accepted - avg_conf_rejected
        stability_gap = avg_stab_accepted - avg_stab_rejected

        # Cascading speedup
        cascading_speedup = 0.0
        if self.total_predictions > 0:
            cascading_speedup = (
                self.fast_accepts + self.fast_rejects
            ) / self.total_predictions

        # Average inference time
        avg_inference_time = 0.0
        if self.total_predictions > 0:
            avg_inference_time = (
                self.metrics["total_inference_time"] / self.total_predictions
            )

        return {
            "total_predictions": self.total_predictions,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "coverage": self.coverage,
            "confidence_threshold": self.confidence_threshold,
            "stability_threshold": self.stability_threshold,
            "strategy": self.strategy.value,
            "fast_accepts": self.fast_accepts,
            "fast_rejects": self.fast_rejects,
            "robust_accepts": self.robust_accepts,
            "robust_rejects": self.robust_rejects,
            "avg_confidence_accepted": float(avg_conf_accepted),
            "avg_confidence_rejected": float(avg_conf_rejected),
            "avg_stability_accepted": float(avg_stab_accepted),
            "avg_stability_rejected": float(avg_stab_rejected),
            "confidence_gap": float(confidence_gap),
            "stability_gap": float(stability_gap),
            "rejection_breakdown": dict(self.metrics["rejection_reasons"]),
            "decision_breakdown": dict(self.metrics["decision_strategies"]),
            "avg_inference_time": float(avg_inference_time),
            "total_inference_time": float(self.metrics["total_inference_time"]),
            "confidence_computation_time": float(
                self.metrics["confidence_computation_time"]
            ),
            "stability_computation_time": float(
                self.metrics["stability_computation_time"]
            ),
            "gating_computation_time": float(self.metrics["gating_computation_time"]),
            "cascading_speedup": float(cascading_speedup),
        }

    def tune_thresholds_for_coverage(
        self,
        images: Tensor,
        labels: Union[List[int], Tensor],
        target_coverage: float = 0.90,
        confidence_grid: Optional[List[float]] = None,
        stability_grid: Optional[List[float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Tune thresholds to achieve target coverage on validation set.

        This method performs grid search to find optimal thresholds that:
        1. Achieve target coverage (e.g., 90%)
        2. Maximize selective accuracy on accepted samples

        Parameters
        ----------
        images : Tensor
            Validation images, shape (N, C, H, W).

        labels : List[int] or Tensor
            Ground truth labels.

        target_coverage : float, optional (default=0.90)
            Desired coverage rate [0, 1].

        confidence_grid : List[float], optional
            Grid of confidence thresholds to search.
            Default: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        stability_grid : List[float], optional
            Grid of stability thresholds to search.
            Default: [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

        Returns
        -------
        best_conf_threshold : float
            Optimal confidence threshold.

        best_stab_threshold : float
            Optimal stability threshold.

        best_selective_acc : float
            Selective accuracy achieved at target coverage.

        Examples
        --------
        >>> # Tune on validation set
        >>> best_conf, best_stab, acc = predictor.tune_thresholds_for_coverage(
        ...     val_images, val_labels, target_coverage=0.90
        ... )
        >>> print(f"Best thresholds: conf={best_conf:.3f}, stab={best_stab:.3f}")
        >>> print(f"Selective accuracy @ 90% coverage: {acc:.2%}")
        >>>
        >>> # Update predictor with tuned thresholds
        >>> predictor.confidence_threshold = best_conf
        >>> predictor.stability_threshold = best_stab
        """
        if self.verbose:
            logger.info(f"Tuning thresholds for {target_coverage:.1%} coverage...")

        # Default grids
        if confidence_grid is None:
            confidence_grid = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        if stability_grid is None:
            stability_grid = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

        # Convert labels if needed
        if isinstance(labels, Tensor):
            labels = labels.cpu().tolist()

        # Pre-compute all predictions, confidences, and stabilities
        # This is the KEY optimization: compute once, use many times in grid search
        if self.verbose:
            logger.info(
                "Pre-computing scores for all samples (will be cached for grid search)..."
            )

        self.reset_statistics()  # Don't count tuning in statistics

        # Disable cascading during threshold tuning to get true stability scores
        original_cascading = self.enable_cascading
        self.enable_cascading = False

        results = self.predict_batch(
            images, true_labels=labels, return_all_metadata=True
        )

        # Restore cascading setting
        self.enable_cascading = original_cascading

        # Extract scores and labels
        predictions = np.array([r.prediction for r in results])
        confidences = np.array([r.confidence for r in results])
        stabilities = np.array([r.stability for r in results])
        true_labels = np.array(labels)

        # Grid search
        best_conf_threshold = self.confidence_threshold
        best_stab_threshold = self.stability_threshold
        best_selective_acc = 0.0
        best_coverage_diff = float("inf")

        if self.verbose:
            total_combinations = len(confidence_grid) * len(stability_grid)
            pbar = tqdm(total=total_combinations, desc="Grid search")

        for conf_thresh in confidence_grid:
            for stab_thresh in stability_grid:
                # Apply gating logic
                if self.strategy == GatingStrategy.CONFIDENCE_ONLY:
                    accepted_mask = confidences >= conf_thresh
                elif self.strategy == GatingStrategy.STABILITY_ONLY:
                    accepted_mask = stabilities >= stab_thresh
                else:  # COMBINED
                    accepted_mask = (confidences >= conf_thresh) & (
                        stabilities >= stab_thresh
                    )

                # Compute coverage
                coverage = accepted_mask.mean()
                coverage_diff = abs(coverage - target_coverage)

                # Compute selective accuracy on accepted samples
                if accepted_mask.sum() > 0:
                    accepted_preds = predictions[accepted_mask]
                    accepted_labels = true_labels[accepted_mask]
                    selective_acc = (accepted_preds == accepted_labels).mean()
                else:
                    selective_acc = 0.0

                # Update best if:
                # 1. Coverage is closer to target AND accuracy is not worse
                # 2. Coverage is same but accuracy is better
                if coverage_diff < best_coverage_diff:
                    best_conf_threshold = conf_thresh
                    best_stab_threshold = stab_thresh
                    best_selective_acc = selective_acc
                    best_coverage_diff = coverage_diff
                elif (
                    coverage_diff == best_coverage_diff
                    and selective_acc > best_selective_acc
                ):
                    best_conf_threshold = conf_thresh
                    best_stab_threshold = stab_thresh
                    best_selective_acc = selective_acc

                if self.verbose:
                    pbar.update(1)

        if self.verbose:
            pbar.close()
            logger.info(f"Best thresholds found:")
            logger.info(f"  Confidence: {best_conf_threshold:.3f}")
            logger.info(f"  Stability: {best_stab_threshold:.3f}")
            logger.info(f"  Selective Accuracy: {best_selective_acc:.2%}")
            logger.info(f"  Coverage difference: {best_coverage_diff:.4f}")

        self.reset_statistics()  # Reset after tuning

        return best_conf_threshold, best_stab_threshold, best_selective_acc

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SelectivePredictor("
            f"strategy={self.strategy.value}, "
            f"conf_thresh={self.confidence_threshold:.3f}, "
            f"stab_thresh={self.stability_threshold:.3f}, "
            f"coverage={self.coverage:.2%})"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_selective_metrics(
    results: List[SelectionResult], verbose: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive selective prediction metrics.

    Parameters
    ----------
    results : List[SelectionResult]
        List of prediction results.

    verbose : bool, optional (default=True)
        Whether to print metrics.

    Returns
    -------
    dict
        Dictionary containing:
        - coverage: Fraction of samples accepted
        - selective_accuracy: Accuracy on accepted samples
        - overall_accuracy: Accuracy on all samples
        - risk_on_rejected: Error rate on rejected samples
        - mean_confidence_accepted: Average confidence on accepted
        - mean_confidence_rejected: Average confidence on rejected
        - mean_stability_accepted: Average stability on accepted
        - mean_stability_rejected: Average stability on rejected

    Examples
    --------
    >>> results = predictor.predict_batch(test_images, test_labels)
    >>> metrics = compute_selective_metrics(results)
    >>> print(f"Coverage: {metrics['coverage']:.2%}")
    >>> print(f"Selective Accuracy: {metrics['selective_accuracy']:.2%}")
    """
    # Split by acceptance
    accepted = [r for r in results if r.is_accepted]
    rejected = [r for r in results if not r.is_accepted]

    # Coverage
    coverage = len(accepted) / len(results) if len(results) > 0 else 0.0

    # Selective accuracy (only on accepted with known labels)
    accepted_with_labels = [r for r in accepted if r.true_label is not None]
    if len(accepted_with_labels) > 0:
        correct_accepted = sum(
            r.prediction == r.true_label for r in accepted_with_labels
        )
        selective_accuracy = correct_accepted / len(accepted_with_labels)
    else:
        selective_accuracy = 0.0

    # Overall accuracy
    with_labels = [r for r in results if r.true_label is not None]
    if len(with_labels) > 0:
        correct_overall = sum(r.prediction == r.true_label for r in with_labels)
        overall_accuracy = correct_overall / len(with_labels)
    else:
        overall_accuracy = 0.0

    # Risk on rejected
    rejected_with_labels = [r for r in rejected if r.true_label is not None]
    if len(rejected_with_labels) > 0:
        incorrect_rejected = sum(
            r.prediction != r.true_label for r in rejected_with_labels
        )
        risk_on_rejected = incorrect_rejected / len(rejected_with_labels)
    else:
        risk_on_rejected = 0.0

    # Mean scores
    mean_conf_accepted = (
        np.mean([r.confidence for r in accepted]) if len(accepted) > 0 else 0.0
    )
    mean_conf_rejected = (
        np.mean([r.confidence for r in rejected]) if len(rejected) > 0 else 0.0
    )
    mean_stab_accepted = (
        np.mean([r.stability for r in accepted]) if len(accepted) > 0 else 0.0
    )
    mean_stab_rejected = (
        np.mean([r.stability for r in rejected]) if len(rejected) > 0 else 0.0
    )

    metrics = {
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "overall_accuracy": overall_accuracy,
        "risk_on_rejected": risk_on_rejected,
        "mean_confidence_accepted": mean_conf_accepted,
        "mean_confidence_rejected": mean_conf_rejected,
        "mean_stability_accepted": mean_stab_accepted,
        "mean_stability_rejected": mean_stab_rejected,
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "total": len(results),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("SELECTIVE PREDICTION METRICS")
        print("=" * 80)
        print(f"\n📊 Coverage:")
        print(
            f"   Accepted: {metrics['num_accepted']}/{metrics['total']} ({metrics['coverage']:.2%})"
        )
        print(
            f"   Rejected: {metrics['num_rejected']}/{metrics['total']} ({1 - metrics['coverage']:.2%})"
        )
        print(f"\n🎯 Accuracy:")
        print(f"   Selective (accepted only): {metrics['selective_accuracy']:.2%}")
        print(f"   Overall (all samples):     {metrics['overall_accuracy']:.2%}")
        print(
            f"   Improvement: {(metrics['selective_accuracy'] - metrics['overall_accuracy']) * 100:+.2f}pp"
        )
        print(f"\n⚠️  Risk:")
        print(f"   Error rate on rejected: {metrics['risk_on_rejected']:.2%}")
        print(f"\n📈 Mean Scores:")
        print(f"   Confidence (accepted): {metrics['mean_confidence_accepted']:.3f}")
        print(f"   Confidence (rejected): {metrics['mean_confidence_rejected']:.3f}")
        print(f"   Stability (accepted):  {metrics['mean_stability_accepted']:.3f}")
        print(f"   Stability (rejected):  {metrics['mean_stability_rejected']:.3f}")
        print("=" * 80 + "\n")

    return metrics


if __name__ == "__main__":
    # This block is for testing only
    print("SelectivePredictor module loaded successfully!")
    print(f"Available strategies: {[s.value for s in GatingStrategy]}")
    print(
        "\nFor usage examples, see the docstrings or notebooks/PHASE_8_SELECTIVE_PREDICTION.ipynb"
    )
