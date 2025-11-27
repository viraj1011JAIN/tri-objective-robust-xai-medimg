"""
Confidence Scoring Module for Selective Prediction (Phase 8.1).

This module implements multiple confidence estimation methods for selective
prediction in medical imaging. These methods are critical for clinical deployment
where the model can reject uncertain predictions.

Implemented Confidence Scores
------------------------------
1. **Softmax Maximum**: max(softmax(logits)) - Simple, fast, but often overconfident
2. **Predictive Entropy**: H(p) = -Σ p_i log(p_i) - Better uncertainty quantification
3. **MC Dropout**: Multiple stochastic forward passes for uncertainty estimation
4. **Temperature Scaling**: Post-hoc calibration for better confidence estimates

Key Metrics
-----------
- **Confidence Score**: Higher = more confident prediction
- **Uncertainty Score**: Higher = more uncertain (inverse of confidence)
- **Epistemic Uncertainty**: Model uncertainty (reducible with more data)
- **Aleatoric Uncertainty**: Data noise (irreducible)

References
----------
1. Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
2. Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
3. Geifman & El-Yaniv, "Selective Classification" (JMLR 2017)

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..losses.calibration_loss import TemperatureScaling


class ConfidenceMethod(Enum):
    """Available confidence scoring methods."""

    SOFTMAX_MAX = "softmax_max"  # Maximum softmax probability
    ENTROPY = "entropy"  # Predictive entropy
    MC_DROPOUT = "mc_dropout"  # Monte Carlo Dropout
    TEMPERATURE_SCALED = "temperature_scaled"  # Temperature-scaled softmax


@dataclass
class ConfidenceScore:
    """
    Container for confidence scores and related metrics.

    Attributes
    ----------
    confidence : float
        Confidence score in [0, 1]. Higher = more confident.
    uncertainty : float
        Uncertainty score (typically 1 - confidence or entropy).
    method : ConfidenceMethod
        Method used to compute confidence.
    prediction : int
        Predicted class index.
    probabilities : np.ndarray
        Full probability distribution (num_classes,).
    metadata : Dict[str, Any]
        Additional method-specific information (e.g., MC Dropout variance).
    """

    confidence: float
    uncertainty: float
    method: ConfidenceMethod
    prediction: int
    probabilities: np.ndarray
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate confidence and uncertainty ranges."""
        # Clip confidence to [0, 1] for numerical stability
        # (floating point errors can produce values like -1e-7 or 1.0000001)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

        if self.uncertainty < 0:
            raise ValueError(f"Uncertainty must be >= 0, got {self.uncertainty}")


class SoftmaxMaxScorer:
    """
    Softmax maximum confidence scorer.

    Returns max(softmax(logits)) as confidence. This is the simplest method
    but tends to be overconfident for neural networks.

    Parameters
    ----------
    temperature : float, default=1.0
        Temperature for softmax scaling. T > 1 produces softer probabilities.

    Example
    -------
    >>> scorer = SoftmaxMaxScorer(temperature=1.0)
    >>> logits = torch.tensor([[2.0, 1.0, 0.5]])
    >>> score = scorer(logits)
    >>> print(f"Confidence: {score.confidence:.3f}")
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize softmax maximum scorer."""
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> ConfidenceScore:
        """
        Compute confidence score from logits.

        Parameters
        ----------
        logits : Tensor
            Model logits. Shape: (batch_size, num_classes) or (num_classes,).

        Returns
        -------
        ConfidenceScore
            Confidence score object. If batch_size > 1, returns score for first sample.
        """
        # Handle single sample or batch
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute softmax probabilities
        probs = F.softmax(scaled_logits, dim=1)

        # Get prediction and confidence (for first sample if batch)
        probs_sample = probs[0].detach().cpu().numpy()
        prediction = int(np.argmax(probs_sample))
        confidence = float(np.max(probs_sample))

        # Uncertainty as 1 - confidence (simple measure)
        uncertainty = 1.0 - confidence

        return ConfidenceScore(
            confidence=confidence,
            uncertainty=uncertainty,
            method=ConfidenceMethod.SOFTMAX_MAX,
            prediction=prediction,
            probabilities=probs_sample,
            metadata={"temperature": self.temperature},
        )


class EntropyScorer:
    """
    Predictive entropy confidence scorer.

    Computes H(p) = -Σ p_i log(p_i) where p is the predictive distribution.
    Entropy is 0 for deterministic predictions and log(C) for uniform distribution.

    Confidence is computed as: confidence = 1 - (entropy / log(num_classes))
    This normalizes entropy to [0, 1] range.

    Parameters
    ----------
    temperature : float, default=1.0
        Temperature for softmax scaling.
    epsilon : float, default=1e-10
        Small constant for numerical stability in log computation.

    Example
    -------
    >>> scorer = EntropyScorer(temperature=1.0)
    >>> logits = torch.tensor([[2.0, 1.0, 0.5]])
    >>> score = scorer(logits)
    >>> print(f"Entropy: {score.metadata['entropy']:.3f}")
    """

    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-10) -> None:
        """Initialize entropy scorer."""
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        self.temperature = temperature
        self.epsilon = epsilon

    def __call__(self, logits: Tensor) -> ConfidenceScore:
        """
        Compute entropy-based confidence score.

        Parameters
        ----------
        logits : Tensor
            Model logits. Shape: (batch_size, num_classes) or (num_classes,).

        Returns
        -------
        ConfidenceScore
            Confidence score with entropy as uncertainty measure.
        """
        # Handle single sample or batch
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        num_classes = logits.shape[1]

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute softmax probabilities
        probs = F.softmax(scaled_logits, dim=1)

        # Get prediction (for first sample if batch)
        probs_sample = probs[0].detach().cpu().numpy()
        prediction = int(np.argmax(probs_sample))

        # Compute predictive entropy
        # H(p) = -Σ p_i log(p_i)
        probs_stable = np.clip(probs_sample, self.epsilon, 1.0 - self.epsilon)
        entropy = -np.sum(probs_stable * np.log(probs_stable))

        # Normalize entropy to [0, 1] by dividing by max entropy
        max_entropy = np.log(num_classes)
        normalized_entropy = entropy / max_entropy

        # Convert entropy to confidence (inverse relationship)
        confidence = 1.0 - normalized_entropy

        return ConfidenceScore(
            confidence=confidence,
            uncertainty=entropy,
            method=ConfidenceMethod.ENTROPY,
            prediction=prediction,
            probabilities=probs_sample,
            metadata={
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "max_entropy": float(max_entropy),
                "temperature": self.temperature,
            },
        )


class MCDropoutScorer:
    """
    Monte Carlo Dropout uncertainty estimator.

    Performs multiple stochastic forward passes with dropout enabled at test time
    to estimate epistemic uncertainty. This captures model uncertainty and is
    particularly useful for out-of-distribution detection.

    The method approximates Bayesian inference by treating dropout as variational
    Bayesian inference (Gal & Ghahramani, 2016).

    Parameters
    ----------
    model : nn.Module
        Neural network model with dropout layers.
    num_samples : int, default=20
        Number of forward passes for Monte Carlo estimation.
        Typical values: 10-50 (trade-off between accuracy and speed).
    temperature : float, default=1.0
        Temperature for softmax scaling.
    device : torch.device or str, default="cuda"
        Device for computation.

    Example
    -------
    >>> scorer = MCDropoutScorer(model, num_samples=20, device="cuda")
    >>> x = torch.randn(1, 3, 224, 224).cuda()
    >>> score = scorer(x)
    >>> print(f"Mean confidence: {score.confidence:.3f}")
    >>> print(f"Predictive variance: {score.metadata['variance']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 20,
        temperature: float = 1.0,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """Initialize MC Dropout scorer."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.model = model
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = torch.device(device) if isinstance(device, str) else device

        # Move model to device
        self.model.to(self.device)

        # Check if model has dropout layers
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())
        if not has_dropout:
            warnings.warn(
                "Model has no Dropout layers. MC Dropout requires dropout layers "
                "to estimate uncertainty. Results may not be meaningful.",
                UserWarning,
            )

    def _enable_dropout(self) -> None:
        """Enable dropout in all Dropout layers (even in eval mode)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def __call__(self, x: Tensor) -> ConfidenceScore:
        """
        Compute MC Dropout confidence score.

        Parameters
        ----------
        x : Tensor
            Input tensor. Shape: (batch_size, C, H, W) or (C, H, W).

        Returns
        -------
        ConfidenceScore
            Confidence score with MC Dropout variance as uncertainty.
        """
        # Handle single sample
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Only process first sample if batch
        if x.shape[0] > 1:
            warnings.warn(
                f"Batch size > 1 detected ({x.shape[0]} samples). "
                "Processing only the first sample for efficiency.",
                UserWarning,
            )
            x = x[0:1]

        x = x.to(self.device)

        # Enable dropout for stochastic forward passes
        self._enable_dropout()

        # Collect predictions from multiple forward passes
        predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                # Forward pass with dropout enabled
                logits = self.model(x)

                # Apply temperature scaling
                scaled_logits = logits / self.temperature

                # Get probabilities
                probs = F.softmax(scaled_logits, dim=1)
                predictions.append(probs.cpu().numpy())

        # Restore model to eval mode
        self.model.eval()

        # Stack predictions: (num_samples, num_classes)
        predictions = np.array(predictions).squeeze(axis=1)

        # Compute mean prediction
        mean_probs = np.mean(predictions, axis=0)

        # Compute predictive variance (epistemic uncertainty)
        variance = np.mean(np.var(predictions, axis=0))

        # Compute predictive entropy from mean probabilities
        epsilon = 1e-10
        mean_probs_stable = np.clip(mean_probs, epsilon, 1.0 - epsilon)
        entropy = -np.sum(mean_probs_stable * np.log(mean_probs_stable))

        # Prediction and confidence
        prediction = int(np.argmax(mean_probs))
        confidence = float(mean_probs[prediction])

        # Alternative confidence: use variance as uncertainty
        # Lower variance = higher confidence
        normalized_variance = float(np.clip(variance, 0, 1))
        confidence_from_variance = 1.0 - normalized_variance

        return ConfidenceScore(
            confidence=confidence,
            uncertainty=entropy,
            method=ConfidenceMethod.MC_DROPOUT,
            prediction=prediction,
            probabilities=mean_probs,
            metadata={
                "variance": float(variance),
                "normalized_variance": normalized_variance,
                "confidence_from_variance": confidence_from_variance,
                "entropy": float(entropy),
                "num_samples": self.num_samples,
                "temperature": self.temperature,
                "individual_predictions": predictions.tolist(),
            },
        )


class TemperatureScaledScorer:
    """
    Temperature-scaled confidence scorer.

    Uses a learned temperature parameter to calibrate model predictions.
    Temperature scaling is a simple post-hoc calibration method that improves
    confidence estimates without retraining the model.

    The temperature parameter should be fitted on a validation set after training.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    temperature_module : TemperatureScaling or None, default=None
        Pre-fitted temperature scaling module. If None, uses temperature=1.5.
    device : torch.device or str, default="cuda"
        Device for computation.

    Example
    -------
    >>> # Fit temperature on validation set
    >>> temp_module = TemperatureScaling(init_temperature=1.0)
    >>> optimizer = torch.optim.LBFGS([temp_module.log_temperature], lr=0.01)
    >>> # ... fit temperature ...
    >>>
    >>> # Use fitted temperature for scoring
    >>> scorer = TemperatureScaledScorer(model, temp_module, device="cuda")
    >>> x = torch.randn(1, 3, 224, 224).cuda()
    >>> score = scorer(x)
    >>> print(f"Calibrated confidence: {score.confidence:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        temperature_module: Optional[TemperatureScaling] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """Initialize temperature-scaled scorer."""
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device

        # Use provided temperature module or create default
        if temperature_module is None:
            warnings.warn(
                "No temperature module provided. Using default temperature=1.5. "
                "For best results, fit temperature on validation set.",
                UserWarning,
            )
            self.temperature_module = TemperatureScaling(init_temperature=1.5)
        else:
            self.temperature_module = temperature_module

        # Move modules to device
        self.model.to(self.device)
        self.temperature_module.to(self.device)

    def __call__(self, x: Tensor) -> ConfidenceScore:
        """
        Compute temperature-scaled confidence score.

        Parameters
        ----------
        x : Tensor
            Input tensor. Shape: (batch_size, C, H, W) or (C, H, W).

        Returns
        -------
        ConfidenceScore
            Temperature-calibrated confidence score.
        """
        # Handle single sample
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Only process first sample if batch
        if x.shape[0] > 1:
            warnings.warn(
                f"Batch size > 1 detected ({x.shape[0]} samples). "
                "Processing only the first sample.",
                UserWarning,
            )
            x = x[0:1]

        x = x.to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

            # Apply temperature scaling
            calibrated_probs = self.temperature_module(logits)

        # Get prediction and confidence
        probs_sample = calibrated_probs[0].cpu().numpy()
        prediction = int(np.argmax(probs_sample))
        confidence = float(np.max(probs_sample))

        # Compute entropy for uncertainty
        epsilon = 1e-10
        probs_stable = np.clip(probs_sample, epsilon, 1.0 - epsilon)
        entropy = -np.sum(probs_stable * np.log(probs_stable))

        # Get current temperature value
        current_temperature = self.temperature_module.get_temperature()

        return ConfidenceScore(
            confidence=confidence,
            uncertainty=entropy,
            method=ConfidenceMethod.TEMPERATURE_SCALED,
            prediction=prediction,
            probabilities=probs_sample,
            metadata={
                "temperature": float(current_temperature),
                "entropy": float(entropy),
                "is_fitted": hasattr(self.temperature_module, "log_temperature"),
            },
        )


class ConfidenceScorer:
    """
    Unified interface for all confidence scoring methods.

    This class provides a high-level API for computing confidence scores using
    different methods. It automatically selects the appropriate scorer based on
    the specified method.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    method : ConfidenceMethod or str, default="softmax_max"
        Confidence scoring method to use.
    temperature : float, default=1.0
        Temperature for softmax scaling.
    num_mc_samples : int, default=20
        Number of samples for MC Dropout (only used if method="mc_dropout").
    temperature_module : TemperatureScaling or None, default=None
        Pre-fitted temperature module (only used if method="temperature_scaled").
    device : torch.device or str, default="cuda"
        Device for computation.

    Example
    -------
    >>> # Simple usage with softmax maximum
    >>> scorer = ConfidenceScorer(model, method="softmax_max", device="cuda")
    >>> x = torch.randn(1, 3, 224, 224).cuda()
    >>> score = scorer(x)
    >>>
    >>> # MC Dropout for uncertainty estimation
    >>> scorer = ConfidenceScorer(
    ...     model, method="mc_dropout", num_mc_samples=20, device="cuda"
    ... )
    >>> score = scorer(x)
    >>> print(f"Variance: {score.metadata['variance']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        method: Union[ConfidenceMethod, str] = "softmax_max",
        temperature: float = 1.0,
        num_mc_samples: int = 20,
        temperature_module: Optional[TemperatureScaling] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """Initialize confidence scorer."""
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device

        # Convert string to enum if needed
        if isinstance(method, str):
            try:
                method = ConfidenceMethod(method)
            except ValueError:
                valid_methods = [m.value for m in ConfidenceMethod]
                raise ValueError(
                    f"Invalid method '{method}'. Choose from: {valid_methods}"
                )

        self.method = method

        # Initialize appropriate scorer
        if method == ConfidenceMethod.SOFTMAX_MAX:
            self.scorer = SoftmaxMaxScorer(temperature=temperature)
        elif method == ConfidenceMethod.ENTROPY:
            self.scorer = EntropyScorer(temperature=temperature)
        elif method == ConfidenceMethod.MC_DROPOUT:
            self.scorer = MCDropoutScorer(
                model=model,
                num_samples=num_mc_samples,
                temperature=temperature,
                device=device,
            )
        elif method == ConfidenceMethod.TEMPERATURE_SCALED:
            self.scorer = TemperatureScaledScorer(
                model=model,
                temperature_module=temperature_module,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

    def __call__(
        self, x: Union[Tensor, Any]
    ) -> Union[ConfidenceScore, List[ConfidenceScore]]:
        """
        Compute confidence score.

        Parameters
        ----------
        x : Tensor or Any
            Input tensor or logits depending on the method.
            - For softmax_max and entropy: Can pass logits directly (faster).
            - For mc_dropout and temperature_scaled: Must pass input tensor.

        Returns
        -------
        ConfidenceScore or List[ConfidenceScore]
            Confidence score(s). Returns list if batch_size > 1.
        """
        # For softmax_max and entropy, support both logits and input tensors
        if self.method in [ConfidenceMethod.SOFTMAX_MAX, ConfidenceMethod.ENTROPY]:
            # Check if input is logits (2D) or image (4D)
            if isinstance(x, Tensor) and x.dim() in [1, 2]:
                # Input is logits, use directly
                return self.scorer(x)
            else:
                # Input is image, need forward pass
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                x = x.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(x)
                return self.scorer(logits)
        else:
            # MC Dropout and Temperature Scaled need input tensor
            return self.scorer(x)

    def batch_score(
        self, inputs: Tensor, batch_size: int = 32
    ) -> List[ConfidenceScore]:
        """
        Compute confidence scores for a batch of inputs.

        Parameters
        ----------
        inputs : Tensor
            Batch of input tensors. Shape: (N, C, H, W).
        batch_size : int, default=32
            Processing batch size.

        Returns
        -------
        List[ConfidenceScore]
            List of confidence scores for each input.

        Raises
        ------
        ValueError
            If inputs batch is empty (shape[0] == 0).
        """
        # Validate inputs
        if inputs.shape[0] == 0:
            raise ValueError("Cannot compute scores for empty batch")

        scores = []

        # Process in batches
        num_samples = inputs.shape[0]
        for i in range(0, num_samples, batch_size):
            batch = inputs[i : i + batch_size]

            # Process each sample individually (required for MC Dropout)
            for j in range(batch.shape[0]):
                sample = batch[j]
                score = self(sample)
                scores.append(score)

        return scores


def compute_confidence_metrics(
    scores: List[ConfidenceScore],
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute aggregate confidence metrics.

    Parameters
    ----------
    scores : List[ConfidenceScore]
        List of confidence scores.
    labels : np.ndarray or None, default=None
        True labels for accuracy computation (optional).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - mean_confidence: Average confidence score
        - std_confidence: Standard deviation of confidence
        - mean_uncertainty: Average uncertainty
        - accuracy: Classification accuracy (if labels provided)
        - correct_confidence: Mean confidence on correct predictions
        - incorrect_confidence: Mean confidence on incorrect predictions
    """
    confidences = np.array([s.confidence for s in scores])
    uncertainties = np.array([s.uncertainty for s in scores])
    predictions = np.array([s.prediction for s in scores])

    metrics = {
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "method": scores[0].method.value if scores else None,
    }

    if labels is not None:
        correct = predictions == labels
        accuracy = float(np.mean(correct))

        metrics["accuracy"] = accuracy
        metrics["correct_confidence"] = float(np.mean(confidences[correct]))
        metrics["incorrect_confidence"] = float(np.mean(confidences[~correct]))

    return metrics


# Export public API
__all__ = [
    "ConfidenceMethod",
    "ConfidenceScore",
    "SoftmaxMaxScorer",
    "EntropyScorer",
    "MCDropoutScorer",
    "TemperatureScaledScorer",
    "ConfidenceScorer",
    "compute_confidence_metrics",
]
