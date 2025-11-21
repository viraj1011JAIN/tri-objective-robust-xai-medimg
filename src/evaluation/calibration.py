"""
Calibration Evaluation Module for Phase 3.3 Baseline Training Integration.

This module implements calibration metrics and visualization tools:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagrams (calibration plots)
- Confidence histograms

These metrics evaluate whether predicted confidences match true probabilities,
which is critical for trustworthy medical AI systems.

Usage:
    from src.evaluation.calibration import (
        calculate_ece,
        calculate_mce,
        plot_reliability_diagram,
    )

    # Calculate ECE
    ece = calculate_ece(predictions, labels, num_bins=15)

    # Plot reliability diagram
    fig = plot_reliability_diagram(predictions, labels, num_bins=15)
    fig.savefig("calibration_plot.png")

Phase 3.3 Baseline Training Integration - Master's Dissertation
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def calculate_ece(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_bins: int = 15,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual
    accuracy across equally-sized bins.

    Args:
        predictions: Predicted probabilities (N, C) or confidence scores (N,)
        labels: True class labels (N,)
        num_bins: Number of bins for calibration (typically 10-20)

    Returns:
        ECE value (0 = perfect calibration, higher = worse)

    Reference:
        Naeini et al. (2015). "Obtaining Well Calibrated Probabilities
        Using Bayesian Binning"
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get confidence scores (max probability)
    if predictions.ndim == 2:
        # Multi-class: (N, C)
        confidences = np.max(predictions, axis=1)
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        # Binary or already max confidence: (N,)
        confidences = predictions
        predicted_labels = (predictions > 0.5).astype(int)

    # Check if predicted labels match true labels
    correct = predicted_labels == labels

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Calculate ECE
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum() / len(confidences)

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = correct[in_bin].mean()
            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Add weighted difference to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def calculate_mce(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_bins: int = 15,
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    MCE is the maximum difference between confidence and accuracy across bins.
    It captures the worst-case calibration error.

    Args:
        predictions: Predicted probabilities (N, C) or confidence scores (N,)
        labels: True class labels (N,)
        num_bins: Number of bins for calibration

    Returns:
        MCE value (0 = perfect calibration, higher = worse)
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get confidence scores
    if predictions.ndim == 2:
        confidences = np.max(predictions, axis=1)
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        confidences = predictions
        predicted_labels = (predictions > 0.5).astype(int)

    correct = predicted_labels == labels

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Calculate MCE
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = max(mce, bin_error)

    return float(mce)


def plot_reliability_diagram(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot reliability diagram (calibration plot).

    The reliability diagram plots predicted confidence vs actual accuracy.
    A perfectly calibrated model lies on the diagonal.

    Args:
        predictions: Predicted probabilities (N, C) or confidence scores (N,)
        labels: True class labels (N,)
        num_bins: Number of bins for calibration
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object

    Reference:
        DeGroot & Fienberg (1983). "The Comparison and Evaluation of
        Forecasters"
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get confidence scores
    if predictions.ndim == 2:
        confidences = np.max(predictions, axis=1)
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        confidences = predictions
        predicted_labels = (predictions > 0.5).astype(int)

    correct = predicted_labels == labels

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    # Calculate bin statistics
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = in_bin.sum()

        if count > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Calculate ECE for display
    ece = calculate_ece(predictions, labels, num_bins)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line (diagonal)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

    # Plot reliability curve
    ax.plot(
        bin_confidences,
        bin_accuracies,
        "o-",
        label=f"Model (ECE={ece:.4f})",
        linewidth=2,
        markersize=8,
    )

    # Add bar chart showing sample distribution
    bin_counts_sum = float(np.sum(bin_counts))
    ax.bar(
        bin_centers,
        bin_counts / bin_counts_sum,
        width=1.0 / num_bins,
        alpha=0.3,
        color="gray",
        label="Sample Distribution",
    )

    # Formatting
    ax.set_xlabel("Confidence (Predicted Probability)", fontsize=14)
    ax.set_ylabel("Accuracy (Fraction Correct)", fontsize=14)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=16)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Reliability diagram saved to {save_path}")

    return fig


def plot_confidence_histogram(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_bins: int = 50,
    title: str = "Confidence Histogram",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot confidence histogram showing distribution of predicted confidences.

    Args:
        predictions: Predicted probabilities (N, C) or confidence scores (N,)
        labels: True class labels (N,)
        num_bins: Number of histogram bins
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get confidence scores
    if predictions.ndim == 2:
        confidences = np.max(predictions, axis=1)
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        confidences = predictions
        predicted_labels = (predictions > 0.5).astype(int)

    correct = predicted_labels == labels

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms for correct and incorrect predictions
    ax.hist(
        confidences[correct],
        bins=num_bins,
        alpha=0.7,
        label="Correct Predictions",
        color="green",
        edgecolor="black",
    )
    ax.hist(
        confidences[~correct],
        bins=num_bins,
        alpha=0.7,
        label="Incorrect Predictions",
        color="red",
        edgecolor="black",
    )

    # Formatting
    ax.set_xlabel("Confidence (Predicted Probability)", fontsize=14)
    ax.set_ylabel("Number of Samples", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    avg_conf_correct = confidences[correct].mean()
    avg_conf_incorrect = confidences[~correct].mean()
    ax.text(
        0.98,
        0.98,
        f"Avg Conf (Correct): {avg_conf_correct:.4f}\n"
        f"Avg Conf (Incorrect): {avg_conf_incorrect:.4f}\n"
        f"Accuracy: {correct.mean():.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confidence histogram saved to {save_path}")

    return fig


def evaluate_calibration(
    predictions: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_bins: int = 15,
    output_dir: Optional[str] = None,
) -> dict[str, float]:
    """
    Comprehensive calibration evaluation.

    Calculate all calibration metrics and optionally generate plots.

    Args:
        predictions: Predicted probabilities (N, C) or confidence scores (N,)
        labels: True class labels (N,)
        num_bins: Number of bins for calibration
        output_dir: Optional directory to save plots

    Returns:
        Dictionary containing:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - accuracy: Overall accuracy
            - avg_confidence: Average predicted confidence
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get confidence scores
    if predictions.ndim == 2:
        confidences = np.max(predictions, axis=1)
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        confidences = predictions
        predicted_labels = (predictions > 0.5).astype(int)

    correct = predicted_labels == labels

    # Calculate metrics
    ece = calculate_ece(predictions, labels, num_bins)
    mce = calculate_mce(predictions, labels, num_bins)
    accuracy = correct.mean()
    avg_confidence = confidences.mean()

    metrics = {
        "ece": float(ece),
        "mce": float(mce),
        "accuracy": float(accuracy),
        "avg_confidence": float(avg_confidence),
    }

    logger.info("Calibration Metrics:")
    logger.info(f"  ECE: {ece:.4f}")
    logger.info(f"  MCE: {mce:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Avg Confidence: {avg_confidence:.4f}")

    # Generate plots if output directory provided
    if output_dir:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Reliability diagram
        fig1 = plot_reliability_diagram(
            predictions,
            labels,
            num_bins,
            save_path=str(output_path / "reliability_diagram.png"),
        )
        plt.close(fig1)

        # Confidence histogram
        fig2 = plot_confidence_histogram(
            predictions,
            labels,
            save_path=str(output_path / "confidence_histogram.png"),
        )
        plt.close(fig2)

        logger.info(f"Plots saved to {output_path}")

    return metrics
