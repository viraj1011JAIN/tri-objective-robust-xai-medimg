"""
Base loss module.

Defines :class:`BaseLoss`, an abstract superclass for all custom loss
functions in the Tri-Objective Robust XAI pipeline.

Key responsibilities
--------------------
* Enforce a consistent ``reduction`` API ("none", "mean", "sum")
* Provide shared input validation helpers
* Track basic loss statistics (mean / min / max across calls)
* Offer a clean, inspectable representation for logging/debugging

All concrete loss implementations (task loss, calibration loss,
explanation losses, etc.) should inherit from :class:`BaseLoss`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom losses.

    Parameters
    ----------
    reduction :
        Reduction method for the loss output. Must be one of:
        - "none": no reduction, return per-element/per-sample losses
        - "mean": average over all elements
        - "sum": sum over all elements
    name :
        Optional human-readable name for this loss. If omitted, the
        class name is used.

    Attributes
    ----------
    reduction : str
        Reduction mode used by the loss.
    name : str
        Human-readable name.
    _num_calls : int
        Number of times the loss has been called.
    _total_loss : float
        Accumulated (scalar) loss across calls.
    _min_loss : float
        Minimum scalar loss observed so far.
    _max_loss : float
        Maximum scalar loss observed so far.
    """

    def __init__(
        self,
        reduction: str = "mean",
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        reduction = reduction.lower()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Must be one of "
                "{'none', 'mean', 'sum'}."
            )

        self.reduction: str = reduction
        self.name: str = name or self.__class__.__name__

        # Statistics tracking (for logging/monitoring)
        self._num_calls: int = 0
        self._total_loss: float = 0.0
        self._min_loss: float = float("inf")
        self._max_loss: float = float("-inf")

    # --------------------------------------------------------------------- #
    # Core interface                                                        #
    # --------------------------------------------------------------------- #

    @abstractmethod
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute the loss value.

        Concrete subclasses must implement this method and are expected
        to respect the ``reduction`` attribute when aggregating losses.

        Parameters
        ----------
        predictions :
            Model outputs (e.g., logits). Shape and type depend on the
            concrete loss.
        targets :
            Ground-truth targets (e.g., class indices or binary labels).
        **kwargs :
            Optional extra arguments specific to the loss.

        Returns
        -------
        Tensor
            Loss tensor. For ``reduction != 'none'`` this is usually a
            scalar; for ``reduction == 'none'`` this may be a batch
            vector or matrix.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Shared helpers                                                        #
    # --------------------------------------------------------------------- #

    def _validate_inputs(self, predictions: Tensor, targets: Tensor) -> None:
        """
        Common validation for ``predictions`` and ``targets``.

        This method is intended to catch obvious issues early. Concrete
        losses are free to perform additional, stricter validation on
        shapes, dtypes, etc.

        Checks performed
        ----------------
        * Both inputs must be ``torch.Tensor`` instances.
        * Batch dimension (dim 0) must match.
        * ``predictions`` must not contain NaN or +/-Inf.

        Parameters
        ----------
        predictions :
            Model predictions.
        targets :
            Ground-truth targets.

        Raises
        ------
        TypeError
            If either argument is not a tensor.
        ValueError
            If batch sizes differ, or if predictions contain NaN/Inf.
        """
        if not isinstance(predictions, Tensor):
            raise TypeError(
                f"predictions must be a torch.Tensor, got {type(predictions)}"
            )
        if not isinstance(targets, Tensor):
            raise TypeError(f"targets must be a torch.Tensor, got {type(targets)}")

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"predictions batch={predictions.shape[0]} vs "
                f"targets batch={targets.shape[0]}"
            )

        if torch.isnan(predictions).any():
            raise ValueError("predictions contains NaN values")
        if torch.isinf(predictions).any():
            raise ValueError("predictions contains Inf values")

    def _update_statistics(self, loss: Tensor) -> None:
        """
        Update internal statistics based on the given loss tensor.

        The loss may be scalar or a batch of values (e.g. when
        ``reduction == 'none'``). In the latter case we track the mean.

        Parameters
        ----------
        loss :
            Loss tensor for a single forward call.
        """
        with torch.no_grad():
            # Handle scalar vs vector/matrix losses
            if loss.numel() == 1:
                value = float(loss.item())
            else:
                value = float(loss.mean().item())

            self._num_calls += 1
            self._total_loss += value
            self._min_loss = min(self._min_loss, value)
            self._max_loss = max(self._max_loss, value)

    # --------------------------------------------------------------------- #
    # Statistics / introspection                                            #
    # --------------------------------------------------------------------- #

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return basic descriptive statistics for this loss instance.

        Returns
        -------
        dict
            Dictionary with keys:
            - "name"
            - "num_calls"
            - "mean_loss"
            - "min_loss"
            - "max_loss"
        """
        if self._num_calls > 0:
            mean_loss = self._total_loss / float(self._num_calls)
            min_loss = self._min_loss
            max_loss = self._max_loss
        else:
            mean_loss = 0.0
            min_loss = 0.0
            max_loss = 0.0

        return {
            "name": self.name,
            "num_calls": self._num_calls,
            "mean_loss": float(mean_loss),
            "min_loss": float(min_loss),
            "max_loss": float(max_loss),
        }

    def reset_statistics(self) -> None:
        """Reset all accumulated statistics to their initial state."""
        self._num_calls = 0
        self._total_loss = 0.0
        self._min_loss = float("inf")
        self._max_loss = float("-inf")

    # --------------------------------------------------------------------- #
    # nn.Module / repr integration                                          #
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        """Compact string representation for logging and debugging."""
        return f"{self.__class__.__name__}(name={self.name!r}, reduction={self.reduction!r})"

    def extra_repr(self) -> str:
        """Hook for ``nn.Module`` pretty-printing."""
        stats = self.get_statistics()
        # Only show stats after at least one call to avoid noise.
        if stats["num_calls"] == 0:
            return f"name={self.name!r}, reduction={self.reduction!r}"
        return (
            f"name={self.name!r}, reduction={self.reduction!r}, "
            f"calls={stats['num_calls']}, "
            f"mean_loss={stats['mean_loss']:.4f}, "
            f"min_loss={stats['min_loss']:.4f}, "
            f"max_loss={stats['max_loss']:.4f}"
        )


__all__ = ["BaseLoss"]
