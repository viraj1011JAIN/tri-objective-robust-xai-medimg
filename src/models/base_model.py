from __future__ import annotations

"""
Base model abstraction for classification architectures.

All CNN / ViT backbones in the Tri-Objective Robust XAI pipeline should
inherit from :class:`BaseModel`.

Responsibilities
----------------
* Provide a consistent interface for:
    - number of classes
    - architecture name / key
    - input channels
* Define abstract hooks:
    - ``forward``           -> logits
    - ``get_feature_maps``  -> intermediate feature maps
* Accept typical configuration arguments (``pretrained``, ``in_channels``)
  without forcing every subclass to re-implement boilerplate.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["BaseModel"]


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all classification models.

    Parameters
    ----------
    num_classes:
        Number of output classes for the classifier head.
    architecture:
        Optional human-readable architecture key, e.g. "resnet50".
        If omitted, defaults to ``self.__class__.__name__.lower()``.
    in_channels:
        Number of input channels (usually 1 for grayscale, 3 for RGB).
    pretrained:
        Optional flag or identifier indicating whether / how to load
        pretrained weights. This is stored but not acted upon here;
        concrete subclasses decide what it means.
    **extra:
        Additional keyword arguments are accepted and stored in
        ``extra_config`` to keep the base class tolerant to future
        configuration fields.

    Attributes
    ----------
    num_classes: int
        Number of output classes.
    architecture: str
        Short architecture identifier (used in logging / registry).
    in_channels: int
        Number of input channels.
    pretrained: bool | str | None
        Pretraining flag / tag as passed to the constructor.
    extra_config: dict
        Any extra keyword arguments that were not explicitly handled.
    """

    def __init__(
        self,
        num_classes: int,
        architecture: Optional[str] = None,
        in_channels: int = 3,
        pretrained: Optional[bool | str] = None,
        **extra: Any,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes!r}")

        self.num_classes: int = int(num_classes)
        self.in_channels: int = int(in_channels)
        self.pretrained: Optional[bool | str] = pretrained
        self.architecture: str = architecture or self.__class__.__name__.lower()
        self.extra_config: Dict[str, Any] = dict(extra)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, H, W)`` or similar.

        Returns
        -------
        Tensor
            Logits of shape ``(B, num_classes)``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_maps(self, x: Tensor) -> Tensor:
        """
        Return intermediate feature maps for interpretability.

        Typically this corresponds to the last convolutional feature map
        before global pooling / classification.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Feature maps of shape ``(B, C_feat, H_feat, W_feat)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Convert logits to probabilities via softmax.

        This is a convenience wrapper around ``forward`` and does not
        change gradients (softmax is differentiable).

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Probabilities with shape ``(B, num_classes)``.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def num_parameters(self, trainable_only: bool = False) -> int:
        """
        Return the number of parameters in the model.

        Parameters
        ----------
        trainable_only:
            If True, count only parameters with ``requires_grad=True``.

        Returns
        -------
        int
            Total number of parameters.
        """
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        else:
            params = self.parameters()
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # Introspection / repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """
        Hook used by ``nn.Module.__repr__`` for pretty-printing.
        """
        extras = ", ".join(f"{k}={v!r}" for k, v in self.extra_config.items())
        base = (
            f"architecture={self.architecture!r}, "
            f"num_classes={self.num_classes}, "
            f"in_channels={self.in_channels}, "
            f"pretrained={self.pretrained!r}"
        )
        return f"{base}" + (f", {extras}" if extras else "")
