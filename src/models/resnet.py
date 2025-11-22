"""
ResNet-based classifier architectures for medical image analysis.

This module provides a ResNet-50-based classifier that integrates with the
Tri-Objective Robust XAI pipeline via the :class:`BaseModel` interface.

Key features
------------
- Uses torchvision's ResNet-50 backbone (with optional ImageNet-1K weights)
- Supports arbitrary input channels (for example, 1-channel chest X-rays)
- Exposes intermediate feature maps for XAI (Grad-CAM etc.)
- Provides a clean embedding interface for TCAV and representation analysis
- Compatible with BaseModel.freeze_backbone() and metadata utilities

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base_model import BaseModel

try:
    # torchvision >= 0.13 style weights API
    from torchvision.models import ResNet50_Weights, resnet50

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - defensive fallback
    _HAS_TORCHVISION = False
    ResNet50_Weights = None  # type: ignore[assignment]
    resnet50 = None  # type: ignore[assignment]


class ResNet50Classifier(BaseModel):
    """
    ResNet-50 classifier tailored for medical image classification.

    This class wraps a torchvision ResNet-50 backbone and adapts it to the
    :class:`BaseModel` interface used in the Tri-Objective Robust XAI pipeline.

    Design choices
    --------------
    - Uses torchvision's official ResNet-50 implementation.
    - Replaces the default 1000-class FC layer with a task-specific head in
      ``self.fc`` so that :meth:`BaseModel.freeze_backbone` can keep it
      trainable by default.
    - Exposes feature maps for ``["layer1", "layer2", "layer3", "layer4"]``
      to support Grad-CAM and related XAI techniques.
    - Provides a penultimate-layer embedding via :meth:`get_embedding`.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the classifier head. Must be greater than 0.
    pretrained : bool, default=True
        If True, load ImageNet-1K pretrained weights for the backbone.
    in_channels : int, default=3
        Number of input channels. If not 3 and ``pretrained=True``, the first
        convolution layer is adapted from the 3-channel weights.
    dropout : float, default=0.0
        Dropout probability in [0.0, 1.0]. If 0, no dropout layer is used.
    global_pool : str, default="avg"
        Type of global pooling. One of {"avg", "max"}.
    **kwargs : Any
        Additional configuration options passed to :class:`BaseModel` and
        stored in ``_extra_config`` for experiment tracking.

    Attributes
    ----------
    backbone : nn.Module
        The ResNet-50 backbone up to (but not including) the classifier head.
    fc : nn.Module
        The final classification head mapping embeddings to logits.
    embedding_dim : int
        Dimensionality of the penultimate feature vector (typically 2048).
    feature_layers : List[str]
        Feature layer names available from :meth:`get_feature_maps`.
    dropout_prob : float
        Stored dropout probability for metadata and config inspection.
    global_pool_mode : str
        Stored global pooling mode ("avg" or "max").
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        global_pool: str = "avg",
        **kwargs: Any,
    ) -> None:
        # Basic value checks to match test expectations
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}")

        if global_pool not in {"avg", "max"}:
            raise ValueError(
                f"Unsupported global_pool: {global_pool}. "
                "Supported values are 'avg' and 'max'."
            )

        if not _HAS_TORCHVISION:
            raise RuntimeError(
                "torchvision is required to use ResNet50Classifier, but it could not "
                "be imported. Please install torchvision."
            )

        # Store config fields
        self.in_channels: int = int(in_channels)
        self.dropout_prob: float = float(dropout)
        self.global_pool_mode: str = str(global_pool)

        # Let BaseModel store extra configuration for metadata
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=self.in_channels,
            dropout=self.dropout_prob,
            global_pool=self.global_pool_mode,
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Backbone setup
        # ------------------------------------------------------------------
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = resnet50(weights=weights)
        # Original FC in_features before replacing it
        backbone_out_features: int = backbone.fc.in_features  # type: ignore[assignment]

        # Replace the original classifier with identity; we attach our own head.
        backbone.fc = nn.Identity()
        self.backbone: nn.Module = backbone
        self.embedding_dim: int = backbone_out_features

        # Adapt the first conv layer for non-3-channel inputs
        if self.in_channels != 3:
            self._adapt_first_conv(in_channels=self.in_channels, pretrained=pretrained)

        # Feature layers exposed for XAI
        self.feature_layers: List[str] = ["layer1", "layer2", "layer3", "layer4"]

        # ------------------------------------------------------------------
        # Global pooling
        # ------------------------------------------------------------------
        if self.global_pool_mode == "avg":
            self.global_pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        else:  # "max"
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        head_modules: List[nn.Module] = []
        if self.dropout_prob > 0.0:
            head_modules.append(nn.Dropout(p=self.dropout_prob))
        head_modules.append(nn.Linear(self.embedding_dim, self.num_classes))

        if len(head_modules) == 1:
            self.fc: nn.Module = head_modules[0]
        else:
            self.fc = nn.Sequential(*head_modules)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _adapt_first_conv(self, in_channels: int, pretrained: bool) -> None:
        """
        Adapt the first convolution layer to support arbitrary input channels.

        When using pretrained weights and ``in_channels == 1``, the new weights
        are initialized as the mean over the RGB channels.

        For other values of ``in_channels``, this performs a best-effort
        initialization by copying available channels and zero-initializing
        any extra channels.
        """
        backbone = self.backbone
        if not hasattr(backbone, "conv1"):
            return

        old_conv: nn.Conv2d = backbone.conv1  # type: ignore[assignment]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )

        if pretrained and isinstance(old_conv.weight, Tensor):
            with torch.no_grad():
                if in_channels == 1:
                    # Average RGB weights to get a single-channel kernel
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    # Best-effort copy of as many channels as possible
                    new_conv.weight.zero_()
                    num_copy = min(3, in_channels, old_conv.weight.shape[1])
                    new_conv.weight[:, :num_copy, :, :].copy_(
                        old_conv.weight[:, :num_copy, :, :]
                    )

        backbone.conv1 = new_conv  # type: ignore[assignment]

    def _extract_backbone_features(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Run the ResNet backbone and return embedding and intermediate features.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Tuple[Tensor, Dict[str, Tensor]]
            - Embeddings of shape (B, D) after global pooling and flattening.
            - Dictionary mapping layer names to feature maps:
              {"layer1": ..., "layer2": ..., "layer3": ..., "layer4": ...}.
        """
        backbone = self.backbone  # type: ignore[assignment]
        features: Dict[str, Tensor] = {}

        # Stem
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        # Residual layers with feature capture
        x = backbone.layer1(x)
        features["layer1"] = x

        x = backbone.layer2(x)
        features["layer2"] = x

        x = backbone.layer3(x)
        features["layer3"] = x

        x = backbone.layer4(x)
        features["layer4"] = x

        # Global pooling and flatten
        x = self.global_pool(x)
        embedding = torch.flatten(x, 1)

        return embedding, features

    # ----------------------------------------------------------------------
    # BaseModel interface implementations
    # ----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass through the ResNet-50 classifier.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).
        return_features : bool, default=False
            If True, also return a dictionary of intermediate feature maps
            suitable for XAI methods.

        Returns
        -------
        Tensor or Tuple[Tensor, Dict[str, Tensor]]
            If return_features is False:
                - Logits of shape (B, num_classes).
            If return_features is True:
                - Tuple of (logits, features_dict), where features_dict maps
                  layer names to activation tensors.
        """
        embedding, features = self._extract_backbone_features(x)
        logits = self.fc(embedding)

        if return_features:
            selected_features = {
                name: tensor
                for name, tensor in features.items()
                if name in self.feature_layers
            }
            return logits, selected_features

        return logits

    def get_feature_maps(
        self,
        x: Tensor,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Extract intermediate feature maps for XAI.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).
        layer_names : Optional[List[str]], default=None
            Optional subset of feature layer names to return. If None,
            the deepest spatial layer ("layer4") is returned.

        Returns
        -------
        Dict[str, Tensor]
            Mapping from layer name to activation tensor.

        Raises
        ------
        ValueError
            If any requested layer name is not available.
        """
        _, features = self._extract_backbone_features(x)

        available = set(features.keys())

        if layer_names is None:
            return {"layer4": features["layer4"]}

        invalid = [name for name in layer_names if name not in available]
        if invalid:
            raise ValueError(
                f"Invalid layer name(s): {invalid}. "
                f"Available layers: {sorted(available)}"
            )

        return {name: features[name] for name in layer_names}

    def get_classifier(self) -> nn.Module:
        """
        Return the final classification head.

        Returns
        -------
        nn.Module
            The classifier head mapping embeddings to logits.
        """
        return self.fc

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Extract penultimate-layer embeddings before the classification head.

        Useful for:
        - t-SNE and UMAP visualization
        - Nearest-neighbor analysis
        - Concept Activation Vectors (TCAV)
        - Representation analysis and clustering

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Embeddings of shape (B, embedding_dim).
        """
        embedding, _ = self._extract_backbone_features(x)
        return embedding

    # ----------------------------------------------------------------------
    # Extra helpers used in tests and workflows
    # ----------------------------------------------------------------------

    def freeze_backbone_except_bn(self) -> None:
        """
        Freeze backbone convolutional weights while keeping BatchNorm trainable.

        This is a common fine-tuning strategy for transfer learning:
        - All convolutional weights in the backbone are frozen.
        - BatchNorm parameters remain trainable.
        - The classifier head (self.fc) remains trainable.
        """
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze BatchNorm parameters in backbone
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

        # Ensure classifier head remains trainable
        for param in self.fc.parameters():
            param.requires_grad = True

    def get_layer_output_shapes(
        self,
        input_resolution: Tuple[int, int],
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Compute output shapes (C, H, W) for feature layers given an input size.

        This runs a lightweight forward pass with a dummy tensor (no gradients)
        to infer the shapes. It is primarily intended for debugging,
        configuration validation, and documentation.

        Parameters
        ----------
        input_resolution : Tuple[int, int]
            Spatial resolution as (height, width).

        Returns
        -------
        Dict[str, Tuple[int, int, int]]
            Mapping from layer name to (channels, height, width).
        """
        height, width = input_resolution
        # Get device from one of the model's parameters
        device = next(self.parameters()).device

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.in_channels,
                int(height),
                int(width),
                device=device,
            )
            _, features = self._extract_backbone_features(dummy)

        shapes: Dict[str, Tuple[int, int, int]] = {}
        for name in self.feature_layers:
            feat = features[name]
            _, c, h, w = feat.shape
            shapes[name] = (int(c), int(h), int(w))

        return shapes


__all__ = ["ResNet50Classifier"]
