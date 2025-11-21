"""
EfficientNet-B0 classifier for medical image analysis.

This module provides an EfficientNet-B0-based classifier that integrates with
the Tri-Objective Robust XAI pipeline via the :class:`BaseModel` interface.

Key features
------------
- Uses torchvision's EfficientNet-B0 backbone (~5.3M parameters)
- Supports arbitrary input channels (e.g., 1-channel chest X-rays)
- Exposes intermediate feature maps for XAI methods (Grad-CAM, etc.)
- Provides a clean embedding interface for TCAV and representation analysis
- Compatible with BaseModel.freeze_backbone() and metadata utilities
- Compound scaling: depth=1.0, width=1.0, resolution=224

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base_model import BaseModel

logger = logging.getLogger(__name__)

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False
    EfficientNet_B0_Weights = None  # type: ignore[assignment,misc]
    efficientnet_b0 = None  # type: ignore[assignment,misc]


class EfficientNetB0Classifier(BaseModel):
    """
    EfficientNet-B0 classifier for medical image classification.

    This class wraps torchvision's EfficientNet-B0 backbone and adapts it to
    the :class:`BaseModel` interface used in the Tri-Objective Robust XAI
    pipeline.

    Architecture details
    --------------------
    - Backbone: EfficientNet-B0 (5.3M parameters)
    - Feature dimension: 1280-D before classification head
    - Extractable layers: multiple MBConv blocks + final features
    - Input resolution: flexible (commonly 224×224 or 512×512)
    - Compound scaling: baseline coefficients (depth=1.0, width=1.0)

    Parameters
    ----------
    num_classes : int
        Number of output classes for the classifier head. Must be > 0.
    pretrained : bool, default=True
        If True, load ImageNet-1K pretrained weights for the backbone.
    in_channels : int, default=3
        Number of input channels. Common values:
            - 1: Grayscale (chest X-rays, mammograms)
            - 3: RGB (dermoscopy, histopathology)
        If not 3 and pretrained=True, adapts first conv layer intelligently.
    dropout : float, default=0.2
        Dropout probability before final linear layer. If 0, no dropout used.
    global_pool : str, default="avg"
        Type of global pooling. Options: "avg", "max".
    **kwargs : Any
        Additional configuration passed to BaseModel for experiment tracking.

    Attributes
    ----------
    backbone : nn.Module
        EfficientNet-B0 feature extractor (features module).
    fc : nn.Module
        Final classification head (may include dropout).
    embedding_dim : int
        Dimensionality of penultimate features (1280 for EfficientNet-B0).
    feature_layers : List[str]
        Available feature layers for extraction.

    Raises
    ------
    RuntimeError
        If torchvision is not installed.
    ValueError
        If in_channels <= 0 or dropout not in [0, 1).

    Examples
    --------
    >>> model = EfficientNetB0Classifier(num_classes=7, pretrained=True)
    >>> x = torch.randn(4, 3, 224, 224)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 7])

    >>> logits, features = model(x, return_features=True)
    >>> features["features"].shape
    torch.Size([4, 1280, 7, 7])
    """

    # Expected spatial sizes at 224×224 input (for documentation)
    FEATURE_SPATIAL_SIZES: Dict[str, Tuple[int, int]] = {
        "block3": (28, 28),  # After ~8× downsampling
        "block5": (14, 14),  # After ~16× downsampling
        "block7": (7, 7),  # After ~32× downsampling (deepest block)
        "features": (7, 7),  # Same as block7
    }

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.2,
        global_pool: str = "avg",
        **kwargs: Any,
    ) -> None:
        """Initialize EfficientNet-B0 classifier with validation and setup."""
        super().__init__(num_classes=num_classes, pretrained=pretrained, **kwargs)

        if not _HAS_TORCHVISION:
            raise RuntimeError(
                "torchvision is required for EfficientNetB0Classifier. "
                "Install with: pip install torchvision"
            )

        # Validate parameters
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if global_pool not in {"avg", "max"}:
            raise ValueError(
                f"Unsupported global_pool: {global_pool}. Use 'avg' or 'max'."
            )

        self.in_channels: int = int(in_channels)
        self.dropout_prob: float = float(dropout)
        self.global_pool_type: str = str(global_pool)
        self.embedding_dim: int = 1280  # Fixed for EfficientNet-B0

        # ------------------------------------------------------------------
        # Backbone setup
        # ------------------------------------------------------------------
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        if efficientnet_b0 is None:
            raise RuntimeError(
                "efficientnet_b0 constructor not available. "
                "Please upgrade torchvision."
            )

        backbone = efficientnet_b0(weights=weights)

        # Remove original classifier (we'll add our own)
        backbone.classifier = nn.Identity()
        self.backbone: nn.Module = backbone

        # Adapt first conv for non-RGB inputs
        if self.in_channels != 3:
            self._adapt_first_conv(in_channels=self.in_channels, pretrained=pretrained)

        # Define extractable feature layers
        # EfficientNet-B0 has multiple MBConv blocks; we expose key intermediate ones
        self.feature_layers: List[str] = ["block3", "block5", "block7", "features"]

        # ------------------------------------------------------------------
        # Global pooling
        # ------------------------------------------------------------------
        if global_pool == "avg":
            self.global_pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        else:  # "max"
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        head_modules: List[nn.Module] = []
        if self.dropout_prob > 0.0:
            head_modules.append(nn.Dropout(p=self.dropout_prob, inplace=True))
        head_modules.append(nn.Linear(self.embedding_dim, self.num_classes))

        self.fc: nn.Module = (
            nn.Sequential(*head_modules) if len(head_modules) > 1 else head_modules[0]
        )

        logger.info(
            "Initialized EfficientNet-B0: %s params, in_channels=%d, dropout=%.3f",
            f"{self.num_parameters():,}",
            self.in_channels,
            self.dropout_prob,
        )

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _adapt_first_conv(self, in_channels: int, pretrained: bool) -> None:
        """
        Adapt first convolution layer for arbitrary input channels.

        Strategy for pretrained weights:
        - in_channels=1: average RGB weights → single channel
        - in_channels=2: take first 2 RGB channels
        - in_channels>3: repeat RGB channels with normalization

        Parameters
        ----------
        in_channels : int
            Target number of input channels.
        pretrained : bool
            Whether pretrained weights are being used.

        Raises
        ------
        RuntimeError
            If first Conv2d layer cannot be located or replaced.
        """
        # Locate first conv in the features sequential module
        first_conv: Optional[nn.Conv2d] = None
        for module in self.backbone.features.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break

        if first_conv is None:
            raise RuntimeError(
                "Could not locate first Conv2d in EfficientNet features."
            )

        # Create new conv with adapted input channels
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None),
        )

        # Adapt pretrained weights if available
        if pretrained and isinstance(first_conv.weight, Tensor):
            with torch.no_grad():
                old_weight = first_conv.weight  # (out_channels, 3, kH, kW)

                if in_channels == 1:
                    # Average RGB → grayscale
                    new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
                elif in_channels == 2:
                    # Take first two RGB channels
                    new_conv.weight.copy_(old_weight[:, :2, :, :])
                elif in_channels > 3:
                    # Repeat RGB channels with normalization
                    repeat_factor = (in_channels + 2) // 3
                    expanded = old_weight.repeat(1, repeat_factor, 1, 1)[
                        :, :in_channels, :, :
                    ]
                    new_conv.weight.copy_(expanded / float(repeat_factor))
                else:
                    # in_channels == 3, direct copy (should not normally reach here)
                    new_conv.weight.copy_(old_weight)

                # Copy bias if present
                if first_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
        # If not pretrained, we keep the default random initialization

        # Replace conv in module hierarchy
        replaced = False
        for module in self.backbone.features.modules():
            for child_name, child in module.named_children():
                if child is first_conv:
                    setattr(module, child_name, new_conv)
                    replaced = True
                    break
            if replaced:
                break

        if not replaced:
            raise RuntimeError(
                "Failed to replace first Conv2d when adapting in_channels."
            )

        logger.info("Adapted EfficientNet first conv to %d input channels", in_channels)

    def _extract_features(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Extract features from multiple EfficientNet blocks.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, H, W).

        Returns
        -------
        Tuple[Tensor, Dict[str, Tensor]]
            - Embeddings of shape (B, 1280) after global pooling.
            - Dictionary of intermediate feature maps.

        Notes
        -----
        EfficientNet-B0 block structure (high-level):
        - Early blocks: resolution /4
        - Mid blocks: resolution /8  → "block3"
        - Deeper blocks: resolution /16 → "block5"
        - Deepest blocks: resolution /32 → "block7" and "features"
        """
        features_dict: Dict[str, Tensor] = {}

        x_curr = x
        for idx, block in enumerate(self.backbone.features):
            x_curr = block(x_curr)

            # Capture specific blocks for XAI (indices based on torchvision impl)
            if idx == 3:
                features_dict["block3"] = x_curr
            elif idx == 5:
                features_dict["block5"] = x_curr
            elif idx == 7:
                features_dict["block7"] = x_curr

        # Final features (after all blocks)
        features_dict["features"] = x_curr

        # Global pooling → embedding
        pooled = self.global_pool(x_curr)
        embedding = pooled.flatten(1)

        return embedding, features_dict

    # ----------------------------------------------------------------------
    # BaseModel interface implementations
    # ----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass through EfficientNet-B0 classifier.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, in_channels, H, W).
        return_features : bool, default=False
            If True, also return intermediate feature maps for XAI.

        Returns
        -------
        Tensor or Tuple[Tensor, Dict[str, Tensor]]
            If return_features is False:
                - Logits of shape (B, num_classes).
            If return_features is True:
                - Tuple of (logits, features_dict) where features_dict maps
                  layer names to activation tensors.

        Raises
        ------
        ValueError
            If input tensor does not have 4 dimensions.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )

        embedding, features = self._extract_features(x)
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
        Extract intermediate feature maps for XAI methods.

        This is the primary interface for explainability techniques:
        - Grad-CAM: typically uses "features" (deepest spatial layer).
        - Multi-scale CAM: may use multiple blocks.
        - TCAV: can use any layer for concept activation.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, in_channels, H, W).
        layer_names : Optional[List[str]], default=None
            Specific layers to extract. If None, returns "features" only.
            Valid options: ["block3", "block5", "block7", "features"].

        Returns
        -------
        Dict[str, Tensor]
            Mapping from layer name to feature tensor.

        Raises
        ------
        ValueError
            If any requested layer name is invalid.
        """
        _, features = self._extract_features(x)

        if layer_names is None:
            return {"features": features["features"]}

        available = set(features.keys())
        invalid = [name for name in layer_names if name not in available]
        if invalid:
            raise ValueError(
                f"Invalid layer name(s): {invalid}. " f"Available: {sorted(available)}"
            )

        return {name: features[name] for name in layer_names}

    def get_classifier(self) -> nn.Module:
        """Return the final classification head."""
        return self.fc

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Extract penultimate-layer embeddings (1280-D for EfficientNet-B0).

        Parameters
        ----------
        x : Tensor
            Input of shape (B, in_channels, H, W).

        Returns
        -------
        Tensor
            Embeddings of shape (B, 1280).

        Raises
        ------
        ValueError
            If input does not have 4 dimensions.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )

        embedding, _ = self._extract_features(x)
        return embedding

    # ----------------------------------------------------------------------
    # Additional utilities
    # ----------------------------------------------------------------------

    def freeze_backbone_except_bn(self) -> None:
        """
        Freeze backbone except BatchNorm / GroupNorm layers.

        This is useful for:
        - Domain adaptation where target statistics differ.
        - Fine-tuning with small datasets.
        - Preserving learned features while adapting normalization statistics.
        """
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                for param in module.parameters():
                    param.requires_grad = False

        # Ensure classifier remains trainable
        for param in self.fc.parameters():
            param.requires_grad = True

    def get_layer_output_shapes(
        self, input_size: Tuple[int, int] = (224, 224)
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Get expected output shapes for each feature layer.

        Useful for:
        - Debugging shape mismatches.
        - Planning Grad-CAM upsampling.
        - Understanding receptive fields.

        Parameters
        ----------
        input_size : Tuple[int, int], default=(224, 224)
            Expected input spatial dimensions (H, W).

        Returns
        -------
        Dict[str, Tuple[int, int, int]]
            Maps layer names to (C, H, W) shapes.
        """
        h, w = input_size
        return {
            "block3": (40, h // 8, w // 8),  # MBConv after block 3
            "block5": (112, h // 16, w // 16),  # MBConv after block 5
            "block7": (320, h // 32, w // 32),  # MBConv after block 7
            "features": (1280, h // 32, w // 32),  # Final conv features
        }


__all__ = ["EfficientNetB0Classifier"]
