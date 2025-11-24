"""
ViT-B/16 classifier for medical image analysis.

This module provides a Vision Transformer (ViT-B/16)-based classifier that
integrates with the Tri-Objective Robust XAI pipeline via the
:class:`BaseModel` interface.

Key features
------------
- Uses torchvision's official ViT-B/16 backbone (~86M parameters)
- Supports arbitrary input channels (e.g., 1-channel chest X-rays)
- Exposes token-level features for transformer-based XAI
- Implements attention rollout for explanation maps
- Provides a clean embedding interface for TCAV and representation analysis
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
    # Newer torchvision API with weights enums
    from torchvision.models import (  # type: ignore[attr-defined]
        ViT_B_16_Weights,
        vit_b_16,
    )

    _HAS_TORCHVISION = True
    _HAS_VIT_WEIGHTS = True
except ImportError:  # pragma: no cover
    try:
        # Older API without weights enums
        from torchvision.models import vit_b_16  # type: ignore[assignment]

        ViT_B_16_Weights = None  # type: ignore[assignment]
        _HAS_TORCHVISION = True
        _HAS_VIT_WEIGHTS = False
    except ImportError:  # pragma: no cover
        vit_b_16 = None  # type: ignore[assignment]
        ViT_B_16_Weights = None  # type: ignore[assignment]
        _HAS_TORCHVISION = False
        _HAS_VIT_WEIGHTS = False


class ViTB16Classifier(BaseModel):
    """
    Vision Transformer (ViT-B/16) classifier for medical image analysis.

    This class wraps torchvision's ViT-B/16 implementation and adapts it to the
    :class:`BaseModel` interface used in the Tri-Objective Robust XAI pipeline.

    Key properties
    --------------
    - Backbone: ViT-B/16 (patch size 16×16, 12 layers, 12 heads)
    - Embedding dimension: 768
    - Parameters: ~86M (with ImageNet-1K head)
    - Input resolution: 224×224 by default
    - Tokens: 1 class token + 14×14 patch tokens

    The class provides:
    - Configurable classifier head (num_classes, dropout)
    - Support for arbitrary input channels (e.g., 1-channel X-rays)
    - Feature extraction interfaces for XAI
    - Attention rollout for transformer-based explanations

    Parameters
    ----------
    num_classes:
        Number of output classes. Must be > 0.
    pretrained:
        If True, load ImageNet-1K pretrained weights.
    in_channels:
        Number of input channels (1 for grayscale, 3 for RGB).
    dropout:
        Dropout probability before the final classification head.
    global_pool:
        How to derive the embedding from tokens:
        - "cls": use the [CLS] token (default)
        - "mean": mean over patch tokens
    image_size:
        Input image size (height and width). Defaults to 224.
    **kwargs:
        Passed through to :class:`BaseModel` for experiment metadata.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        global_pool: str = "cls",
        image_size: int = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained, **kwargs)

        if not _HAS_TORCHVISION or vit_b_16 is None:
            raise RuntimeError(
                "torchvision with ViT support is required for ViTB16Classifier. "
                "Install with: pip install torchvision"
            )

        if num_classes <= 0:  # pragma: no cover
            # Defensive check; BaseModel already validates this
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if global_pool not in {"cls", "mean"}:
            raise ValueError(
                f"Unsupported global_pool: {global_pool}. Use 'cls' or 'mean'."
            )

        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")

        self.in_channels: int = int(in_channels)
        self.dropout_prob: float = float(dropout)
        self.global_pool_type: str = str(global_pool)
        self.image_size: int = int(image_size)

        # ------------------------------------------------------------------
        # Backbone construction
        # ------------------------------------------------------------------
        if _HAS_VIT_WEIGHTS and ViT_B_16_Weights is not None:
            if pretrained:
                weights = ViT_B_16_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
            else:
                weights = None
            backbone = vit_b_16(weights=weights, image_size=self.image_size)
        else:
            # Older torchvision API
            backbone = vit_b_16(pretrained=pretrained)  # type: ignore[call-arg]
            # image_size is fixed to 224 in this case

        self.backbone = backbone
        self.embedding_dim: int = int(getattr(self.backbone, "hidden_dim", 768))

        # Adapt conv stem for non-RGB inputs
        if self.in_channels != 3:
            self._adapt_conv_proj(in_channels=self.in_channels, pretrained=pretrained)

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

        # Replace backbone heads with identity; we own the classifier head
        if hasattr(self.backbone, "heads"):  # pragma: no branch
            self.backbone.heads = nn.Identity()

        # Names exposed via get_feature_maps
        self.feature_layers: List[str] = [
            "cls_token",
            "patch_tokens",
            "patch_map",
            "tokens",
            "attention_rollout",
        ]

        logger.info(
            "Initialized ViT-B/16: %s params, in_channels=%d, dropout=%.3f, pool=%s",
            f"{self.num_parameters():,}",
            self.in_channels,
            self.dropout_prob,
            self.global_pool_type,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt_conv_proj(self, in_channels: int, pretrained: bool) -> None:
        """
        Adapt ViT conv projection for arbitrary input channels.

        The default ViT-B/16 expects 3-channel RGB inputs. This method adapts the
        patch embedding stem to support grayscale (1-channel) or other channel
        configurations while preserving pretrained information where possible.
        """
        conv = getattr(self.backbone, "conv_proj", None)
        if not isinstance(conv, nn.Conv2d):
            raise RuntimeError(
                "Expected backbone.conv_proj to be nn.Conv2d for ViT-B/16, "
                f"got {type(conv).__name__!s}"
            )

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None),
        )

        if pretrained and isinstance(conv.weight, Tensor):
            with torch.no_grad():
                old_weight = conv.weight  # (out_channels, 3, kH, kW)
                if in_channels == 1:
                    # Average RGB channels → grayscale
                    new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
                elif in_channels == 2:
                    # Use first two channels
                    new_conv.weight.copy_(old_weight[:, :2, :, :])
                elif in_channels > 3:
                    # Repeat and normalize RGB channels
                    repeat_factor = (in_channels + 2) // 3
                    expanded = old_weight.repeat(1, repeat_factor, 1, 1)[
                        :, :in_channels, :, :
                    ]
                    new_conv.weight.copy_(expanded / float(repeat_factor))
                else:  # pragma: no cover
                    # in_channels == 3: direct copy (for completeness)
                    # Note: This is unreachable since _adapt_conv_proj is only
                    # called when in_channels != 3 (see __init__ line 154-155)
                    new_conv.weight.copy_(old_weight)

                if conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)

        # Replace conv_proj
        self.backbone.conv_proj = new_conv
        logger.info(
            "Adapted ViT conv_proj from 3 to %d input channels (pretrained=%s)",
            in_channels,
            pretrained,
        )

    def _encode_with_attn(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Run ViT encoder while capturing attention matrices.

        This re-implements the encoder forward using the same modules as
        torchvision's implementation, but forces ``need_weights=True`` and
        ``average_attn_weights=False`` to obtain per-head attention maps.

        Returns
        -------
        encoded_tokens:
            Tensor of shape (B, 1 + N_patches, D).
        attn_weights_list:
            List of attention tensors, one per layer, each of shape
            (B, num_heads, 1 + N_patches, 1 + N_patches).
        """
        # Shape checks are delegated to _process_input
        tokens = self.backbone._process_input(x)  # (B, N_patches, D)
        batch_size = tokens.shape[0]

        # Prepend class token
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x_tokens = torch.cat([cls_token, tokens], dim=1)  # (B, 1 + N, D)

        # Positional embeddings and dropout
        encoder = self.backbone.encoder
        x_tokens = x_tokens + encoder.pos_embedding
        x_tokens = encoder.dropout(x_tokens)

        attn_weights_list: List[Tensor] = []
        for block in encoder.layers:  # type: ignore[attr-defined]
            # This mirrors EncoderBlock.forward, but with attention weights
            input_tokens = x_tokens
            x_norm = block.ln_1(input_tokens)
            attn_output, attn_weights = block.self_attention(  # type: ignore[call-arg]
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=True,
                average_attn_weights=False,
            )
            x_attn = block.dropout(attn_output)
            x_tokens = x_attn + input_tokens

            y = block.ln_2(x_tokens)
            y = block.mlp(y)
            x_tokens = x_tokens + y

            attn_weights_list.append(attn_weights)

        encoded_tokens = encoder.ln(x_tokens)
        return encoded_tokens, attn_weights_list

    @staticmethod
    def _compute_attention_rollout(attn_weights_list: List[Tensor]) -> Tensor:
        """
        Compute attention rollout map from per-layer attention weights.

        Implements the standard attention rollout algorithm:

        - Average over heads
        - Add identity to each layer's attention
        - Row-normalize
        - Multiply attentions from first to last layer

        Returns
        -------
        Tensor
            Rollout maps of shape (B, S, S) where S is the number of tokens.
        """
        if not attn_weights_list:
            raise ValueError("attn_weights_list must not be empty.")

        # All layers share the same sequence length
        first = attn_weights_list[0]
        if first.dim() != 4:
            raise ValueError(
                f"Expected attention weights of shape (B, H, S, S), got {tuple(first.shape)}"
            )

        batch_size = first.shape[0]
        seq_len = first.shape[-1]
        device = first.device
        dtype = first.dtype

        rollout = torch.eye(seq_len, device=device, dtype=dtype).unsqueeze(0)
        rollout = rollout.expand(batch_size, seq_len, seq_len).clone()

        for attn in attn_weights_list:
            # (B, num_heads, S, S) -> (B, S, S)
            attn_mean = attn.mean(dim=1)

            # Add identity and normalize rows
            attn_aug = attn_mean + torch.eye(seq_len, device=device, dtype=dtype)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)

            rollout = torch.bmm(attn_aug, rollout)

        return rollout  # (B, S, S)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass through ViT-B/16 classifier.

        Parameters
        ----------
        x:
            Input tensor of shape (B, in_channels, H, W).
        return_features:
            If True, also return feature maps and attention rollout.

        Returns
        -------
        logits or (logits, features)
            - logits: (B, num_classes)
            - features (if requested): dict with keys among
              ``['cls_token', 'patch_tokens', 'patch_map', 'tokens', 'attention_rollout']``.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )

        encoded_tokens, attn_weights_list = self._encode_with_attn(x)

        cls_token = encoded_tokens[:, 0]  # (B, D)
        patch_tokens = encoded_tokens[:, 1:]  # (B, N, D)
        num_patches = patch_tokens.shape[1]

        # Derive embedding
        if self.global_pool_type == "cls":
            embedding = cls_token
        else:
            # Mean over patch tokens
            embedding = patch_tokens.mean(dim=1)

        logits = self.fc(embedding)

        if not return_features:
            return logits

        # Construct feature dict
        features: Dict[str, Tensor] = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "tokens": encoded_tokens,
        }

        # Spatial map from patch tokens if grid is square (e.g., 14×14)
        grid_size_float = num_patches**0.5
        grid_size = int(grid_size_float)
        if grid_size * grid_size == num_patches:
            patch_map = patch_tokens.transpose(1, 2).reshape(
                patch_tokens.size(0),
                patch_tokens.size(2),
                grid_size,
                grid_size,
            )
            features["patch_map"] = patch_map

            # Attention rollout → class-to-patch relevance
            rollout = self._compute_attention_rollout(attn_weights_list)
            # Take class token row and discard class token column
            cls_to_patches = rollout[:, 0, 1:]  # (B, N)
            attn_map = cls_to_patches.reshape(
                patch_tokens.size(0),
                1,
                grid_size,
                grid_size,
            )
            features["attention_rollout"] = attn_map

        return logits, features

    def get_feature_maps(
        self,
        x: Tensor,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Extract ViT feature representations and attention rollout.

        Parameters
        ----------
        x:
            Input tensor of shape (B, in_channels, H, W).
        layer_names:
            Optional subset of layers to return. If None, returns only
            ``{"tokens": encoded_tokens}``.

        Returns
        -------
        Dict[str, Tensor]
            Mapping from layer name to tensor.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )

        encoded_tokens, attn_weights_list = self._encode_with_attn(x)

        cls_token = encoded_tokens[:, 0]
        patch_tokens = encoded_tokens[:, 1:]
        num_patches = patch_tokens.size(1)

        features: Dict[str, Tensor] = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "tokens": encoded_tokens,
        }

        grid_size_float = num_patches**0.5
        grid_size = int(grid_size_float)
        if grid_size * grid_size == num_patches:
            patch_map = patch_tokens.transpose(1, 2).reshape(
                patch_tokens.size(0),
                patch_tokens.size(2),
                grid_size,
                grid_size,
            )
            features["patch_map"] = patch_map

            rollout = self._compute_attention_rollout(attn_weights_list)
            cls_to_patches = rollout[:, 0, 1:]
            attn_map = cls_to_patches.reshape(
                patch_tokens.size(0),
                1,
                grid_size,
                grid_size,
            )
            features["attention_rollout"] = attn_map

        if layer_names is None:
            # Default minimal subset to avoid surprises
            return {"tokens": features["tokens"]}

        invalid = [name for name in layer_names if name not in features]
        if invalid:
            raise ValueError(
                f"Invalid layer name(s): {invalid}. "
                f"Available: {sorted(features.keys())}"
            )

        return {name: features[name] for name in layer_names}

    def get_classifier(self) -> nn.Module:
        """Return the final classification head."""
        return self.fc

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Extract ViT embedding vector (CLS or mean-pooled patch tokens).

        Parameters
        ----------
        x:
            Input tensor of shape (B, in_channels, H, W).

        Returns
        -------
        Tensor
            Embeddings of shape (B, embedding_dim).
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )

        encoded_tokens, _ = self._encode_with_attn(x)
        cls_token = encoded_tokens[:, 0]
        patch_tokens = encoded_tokens[:, 1:]

        if self.global_pool_type == "cls":
            return cls_token
        return patch_tokens.mean(dim=1)

    def get_attention_rollout(self, x: Tensor) -> Tensor:
        """
        Convenience wrapper to compute attention rollout maps.

        Parameters
        ----------
        x:
            Input tensor of shape (B, in_channels, H, W).

        Returns
        -------
        Tensor
            Attention rollout maps of shape (B, 1, H_p, W_p) where
            ``H_p = W_p = image_size / patch_size`` (typically 14 for 224×224).
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}"
            )

        encoded_tokens, attn_weights_list = self._encode_with_attn(x)
        patch_tokens = encoded_tokens[:, 1:]
        num_patches = patch_tokens.size(1)

        grid_size_float = num_patches**0.5
        grid_size = int(grid_size_float)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(
                f"Number of patches ({num_patches}) is not a perfect square; "
                "cannot reshape attention rollout to spatial map."
            )

        rollout = self._compute_attention_rollout(attn_weights_list)
        cls_to_patches = rollout[:, 0, 1:]
        attn_map = cls_to_patches.reshape(
            patch_tokens.size(0),
            1,
            grid_size,
            grid_size,
        )
        return attn_map

    def get_layer_output_shapes(
        self,
        input_size: Tuple[int, int] = (224, 224),
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Get expected output shapes for ViT feature layers.

        Parameters
        ----------
        input_size:
            Input image size (H, W). Must be divisible by patch size.

        Returns
        -------
        Dict[str, Tuple[int, ...]]
            Mapping from layer name to shape (excluding batch dimension).
        """
        h, w = input_size
        patch_size = int(getattr(self.backbone, "patch_size", 16))

        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"Input size {input_size} not divisible by patch size {patch_size}."
            )

        grid_h = h // patch_size
        grid_w = w // patch_size
        num_patches = grid_h * grid_w

        return {
            "cls_token": (self.embedding_dim,),
            "patch_tokens": (num_patches, self.embedding_dim),
            "tokens": (1 + num_patches, self.embedding_dim),
            "patch_map": (self.embedding_dim, grid_h, grid_w),
            "attention_rollout": (1, grid_h, grid_w),
        }


__all__ = ["ViTB16Classifier"]
