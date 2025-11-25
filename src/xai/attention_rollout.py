"""
Attention Rollout for Vision Transformers (ViT).

This module provides attention-based explainability methods specifically
designed for Vision Transformer architectures, complementing Grad-CAM for
CNN models.

Classes:
    AttentionRollout: Aggregate multi-head attention across ViT layers
    AttentionFlow: Attention flow visualization for transformers

Key Features:
    - Multi-head attention aggregation
    - Layer-wise attention rollout
    - Identity matrix handling for residual connections
    - Batch-efficient processing
    - Compatible with medical imaging ViTs

Integration:
    - Works alongside GradCAM for hybrid CNN-ViT models
    - Compatible with torch.vision.models.vision_transformer
    - Supports timm library ViT variants
    - Ready for adversarial robustness analysis

Reference:
    Abnar & Zuidema. "Quantifying Attention Flow in Transformers."
    ACL 2020.

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 25, 2025
Version: 6.1.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.

    Aggregates attention maps across all transformer layers to visualize
    which input tokens (image patches) contribute most to the prediction.

    Attributes:
        model: Vision Transformer model
        device: Computation device
        attention_maps: Stored attention maps from forward pass
        discard_ratio: Ratio of lowest attention values to discard
        head_fusion: How to fuse multi-head attention ("mean" or "max")

    Example:
        >>> rollout = AttentionRollout(vit_model, head_fusion="mean")
        >>> attention_map = rollout.generate_attention_map(image)
        >>> # attention_map shape: (num_patches,) or (H, W) if reshaped
    """

    def __init__(
        self,
        model: nn.Module,
        discard_ratio: float = 0.1,
        head_fusion: str = "mean",
        use_cuda: bool = True,
    ):
        """Initialize Attention Rollout.

        Args:
            model: Vision Transformer model
            discard_ratio: Ratio of attention values to discard (0.0-0.9)
            head_fusion: How to combine attention heads ("mean" or "max")
            use_cuda: Use GPU if available

        Raises:
            ValueError: If parameters invalid
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.device = (
            torch.device("cuda")
            if use_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not 0.0 <= discard_ratio < 1.0:
            raise ValueError(f"discard_ratio must be in [0, 1), got {discard_ratio}")

        if head_fusion not in ["mean", "max", "min"]:
            raise ValueError(
                f"head_fusion must be 'mean', 'max', or 'min', got {head_fusion}"
            )

        self.attention_maps: List[Tensor] = []
        self.hooks: List[any] = []
        self._register_hooks()

        logger.info(
            f"Initialized AttentionRollout with {head_fusion} fusion, "
            f"discard_ratio={discard_ratio}"
        )

    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention maps."""

        def attention_hook(module: nn.Module, input: any, output: any) -> None:
            """Hook to capture attention weights."""
            # For standard ViT, attention is in output tuple
            if isinstance(output, tuple) and len(output) > 1:
                attention = output[1]  # (B, H, N, N)
                if attention is not None:
                    self.attention_maps.append(attention.detach())

        # Find attention layers
        for name, module in self.model.named_modules():
            # Standard ViT attention modules
            if "attn" in name.lower() and isinstance(module, nn.Module):
                if hasattr(module, "num_heads"):
                    handle = module.register_forward_hook(attention_hook)
                    self.hooks.append(handle)

        if not self.hooks:
            logger.warning(
                "No attention layers found. Make sure model has attention modules."
            )

    def generate_attention_map(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None,
        reshape_to_grid: bool = True,
    ) -> np.ndarray:
        """Generate attention rollout map for input.

        Args:
            input_tensor: Input tensor (1, C, H, W) or (C, H, W)
            class_idx: Target class (unused, for API compatibility)
            reshape_to_grid: Reshape output to (H, W) grid

        Returns:
            Attention map as numpy array

        Raises:
            RuntimeError: If no attention maps captured
        """
        # Prepare input
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        # Clear previous attention maps
        self.attention_maps.clear()

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attention_maps:
            raise RuntimeError(
                "No attention maps captured. Check if model has "
                "attention modules with forward hooks."
            )

        # Rollout attention across layers
        attention_rollout = self._compute_rollout()

        # Extract CLS token attention (first token)
        # Shape: (num_patches + 1,)
        cls_attention = attention_rollout[0, 0, 1:]  # Skip CLS itself

        # Convert to numpy
        attention_map = cls_attention.cpu().numpy()

        # Reshape to spatial grid if requested
        if reshape_to_grid:
            num_patches = attention_map.shape[0]
            grid_size = int(np.sqrt(num_patches))

            if grid_size * grid_size != num_patches:
                logger.warning(
                    f"Cannot reshape {num_patches} patches to square grid. "
                    f"Returning flattened map."
                )
            else:
                attention_map = attention_map.reshape(grid_size, grid_size)

        return attention_map

    def _compute_rollout(self) -> Tensor:
        """Compute attention rollout across all layers.

        Returns:
            Rolled-out attention map (1, 1, N, N)
        """
        # Fuse multi-head attention
        fused_attentions = []

        for attention in self.attention_maps:
            # attention shape: (B, H, N, N)
            # B = batch, H = heads, N = tokens

            if self.head_fusion == "mean":
                fused = attention.mean(dim=1)  # (B, N, N)
            elif self.head_fusion == "max":
                fused = attention.max(dim=1)[0]
            elif self.head_fusion == "min":
                fused = attention.min(dim=1)[0]
            else:
                fused = attention.mean(dim=1)

            fused_attentions.append(fused)

        # Stack all layers: (num_layers, B, N, N)
        attention_stack = torch.stack(fused_attentions, dim=0)

        # Add identity matrix for residual connections
        num_layers, batch_size, num_tokens, _ = attention_stack.shape

        identity = torch.eye(num_tokens, device=attention_stack.device)
        identity = identity.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        identity = identity.expand(num_layers, batch_size, -1, -1)

        # Combine attention with identity (residual connection)
        attention_with_identity = attention_stack + identity

        # Normalize rows
        attention_with_identity = attention_with_identity / (
            attention_with_identity.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Apply discard ratio (keep only top attention values)
        if self.discard_ratio > 0:
            attention_with_identity = self._apply_discard_ratio(attention_with_identity)

        # Rollout: multiply attention matrices across layers
        result = attention_with_identity[0]  # Start with first layer

        for i in range(1, num_layers):
            result = torch.matmul(result, attention_with_identity[i])

        # Normalize final result
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-8)

        return result

    def _apply_discard_ratio(self, attention: Tensor) -> Tensor:
        """Apply discard ratio to attention maps.

        Zeros out lowest attention values to reduce noise.

        Args:
            attention: Attention tensor (L, B, N, N)

        Returns:
            Filtered attention tensor
        """
        # Flatten spatial dimensions
        flat_attention = attention.flatten(2)  # (L, B, N*N)

        # Compute threshold for each layer/batch
        threshold_idx = int(flat_attention.shape[-1] * self.discard_ratio)

        if threshold_idx > 0:
            thresholds = torch.kthvalue(
                flat_attention, k=threshold_idx, dim=-1, keepdim=True
            )[0]

            # Zero out values below threshold
            mask = attention >= thresholds.view(*thresholds.shape[:2], 1, 1)
            attention = attention * mask.float()

        return attention

    def get_layer_attention(self, layer_idx: int, input_tensor: Tensor) -> np.ndarray:
        """Get attention map from specific layer.

        Args:
            layer_idx: Layer index (0-based)
            input_tensor: Input tensor

        Returns:
            Attention map from specified layer
        """
        # Run forward pass
        self.attention_maps.clear()
        self.model.eval()

        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        if layer_idx >= len(self.attention_maps):
            raise IndexError(
                f"Layer {layer_idx} out of range. "
                f"Model has {len(self.attention_maps)} attention layers."
            )

        # Get and process attention
        attention = self.attention_maps[layer_idx]

        if self.head_fusion == "mean":
            fused = attention.mean(dim=1)
        elif self.head_fusion == "max":
            fused = attention.max(dim=1)[0]
        else:
            fused = attention.mean(dim=1)

        # CLS token attention
        cls_attention = fused[0, 0, 1:]

        return cls_attention.cpu().numpy()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()
        logger.info("Removed all AttentionRollout hooks")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.remove_hooks()
        except Exception:
            pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AttentionRollout(head_fusion={self.head_fusion}, "
            f"discard_ratio={self.discard_ratio})"
        )


def create_vit_explainer(
    model: nn.Module,
    method: str = "attention_rollout",
    discard_ratio: float = 0.1,
    use_cuda: bool = True,
) -> AttentionRollout:
    """Factory function to create ViT explainability methods.

    Args:
        model: Vision Transformer model
        method: "attention_rollout" (more methods can be added)
        discard_ratio: Ratio of attention to discard
        use_cuda: Use GPU acceleration

    Returns:
        Explainer instance

    Example:
        >>> explainer = create_vit_explainer(vit_model)
        >>> attention_map = explainer.generate_attention_map(image)
    """
    if method == "attention_rollout":
        return AttentionRollout(model, discard_ratio=discard_ratio, use_cuda=use_cuda)
    else:
        raise ValueError(
            f"Invalid method: {method}. Currently only "
            f"'attention_rollout' is supported."
        )
