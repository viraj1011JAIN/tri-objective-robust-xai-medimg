"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" ICCV 2017.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for generating visual explanations."""

    def __init__(self, model: nn.Module, target_layer: str, device: torch.device):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Layer name to visualize (e.g., 'layer4')
            device: Computation device
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

        # Storage
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Get target layer
        target_layer_module = dict(self.model.named_modules())[self.target_layer]

        # Attach hooks
        self.forward_handle = target_layer_module.register_forward_hook(forward_hook)
        self.backward_handle = target_layer_module.register_full_backward_hook(
            backward_hook
        )

    def generate_heatmap(
        self,
        images: torch.Tensor,
        target_class: Optional[torch.Tensor] = None,
        use_relu: bool = True,
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmaps.

        Args:
            images: (B, C, H, W) input images
            target_class: (B,) target class indices. If None, use predicted class
            use_relu: Apply ReLU to heatmap (positive attributions only)

        Returns:
            heatmaps: (B, H, W) heatmaps in range [0, 1]
        """

        self.model.eval()
        self.model.zero_grad()

        images = images.to(self.device)
        images.requires_grad_(True)

        # Forward pass
        logits = self.model(images)

        # Use predicted class if not specified
        if target_class is None:
            target_class = logits.argmax(dim=1)
        else:
            target_class = target_class.to(self.device)

        # Get target logits
        batch_size = images.size(0)
        target_logits = logits[range(batch_size), target_class]

        # Backward pass
        target_logits.sum().backward()

        # Get gradients and activations
        gradients = self.gradients  # (B, C, H', W')
        activations = self.activations  # (B, C, H', W')

        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        # Weighted combination
        weighted_activations = (weights * activations).sum(dim=1)  # (B, H', W')

        # ReLU (only positive contributions)
        if use_relu:
            weighted_activations = F.relu(weighted_activations)

        # Normalize to [0, 1]
        heatmaps = []
        for i in range(batch_size):
            heatmap = weighted_activations[i]

            # Min-max normalization
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()

            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap = torch.zeros_like(heatmap)

            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps)  # (B, H', W')

        # Upsize to input resolution
        heatmaps = F.interpolate(
            heatmaps.unsqueeze(1),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # (B, H, W)

        return heatmaps

    def cleanup(self):
        """Remove hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
