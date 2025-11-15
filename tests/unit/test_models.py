"""Tests for a simple CNN used in CIFAR-style debug runs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDebugNet(nn.Module):
    """Tiny convnet just for unit tests."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TestSimpleDebugNet:
    def test_forward_shape(self, device: torch.device) -> None:
        model = SimpleDebugNet().to(device)
        x = torch.randn(4, 3, 32, 32, device=device)

        out = model(x)

        assert out.shape == (4, 10)
        assert torch.isfinite(out).all()

    def test_backward_pass(self, device: torch.device) -> None:
        model = SimpleDebugNet().to(device)
        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()
