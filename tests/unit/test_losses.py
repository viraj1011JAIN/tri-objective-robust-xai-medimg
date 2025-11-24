"""Basic and TRADES-style loss tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TestBasicLosses:
    def test_cross_entropy_non_negative(self, device: torch.device) -> None:
        logits = torch.randn(8, 5, device=device)
        targets = torch.randint(0, 5, (8,), device=device)

        loss = F.cross_entropy(logits, targets)

        assert loss.item() >= 0.0
        assert torch.isfinite(loss)


class TestTradesStyleLoss:
    def test_kl_div_non_negative(self, device: torch.device) -> None:
        """KL divergence should be >= ~0 up to tiny numerical noise."""
        p_logits = torch.randn(4, 10, device=device)
        q_logits = torch.randn(4, 10, device=device)

        p_log = F.log_softmax(p_logits, dim=1)
        q = F.softmax(q_logits, dim=1)

        kl = F.kl_div(p_log, q, reduction="batchmean")

        # Allow a tiny negative due to FP rounding
        assert kl.item() > -1e-6
