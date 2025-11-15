"""
Tests for simple FGSM- and PGD-style adversarial perturbations.
These are generic sanity checks, not tied to your attack implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _toy_model(device):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)


class TestFGSMSanity:
    def test_fgsm_perturbation_bounded(self, device):
        model = _toy_model(device)
        model.eval()

        x = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
        y = torch.randint(0, 10, (2,), device=device)
        eps = 8.0 / 255.0

        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        x_adv = torch.clamp(x + eps * x.grad.sign(), 0.0, 1.0)
        diff = (x_adv - x).abs()
        assert diff.max().item() <= eps + 1e-6
        assert (x_adv >= 0).all()
        assert (x_adv <= 1).all()


class TestPGDSanity:
    def test_pgd_stays_in_linf_ball(self, device):
        model = _toy_model(device)
        model.eval()

        x = torch.rand(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)
        eps = 8.0 / 255.0
        alpha = 2.0 / 255.0
        steps = 5

        x_adv = x.clone().detach()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            out = model(x_adv)
            loss = F.cross_entropy(out, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -eps, eps)
                x_adv = torch.clamp(x + delta, 0.0, 1.0)

        final_delta = (x_adv - x).abs()
        assert final_delta.max().item() <= eps + 1e-6
