"""
Tests for seed-based reproducibility and determinism.
"""

import numpy as np
import torch
import torch.nn as nn


class TestSeedDeterminism:
    def test_torch_random_seed(self):
        torch.manual_seed(42)
        a = torch.randn(3, 3)

        torch.manual_seed(42)
        b = torch.randn(3, 3)

        assert torch.allclose(a, b)

    def test_numpy_random_seed(self):
        np.random.seed(42)
        a = np.random.randn(3, 3)

        np.random.seed(42)
        b = np.random.randn(3, 3)

        assert np.allclose(a, b)

    def test_model_init_deterministic(self):
        torch.manual_seed(42)
        m1 = nn.Linear(10, 5)
        w1 = m1.weight.clone()

        torch.manual_seed(42)
        m2 = nn.Linear(10, 5)
        w2 = m2.weight.clone()

        assert torch.allclose(w1, w2)
