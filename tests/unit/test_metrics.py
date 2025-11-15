"""Tests for basic classification and robustness metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class TestClassificationMetrics:
    def test_accuracy_matches_sklearn(self) -> None:
        y_true = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 1, 2, 1, 1])

        acc_manual = float((y_true == y_pred).mean())
        acc_sklearn = accuracy_score(y_true, y_pred)

        assert abs(acc_manual - acc_sklearn) < 1e-8

    def test_binary_f1_range(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        f1 = f1_score(y_true, y_pred)

        assert 0.0 <= f1 <= 1.0
