"""
Training Infrastructure for Tri-Objective Robust XAI
=====================================================

This package provides comprehensive training infrastructure for:
1. Baseline training (standard supervised learning)
2. Adversarial training (TRADES, MART, standard AT)
3. Tri-objective training (accuracy + robustness + explainability)

Phase 5.1 Components:
---------------------
- AdversarialTrainer: High-level adversarial training coordinator
- AdversarialTrainingConfig: Configuration dataclass
- train_adversarial_epoch: Standalone training function
- validate_robust: Robust accuracy evaluation

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.1.0
"""

from __future__ import annotations

# Phase 5.1: Adversarial training infrastructure
from .adversarial_trainer import (
    AdversarialTrainer,
    AdversarialTrainingConfig,
    train_adversarial_epoch,
    validate_robust,
)
from .base_trainer import BaseTrainer
from .baseline_trainer import BaselineTrainer
from .tri_objective_trainer import TriObjectiveTrainer

__all__ = [
    # Base infrastructure
    "BaseTrainer",
    "BaselineTrainer",
    "TriObjectiveTrainer",
    # Phase 5.1: Adversarial training
    "AdversarialTrainer",
    "AdversarialTrainingConfig",
    "train_adversarial_epoch",
    "validate_robust",
]

__version__ = "5.1.0"
