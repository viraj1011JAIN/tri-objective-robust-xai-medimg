"""
Adversarial Attacks Module
===========================

Comprehensive adversarial attack implementations for robustness evaluation
of medical imaging models.

This module provides:
- FGSM: Fast Gradient Sign Method (single-step L∞ attack)
- PGD: Projected Gradient Descent (multi-step L∞ attack)
- C&W: Carlini-Wagner L2 attack (optimization-based)
- AutoAttack: Ensemble of diverse parameter-free attacks

All attacks are designed for medical imaging pipelines with:
- Support for [0, 1] pixel ranges
- Optional normalization (ImageNet, custom)
- Proper model train/eval mode handling
- Comprehensive logging and statistics
- Type hints and production-grade error handling

Examples:
    >>> from src.attacks import FGSM, PGD, AutoAttack
    >>> from src.attacks import fgsm_attack, pgd_attack
    >>>
    >>> # Functional API
    >>> x_adv = fgsm_attack(model, images, labels, epsilon=8/255)
    >>>
    >>> # Class-based API
    >>> attack = PGD(epsilon=8/255, num_steps=10)
    >>> x_adv = attack(model, images, labels)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: January 2025
"""

from __future__ import annotations

# Base classes
from .base import BaseAttack, AttackConfig, AttackResult

# Attack implementations
from .fgsm import FGSM, fgsm_attack
from .pgd import PGD, pgd_attack
from .cw import CarliniWagner, cw_attack
from .auto_attack import AutoAttack, autoattack

__version__ = "0.4.1"

__all__ = [
    # Base
    "BaseAttack",
    "AttackConfig",
    "AttackResult",
    # FGSM
    "FGSM",
    "fgsm_attack",
    # PGD
    "PGD",
    "pgd_attack",
    # C&W
    "CarliniWagner",
    "cw_attack",
    # AutoAttack
    "AutoAttack",
    "autoattack",
]
