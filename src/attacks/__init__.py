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

from .auto_attack import AutoAttack, AutoAttackConfig, autoattack

# Base classes
from .base import AttackConfig, AttackResult, BaseAttack
from .cw import CarliniWagner, CWConfig, cw_attack

# Attack implementations
from .fgsm import FGSM, FGSMConfig, fgsm_attack
from .pgd import PGD, PGDConfig, pgd_attack

__version__ = "0.4.2"

__all__ = [
    # Base
    "BaseAttack",
    "AttackConfig",
    "AttackResult",
    # FGSM
    "FGSM",
    "FGSMConfig",
    "fgsm_attack",
    # PGD
    "PGD",
    "PGDConfig",
    "pgd_attack",
    # C&W
    "CarliniWagner",
    "CWConfig",
    "cw_attack",
    # AutoAttack
    "AutoAttack",
    "AutoAttackConfig",
    "autoattack",
]
