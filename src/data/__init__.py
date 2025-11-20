# src/data/__init__.py
"""
Data utilities package.

Currently exposes the dataset readiness checker.
"""

from .ready_check import DatasetReadyReport, run_ready_check, save_report  # noqa: F401

__all__ = ["DatasetReadyReport", "run_ready_check", "save_report"]
