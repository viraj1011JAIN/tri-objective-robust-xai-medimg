# src/datasets/data_governance.py
"""
Centralised data governance and compliance utilities.

This module is deliberately lightweight (stdlib only) and can be imported
from any script in the project (preprocessing, training, evaluation).

Main responsibilities
---------------------
- Data access logging  (who/what read which dataset, when, for what purpose)
- Data provenance      (which inputs -> which outputs, with what params)
- Compliance checks    (is this dataset being used in an allowed way?)
- Dataset metadata     (source, license summary, usage restrictions)

Typical usage
-------------
from src.datasets import data_governance as gov

gov.assert_data_usage_allowed("isic2020", purpose="research")

gov.log_data_access(
    dataset_name="isic2020",
    split="train",
    purpose="training",
    num_samples=len(train_dataset),
)

gov.log_provenance(
    stage="preprocess_isic2020",
    dataset_name="isic2020",
    input_paths=["F:/data/isic_2020/metadata.csv"],
    output_paths=["data/processed/isic2020/isic2020_train.h5"],
    params={"image_size": 224, "normalize": "zero_one"},
)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetLicenseInfo:
    """Minimal, *non-authoritative* summary of dataset licensing/terms.

    This is for runtime hints and documentation. It does NOT replace
    reading and complying with each dataset's official license/DUA.
    """

    name: str
    url: Optional[str]
    summary: str


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    display_name: str
    source_url: str
    license: DatasetLicenseInfo
    allowed_purposes: Sequence[str] = field(
        default_factory=lambda: ("research", "education")
    )
    allow_commercial: bool = False
    contains_direct_identifiers: bool = False
    notes: str = ""


def _dataset_info_registry() -> Dict[str, DatasetInfo]:
    """Build the dataset registry.

    Keys are canonical slugs (e.g. "isic2018", "nih_cxr").
    """
    return {
        # ISIC challenge dermoscopy datasets
        "isic2018": DatasetInfo(
            key="isic2018",
            display_name="ISIC 2018 Dermoscopy",
            source_url="https://challenge2018.isic-archive.com/",
            license=DatasetLicenseInfo(
                name="ISIC Archive / challenge terms",
                url="https://www.isic-archive.com/#terms-of-use",
                summary=(
                    "Public dermoscopy images for research and education; "
                    "no clinical use; follow ISIC terms and cite organisers."
                ),
            ),
            notes="Multi-class lesion classification benchmark.",
        ),
        "isic2019": DatasetInfo(
            key="isic2019",
            display_name="ISIC 2019 Dermoscopy",
            source_url="https://challenge2019.isic-archive.com/",
            license=DatasetLicenseInfo(
                name="ISIC Archive / challenge terms",
                url="https://www.isic-archive.com/#terms-of-use",
                summary=(
                    "Public dermoscopy dataset for non-clinical research; "
                    "subject to ISIC terms of use and challenge rules."
                ),
            ),
            notes="Larger, multi-institutional extension of ISIC 2018.",
        ),
        "isic2020": DatasetInfo(
            key="isic2020",
            display_name="ISIC 2020 Dermoscopy",
            source_url="https://challenge2020.isic-archive.com/",
            license=DatasetLicenseInfo(
                name="ISIC Archive / Kaggle challenge terms",
                url="https://www.isic-archive.com/#terms-of-use",
                summary=(
                    "Dermoscopy images provided for research and algorithm "
                    "development; no direct diagnostic use."
                ),
            ),
            notes="Melanoma vs non-melanoma classification.",
        ),
        # Derm7pt
        "derm7pt": DatasetInfo(
            key="derm7pt",
            display_name="Derm7pt Dermoscopy",
            source_url="https://github.com/jeremykawahara/derm7pt",
            license=DatasetLicenseInfo(
                name="Derm7pt dataset license",
                url="https://github.com/jeremykawahara/derm7pt",
                summary=(
                    "Research dermoscopy dataset; requires citing the "
                    "original paper and following repository/license terms."
                ),
            ),
            notes="Includes 7-point criteria and diagnostic labels.",
        ),
        # NIH ChestXray14
        "nih_cxr": DatasetInfo(
            key="nih_cxr",
            display_name="NIH ChestXray14",
            source_url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
            license=DatasetLicenseInfo(
                name="NIH data use terms",
                url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
                summary=(
                    "De-identified chest radiographs released by NIH for "
                    "research; subject to NIH data use agreement; no "
                    "re-identification or clinical use."
                ),
            ),
            contains_direct_identifiers=False,
            notes="Multi-label thoracic disease classification.",
        ),
        # PadChest
        "padchest": DatasetInfo(
            key="padchest",
            display_name="PadChest",
            source_url="https://bimcv.cipf.es/bimcv-projects/padchest/",
            license=DatasetLicenseInfo(
                name="PadChest research-only license",
                url="https://bimcv.cipf.es/bimcv-projects/padchest/",
                summary=(
                    "Spanish chest X-ray dataset for scientific research; "
                    "requires signed data usage agreement; sale or "
                    "redistribution is forbidden; not for diagnosis."
                ),
            ),
            contains_direct_identifiers=False,
            notes="Multi-label dataset with detailed text labels.",
        ),
    }


_DATASET_REGISTRY: Dict[str, DatasetInfo] = _dataset_info_registry()

# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("data_governance")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _detect_project_root() -> Path:
    """Try to locate the Git repo root; fall back to parent dirs.

    This avoids hard-coding paths and behaves well in tests and notebooks.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").is_dir():
            return parent
    # Fallback: src/.. if possible, else current working directory
    if len(here.parents) >= 2:
        return here.parents[2 - 1]  # usually <repo>/src/datasets/file.py
    return Path.cwd()


def get_governance_dir() -> Path:
    """Return the base directory where governance logs are written.

    Priority:
    1. DATA_GOVERNANCE_DIR environment variable (if set)
    2. <repo_root>/logs/data_governance
    """
    env = os.environ.get("DATA_GOVERNANCE_DIR")
    if env:
        base = Path(env).expanduser().resolve()
    else:
        base = _detect_project_root() / "logs" / "data_governance"
    base.mkdir(parents=True, exist_ok=True)
    return base


AUDIT_LOG_FILENAME = "data_access.jsonl"
PROVENANCE_LOG_FILENAME = "data_provenance.jsonl"
COMPLIANCE_LOG_FILENAME = "compliance_checks.jsonl"


def _append_json_record(
    filename: str,
    record: Mapping[str, Any],
    governance_dir: Optional[Path] = None,
) -> None:
    """Append a JSON record to the specified log (one record per line).

    Failures are logged but never raise, so governance never breaks
    the main training / preprocessing logic.
    """
    logger = _get_logger()
    base = governance_dir or get_governance_dir()
    path = base / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, default=str)
            f.write("\n")
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to append governance record to %s", path)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_script_name() -> str:
    if getattr(sys, "argv", None):
        return os.path.basename(sys.argv[0])
    return "<interactive>"


def _get_git_commit() -> Optional[str]:
    """Return current Git commit hash if available, else None."""
    try:
        import subprocess

        root = _detect_project_root()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # pragma: no cover - environment dependent
        return None


def canonicalize_dataset_key(name: str) -> str:
    """Map various spellings to a canonical dataset key."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    synonyms = {
        "isic18": "isic2018",
        "isic_2018": "isic2018",
        "isic19": "isic2019",
        "isic_2019": "isic2019",
        "isic20": "isic2020",
        "isic_2020": "isic2020",
        "nih": "nih_cxr",
        "nih_chestxray": "nih_cxr",
        "chestxray14": "nih_cxr",
        "nih_cxr": "nih_cxr",
        "pad_chest": "padchest",
        "pad-chest": "padchest",
    }
    return synonyms.get(key, key)


# ---------------------------------------------------------------------------
# Public dataset metadata API
# ---------------------------------------------------------------------------


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Return DatasetInfo for the given dataset name (canonicalising first)."""
    key = canonicalize_dataset_key(dataset_name)
    try:
        return _DATASET_REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown dataset for governance: {dataset_name!r} (key={key})")


def list_datasets() -> Dict[str, DatasetInfo]:
    """Return a copy of the internal dataset registry."""
    return dict(_DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# Data access logging
# ---------------------------------------------------------------------------


def log_data_access(
    dataset_name: str,
    split: Optional[str] = None,
    *,
    action: str = "read",
    purpose: str = "training",
    num_samples: Optional[int] = None,
    user: Optional[str] = None,
    script: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    governance_dir: Optional[Path] = None,
) -> None:
    """Log a single data access event.

    Parameters
    ----------
    dataset_name:
        Name or slug of the dataset (any spelling; will be canonicalised).
    split:
        Data split (train/val/test) if applicable.
    action:
        What happened: "read", "write", "download", etc.
    purpose:
        High-level purpose: "training", "validation", "evaluation",
        "debugging", "exploration", etc.
    num_samples:
        Optional size of the accessed batch/dataset.
    user:
        Optional user identifier; falls back to OS user.
    script:
        Optional script/module name; falls back to sys.argv[0].
    extra:
        Arbitrary additional metadata (serialisable to JSON).
    governance_dir:
        Override base log directory; normally leave as None.
    """
    logger = _get_logger()
    key = canonicalize_dataset_key(dataset_name)
    info = _DATASET_REGISTRY.get(key)

    record: Dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "event": "data_access",
        "dataset": key,
        "split": split,
        "action": action,
        "purpose": purpose,
        "num_samples": num_samples,
        "user": user or os.environ.get("USERNAME") or os.environ.get("USER"),
        "script": script or _infer_script_name(),
        "cwd": str(Path.cwd()),
        "git_commit": _get_git_commit(),
    }

    if info is not None:
        record["dataset_display_name"] = info.display_name
        record["license_name"] = info.license.name
        record["license_url"] = info.license.url
        record["allowed_purposes"] = list(info.allowed_purposes)
        record["allow_commercial"] = info.allow_commercial

    if extra:
        record["extra"] = dict(extra)

    _append_json_record(AUDIT_LOG_FILENAME, record, governance_dir)
    logger.debug("Logged data access: %s", record)


# ---------------------------------------------------------------------------
# Provenance logging
# ---------------------------------------------------------------------------


def log_provenance(
    stage: str,
    *,
    dataset_name: Optional[str] = None,
    input_paths: Optional[Iterable[str | Path]] = None,
    output_paths: Optional[Iterable[str | Path]] = None,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    governance_dir: Optional[Path] = None,
) -> None:
    """Log a provenance event (inputs -> outputs with parameters).

    This is meant to complement DVC: DVC tracks file-level provenance
    and this function tracks *semantic* provenance at the Python level.

    Parameters
    ----------
    stage:
        High-level stage name, e.g. "preprocess_isic2020" or "train_baseline".
    dataset_name:
        Optional dataset this stage primarily relates to.
    input_paths:
        Iterable of input files/directories (will be stringified).
    output_paths:
        Iterable of output files/directories (will be stringified).
    params:
        Dictionary of key hyper-parameters or configuration values.
    tags:
        Short string tags (e.g. {"experiment": "rq1_baseline"}).
    governance_dir:
        Override base log directory.
    """
    logger = _get_logger()
    key = canonicalize_dataset_key(dataset_name) if dataset_name else None

    def _to_str_list(paths: Optional[Iterable[str | Path]]) -> list[str]:
        if not paths:
            return []
        return [str(Path(p)) for p in paths]

    record: Dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "event": "provenance",
        "stage": stage,
        "dataset": key,
        "inputs": _to_str_list(input_paths),
        "outputs": _to_str_list(output_paths),
        "params": dict(params) if params else {},
        "tags": dict(tags) if tags else {},
        "script": _infer_script_name(),
        "cwd": str(Path.cwd()),
        "git_commit": _get_git_commit(),
    }

    _append_json_record(PROVENANCE_LOG_FILENAME, record, governance_dir)
    logger.debug("Logged provenance: %s", record)


# ---------------------------------------------------------------------------
# Compliance checks
# ---------------------------------------------------------------------------


def assert_data_usage_allowed(
    dataset_name: str,
    *,
    purpose: str = "research",
    commercial: bool = False,
    country: Optional[str] = None,
    governance_dir: Optional[Path] = None,
) -> None:
    """Soft compliance guard: raise if usage clearly violates basic rules.

    This is intentionally conservative and high-level. It does NOT provide
    legal guarantees; it is a reminder and a loggable guard.

    Typical usage
    -------------
    # Will raise if you accidentally try commercial usage on a
    # research-only dataset.
    assert_data_usage_allowed("padchest", purpose="research", commercial=False)
    """
    logger = _get_logger()
    info = get_dataset_info(dataset_name)
    purpose_norm = purpose.strip().lower()

    allowed_purposes = {p.lower() for p in info.allowed_purposes}
    commercial_allowed = info.allow_commercial

    result = "allowed"
    reason: Optional[str] = None

    if purpose_norm not in allowed_purposes:
        result = "denied"
        reason = f"purpose {purpose!r} not in allowed_purposes={allowed_purposes}"
    elif commercial and not commercial_allowed:
        result = "denied"
        reason = "commercial usage is not allowed by this configuration"

    record: Dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "event": "compliance_check",
        "dataset": info.key,
        "dataset_display_name": info.display_name,
        "purpose": purpose,
        "commercial": bool(commercial),
        "country": country,
        "result": result,
        "reason": reason,
        "license": asdict(info.license),
        "script": _infer_script_name(),
        "cwd": str(Path.cwd()),
        "git_commit": _get_git_commit(),
    }

    _append_json_record(COMPLIANCE_LOG_FILENAME, record, governance_dir)

    if result != "allowed":
        logger.error(
            "Compliance check FAILED for dataset=%s: %s", info.key, reason or ""
        )
        raise PermissionError(
            f"Data usage for dataset={info.key!r} failed compliance check: {reason}"
        )

    logger.debug(
        "Compliance check passed for dataset=%s (purpose=%s, commercial=%s)",
        info.key,
        purpose,
        commercial,
    )


__all__ = [
    "DatasetLicenseInfo",
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    "canonicalize_dataset_key",
    "get_governance_dir",
    "log_data_access",
    "log_provenance",
    "assert_data_usage_allowed",
]
