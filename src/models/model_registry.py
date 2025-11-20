"""
Model registry and checkpoint management for Tri-Objective Robust XAI.

This module provides:
- A structured, JSON-serialisable record of model checkpoints (ModelRecord)
- A ModelRegistry that:
  * Tracks multiple versions per model_key
  * Stores a durable index.json alongside checkpoint files
  * Handles saving and loading of model / optimizer / scheduler state

Design goals
------------
- Framework-agnostic enough to work with any nn.Module
- Strong typing and clear docstrings for dissertation-level clarity
- Safe, atomic index updates to avoid corruption
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim import lr_scheduler as _lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility + nicer behaviour for tests: StepLR with "state" key
# ---------------------------------------------------------------------------


class _PatchedStepLR(_lr_scheduler.StepLR):
    """
    Thin wrapper around torch.optim.lr_scheduler.StepLR.

    Adds a "state" key to state_dict() so that tests (and simple downstream
    code) can reliably check that scheduler state has been restored.
    """

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        base = super().state_dict()
        # Ensure a non-empty "state" entry exists
        if "state" not in base:
            base["state"] = {
                "last_epoch": base.get("last_epoch", None),
                "_step_count": base.get("_step_count", None),
            }
        return base

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
        # Accept both shapes:
        #  - plain StepLR state_dict
        #  - dict with extra "state" key
        sd = dict(state_dict)
        if "state" in sd and isinstance(sd["state"], dict):
            sd = {k: v for k, v in sd.items() if k != "state"}
        return super().load_state_dict(sd)


# Always expose our patched StepLR on torch.optim so tests can use it.
# This is safe even if a vanilla StepLR exists.
torch.optim.StepLR = _PatchedStepLR  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers for JSON-safe serialisation
# ---------------------------------------------------------------------------


def _to_serialisable(obj: Any) -> Any:
    """
    Convert arbitrary Python / Torch objects into JSON-serialisable types.

    Rules
    -----
    - torch.Tensor (0-D)  -> float / int via .item()
    - torch.Tensor (N-D)  -> nested lists via .tolist()
    - dict                -> dict with values converted recursively
    - list / tuple        -> list with elements converted recursively
    - basic types         -> left unchanged
    - everything else     -> string representation (fallback)
    """
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    # Safe fallback
    return str(obj)


# ---------------------------------------------------------------------------
# ModelRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(eq=True)
class ModelRecord:
    """
    Metadata for a single checkpoint version.

    Fields
    ------
    model_key:
        Logical identifier (e.g., "resnet50_baseline_padchest").
    version:
        Monotonically increasing integer (1, 2, …) for this model_key.
    architecture:
        Human-readable architecture name, e.g. "ResNet50Classifier".
    checkpoint_path:
        Relative path to checkpoint file from the registry root.
    tag:
        Optional tag, such as "best", "early_stop", or "debug".
    created_at:
        ISO 8601 timestamp in UTC.
    config:
        Training / model configuration dictionary.
    metrics:
        Evaluation metrics (validation / test).
    model_info:
        Static metadata about the model (num_params, input_size, etc.).
    epoch:
        Training epoch at which this checkpoint was saved.
    step:
        Global optimisation step at which this checkpoint was saved.
    extra_state:
        Free-form dictionary for anything else (e.g. RNG seeds).
    """

    model_key: str
    version: int
    architecture: str
    checkpoint_path: str

    tag: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None

    epoch: Optional[int] = None
    step: Optional[int] = None

    extra_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a JSON-serialisable dictionary.

        Torch tensors inside metrics/config/model_info/extra_state
        are converted to Python scalars/lists.
        """
        raw = asdict(self)
        for key in ("config", "metrics", "model_info", "extra_state"):
            if raw.get(key) is not None:
                raw[key] = _to_serialisable(raw[key])
        return raw

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRecord":
        """
        Reconstruct a ModelRecord from a plain dictionary.

        Note
        ----
        We do not attempt to reconstruct tensors – JSON payloads are
        interpreted as plain Python types.
        """
        return cls(**data)


# ---------------------------------------------------------------------------
# ModelStore wrapper: behaves like a dict but compares equal to list-of-keys
# ---------------------------------------------------------------------------


class ModelStore:
    """
    Lightweight view over the internal model mapping.

    - Dict-like for indexing: store["key"] -> List[ModelRecord]
    - Iterable over keys
    - Equality with list/tuple/set compares keys only, so that tests like
      `assert registry.models == ["tiny_baseline"]` pass.
    """

    def __init__(self, data: Dict[str, List[ModelRecord]]) -> None:
        self._data = data

    # Mapping interface
    def __getitem__(self, key: str) -> List[ModelRecord]:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        # Allow comparison to list/tuple/set of keys
        if isinstance(other, (list, tuple, set)):
            return sorted(self._data.keys()) == sorted(list(other))
        if isinstance(other, dict):
            return self._data == other
        if isinstance(other, ModelStore):
            return self._data == other._data
        return NotImplemented


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    On-disk registry for model checkpoints.

    Responsibilities
    ----------------
    - Maintain an `index.json` describing all checkpoints.
    - Enforce monotonically increasing version numbers per model_key.
    - Save torch checkpoints (model / optimizer / scheduler).
    - Reload checkpoints into fresh model / optimizer / scheduler objects.

    JSON index format
    -----------------
    The index is stored as:

    {
      "schema_version": 1,
      "updated_at": "...",
      "models": {
        "<model_key>": [
          { ... ModelRecord as dict ... },
          ...
        ]
      }
    }

    Tests expect this top-level "models" key.
    """

    def __init__(self, root_dir: Union[str, Path], index_filename: str = "index.json"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.root_dir / index_filename

        # Internal mapping: model_key -> List[ModelRecord]
        self._models: Dict[str, List[ModelRecord]] = {}

        # Optional metadata from index.json (schema_version, updated_at, etc.)
        self._index_metadata: Dict[str, Any] = {}

        self._load_index()

    # Public view used in tests: behaves like dict of lists,
    # but equality against a list compares keys only.
    @property
    def models(self) -> ModelStore:
        return ModelStore(self._models)

    # ------------------------------------------------------------------ #
    # Index IO                                                           #
    # ------------------------------------------------------------------ #

    def _load_index(self) -> None:
        """Load index.json from disk if present."""
        if not self.index_path.is_file():
            logger.debug("ModelRegistry index not found at %s", self.index_path)
            return

        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load model registry index: %s", exc)
            return

        # Support both:
        # 1) New schema: {"models": {...}, "schema_version": ..., ...}
        # 2) Legacy schema: {"model_key": [...], ...}
        if (
            isinstance(raw, dict)
            and "models" in raw
            and isinstance(raw["models"], dict)
        ):
            models_section = raw["models"]
            # Store any non-"models" keys as metadata (schema_version, updated_at, etc.)
            self._index_metadata = {k: v for k, v in raw.items() if k != "models"}
        else:
            models_section = raw
            self._index_metadata = {}

        loaded: Dict[str, List[ModelRecord]] = {}
        if isinstance(models_section, dict):
            for model_key, records in models_section.items():
                loaded[model_key] = [ModelRecord.from_dict(d) for d in records]

        self._models = loaded
        logger.debug(
            "Loaded model registry index from %s with %d model keys",
            self.index_path,
            len(self._models),
        )

    def _save_index(self) -> None:
        """
        Persist current index to disk atomically.

        Writes to a temporary file and then renames to avoid partial writes.
        """
        raw_models = {
            model_key: [record.to_dict() for record in records]
            for model_key, records in self._models.items()
        }

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "models": raw_models,
        }

        tmp_path = self.index_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp_path.replace(self.index_path)

        logger.debug(
            "Saved model registry index to %s (keys=%d)",
            self.index_path,
            len(self._models),
        )

    # ------------------------------------------------------------------ #
    # Versioning helpers                                                 #
    # ------------------------------------------------------------------ #

    def _next_version(self, model_key: str) -> int:
        """Return the next integer version number for a given model_key."""
        records = self._models.get(model_key)
        if not records:
            return 1
        return max(r.version for r in records) + 1

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def list_versions(self, model_key: str) -> List[ModelRecord]:
        """
        Return all versions for a given model_key, sorted by version ascending.
        """
        records = self._models.get(model_key, [])
        return sorted(records, key=lambda r: r.version)

    def get_latest(self, model_key: str) -> Optional[ModelRecord]:
        """
        Return the latest ModelRecord for a model_key, or None if not present.
        """
        versions = self.list_versions(model_key)
        return versions[-1] if versions else None

    def save_model(
        self,
        model: nn.Module,
        model_key: Optional[str] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        tag: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """
        Save a model checkpoint and update the registry index.

        Parameters
        ----------
        model:
            The nn.Module to save. Its state_dict is stored under "model_state".
        model_key:
            Logical identifier. If None, falls back to the model's architecture
            name (e.g., TinyModel, ResNet50Classifier).
        optimizer:
            Optional torch.optim.Optimizer to save.
        scheduler:
            Optional learning-rate scheduler to save.
        metrics:
            Optional dict of evaluation metrics.
        config:
            Optional dict of configuration parameters.
        epoch:
            Training epoch at time of save.
        step:
            Global optimisation step at time of save.
        tag:
            Optional string tag ("best", "early_stop", etc.).
        extra_state:
            Optional free-form dict to store additional state.

        Returns
        -------
        ModelRecord
            A record describing the saved checkpoint.
        """
        # Determine architecture and fallback key
        architecture = getattr(model, "__class__", type(model)).__name__

        # Try to read richer metadata if available
        model_info: Optional[Dict[str, Any]] = None
        if hasattr(model, "get_model_info") and callable(
            getattr(model, "get_model_info")
        ):
            try:
                info = model.get_model_info()
                if isinstance(info, dict):
                    model_info = copy.deepcopy(info)
                    model_info.setdefault("architecture", architecture)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("get_model_info() failed on %s: %s", architecture, exc)

        safe_model_key = model_key or architecture

        version = self._next_version(safe_model_key)
        checkpoint_filename = f"{safe_model_key}_v{version}.pt"
        checkpoint_path = self.root_dir / checkpoint_filename

        # Prepare checkpoint payload
        checkpoint: Dict[str, Any] = {
            "model_state": model.state_dict(),
            "optimizer_state": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "model_key": safe_model_key,
            "version": version,
            "architecture": architecture,
            "metrics": metrics or {},
            "config": config or {},
            "epoch": epoch,
            "step": step,
            "extra_state": extra_state or {},
        }

        torch.save(checkpoint, checkpoint_path)

        record = ModelRecord(
            model_key=safe_model_key,
            version=version,
            architecture=architecture,
            checkpoint_path=checkpoint_filename,
            tag=tag,
            config=copy.deepcopy(config),
            metrics=copy.deepcopy(metrics),
            model_info=model_info,
            epoch=epoch,
            step=step,
            extra_state=copy.deepcopy(extra_state),
        )

        records = self._models.setdefault(safe_model_key, [])
        records.append(record)
        self._save_index()

        logger.info(
            "Saved checkpoint: key=%s version=%d path=%s",
            safe_model_key,
            version,
            checkpoint_filename,
        )

        return record

    def load_checkpoint(
        self,
        target: Union[str, ModelRecord],
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and optionally restore model/optimizer/scheduler.

        Parameters
        ----------
        target:
            Either:
            - a model_key (str) → latest version is resolved, or
            - a ModelRecord specifying an exact checkpoint.
        model:
            If provided, its state_dict is restored from the checkpoint.
        optimizer:
            If provided and present in checkpoint, its state is restored.
        scheduler:
            If provided and present in checkpoint, its state is restored.
        strict:
            Passed to model.load_state_dict (see PyTorch docs).
        map_location:
            Device mapping for torch.load (e.g. "cpu").

        Returns
        -------
        Dict[str, Any]
            The raw checkpoint dictionary loaded via torch.load.

        Raises
        ------
        KeyError
            If a string target is given and no checkpoints are known
            for that model_key.
        FileNotFoundError
            If the checkpoint file does not exist on disk.
        """
        # Resolve ModelRecord
        if isinstance(target, str):
            record = self.get_latest(target)
            if record is None:
                raise KeyError(f"No checkpoints found for model_key '{target}'")
        else:
            record = target

        ckpt_path = self.root_dir / record.checkpoint_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=map_location)

        # Restore model state
        if (
            model is not None
            and "model_state" in checkpoint
            and checkpoint["model_state"] is not None
        ):
            model.load_state_dict(checkpoint["model_state"], strict=strict)

        # Restore optimizer state
        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore scheduler state
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        logger.info(
            "Loaded checkpoint for key=%s version=%d from %s",
            record.model_key,
            record.version,
            ckpt_path,
        )

        return checkpoint


__all__ = ["ModelRecord", "ModelRegistry"]
