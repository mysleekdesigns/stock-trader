"""File-based model registry for versioned model artifact management.

Provides promotion stages (staging -> production -> archived) and metadata
tracking without requiring external services like MLflow.
"""

from __future__ import annotations

import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import structlog

from src.core.exceptions import ModelError, ModelLoadError
from src.models.base import BaseModelPredictor

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """File-based model registry with versioning and stage promotion.

    Directory layout::

        registry_dir/
            {name}/
                {version}/
                    metadata.json
                    model.pkl

    Parameters
    ----------
    registry_dir:
        Root directory for all registered models.
    """

    VALID_STAGES = ("staging", "production", "archived")

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        logger.info("model_registry_init", registry_dir=str(self.registry_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: BaseModelPredictor,
        name: str,
        version: str,
        metrics: dict[str, Any],
        stage: str = "staging",
    ) -> str:
        """Register a model artifact with metadata.

        Parameters
        ----------
        model:
            Trained predictor instance.
        name:
            Logical model name (e.g. ``"lightgbm_v1"``).
        version:
            Semantic version string (e.g. ``"1.0.0"``).
        metrics:
            Performance metrics dict persisted alongside the model.
        stage:
            Initial deployment stage.

        Returns
        -------
        str
            Unique model ID (``{name}/{version}``).
        """
        self._validate_stage(stage)

        model_id = f"{name}/{version}"
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifact
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # If model has its own save, also call it for native format
        native_dir = model_dir / "native"
        try:
            model.save(native_dir)
        except Exception:
            logger.debug("registry_native_save_skipped", model_id=model_id)

        # Save metadata
        metadata = {
            "name": name,
            "version": version,
            "stage": stage,
            "metrics": metrics,
            "model_class": type(model).__name__,
            "is_fitted": getattr(model, "is_fitted", None),
            "feature_names": getattr(model, "feature_names", None),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = model_dir / "metadata.json"
        meta_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

        logger.info(
            "model_registered",
            model_id=model_id,
            stage=stage,
            metrics=metrics,
        )
        return model_id

    def promote(self, model_id: str, stage: str = "production") -> None:
        """Promote (or demote) a model to the given stage.

        If promoting to ``production``, any existing production model of the
        same name is automatically archived.

        Parameters
        ----------
        model_id:
            The model ID (``{name}/{version}``).
        stage:
            Target stage.
        """
        self._validate_stage(stage)

        name, version = self._parse_model_id(model_id)
        model_dir = self.registry_dir / name / version

        if not model_dir.exists():
            raise ModelError(
                f"Model not found: {model_id}",
                details={"model_id": model_id},
            )

        # Archive the current production model if promoting to production
        if stage == "production":
            for existing in self._find_models(name=name, stage="production"):
                if existing["version"] != version:
                    self._update_stage(name, existing["version"], "archived")

        self._update_stage(name, version, stage)
        logger.info("model_promoted", model_id=model_id, stage=stage)

    def get_model(self, name: str, stage: str = "production") -> BaseModelPredictor:
        """Load the model artifact for *name* at the given *stage*.

        If multiple versions share the same stage, the most recently
        registered one is returned.
        """
        models = self._find_models(name=name, stage=stage)
        if not models:
            raise ModelLoadError(
                f"No model found: name={name}, stage={stage}",
                details={"name": name, "stage": stage},
            )

        # Pick the latest by registration time
        latest = sorted(models, key=lambda m: m.get("registered_at", ""))[-1]
        return self._load_artifact(name, latest["version"])

    def get_latest(self, name: str) -> BaseModelPredictor:
        """Load the most recently registered version of *name* regardless of stage."""
        models = self._find_models(name=name)
        if not models:
            raise ModelLoadError(
                f"No model found: name={name}",
                details={"name": name},
            )

        latest = sorted(models, key=lambda m: m.get("registered_at", ""))[-1]
        return self._load_artifact(name, latest["version"])

    def list_models(
        self,
        name: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered models, optionally filtered by name and/or stage.

        Returns
        -------
        list of dict
            Metadata dicts for each matching model.
        """
        if name is not None:
            return self._find_models(name=name, stage=stage)

        results: list[dict[str, Any]] = []
        if not self.registry_dir.exists():
            return results

        for name_dir in sorted(self.registry_dir.iterdir()):
            if name_dir.is_dir():
                results.extend(self._find_models(name=name_dir.name, stage=stage))

        return results

    def delete(self, model_id: str) -> None:
        """Permanently remove a model artifact and its metadata."""
        name, version = self._parse_model_id(model_id)
        model_dir = self.registry_dir / name / version

        if not model_dir.exists():
            raise ModelError(
                f"Model not found: {model_id}",
                details={"model_id": model_id},
            )

        shutil.rmtree(model_dir)
        logger.info("model_deleted", model_id=model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_models(
        self,
        name: str,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find all versions of *name*, optionally filtered by *stage*."""
        name_dir = self.registry_dir / name
        if not name_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for version_dir in sorted(name_dir.iterdir()):
            meta_path = version_dir / "metadata.json"
            if not meta_path.exists():
                continue

            metadata = orjson.loads(meta_path.read_bytes())
            if stage is not None and metadata.get("stage") != stage:
                continue
            results.append(metadata)

        return results

    def _load_artifact(self, name: str, version: str) -> BaseModelPredictor:
        """Load a pickled model artifact."""
        model_path = self.registry_dir / name / version / "model.pkl"
        if not model_path.exists():
            raise ModelLoadError(
                f"Model artifact not found: {name}/{version}",
                details={"path": str(model_path)},
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301

        logger.info("model_loaded", name=name, version=version)
        return model

    def _update_stage(self, name: str, version: str, stage: str) -> None:
        """Update the stage field in a model's metadata."""
        meta_path = self.registry_dir / name / version / "metadata.json"
        if not meta_path.exists():
            return

        metadata = orjson.loads(meta_path.read_bytes())
        old_stage = metadata.get("stage")
        metadata["stage"] = stage
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

        logger.info(
            "model_stage_updated",
            name=name,
            version=version,
            old_stage=old_stage,
            new_stage=stage,
        )

    def _validate_stage(self, stage: str) -> None:
        if stage not in self.VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}"
            )

    @staticmethod
    def _parse_model_id(model_id: str) -> tuple[str, str]:
        parts = model_id.split("/", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid model_id '{model_id}'. Expected format: 'name/version'"
            )
        return parts[0], parts[1]
