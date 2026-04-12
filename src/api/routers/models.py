"""Model performance endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])

# Default registry path — overridden by DI or config in production.
_REGISTRY_DIR = Path("model_registry")


def _get_registry() -> Any:
    """Lazily import and construct the ModelRegistry."""
    from src.models.training.registry import ModelRegistry

    return ModelRegistry(registry_dir=_REGISTRY_DIR)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_models() -> list[dict[str, Any]]:
    """Return metadata and performance metrics for all registered models."""
    try:
        registry = _get_registry()
        return registry.list_models()
    except Exception as exc:
        logger.warning("models.list_error", error=str(exc))
        return []


@router.get("/{name}")
async def get_model_details(name: str) -> dict[str, Any]:
    """Return details for a specific model, including feature importance."""
    try:
        registry = _get_registry()
        entries = registry.list_models(name=name)
        if not entries:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        # Pick the most recent entry
        entry = sorted(entries, key=lambda e: e.get("registered_at", ""))[-1]

        # Attempt to load the model to extract feature importance
        feature_importance: dict[str, float] = {}
        try:
            model = registry.get_latest(name)
            if hasattr(model, "get_feature_importance"):
                feature_importance = model.get_feature_importance()
        except Exception:
            logger.debug("models.feature_importance_unavailable", name=name)

        return {
            **entry,
            "feature_importance": feature_importance,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("models.detail_error", name=name, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to load model details") from exc
