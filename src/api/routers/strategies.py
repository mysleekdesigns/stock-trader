"""Strategy management endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_event_bus
from src.api.schemas.strategy import StrategyResponse, StrategyToggle
from src.core.events import EventBus

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# ---------------------------------------------------------------------------
# In-process strategy registry
# ---------------------------------------------------------------------------
# In a real deployment this would be backed by the ExecutionEngine's strategy
# list.  Here we keep a module-level dict so routers remain self-contained
# during testing or when the engine is not yet running.

_strategies: dict[str, dict[str, Any]] = {}


def register_strategy(name: str, strategy: Any) -> None:
    """Register a strategy instance so the API can expose it."""
    _strategies[name] = {
        "instance": strategy,
        "signal_count": 0,
        "last_signal": None,
    }


def _get_strategy_response(name: str) -> StrategyResponse:
    entry = _strategies.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    strategy = entry["instance"]
    return StrategyResponse(
        name=strategy.name,
        enabled=strategy.enabled,
        weight=getattr(strategy, "weight", 1.0),
        params=getattr(strategy, "params", {}),
        last_signal=entry.get("last_signal"),
        signal_count=entry.get("signal_count", 0),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[StrategyResponse])
async def list_strategies() -> list[StrategyResponse]:
    """Return all registered strategies."""
    return [_get_strategy_response(name) for name in _strategies]


@router.get("/{name}", response_model=StrategyResponse)
async def get_strategy(name: str) -> StrategyResponse:
    """Return details for a single strategy."""
    return _get_strategy_response(name)


@router.post("/{name}/toggle", response_model=StrategyResponse)
async def toggle_strategy(
    name: str,
    body: StrategyToggle,
    event_bus: EventBus = Depends(get_event_bus),
) -> StrategyResponse:
    """Enable or disable a strategy."""
    entry = _strategies.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    strategy = entry["instance"]
    previous = strategy.enabled
    strategy.enabled = body.enabled

    logger.info(
        "strategy.toggled",
        name=name,
        previous=previous,
        enabled=body.enabled,
    )

    return _get_strategy_response(name)
