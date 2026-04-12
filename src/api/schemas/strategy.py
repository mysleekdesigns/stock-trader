"""Pydantic models for strategy-related API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StrategyResponse(BaseModel):
    """Public representation of a registered strategy."""

    name: str
    enabled: bool
    weight: float
    params: dict[str, Any] = Field(default_factory=dict)
    last_signal: dict[str, Any] | None = None
    signal_count: int = 0


class StrategyToggle(BaseModel):
    """Request body to enable or disable a strategy."""

    enabled: bool
