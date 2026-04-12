"""Pydantic models for backtest-related API endpoints."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from src.core.types import TimeFrame


class BacktestRequest(BaseModel):
    """Payload for launching a new backtest."""

    symbols: list[str] = Field(min_length=1)
    start_date: date
    end_date: date
    strategy: str = Field(description="Strategy name to backtest")
    initial_capital: float = Field(default=100_000.0, gt=0)
    timeframe: TimeFrame = TimeFrame.DAILY


class BacktestResponse(BaseModel):
    """Result of a completed (or in-progress) backtest."""

    id: str
    status: str = "pending"
    metrics: dict[str, Any] = Field(default_factory=dict)
    equity_curve: list[float] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    report_url: str | None = None
