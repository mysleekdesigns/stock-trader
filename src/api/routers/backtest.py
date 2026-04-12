"""Backtest endpoints — launch and retrieve backtest results."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas.backtest import BacktestRequest, BacktestResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# ---------------------------------------------------------------------------
# In-memory result store
# ---------------------------------------------------------------------------

_results: dict[str, BacktestResponse] = {}


async def _run_backtest(backtest_id: str, request: BacktestRequest) -> None:
    """Execute a backtest in the background and store the result.

    Imports the engine lazily so the router module stays lightweight.
    """
    try:
        from src.backtest.engine import BacktestEngine

        engine = BacktestEngine(config={"strategy": None})

        result = await engine.run_async(
            symbols=request.symbols,
            start=datetime.combine(request.start_date, datetime.min.time()),
            end=datetime.combine(request.end_date, datetime.min.time()),
            timeframe=request.timeframe,
            initial_capital=request.initial_capital,
        )

        _results[backtest_id] = BacktestResponse(
            id=backtest_id,
            status="completed",
            metrics=result.metrics,
            equity_curve=result.equity_curve,
            trades=result.trades,
        )

        logger.info("backtest.completed", backtest_id=backtest_id)

    except Exception as exc:
        logger.error("backtest.failed", backtest_id=backtest_id, error=str(exc))
        _results[backtest_id] = BacktestResponse(
            id=backtest_id,
            status="failed",
            metrics={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=BacktestResponse, status_code=202)
async def launch_backtest(body: BacktestRequest) -> BacktestResponse:
    """Launch a backtest as an async background task.

    Returns immediately with a ``pending`` status and a backtest ID that
    can be polled via ``GET /api/backtest/{id}``.
    """
    backtest_id = str(uuid.uuid4())

    response = BacktestResponse(id=backtest_id, status="pending")
    _results[backtest_id] = response

    # Fire-and-forget background task
    asyncio.create_task(_run_backtest(backtest_id, body))

    logger.info(
        "backtest.launched",
        backtest_id=backtest_id,
        symbols=body.symbols,
        strategy=body.strategy,
    )

    return response


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(backtest_id: str) -> BacktestResponse:
    """Retrieve the current status and results of a backtest."""
    result = _results.get(backtest_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Backtest '{backtest_id}' not found")
    return result
