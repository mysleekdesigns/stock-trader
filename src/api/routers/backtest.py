"""Backtest endpoints — launch and retrieve backtest results."""

from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas.backtest import BacktestRequest, BacktestResponse
from src.backtest.runner import (
    NoDataError,
    UnsupportedStrategyError,
)
from src.backtest.runner import (
    run_backtest as run_real_backtest,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# ---------------------------------------------------------------------------
# In-memory result store
# ---------------------------------------------------------------------------

_results: dict[str, BacktestResponse] = {}


async def _run_backtest(backtest_id: str, request: BacktestRequest) -> None:
    """Execute a real backtest in the background and store the result."""
    try:
        logger.info(
            "backtest.running",
            backtest_id=backtest_id,
            strategy=request.strategy,
            symbols=request.symbols,
        )
        metrics, equity, trades, timestamps = await run_real_backtest(request)
        _results[backtest_id] = BacktestResponse(
            id=backtest_id,
            status="completed",
            metrics=metrics,
            equity_curve=equity,
            timestamps=timestamps,
            trades=trades,
        )
        logger.info(
            "backtest.completed",
            backtest_id=backtest_id,
            bars=len(equity),
            trades=len(trades),
        )

    except (UnsupportedStrategyError, NoDataError) as exc:
        # Expected, user-facing failures — report the message cleanly.
        logger.warning("backtest.rejected", backtest_id=backtest_id, error=str(exc))
        _results[backtest_id] = BacktestResponse(
            id=backtest_id,
            status="failed",
            metrics={"error": str(exc)},
        )
    except Exception as exc:
        logger.error("backtest.failed", backtest_id=backtest_id, error=str(exc), exc_info=exc)
        _results[backtest_id] = BacktestResponse(
            id=backtest_id,
            status="failed",
            metrics={"error": f"Backtest failed: {exc}"},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=BacktestResponse, status_code=202)
async def launch_backtest(body: BacktestRequest) -> BacktestResponse:
    """Launch a backtest as an async background task."""
    backtest_id = str(uuid.uuid4())
    response = BacktestResponse(id=backtest_id, status="pending")
    _results[backtest_id] = response

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
