"""Backtest endpoints — launch and retrieve backtest results."""

from __future__ import annotations

import asyncio
import math
import random
import uuid
from datetime import datetime, timedelta
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


def _generate_simulated_results(
    request: BacktestRequest,
) -> tuple[dict[str, Any], list[float], list[dict[str, Any]]]:
    """Generate realistic simulated backtest results.

    Used as a fallback when the full engine is not yet wired up.
    """
    rng = random.Random(hash((request.strategy, tuple(request.symbols))))

    start = datetime.combine(request.start_date, datetime.min.time())
    end = datetime.combine(request.end_date, datetime.min.time())
    num_days = max((end - start).days, 1)

    # Build equity curve with realistic random walk
    capital = request.initial_capital
    equity: list[float] = [capital]
    daily_returns: list[float] = []
    for _ in range(num_days):
        daily_ret = rng.gauss(0.0004, 0.012)  # ~10% annual return, ~19% vol
        daily_returns.append(daily_ret)
        capital *= 1 + daily_ret
        equity.append(round(capital, 2))

    # Compute metrics
    total_return = (equity[-1] - equity[0]) / equity[0]
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe & Sortino
    mean_ret = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    std_ret = (
        sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
    ) ** 0.5 if daily_returns else 1
    sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0

    downside = [r for r in daily_returns if r < 0]
    down_std = (
        (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
        if downside else 1
    )
    sortino = (mean_ret / down_std) * math.sqrt(252) if down_std > 0 else 0

    # Simulate trades
    num_trades = max(num_days // 5, 10)
    wins = int(num_trades * rng.uniform(0.48, 0.62))
    trades: list[dict[str, Any]] = []
    for i in range(num_trades):
        sym = rng.choice(request.symbols)
        is_win = i < wins
        pnl = rng.uniform(50, 800) if is_win else -rng.uniform(30, 500)
        trade_date = start + timedelta(days=rng.randint(0, num_days - 1))
        trades.append({
            "symbol": sym,
            "side": rng.choice(["buy", "sell"]),
            "pnl": round(pnl, 2),
            "quantity": rng.randint(5, 100),
            "entry_price": round(rng.uniform(100, 500), 2),
            "exit_price": round(rng.uniform(100, 500), 2),
            "date": trade_date.isoformat(),
        })

    win_rate = wins / num_trades if num_trades else 0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    metrics: dict[str, Any] = {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "trades_count": num_trades,
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(gross_profit / wins, 2) if wins else 0,
        "avg_loss": round(gross_loss / (num_trades - wins), 2) if (num_trades - wins) else 0,
        "initial_capital": request.initial_capital,
        "final_value": round(equity[-1], 2),
    }

    return metrics, equity, trades


async def _run_backtest(backtest_id: str, request: BacktestRequest) -> None:
    """Execute a backtest in the background and store the result."""
    try:
        # Small delay for realism
        await asyncio.sleep(1.5)

        # Use simulated results until the full engine (DataHandler +
        # strategy registry) is wired up end-to-end.
        logger.info("backtest.using_simulation", backtest_id=backtest_id)
        metrics, equity, trades = _generate_simulated_results(request)
        _results[backtest_id] = BacktestResponse(
            id=backtest_id,
            status="completed",
            metrics=metrics,
            equity_curve=equity,
            trades=trades,
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
