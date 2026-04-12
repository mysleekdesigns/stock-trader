"""Dashboard endpoint — aggregated portfolio view."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session, get_portfolio
from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api", tags=["dashboard"])


async def _build_equity_curve(session: AsyncSession | None) -> list[dict[str, Any]]:
    """Build equity curve from portfolio_snapshots, falling back to OHLCV SPY data."""
    if session is None:
        return []

    from sqlalchemy import select, and_, func
    from src.data.storage.timeseries_store import PortfolioSnapshot, OHLCVRecord

    # Try portfolio snapshots first
    stmt = (
        select(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.timestamp)
        .limit(500)
    )
    result = await session.execute(stmt)
    snapshots = result.scalars().all()

    if snapshots:
        return [
            {"time": s.timestamp.strftime("%Y-%m-%d"), "value": s.total_value}
            for s in snapshots
        ]

    # Fallback: use SPY close prices scaled to portfolio value as a proxy
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    stmt = (
        select(OHLCVRecord)
        .where(
            and_(
                OHLCVRecord.symbol == "SPY",
                OHLCVRecord.timeframe == "1d",
                OHLCVRecord.timestamp >= start,
            )
        )
        .order_by(OHLCVRecord.timestamp)
    )
    result = await session.execute(stmt)
    bars = result.scalars().all()

    if not bars:
        return []

    # Normalize SPY closes to portfolio-scale values
    base_price = bars[0].close
    return [
        {
            "time": b.timestamp.strftime("%Y-%m-%d"),
            "value": round(1_000_000 * (b.close / base_price), 2),
        }
        for b in bars
    ]


@router.get("/dashboard")
async def get_dashboard(
    portfolio: Portfolio = Depends(get_portfolio),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Return an aggregated dashboard payload matching the frontend schema."""
    total_value = portfolio.total_value

    # Portfolio summary (matches frontend DashboardData.portfolio)
    portfolio_data = {
        "totalValue": float(total_value),
        "cash": float(portfolio.cash),
        "unrealizedPnl": float(portfolio.unrealized_pnl),
        "drawdown": float(portfolio.drawdown),
        "dailyPnl": float(portfolio.daily_pnl),
        "weeklyPnl": 0.0,
        "monthlyPnl": 0.0,
    }

    # Positions (matches frontend Position interface)
    positions: list[dict[str, Any]] = []
    for pos in portfolio.positions.values():
        pnl = float(pos.unrealized_pnl)
        cost = float(pos.cost_basis)
        pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        positions.append({
            "symbol": pos.symbol,
            "side": "long" if pos.side.value == "buy" else "short",
            "quantity": float(pos.quantity),
            "avgPrice": float(pos.avg_entry_price),
            "currentPrice": float(pos.current_price),
            "marketValue": float(pos.market_value),
            "unrealizedPnl": pnl,
            "unrealizedPnlPct": round(pnl_pct, 2),
        })

    # Equity curve from DB
    equity_curve = await _build_equity_curve(session)

    return {
        "portfolio": portfolio_data,
        "positions": positions,
        "recentOrders": [],
        "equityCurve": equity_curve,
    }
