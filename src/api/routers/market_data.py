"""Market data endpoints — historical bars from TimescaleDB."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api", tags=["market_data"])


@router.get("/market-data/bars")
async def get_bars(
    symbol: str = Query(..., description="Ticker symbol"),
    timeframe: str = Query("1d", description="Bar timeframe (1m, 5m, 15m, 1h, 1d, 1w)"),
    days: int = Query(30, description="Number of days of history"),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Return historical bars for a symbol from TimescaleDB."""
    if session is None:
        return {"bars": [], "error": "Database not available"}

    from sqlalchemy import select, and_
    from src.data.storage.timeseries_store import OHLCVRecord

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    stmt = (
        select(OHLCVRecord)
        .where(
            and_(
                OHLCVRecord.symbol == symbol,
                OHLCVRecord.timeframe == timeframe,
                OHLCVRecord.timestamp >= start,
                OHLCVRecord.timestamp <= end,
            )
        )
        .order_by(OHLCVRecord.timestamp)
    )

    result = await session.execute(stmt)
    records = result.scalars().all()

    bars = [
        {
            "time": r.timestamp.isoformat(),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in records
    ]

    return {"symbol": symbol, "timeframe": timeframe, "bars": bars}


@router.get("/market-data/symbols")
async def get_available_symbols(
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Return a list of symbols that have data in TimescaleDB."""
    if session is None:
        return {"symbols": []}

    from sqlalchemy import select, distinct, func
    from src.data.storage.timeseries_store import OHLCVRecord

    stmt = (
        select(
            OHLCVRecord.symbol,
            func.count().label("bar_count"),
            func.min(OHLCVRecord.timestamp).label("first_bar"),
            func.max(OHLCVRecord.timestamp).label("last_bar"),
        )
        .group_by(OHLCVRecord.symbol)
        .order_by(OHLCVRecord.symbol)
    )

    result = await session.execute(stmt)
    rows = result.all()

    symbols = [
        {
            "symbol": row.symbol,
            "barCount": row.bar_count,
            "firstBar": row.first_bar.isoformat() if row.first_bar else None,
            "lastBar": row.last_bar.isoformat() if row.last_bar else None,
        }
        for row in rows
    ]

    return {"symbols": symbols}
