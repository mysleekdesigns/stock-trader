"""Opening Range Breakout (ORB) scanner and signal endpoints.

Provides endpoints to scan symbols for ORB setups, retrieve the current
ORB state (opening range levels, VWAP, signals), and configure the strategy.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.deps import get_event_bus
from src.core.events import EventBus, SignalEvent
from src.core.types import Bar, TimeFrame
from src.strategies.opening_range_breakout import ORBConfig, ORBStrategy

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/orb", tags=["orb"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ORBState(BaseModel):
    """Current ORB state for a symbol."""

    symbol: str
    date: str | None = None
    or_high: float | None = None
    or_low: float | None = None
    or_avg_volume: float | None = None
    or_bar_count: int = 0
    or_complete: bool = False
    breached: bool = False
    vwap: float | None = None


class ORBSignalResponse(BaseModel):
    """ORB breakout signal."""

    symbol: str
    direction: str
    strength: float
    confidence: float
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ORBScanResult(BaseModel):
    """Result from scanning a symbol for ORB setup."""

    symbol: str
    state: ORBState
    signal: ORBSignalResponse | None = None
    bars: list[dict[str, Any]] = Field(default_factory=list)


class ORBConfigRequest(BaseModel):
    """Request body for configuring the ORB scanner."""

    volume_multiplier: float = 1.5
    signal_cutoff: str = "11:30"


# ---------------------------------------------------------------------------
# Module-level strategy instances (one per scanned symbol)
# ---------------------------------------------------------------------------

_orb_instances: dict[str, ORBStrategy] = {}
_orb_signals: list[ORBSignalResponse] = []
_orb_config = ORBConfig()


def _get_or_create(symbol: str) -> ORBStrategy:
    """Get or create an ORB strategy instance for a symbol."""
    if symbol not in _orb_instances:
        _orb_instances[symbol] = ORBStrategy(symbol, config=_orb_config)
    return _orb_instances[symbol]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/state/{symbol}", response_model=ORBState)
async def get_orb_state(symbol: str) -> ORBState:
    """Return the current ORB state for a symbol."""
    strategy = _get_or_create(symbol.upper())
    snap = strategy.state_snapshot
    return ORBState(symbol=symbol.upper(), **snap)


@router.get("/signals", response_model=list[ORBSignalResponse])
async def get_orb_signals(
    limit: int = Query(50, ge=1, le=500),
) -> list[ORBSignalResponse]:
    """Return recent ORB breakout signals across all scanned symbols."""
    return _orb_signals[:limit]


@router.post("/scan/{symbol}", response_model=ORBScanResult)
async def scan_symbol(
    symbol: str,
    event_bus: EventBus = Depends(get_event_bus),
) -> ORBScanResult:
    """Scan a symbol for ORB setup using recent intraday data.

    Fetches 1-minute bars for the current session from the data provider
    and replays them through the ORB strategy.
    """
    symbol = symbol.upper()
    strategy = ORBStrategy(symbol, config=_orb_config)
    _orb_instances[symbol] = strategy

    # Fetch intraday bars from data provider
    bars_data: list[dict[str, Any]] = []
    signal_response: ORBSignalResponse | None = None

    try:
        from src.data.providers.alpaca_provider import AlpacaDataProvider
        from src.core.config import get_settings

        settings = get_settings()
        if settings.alpaca_api_key and settings.alpaca_secret_key:
            provider = AlpacaDataProvider(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                base_url=settings.alpaca_base_url,
            )
            await provider.connect()

            end = datetime.utcnow()
            start = end.replace(hour=14, minute=30, second=0, microsecond=0)
            if end.hour < 14:
                start -= timedelta(days=1)

            bars = await provider.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.MINUTE_1,
                start=start,
                end=end,
            )
            await provider.disconnect()

            for bar in bars:
                result = strategy.on_bar(bar)
                bars_data.append({
                    "time": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                })

                if result is not None:
                    signal_response = ORBSignalResponse(
                        symbol=result.symbol,
                        direction=result.direction.value,
                        strength=result.strength,
                        confidence=result.confidence,
                        timestamp=result.timestamp.isoformat(),
                        metadata=result.metadata,
                    )
                    _orb_signals.insert(0, signal_response)
                    if len(_orb_signals) > 500:
                        _orb_signals.pop()

                    await event_bus.publish(SignalEvent(signal=result))

    except ImportError:
        logger.warning("orb.scan_no_provider", symbol=symbol)
    except Exception as exc:
        logger.error("orb.scan_failed", symbol=symbol, error=str(exc))

    # If no live data, return simulated state for demo purposes
    if not bars_data:
        bars_data = _generate_demo_bars(symbol)
        for bar_dict in bars_data:
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.fromisoformat(bar_dict["time"]),
                open=bar_dict["open"],
                high=bar_dict["high"],
                low=bar_dict["low"],
                close=bar_dict["close"],
                volume=bar_dict["volume"],
                timeframe=TimeFrame.MINUTE_1,
            )
            result = strategy.on_bar(bar)
            if result is not None:
                signal_response = ORBSignalResponse(
                    symbol=result.symbol,
                    direction=result.direction.value,
                    strength=result.strength,
                    confidence=result.confidence,
                    timestamp=result.timestamp.isoformat(),
                    metadata=result.metadata,
                )
                _orb_signals.insert(0, signal_response)

    snap = strategy.state_snapshot
    state = ORBState(symbol=symbol, **snap)

    return ORBScanResult(
        symbol=symbol,
        state=state,
        signal=signal_response,
        bars=bars_data,
    )


@router.post("/config", response_model=dict[str, Any])
async def update_config(body: ORBConfigRequest) -> dict[str, Any]:
    """Update the ORB scanner configuration."""
    global _orb_config  # noqa: PLW0603

    parts = body.signal_cutoff.split(":")
    cutoff_hour = int(parts[0])
    cutoff_minute = int(parts[1]) if len(parts) > 1 else 0

    from datetime import time as dt_time

    _orb_config = ORBConfig(
        volume_multiplier=body.volume_multiplier,
        signal_cutoff=dt_time(cutoff_hour, cutoff_minute),
    )

    # Reset all instances to pick up new config
    _orb_instances.clear()

    logger.info(
        "orb.config_updated",
        volume_multiplier=body.volume_multiplier,
        signal_cutoff=body.signal_cutoff,
    )

    return {
        "volume_multiplier": _orb_config.volume_multiplier,
        "or_start": _orb_config.or_start.strftime("%H:%M"),
        "or_end": _orb_config.or_end.strftime("%H:%M"),
        "signal_cutoff": _orb_config.signal_cutoff.strftime("%H:%M"),
    }


@router.get("/config", response_model=dict[str, Any])
async def get_config() -> dict[str, Any]:
    """Return the current ORB scanner configuration."""
    return {
        "volume_multiplier": _orb_config.volume_multiplier,
        "or_start": _orb_config.or_start.strftime("%H:%M"),
        "or_end": _orb_config.or_end.strftime("%H:%M"),
        "signal_cutoff": _orb_config.signal_cutoff.strftime("%H:%M"),
    }


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------


def _generate_demo_bars(symbol: str) -> list[dict[str, Any]]:
    """Generate realistic demo 1-minute bars for an ORB session.

    This creates a session where the opening range is established,
    then a clean breakout occurs with volume confirmation.
    """
    import random

    random.seed(hash(symbol + datetime.utcnow().strftime("%Y-%m-%d")))

    base_price = 150.0 + random.uniform(-30, 30)
    base_volume = 50000 + random.randint(-20000, 20000)
    bars: list[dict[str, Any]] = []

    today = datetime.utcnow().date()
    # Use 14:30 UTC = 9:30 AM EST
    session_start = datetime(today.year, today.month, today.day, 14, 30)

    price = base_price
    or_high = 0.0

    for i in range(120):  # 2 hours of 1-min bars
        ts = session_start + timedelta(minutes=i)
        minute = i

        # Opening range: first 30 minutes — range-bound
        if minute < 30:
            change = random.uniform(-0.15, 0.18)
            vol_mult = random.uniform(0.8, 1.5)
        # Post-OR breakout zone (30-45 min): volume surge + breakout
        elif minute < 45:
            change = random.uniform(0.05, 0.35)
            vol_mult = random.uniform(1.5, 3.0)
        # Continuation (45+ min)
        else:
            change = random.uniform(-0.1, 0.15)
            vol_mult = random.uniform(0.6, 1.2)

        open_p = price
        close_p = price + change
        high_p = max(open_p, close_p) + abs(random.uniform(0, 0.1))
        low_p = min(open_p, close_p) - abs(random.uniform(0, 0.1))
        vol = int(base_volume * vol_mult)

        bars.append({
            "time": ts.isoformat(),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": vol,
        })

        price = close_p

        if minute < 30:
            or_high = max(or_high, high_p)

    return bars
