"""Shared pytest fixtures for the trading system test suite.

Provides synthetic market data, event bus instances, and sample domain objects
so that unit tests can run without external dependencies (databases, APIs).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.core.types import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalDirection,
    TimeFrame,
)


# ---------------------------------------------------------------------------
# Synthetic OHLCV data (random walk)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bars():
    """Return a factory that generates a list of synthetic Bar objects.

    Usage in tests::

        bars = sample_bars()            # 252 bars for SPY
        bars = sample_bars("AAPL", 60)  # 60 bars for AAPL
    """

    def _factory(symbol: str = "SPY", n: int = 252, seed: int = 42) -> list[Bar]:
        rng = np.random.RandomState(seed)
        base_price = 100.0
        base_volume = 1_000_000
        start_date = datetime(2023, 1, 3)

        prices = [base_price]
        for _ in range(n - 1):
            ret = rng.normal(0.0005, 0.015)
            prices.append(prices[-1] * (1 + ret))

        bars: list[Bar] = []
        for i in range(n):
            close = prices[i]
            # Generate realistic OHLC from close
            daily_range = abs(rng.normal(0, 0.01)) * close
            high = close + rng.uniform(0, daily_range)
            low = close - rng.uniform(0, daily_range)
            open_ = close + rng.normal(0, daily_range * 0.3)

            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_, close)
            low = min(low, open_, close)
            low = max(low, 0.01)  # no negative prices

            volume = int(base_volume * rng.uniform(0.5, 2.0))
            timestamp = start_date + timedelta(days=i)

            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=round(open_, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close, 4),
                    volume=volume,
                    timeframe=TimeFrame.DAILY,
                )
            )

        return bars

    return _factory


@pytest.fixture
def sample_df():
    """Return a factory that generates a synthetic OHLCV DataFrame.

    The DataFrame has a DatetimeIndex and columns: open, high, low, close, volume.

    Usage in tests::

        df = sample_df()              # 252 rows
        df = sample_df(n=60, seed=7)  # 60 rows with different seed
    """

    def _factory(n: int = 252, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        base_price = 100.0
        base_volume = 1_000_000
        start_date = datetime(2023, 1, 3)

        prices = [base_price]
        for _ in range(n - 1):
            ret = rng.normal(0.0005, 0.015)
            prices.append(prices[-1] * (1 + ret))

        records = []
        timestamps = []
        for i in range(n):
            close = prices[i]
            daily_range = abs(rng.normal(0, 0.01)) * close
            high = close + rng.uniform(0, daily_range)
            low = close - rng.uniform(0, daily_range)
            open_ = close + rng.normal(0, daily_range * 0.3)
            high = max(high, open_, close)
            low = min(low, open_, close)
            low = max(low, 0.01)
            volume = int(base_volume * rng.uniform(0.5, 2.0))

            records.append(
                {
                    "open": round(open_, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": volume,
                }
            )
            timestamps.append(start_date + timedelta(days=i))

        df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps))
        df.index.name = "timestamp"
        return df

    return _factory


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    """Return a fresh EventBus instance."""
    from src.core.events import EventBus

    return EventBus(max_history=100)


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signal():
    """Return a factory that produces Signal objects."""

    def _factory(
        symbol: str = "SPY",
        direction: SignalDirection = SignalDirection.LONG,
        strength: float = 0.8,
        confidence: float = 0.7,
        strategy_name: str = "test_strategy",
    ) -> Signal:
        return Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            strategy_name=strategy_name,
        )

    return _factory


@pytest.fixture
def sample_order():
    """Return a factory that produces Order objects."""

    def _factory(
        symbol: str = "SPY",
        side: OrderSide = OrderSide.BUY,
        quantity: Decimal = Decimal("100"),
        order_type: OrderType = OrderType.MARKET,
        status: OrderStatus = OrderStatus.PENDING,
    ) -> Order:
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=status,
            strategy_name="test_strategy",
        )

    return _factory


@pytest.fixture
def sample_position():
    """Return a factory that produces Position objects."""

    def _factory(
        symbol: str = "SPY",
        side: OrderSide = OrderSide.BUY,
        quantity: Decimal = Decimal("100"),
        avg_entry_price: Decimal = Decimal("450.00"),
        current_price: Decimal = Decimal("455.00"),
    ) -> Position:
        unrealized = (current_price - avg_entry_price) * quantity
        if side == OrderSide.SELL:
            unrealized = (avg_entry_price - current_price) * quantity
        return Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            avg_entry_price=avg_entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized,
        )

    return _factory


@pytest.fixture
def sample_portfolio():
    """Return a factory that creates a Portfolio with initial cash.

    Tries to import the real Portfolio class; falls back to a dict-based
    representation if the module is not yet available.
    """

    def _factory(initial_cash: float = 100_000.0) -> object:
        try:
            from src.risk.portfolio import Portfolio

            return Portfolio(
                cash=Decimal(str(initial_cash)),
                initial_value=Decimal(str(initial_cash)),
            )
        except ImportError:
            # Fallback for when the risk module is incomplete
            return {
                "cash": Decimal(str(initial_cash)),
                "initial_value": Decimal(str(initial_cash)),
                "positions": {},
            }

    return _factory
