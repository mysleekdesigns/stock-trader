"""Core domain types for the trading system.

Defines enums, value objects, and data transfer objects used across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import uuid


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class AssetClass(str, Enum):
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"


class TimeFrame(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class Signal:
    """A trading signal produced by a strategy."""

    symbol: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class Quote:
    """A point-in-time bid/ask quote for a symbol."""

    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    timestamp: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass
class Bar:
    """An OHLCV price bar for a given symbol and timeframe."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: TimeFrame
    vwap: float | None = None


@dataclass
class Position:
    """A current position held in a portfolio."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    opened_at: datetime = field(default_factory=datetime.utcnow)
    asset_class: AssetClass = AssetClass.US_EQUITY

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.avg_entry_price


@dataclass
class Order:
    """An order to be submitted to a broker."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
