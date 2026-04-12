"""Abstract base class for all data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import structlog

from src.core.types import Bar, Quote, TimeFrame

logger = structlog.get_logger(__name__)

# Type alias for streaming callbacks.
BarCallback = Callable[[Bar], Awaitable[None]]
TradeCallback = Callable[[dict[str, Any]], Awaitable[None]]


class DataProviderBase(ABC):
    """Contract that every data-provider adapter must fulfil.

    Lifecycle
    ---------
    1. Instantiate the provider.
    2. Call ``await provider.connect()`` to open any persistent connections.
    3. Use ``get_bars`` / ``get_quote`` / ``subscribe_*`` as needed.
    4. Call ``await provider.disconnect()`` for graceful teardown.
    """

    # ── lifecycle ───────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open connections, authenticate, and prepare the provider."""
        logger.info("data_provider.connect", provider=self.__class__.__name__)

    async def disconnect(self) -> None:
        """Tear down connections and release resources."""
        logger.info("data_provider.disconnect", provider=self.__class__.__name__)

    # ── historical data ─────────────────────────────────────────────────

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Bar]:
        """Return OHLCV bars for *symbol* in the given time range."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest bid/ask quote for *symbol*."""

    # ── streaming / real-time ───────────────────────────────────────────

    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        callback: BarCallback,
    ) -> None:
        """Subscribe to real-time bar updates."""

    @abstractmethod
    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        """Subscribe to real-time trade updates."""
