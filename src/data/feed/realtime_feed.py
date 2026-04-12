"""WebSocket multiplexer that fans-in real-time data from multiple providers.

The :class:`RealtimeFeed` connects to every configured
:class:`DataProviderBase`, subscribes to bars and trades for the requested
symbols, normalises incoming data, and publishes :class:`MarketDataEvent`
instances on the shared :class:`EventBus`.

Reconnection uses truncated exponential back-off per provider.  Provider
health (connected/disconnected) and latency are tracked for observability.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.core.events import EventBus, MarketDataEvent
from src.core.exceptions import DataProviderError
from src.core.types import Bar, TimeFrame
from src.data.providers.base import DataProviderBase

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider health tracking
# ---------------------------------------------------------------------------

_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0
_BACKOFF_FACTOR = 2.0


@dataclass
class _ProviderHealth:
    """Mutable health record for a single provider."""

    provider_name: str
    connected: bool = False
    last_data_at: float | None = None
    latency_ms: float = 0.0
    reconnect_attempts: int = 0
    last_error: str | None = None
    _backoff: float = field(default=_INITIAL_BACKOFF_S, repr=False)

    def record_data(self) -> None:
        now = time.monotonic()
        if self.last_data_at is not None:
            self.latency_ms = (now - self.last_data_at) * 1000.0
        self.last_data_at = now

    def record_connect(self) -> None:
        self.connected = True
        self.reconnect_attempts = 0
        self._backoff = _INITIAL_BACKOFF_S

    def record_disconnect(self, error: str | None = None) -> None:
        self.connected = False
        self.last_error = error

    @property
    def next_backoff(self) -> float:
        return self._backoff

    def advance_backoff(self) -> float:
        delay = self._backoff
        self._backoff = min(self._backoff * _BACKOFF_FACTOR, _MAX_BACKOFF_S)
        self.reconnect_attempts += 1
        return delay


# ---------------------------------------------------------------------------
# RealtimeFeed
# ---------------------------------------------------------------------------


class RealtimeFeed:
    """Multiplexes real-time bar and trade data from multiple providers.

    Parameters
    ----------
    providers:
        Ordered list of :class:`DataProviderBase` instances.  The first
        provider that is healthy for a given symbol wins; the rest serve as
        fallbacks.
    event_bus:
        Shared event bus where :class:`MarketDataEvent` will be published.
    """

    def __init__(
        self,
        providers: list[DataProviderBase],
        event_bus: EventBus,
    ) -> None:
        if not providers:
            raise ValueError("At least one data provider is required")

        self._providers = providers
        self._event_bus = event_bus
        self._running = False

        # provider class-name -> health
        self._health: dict[str, _ProviderHealth] = {
            p.__class__.__name__: _ProviderHealth(provider_name=p.__class__.__name__)
            for p in providers
        }

        # Background reconnection tasks
        self._reconnect_tasks: dict[str, asyncio.Task[None]] = {}

        # Track symbols/timeframes for reconnection resubscription
        self._symbols: list[str] = []
        self._timeframes: list[TimeFrame] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(
        self,
        symbols: list[str],
        timeframes: list[TimeFrame],
    ) -> None:
        """Connect to all providers and subscribe to bars/trades.

        Subscriptions are attempted on every provider.  If a provider fails
        to connect, a background reconnection loop is spawned for it.
        """
        self._symbols = list(symbols)
        self._timeframes = list(timeframes)
        self._running = True

        logger.info(
            "realtime_feed.starting",
            provider_count=len(self._providers),
            symbols=symbols,
            timeframes=[tf.value for tf in timeframes],
        )

        connect_tasks = [
            self._connect_provider(provider)
            for provider in self._providers
        ]
        await asyncio.gather(*connect_tasks, return_exceptions=True)

        logger.info(
            "realtime_feed.started",
            healthy=[
                name for name, h in self._health.items() if h.connected
            ],
        )

    async def stop(self) -> None:
        """Disconnect all providers and cancel reconnection loops."""
        self._running = False

        # Cancel pending reconnection tasks
        for name, task in self._reconnect_tasks.items():
            task.cancel()
            logger.debug("realtime_feed.cancel_reconnect", provider=name)

        cancel_results = await asyncio.gather(
            *self._reconnect_tasks.values(), return_exceptions=True
        )
        for name, result in zip(self._reconnect_tasks, cancel_results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.warning(
                    "realtime_feed.reconnect_cancel_error",
                    provider=name,
                    error=str(result),
                )
        self._reconnect_tasks.clear()

        # Disconnect providers
        for provider in self._providers:
            name = provider.__class__.__name__
            try:
                await provider.disconnect()
                health = self._health[name]
                health.record_disconnect()
                logger.info("realtime_feed.provider_disconnected", provider=name)
            except Exception as exc:
                logger.error(
                    "realtime_feed.disconnect_error",
                    provider=name,
                    error=str(exc),
                    exc_info=exc,
                )

        logger.info("realtime_feed.stopped")

    def get_provider_health(self) -> dict[str, _ProviderHealth]:
        """Return a snapshot of provider health records."""
        return dict(self._health)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _connect_provider(self, provider: DataProviderBase) -> None:
        """Connect a single provider and subscribe to data feeds."""
        name = provider.__class__.__name__
        health = self._health[name]

        try:
            await provider.connect()
            health.record_connect()
            logger.info("realtime_feed.provider_connected", provider=name)

            await self._subscribe_provider(provider)

        except Exception as exc:
            health.record_disconnect(error=str(exc))
            logger.error(
                "realtime_feed.provider_connect_failed",
                provider=name,
                error=str(exc),
                exc_info=exc,
            )
            self._schedule_reconnect(provider)

    async def _subscribe_provider(self, provider: DataProviderBase) -> None:
        """Subscribe to bars (for each timeframe) and trades on *provider*."""
        name = provider.__class__.__name__

        for tf in self._timeframes:
            try:
                await provider.subscribe_bars(
                    self._symbols,
                    tf,
                    callback=self._make_bar_callback(name),
                )
                logger.debug(
                    "realtime_feed.subscribed_bars",
                    provider=name,
                    timeframe=tf.value,
                    symbols=self._symbols,
                )
            except Exception as exc:
                logger.error(
                    "realtime_feed.subscribe_bars_error",
                    provider=name,
                    timeframe=tf.value,
                    error=str(exc),
                    exc_info=exc,
                )

        try:
            await provider.subscribe_trades(
                self._symbols,
                callback=self._make_trade_callback(name),
            )
            logger.debug(
                "realtime_feed.subscribed_trades",
                provider=name,
                symbols=self._symbols,
            )
        except Exception as exc:
            logger.error(
                "realtime_feed.subscribe_trades_error",
                provider=name,
                error=str(exc),
                exc_info=exc,
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _make_bar_callback(self, provider_name: str):
        """Return an async callback that normalises a bar and publishes it."""

        async def _on_bar(bar: Bar) -> None:
            if not self._running:
                return

            health = self._health[provider_name]
            health.record_data()

            event = MarketDataEvent(
                symbol=bar.symbol,
                data_type="bar",
                payload={
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "timeframe": bar.timeframe.value,
                    "vwap": bar.vwap,
                    "provider": provider_name,
                },
            )

            logger.debug(
                "realtime_feed.bar_received",
                provider=provider_name,
                symbol=bar.symbol,
                timeframe=bar.timeframe.value,
                close=bar.close,
            )

            await self._event_bus.publish(event)

        return _on_bar

    def _make_trade_callback(self, provider_name: str):
        """Return an async callback that normalises a trade and publishes it."""

        async def _on_trade(trade: dict[str, Any]) -> None:
            if not self._running:
                return

            health = self._health[provider_name]
            health.record_data()

            symbol = trade.get("symbol", "")
            event = MarketDataEvent(
                symbol=symbol,
                data_type="trade",
                payload={
                    **trade,
                    "provider": provider_name,
                },
            )

            logger.debug(
                "realtime_feed.trade_received",
                provider=provider_name,
                symbol=symbol,
                price=trade.get("price"),
                volume=trade.get("volume"),
            )

            await self._event_bus.publish(event)

        return _on_trade

    # ------------------------------------------------------------------
    # Reconnection with exponential back-off
    # ------------------------------------------------------------------

    def _schedule_reconnect(self, provider: DataProviderBase) -> None:
        """Spawn a background task that will attempt to reconnect *provider*."""
        name = provider.__class__.__name__

        if name in self._reconnect_tasks and not self._reconnect_tasks[name].done():
            logger.debug("realtime_feed.reconnect_already_scheduled", provider=name)
            return

        task = asyncio.create_task(
            self._reconnect_loop(provider),
            name=f"reconnect-{name}",
        )
        self._reconnect_tasks[name] = task

    async def _reconnect_loop(self, provider: DataProviderBase) -> None:
        """Keep trying to reconnect *provider* with exponential back-off."""
        name = provider.__class__.__name__
        health = self._health[name]

        while self._running and not health.connected:
            delay = health.advance_backoff()
            logger.info(
                "realtime_feed.reconnecting",
                provider=name,
                attempt=health.reconnect_attempts,
                delay_seconds=round(delay, 1),
            )
            await asyncio.sleep(delay)

            if not self._running:
                break

            try:
                await provider.connect()
                health.record_connect()
                logger.info(
                    "realtime_feed.reconnected",
                    provider=name,
                    after_attempts=health.reconnect_attempts,
                )
                await self._subscribe_provider(provider)
            except Exception as exc:
                health.record_disconnect(error=str(exc))
                logger.warning(
                    "realtime_feed.reconnect_failed",
                    provider=name,
                    attempt=health.reconnect_attempts,
                    error=str(exc),
                )
