"""Async pub/sub event bus for the trading system.

Provides event dataclasses and a concurrent-safe EventBus with optional
bounded replay history.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Coroutine

import structlog

from src.core.types import Order, OrderStatus, Position, Signal

logger = structlog.get_logger(__name__)

# Type alias for async event handlers
EventHandler = Callable[["Event"], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Base event. All domain events inherit from this."""

    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class MarketDataEvent(Event):
    """Emitted when new market data (bar, quote, trade) arrives."""

    event_type: str = "market_data"
    symbol: str = ""
    data_type: str = ""  # "bar", "quote", "trade"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalEvent(Event):
    """Emitted when a strategy produces a trading signal."""

    event_type: str = "signal"
    signal: Signal | None = None


@dataclass(frozen=True)
class OrderEvent(Event):
    """Emitted when an order is created, updated, or cancelled."""

    event_type: str = "order"
    order: Order | None = None
    previous_status: OrderStatus | None = None


@dataclass(frozen=True)
class FillEvent(Event):
    """Emitted when an order (partially or fully) fills."""

    event_type: str = "fill"
    order_id: str = ""
    symbol: str = ""
    filled_quantity: Decimal = Decimal("0")
    filled_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")


@dataclass(frozen=True)
class RiskBreachEvent(Event):
    """Emitted when a risk limit is breached."""

    event_type: str = "risk_breach"
    rule_name: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioUpdateEvent(Event):
    """Emitted when the portfolio state changes."""

    event_type: str = "portfolio_update"
    positions: list[Position] = field(default_factory=list)
    total_equity: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Async pub/sub event bus with bounded replay history.

    Handlers are invoked concurrently via ``asyncio.gather`` when an event
    is published.  A per-bus ``asyncio.Lock`` ensures that subscribe /
    unsubscribe / publish operations are safe across concurrent tasks.

    Parameters
    ----------
    max_history:
        Maximum number of events to retain for replay. Set to 0 to disable.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._lock = asyncio.Lock()
        self._history: deque[Event] = deque(maxlen=max_history if max_history > 0 else None)
        self._max_history = max_history

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* for events of *event_type*."""
        async with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)
            logger.debug("event_bus.subscribe", event_type=event_type, handler=handler.__qualname__)

    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove *handler* from *event_type* subscribers."""
        async with self._lock:
            handlers = self._subscribers.get(event_type, [])
            try:
                handlers.remove(handler)
                logger.debug("event_bus.unsubscribe", event_type=event_type, handler=handler.__qualname__)
            except ValueError:
                logger.warning(
                    "event_bus.unsubscribe_missing",
                    event_type=event_type,
                    handler=handler.__qualname__,
                )

    async def publish(self, event: Event) -> None:
        """Publish *event* to all registered handlers concurrently."""
        async with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))
            if self._max_history > 0:
                self._history.append(event)

        if not handlers:
            logger.debug("event_bus.publish_no_handlers", event_type=event.event_type)
            return

        logger.info(
            "event_bus.publish",
            event_type=event.event_type,
            handler_count=len(handlers),
        )

        results = await asyncio.gather(
            *(h(event) for h in handlers),
            return_exceptions=True,
        )

        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.error(
                    "event_bus.handler_error",
                    event_type=event.event_type,
                    handler=handler.__qualname__,
                    error=str(result),
                    exc_info=result,
                )

    async def replay(self, event_type: str | None = None) -> list[Event]:
        """Return buffered history, optionally filtered by *event_type*."""
        async with self._lock:
            if event_type is None:
                return list(self._history)
            return [e for e in self._history if e.event_type == event_type]
