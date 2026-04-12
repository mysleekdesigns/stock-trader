"""Abstract base class for broker adapters.

Every broker integration (Alpaca, Interactive Brokers, simulated, etc.) must
implement the :class:`BrokerAdapter` interface so the execution engine can
submit and manage orders in a provider-agnostic way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from decimal import Decimal
from typing import Any

import structlog

from src.core.types import Order, Position

logger = structlog.get_logger(__name__)

# Callback invoked when the broker pushes an order status update.
OrderUpdateCallback = Callable[[Order], Coroutine[Any, Any, None]]


class BrokerAdapter(ABC):
    """Contract that every broker adapter must fulfil.

    Lifecycle
    ---------
    1. Instantiate the adapter with configuration / credentials.
    2. ``await adapter.connect()`` to open connections.
    3. Use ``submit_order``, ``cancel_order``, etc.
    4. ``await adapter.disconnect()`` for graceful teardown.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connections and authenticate with the broker."""
        logger.info("broker.connect", broker=self.__class__.__name__)

    async def disconnect(self) -> None:
        """Close connections and release resources."""
        logger.info("broker.disconnect", broker=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit *order* to the broker and return it with updated status/ID.

        Raises
        ------
        OrderSubmissionError
            If the broker rejects the request.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.  Returns ``True`` if cancellation was accepted."""

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch all open positions from the broker."""

    @abstractmethod
    async def get_account(self) -> dict[str, Any]:
        """Fetch account summary (cash, buying power, equity, etc.)."""

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @abstractmethod
    async def subscribe_order_updates(self, callback: OrderUpdateCallback) -> None:
        """Register *callback* to receive real-time order status updates.

        The broker adapter is responsible for mapping broker-native events to
        :class:`Order` instances before invoking the callback.
        """
