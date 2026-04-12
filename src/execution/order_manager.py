"""Order lifecycle management.

The :class:`OrderManager` tracks every order from creation through terminal
state, enforces fill timeouts, and publishes :class:`OrderEvent` /
:class:`FillEvent` instances on the event bus.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog

from src.core.events import EventBus, FillEvent, OrderEvent
from src.core.exceptions import OrderCancellationError, OrderSubmissionError
from src.core.types import Order, OrderStatus
from src.execution.brokers.base import BrokerAdapter

logger = structlog.get_logger(__name__)

# Terminal states — orders in these states will never change again.
_TERMINAL_STATES = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
})


class OrderManager:
    """Manages the full order lifecycle: submit, track, timeout, cancel.

    Parameters
    ----------
    broker:
        The broker adapter used to send orders to the market.
    event_bus:
        Async event bus for publishing order/fill events.
    order_timeout_seconds:
        Maximum seconds an order may remain open before automatic cancellation.
        Set to ``0`` to disable timeout enforcement.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        event_bus: EventBus,
        order_timeout_seconds: int = 300,
    ) -> None:
        self._broker = broker
        self._event_bus = event_bus
        self._order_timeout = order_timeout_seconds

        self._orders: dict[str, Order] = {}
        self._timeout_tasks: dict[str, asyncio.Task[None]] = {}

        logger.info(
            "order_manager.init",
            timeout_seconds=order_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Submit *order* via the broker and begin tracking it.

        Publishes an :class:`OrderEvent` on submission and schedules a timeout
        task if configured.

        Raises
        ------
        OrderSubmissionError
            Propagated from the broker adapter.
        """
        log = logger.bind(order_id=order.id, symbol=order.symbol)
        log.info(
            "order_manager.submitting",
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=str(order.quantity),
        )

        previous_status = order.status

        try:
            order = await self._broker.submit_order(order)
        except OrderSubmissionError:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.utcnow()
            self._orders[order.id] = order
            await self._publish_order_event(order, previous_status)
            raise

        self._orders[order.id] = order
        await self._publish_order_event(order, previous_status)

        # Schedule timeout if the order is not already terminal
        if order.status not in _TERMINAL_STATES and self._order_timeout > 0:
            self._schedule_timeout(order.id)

        # If the broker filled it immediately (e.g. simulated market order),
        # emit a fill event right away.
        if order.status == OrderStatus.FILLED:
            await self._publish_fill_event(order)

        log.info(
            "order_manager.submitted",
            status=order.status.value,
        )
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Returns ``True`` if cancellation was accepted by the broker.

        Raises
        ------
        OrderCancellationError
            If the broker rejects the cancellation.
        """
        log = logger.bind(order_id=order_id)
        order = self._orders.get(order_id)
        if order is None:
            raise OrderCancellationError(
                f"Unknown order {order_id}",
                details={"order_id": order_id},
            )

        if order.status in _TERMINAL_STATES:
            log.warning(
                "order_manager.cancel_terminal",
                status=order.status.value,
            )
            return False

        previous_status = order.status
        result = await self._broker.cancel_order(order_id)

        if result:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            self._cancel_timeout(order_id)
            await self._publish_order_event(order, previous_status)
            log.info("order_manager.cancelled")

        return result

    async def get_open_orders(self) -> list[Order]:
        """Return all orders that are not yet in a terminal state."""
        return [
            o for o in self._orders.values()
            if o.status not in _TERMINAL_STATES
        ]

    def get_order(self, order_id: str) -> Order | None:
        """Look up a tracked order by ID."""
        return self._orders.get(order_id)

    @property
    def all_orders(self) -> dict[str, Order]:
        """Read-only view of the full order book (for inspection/testing)."""
        return dict(self._orders)

    # ------------------------------------------------------------------
    # Broker callback
    # ------------------------------------------------------------------

    async def on_order_update(self, order: Order) -> None:
        """Handle an order-status update pushed by the broker.

        This method is intended to be registered as an
        :pydata:`OrderUpdateCallback` via
        ``broker.subscribe_order_updates(order_manager.on_order_update)``.
        """
        log = logger.bind(order_id=order.id, symbol=order.symbol)
        existing = self._orders.get(order.id)

        if existing is None:
            # Order we didn't originate (e.g. manual trade) — track it anyway
            self._orders[order.id] = order
            log.info(
                "order_manager.tracking_external_order",
                status=order.status.value,
            )
            return

        previous_status = existing.status

        if order.status == previous_status:
            return  # Duplicate update — skip

        log.info(
            "order_manager.state_transition",
            from_status=previous_status.value,
            to_status=order.status.value,
        )

        # Merge updated fields
        existing.status = order.status
        existing.filled_quantity = order.filled_quantity
        existing.filled_avg_price = order.filled_avg_price
        existing.updated_at = datetime.utcnow()

        await self._publish_order_event(existing, previous_status)

        if order.status in _TERMINAL_STATES:
            self._cancel_timeout(order.id)

        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL):
            await self._publish_fill_event(existing)

    # ------------------------------------------------------------------
    # Timeout management
    # ------------------------------------------------------------------

    def _schedule_timeout(self, order_id: str) -> None:
        """Schedule automatic cancellation after the configured timeout."""
        self._cancel_timeout(order_id)  # Clear any existing task

        async def _timeout() -> None:
            await asyncio.sleep(self._order_timeout)
            order = self._orders.get(order_id)
            if order is not None and order.status not in _TERMINAL_STATES:
                logger.warning(
                    "order_manager.timeout",
                    order_id=order_id,
                    symbol=order.symbol,
                    elapsed_seconds=self._order_timeout,
                )
                try:
                    await self.cancel_order(order_id)
                except OrderCancellationError:
                    logger.exception(
                        "order_manager.timeout_cancel_failed",
                        order_id=order_id,
                    )

        task = asyncio.create_task(_timeout(), name=f"order-timeout-{order_id}")
        self._timeout_tasks[order_id] = task

    def _cancel_timeout(self, order_id: str) -> None:
        """Cancel a pending timeout task."""
        task = self._timeout_tasks.pop(order_id, None)
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_order_event(
        self,
        order: Order,
        previous_status: OrderStatus,
    ) -> None:
        """Publish an :class:`OrderEvent` to the event bus."""
        event = OrderEvent(
            order=order,
            previous_status=previous_status,
        )
        await self._event_bus.publish(event)

    async def _publish_fill_event(self, order: Order) -> None:
        """Publish a :class:`FillEvent` to the event bus."""
        event = FillEvent(
            order_id=order.id,
            symbol=order.symbol,
            filled_quantity=order.filled_quantity,
            filled_price=order.filled_avg_price or Decimal("0"),
        )
        await self._event_bus.publish(event)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cancel_all_open(self) -> int:
        """Best-effort cancellation of every open order.  Returns the count cancelled."""
        open_orders = await self.get_open_orders()
        cancelled = 0
        for order in open_orders:
            try:
                if await self.cancel_order(order.id):
                    cancelled += 1
            except OrderCancellationError:
                logger.exception(
                    "order_manager.cancel_all_failed",
                    order_id=order.id,
                )
        logger.info("order_manager.cancel_all_complete", cancelled=cancelled)
        return cancelled

    async def shutdown(self) -> None:
        """Cancel all timeouts and attempt to cancel open orders."""
        for task in self._timeout_tasks.values():
            task.cancel()
        self._timeout_tasks.clear()
        await self.cancel_all_open()
        logger.info("order_manager.shutdown")
