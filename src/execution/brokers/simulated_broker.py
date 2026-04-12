"""In-memory simulated broker for local development and backtesting.

Provides instant market-order fills (with configurable slippage) and
pending-order management for limit/stop orders that trigger on subsequent
bar data via :meth:`tick`.
"""

from __future__ import annotations

import random
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog

from src.core.exceptions import OrderCancellationError, OrderSubmissionError
from src.core.types import Bar, Order, OrderSide, OrderStatus, OrderType, Position
from src.execution.brokers.base import BrokerAdapter, OrderUpdateCallback

logger = structlog.get_logger(__name__)


class SimulatedBroker(BrokerAdapter):
    """Broker that fills orders in-memory without touching any external API.

    Parameters
    ----------
    initial_cash:
        Starting cash balance.
    slippage_bps:
        Random slippage applied to market fills, in basis points.  A value of
        ``5`` means up to 0.05 % adverse slippage per fill.
    commission_per_share:
        Flat per-share commission charged on each fill.
    """

    def __init__(
        self,
        initial_cash: Decimal = Decimal("100000"),
        slippage_bps: int = 5,
        commission_per_share: Decimal = Decimal("0"),
    ) -> None:
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._slippage_bps = slippage_bps
        self._commission_per_share = commission_per_share

        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._pending_orders: dict[str, Order] = {}
        self._update_callbacks: list[OrderUpdateCallback] = []
        self._last_prices: dict[str, Decimal] = {}

        logger.info(
            "simulated_broker.init",
            initial_cash=str(initial_cash),
            slippage_bps=slippage_bps,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        logger.info("simulated_broker.connect")

    async def disconnect(self) -> None:
        logger.info(
            "simulated_broker.disconnect",
            open_orders=len(self._pending_orders),
            positions=len(self._positions),
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        if order.quantity <= Decimal("0"):
            raise OrderSubmissionError(
                "Order quantity must be positive",
                details={"order_id": order.id, "quantity": str(order.quantity)},
            )

        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.utcnow()
        self._orders[order.id] = order

        logger.info(
            "simulated_broker.order_submitted",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=str(order.quantity),
        )

        # Market orders fill immediately at last known price.
        if order.order_type == OrderType.MARKET:
            last_price = self._last_prices.get(order.symbol)
            if last_price is not None:
                fill_price = self._apply_slippage(last_price, order.side)
                await self._fill_order(order, fill_price)
            else:
                # No price available yet — queue it to fill on next tick.
                self._pending_orders[order.id] = order
                logger.warning(
                    "simulated_broker.no_price_for_market_order",
                    order_id=order.id,
                    symbol=order.symbol,
                )
        else:
            # Limit / stop / stop-limit orders wait for price triggers.
            self._pending_orders[order.id] = order

        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self._pending_orders.pop(order_id, None)
        if order is None:
            stored = self._orders.get(order_id)
            if stored and stored.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
                raise OrderCancellationError(
                    f"Order {order_id} already {stored.status.value}",
                    details={"order_id": order_id},
                )
            raise OrderCancellationError(
                f"Order {order_id} not found in pending orders",
                details={"order_id": order_id},
            )

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        logger.info("simulated_broker.order_cancelled", order_id=order_id)
        await self._notify_update(order)
        return True

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_account(self) -> dict[str, Any]:
        position_value = sum(
            p.market_value for p in self._positions.values()
        )
        equity = self._cash + position_value
        return {
            "cash": str(self._cash),
            "equity": str(equity),
            "buying_power": str(self._cash),
            "initial_cash": str(self._initial_cash),
            "position_count": len(self._positions),
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def subscribe_order_updates(self, callback: OrderUpdateCallback) -> None:
        self._update_callbacks.append(callback)
        logger.info("simulated_broker.subscribed_order_updates")

    # ------------------------------------------------------------------
    # Tick processing (simulated price feed)
    # ------------------------------------------------------------------

    async def tick(self, bar: Bar) -> None:
        """Process a price bar, potentially filling pending limit/stop orders.

        Call this method for every new bar to simulate order matching.
        """
        price = Decimal(str(bar.close))
        self._last_prices[bar.symbol] = price

        # Update position marks
        if bar.symbol in self._positions:
            pos = self._positions[bar.symbol]
            pos.current_price = price
            if pos.side == OrderSide.BUY:
                pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.avg_entry_price - price) * pos.quantity

        # Check pending orders
        to_fill: list[tuple[Order, Decimal]] = []
        for order in list(self._pending_orders.values()):
            if order.symbol != bar.symbol:
                continue

            fill_price = self._check_trigger(order, bar)
            if fill_price is not None:
                to_fill.append((order, fill_price))

        for order, fill_price in to_fill:
            await self._fill_order(order, fill_price)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trigger(self, order: Order, bar: Bar) -> Decimal | None:
        """Return the fill price if *order* triggers on *bar*, else ``None``."""
        high = Decimal(str(bar.high))
        low = Decimal(str(bar.low))
        close = Decimal(str(bar.close))

        if order.order_type == OrderType.MARKET:
            return self._apply_slippage(close, order.side)

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and low <= order.limit_price:
                return min(order.limit_price, close)
            if order.side == OrderSide.SELL and high >= order.limit_price:
                return max(order.limit_price, close)

        if order.order_type == OrderType.STOP and order.stop_price is not None:
            if order.side == OrderSide.BUY and high >= order.stop_price:
                return self._apply_slippage(
                    max(order.stop_price, close), order.side,
                )
            if order.side == OrderSide.SELL and low <= order.stop_price:
                return self._apply_slippage(
                    min(order.stop_price, close), order.side,
                )

        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is not None and order.limit_price is not None:
                # Stop triggered?
                stop_triggered = False
                if order.side == OrderSide.BUY and high >= order.stop_price:
                    stop_triggered = True
                elif order.side == OrderSide.SELL and low <= order.stop_price:
                    stop_triggered = True

                if stop_triggered:
                    # Then check limit
                    if order.side == OrderSide.BUY and low <= order.limit_price:
                        return min(order.limit_price, close)
                    if order.side == OrderSide.SELL and high >= order.limit_price:
                        return max(order.limit_price, close)

        return None

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        """Apply random adverse slippage to a fill price."""
        if self._slippage_bps <= 0:
            return price
        slip_frac = Decimal(str(random.uniform(0, self._slippage_bps / 10_000)))
        if side == OrderSide.BUY:
            return price * (Decimal("1") + slip_frac)
        return price * (Decimal("1") - slip_frac)

    async def _fill_order(self, order: Order, fill_price: Decimal) -> None:
        """Execute a fill: update order, positions, and cash."""
        self._pending_orders.pop(order.id, None)

        order.filled_quantity = order.quantity
        order.filled_avg_price = fill_price
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.utcnow()

        commission = self._commission_per_share * order.quantity
        fill_cost = fill_price * order.quantity

        if order.side == OrderSide.BUY:
            self._cash -= fill_cost + commission
        else:
            self._cash += fill_cost - commission

        self._update_position(order, fill_price)

        logger.info(
            "simulated_broker.order_filled",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=str(order.quantity),
            fill_price=str(fill_price),
            commission=str(commission),
            cash_remaining=str(self._cash),
        )

        await self._notify_update(order)

    def _update_position(self, order: Order, fill_price: Decimal) -> None:
        """Update the position book after a fill."""
        existing = self._positions.get(order.symbol)

        if existing is None:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                avg_entry_price=fill_price,
                current_price=fill_price,
            )
            return

        # Same side: increase position
        if existing.side == order.side:
            total_qty = existing.quantity + order.quantity
            if total_qty > Decimal("0"):
                existing.avg_entry_price = (
                    existing.avg_entry_price * existing.quantity
                    + fill_price * order.quantity
                ) / total_qty
            existing.quantity = total_qty
            existing.current_price = fill_price
        else:
            # Opposite side: reduce or flip position
            if order.quantity >= existing.quantity:
                remainder = order.quantity - existing.quantity
                if remainder > Decimal("0"):
                    self._positions[order.symbol] = Position(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=remainder,
                        avg_entry_price=fill_price,
                        current_price=fill_price,
                    )
                else:
                    del self._positions[order.symbol]
            else:
                existing.quantity -= order.quantity
                existing.current_price = fill_price

        # Recompute unrealized P&L
        pos = self._positions.get(order.symbol)
        if pos is not None:
            if pos.side == OrderSide.BUY:
                pos.unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.avg_entry_price - pos.current_price) * pos.quantity

    async def _notify_update(self, order: Order) -> None:
        """Invoke all registered order-update callbacks."""
        for cb in self._update_callbacks:
            try:
                await cb(order)
            except Exception:
                logger.exception(
                    "simulated_broker.callback_error",
                    order_id=order.id,
                )
