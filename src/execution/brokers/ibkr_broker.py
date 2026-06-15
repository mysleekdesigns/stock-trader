"""Interactive Brokers (IBKR) broker adapter.

Connects to a running **TWS** or **IB Gateway** instance via the ``ib_insync``
library and implements the :class:`~src.execution.brokers.base.BrokerAdapter`
contract.  Unlike the equities-only Alpaca adapter, IBKR gives access to
futures and options, so this adapter builds the appropriate IB ``Contract`` from
the order's ``metadata`` (``asset_class`` plus any contract specifics).

Connection settings (host / port / client id) come from the constructor or the
``IBKR_HOST`` / ``IBKR_PORT`` / ``IBKR_CLIENT_ID`` environment variables.  Common
ports: ``7497`` (TWS paper), ``7496`` (TWS live), ``4002`` (Gateway paper),
``4001`` (Gateway live).

``ib_insync`` is imported lazily so the rest of the system runs without it
installed; ``connect()`` raises a clear error if the dependency is missing
(``uv add ib_insync``).
"""

from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog

from src.core.exceptions import OrderCancellationError, OrderSubmissionError
from src.core.types import (
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.brokers.base import BrokerAdapter, OrderUpdateCallback

logger = structlog.get_logger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7497  # TWS paper trading
_DEFAULT_CLIENT_ID = 1


class IBKRBroker(BrokerAdapter):
    """Broker adapter for Interactive Brokers (TWS / IB Gateway).

    Parameters
    ----------
    host:
        TWS/Gateway host.  Falls back to ``IBKR_HOST`` then ``127.0.0.1``.
    port:
        TWS/Gateway API port.  Falls back to ``IBKR_PORT`` then ``7497``.
    client_id:
        Unique API client id.  Falls back to ``IBKR_CLIENT_ID`` then ``1``.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
    ) -> None:
        # Use explicit None checks so a valid 0 (e.g. IBKR master client id 0)
        # is not treated as falsy and overridden by the env var / default.
        self._host = host or os.environ.get("IBKR_HOST", _DEFAULT_HOST)
        self._port = int(port if port is not None else os.environ.get("IBKR_PORT", _DEFAULT_PORT))
        self._client_id = int(
            client_id if client_id is not None else os.environ.get("IBKR_CLIENT_ID", _DEFAULT_CLIENT_ID)
        )

        self._ib: Any = None
        self._update_callbacks: list[OrderUpdateCallback] = []
        # Map our domain order id -> ib_insync Trade, and IB orderId -> our id.
        self._trades: dict[str, Any] = {}
        self._ib_id_to_order_id: dict[int, str] = {}

        is_paper = self._port in (7497, 4002)
        logger.info(
            "ibkr_broker.init",
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            paper=is_paper,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to TWS / IB Gateway and wire up order-status events."""
        try:
            from ib_insync import IB
        except ImportError:
            logger.error(
                "ibkr_broker.missing_dependency",
                message="ib_insync is not installed. Run: uv add ib_insync",
            )
            raise

        try:
            self._ib = IB()
            await self._ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
            )
            # ib_insync fires orderStatusEvent on every order state change.
            self._ib.orderStatusEvent += self._on_order_status
            logger.info("ibkr_broker.connected", host=self._host, port=self._port)
        except Exception as exc:
            logger.error("ibkr_broker.connect_failed", error=str(exc))
            raise

    async def disconnect(self) -> None:
        """Disconnect from TWS / IB Gateway."""
        if self._ib is not None:
            try:
                self._ib.orderStatusEvent -= self._on_order_status
            except Exception:
                pass
            try:
                self._ib.disconnect()
            except Exception:
                logger.exception("ibkr_broker.disconnect_error")
            self._ib = None
        logger.info("ibkr_broker.disconnected")

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        if self._ib is None:
            raise OrderSubmissionError("Broker not connected — call connect() first")

        try:
            contract = self._build_contract(order)
            ib_order = self._build_ib_order(order)

            trade = self._ib.placeOrder(contract, ib_order)

            ib_order_id = int(getattr(trade.order, "orderId", 0))
            order.metadata["ibkr_order_id"] = ib_order_id
            order.metadata["ibkr_perm_id"] = int(getattr(trade.order, "permId", 0) or 0)
            order.status = self._map_status(str(trade.orderStatus.status))
            order.updated_at = datetime.utcnow()

            filled = getattr(trade.orderStatus, "filled", 0) or 0
            avg_price = getattr(trade.orderStatus, "avgFillPrice", 0) or 0
            if filled:
                order.filled_quantity = Decimal(str(filled))
            if avg_price:
                order.filled_avg_price = Decimal(str(avg_price))

            self._trades[order.id] = trade
            if ib_order_id:
                self._ib_id_to_order_id[ib_order_id] = order.id

            logger.info(
                "ibkr_broker.order_submitted",
                order_id=order.id,
                ibkr_order_id=ib_order_id,
                symbol=order.symbol,
                status=order.status.value,
            )
            return order

        except OrderSubmissionError:
            raise
        except Exception as exc:
            logger.error("ibkr_broker.submit_failed", order_id=order.id, error=str(exc))
            raise OrderSubmissionError(
                f"IBKR order submission failed: {exc}",
                details={"order_id": order.id, "error": str(exc)},
            ) from exc

    async def cancel_order(self, order_id: str) -> bool:
        if self._ib is None:
            raise OrderCancellationError("Broker not connected")

        trade = self._trades.get(order_id)
        if trade is None:
            # Allow cancelling by the raw IBKR order id as well.
            mapped = self._ib_id_to_order_id.get(_safe_int(order_id))
            trade = self._trades.get(mapped) if mapped else None
        if trade is None:
            raise OrderCancellationError(
                f"Unknown order id: {order_id}",
                details={"order_id": order_id},
            )

        try:
            self._ib.cancelOrder(trade.order)
            logger.info("ibkr_broker.order_cancelled", order_id=order_id)
            return True
        except Exception as exc:
            logger.error("ibkr_broker.cancel_failed", order_id=order_id, error=str(exc))
            raise OrderCancellationError(
                f"IBKR cancel failed: {exc}",
                details={"order_id": order_id, "error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        if self._ib is None:
            return []
        try:
            return [self._map_position(p) for p in self._ib.positions()]
        except Exception as exc:
            logger.error("ibkr_broker.get_positions_failed", error=str(exc))
            return []

    async def get_account(self) -> dict[str, Any]:
        if self._ib is None:
            return {}
        try:
            summary = {row.tag: row.value for row in self._ib.accountSummary()}
            return {
                "cash": summary.get("TotalCashValue", "0"),
                "equity": summary.get("NetLiquidation", "0"),
                "buying_power": summary.get("BuyingPower", "0"),
                "portfolio_value": summary.get("GrossPositionValue", "0"),
                "currency": summary.get("Currency", "USD"),
                "maintenance_margin": summary.get("MaintMarginReq", "0"),
                "available_funds": summary.get("AvailableFunds", "0"),
            }
        except Exception as exc:
            logger.error("ibkr_broker.get_account_failed", error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def subscribe_order_updates(self, callback: OrderUpdateCallback) -> None:
        self._update_callbacks.append(callback)
        logger.info("ibkr_broker.order_updates_subscribed", n_callbacks=len(self._update_callbacks))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_order_status(self, trade: Any) -> None:
        """ib_insync ``orderStatusEvent`` handler — maps to a domain Order."""
        try:
            ib_order_id = int(getattr(trade.order, "orderId", 0))
            order_id = self._ib_id_to_order_id.get(ib_order_id, str(ib_order_id))

            order = Order(
                symbol=str(getattr(trade.contract, "symbol", "")),
                side=(
                    OrderSide.BUY
                    if str(getattr(trade.order, "action", "BUY")).upper() == "BUY"
                    else OrderSide.SELL
                ),
                quantity=Decimal(str(getattr(trade.order, "totalQuantity", 0) or 0)),
                order_type=self._map_ib_order_type(str(getattr(trade.order, "orderType", "MKT"))),
                status=self._map_status(str(trade.orderStatus.status)),
                id=order_id,
            )
            filled = getattr(trade.orderStatus, "filled", 0) or 0
            avg_price = getattr(trade.orderStatus, "avgFillPrice", 0) or 0
            if filled:
                order.filled_quantity = Decimal(str(filled))
            if avg_price:
                order.filled_avg_price = Decimal(str(avg_price))
            order.metadata["ibkr_order_id"] = ib_order_id

            # ib_insync events are synchronous; schedule async callbacks safely.
            import asyncio

            for cb in self._update_callbacks:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(cb(order))
                except RuntimeError:
                    # No running loop (e.g. in tests) — run to completion.
                    asyncio.run(cb(order))
                except Exception:
                    logger.exception("ibkr_broker.callback_error", order_id=order.id)
        except Exception:
            logger.exception("ibkr_broker.order_status_parse_error")

    def _build_contract(self, order: Order) -> Any:
        """Build an IB ``Contract`` from the order and its metadata."""
        from ib_insync import Future, Option, Stock

        meta = order.metadata
        asset_class = meta.get("asset_class", AssetClass.US_EQUITY)
        if isinstance(asset_class, AssetClass):
            asset_class = asset_class.value
        exchange = meta.get("exchange", "SMART")
        currency = meta.get("currency", "USD")

        if asset_class == AssetClass.FUTURES.value:
            return Future(
                symbol=order.symbol,
                lastTradeDateOrContractMonth=str(meta.get("expiry", "")),
                exchange=meta.get("exchange", "GLOBEX"),
                currency=currency,
            )
        if asset_class == AssetClass.OPTIONS.value:
            return Option(
                symbol=order.symbol,
                lastTradeDateOrContractMonth=str(meta.get("expiry", "")),
                strike=float(meta.get("strike", 0.0)),
                right=str(meta.get("right", "C")),
                exchange=exchange,
                currency=currency,
            )
        return Stock(order.symbol, exchange, currency)

    @staticmethod
    def _build_ib_order(order: Order) -> Any:
        """Build an IB order object from our domain ``Order``."""
        from ib_insync import LimitOrder, MarketOrder, StopOrder
        from ib_insync import Order as IBOrder

        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        qty = float(order.quantity)

        if order.order_type == OrderType.MARKET:
            return MarketOrder(action, qty)
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise OrderSubmissionError("Limit order requires a limit_price")
            return LimitOrder(action, qty, float(order.limit_price))
        if order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise OrderSubmissionError("Stop order requires a stop_price")
            return StopOrder(action, qty, float(order.stop_price))
        if order.order_type == OrderType.STOP_LIMIT:
            if order.limit_price is None or order.stop_price is None:
                raise OrderSubmissionError("Stop-limit order requires limit_price and stop_price")
            ib_order = IBOrder(
                action=action,
                totalQuantity=qty,
                orderType="STP LMT",
                lmtPrice=float(order.limit_price),
                auxPrice=float(order.stop_price),
            )
            return ib_order
        raise OrderSubmissionError(f"Unsupported order type: {order.order_type}")

    @staticmethod
    def _map_status(ib_status: str) -> OrderStatus:
        """Map an ib_insync order-status string to our ``OrderStatus`` enum."""
        mapping: dict[str, OrderStatus] = {
            "pendingsubmit": OrderStatus.PENDING,
            "pendingcancel": OrderStatus.SUBMITTED,
            "presubmitted": OrderStatus.SUBMITTED,
            "submitted": OrderStatus.SUBMITTED,
            "apipending": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "apicancelled": OrderStatus.CANCELLED,
            "inactive": OrderStatus.REJECTED,
        }
        return mapping.get(ib_status.lower(), OrderStatus.SUBMITTED)

    @staticmethod
    def _map_ib_order_type(ib_type: str) -> OrderType:
        """Map an ib_insync order-type string to our ``OrderType`` enum."""
        mapping: dict[str, OrderType] = {
            "mkt": OrderType.MARKET,
            "lmt": OrderType.LIMIT,
            "stp": OrderType.STOP,
            "stp lmt": OrderType.STOP_LIMIT,
        }
        return mapping.get(ib_type.lower(), OrderType.MARKET)

    @staticmethod
    def _map_position(ib_pos: Any) -> Position:
        """Map an ib_insync ``Position`` namedtuple to our ``Position``."""
        qty = Decimal(str(getattr(ib_pos, "position", 0) or 0))
        side = OrderSide.BUY if qty >= 0 else OrderSide.SELL
        contract = getattr(ib_pos, "contract", None)
        symbol = str(getattr(contract, "symbol", "")) if contract else ""
        sec_type = str(getattr(contract, "secType", "STK")) if contract else "STK"
        return Position(
            symbol=symbol,
            side=side,
            quantity=abs(qty),
            avg_entry_price=Decimal(str(getattr(ib_pos, "avgCost", 0) or 0)),
            current_price=Decimal("0"),  # filled from market data elsewhere
            asset_class=_SEC_TYPE_TO_ASSET_CLASS.get(sec_type, AssetClass.US_EQUITY),
        )


_SEC_TYPE_TO_ASSET_CLASS: dict[str, AssetClass] = {
    "STK": AssetClass.US_EQUITY,
    "FUT": AssetClass.FUTURES,
    "OPT": AssetClass.OPTIONS,
    "FOP": AssetClass.OPTIONS,
    "CRYPTO": AssetClass.CRYPTO,
}


def _safe_int(value: str) -> int | None:
    """Best-effort int parse; returns ``None`` for non-numeric input."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
