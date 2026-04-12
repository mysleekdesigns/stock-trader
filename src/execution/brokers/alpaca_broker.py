"""Alpaca broker adapter for paper and live trading.

Uses the ``alpaca-py`` SDK for both REST order management and WebSocket
streaming of trade updates.  Paper vs. live is controlled by the
``ALPACA_BASE_URL`` environment variable (or constructor parameter).
"""

from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog

from src.core.exceptions import OrderCancellationError, OrderSubmissionError
from src.core.types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.brokers.base import BrokerAdapter, OrderUpdateCallback

logger = structlog.get_logger(__name__)

# Alpaca paper trading base URL
_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaBroker(BrokerAdapter):
    """Broker adapter for the Alpaca trading platform.

    Parameters
    ----------
    api_key:
        Alpaca API key.  Falls back to ``ALPACA_API_KEY`` env var.
    secret_key:
        Alpaca secret key.  Falls back to ``ALPACA_SECRET_KEY`` env var.
    base_url:
        REST base URL.  Falls back to ``ALPACA_BASE_URL`` env var, defaulting
        to the paper-trading endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = base_url or os.environ.get("ALPACA_BASE_URL", _PAPER_BASE_URL)

        self._trading_client: Any = None
        self._trading_stream: Any = None
        self._update_callbacks: list[OrderUpdateCallback] = []

        is_paper = "paper" in self._base_url.lower()
        logger.info(
            "alpaca_broker.init",
            base_url=self._base_url,
            paper=is_paper,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the Alpaca REST client and streaming connection."""
        try:
            from alpaca.trading.client import TradingClient

            is_paper = "paper" in self._base_url.lower()
            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=is_paper,
            )
            logger.info("alpaca_broker.connected", base_url=self._base_url)
        except ImportError:
            logger.error(
                "alpaca_broker.missing_dependency",
                message="alpaca-py is not installed. Run: uv add alpaca-py",
            )
            raise
        except Exception as exc:
            logger.error("alpaca_broker.connect_failed", error=str(exc))
            raise

    async def disconnect(self) -> None:
        """Close the streaming connection."""
        if self._trading_stream is not None:
            try:
                await self._trading_stream.close()
            except Exception:
                logger.exception("alpaca_broker.stream_close_error")
            self._trading_stream = None
        logger.info("alpaca_broker.disconnected")

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        if self._trading_client is None:
            raise OrderSubmissionError("Broker not connected — call connect() first")

        try:
            from alpaca.trading.requests import (
                LimitOrderRequest,
                MarketOrderRequest,
                StopLimitOrderRequest,
                StopOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpacaSide
            from alpaca.trading.enums import TimeInForce

            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
            qty = float(order.quantity)

            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=float(order.limit_price) if order.limit_price else None,
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                request = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=float(order.limit_price) if order.limit_price else None,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                )
            else:
                raise OrderSubmissionError(
                    f"Unsupported order type: {order.order_type}",
                    details={"order_id": order.id},
                )

            alpaca_order = self._trading_client.submit_order(order_data=request)

            # Map Alpaca response back to our domain Order
            order.status = self._map_status(str(alpaca_order.status))
            order.metadata["alpaca_id"] = str(alpaca_order.id)
            order.updated_at = datetime.utcnow()

            if alpaca_order.filled_qty is not None:
                order.filled_quantity = Decimal(str(alpaca_order.filled_qty))
            if alpaca_order.filled_avg_price is not None:
                order.filled_avg_price = Decimal(str(alpaca_order.filled_avg_price))

            logger.info(
                "alpaca_broker.order_submitted",
                order_id=order.id,
                alpaca_id=order.metadata.get("alpaca_id"),
                symbol=order.symbol,
                status=order.status.value,
            )
            return order

        except (OrderSubmissionError,):
            raise
        except Exception as exc:
            logger.error(
                "alpaca_broker.submit_failed",
                order_id=order.id,
                error=str(exc),
            )
            raise OrderSubmissionError(
                f"Alpaca order submission failed: {exc}",
                details={"order_id": order.id, "error": str(exc)},
            ) from exc

    async def cancel_order(self, order_id: str) -> bool:
        if self._trading_client is None:
            raise OrderCancellationError("Broker not connected")

        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info("alpaca_broker.order_cancelled", order_id=order_id)
            return True
        except Exception as exc:
            logger.error(
                "alpaca_broker.cancel_failed",
                order_id=order_id,
                error=str(exc),
            )
            raise OrderCancellationError(
                f"Alpaca cancel failed: {exc}",
                details={"order_id": order_id, "error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        if self._trading_client is None:
            return []

        try:
            alpaca_positions = self._trading_client.get_all_positions()
            return [self._map_position(p) for p in alpaca_positions]
        except Exception as exc:
            logger.error("alpaca_broker.get_positions_failed", error=str(exc))
            return []

    async def get_account(self) -> dict[str, Any]:
        if self._trading_client is None:
            return {}

        try:
            acct = self._trading_client.get_account()
            return {
                "cash": str(acct.cash),
                "equity": str(acct.equity),
                "buying_power": str(acct.buying_power),
                "portfolio_value": str(acct.portfolio_value),
                "currency": str(acct.currency),
                "pattern_day_trader": bool(acct.pattern_day_trader),
                "trading_blocked": bool(acct.trading_blocked),
                "account_blocked": bool(acct.account_blocked),
            }
        except Exception as exc:
            logger.error("alpaca_broker.get_account_failed", error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def subscribe_order_updates(self, callback: OrderUpdateCallback) -> None:
        self._update_callbacks.append(callback)

        if self._trading_stream is None:
            try:
                from alpaca.trading.stream import TradingStream

                self._trading_stream = TradingStream(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                    url_override=self._base_url,
                )
                self._trading_stream.subscribe_trade_updates(self._handle_trade_update)

                logger.info("alpaca_broker.stream_subscribed")
            except ImportError:
                logger.error(
                    "alpaca_broker.missing_dependency",
                    message="alpaca-py is not installed for streaming",
                )
            except Exception as exc:
                logger.error("alpaca_broker.stream_subscribe_failed", error=str(exc))

    async def start_stream(self) -> None:
        """Start the Alpaca trade-update WebSocket stream.

        This is a blocking call that should be run in a background task.
        """
        if self._trading_stream is not None:
            logger.info("alpaca_broker.stream_starting")
            await self._trading_stream._run_forever()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_trade_update(self, data: Any) -> None:
        """Process an Alpaca trade update and invoke registered callbacks."""
        try:
            event_type = getattr(data, "event", None) or str(data.get("event", ""))
            alpaca_order = getattr(data, "order", None) or data.get("order", {})

            order = Order(
                symbol=str(getattr(alpaca_order, "symbol", "")),
                side=(
                    OrderSide.BUY
                    if str(getattr(alpaca_order, "side", "buy")) == "buy"
                    else OrderSide.SELL
                ),
                quantity=Decimal(str(getattr(alpaca_order, "qty", 0))),
                order_type=self._map_order_type(
                    str(getattr(alpaca_order, "type", "market"))
                ),
                status=self._map_status(event_type),
                id=str(getattr(alpaca_order, "id", "")),
            )

            filled_qty = getattr(alpaca_order, "filled_qty", None)
            if filled_qty is not None:
                order.filled_quantity = Decimal(str(filled_qty))

            filled_price = getattr(alpaca_order, "filled_avg_price", None)
            if filled_price is not None:
                order.filled_avg_price = Decimal(str(filled_price))

            order.metadata["alpaca_id"] = str(getattr(alpaca_order, "id", ""))

            logger.info(
                "alpaca_broker.trade_update",
                event=event_type,
                order_id=order.id,
                status=order.status.value,
            )

            for cb in self._update_callbacks:
                try:
                    await cb(order)
                except Exception:
                    logger.exception(
                        "alpaca_broker.callback_error",
                        order_id=order.id,
                    )
        except Exception:
            logger.exception("alpaca_broker.trade_update_parse_error")

    @staticmethod
    def _map_status(alpaca_status: str) -> OrderStatus:
        """Map an Alpaca order status string to our ``OrderStatus`` enum."""
        mapping: dict[str, OrderStatus] = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIAL_FILL,
            "fill": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "replaced": OrderStatus.SUBMITTED,
            "rejected": OrderStatus.REJECTED,
            "pending_cancel": OrderStatus.SUBMITTED,
            "pending_replace": OrderStatus.SUBMITTED,
            "stopped": OrderStatus.FILLED,
            "suspended": OrderStatus.REJECTED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.SUBMITTED)

    @staticmethod
    def _map_order_type(alpaca_type: str) -> OrderType:
        """Map an Alpaca order type string to our ``OrderType`` enum."""
        mapping: dict[str, OrderType] = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
        }
        return mapping.get(alpaca_type.lower(), OrderType.MARKET)

    @staticmethod
    def _map_position(alpaca_pos: Any) -> Position:
        """Map an Alpaca position object to our ``Position`` dataclass."""
        qty = Decimal(str(getattr(alpaca_pos, "qty", 0)))
        side = (
            OrderSide.BUY
            if str(getattr(alpaca_pos, "side", "long")) == "long"
            else OrderSide.SELL
        )
        return Position(
            symbol=str(getattr(alpaca_pos, "symbol", "")),
            side=side,
            quantity=abs(qty),
            avg_entry_price=Decimal(str(getattr(alpaca_pos, "avg_entry_price", 0))),
            current_price=Decimal(str(getattr(alpaca_pos, "current_price", 0))),
            unrealized_pnl=Decimal(str(getattr(alpaca_pos, "unrealized_pl", 0))),
        )
