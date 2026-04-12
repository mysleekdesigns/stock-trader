"""Order management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_event_bus, get_portfolio
from src.api.schemas.order import OrderFilter, OrderRequest, OrderResponse
from src.core.events import EventBus, OrderEvent
from src.core.types import Order, OrderSide, OrderStatus, OrderType
from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/orders", tags=["orders"])

# ---------------------------------------------------------------------------
# In-memory order store (would be backed by DB in production)
# ---------------------------------------------------------------------------

_orders: dict[str, Order] = {}


def _order_to_response(order: Order) -> OrderResponse:
    return OrderResponse(
        id=order.id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        order_type=order.order_type,
        status=order.status,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
        filled_quantity=order.filled_quantity,
        filled_avg_price=order.filled_avg_price,
        strategy_name=order.strategy_name,
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[OrderResponse])
async def list_orders(
    symbol: str | None = Query(default=None),
    status: OrderStatus | None = Query(default=None),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[OrderResponse]:
    """List orders with optional filters."""
    results = list(_orders.values())

    if symbol:
        results = [o for o in results if o.symbol == symbol.upper()]
    if status:
        results = [o for o in results if o.status == status]
    if start_date:
        start_dt = datetime.fromisoformat(start_date)
        results = [o for o in results if o.created_at >= start_dt]
    if end_date:
        end_dt = datetime.fromisoformat(end_date)
        results = [o for o in results if o.created_at <= end_dt]

    # Sort by most recent first
    results.sort(key=lambda o: o.created_at, reverse=True)

    # Paginate
    results = results[offset : offset + limit]

    return [_order_to_response(o) for o in results]


@router.post("", response_model=OrderResponse, status_code=201)
async def submit_order(
    body: OrderRequest,
    portfolio: Portfolio = Depends(get_portfolio),
    event_bus: EventBus = Depends(get_event_bus),
) -> OrderResponse:
    """Submit a manual order."""
    order = Order(
        symbol=body.symbol.upper(),
        side=body.side,
        quantity=body.quantity,
        order_type=body.order_type,
        limit_price=body.limit_price,
        stop_price=body.stop_price,
        strategy_name="manual",
    )

    _orders[order.id] = order

    # Publish order event
    await event_bus.publish(OrderEvent(order=order))

    logger.info(
        "order.submitted",
        order_id=order.id,
        symbol=order.symbol,
        side=order.side.value,
        quantity=str(order.quantity),
    )

    return _order_to_response(order)


@router.delete("/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    event_bus: EventBus = Depends(get_event_bus),
) -> OrderResponse:
    """Cancel a pending order."""
    order = _orders.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' not found")

    if order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel order in status '{order.status.value}'",
        )

    previous_status = order.status
    order.status = OrderStatus.CANCELLED
    order.updated_at = datetime.now(timezone.utc)

    await event_bus.publish(
        OrderEvent(order=order, previous_status=previous_status),
    )

    logger.info("order.cancelled", order_id=order_id)

    return _order_to_response(order)
