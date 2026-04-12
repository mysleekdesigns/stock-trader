"""Pydantic models for order-related API endpoints."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from src.core.types import OrderSide, OrderStatus, OrderType


class OrderRequest(BaseModel):
    """Payload for submitting a new manual order."""

    symbol: str = Field(min_length=1, max_length=10)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)


class OrderResponse(BaseModel):
    """Serialised representation of an Order."""

    id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    status: OrderStatus
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal
    filled_avg_price: Decimal | None = None
    strategy_name: str
    created_at: datetime
    updated_at: datetime


class OrderFilter(BaseModel):
    """Query parameters for listing orders."""

    symbol: str | None = None
    status: OrderStatus | None = None
    start_date: date | None = None
    end_date: date | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
