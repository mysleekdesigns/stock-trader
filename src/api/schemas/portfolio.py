"""Pydantic models for portfolio-related API responses."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class PortfolioSummary(BaseModel):
    """Top-level portfolio snapshot returned by the dashboard endpoint."""

    total_value: Decimal = Field(description="Total portfolio value (cash + positions)")
    cash: Decimal = Field(description="Available cash balance")
    positions_value: Decimal = Field(description="Aggregate market value of open positions")
    unrealized_pnl: Decimal = Field(description="Sum of unrealized P&L across positions")
    realized_pnl: Decimal = Field(description="Cumulative realized P&L")
    drawdown: Decimal = Field(description="Current drawdown as a fraction of peak value")
    gross_exposure: Decimal = Field(description="Gross exposure as a fraction of total value")
    net_exposure: Decimal = Field(description="Net (directional) exposure as a fraction")

    model_config = {"json_schema_extra": {"example": {
        "total_value": "105234.50",
        "cash": "45000.00",
        "positions_value": "60234.50",
        "unrealized_pnl": "1234.50",
        "realized_pnl": "3200.00",
        "drawdown": "0.012",
        "gross_exposure": "0.572",
        "net_exposure": "0.350",
    }}}


class PositionResponse(BaseModel):
    """A single open position."""

    symbol: str
    side: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    market_value: Decimal
    pct_of_portfolio: Decimal = Field(
        description="Position market value as a percentage of total portfolio value",
    )


class DailyPnL(BaseModel):
    """One day of P&L history."""

    date: date
    pnl: Decimal
    pnl_pct: Decimal
    cumulative_pnl: Decimal
