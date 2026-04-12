"""Portfolio state tracking.

Maintains live position data, cash balances, P&L, exposure metrics, and
drawdown tracking.  All monetary values use ``Decimal`` for precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import structlog

from src.core.types import OrderSide, Position

logger = structlog.get_logger(__name__)


@dataclass
class Portfolio:
    """Tracks portfolio state including positions, cash, P&L, and exposure.

    Parameters
    ----------
    cash:
        Available cash balance.
    initial_value:
        Starting portfolio value for drawdown computation.
    """

    cash: Decimal = Decimal("0")
    initial_value: Decimal = Decimal("0")
    positions: dict[str, Position] = field(default_factory=dict)

    # Internal bookkeeping
    _peak_value: Decimal = field(default=Decimal("0"), repr=False)
    _realized_pnl: Decimal = field(default=Decimal("0"), repr=False)
    _daily_pnl: Decimal = field(default=Decimal("0"), repr=False)
    _daily_pnl_date: date | None = field(default=None, repr=False)
    _starting_day_value: Decimal = field(default=Decimal("0"), repr=False)

    def __post_init__(self) -> None:
        if self._peak_value == Decimal("0"):
            self._peak_value = self.initial_value
        if self._starting_day_value == Decimal("0"):
            self._starting_day_value = self.initial_value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_value(self) -> Decimal:
        """Cash plus the market value of all open positions."""
        position_value = sum(
            (p.market_value for p in self.positions.values()),
            Decimal("0"),
        )
        return self.cash + position_value

    @property
    def unrealized_pnl(self) -> Decimal:
        """Sum of unrealized P&L across all open positions."""
        return sum(
            (p.unrealized_pnl for p in self.positions.values()),
            Decimal("0"),
        )

    @property
    def realized_pnl(self) -> Decimal:
        """Cumulative realized P&L."""
        return self._realized_pnl

    @property
    def drawdown(self) -> Decimal:
        """Current drawdown as a fraction: (peak - current) / peak.

        Returns ``Decimal("0")`` when peak is zero or current value exceeds peak.
        """
        if self._peak_value <= Decimal("0"):
            return Decimal("0")
        current = self.total_value
        if current >= self._peak_value:
            self._peak_value = current
            return Decimal("0")
        return (self._peak_value - current) / self._peak_value

    @property
    def daily_pnl(self) -> Decimal:
        """Intra-day P&L based on the starting-of-day value snapshot."""
        today = datetime.utcnow().date()
        if self._daily_pnl_date != today:
            # New trading day: reset baseline
            self._daily_pnl_date = today
            self._starting_day_value = self.total_value
            return Decimal("0")
        return self.total_value - self._starting_day_value

    @property
    def daily_pnl_pct(self) -> Decimal:
        """Daily P&L as a fraction of the starting-of-day value."""
        if self._starting_day_value <= Decimal("0"):
            return Decimal("0")
        return self.daily_pnl / self._starting_day_value

    @property
    def gross_exposure(self) -> Decimal:
        """Sum of absolute market values divided by total portfolio value."""
        tv = self.total_value
        if tv <= Decimal("0"):
            return Decimal("0")
        gross = sum(
            (abs(p.market_value) for p in self.positions.values()),
            Decimal("0"),
        )
        return gross / tv

    @property
    def net_exposure(self) -> Decimal:
        """Signed sum of market values divided by total portfolio value."""
        tv = self.total_value
        if tv <= Decimal("0"):
            return Decimal("0")
        net = sum(
            (
                p.market_value if p.side == OrderSide.BUY else -p.market_value
                for p in self.positions.values()
            ),
            Decimal("0"),
        )
        return net / tv

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """Add or overwrite a position."""
        self.positions[position.symbol] = position
        logger.info(
            "portfolio.position_added",
            symbol=position.symbol,
            side=position.side.value,
            quantity=str(position.quantity),
        )

    def remove_position(self, symbol: str) -> Position | None:
        """Remove and return a position, recording realized P&L."""
        position = self.positions.pop(symbol, None)
        if position is not None:
            self._realized_pnl += position.unrealized_pnl
            logger.info(
                "portfolio.position_removed",
                symbol=symbol,
                realized_pnl=str(position.unrealized_pnl),
            )
        return position

    def update_position(self, symbol: str, **kwargs: Any) -> None:
        """Update fields on an existing position in-place."""
        position = self.positions.get(symbol)
        if position is None:
            logger.warning("portfolio.update_missing_position", symbol=symbol)
            return
        for attr, value in kwargs.items():
            if hasattr(position, attr):
                setattr(position, attr, value)
        # Recompute unrealized P&L
        if position.side == OrderSide.BUY:
            position.unrealized_pnl = (
                position.current_price - position.avg_entry_price
            ) * position.quantity
        else:
            position.unrealized_pnl = (
                position.avg_entry_price - position.current_price
            ) * position.quantity

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Refresh current_price and unrealized P&L for all matching positions."""
        for symbol, price in prices.items():
            position = self.positions.get(symbol)
            if position is None:
                continue
            position.current_price = price
            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (price - position.avg_entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.avg_entry_price - price) * position.quantity

        # Update peak for drawdown tracking
        current = self.total_value
        if current > self._peak_value:
            self._peak_value = current

    # ------------------------------------------------------------------
    # Sector exposure
    # ------------------------------------------------------------------

    def get_sector_exposures(self, sector_map: dict[str, str]) -> dict[str, Decimal]:
        """Return per-sector exposure as a fraction of total portfolio value.

        Parameters
        ----------
        sector_map:
            Mapping from symbol to sector name.
        """
        tv = self.total_value
        if tv <= Decimal("0"):
            return {}

        sector_values: dict[str, Decimal] = {}
        for symbol, position in self.positions.items():
            sector = sector_map.get(symbol, "unknown")
            sector_values[sector] = sector_values.get(sector, Decimal("0")) + abs(position.market_value)

        return {sector: value / tv for sector, value in sector_values.items()}

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def reset_daily_pnl(self) -> None:
        """Explicitly reset the daily P&L baseline to the current portfolio value."""
        self._daily_pnl_date = datetime.utcnow().date()
        self._starting_day_value = self.total_value
        logger.info("portfolio.daily_pnl_reset", starting_value=str(self._starting_day_value))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the portfolio state."""
        return {
            "cash": str(self.cash),
            "total_value": str(self.total_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self._realized_pnl),
            "drawdown": str(self.drawdown),
            "gross_exposure": str(self.gross_exposure),
            "net_exposure": str(self.net_exposure),
            "daily_pnl": str(self.daily_pnl),
            "peak_value": str(self._peak_value),
            "positions": {
                symbol: {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "quantity": str(p.quantity),
                    "avg_entry_price": str(p.avg_entry_price),
                    "current_price": str(p.current_price),
                    "unrealized_pnl": str(p.unrealized_pnl),
                    "realized_pnl": str(p.realized_pnl),
                    "market_value": str(p.market_value),
                }
                for symbol, p in self.positions.items()
            },
        }
