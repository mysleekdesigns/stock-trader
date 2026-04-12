"""Risk limit definitions.

Each limit implements a uniform ``check`` interface so the :class:`RiskManager`
can iterate over them generically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from src.core.types import Order, OrderSide

if TYPE_CHECKING:
    from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)


class RiskLimit(ABC):
    """Abstract base class for a single risk limit."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and breach reports."""

    @abstractmethod
    def check(
        self,
        portfolio: Portfolio,
        order: Order | None = None,
    ) -> tuple[bool, str]:
        """Check whether the limit is satisfied.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when the limit passes, or ``(False, reason)``
            when breached.
        """


# ---------------------------------------------------------------------------
# Concrete limits
# ---------------------------------------------------------------------------


class DrawdownLimit(RiskLimit):
    """Rejects activity when portfolio drawdown exceeds a threshold."""

    def __init__(self, max_drawdown: float = 0.10) -> None:
        self._max_drawdown = Decimal(str(max_drawdown))

    @property
    def name(self) -> str:
        return "drawdown_limit"

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        dd = portfolio.drawdown
        if dd > self._max_drawdown:
            msg = (
                f"Drawdown {dd:.4f} exceeds limit {self._max_drawdown:.4f}"
            )
            logger.warning("risk.drawdown_breach", drawdown=str(dd), limit=str(self._max_drawdown))
            return False, msg
        return True, ""


class DailyLossLimit(RiskLimit):
    """Rejects activity when intra-day loss exceeds a threshold."""

    def __init__(self, max_daily_loss: float = 0.03) -> None:
        self._max_daily_loss = Decimal(str(max_daily_loss))

    @property
    def name(self) -> str:
        return "daily_loss_limit"

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        daily_pnl_pct = portfolio.daily_pnl_pct
        if daily_pnl_pct < -self._max_daily_loss:
            msg = (
                f"Daily loss {daily_pnl_pct:.4f} exceeds limit -{self._max_daily_loss:.4f}"
            )
            logger.warning(
                "risk.daily_loss_breach",
                daily_pnl_pct=str(daily_pnl_pct),
                limit=str(self._max_daily_loss),
            )
            return False, msg
        return True, ""


class GrossExposureLimit(RiskLimit):
    """Rejects orders that would push gross exposure above the threshold."""

    def __init__(self, max_exposure: float = 2.0) -> None:
        self._max_exposure = Decimal(str(max_exposure))

    @property
    def name(self) -> str:
        return "gross_exposure_limit"

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        current = portfolio.gross_exposure
        additional = Decimal("0")

        if order is not None:
            tv = portfolio.total_value
            if tv > Decimal("0"):
                price = order.limit_price or order.stop_price or Decimal("0")
                additional = (order.quantity * price) / tv if price > Decimal("0") else Decimal("0")

        projected = current + additional
        if projected > self._max_exposure:
            msg = (
                f"Gross exposure {projected:.4f} would exceed limit {self._max_exposure:.4f}"
            )
            logger.warning(
                "risk.gross_exposure_breach",
                current=str(current),
                additional=str(additional),
                limit=str(self._max_exposure),
            )
            return False, msg
        return True, ""


class NetExposureLimit(RiskLimit):
    """Rejects orders that would push net exposure outside bounds."""

    def __init__(self, max_exposure: float = 1.0) -> None:
        self._max_exposure = Decimal(str(max_exposure))

    @property
    def name(self) -> str:
        return "net_exposure_limit"

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        current = portfolio.net_exposure
        additional = Decimal("0")

        if order is not None:
            tv = portfolio.total_value
            if tv > Decimal("0"):
                price = order.limit_price or order.stop_price or Decimal("0")
                if price > Decimal("0"):
                    delta = (order.quantity * price) / tv
                    if order.side == OrderSide.SELL:
                        delta = -delta
                    additional = delta

        projected = current + additional
        if abs(projected) > self._max_exposure:
            msg = (
                f"Net exposure {projected:.4f} would exceed limit "
                f"+/-{self._max_exposure:.4f}"
            )
            logger.warning(
                "risk.net_exposure_breach",
                current=str(current),
                additional=str(additional),
                limit=str(self._max_exposure),
            )
            return False, msg
        return True, ""


class PositionConcentrationLimit(RiskLimit):
    """Rejects orders that would make a single position too large."""

    def __init__(self, max_fraction: float = 0.05) -> None:
        self._max_fraction = Decimal(str(max_fraction))

    @property
    def name(self) -> str:
        return "position_concentration_limit"

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        tv = portfolio.total_value
        if tv <= Decimal("0"):
            return True, ""

        if order is None:
            # Portfolio-level: check each position
            for symbol, position in portfolio.positions.items():
                fraction = abs(position.market_value) / tv
                if fraction > self._max_fraction:
                    msg = (
                        f"Position {symbol} is {fraction:.4f} of portfolio, "
                        f"exceeds limit {self._max_fraction:.4f}"
                    )
                    return False, msg
            return True, ""

        # Pre-trade: project new position size
        existing_value = Decimal("0")
        existing_pos = portfolio.positions.get(order.symbol)
        if existing_pos is not None:
            existing_value = abs(existing_pos.market_value)

        price = order.limit_price or order.stop_price or Decimal("0")
        order_value = order.quantity * price if price > Decimal("0") else Decimal("0")

        # For sells closing a long (or buys closing a short), position shrinks
        if existing_pos is not None:
            if (order.side == OrderSide.SELL and existing_pos.side == OrderSide.BUY) or (
                order.side == OrderSide.BUY and existing_pos.side == OrderSide.SELL
            ):
                projected = existing_value - order_value
            else:
                projected = existing_value + order_value
        else:
            projected = order_value

        fraction = projected / tv if tv > Decimal("0") else Decimal("0")
        if fraction > self._max_fraction:
            msg = (
                f"Position {order.symbol} would be {fraction:.4f} of portfolio, "
                f"exceeds limit {self._max_fraction:.4f}"
            )
            logger.warning(
                "risk.position_concentration_breach",
                symbol=order.symbol,
                fraction=str(fraction),
                limit=str(self._max_fraction),
            )
            return False, msg
        return True, ""


class SectorExposureLimit(RiskLimit):
    """Rejects orders that would push a sector above the exposure threshold."""

    def __init__(
        self,
        max_fraction: float = 0.25,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self._max_fraction = Decimal(str(max_fraction))
        self._sector_map: dict[str, str] = sector_map or {}

    @property
    def name(self) -> str:
        return "sector_exposure_limit"

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        """Update the symbol-to-sector mapping at runtime."""
        self._sector_map = sector_map

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        if not self._sector_map:
            return True, ""

        exposures = portfolio.get_sector_exposures(self._sector_map)

        # If order provided, project the change
        if order is not None:
            tv = portfolio.total_value
            if tv > Decimal("0"):
                sector = self._sector_map.get(order.symbol, "unknown")
                price = order.limit_price or order.stop_price or Decimal("0")
                if price > Decimal("0"):
                    additional = (order.quantity * price) / tv
                    current_sector = exposures.get(sector, Decimal("0"))
                    projected = current_sector + additional
                    if projected > self._max_fraction:
                        msg = (
                            f"Sector {sector!r} exposure {projected:.4f} would "
                            f"exceed limit {self._max_fraction:.4f}"
                        )
                        logger.warning(
                            "risk.sector_exposure_breach",
                            sector=sector,
                            projected=str(projected),
                            limit=str(self._max_fraction),
                        )
                        return False, msg
            return True, ""

        # Portfolio-level check
        for sector, exposure in exposures.items():
            if exposure > self._max_fraction:
                msg = (
                    f"Sector {sector!r} exposure {exposure:.4f} "
                    f"exceeds limit {self._max_fraction:.4f}"
                )
                return False, msg
        return True, ""


class CorrelatedPositionsLimit(RiskLimit):
    """Limits the number of highly-correlated positions held simultaneously.

    By default this uses a simple sector-based proxy: positions in the same
    sector are treated as correlated.  A full correlation-matrix approach can
    be plugged in by overriding ``_get_groups``.
    """

    def __init__(
        self,
        max_correlated: int = 3,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self._max_correlated = max_correlated
        self._sector_map: dict[str, str] = sector_map or {}

    @property
    def name(self) -> str:
        return "correlated_positions_limit"

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        """Update the symbol-to-sector mapping at runtime."""
        self._sector_map = sector_map

    def _get_groups(self, portfolio: Portfolio) -> dict[str, list[str]]:
        """Group open positions by sector (proxy for correlation)."""
        groups: dict[str, list[str]] = {}
        for symbol in portfolio.positions:
            sector = self._sector_map.get(symbol, "unknown")
            groups.setdefault(sector, []).append(symbol)
        return groups

    def check(self, portfolio: Portfolio, order: Order | None = None) -> tuple[bool, str]:
        if not self._sector_map:
            return True, ""

        groups = self._get_groups(portfolio)

        # If an order is provided, simulate adding it
        if order is not None:
            sector = self._sector_map.get(order.symbol, "unknown")
            group = groups.get(sector, [])
            # Only count if the symbol is not already in the group
            if order.symbol not in group:
                simulated_count = len(group) + 1
                if simulated_count > self._max_correlated:
                    msg = (
                        f"Adding {order.symbol} would create {simulated_count} "
                        f"correlated positions in sector {sector!r}, "
                        f"exceeds limit {self._max_correlated}"
                    )
                    logger.warning(
                        "risk.correlated_positions_breach",
                        sector=sector,
                        count=simulated_count,
                        limit=self._max_correlated,
                    )
                    return False, msg
            return True, ""

        # Portfolio-level check
        for sector, symbols in groups.items():
            if len(symbols) > self._max_correlated:
                msg = (
                    f"Sector {sector!r} has {len(symbols)} correlated positions, "
                    f"exceeds limit {self._max_correlated}"
                )
                return False, msg
        return True, ""
