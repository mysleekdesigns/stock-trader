"""Realistic order fill simulation for backtesting.

Models slippage (volume-dependent), commission (per-share or per-trade),
and Almgren-Chriss market impact (permanent + temporary components).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

import structlog

from src.core.types import Bar, Order, OrderSide, OrderType

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FillResult:
    """Outcome of simulating a single order fill."""

    filled_price: Decimal
    filled_quantity: Decimal
    commission: Decimal
    slippage: Decimal
    market_impact: Decimal

    @property
    def total_cost(self) -> Decimal:
        """Absolute cost including commission and market impact."""
        return self.commission + abs(self.slippage) + abs(self.market_impact)


class FillSimulator:
    """Simulate realistic order fills against historical bars.

    Parameters
    ----------
    config:
        Optional overrides. Recognized keys:

        - ``base_slippage`` (float): baseline slippage fraction (default 0.0005).
        - ``commission_mode`` ("per_share" | "per_trade"): default "per_share".
        - ``commission_per_share`` (float): default 0.005 ($ per share).
        - ``commission_per_trade`` (float): default 1.00 ($ per trade).
        - ``max_volume_pct`` (float): reject if order > this fraction of bar
          volume (default 0.10).
        - ``permanent_impact_gamma`` (float): Almgren-Chriss gamma
          (default 0.1).
        - ``temporary_impact_eta`` (float): Almgren-Chriss eta (default 0.01).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._base_slippage: float = cfg.get("base_slippage", 0.0005)
        self._commission_mode: str = cfg.get("commission_mode", "per_share")
        self._commission_per_share: float = cfg.get("commission_per_share", 0.005)
        self._commission_per_trade: float = cfg.get("commission_per_trade", 1.00)
        self._max_volume_pct: float = cfg.get("max_volume_pct", 0.10)
        self._permanent_gamma: float = cfg.get("permanent_impact_gamma", 0.1)
        self._temporary_eta: float = cfg.get("temporary_impact_eta", 0.01)

        logger.info(
            "fill_simulator.init",
            base_slippage=self._base_slippage,
            commission_mode=self._commission_mode,
            max_volume_pct=self._max_volume_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_fill(
        self,
        order: Order,
        bar: Bar,
        portfolio_value: float = 0.0,
    ) -> FillResult | None:
        """Simulate filling *order* against *bar*.

        Returns a :class:`FillResult` on success, or ``None`` if the fill
        is rejected (e.g. volume constraint violated or limit price not met).
        """
        qty = float(order.quantity)
        volume = bar.volume

        # --- Volume rejection check ---
        if volume <= 0:
            logger.warning(
                "fill_simulator.zero_volume",
                symbol=order.symbol,
                timestamp=str(bar.timestamp),
            )
            return None

        participation_rate = qty / volume
        if participation_rate > self._max_volume_pct:
            logger.warning(
                "fill_simulator.volume_rejection",
                symbol=order.symbol,
                order_qty=qty,
                bar_volume=volume,
                participation_pct=round(participation_rate * 100, 2),
                max_pct=round(self._max_volume_pct * 100, 2),
            )
            return None

        # --- Determine base execution price ---
        base_price = self._get_base_price(order, bar)
        if base_price is None:
            # Limit / stop price not met during this bar.
            return None

        # --- Slippage (volume-dependent) ---
        slippage_frac = self._compute_slippage(qty, volume)
        slippage_amount = base_price * slippage_frac

        # --- Almgren-Chriss market impact ---
        permanent, temporary = self._compute_market_impact(
            qty, volume, base_price,
        )
        total_impact = permanent + temporary

        # --- Apply directional adjustments ---
        if order.side == OrderSide.BUY:
            filled_price = base_price + slippage_amount + total_impact
        else:
            filled_price = base_price - slippage_amount - total_impact

        # Price cannot go below zero.
        filled_price = max(filled_price, 0.0001)

        # --- Commission ---
        commission = self._compute_commission(qty)

        filled_price_dec = Decimal(str(filled_price)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )
        filled_qty_dec = order.quantity
        commission_dec = Decimal(str(commission)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        slippage_dec = Decimal(str(slippage_amount)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )
        impact_dec = Decimal(str(total_impact)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        logger.debug(
            "fill_simulator.fill",
            symbol=order.symbol,
            side=order.side.value,
            qty=str(filled_qty_dec),
            base_price=round(base_price, 4),
            filled_price=str(filled_price_dec),
            slippage=str(slippage_dec),
            market_impact=str(impact_dec),
            commission=str(commission_dec),
        )

        return FillResult(
            filled_price=filled_price_dec,
            filled_quantity=filled_qty_dec,
            commission=commission_dec,
            slippage=slippage_dec,
            market_impact=impact_dec,
        )

    # ------------------------------------------------------------------
    # Price determination
    # ------------------------------------------------------------------

    def _get_base_price(self, order: Order, bar: Bar) -> float | None:
        """Determine the base execution price given order type and bar.

        Returns ``None`` when a limit/stop condition is not met.
        """
        if order.order_type == OrderType.MARKET:
            # Use the bar open as proxy (order arrives at bar open).
            return bar.open

        if order.order_type == OrderType.LIMIT:
            limit = float(order.limit_price) if order.limit_price is not None else None
            if limit is None:
                return bar.open
            if order.side == OrderSide.BUY and bar.low <= limit:
                return min(limit, bar.open)
            if order.side == OrderSide.SELL and bar.high >= limit:
                return max(limit, bar.open)
            return None  # Limit not reached.

        if order.order_type == OrderType.STOP:
            stop = float(order.stop_price) if order.stop_price is not None else None
            if stop is None:
                return bar.open
            if order.side == OrderSide.BUY and bar.high >= stop:
                return max(stop, bar.open)
            if order.side == OrderSide.SELL and bar.low <= stop:
                return min(stop, bar.open)
            return None  # Stop not triggered.

        if order.order_type == OrderType.STOP_LIMIT:
            stop = float(order.stop_price) if order.stop_price is not None else None
            limit = float(order.limit_price) if order.limit_price is not None else None
            if stop is None or limit is None:
                return bar.open
            # Stop must be triggered first.
            if order.side == OrderSide.BUY:
                if bar.high >= stop and bar.low <= limit:
                    return min(limit, max(stop, bar.open))
            else:
                if bar.low <= stop and bar.high >= limit:
                    return max(limit, min(stop, bar.open))
            return None

        return bar.open

    # ------------------------------------------------------------------
    # Cost models
    # ------------------------------------------------------------------

    def _compute_slippage(self, qty: float, volume: float) -> float:
        """Volume-dependent slippage: base * sqrt(qty / volume)."""
        if volume <= 0:
            return self._base_slippage
        ratio = qty / volume
        return self._base_slippage * math.sqrt(ratio)

    def _compute_market_impact(
        self,
        qty: float,
        volume: float,
        price: float,
    ) -> tuple[float, float]:
        """Almgren-Chriss permanent and temporary market impact.

        Returns (permanent_amount, temporary_amount) in price units.
        """
        if volume <= 0:
            return 0.0, 0.0

        ratio = qty / volume
        permanent = self._permanent_gamma * ratio * price
        temporary = self._temporary_eta * (ratio ** 0.6) * price
        return permanent, temporary

    def _compute_commission(self, qty: float) -> float:
        """Calculate commission based on configured mode."""
        if self._commission_mode == "per_share":
            return qty * self._commission_per_share
        return self._commission_per_trade
