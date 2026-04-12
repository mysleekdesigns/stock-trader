"""Risk manager — the central pre-trade and portfolio-level risk gate.

All order flow passes through :meth:`RiskManager.check_order` before reaching
the broker adapter.  The manager also runs periodic portfolio-level sweeps via
:meth:`check_portfolio`.
"""

from __future__ import annotations

from copy import copy
from decimal import Decimal
from typing import Any

import structlog

from src.core.events import EventBus, RiskBreachEvent
from src.core.types import Order
from src.risk.limits import (
    CorrelatedPositionsLimit,
    DailyLossLimit,
    DrawdownLimit,
    GrossExposureLimit,
    NetExposureLimit,
    PositionConcentrationLimit,
    RiskLimit,
    SectorExposureLimit,
)
from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)


class RiskManager:
    """Centralised risk gate that enforces pre-trade and portfolio-level limits.

    Parameters
    ----------
    event_bus:
        Async event bus used to publish :class:`RiskBreachEvent` instances.
    config:
        Risk configuration dict, typically loaded from ``config/risk_limits.yaml``.
    """

    def __init__(self, event_bus: EventBus, config: dict[str, Any] | None = None) -> None:
        self._event_bus = event_bus
        self._config = config or {}
        self._limits: list[RiskLimit] = []
        self._pending_events: list[RiskBreachEvent] = []

        self._build_limits()
        logger.info("risk_manager.initialised", limit_count=len(self._limits))

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_limits(self) -> None:
        """Construct limit objects from config."""
        portfolio_cfg = self._config.get("portfolio", {})
        position_cfg = self._config.get("position", {})
        sector_map: dict[str, str] = self._config.get("sector_map", {})

        self._limits = [
            DrawdownLimit(
                max_drawdown=portfolio_cfg.get("max_drawdown", 0.10),
            ),
            DailyLossLimit(
                max_daily_loss=portfolio_cfg.get("max_daily_loss", 0.03),
            ),
            GrossExposureLimit(
                max_exposure=portfolio_cfg.get("max_gross_exposure", 2.0),
            ),
            NetExposureLimit(
                max_exposure=portfolio_cfg.get("max_net_exposure", 1.0),
            ),
            PositionConcentrationLimit(
                max_fraction=position_cfg.get("max_single_position", 0.05),
            ),
            SectorExposureLimit(
                max_fraction=position_cfg.get("max_sector_exposure", 0.25),
                sector_map=sector_map,
            ),
            CorrelatedPositionsLimit(
                max_correlated=position_cfg.get("max_correlated_positions", 3),
                sector_map=sector_map,
            ),
        ]

    @property
    def limits(self) -> list[RiskLimit]:
        """Expose configured limits for introspection."""
        return list(self._limits)

    def update_sector_map(self, sector_map: dict[str, str]) -> None:
        """Hot-update the sector mapping on sector-aware limits."""
        for limit in self._limits:
            if isinstance(limit, (SectorExposureLimit, CorrelatedPositionsLimit)):
                limit.set_sector_map(sector_map)
        logger.info("risk_manager.sector_map_updated", sector_count=len(sector_map))

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def check_order(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """Run all risk limits against a proposed order.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if the order passes all checks, or
            ``(False, reason)`` with the first breach reason.
        """
        for limit in self._limits:
            passed, reason = limit.check(portfolio, order)
            if not passed:
                event = RiskBreachEvent(
                    rule_name=limit.name,
                    message=reason,
                    details={
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": str(order.quantity),
                    },
                )
                self._pending_events.append(event)
                logger.warning(
                    "risk_manager.order_rejected",
                    order_id=order.id,
                    symbol=order.symbol,
                    rule=limit.name,
                    reason=reason,
                )
                return False, reason

        logger.info(
            "risk_manager.order_approved",
            order_id=order.id,
            symbol=order.symbol,
        )
        return True, ""

    # ------------------------------------------------------------------
    # Portfolio-level sweep
    # ------------------------------------------------------------------

    def check_portfolio(self, portfolio: Portfolio) -> list[RiskBreachEvent]:
        """Run all limits at the portfolio level (no specific order).

        Returns
        -------
        list[RiskBreachEvent]
            A list of breach events for any limits that are currently violated.
        """
        breaches: list[RiskBreachEvent] = []

        for limit in self._limits:
            passed, reason = limit.check(portfolio, order=None)
            if not passed:
                event = RiskBreachEvent(
                    rule_name=limit.name,
                    message=reason,
                )
                breaches.append(event)
                self._pending_events.append(event)
                logger.warning(
                    "risk_manager.portfolio_breach",
                    rule=limit.name,
                    reason=reason,
                )

        return breaches

    # ------------------------------------------------------------------
    # Order adjustment
    # ------------------------------------------------------------------

    def adjust_order_for_risk(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> Order:
        """Reduce order size if necessary to stay within risk limits.

        The method iteratively shrinks the order quantity until all limits
        pass, or the quantity hits zero.

        Returns
        -------
        Order
            A (possibly modified) copy of the order with reduced quantity.
        """
        adjusted = copy(order)

        # Quick check: if the order already passes, return immediately
        passed, _ = self.check_order(adjusted, portfolio)
        if passed:
            return adjusted

        # Binary search for the maximum acceptable quantity
        original_qty = adjusted.quantity
        low = Decimal("0")
        high = original_qty
        best_qty = Decimal("0")

        for _ in range(20):  # 20 iterations gives ~1e-6 precision
            mid = (low + high) / 2
            if mid <= Decimal("0"):
                break
            adjusted.quantity = mid
            # Clear pending events from failed checks during search
            snapshot_len = len(self._pending_events)
            passed, _ = self.check_order(adjusted, portfolio)
            if passed:
                best_qty = mid
                low = mid
            else:
                # Discard breach events generated during search
                self._pending_events = self._pending_events[:snapshot_len]
                high = mid

        adjusted.quantity = best_qty

        if best_qty < original_qty:
            logger.info(
                "risk_manager.order_adjusted",
                order_id=order.id,
                symbol=order.symbol,
                original_qty=str(original_qty),
                adjusted_qty=str(best_qty),
            )

        return adjusted

    # ------------------------------------------------------------------
    # Event flushing
    # ------------------------------------------------------------------

    async def flush_events(self) -> None:
        """Publish all pending breach events to the event bus and clear the buffer."""
        events = self._pending_events
        self._pending_events = []
        for event in events:
            await self._event_bus.publish(event)
        if events:
            logger.info("risk_manager.events_flushed", count=len(events))
