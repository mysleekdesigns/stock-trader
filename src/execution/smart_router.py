"""Smart order routing with TWAP, VWAP, and iceberg execution algorithms.

Splits a parent order into child orders according to the selected execution
strategy, scheduling them across a time window to minimise market impact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from src.core.types import Order, OrderSide, OrderType, OrderStatus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ExecutionAlgo(str, Enum):
    """Supported execution algorithms."""

    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


@dataclass
class ChildOrder:
    """A child order slice produced by an execution algorithm."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    scheduled_time: datetime
    limit_price: Decimal | None = None
    slice_index: int = 0
    total_slices: int = 1
    parent_order_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan returned by an executor."""

    parent_order: Order
    algo: ExecutionAlgo
    child_orders: list[ChildOrder]
    window_start: datetime
    window_end: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TWAP Executor
# ---------------------------------------------------------------------------

class TWAPExecutor:
    """Time-Weighted Average Price executor.

    Splits a parent order into equal-sized slices distributed uniformly
    across the execution window.

    Parameters
    ----------
    num_slices:
        Number of child order slices to generate.
    """

    def __init__(self, num_slices: int = 10) -> None:
        self.num_slices = max(num_slices, 1)

    def generate(
        self,
        order: Order,
        start: datetime,
        end: datetime,
    ) -> ExecutionPlan:
        """Create equally-spaced child orders across ``[start, end]``."""
        total_qty = order.quantity
        n = self.num_slices
        base_qty = total_qty // n
        remainder = total_qty - base_qty * n

        window = (end - start).total_seconds()
        interval = window / n if n > 1 else 0.0

        children: list[ChildOrder] = []
        for i in range(n):
            qty = base_qty + (Decimal("1") if i < int(remainder) else Decimal("0"))
            if qty <= 0:
                continue

            scheduled = start + timedelta(seconds=interval * i)
            children.append(
                ChildOrder(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    order_type=OrderType.LIMIT if order.limit_price else OrderType.MARKET,
                    scheduled_time=scheduled,
                    limit_price=order.limit_price,
                    slice_index=i,
                    total_slices=n,
                    parent_order_id=order.id,
                    metadata={"algo": ExecutionAlgo.TWAP.value},
                )
            )

        logger.info(
            "twap.plan_created",
            symbol=order.symbol,
            total_qty=str(total_qty),
            slices=len(children),
            window_seconds=round(window, 1),
        )

        return ExecutionPlan(
            parent_order=order,
            algo=ExecutionAlgo.TWAP,
            child_orders=children,
            window_start=start,
            window_end=end,
        )


# ---------------------------------------------------------------------------
# VWAP Executor
# ---------------------------------------------------------------------------

class VWAPExecutor:
    """Volume-Weighted Average Price executor.

    Distributes child order sizes proportionally to an expected volume
    profile so that execution tracks the market's VWAP.

    Parameters
    ----------
    num_slices:
        Number of child order slices to generate.
    """

    def __init__(self, num_slices: int = 10) -> None:
        self.num_slices = max(num_slices, 1)

    def generate(
        self,
        order: Order,
        start: datetime,
        end: datetime,
        volume_profile: list[float] | None = None,
    ) -> ExecutionPlan:
        """Create child orders weighted by *volume_profile*.

        Parameters
        ----------
        volume_profile:
            Relative volume weights per slice.  If ``None``, a U-shaped
            intraday profile is synthesised (higher volume at open/close).
        """
        n = self.num_slices
        total_qty = order.quantity

        if volume_profile is None:
            volume_profile = self._default_profile(n)
        else:
            # Resample to match num_slices
            volume_profile = self._resample_profile(volume_profile, n)

        profile_sum = sum(volume_profile)
        if profile_sum <= 0:
            volume_profile = [1.0] * n
            profile_sum = float(n)

        window = (end - start).total_seconds()
        interval = window / n if n > 1 else 0.0

        children: list[ChildOrder] = []
        allocated = Decimal("0")

        for i in range(n):
            weight = volume_profile[i] / profile_sum
            qty = Decimal(str(round(float(total_qty) * weight)))

            # Last slice absorbs rounding residual
            if i == n - 1:
                qty = total_qty - allocated
            allocated += qty

            if qty <= 0:
                continue

            scheduled = start + timedelta(seconds=interval * i)
            children.append(
                ChildOrder(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    order_type=OrderType.LIMIT if order.limit_price else OrderType.MARKET,
                    scheduled_time=scheduled,
                    limit_price=order.limit_price,
                    slice_index=i,
                    total_slices=n,
                    parent_order_id=order.id,
                    metadata={
                        "algo": ExecutionAlgo.VWAP.value,
                        "volume_weight": round(weight, 6),
                    },
                )
            )

        logger.info(
            "vwap.plan_created",
            symbol=order.symbol,
            total_qty=str(total_qty),
            slices=len(children),
            window_seconds=round(window, 1),
        )

        return ExecutionPlan(
            parent_order=order,
            algo=ExecutionAlgo.VWAP,
            child_orders=children,
            window_start=start,
            window_end=end,
            metadata={"volume_profile": [round(v, 4) for v in volume_profile]},
        )

    # ------------------------------------------------------------------
    # Volume profile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_profile(n: int) -> list[float]:
        """Synthesise a U-shaped intraday volume curve.

        Higher volume at the start and end of the window (mimicking
        open/close patterns), with a trough in the middle.
        """
        if n <= 1:
            return [1.0]
        profile: list[float] = []
        for i in range(n):
            # Normalised position in [0, 1]
            t = i / (n - 1)
            # U-shape: 1 + cos(pi * t - pi) * 0.5
            weight = 1.0 + 0.5 * math.cos(math.pi * (2 * t - 1))
            profile.append(max(weight, 0.1))
        return profile

    @staticmethod
    def _resample_profile(profile: list[float], n: int) -> list[float]:
        """Linearly resample *profile* to length *n*."""
        m = len(profile)
        if m == n:
            return list(profile)
        if m == 0:
            return [1.0] * n

        result: list[float] = []
        for i in range(n):
            pos = i * (m - 1) / (n - 1) if n > 1 else 0
            lo = int(pos)
            hi = min(lo + 1, m - 1)
            frac = pos - lo
            result.append(profile[lo] * (1 - frac) + profile[hi] * frac)
        return result


# ---------------------------------------------------------------------------
# Iceberg Executor
# ---------------------------------------------------------------------------

class IcebergExecutor:
    """Iceberg executor that reveals only a small visible portion of the order.

    Each child order is sized at the ``display_quantity``; the series
    continues until the full quantity is covered.

    Parameters
    ----------
    display_quantity:
        Visible quantity per child slice.
    interval_seconds:
        Minimum time between successive child orders.
    """

    def __init__(
        self,
        display_quantity: Decimal = Decimal("100"),
        interval_seconds: float = 5.0,
    ) -> None:
        self.display_quantity = display_quantity
        self.interval_seconds = interval_seconds

    def generate(
        self,
        order: Order,
        start: datetime,
    ) -> ExecutionPlan:
        """Create iceberg child orders starting at *start*."""
        total_qty = order.quantity
        remaining = total_qty
        children: list[ChildOrder] = []
        i = 0

        while remaining > 0:
            qty = min(self.display_quantity, remaining)
            scheduled = start + timedelta(seconds=self.interval_seconds * i)

            children.append(
                ChildOrder(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    order_type=OrderType.LIMIT if order.limit_price else OrderType.MARKET,
                    scheduled_time=scheduled,
                    limit_price=order.limit_price,
                    slice_index=i,
                    total_slices=-1,  # unknown upfront for iceberg
                    parent_order_id=order.id,
                    metadata={"algo": ExecutionAlgo.ICEBERG.value},
                )
            )
            remaining -= qty
            i += 1

        # Backfill total_slices now that we know the count
        for child in children:
            child.total_slices = len(children)

        end = children[-1].scheduled_time if children else start

        logger.info(
            "iceberg.plan_created",
            symbol=order.symbol,
            total_qty=str(total_qty),
            display_qty=str(self.display_quantity),
            slices=len(children),
        )

        return ExecutionPlan(
            parent_order=order,
            algo=ExecutionAlgo.ICEBERG,
            child_orders=children,
            window_start=start,
            window_end=end,
        )


# ---------------------------------------------------------------------------
# Smart Router
# ---------------------------------------------------------------------------

@dataclass
class RoutingConfig:
    """Per-order routing configuration."""

    algo: ExecutionAlgo = ExecutionAlgo.TWAP
    num_slices: int = 10
    window_seconds: float = 300.0
    display_quantity: Decimal = Decimal("100")
    iceberg_interval_seconds: float = 5.0
    volume_profile: list[float] | None = None


class SmartRouter:
    """Dispatches parent orders to the appropriate execution algorithm.

    Usage::

        router = SmartRouter()
        plan = router.route(order, config=RoutingConfig(algo=ExecutionAlgo.VWAP))
        for child in plan.child_orders:
            ...
    """

    def route(
        self,
        order: Order,
        config: RoutingConfig | None = None,
        start: datetime | None = None,
    ) -> ExecutionPlan:
        """Route *order* through the configured execution algorithm.

        Parameters
        ----------
        order:
            Parent order to be sliced.
        config:
            Routing/algorithm parameters.  Defaults to TWAP with 10 slices
            over a 5-minute window.
        start:
            Execution window start time.  Defaults to ``datetime.utcnow()``.
        """
        cfg = config or RoutingConfig()
        start = start or datetime.utcnow()
        end = start + timedelta(seconds=cfg.window_seconds)

        logger.info(
            "smart_router.routing",
            symbol=order.symbol,
            algo=cfg.algo.value,
            quantity=str(order.quantity),
        )

        if cfg.algo is ExecutionAlgo.TWAP:
            executor = TWAPExecutor(num_slices=cfg.num_slices)
            return executor.generate(order, start, end)

        if cfg.algo is ExecutionAlgo.VWAP:
            executor_vwap = VWAPExecutor(num_slices=cfg.num_slices)
            return executor_vwap.generate(
                order, start, end, volume_profile=cfg.volume_profile,
            )

        if cfg.algo is ExecutionAlgo.ICEBERG:
            executor_ice = IcebergExecutor(
                display_quantity=cfg.display_quantity,
                interval_seconds=cfg.iceberg_interval_seconds,
            )
            return executor_ice.generate(order, start)

        raise ValueError(f"Unsupported execution algorithm: {cfg.algo}")
