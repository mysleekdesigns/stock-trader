"""Stop-loss management.

Provides trailing ATR stops, time-based exits, and a manager that tracks
stop states for all open positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import structlog

from src.core.types import Bar, OrderSide, Position

logger = structlog.get_logger(__name__)


class StopLoss(ABC):
    """Abstract base for a stop-loss rule."""

    @abstractmethod
    def check(self, position: Position, current_bar: Bar) -> bool:
        """Return ``True`` if the position should be exited."""


# ---------------------------------------------------------------------------
# Trailing ATR stop
# ---------------------------------------------------------------------------


@dataclass
class TrailingATRStop(StopLoss):
    """Trailing stop keyed off Average True Range.

    For longs the stop trails below the highest price observed since entry.
    For shorts the stop trails above the lowest price observed since entry.
    """

    atr_multiplier: float = 2.5

    # Internal tracking per symbol
    _peaks: dict[str, float] = field(default_factory=dict, repr=False)
    _troughs: dict[str, float] = field(default_factory=dict, repr=False)
    _atrs: dict[str, float] = field(default_factory=dict, repr=False)

    def register(self, symbol: str, initial_price: float, atr: float) -> None:
        """Initialise tracking for a new position."""
        self._peaks[symbol] = initial_price
        self._troughs[symbol] = initial_price
        self._atrs[symbol] = atr
        logger.debug(
            "stop_loss.trailing_atr_register",
            symbol=symbol,
            initial_price=initial_price,
            atr=atr,
        )

    def update(self, bar: Bar, atr: float) -> None:
        """Update peak/trough and ATR for an existing position."""
        symbol = bar.symbol
        self._atrs[symbol] = atr

        current_peak = self._peaks.get(symbol)
        if current_peak is not None:
            self._peaks[symbol] = max(current_peak, bar.high)

        current_trough = self._troughs.get(symbol)
        if current_trough is not None:
            self._troughs[symbol] = min(current_trough, bar.low)

    def remove(self, symbol: str) -> None:
        """Remove tracking state for a closed position."""
        self._peaks.pop(symbol, None)
        self._troughs.pop(symbol, None)
        self._atrs.pop(symbol, None)

    def check(self, position: Position, current_bar: Bar) -> bool:
        """Return ``True`` if the trailing stop has been triggered."""
        symbol = position.symbol
        atr = self._atrs.get(symbol)
        if atr is None or atr <= 0:
            return False

        stop_distance = self.atr_multiplier * atr

        if position.side == OrderSide.BUY:
            peak = self._peaks.get(symbol, float(position.avg_entry_price))
            stop_level = peak - stop_distance
            triggered = current_bar.low <= stop_level
        else:
            trough = self._troughs.get(symbol, float(position.avg_entry_price))
            stop_level = trough + stop_distance
            triggered = current_bar.high >= stop_level

        if triggered:
            logger.info(
                "stop_loss.trailing_atr_triggered",
                symbol=symbol,
                side=position.side.value,
                stop_level=round(stop_level, 4),
                bar_low=current_bar.low,
                bar_high=current_bar.high,
            )
        return triggered


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------


@dataclass
class TimeBasedExit(StopLoss):
    """Forces exit after a position has been held for too many days."""

    max_hold_days: int = 20

    def check(self, position: Position, current_bar: Bar) -> bool:
        """Return ``True`` if the holding period has exceeded the limit."""
        return self.check_time(position, current_bar.timestamp)

    def check_time(self, position: Position, current_time: datetime) -> bool:
        """Check using an explicit timestamp (useful when no bar is available)."""
        days_held = (current_time - position.opened_at).days
        if days_held >= self.max_hold_days:
            logger.info(
                "stop_loss.time_exit_triggered",
                symbol=position.symbol,
                days_held=days_held,
                max_hold_days=self.max_hold_days,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Stop-loss manager
# ---------------------------------------------------------------------------


class StopLossManager:
    """Manages stop-loss state across all open positions.

    Combines :class:`TrailingATRStop` and :class:`TimeBasedExit` into a
    single update loop.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.5,
        max_hold_days: int = 20,
    ) -> None:
        self._trailing_stop = TrailingATRStop(atr_multiplier=atr_multiplier)
        self._time_exit = TimeBasedExit(max_hold_days=max_hold_days)
        self._positions: dict[str, Position] = {}

    def register_position(self, position: Position, atr: float) -> None:
        """Start tracking stops for a newly opened position."""
        self._positions[position.symbol] = position
        self._trailing_stop.register(
            symbol=position.symbol,
            initial_price=float(position.avg_entry_price),
            atr=atr,
        )
        logger.info(
            "stop_loss_manager.registered",
            symbol=position.symbol,
            side=position.side.value,
            atr=atr,
        )

    def remove_position(self, symbol: str) -> None:
        """Stop tracking a position (e.g. after exit)."""
        self._positions.pop(symbol, None)
        self._trailing_stop.remove(symbol)
        logger.info("stop_loss_manager.removed", symbol=symbol)

    def update_all(
        self,
        bars: dict[str, Bar],
        atrs: dict[str, float],
    ) -> list[str]:
        """Update all stops and return a list of symbols that should be exited.

        Parameters
        ----------
        bars:
            Latest bar per symbol.
        atrs:
            Latest ATR value per symbol.

        Returns
        -------
        list[str]
            Symbols whose stop conditions have been triggered.
        """
        exits: list[str] = []

        for symbol, position in list(self._positions.items()):
            bar = bars.get(symbol)
            if bar is None:
                continue

            # Update trailing stop state
            atr = atrs.get(symbol, 0.0)
            if atr > 0:
                self._trailing_stop.update(bar, atr)

            # Check trailing ATR stop
            if self._trailing_stop.check(position, bar):
                exits.append(symbol)
                continue

            # Check time-based exit
            if self._time_exit.check(position, bar):
                exits.append(symbol)
                continue

        if exits:
            logger.info("stop_loss_manager.exits_triggered", symbols=exits)

        return exits
