"""Shared scaffolding for stateful, transition-based strategies.

The backtest/execution engine sizes a *fresh* order from every non-FLAT signal
it receives (see ``BacktestEngine._create_orders_from_signals``).  A strategy
that re-emits its desired position on every bar would therefore pyramid into
the position bar-after-bar until cash runs out.  The built-in event strategies
(momentum, mean-reversion, ORB) avoid this by emitting a signal *only when their
desired position changes*.

:class:`PositionStateStrategy` centralises that pattern.  A subclass computes the
*desired* :class:`SignalDirection` each bar and calls :meth:`_transition`, which
returns the minimal list of signals needed to move the position from its current
state to the desired one (closing first on a reversal) and remembers the new
state.  Subclasses never emit signals directly.

This module also provides small, dependency-free incremental indicator helpers
(Wilder RSI, EMA) so strategies can compute their own indicators from the raw
OHLCV the engine always injects, without coupling to the feature registry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from src.core.types import Signal, SignalDirection
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)


def clamp01(value: float) -> float:
    """Clamp *value* into the ``[0.0, 1.0]`` range expected of signal scores."""
    if value != value:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, float(value)))


def wilder_rsi(closes: list[float], period: int) -> float | None:
    """Compute Wilder's RSI of *period* over *closes* (oldest → newest).

    Returns ``None`` when there are not enough observations.  Wilder's
    smoothing is seeded with the simple average of the first *period* gains and
    losses, then smoothed recursively — the standard RSI definition used by
    Connors' RSI(2) work.
    """
    if len(closes) < period + 1:
        return None

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class _Ema:
    """Incremental exponential moving average updated one sample at a time."""

    __slots__ = ("_alpha", "value", "_seeded")

    def __init__(self, period: int) -> None:
        self._alpha = 2.0 / (period + 1.0)
        self.value: float | None = None
        self._seeded = False

    def update(self, sample: float) -> float:
        if not self._seeded:
            self.value = sample
            self._seeded = True
        else:
            self.value = sample * self._alpha + self.value * (1.0 - self._alpha)
        return self.value


class PositionStateStrategy(BaseStrategyABC):
    """Base class for strategies that manage a single long/short/flat position.

    Subclasses implement :meth:`get_required_features` and a ``_evaluate`` that
    computes the desired :class:`SignalDirection`, then drive position changes
    through :meth:`_transition`.

    Parameters
    ----------
    name:
        Unique strategy name.
    symbol:
        Ticker symbol this instance trades.
    enabled, weight:
        Forwarded to :class:`BaseStrategyABC`.
    """

    def __init__(
        self,
        name: str,
        symbol: str,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, enabled=enabled, weight=weight)
        self.symbol = symbol
        self._state: SignalDirection = SignalDirection.FLAT

    @property
    def state(self) -> SignalDirection:
        """The position state the strategy currently believes it holds."""
        return self._state

    def _transition(
        self,
        desired: SignalDirection,
        *,
        timestamp: datetime,
        strength: float = 1.0,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Return the minimal signals to move from the current state to *desired*.

        - No change → ``[]`` (the position is simply held; emitting nothing is
          what prevents the engine from pyramiding every bar).
        - Open from flat → a single directional signal.
        - Close to flat → a single FLAT signal.
        - Reversal (long↔short) → a FLAT signal (close) followed by the new
          directional signal (open).  The engine fills these on the next bar in
          order, so the existing position is closed before the new one opens.
        """
        current = self._state
        if desired == current:
            return []

        meta = dict(metadata or {})
        strength = clamp01(strength)
        confidence = clamp01(confidence)
        signals: list[Signal] = []

        if current != SignalDirection.FLAT:
            signals.append(
                Signal(
                    symbol=self.symbol,
                    direction=SignalDirection.FLAT,
                    strength=1.0,
                    confidence=confidence,
                    strategy_name=self.name,
                    timestamp=timestamp,
                    metadata={**meta, "action": "close", "from": current.value},
                )
            )

        if desired != SignalDirection.FLAT:
            signals.append(
                Signal(
                    symbol=self.symbol,
                    direction=desired,
                    strength=strength,
                    confidence=confidence,
                    strategy_name=self.name,
                    timestamp=timestamp,
                    metadata={**meta, "action": "open", "to": desired.value},
                )
            )

        self._state = desired
        logger.info(
            "strategy.transition",
            strategy=self.name,
            symbol=self.symbol,
            frm=current.value,
            to=desired.value,
        )
        return signals
