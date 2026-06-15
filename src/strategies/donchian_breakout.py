"""Donchian channel breakout strategy (Turtle-style trend following).

A systematic trend-following strategy based on the Turtle Trading rules of
Richard Dennis: enter when price breaks out to a new N-bar high (long) or
N-bar low (short), and exit when price retraces to an M-bar extreme in the
opposite direction (M < N), with an ATR-based protective stop.

Defaults follow the classic Turtle "System 1": a 20-bar entry channel with a
10-bar exit channel and a 2-ATR stop.  Pure price/volatility logic with no
fitted parameters makes it robust across both equities and crypto.

References
----------
- Curtis Faith, *Way of the Turtle*; the original Dennis/Eckhardt Turtle rules.
- https://www.altrady.com/blog/crypto-trading-strategies/turtle-trading-strategy-rules
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Signal, SignalDirection
from src.strategies.stateful import PositionStateStrategy, clamp01

logger = structlog.get_logger(__name__)


@dataclass
class DonchianConfig:
    """Tunable parameters for :class:`DonchianBreakoutStrategy`."""

    entry_period: int = 20
    """Look-back (bars) for the breakout channel that triggers entries."""

    exit_period: int = 10
    """Look-back (bars) for the opposite-extreme channel that triggers exits."""

    atr_stop_mult: float = 2.0
    """Protective stop distance in ATR units from the entry price."""

    allow_short: bool = True
    """If ``False`` the strategy is long/flat only (e.g. spot-only accounts)."""


_REQUIRED_FEATURES: list[str] = ["atr_14"]


class DonchianBreakoutStrategy(PositionStateStrategy):
    """Turtle-style Donchian channel breakout (trend following).

    Parameters
    ----------
    symbol:
        Ticker symbol this instance trades.
    config:
        Tunable hyper-parameters; sensible Turtle defaults when omitted.
    enabled, weight:
        Forwarded to the base strategy.
    """

    def __init__(
        self,
        symbol: str,
        config: DonchianConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="donchian_breakout", symbol=symbol, enabled=enabled, weight=weight)
        self.config = config or DonchianConfig()

        # Rolling windows of *prior* bar extremes (current bar excluded so a
        # breakout is measured against completed bars only).
        self._highs: deque[float] = deque(maxlen=self.config.entry_period)
        self._lows: deque[float] = deque(maxlen=self.config.entry_period)

        # Entry bookkeeping for the ATR stop.
        self._entry_price: float | None = None
        self._entry_atr: float | None = None

    def get_required_features(self) -> list[str]:
        return list(_REQUIRED_FEATURES)

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        self.validate_features(features)
        try:
            return self._evaluate(features, timestamp)
        except Exception as exc:
            logger.error(
                "donchian_breakout.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Donchian breakout strategy failed for {self.symbol}: {exc}",
                details={"symbol": self.symbol},
            ) from exc

    def _evaluate(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        cfg = self.config
        close = float(features["close"])
        high = float(features["high"])
        low = float(features["low"])
        atr = float(features["atr_14"])

        # Need a full entry channel of *prior* bars before trading.
        if len(self._highs) < cfg.entry_period:
            self._highs.append(high)
            self._lows.append(low)
            return []

        entry_high = max(self._highs)
        entry_low = min(self._lows)
        # Exit channel is the most recent ``exit_period`` prior extremes.
        recent_highs = list(self._highs)[-cfg.exit_period:]
        recent_lows = list(self._lows)[-cfg.exit_period:]
        exit_high = max(recent_highs)
        exit_low = min(recent_lows)

        desired = self._state
        metadata = {
            "entry_high": round(entry_high, 4),
            "entry_low": round(entry_low, 4),
            "exit_high": round(exit_high, 4),
            "exit_low": round(exit_low, 4),
            "atr": round(atr, 4),
        }

        if self._state == SignalDirection.LONG:
            stop_hit = (
                self._entry_price is not None
                and self._entry_atr is not None
                and close < self._entry_price - cfg.atr_stop_mult * self._entry_atr
            )
            if close < exit_low or stop_hit:
                desired = SignalDirection.FLAT
        elif self._state == SignalDirection.SHORT:
            stop_hit = (
                self._entry_price is not None
                and self._entry_atr is not None
                and close > self._entry_price + cfg.atr_stop_mult * self._entry_atr
            )
            if close > exit_high or stop_hit:
                desired = SignalDirection.FLAT

        # From flat (or after the exit above) look for a fresh breakout.
        if desired == SignalDirection.FLAT:
            if close > entry_high:
                desired = SignalDirection.LONG
            elif cfg.allow_short and close < entry_low:
                desired = SignalDirection.SHORT

        signals: list[Signal] = []
        if desired != self._state:
            if desired in (SignalDirection.LONG, SignalDirection.SHORT):
                ref = entry_high if desired == SignalDirection.LONG else entry_low
                breakout = abs(close - ref)
                strength = clamp01(breakout / atr) if atr > 0 else 0.5
                self._entry_price = close
                self._entry_atr = atr
            else:
                strength = 1.0
                self._entry_price = None
                self._entry_atr = None
            signals = self._transition(
                desired,
                timestamp=timestamp,
                strength=max(strength, 0.25),
                confidence=0.6,
                metadata=metadata,
            )

        # Append *after* the decision so the channel always reflects prior bars.
        self._highs.append(high)
        self._lows.append(low)
        return signals
