"""Supertrend strategy (ATR-band trend following).

The Supertrend indicator (Olivier Seban) plots a trailing stop line offset from
the candle mid-price by a multiple of ATR.  Price closing above the line marks
an up-trend (go/stay long); closing below marks a down-trend (go/stay short or
flat).  The indicator only flips on a confirmed close beyond the opposite band,
which filters out much of the intrabar noise.

Defaults use the canonical ``multiplier = 3``.  This implementation uses the
engine's 14-period ATR (``atr_14``) for the band width — a common, slightly
smoother variant of the original 10-period setting.

References
----------
- https://www.elearnmarkets.com/blog/supertrend-indicator-strategy-trading/
- https://quantifiedstrategies.substack.com/p/supertrend-indicator
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Signal, SignalDirection
from src.strategies.stateful import PositionStateStrategy, clamp01

logger = structlog.get_logger(__name__)


@dataclass
class SupertrendConfig:
    """Tunable parameters for :class:`SupertrendStrategy`."""

    atr_multiplier: float = 3.0
    """Band offset from the (high+low)/2 mid-price, in ATR units."""

    allow_short: bool = True
    """If ``False`` a down-trend exits to flat instead of going short."""


_REQUIRED_FEATURES: list[str] = ["atr_14"]


class SupertrendStrategy(PositionStateStrategy):
    """Supertrend (ATR-band) trend-following strategy.

    Parameters
    ----------
    symbol:
        Ticker symbol this instance trades.
    config:
        Tunable hyper-parameters; canonical defaults when omitted.
    enabled, weight:
        Forwarded to the base strategy.
    """

    def __init__(
        self,
        symbol: str,
        config: SupertrendConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="supertrend", symbol=symbol, enabled=enabled, weight=weight)
        self.config = config or SupertrendConfig()

        # Supertrend recursion state.
        self._final_upper: float | None = None  # resistance band (down-trend line)
        self._final_lower: float | None = None  # support band (up-trend line)
        self._trend: int = 1  # +1 up-trend, -1 down-trend
        self._prev_close: float | None = None

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
                "supertrend.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Supertrend strategy failed for {self.symbol}: {exc}",
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

        hl2 = (high + low) / 2.0
        basic_upper = hl2 + cfg.atr_multiplier * atr
        basic_lower = hl2 - cfg.atr_multiplier * atr

        # First observed bar: seed bands and remember the close.
        if self._prev_close is None or self._final_upper is None or self._final_lower is None:
            self._final_upper = basic_upper
            self._final_lower = basic_lower
            self._trend = 1
            self._prev_close = close
            return []

        prev_close = self._prev_close
        prev_upper = self._final_upper
        prev_lower = self._final_lower

        # Carry the band forward unless price has invalidated it (standard rule).
        keep_upper = basic_upper >= prev_upper and prev_close <= prev_upper
        keep_lower = basic_lower <= prev_lower and prev_close >= prev_lower
        final_upper = prev_upper if keep_upper else basic_upper
        final_lower = prev_lower if keep_lower else basic_lower

        prev_trend = self._trend
        if prev_trend == 1:
            trend = -1 if close < final_lower else 1
        else:
            trend = 1 if close > final_upper else -1

        # Persist recursion state for the next bar.
        self._final_upper = final_upper
        self._final_lower = final_lower
        self._trend = trend
        self._prev_close = close

        line = final_lower if trend == 1 else final_upper
        strength = clamp01(abs(close - line) / atr) if atr > 0 else 0.5
        metadata = {
            "trend": trend,
            "supertrend_line": round(line, 4),
            "final_upper": round(final_upper, 4),
            "final_lower": round(final_lower, 4),
            "atr": round(atr, 4),
        }

        if trend == 1:
            desired = SignalDirection.LONG
        else:
            desired = SignalDirection.SHORT if cfg.allow_short else SignalDirection.FLAT

        return self._transition(
            desired,
            timestamp=timestamp,
            strength=max(strength, 0.3),
            confidence=0.6,
            metadata=metadata,
        )
