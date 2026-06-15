"""MACD trend-following strategy with an optional long-term trend filter.

Classic MACD(12, 26, 9): go long when the MACD line crosses above its signal
line, go short (or flat) when it crosses below.  An optional 200-period SMA
trend filter only permits longs above the SMA and shorts below it, which sharply
reduces whipsaw losses in ranging markets and is the difference between a noisy
oscillator and a usable trend strategy.

MACD lines are computed incrementally from the raw close series, so the strategy
needs no feature beyond the engine's always-present OHLCV.

References
----------
- Gerald Appel's MACD; the 12/26/9 EMA configuration.
- https://www.quantifiedstrategies.com/macd-trading-strategy/
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Signal, SignalDirection
from src.strategies.stateful import PositionStateStrategy, _Ema, clamp01

logger = structlog.get_logger(__name__)


@dataclass
class MACDTrendConfig:
    """Tunable parameters for :class:`MACDTrendStrategy`."""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    use_trend_filter: bool = True
    trend_sma: int = 200
    allow_short: bool = True


_REQUIRED_FEATURES: list[str] = ["atr_14"]


class MACDTrendStrategy(PositionStateStrategy):
    """MACD(12,26,9) crossover trend following with a 200-SMA regime filter.

    Parameters
    ----------
    symbol:
        Ticker symbol this instance trades.
    config:
        Tunable hyper-parameters; classic defaults when omitted.
    enabled, weight:
        Forwarded to the base strategy.
    """

    def __init__(
        self,
        symbol: str,
        config: MACDTrendConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="macd_trend", symbol=symbol, enabled=enabled, weight=weight)
        self.config = config or MACDTrendConfig()

        self._ema_fast = _Ema(self.config.fast_period)
        self._ema_slow = _Ema(self.config.slow_period)
        self._signal_ema = _Ema(self.config.signal_period)
        self._prev_macd_above: bool | None = None
        self._bars_seen = 0
        self._closes: deque[float] = deque(maxlen=self.config.trend_sma + 5)

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
                "macd_trend.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"MACD trend strategy failed for {self.symbol}: {exc}",
                details={"symbol": self.symbol},
            ) from exc

    def _evaluate(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        cfg = self.config
        close = float(features["close"])
        atr = float(features["atr_14"])

        self._bars_seen += 1
        self._closes.append(close)

        ema_fast = self._ema_fast.update(close)
        ema_slow = self._ema_slow.update(close)
        macd = ema_fast - ema_slow
        signal_line = self._signal_ema.update(macd)
        macd_above = macd > signal_line

        # Warm up the slow EMA + signal before acting on the first crossover.
        warmup = cfg.slow_period + cfg.signal_period
        if self._bars_seen < warmup:
            self._prev_macd_above = macd_above
            return []

        cross_up = self._prev_macd_above is False and macd_above
        cross_down = self._prev_macd_above is True and not macd_above
        self._prev_macd_above = macd_above

        # Trend regime filter.
        above_trend: bool | None = None
        if cfg.use_trend_filter and len(self._closes) >= cfg.trend_sma:
            sma_trend = sum(list(self._closes)[-cfg.trend_sma:]) / cfg.trend_sma
            above_trend = close > sma_trend

        def filter_ok(direction: SignalDirection) -> bool:
            if above_trend is None:
                return True
            return above_trend if direction == SignalDirection.LONG else not above_trend

        desired = self._state

        if cross_up:
            desired = (
                SignalDirection.LONG if filter_ok(SignalDirection.LONG) else SignalDirection.FLAT
            )
        elif cross_down:
            if cfg.allow_short and filter_ok(SignalDirection.SHORT):
                desired = SignalDirection.SHORT
            else:
                desired = SignalDirection.FLAT
        elif above_trend is not None:
            # No crossover this bar — exit only if the trend filter flipped
            # against an open position.
            long_against_trend = self._state == SignalDirection.LONG and not above_trend
            short_against_trend = self._state == SignalDirection.SHORT and above_trend
            if long_against_trend or short_against_trend:
                desired = SignalDirection.FLAT

        if desired == self._state:
            return []

        strength = clamp01(abs(macd - signal_line) / atr) if atr > 0 else 0.5
        return self._transition(
            desired,
            timestamp=timestamp,
            strength=max(strength, 0.3),
            confidence=0.6,
            metadata={
                "macd": round(macd, 4),
                "macd_signal": round(signal_line, 4),
                "above_trend": above_trend,
                "atr": round(atr, 4),
            },
        )
