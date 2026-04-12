"""Momentum / trend-following strategy.

Generates LONG and SHORT signals based on moving-average crossovers,
ADX trend strength, breakout detection, and volume confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Signal, SignalDirection
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MomentumConfig:
    """Tunable parameters for :class:`MomentumStrategy`."""

    fast_ma_period: int = 10
    slow_ma_period: int = 50
    adx_threshold: float = 15.0
    breakout_lookback: int = 20
    min_volume_ratio: float = 1.0


# ---------------------------------------------------------------------------
# Required feature keys
# ---------------------------------------------------------------------------

_REQUIRED_FEATURES: list[str] = [
    "ema_10",
    "ema_50",
    "adx",
    "atr_14",
    "volume_sma_20",
    "sma_20",
    "sma_200",
    "rsi_14",
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategyABC):
    """Trend-following strategy driven by MA crossovers, ADX, volume, and
    breakout conditions.

    Parameters
    ----------
    symbol:
        Ticker symbol this strategy instance is responsible for.
    config:
        Tunable hyper-parameters; uses sensible defaults when omitted.
    enabled:
        If ``False`` the engine should skip this strategy.
    weight:
        Relative importance when signals are aggregated across strategies.
    """

    def __init__(
        self,
        symbol: str,
        config: MomentumConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="momentum", enabled=enabled, weight=weight)
        self.symbol = symbol
        self.config = config or MomentumConfig()

        # Internal state: track previous bar's MA relationship for crossover
        self._prev_fast_above_slow: bool | None = None

    # ------------------------------------------------------------------
    # BaseStrategyABC interface
    # ------------------------------------------------------------------

    def get_required_features(self) -> list[str]:
        return list(_REQUIRED_FEATURES)

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Evaluate momentum conditions and emit up to one signal."""
        self.validate_features(features)

        try:
            return self._evaluate(features, timestamp)
        except Exception as exc:
            logger.error(
                "momentum.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Momentum strategy failed for {self.symbol}: {exc}",
                details={"symbol": self.symbol},
            ) from exc

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        fast_ma: float = float(features["ema_10"])
        slow_ma: float = float(features["ema_50"])
        adx: float = float(features["adx"])
        atr: float = float(features["atr_14"])
        volume_sma: float = float(features["volume_sma_20"])
        rsi: float = float(features["rsi_14"])
        close: float = float(features.get("close", fast_ma))
        current_volume: float = float(features.get("volume", volume_sma))
        highest_high: float | None = (
            float(features["highest_high"])
            if features.get("highest_high") is not None
            else None
        )
        lowest_low: float | None = (
            float(features["lowest_low"])
            if features.get("lowest_low") is not None
            else None
        )

        fast_above_slow = fast_ma > slow_ma

        # ---- Crossover detection ------------------------------------
        crossover_long = False
        crossover_short = False
        if self._prev_fast_above_slow is not None:
            crossover_long = fast_above_slow and not self._prev_fast_above_slow
            crossover_short = not fast_above_slow and self._prev_fast_above_slow
        self._prev_fast_above_slow = fast_above_slow

        # ---- Trend strength gate ------------------------------------
        trend_strong = adx > self.config.adx_threshold

        # ---- Volume confirmation ------------------------------------
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 0.0
        volume_confirmed = volume_ratio > self.config.min_volume_ratio

        # ---- Breakout detection -------------------------------------
        breakout_long = (
            highest_high is not None and close > highest_high
        )
        breakout_short = (
            lowest_low is not None and close < lowest_low
        )

        # ---- Determine direction ------------------------------------
        direction: SignalDirection | None = None

        if crossover_long and trend_strong and volume_confirmed:
            direction = SignalDirection.LONG
        elif crossover_short and trend_strong and volume_confirmed:
            direction = SignalDirection.SHORT
        elif breakout_long and trend_strong and volume_confirmed:
            direction = SignalDirection.LONG
        elif breakout_short and trend_strong and volume_confirmed:
            direction = SignalDirection.SHORT

        if direction is None:
            logger.debug(
                "momentum.no_signal",
                symbol=self.symbol,
                crossover_long=crossover_long,
                crossover_short=crossover_short,
                trend_strong=trend_strong,
                volume_confirmed=volume_confirmed,
                breakout_long=breakout_long,
                breakout_short=breakout_short,
            )
            return []

        # ---- Confidence & strength ----------------------------------
        confidence = self._compute_confidence(adx, rsi, volume_ratio, direction)
        strength = self._compute_strength(fast_ma, slow_ma, atr)

        signal = Signal(
            symbol=self.symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            strategy_name=self.name,
            timestamp=timestamp,
            metadata={
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "adx": adx,
                "rsi": rsi,
                "atr": atr,
                "volume_ratio": round(volume_ratio, 4),
                "crossover_long": crossover_long,
                "crossover_short": crossover_short,
                "breakout_long": breakout_long,
                "breakout_short": breakout_short,
            },
        )

        logger.info(
            "momentum.signal",
            symbol=self.symbol,
            direction=direction.value,
            confidence=round(confidence, 4),
            strength=round(strength, 4),
        )
        return [signal]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        adx: float,
        rsi: float,
        volume_ratio: float,
        direction: SignalDirection,
    ) -> float:
        """Weighted combination of ADX strength, RSI positioning, and volume.

        Components (weights sum to 1.0):
          - ADX component  (0.4): normalized ADX / 100 (capped at 1)
          - RSI component  (0.3): how extreme RSI is in the signal direction
          - Volume component (0.3): capped volume ratio / 5
        """
        # ADX: higher is more confident, normalise to [0, 1]
        adx_score = min(adx / 100.0, 1.0)

        # RSI: for LONG, higher RSI (momentum) = more confidence unless
        # overbought; for SHORT, lower RSI = more confidence.
        if direction is SignalDirection.LONG:
            rsi_score = min(rsi / 100.0, 1.0)
        else:
            rsi_score = min((100.0 - rsi) / 100.0, 1.0)

        # Volume: higher ratio = more confirmation
        vol_score = min(volume_ratio / 5.0, 1.0)

        confidence = 0.4 * adx_score + 0.3 * rsi_score + 0.3 * vol_score
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _compute_strength(fast_ma: float, slow_ma: float, atr: float) -> float:
        """Normalised distance of fast MA from slow MA, scaled by ATR."""
        if atr <= 0:
            return 0.0
        raw = abs(fast_ma - slow_ma) / atr
        # Sigmoid-style squash into [0, 1] — 2 ATR separation ≈ 0.76
        return min(raw / (1.0 + raw), 1.0)
