"""Dual-momentum (absolute / time-series momentum) strategy.

Implements the *absolute momentum* engine at the heart of Gary Antonacci's Dual
Momentum: hold the asset only while its own trailing return is positive (the
"risk-on" filter), otherwise stand aside in cash.  A long-term moving-average
filter is layered on as a second confirmation of the up-trend.

Antonacci's full Dual Momentum also ranks a *universe* of assets by relative
strength and rotates into the strongest; that cross-sectional step requires a
multi-asset wrapper.  This per-symbol strategy implements the absolute-momentum
sleeve, which on its own historically sidesteps the deepest equity and crypto
draw-downs by going to cash in sustained down-trends.

References
----------
- Gary Antonacci, *Dual Momentum Investing*.
- https://www.quantifiedstrategies.com/dual-momentum-trading-strategy/
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
class DualMomentumConfig:
    """Tunable parameters for :class:`DualMomentumStrategy`."""

    lookback: int = 126
    """Trailing return window in bars (~6 months of daily bars)."""

    trend_sma: int = 200
    """Long-term SMA the price must exceed to confirm the up-trend."""

    min_momentum: float = 0.0
    """Minimum trailing return (fraction) required to be invested."""


_REQUIRED_FEATURES: list[str] = ["atr_14"]


class DualMomentumStrategy(PositionStateStrategy):
    """Absolute / time-series momentum with a long-term trend filter (long/flat).

    Parameters
    ----------
    symbol:
        Ticker symbol this instance trades.
    config:
        Tunable hyper-parameters; sensible defaults when omitted.
    enabled, weight:
        Forwarded to the base strategy.
    """

    def __init__(
        self,
        symbol: str,
        config: DualMomentumConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="dual_momentum", symbol=symbol, enabled=enabled, weight=weight)
        self.config = config or DualMomentumConfig()
        maxlen = max(self.config.lookback, self.config.trend_sma) + 5
        self._closes: deque[float] = deque(maxlen=maxlen)

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
                "dual_momentum.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Dual momentum strategy failed for {self.symbol}: {exc}",
                details={"symbol": self.symbol},
            ) from exc

    def _evaluate(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        cfg = self.config
        close = float(features["close"])
        self._closes.append(close)

        need = max(cfg.lookback + 1, cfg.trend_sma)
        if len(self._closes) < need:
            return []

        closes = list(self._closes)
        past = closes[-(cfg.lookback + 1)]
        momentum = (close / past) - 1.0 if past > 0 else 0.0
        sma_trend = sum(closes[-cfg.trend_sma:]) / cfg.trend_sma

        invest = momentum > cfg.min_momentum and close > sma_trend
        desired = SignalDirection.LONG if invest else SignalDirection.FLAT

        if desired == self._state:
            return []

        strength = clamp01(momentum * 3.0) if desired == SignalDirection.LONG else 1.0
        return self._transition(
            desired,
            timestamp=timestamp,
            strength=max(strength, 0.4),
            confidence=0.6,
            metadata={
                "momentum": round(momentum, 4),
                "sma_trend": round(sma_trend, 4),
                "lookback": cfg.lookback,
            },
        )
