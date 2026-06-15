"""Connors RSI-2 mean-reversion strategy.

Larry Connors' classic short-term mean-reversion system: buy short-term
weakness *within* a longer-term up-trend, and exit on the snap-back.

Rules (long-only):
- Trend filter: only trade when ``close`` is above its 200-period SMA.
- Entry: buy when the 2-period RSI drops below ``rsi_buy`` (default 5).
- Exit: sell when ``close`` closes back above its 5-period SMA, or the 2-period
  RSI rises above ``rsi_exit`` (default 65), whichever comes first.
- Safety exit: close if price falls below the 200-SMA (trend broke).

Connors/Alvarez reported win rates well above 70% on broad equity indices with
an average hold of ~2 bars; the edge is strongest on index ETFs and weakens on
single names and trend-dominated crypto, which the backtest reflects.

References
----------
- https://www.quantifiedstrategies.com/rsi-2-strategy/
- https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Signal, SignalDirection
from src.strategies.stateful import PositionStateStrategy, clamp01, wilder_rsi

logger = structlog.get_logger(__name__)


@dataclass
class ConnorsRSI2Config:
    """Tunable parameters for :class:`ConnorsRSI2Strategy`."""

    rsi_period: int = 2
    rsi_buy: float = 5.0
    rsi_exit: float = 65.0
    trend_sma: int = 200
    exit_sma: int = 5


_REQUIRED_FEATURES: list[str] = ["atr_14"]


class ConnorsRSI2Strategy(PositionStateStrategy):
    """Connors RSI-2 short-term mean reversion (long/flat).

    Parameters
    ----------
    symbol:
        Ticker symbol this instance trades.
    config:
        Tunable hyper-parameters; Connors' defaults when omitted.
    enabled, weight:
        Forwarded to the base strategy.
    """

    def __init__(
        self,
        symbol: str,
        config: ConnorsRSI2Config | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="connors_rsi2", symbol=symbol, enabled=enabled, weight=weight)
        self.config = config or ConnorsRSI2Config()
        # Keep enough closes for the RSI seed and the 200-SMA trend filter.
        maxlen = max(self.config.trend_sma, self.config.rsi_period + 1, self.config.exit_sma) + 5
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
                "connors_rsi2.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Connors RSI-2 strategy failed for {self.symbol}: {exc}",
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

        # Need the full trend window before trading.
        if len(self._closes) < cfg.trend_sma:
            return []

        closes = list(self._closes)
        sma_trend = sum(closes[-cfg.trend_sma:]) / cfg.trend_sma
        sma_exit = sum(closes[-cfg.exit_sma:]) / cfg.exit_sma
        rsi = wilder_rsi(closes, cfg.rsi_period)
        if rsi is None:
            return []

        uptrend = close > sma_trend
        desired = self._state

        if self._state == SignalDirection.LONG:
            if (close > sma_exit) or (rsi > cfg.rsi_exit) or (not uptrend):
                desired = SignalDirection.FLAT
        elif uptrend and rsi < cfg.rsi_buy:
            desired = SignalDirection.LONG

        if desired == self._state:
            return []

        # Deeper oversold readings get a stronger entry conviction.
        if desired == SignalDirection.LONG:
            strength = clamp01((cfg.rsi_buy - rsi) / cfg.rsi_buy + 0.5)
        else:
            strength = 1.0
        return self._transition(
            desired,
            timestamp=timestamp,
            strength=max(strength, 0.4),
            confidence=0.65,
            metadata={
                "rsi_2": round(rsi, 2),
                "sma_trend": round(sma_trend, 4),
                "sma_exit": round(sma_exit, 4),
                "uptrend": uptrend,
            },
        )
