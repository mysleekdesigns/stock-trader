"""Mean reversion strategy.

Generates signals when price deviates significantly from its statistical mean,
expecting reversion.  Uses Bollinger Bands, RSI, and z-score as convergence
indicators.
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
class MeanReversionConfig:
    """Tunable parameters for :class:`MeanReversionStrategy`."""

    bb_period: int = 20
    bb_std: float = 2.0
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5


# ---------------------------------------------------------------------------
# Required feature keys
# ---------------------------------------------------------------------------

_REQUIRED_FEATURES: list[str] = [
    "bollinger_%b",
    "rsi_14",
    "mr_zscore_20",
    "atr_14",
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategyABC):
    """Mean-reversion strategy driven by Bollinger Bands, RSI, and z-score.

    The strategy enters when price is stretched far from its mean (extreme
    z-score, outside Bollinger Bands, extreme RSI) and exits when the z-score
    contracts back toward equilibrium.

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
        config: MeanReversionConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="mean_reversion", enabled=enabled, weight=weight)
        self.symbol = symbol
        self.config = config or MeanReversionConfig()

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
        """Evaluate mean-reversion conditions and emit up to one signal."""
        self.validate_features(features)

        try:
            return self._evaluate(features, timestamp)
        except Exception as exc:
            logger.error(
                "mean_reversion.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Mean reversion strategy failed for {self.symbol}: {exc}",
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
        cfg = self.config

        bb_pct_b: float = float(features["bollinger_%b"])
        rsi: float = float(features["rsi_14"])
        zscore: float = float(features["mr_zscore_20"])
        atr: float = float(features["atr_14"])

        direction: SignalDirection | None = None

        # --- Exit signal: z-score returned within exit band ---------------
        if abs(zscore) < cfg.zscore_exit:
            direction = SignalDirection.FLAT

        # --- Long entry: price stretched below mean -----------------------
        elif (
            bb_pct_b < 0.0
            and rsi < cfg.rsi_oversold
            and zscore < -cfg.zscore_entry
        ):
            direction = SignalDirection.LONG

        # --- Short entry: price stretched above mean ----------------------
        elif (
            bb_pct_b > 1.0
            and rsi > cfg.rsi_overbought
            and zscore > cfg.zscore_entry
        ):
            direction = SignalDirection.SHORT

        if direction is None:
            logger.debug(
                "mean_reversion.no_signal",
                symbol=self.symbol,
                bb_pct_b=round(bb_pct_b, 4),
                rsi=round(rsi, 2),
                zscore=round(zscore, 4),
            )
            return []

        confidence = self._compute_confidence(rsi, zscore, direction)
        strength = self._compute_strength(bb_pct_b, atr)

        signal = Signal(
            symbol=self.symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            strategy_name=self.name,
            timestamp=timestamp,
            metadata={
                "bb_pct_b": round(bb_pct_b, 4),
                "rsi": round(rsi, 2),
                "zscore": round(zscore, 4),
                "atr": round(atr, 4),
            },
        )

        logger.info(
            "mean_reversion.signal",
            symbol=self.symbol,
            direction=direction.value,
            confidence=round(confidence, 4),
            strength=round(strength, 4),
        )
        return [signal]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        rsi: float,
        zscore: float,
        direction: SignalDirection,
    ) -> float:
        """Confidence from RSI extremity (0.5) and z-score magnitude (0.5).

        For exit signals confidence is fixed at 0.5 (neutral).
        """
        if direction is SignalDirection.FLAT:
            return 0.5

        # RSI extremity: how far RSI is into oversold / overbought territory
        if direction is SignalDirection.LONG:
            rsi_score = max(0.0, (self.config.rsi_oversold - rsi) / self.config.rsi_oversold)
        else:
            rsi_score = max(
                0.0,
                (rsi - self.config.rsi_overbought) / (100.0 - self.config.rsi_overbought),
            )
        rsi_score = min(rsi_score, 1.0)

        # Z-score magnitude beyond entry threshold
        zscore_excess = (abs(zscore) - self.config.zscore_entry) / self.config.zscore_entry
        zscore_score = min(max(zscore_excess, 0.0), 1.0)

        confidence = 0.5 * rsi_score + 0.5 * zscore_score
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _compute_strength(bb_pct_b: float, atr: float) -> float:
        """Strength from distance outside Bollinger Bands, scaled by ATR.

        %B < 0 means below lower band; %B > 1 means above upper band.
        The further outside, the stronger the signal.
        """
        if bb_pct_b < 0.0:
            raw = abs(bb_pct_b)
        elif bb_pct_b > 1.0:
            raw = bb_pct_b - 1.0
        else:
            # Inside bands — e.g. exit signal
            return 0.0

        # Sigmoid-style squash into [0, 1]
        return min(raw / (1.0 + raw), 1.0)
