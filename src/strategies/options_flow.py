"""Options flow strategy.

Detects unusual options activity -- abnormal volume, large premiums, and
put/call ratio skew -- as a signal for directional moves in the underlying.

This is largely a placeholder implementation with proper structure; the
concrete data feeds for real-time options flow would be wired at integration
time.
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
class OptionsFlowConfig:
    """Tunable parameters for :class:`OptionsFlowStrategy`."""

    unusual_volume_threshold: float = 3.0
    min_premium: float = 100_000.0
    smart_money_weight: float = 0.7
    pc_ratio_bullish: float = 0.5
    pc_ratio_bearish: float = 1.5
    avg_volume_lookback: int = 20


# ---------------------------------------------------------------------------
# Required feature keys (placeholder features)
# ---------------------------------------------------------------------------

_REQUIRED_FEATURES: list[str] = [
    "options_call_volume",
    "options_put_volume",
    "options_premium",
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class OptionsFlowStrategy(BaseStrategyABC):
    """Options-flow driven directional strategy.

    Monitors call/put volume ratios, unusual volume spikes, and premium size
    to infer smart-money positioning and generate directional signals on the
    underlying security.

    Parameters
    ----------
    symbol:
        Ticker symbol of the underlying this strategy tracks.
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
        config: OptionsFlowConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="options_flow", enabled=enabled, weight=weight)
        self.symbol = symbol
        self.config = config or OptionsFlowConfig()

        # Rolling average volumes for unusual-activity detection
        self._avg_call_volume: float = 0.0
        self._avg_put_volume: float = 0.0
        self._volume_history: list[tuple[float, float]] = []

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
        """Evaluate options-flow conditions and emit up to one signal."""
        self.validate_features(features)

        try:
            return self._evaluate(features, timestamp)
        except Exception as exc:
            logger.error(
                "options_flow.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Options flow strategy failed for {self.symbol}: {exc}",
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

        call_volume: float = float(features["options_call_volume"])
        put_volume: float = float(features["options_put_volume"])
        premium: float = float(features["options_premium"])

        # Update rolling averages
        self._update_volume_history(call_volume, put_volume)

        total_volume = call_volume + put_volume

        # --- Unusual activity detection --------------------------------
        call_ratio = (
            call_volume / self._avg_call_volume
            if self._avg_call_volume > 0
            else 0.0
        )
        put_ratio = (
            put_volume / self._avg_put_volume
            if self._avg_put_volume > 0
            else 0.0
        )

        unusual_call = call_ratio > cfg.unusual_volume_threshold
        unusual_put = put_ratio > cfg.unusual_volume_threshold

        # --- Premium filter --------------------------------------------
        large_premium = premium >= cfg.min_premium

        # --- Put/call ratio analysis -----------------------------------
        pc_ratio = put_volume / call_volume if call_volume > 0 else float("inf")

        # --- Smart money flow detection (large block proxy) -------------
        smart_money_bullish = unusual_call and large_premium
        smart_money_bearish = unusual_put and large_premium

        # --- Direction determination -----------------------------------
        direction: SignalDirection | None = None

        if smart_money_bullish and pc_ratio < cfg.pc_ratio_bullish:
            direction = SignalDirection.LONG
        elif smart_money_bearish and pc_ratio > cfg.pc_ratio_bearish:
            direction = SignalDirection.SHORT
        elif unusual_call and not unusual_put and large_premium:
            direction = SignalDirection.LONG
        elif unusual_put and not unusual_call and large_premium:
            direction = SignalDirection.SHORT

        if direction is None:
            logger.debug(
                "options_flow.no_signal",
                symbol=self.symbol,
                call_ratio=round(call_ratio, 2),
                put_ratio=round(put_ratio, 2),
                pc_ratio=round(pc_ratio, 4) if pc_ratio != float("inf") else "inf",
                premium=premium,
                unusual_call=unusual_call,
                unusual_put=unusual_put,
            )
            return []

        confidence = self._compute_confidence(
            call_ratio, put_ratio, pc_ratio, premium, direction,
        )
        strength = self._compute_strength(
            call_ratio, put_ratio, premium, direction,
        )

        signal = Signal(
            symbol=self.symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            strategy_name=self.name,
            timestamp=timestamp,
            metadata={
                "call_volume": call_volume,
                "put_volume": put_volume,
                "premium": premium,
                "pc_ratio": round(pc_ratio, 4) if pc_ratio != float("inf") else None,
                "call_ratio_vs_avg": round(call_ratio, 4),
                "put_ratio_vs_avg": round(put_ratio, 4),
                "smart_money_bullish": smart_money_bullish,
                "smart_money_bearish": smart_money_bearish,
            },
        )

        logger.info(
            "options_flow.signal",
            symbol=self.symbol,
            direction=direction.value,
            confidence=round(confidence, 4),
            strength=round(strength, 4),
        )
        return [signal]

    # ------------------------------------------------------------------
    # Rolling volume tracking
    # ------------------------------------------------------------------

    def _update_volume_history(
        self,
        call_volume: float,
        put_volume: float,
    ) -> None:
        """Maintain a rolling window of call/put volumes."""
        self._volume_history.append((call_volume, put_volume))
        if len(self._volume_history) > self.config.avg_volume_lookback:
            self._volume_history = self._volume_history[
                -self.config.avg_volume_lookback :
            ]

        if self._volume_history:
            self._avg_call_volume = sum(
                v[0] for v in self._volume_history
            ) / len(self._volume_history)
            self._avg_put_volume = sum(
                v[1] for v in self._volume_history
            ) / len(self._volume_history)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        call_ratio: float,
        put_ratio: float,
        pc_ratio: float,
        premium: float,
        direction: SignalDirection,
    ) -> float:
        """Confidence from volume ratio magnitude and premium size.

        Components (weights):
          - Volume ratio (0.4): how many multiples above average
          - Premium size (0.3): scaled by min_premium threshold
          - Smart money weight (0.3): configured conviction multiplier
        """
        cfg = self.config

        # Volume ratio score
        relevant_ratio = call_ratio if direction is SignalDirection.LONG else put_ratio
        vol_score = min((relevant_ratio - 1.0) / (cfg.unusual_volume_threshold * 2), 1.0)
        vol_score = max(vol_score, 0.0)

        # Premium score
        prem_score = min(premium / (cfg.min_premium * 5), 1.0)

        # Smart money component
        sm_score = cfg.smart_money_weight

        confidence = 0.4 * vol_score + 0.3 * prem_score + 0.3 * sm_score
        return min(max(confidence, 0.0), 1.0)

    def _compute_strength(
        self,
        call_ratio: float,
        put_ratio: float,
        premium: float,
        direction: SignalDirection,
    ) -> float:
        """Strength from the magnitude of volume spike and premium."""
        relevant_ratio = call_ratio if direction is SignalDirection.LONG else put_ratio

        # Sigmoid-style squash: 5x average volume -> ~0.83
        ratio_norm = relevant_ratio / (1.0 + relevant_ratio) if relevant_ratio > 0 else 0.0
        prem_norm = min(premium / (self.config.min_premium * 10), 1.0)

        return min(0.6 * ratio_norm + 0.4 * prem_norm, 1.0)
