"""Opening Range Breakout (ORB) strategy.

Generates LONG signals when price breaks above the 30-minute opening range
high with volume confirmation and VWAP filter, within the first 2 hours
of the regular session.

Conditions (all must be true):
  1. Price closes above the opening range high (9:30-10:00 AM EST)
  2. Bar volume >= configurable multiplier of the opening range average volume
  3. Price is above session VWAP
  4. Time is within the signal window (before 11:30 AM EST by default)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any

import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Bar, Signal, SignalDirection
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ORBConfig:
    """Tunable parameters for :class:`ORBStrategy`."""

    volume_multiplier: float = 1.5
    or_start: dt_time = field(default_factory=lambda: dt_time(9, 30))
    or_end: dt_time = field(default_factory=lambda: dt_time(10, 0))
    signal_cutoff: dt_time = field(default_factory=lambda: dt_time(11, 30))
    min_or_bars: int = 1


# ---------------------------------------------------------------------------
# Internal state for a single trading day
# ---------------------------------------------------------------------------

@dataclass
class _DayState:
    """Tracks opening range and VWAP state for a single session."""

    date: datetime | None = None
    or_high: float | None = None
    or_low: float | None = None
    or_volume_sum: float = 0.0
    or_bar_count: int = 0
    or_avg_volume: float | None = None
    or_complete: bool = False
    breached: bool = False

    # Running VWAP accumulators (reset each session)
    vwap_cum_vol: float = 0.0
    vwap_cum_tp_vol: float = 0.0

    @property
    def vwap(self) -> float | None:
        if self.vwap_cum_vol <= 0:
            return None
        return self.vwap_cum_tp_vol / self.vwap_cum_vol

    def update_vwap(self, bar: Bar) -> None:
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self.vwap_cum_vol += bar.volume
        self.vwap_cum_tp_vol += typical_price * bar.volume


# ---------------------------------------------------------------------------
# Required features (we operate on raw bars, but declare these so the
# strategy can also work with the feature pipeline when available)
# ---------------------------------------------------------------------------

_REQUIRED_FEATURES: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ORBStrategy(BaseStrategyABC):
    """Opening Range Breakout strategy.

    This strategy tracks the first 30 minutes of the regular session to
    establish the opening range, then signals a LONG entry when price
    breaks above with volume and VWAP confirmation.

    Parameters
    ----------
    symbol:
        Ticker symbol this strategy instance monitors.
    config:
        Tunable parameters; uses sensible defaults when omitted.
    enabled:
        If ``False`` the engine skips this strategy.
    weight:
        Relative importance when signals are aggregated.
    """

    def __init__(
        self,
        symbol: str,
        config: ORBConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="opening_range_breakout", enabled=enabled, weight=weight)
        self.symbol = symbol
        self.config = config or ORBConfig()
        self._state = _DayState()

    # ------------------------------------------------------------------
    # Public accessors for API / frontend
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "volume_multiplier": self.config.volume_multiplier,
            "or_start": self.config.or_start.strftime("%H:%M"),
            "or_end": self.config.or_end.strftime("%H:%M"),
            "signal_cutoff": self.config.signal_cutoff.strftime("%H:%M"),
        }

    @property
    def state_snapshot(self) -> dict[str, Any]:
        """Return the current day state for the API/frontend."""
        s = self._state
        return {
            "date": s.date.isoformat() if s.date else None,
            "or_high": s.or_high,
            "or_low": s.or_low,
            "or_avg_volume": s.or_avg_volume,
            "or_bar_count": s.or_bar_count,
            "or_complete": s.or_complete,
            "breached": s.breached,
            "vwap": s.vwap,
        }

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
        """Evaluate ORB conditions and emit up to one signal per day."""
        self.validate_features(features)
        try:
            return self._evaluate(features, timestamp)
        except Exception as exc:
            logger.error(
                "orb.signal_generation_failed",
                symbol=self.symbol,
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"ORB strategy failed for {self.symbol}: {exc}",
                details={"symbol": self.symbol},
            ) from exc

    # ------------------------------------------------------------------
    # Bar-level interface (used by the execution engine / scanner)
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> Signal | None:
        """Process a single bar and optionally return a signal.

        This is the primary entry point when the strategy is fed bars
        directly (e.g. from the real-time feed or the ORB scanner).
        """
        features = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        signals = self.generate_signals(features, bar.timestamp)
        return signals[0] if signals else None

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        bar_time = timestamp.time()
        bar_date = timestamp.date()

        # Reset state on a new trading day
        if self._state.date is None or self._state.date != bar_date:
            self._state = _DayState(date=bar_date)

        high = float(features["high"])
        low = float(features["low"])
        close = float(features["close"])
        volume = float(features["volume"])

        # Build a lightweight Bar for VWAP calculation
        _bar = Bar(
            symbol=self.symbol,
            timestamp=timestamp,
            open=float(features["open"]),
            high=high,
            low=low,
            close=close,
            volume=int(volume),
            timeframe=_DUMMY_TF,
        )
        self._state.update_vwap(_bar)

        # --- Phase 1: Build the opening range ---
        if self.config.or_start <= bar_time < self.config.or_end:
            if self._state.or_high is None:
                self._state.or_high = high
                self._state.or_low = low
            else:
                self._state.or_high = max(self._state.or_high, high)
                self._state.or_low = min(self._state.or_low, low)  # type: ignore[arg-type]
            self._state.or_volume_sum += volume
            self._state.or_bar_count += 1
            return []

        # --- Phase 2: Lock in the opening range once it ends ---
        if (
            not self._state.or_complete
            and bar_time >= self.config.or_end
            and self._state.or_bar_count >= self.config.min_or_bars
        ):
            self._state.or_avg_volume = (
                self._state.or_volume_sum / self._state.or_bar_count
            )
            self._state.or_complete = True
            logger.info(
                "orb.range_established",
                symbol=self.symbol,
                or_high=self._state.or_high,
                or_low=self._state.or_low,
                or_avg_volume=round(self._state.or_avg_volume, 2),
                or_bars=self._state.or_bar_count,
            )

        # Can't evaluate until the opening range is established
        if not self._state.or_complete:
            return []

        # --- Phase 3: Check signal window ---
        if bar_time >= self.config.signal_cutoff:
            return []

        # Already fired today
        if self._state.breached:
            return []

        # --- Phase 4: Evaluate breakout conditions ---
        or_high = self._state.or_high
        or_avg_vol = self._state.or_avg_volume
        vwap = self._state.vwap

        if or_high is None or or_avg_vol is None or vwap is None:
            return []

        price_above_or = close > or_high
        volume_confirmed = volume >= self.config.volume_multiplier * or_avg_vol
        above_vwap = close > vwap

        # Bar must close in the top 50% of its range (strong buying pressure)
        bar_range = high - low
        close_in_upper_half = bar_range <= 0 or (close - low) >= 0.5 * bar_range

        if not (price_above_or and volume_confirmed and above_vwap and close_in_upper_half):
            logger.debug(
                "orb.no_signal",
                symbol=self.symbol,
                close=close,
                or_high=or_high,
                volume=volume,
                vol_threshold=round(self.config.volume_multiplier * or_avg_vol, 2),
                vwap=round(vwap, 4),
                price_above_or=price_above_or,
                volume_confirmed=volume_confirmed,
                above_vwap=above_vwap,
                close_in_upper_half=close_in_upper_half,
            )
            return []

        # --- All conditions met ---
        self._state.breached = True

        strength = self._compute_strength(close, or_high, vwap)
        confidence = self._compute_confidence(close, or_high, volume, or_avg_vol, vwap)

        signal = Signal(
            symbol=self.symbol,
            direction=SignalDirection.LONG,
            strength=strength,
            confidence=confidence,
            strategy_name=self.name,
            timestamp=timestamp,
            metadata={
                "or_high": or_high,
                "or_low": self._state.or_low,
                "or_avg_volume": round(or_avg_vol, 2),
                "vwap": round(vwap, 4),
                "close": close,
                "volume": volume,
                "volume_ratio": round(volume / or_avg_vol, 2),
                "close_position_pct": round((close - low) / bar_range * 100, 2) if bar_range > 0 else 100.0,
                "price_vs_or_high_pct": round((close - or_high) / or_high * 100, 4),
                "price_vs_vwap_pct": round((close - vwap) / vwap * 100, 4),
            },
        )

        logger.info(
            "orb.breakout_signal",
            symbol=self.symbol,
            close=close,
            or_high=or_high,
            volume_ratio=round(volume / or_avg_vol, 2),
            vwap=round(vwap, 4),
            confidence=round(confidence, 4),
            strength=round(strength, 4),
        )
        return [signal]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_strength(close: float, or_high: float, vwap: float) -> float:
        """Strength based on how far price has cleared the opening range
        and how far above VWAP it sits.

        Components (equal weight):
          - OR breakout magnitude: (close - or_high) / or_high, capped at 2%
          - VWAP clearance: (close - vwap) / vwap, capped at 1%
        """
        or_pct = (close - or_high) / or_high if or_high > 0 else 0
        vwap_pct = (close - vwap) / vwap if vwap > 0 else 0

        or_score = min(or_pct / 0.02, 1.0)
        vwap_score = min(vwap_pct / 0.01, 1.0)

        return min(max(0.5 * or_score + 0.5 * vwap_score, 0.0), 1.0)

    @staticmethod
    def _compute_confidence(
        close: float,
        or_high: float,
        volume: float,
        or_avg_vol: float,
        vwap: float,
    ) -> float:
        """Confidence based on breakout conviction.

        Components:
          - Volume ratio (0.5): how much volume exceeds the threshold
          - Price above OR high (0.25): breakout magnitude
          - Price above VWAP (0.25): trend confirmation
        """
        vol_ratio = volume / or_avg_vol if or_avg_vol > 0 else 1.0
        vol_score = min(vol_ratio / 5.0, 1.0)

        or_pct = (close - or_high) / or_high if or_high > 0 else 0
        or_score = min(or_pct / 0.02, 1.0)

        vwap_pct = (close - vwap) / vwap if vwap > 0 else 0
        vwap_score = min(vwap_pct / 0.01, 1.0)

        return min(max(0.5 * vol_score + 0.25 * or_score + 0.25 * vwap_score, 0.0), 1.0)


# Sentinel to avoid importing TimeFrame in hot path
from src.core.types import TimeFrame as _TF  # noqa: E402

_DUMMY_TF = _TF.MINUTE_1
