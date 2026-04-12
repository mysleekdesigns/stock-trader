"""Signal aggregation and filtering utilities.

:class:`SignalAggregator` combines signals from multiple strategies into a
single consensus signal.  :class:`SignalFilter` applies quality gates
(minimum confidence, strength, debounce) so that only actionable signals
reach the execution layer.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import structlog

from src.core.types import Signal, SignalDirection

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SignalAggregator
# ---------------------------------------------------------------------------

class SignalAggregator:
    """Combine signals from multiple strategies into a single consensus signal.

    Usage::

        agg = SignalAggregator(symbol="AAPL")
        agg.add_signal(signal_a)
        agg.add_signal(signal_b)
        result = agg.aggregate(method="weighted_average")
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._signals: list[Signal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_signal(self, signal: Signal) -> None:
        """Append a signal for later aggregation."""
        self._signals.append(signal)
        logger.debug(
            "aggregator.add_signal",
            symbol=self.symbol,
            strategy=signal.strategy_name,
            direction=signal.direction.value,
            confidence=signal.confidence,
            strength=signal.strength,
        )

    def aggregate(self, method: str = "weighted_average") -> Signal | None:
        """Return a single consensus :class:`Signal`, or *None* if the
        basket is empty or signals cancel out.

        Parameters
        ----------
        method:
            One of ``"weighted_average"``, ``"majority_vote"``,
            ``"max_confidence"``.
        """
        if not self._signals:
            logger.debug("aggregator.empty", symbol=self.symbol)
            return None

        dispatch = {
            "weighted_average": self._weighted_average,
            "majority_vote": self._majority_vote,
            "max_confidence": self._max_confidence,
        }

        if method not in dispatch:
            raise ValueError(
                f"Unknown aggregation method '{method}'. "
                f"Choose from {sorted(dispatch)}."
            )

        result = dispatch[method]()
        if result is not None:
            logger.info(
                "aggregator.result",
                symbol=self.symbol,
                method=method,
                direction=result.direction.value,
                confidence=round(result.confidence, 4),
                strength=round(result.strength, 4),
                input_count=len(self._signals),
            )
        return result

    def clear(self) -> None:
        """Reset the signal basket."""
        self._signals.clear()

    # ------------------------------------------------------------------
    # Private aggregation methods
    # ------------------------------------------------------------------

    def _weighted_average(self) -> Signal | None:
        """Weight each signal by ``confidence * strength``, separate into
        LONG/SHORT buckets, and produce a net signal.
        """
        long_weight = 0.0
        short_weight = 0.0

        for sig in self._signals:
            w = sig.confidence * sig.strength
            if sig.direction is SignalDirection.LONG:
                long_weight += w
            elif sig.direction is SignalDirection.SHORT:
                short_weight += w
            # FLAT signals are neutral and ignored

        total = long_weight + short_weight
        if total == 0.0:
            return None

        if long_weight > short_weight:
            direction = SignalDirection.LONG
            net_strength = (long_weight - short_weight) / total
        elif short_weight > long_weight:
            direction = SignalDirection.SHORT
            net_strength = (short_weight - long_weight) / total
        else:
            # Perfect tie — emit FLAT
            direction = SignalDirection.FLAT
            net_strength = 0.0

        net_confidence = total / len(self._signals) if self._signals else 0.0
        # Clamp to [0, 1]
        net_confidence = min(max(net_confidence, 0.0), 1.0)
        net_strength = min(max(net_strength, 0.0), 1.0)

        return Signal(
            symbol=self.symbol,
            direction=direction,
            strength=net_strength,
            confidence=net_confidence,
            strategy_name="aggregator:weighted_average",
            timestamp=datetime.utcnow(),
            metadata=self._build_metadata("weighted_average"),
        )

    def _majority_vote(self) -> Signal | None:
        """Each signal is one vote for its direction.  The direction with
        the most votes wins; confidence/strength are averaged from winning
        signals.
        """
        votes: dict[SignalDirection, list[Signal]] = defaultdict(list)
        for sig in self._signals:
            if sig.direction is not SignalDirection.FLAT:
                votes[sig.direction].append(sig)

        if not votes:
            return None

        winner = max(votes, key=lambda d: len(votes[d]))
        winning_signals = votes[winner]

        avg_confidence = sum(s.confidence for s in winning_signals) / len(winning_signals)
        avg_strength = sum(s.strength for s in winning_signals) / len(winning_signals)

        return Signal(
            symbol=self.symbol,
            direction=winner,
            strength=min(max(avg_strength, 0.0), 1.0),
            confidence=min(max(avg_confidence, 0.0), 1.0),
            strategy_name="aggregator:majority_vote",
            timestamp=datetime.utcnow(),
            metadata=self._build_metadata("majority_vote"),
        )

    def _max_confidence(self) -> Signal | None:
        """Pick the single signal with the highest confidence."""
        best = max(self._signals, key=lambda s: s.confidence)
        return Signal(
            symbol=self.symbol,
            direction=best.direction,
            strength=best.strength,
            confidence=best.confidence,
            strategy_name=f"aggregator:max_confidence({best.strategy_name})",
            timestamp=datetime.utcnow(),
            metadata=self._build_metadata("max_confidence"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, method: str) -> dict[str, Any]:
        return {
            "aggregation_method": method,
            "input_signals": [
                {
                    "strategy": s.strategy_name,
                    "direction": s.direction.value,
                    "confidence": s.confidence,
                    "strength": s.strength,
                }
                for s in self._signals
            ],
        }


# ---------------------------------------------------------------------------
# SignalFilter
# ---------------------------------------------------------------------------

class SignalFilter:
    """Apply quality gates to signals before they reach execution.

    Parameters
    ----------
    min_confidence:
        Discard signals below this confidence threshold.
    min_strength:
        Discard signals below this strength threshold.
    debounce_seconds:
        Suppress duplicate signals (same symbol + direction) that arrive
        within this window.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.0,
        min_strength: float = 0.0,
        debounce_seconds: float = 0.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_strength = min_strength
        self.debounce_seconds = debounce_seconds
        # key = (symbol, direction_value), value = monotonic timestamp
        self._last_emitted: dict[tuple[str, str], float] = {}

    def filter(self, signal: Signal) -> Signal | None:
        """Return *signal* if it passes all gates, otherwise ``None``."""

        # --- confidence gate ---
        if signal.confidence < self.min_confidence:
            logger.debug(
                "signal_filter.low_confidence",
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                confidence=signal.confidence,
                threshold=self.min_confidence,
            )
            return None

        # --- strength gate ---
        if signal.strength < self.min_strength:
            logger.debug(
                "signal_filter.low_strength",
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                strength=signal.strength,
                threshold=self.min_strength,
            )
            return None

        # --- debounce gate ---
        if self.debounce_seconds > 0.0:
            key = (signal.symbol, signal.direction.value)
            now = time.monotonic()
            last = self._last_emitted.get(key)
            if last is not None and (now - last) < self.debounce_seconds:
                logger.debug(
                    "signal_filter.debounced",
                    symbol=signal.symbol,
                    strategy=signal.strategy_name,
                    direction=signal.direction.value,
                    seconds_since_last=round(now - last, 2),
                    debounce_window=self.debounce_seconds,
                )
                return None
            self._last_emitted[key] = now

        logger.debug(
            "signal_filter.passed",
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            direction=signal.direction.value,
            confidence=signal.confidence,
            strength=signal.strength,
        )
        return signal

    def filter_many(self, signals: list[Signal]) -> list[Signal]:
        """Apply :meth:`filter` to each signal and return those that pass."""
        return [s for s in (self.filter(sig) for sig in signals) if s is not None]

    def reset_debounce(self) -> None:
        """Clear the debounce cache (useful between trading sessions)."""
        self._last_emitted.clear()
