"""Pairs / cointegration trading strategy.

Identifies mean-reverting spreads between two cointegrated securities and
trades the spread when it deviates beyond statistical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.core.exceptions import SignalGenerationError
from src.core.types import Signal, SignalDirection
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PairsTradingConfig:
    """Tunable parameters for :class:`PairsTradingStrategy`."""

    lookback: int = 252
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    max_holding_period: int = 20


class PairState(str, Enum):
    """Current position state of the pair spread."""

    FLAT = "flat"
    LONG_SPREAD = "long_spread"   # long leg_a, short leg_b
    SHORT_SPREAD = "short_spread"  # short leg_a, long leg_b


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class PairsTradingStrategy(BaseStrategyABC):
    """Statistical arbitrage strategy for a cointegrated pair.

    Computes an OLS hedge ratio between two symbols, tracks the z-score of
    the resulting spread, and generates entry/exit signals when the spread
    diverges from or reverts to its mean.

    Parameters
    ----------
    symbol_a:
        First leg of the pair (the one to go long on a long-spread signal).
    symbol_b:
        Second leg of the pair (hedging instrument).
    config:
        Tunable hyper-parameters; uses sensible defaults when omitted.
    enabled:
        If ``False`` the engine should skip this strategy.
    weight:
        Relative importance when signals are aggregated across strategies.
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        config: PairsTradingConfig | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        name = f"pairs_{symbol_a}_{symbol_b}"
        super().__init__(name=name, enabled=enabled, weight=weight)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.config = config or PairsTradingConfig()

        # Internal pair state
        self._state: PairState = PairState.FLAT
        self._bars_in_position: int = 0
        self._hedge_ratio: float = 1.0
        self._spread_mean: float = 0.0
        self._spread_std: float = 1.0

    # ------------------------------------------------------------------
    # BaseStrategyABC interface
    # ------------------------------------------------------------------

    def get_required_features(self) -> list[str]:
        return [
            f"close_{self.symbol_a}",
            f"close_{self.symbol_b}",
        ]

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Evaluate pairs-spread conditions and emit signals for both legs."""
        self.validate_features(features)

        try:
            return self._evaluate(features, timestamp)
        except Exception as exc:
            logger.error(
                "pairs.signal_generation_failed",
                pair=f"{self.symbol_a}/{self.symbol_b}",
                error=str(exc),
                exc_info=True,
            )
            raise SignalGenerationError(
                f"Pairs strategy failed for {self.symbol_a}/{self.symbol_b}: {exc}",
                details={
                    "symbol_a": self.symbol_a,
                    "symbol_b": self.symbol_b,
                },
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

        price_a: float = float(features[f"close_{self.symbol_a}"])
        price_b: float = float(features[f"close_{self.symbol_b}"])

        # Update hedge ratio and spread statistics from price history when
        # history is provided; otherwise use current single-bar estimate.
        history_a = features.get(f"history_{self.symbol_a}")
        history_b = features.get(f"history_{self.symbol_b}")

        if history_a is not None and history_b is not None:
            self._fit_hedge_ratio(
                np.asarray(history_a, dtype=np.float64),
                np.asarray(history_b, dtype=np.float64),
            )

        spread = price_a - self._hedge_ratio * price_b
        zscore = (
            (spread - self._spread_mean) / self._spread_std
            if self._spread_std > 0
            else 0.0
        )

        # --- Track holding duration ------------------------------------
        if self._state is not PairState.FLAT:
            self._bars_in_position += 1

        # --- Determine desired new state --------------------------------
        new_state = self._state

        # Force-exit on max holding period
        if (
            self._state is not PairState.FLAT
            and self._bars_in_position >= cfg.max_holding_period
        ):
            new_state = PairState.FLAT
            logger.info(
                "pairs.max_holding_exit",
                pair=f"{self.symbol_a}/{self.symbol_b}",
                bars=self._bars_in_position,
            )

        # Exit when spread reverts
        elif self._state is PairState.LONG_SPREAD and zscore >= -cfg.exit_zscore:
            new_state = PairState.FLAT
        elif self._state is PairState.SHORT_SPREAD and zscore <= cfg.exit_zscore:
            new_state = PairState.FLAT

        # Entry signals
        elif self._state is PairState.FLAT:
            if zscore < -cfg.entry_zscore:
                new_state = PairState.LONG_SPREAD
            elif zscore > cfg.entry_zscore:
                new_state = PairState.SHORT_SPREAD

        # No state change -> no signal
        if new_state == self._state:
            logger.debug(
                "pairs.no_signal",
                pair=f"{self.symbol_a}/{self.symbol_b}",
                zscore=round(zscore, 4),
                state=self._state.value,
            )
            return []

        # Update state
        prev_state = self._state
        self._state = new_state
        if new_state is PairState.FLAT:
            self._bars_in_position = 0
        else:
            self._bars_in_position = 1

        return self._build_signals(new_state, prev_state, zscore, timestamp)

    # ------------------------------------------------------------------
    # Hedge-ratio estimation
    # ------------------------------------------------------------------

    def _fit_hedge_ratio(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> None:
        """Estimate OLS hedge ratio and spread statistics.

        Falls back to simple correlation-based estimate when ``statsmodels``
        is not available.
        """
        n = min(len(prices_a), len(prices_b), self.config.lookback)
        a = prices_a[-n:]
        b = prices_b[-n:]

        coint_pvalue: float | None = None

        try:
            from statsmodels.tsa.stattools import coint  # type: ignore[import-untyped]

            _, pvalue, _ = coint(a, b)
            coint_pvalue = float(pvalue)
        except ImportError:
            logger.debug("pairs.statsmodels_unavailable_using_correlation")

        # OLS hedge ratio via normal equations: beta = cov(a,b) / var(b)
        cov_ab = np.cov(a, b, ddof=1)
        var_b = cov_ab[1, 1]
        self._hedge_ratio = float(cov_ab[0, 1] / var_b) if var_b > 0 else 1.0

        spread = a - self._hedge_ratio * b
        self._spread_mean = float(np.mean(spread))
        self._spread_std = float(np.std(spread, ddof=1)) or 1.0

        logger.debug(
            "pairs.fit",
            pair=f"{self.symbol_a}/{self.symbol_b}",
            hedge_ratio=round(self._hedge_ratio, 6),
            spread_mean=round(self._spread_mean, 6),
            spread_std=round(self._spread_std, 6),
            coint_pvalue=round(coint_pvalue, 4) if coint_pvalue is not None else None,
            lookback=n,
        )

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        new_state: PairState,
        prev_state: PairState,
        zscore: float,
        timestamp: datetime,
    ) -> list[Signal]:
        """Build a pair of signals (one per leg) for the state transition."""
        confidence = self._compute_confidence(zscore)

        # Strength: how far z-score is beyond the entry threshold
        strength = min(
            abs(zscore) / (self.config.entry_zscore * 2),
            1.0,
        )

        meta = {
            "hedge_ratio": round(self._hedge_ratio, 6),
            "zscore": round(zscore, 4),
            "spread_mean": round(self._spread_mean, 6),
            "spread_std": round(self._spread_std, 6),
            "prev_state": prev_state.value,
            "new_state": new_state.value,
        }

        signals: list[Signal] = []

        if new_state is PairState.FLAT:
            # Exit both legs
            for sym in (self.symbol_a, self.symbol_b):
                signals.append(Signal(
                    symbol=sym,
                    direction=SignalDirection.FLAT,
                    strength=strength,
                    confidence=confidence,
                    strategy_name=self.name,
                    timestamp=timestamp,
                    metadata=meta,
                ))
        elif new_state is PairState.LONG_SPREAD:
            signals.append(Signal(
                symbol=self.symbol_a,
                direction=SignalDirection.LONG,
                strength=strength,
                confidence=confidence,
                strategy_name=self.name,
                timestamp=timestamp,
                metadata=meta,
            ))
            signals.append(Signal(
                symbol=self.symbol_b,
                direction=SignalDirection.SHORT,
                strength=strength,
                confidence=confidence,
                strategy_name=self.name,
                timestamp=timestamp,
                metadata=meta,
            ))
        elif new_state is PairState.SHORT_SPREAD:
            signals.append(Signal(
                symbol=self.symbol_a,
                direction=SignalDirection.SHORT,
                strength=strength,
                confidence=confidence,
                strategy_name=self.name,
                timestamp=timestamp,
                metadata=meta,
            ))
            signals.append(Signal(
                symbol=self.symbol_b,
                direction=SignalDirection.LONG,
                strength=strength,
                confidence=confidence,
                strategy_name=self.name,
                timestamp=timestamp,
                metadata=meta,
            ))

        for sig in signals:
            logger.info(
                "pairs.signal",
                symbol=sig.symbol,
                direction=sig.direction.value,
                confidence=round(sig.confidence, 4),
                strength=round(sig.strength, 4),
                pair=f"{self.symbol_a}/{self.symbol_b}",
            )

        return signals

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_confidence(self, zscore: float) -> float:
        """Confidence based on how far z-score exceeds the entry threshold.

        For exit signals a moderate confidence of 0.6 is returned.
        """
        if self._state is PairState.FLAT:
            return 0.6

        excess = abs(zscore) - self.config.entry_zscore
        if excess <= 0:
            return 0.3
        # Saturate at ~3x the entry threshold
        return min(0.3 + 0.7 * (excess / self.config.entry_zscore), 1.0)
