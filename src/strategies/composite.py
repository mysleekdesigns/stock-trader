"""Composite strategy that combines signals from multiple sub-strategies.

:class:`CompositeStrategy` extends :class:`BaseStrategyABC` and orchestrates
a collection of child strategies.  For each symbol it collects signals,
resolves conflicts via confidence-weighted agreement scoring, and emits a
single consensus signal when sufficient agreement is reached.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import structlog

from src.core.types import Signal, SignalDirection
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)

# Default configuration values
_DEFAULT_MIN_AGREEMENT = 0.60


class CompositeStrategy(BaseStrategyABC):
    """Multi-strategy combiner with confidence-weighted conflict resolution.

    Parameters
    ----------
    strategies:
        Child strategies to orchestrate.
    config:
        Configuration dictionary.  Recognised keys:

        ``min_agreement`` (float):
            Minimum fraction of total weight that must agree on a direction
            for a signal to be emitted.  Default ``0.60``.
        ``strategy_weights`` (dict[str, float]):
            Optional per-strategy weight overrides keyed by strategy name.
            If a strategy is listed here its ``weight`` attribute is updated.

    Example ``config/strategies.yaml`` section::

        composite:
          min_agreement: 0.60
          strategy_weights:
            momentum_rsi: 1.5
            ml_alpha: 2.0
    """

    def __init__(
        self,
        strategies: list[BaseStrategyABC],
        config: dict[str, Any] | None = None,
    ) -> None:
        config = config or {}

        super().__init__(
            name=config.get("name", "composite"),
            enabled=config.get("enabled", True),
            weight=config.get("weight", 1.0),
        )

        self._strategies: list[BaseStrategyABC] = list(strategies)
        self._min_agreement: float = float(
            config.get("min_agreement", _DEFAULT_MIN_AGREEMENT)
        )

        # Apply per-strategy weight overrides from config.
        weight_overrides: dict[str, float] = config.get("strategy_weights", {})
        for strat in self._strategies:
            if strat.name in weight_overrides:
                strat.weight = float(weight_overrides[strat.name])

        logger.info(
            "composite_strategy.init",
            strategy_count=len(self._strategies),
            min_agreement=self._min_agreement,
            strategies=[
                {"name": s.name, "weight": s.weight, "enabled": s.enabled}
                for s in self._strategies
            ],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Run all enabled sub-strategies and resolve conflicts.

        Returns a list of consensus :class:`Signal` objects (at most one per
        symbol).  Only signals that meet the ``min_agreement`` threshold are
        emitted.
        """
        # 1. Collect raw signals from every enabled child strategy.
        raw_signals: list[Signal] = []

        for strat in self._strategies:
            if not strat.enabled:
                logger.debug(
                    "composite_strategy.skip_disabled",
                    strategy=strat.name,
                )
                continue

            try:
                strat.validate_features(features)
                signals = strat.generate_signals(features, timestamp)
                raw_signals.extend(signals)
                logger.debug(
                    "composite_strategy.sub_signals",
                    strategy=strat.name,
                    signal_count=len(signals),
                )
            except Exception as exc:
                logger.warning(
                    "composite_strategy.sub_strategy_error",
                    strategy=strat.name,
                    error=str(exc),
                    exc_info=exc,
                )

        if not raw_signals:
            logger.debug("composite_strategy.no_raw_signals")
            return []

        # 2. Group signals by symbol.
        by_symbol: dict[str, list[Signal]] = defaultdict(list)
        for sig in raw_signals:
            by_symbol[sig.symbol].append(sig)

        # 3. Resolve each symbol independently.
        resolved: list[Signal] = []
        for symbol, signals in by_symbol.items():
            result = self._resolve_signals(symbol, signals, timestamp)
            if result is not None:
                resolved.append(result)

        logger.info(
            "composite_strategy.resolved",
            raw_count=len(raw_signals),
            resolved_count=len(resolved),
        )
        return resolved

    def get_required_features(self) -> list[str]:
        """Return the union of features required by all sub-strategies."""
        features: set[str] = set()
        for strat in self._strategies:
            if strat.enabled:
                features.update(strat.get_required_features())
        return sorted(features)

    # ------------------------------------------------------------------
    # Strategy management
    # ------------------------------------------------------------------

    def enable_strategy(self, name: str) -> None:
        """Enable a sub-strategy by name."""
        for strat in self._strategies:
            if strat.name == name:
                strat.enabled = True
                logger.info("composite_strategy.enabled", strategy=name)
                return
        logger.warning("composite_strategy.not_found", strategy=name)

    def disable_strategy(self, name: str) -> None:
        """Disable a sub-strategy by name."""
        for strat in self._strategies:
            if strat.name == name:
                strat.enabled = False
                logger.info("composite_strategy.disabled", strategy=name)
                return
        logger.warning("composite_strategy.not_found", strategy=name)

    @property
    def strategies(self) -> list[BaseStrategyABC]:
        """Return the list of sub-strategies (read-only view)."""
        return list(self._strategies)

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_signals(
        self,
        symbol: str,
        signals: list[Signal],
        timestamp: datetime,
    ) -> Signal | None:
        """Confidence-weighted agreement resolution for a single symbol.

        Steps:
        1. Build a map from strategy-name -> weight (from the child strategy
           objects).
        2. Group signals by direction.
        3. For each direction compute ``agreement = sum(weights) / total_weight``.
        4. If the winning direction meets ``min_agreement``, emit a combined
           signal with weighted-average confidence and strength.
        """
        # Build weight lookup.  If the same strategy emits multiple signals
        # for the same symbol, we take the weight once.
        strat_weight: dict[str, float] = {}
        for strat in self._strategies:
            if strat.enabled:
                strat_weight[strat.name] = strat.weight

        # Group by direction.
        direction_signals: dict[SignalDirection, list[Signal]] = defaultdict(list)
        for sig in signals:
            direction_signals[sig.direction].append(sig)

        # Compute total weight across all signalling strategies.
        signalling_names: set[str] = {sig.strategy_name for sig in signals}
        total_weight = sum(
            strat_weight.get(name, 1.0) for name in signalling_names
        )

        if total_weight == 0.0:
            return None

        # Find the direction with the highest weighted agreement.
        best_direction: SignalDirection | None = None
        best_agreement = 0.0
        best_signals: list[Signal] = []

        for direction, dir_sigs in direction_signals.items():
            dir_names = {s.strategy_name for s in dir_sigs}
            dir_weight = sum(strat_weight.get(n, 1.0) for n in dir_names)
            agreement = dir_weight / total_weight

            if agreement > best_agreement:
                best_agreement = agreement
                best_direction = direction
                best_signals = dir_sigs

        if best_direction is None or best_agreement < self._min_agreement:
            logger.debug(
                "composite_strategy.insufficient_agreement",
                symbol=symbol,
                best_agreement=round(best_agreement, 4),
                min_agreement=self._min_agreement,
            )
            return None

        # Weighted average of confidence and strength.
        weighted_confidence = 0.0
        weighted_strength = 0.0
        weight_sum = 0.0

        for sig in best_signals:
            w = strat_weight.get(sig.strategy_name, 1.0)
            weighted_confidence += sig.confidence * w
            weighted_strength += sig.strength * w
            weight_sum += w

        if weight_sum == 0.0:
            return None

        combined_confidence = min(max(weighted_confidence / weight_sum, 0.0), 1.0)
        combined_strength = min(max(weighted_strength / weight_sum, 0.0), 1.0)

        result = Signal(
            symbol=symbol,
            direction=best_direction,
            strength=combined_strength,
            confidence=combined_confidence,
            strategy_name=self.name,
            timestamp=timestamp,
            metadata={
                "agreement": round(best_agreement, 4),
                "min_agreement": self._min_agreement,
                "contributing_strategies": [
                    {
                        "name": s.strategy_name,
                        "direction": s.direction.value,
                        "confidence": s.confidence,
                        "strength": s.strength,
                        "weight": strat_weight.get(s.strategy_name, 1.0),
                    }
                    for s in best_signals
                ],
                "all_signals": [
                    {
                        "name": s.strategy_name,
                        "direction": s.direction.value,
                        "confidence": s.confidence,
                        "strength": s.strength,
                    }
                    for s in signals
                ],
            },
        )

        logger.info(
            "composite_strategy.signal_resolved",
            symbol=symbol,
            direction=best_direction.value,
            agreement=round(best_agreement, 4),
            confidence=round(combined_confidence, 4),
            strength=round(combined_strength, 4),
            contributing_count=len(best_signals),
        )

        return result
