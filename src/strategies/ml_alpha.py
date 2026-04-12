"""ML-driven alpha strategy that converts ensemble predictions into signals.

The :class:`MLAlphaStrategy` runs the stacking ensemble across multiple
forecast horizons and generates :class:`Signal` objects when directional
confidence exceeds a configurable threshold.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.core.types import Signal, SignalDirection
from src.models.ensemble import EnsemblePredictor
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)


class MLAlphaStrategy(BaseStrategyABC):
    """Multi-horizon ML strategy driven by a stacking ensemble.

    Parameters
    ----------
    ensemble:
        Fitted :class:`EnsemblePredictor` instance.
    min_confidence:
        Minimum directional probability (distance from 0.5) required to
        emit a signal.
    horizons:
        Forecast horizons to evaluate.  Shorter horizons receive higher
        weight for entry timing.
    symbols:
        Symbols this strategy covers.  Defaults to ``["SPY"]``.
    """

    # Horizon weights: shorter horizons count more for timing
    _HORIZON_WEIGHTS: dict[int, float] = {
        1: 0.35,
        2: 0.25,
        5: 0.20,
        10: 0.12,
        21: 0.08,
    }

    def __init__(
        self,
        ensemble: EnsemblePredictor,
        min_confidence: float = 0.55,
        horizons: list[int] | None = None,
        symbols: list[str] | None = None,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="ml_alpha", enabled=enabled, weight=weight)
        self.ensemble = ensemble
        self.min_confidence = min_confidence
        self.horizons: list[int] = horizons or [1, 2, 5, 10, 21]
        self.symbols: list[str] = symbols or ["SPY"]

        logger.info(
            "ml_alpha_strategy_init",
            min_confidence=min_confidence,
            horizons=self.horizons,
            symbols=self.symbols,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate trading signals from ensemble predictions.

        Parameters
        ----------
        features:
            Dictionary of feature values. Must include all features required
            by the ensemble.  Values can be scalars or arrays.
        timestamp:
            Current timestamp for the signal.

        Returns
        -------
        list[Signal]
            Zero or more signals, one per symbol where confidence threshold
            is met.
        """
        self.validate_features(features)

        # Build a single-row DataFrame from features dict
        feature_df = self._features_to_dataframe(features)

        try:
            predictions = self.ensemble.predict(feature_df)
            probabilities = self.ensemble.predict_proba(feature_df)
        except Exception:
            logger.error("ml_alpha_prediction_failed", exc_info=True)
            return []

        # Single probability value
        prob = float(probabilities[0]) if len(probabilities) > 0 else 0.5

        signals: list[Signal] = []

        for symbol in self.symbols:
            signal = self._build_signal(predictions, prob, symbol, timestamp)
            if signal is not None:
                signals.append(signal)

        logger.info(
            "ml_alpha_signals_generated",
            n_signals=len(signals),
            probability=round(prob, 4),
            timestamp=str(timestamp),
        )

        return signals

    def get_required_features(self) -> list[str]:
        """Return feature names required by the underlying ensemble."""
        if self.ensemble.feature_names:
            return list(self.ensemble.feature_names)
        # Fallback: aggregate from base models
        all_features: set[str] = set()
        for model in self.ensemble.models:
            if hasattr(model, "feature_names") and model.feature_names:
                all_features.update(model.feature_names)
        return sorted(all_features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        predictions: dict[str, Any],
        probability: float,
        symbol: str,
        timestamp: datetime,
    ) -> Signal | None:
        """Combine multi-horizon predictions into a single signal."""
        # Collect per-horizon signals
        horizon_directions: list[tuple[int, float, float]] = []  # (horizon, direction, strength)

        for h in self.horizons:
            key = f"predictions_h{h}"
            if key not in predictions:
                continue

            pred_return = predictions[key]
            if isinstance(pred_return, np.ndarray):
                pred_return = float(pred_return[0])
            else:
                pred_return = float(pred_return)

            direction = np.sign(pred_return)
            strength = abs(pred_return)
            horizon_directions.append((h, direction, strength))

        if not horizon_directions:
            return None

        # Determine overall direction from probability
        direction = self._determine_direction(probability)
        if direction == SignalDirection.FLAT:
            return None

        # Compute confidence as distance from 0.5, scaled to [0, 1]
        confidence = abs(probability - 0.5) * 2.0
        if confidence < (self.min_confidence - 0.5) * 2.0:
            return None

        # Compute strength as weighted average of predicted returns
        total_weight = 0.0
        weighted_strength = 0.0
        for h, _dir, strength in horizon_directions:
            w = self._HORIZON_WEIGHTS.get(h, 0.05)
            weighted_strength += w * strength
            total_weight += w

        if total_weight > 0:
            weighted_strength /= total_weight

        # Clamp strength to [0, 1]
        signal_strength = float(min(max(weighted_strength * 100, 0.0), 1.0))

        # Build metadata
        metadata: dict[str, Any] = {
            "probability": round(probability, 4),
            "model_weights": self.ensemble.get_model_weights(),
        }
        for h, direction_val, strength in horizon_directions:
            metadata[f"h{h}_direction"] = direction_val
            metadata[f"h{h}_strength"] = round(strength, 6)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=signal_strength,
            confidence=round(confidence, 4),
            strategy_name=self.name,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _determine_direction(self, probability: float) -> SignalDirection:
        """Map probability to direction using thresholds."""
        if probability > self.min_confidence:
            return SignalDirection.LONG
        elif probability < (1.0 - self.min_confidence):
            return SignalDirection.SHORT
        return SignalDirection.FLAT

    @staticmethod
    def _features_to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
        """Convert a feature dict to a single-row DataFrame."""
        row: dict[str, Any] = {}
        for key, val in features.items():
            if isinstance(val, (np.ndarray, pd.Series)):
                if len(val) == 1:
                    row[key] = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val[0])
                else:
                    # Take the last (most recent) value
                    row[key] = float(val.iloc[-1]) if isinstance(val, pd.Series) else float(val[-1])
            elif isinstance(val, (int, float, np.floating, np.integer)):
                row[key] = float(val)
            else:
                row[key] = val

        return pd.DataFrame([row])
