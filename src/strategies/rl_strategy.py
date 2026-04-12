"""RL-based trading strategy.

Wraps an :class:`RLAgent` behind the standard :class:`BaseStrategyABC`
interface, translating raw features into environment observations and
RL actions into :class:`Signal` objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import structlog

from src.core.types import Signal, SignalDirection
from src.models.rl.agent import RLAgent
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)

# Feature keys consumed by the RL observation vector
_REQUIRED_FEATURES = [
    "signal_confidence",
    "position",
    "unrealized_pnl",
    "realized_pnl",
    "volatility",
    "portfolio_value",
]


class RLStrategy(BaseStrategyABC):
    """Strategy that delegates position-sizing decisions to an RL agent.

    Parameters
    ----------
    agent:
        Trained (or in-training) :class:`RLAgent`.
    symbols:
        Symbols this strategy covers.
    min_action_threshold:
        Minimum absolute action magnitude to emit a signal.  Values below
        this threshold are treated as *flat*.
    """

    def __init__(
        self,
        agent: RLAgent,
        symbols: list[str] | None = None,
        min_action_threshold: float = 0.05,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="rl_strategy", enabled=enabled, weight=weight)
        self.agent = agent
        self.symbols: list[str] = symbols or ["SPY"]
        self.min_action_threshold = min_action_threshold

        logger.info(
            "rl_strategy_init",
            symbols=self.symbols,
            min_action_threshold=min_action_threshold,
        )

    # ------------------------------------------------------------------
    # BaseStrategyABC interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Translate features into RL observations, run the agent, and
        convert the resulting action into trading signals.
        """
        self.validate_features(features)

        observation = self._build_observation(features)

        try:
            action = self.agent.predict(observation)
        except Exception:
            logger.error("rl_strategy_prediction_failed", exc_info=True)
            return []

        signals: list[Signal] = []
        for symbol in self.symbols:
            signal = self._action_to_signal(action, symbol, timestamp)
            if signal is not None:
                signals.append(signal)

        logger.info(
            "rl_signals_generated",
            n_signals=len(signals),
            action=round(action, 4),
            timestamp=str(timestamp),
        )
        return signals

    def get_required_features(self) -> list[str]:
        return list(_REQUIRED_FEATURES)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, features: dict[str, Any]) -> np.ndarray:
        """Convert the features dict to a numpy observation vector."""
        values: list[float] = []
        for key in _REQUIRED_FEATURES:
            val = features.get(key, 0.0)
            if isinstance(val, (np.ndarray,)):
                val = float(val[-1]) if len(val) > 0 else 0.0
            values.append(float(val))
        return np.array(values, dtype=np.float64)

    def _action_to_signal(
        self,
        action: float,
        symbol: str,
        timestamp: datetime,
    ) -> Signal | None:
        """Map a continuous action to a discrete Signal."""
        if abs(action) < self.min_action_threshold:
            return None

        if action > 0:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        strength = min(abs(action), 1.0)
        confidence = strength  # use magnitude as confidence proxy

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=round(confidence, 4),
            strategy_name=self.name,
            timestamp=timestamp,
            metadata={
                "rl_action": round(action, 6),
                "position_size": round(action, 6),
            },
        )
