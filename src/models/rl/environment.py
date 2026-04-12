"""Gymnasium-compatible trading environment for RL agents.

The environment simulates a single-asset portfolio where the agent decides
position sizing on every step.  State captures signal confidence, current
position, unrealized/realized P&L, volatility, and portfolio value.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TradingEnv(gym.Env):
    """Gym-compatible trading environment.

    Parameters
    ----------
    prices:
        1-D array of asset prices for each time step.
    features:
        Optional 2-D array of auxiliary features (rows = steps, cols = features).
        If provided these are appended to the core observation vector.
    initial_capital:
        Starting portfolio value.
    max_steps:
        Maximum number of steps per episode.  Defaults to ``len(prices) - 1``.
    transaction_cost:
        Proportional cost applied on each position change (e.g. 0.001 = 10 bps).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray | None = None,
        initial_capital: float = 100_000.0,
        max_steps: int | None = None,
        transaction_cost: float = 0.001,
    ) -> None:
        super().__init__()

        self.prices = np.asarray(prices, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float64) if features is not None else None
        self.initial_capital = initial_capital
        self.max_steps = max_steps or (len(self.prices) - 1)
        self.transaction_cost = transaction_cost

        if len(self.prices) < 2:
            raise ValueError("prices must have at least 2 elements")

        # Core state dimensions: signal_confidence, position, unrealized_pnl,
        # realized_pnl, volatility, portfolio_value (all normalised)
        self._core_state_dim = 6
        extra_dim = self.features.shape[1] if self.features is not None else 0
        self._state_dim = self._core_state_dim + extra_dim

        # Action: continuous [-1, 1] -> full short to full long
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64,
        )

        # Observation: Box space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float64,
        )

        # Internal tracking — initialised in reset()
        self._step_idx: int = 0
        self._position: float = 0.0  # fraction of capital, [-1, 1]
        self._portfolio_value: float = initial_capital
        self._realized_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._entry_price: float = 0.0
        self._returns: list[float] = []

        logger.info(
            "trading_env_init",
            state_dim=self._state_dim,
            max_steps=self.max_steps,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        self._step_idx = 0
        self._position = 0.0
        self._portfolio_value = self.initial_capital
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._entry_price = self.prices[0]
        self._returns = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Parameters
        ----------
        action:
            Array of shape ``(1,)`` with value in ``[-1, 1]``.

        Returns
        -------
        tuple
            (observation, reward, terminated, truncated, info)
        """
        target_position = float(np.clip(action[0], -1.0, 1.0))

        prev_price = self.prices[self._step_idx]
        self._step_idx += 1
        current_price = self.prices[self._step_idx]

        # Transaction cost for position change
        position_change = abs(target_position - self._position)
        cost = position_change * self._portfolio_value * self.transaction_cost

        # P&L from existing position
        if prev_price > 0:
            price_return = (current_price - prev_price) / prev_price
        else:
            price_return = 0.0

        position_pnl = self._position * self._portfolio_value * price_return

        # Update portfolio
        self._portfolio_value += position_pnl - cost
        self._realized_pnl += -cost  # costs are always realised
        self._unrealized_pnl = target_position * self._portfolio_value * (
            (current_price - self._entry_price) / self._entry_price
            if self._entry_price > 0
            else 0.0
        )

        # Track step return
        step_return = (position_pnl - cost) / max(self._portfolio_value, 1.0)
        self._returns.append(step_return)

        # Update position
        if abs(target_position) > 1e-8 and abs(self._position) < 1e-8:
            # Opening new position
            self._entry_price = current_price
        elif abs(target_position) < 1e-8:
            # Closing position — realise P&L
            self._realized_pnl += self._unrealized_pnl
            self._unrealized_pnl = 0.0
            self._entry_price = current_price

        self._position = target_position

        # Termination conditions
        terminated = self._portfolio_value <= 0
        truncated = self._step_idx >= self.max_steps

        obs = self._get_observation()
        reward = step_return
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        # Compute volatility from recent returns
        if len(self._returns) >= 2:
            volatility = float(np.std(self._returns[-20:]))
        else:
            volatility = 0.0

        # Normalise values
        pv_norm = self._portfolio_value / self.initial_capital

        core = np.array(
            [
                0.0,  # signal confidence placeholder (filled by strategy)
                self._position,
                self._unrealized_pnl / self.initial_capital,
                self._realized_pnl / self.initial_capital,
                volatility,
                pv_norm,
            ],
            dtype=np.float64,
        )

        if self.features is not None:
            idx = min(self._step_idx, len(self.features) - 1)
            extra = self.features[idx]
            return np.concatenate([core, extra])

        return core

    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary information."""
        return {
            "step": self._step_idx,
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl,
        }
