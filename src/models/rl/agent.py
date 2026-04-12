"""SAC-based reinforcement learning agent for trading.

Wraps Stable-Baselines3's :class:`SAC` implementation with trading-specific
logging, reward function injection, and model persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src.models.rl.environment import TradingEnv
from src.models.rl.reward import BaseReward

logger = structlog.get_logger(__name__)


class RLAgent:
    """SAC-based trading agent.

    Parameters
    ----------
    env:
        A :class:`TradingEnv` (or any Gymnasium environment).
    reward_fn:
        Reward function used for custom reward shaping during training.
    params:
        Extra keyword arguments forwarded to the SAC constructor
        (e.g. ``learning_rate``, ``batch_size``, ``buffer_size``).
    """

    def __init__(
        self,
        env: TradingEnv,
        reward_fn: BaseReward,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.reward_fn = reward_fn
        self.params = params or {}
        self._model: Any = None  # lazily created SAC instance

        logger.info(
            "rl_agent_init",
            reward_fn=type(reward_fn).__name__,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Lazy model construction
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Create the SAC model if it has not been created yet."""
        if self._model is not None:
            return

        try:
            from stable_baselines3 import SAC  # type: ignore[import-untyped]
        except ImportError as exc:
            logger.error("stable_baselines3_not_installed")
            raise ImportError(
                "stable_baselines3 is required for RLAgent. "
                "Install with: uv add stable-baselines3"
            ) from exc

        default_params: dict[str, Any] = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 100_000,
            "tau": 0.005,
            "gamma": 0.99,
            "verbose": 0,
        }
        default_params.update(self.params)
        policy = default_params.pop("policy")

        self._model = SAC(policy, self.env, **default_params)
        logger.info("sac_model_created", policy=policy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 10_000) -> dict[str, Any]:
        """Train the SAC agent.

        Parameters
        ----------
        total_timesteps:
            Number of environment interactions.

        Returns
        -------
        dict
            Training metrics including total timesteps and final portfolio
            value from the environment info.
        """
        self._ensure_model()

        logger.info("rl_training_start", total_timesteps=total_timesteps)
        self._model.learn(total_timesteps=total_timesteps)

        # Collect metrics from a single evaluation episode
        obs, info = self.env.reset()
        done = False
        episode_returns: list[float] = []
        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_returns.append(float(reward))
            done = terminated or truncated

        metrics: dict[str, Any] = {
            "total_timesteps": total_timesteps,
            "final_portfolio_value": info.get("portfolio_value", 0.0),
            "episode_reward": float(np.sum(episode_returns)),
            "reward_fn_score": self.reward_fn.calculate(np.array(episode_returns)),
        }

        logger.info("rl_training_complete", **metrics)
        return metrics

    def predict(self, observation: np.ndarray) -> float:
        """Return a position-sizing action in ``[-1, 1]``.

        Parameters
        ----------
        observation:
            Environment observation vector.

        Returns
        -------
        float
            Target position size.
        """
        self._ensure_model()

        action, _ = self._model.predict(observation, deterministic=True)
        position = float(np.clip(action[0], -1.0, 1.0))

        logger.debug("rl_predict", position=round(position, 4))
        return position

    def save(self, path: str | Path) -> None:
        """Persist the trained model to *path*."""
        self._ensure_model()

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(save_path))
        logger.info("rl_model_saved", path=str(save_path))

    def load(self, path: str | Path) -> None:
        """Load a previously saved model from *path*."""
        try:
            from stable_baselines3 import SAC  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is required for RLAgent. "
                "Install with: uv add stable-baselines3"
            ) from exc

        load_path = Path(path)
        self._model = SAC.load(str(load_path), env=self.env)
        logger.info("rl_model_loaded", path=str(load_path))
