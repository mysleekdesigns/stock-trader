"""Reward functions for RL trading agents.

Each reward class implements a ``calculate`` method that takes an array of
portfolio returns and produces a scalar reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class BaseReward(ABC):
    """Abstract reward function."""

    @abstractmethod
    def calculate(self, portfolio_returns: np.ndarray) -> float:
        """Compute a scalar reward from a sequence of portfolio returns."""
        ...


class SharpeReward(BaseReward):
    """Rolling Sharpe ratio of returns.

    Parameters
    ----------
    window:
        Number of recent returns to include.  Defaults to 20.
    annualization_factor:
        Scaling factor (e.g. sqrt(252) for daily data).  Defaults to 1.0.
    """

    def __init__(
        self,
        window: int = 20,
        annualization_factor: float = 1.0,
    ) -> None:
        self.window = window
        self.annualization_factor = annualization_factor

    def calculate(self, portfolio_returns: np.ndarray) -> float:
        """Return the Sharpe ratio over the most recent *window* returns."""
        returns = np.asarray(portfolio_returns, dtype=np.float64)
        if len(returns) < 2:
            return 0.0

        recent = returns[-self.window:]
        std = float(np.std(recent, ddof=1))
        if std < 1e-10:
            return 0.0

        mean = float(np.mean(recent))
        sharpe = (mean / std) * self.annualization_factor
        return sharpe


class SortinoReward(BaseReward):
    """Sortino ratio — Sharpe variant using downside deviation only.

    Parameters
    ----------
    window:
        Number of recent returns to include.
    annualization_factor:
        Scaling factor.
    target_return:
        Minimum acceptable return; values below this count as downside.
    """

    def __init__(
        self,
        window: int = 20,
        annualization_factor: float = 1.0,
        target_return: float = 0.0,
    ) -> None:
        self.window = window
        self.annualization_factor = annualization_factor
        self.target_return = target_return

    def calculate(self, portfolio_returns: np.ndarray) -> float:
        """Return the Sortino ratio over the most recent *window* returns."""
        returns = np.asarray(portfolio_returns, dtype=np.float64)
        if len(returns) < 2:
            return 0.0

        recent = returns[-self.window:]
        excess = recent - self.target_return
        downside = excess[excess < 0]

        if len(downside) < 1:
            # No downside returns — perfect score capped to avoid inf
            mean = float(np.mean(recent))
            return mean * self.annualization_factor * 100.0 if mean > 0 else 0.0

        downside_std = float(np.std(downside, ddof=1))
        if downside_std < 1e-10:
            return 0.0

        mean = float(np.mean(recent))
        sortino = (mean / downside_std) * self.annualization_factor
        return sortino


class AsymmetricReward(BaseReward):
    """Asymmetric reward that penalises losses more heavily than it rewards gains.

    Parameters
    ----------
    loss_multiplier:
        Factor by which negative returns are amplified.
    """

    def __init__(self, loss_multiplier: float = 2.0) -> None:
        self.loss_multiplier = loss_multiplier

    def calculate(self, portfolio_returns: np.ndarray) -> float:
        """Compute asymmetric reward from the most recent return.

        Positive returns are passed through; negative returns are scaled by
        ``loss_multiplier``.
        """
        returns = np.asarray(portfolio_returns, dtype=np.float64)
        if len(returns) == 0:
            return 0.0

        last_return = float(returns[-1])
        if last_return < 0:
            return last_return * self.loss_multiplier
        return last_return
