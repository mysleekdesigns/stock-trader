"""Position sizing strategies.

Provides Kelly criterion, volatility-adjusted, and fixed-fractional sizing,
plus a dispatcher that selects the appropriate method at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from src.core.types import Signal, SignalDirection

logger = structlog.get_logger(__name__)


@dataclass
class KellyCriterion:
    """Kelly criterion position sizer.

    Returns the optimal fraction of the portfolio to risk on a single trade,
    clamped between ``fraction_min`` and ``fraction_max``.
    """

    fraction_min: float = 0.25
    fraction_max: float = 0.50

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,
    ) -> float:
        """Compute the Kelly fraction.

        Parameters
        ----------
        win_rate:
            Historical probability of a winning trade (0-1).
        avg_win:
            Average gain on winning trades (as a positive multiple, e.g. 1.5).
        avg_loss:
            Average loss on losing trades (as a positive multiple, e.g. 1.0).
        fraction:
            Fractional Kelly multiplier applied after computation (e.g. 0.25
            for quarter-Kelly).

        Returns
        -------
        float
            Fraction of portfolio to allocate, clamped to configured bounds.
        """
        if avg_loss <= 0 or avg_win <= 0 or not (0 < win_rate < 1):
            logger.warning(
                "position_sizer.kelly_invalid_inputs",
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        # where p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b

        if kelly <= 0:
            return 0.0

        # Apply fractional Kelly
        adjusted = kelly * fraction

        # Clamp to configured bounds
        clamped = max(self.fraction_min, min(self.fraction_max, adjusted))

        logger.debug(
            "position_sizer.kelly_result",
            raw_kelly=round(kelly, 4),
            fraction=fraction,
            adjusted=round(adjusted, 4),
            clamped=round(clamped, 4),
        )
        return clamped


@dataclass
class VolatilityAdjustedSizer:
    """Sizes positions inversely proportional to asset volatility.

    The idea is to allocate a fixed risk budget (``target_risk`` as a fraction
    of portfolio value) and derive the number of shares from the asset's
    realised volatility.
    """

    def calculate(
        self,
        target_risk: float,
        volatility: float,
        portfolio_value: float,
    ) -> float:
        """Return the dollar notional to allocate.

        Parameters
        ----------
        target_risk:
            Fraction of portfolio value to risk (e.g. 0.02 for 2%).
        volatility:
            Annualised or per-bar volatility of the asset price.
        portfolio_value:
            Current portfolio value in dollars.

        Returns
        -------
        float
            Dollar amount to allocate.
        """
        if volatility <= 0 or portfolio_value <= 0:
            logger.warning(
                "position_sizer.vol_invalid_inputs",
                volatility=volatility,
                portfolio_value=portfolio_value,
            )
            return 0.0

        dollar_risk = target_risk * portfolio_value
        notional = dollar_risk / volatility

        logger.debug(
            "position_sizer.vol_adjusted_result",
            target_risk=target_risk,
            volatility=round(volatility, 6),
            notional=round(notional, 2),
        )
        return notional


@dataclass
class FixedFractionalSizer:
    """Allocates a fixed fraction of the portfolio to each trade."""

    def calculate(
        self,
        fraction: float,
        portfolio_value: float,
        price: float,
    ) -> int:
        """Return the number of whole shares to buy.

        Parameters
        ----------
        fraction:
            Fraction of portfolio to allocate (e.g. 0.02 for 2%).
        portfolio_value:
            Current portfolio value in dollars.
        price:
            Current share price.

        Returns
        -------
        int
            Number of whole shares (floored).
        """
        if price <= 0 or portfolio_value <= 0 or fraction <= 0:
            return 0

        dollar_amount = fraction * portfolio_value
        shares = int(math.floor(dollar_amount / price))

        logger.debug(
            "position_sizer.fixed_fractional_result",
            fraction=fraction,
            dollar_amount=round(dollar_amount, 2),
            price=price,
            shares=shares,
        )
        return shares


class PositionSizer:
    """Dispatcher that routes to the appropriate sizer based on configuration.

    Parameters
    ----------
    method:
        One of ``"kelly"``, ``"volatility"``, ``"fixed_fractional"``.
    params:
        Method-specific parameters forwarded to the underlying sizer.
    """

    def __init__(self, method: str = "kelly", params: dict | None = None) -> None:
        self._method = method
        self._params = params or {}

        self._kelly = KellyCriterion(
            fraction_min=self._params.get("kelly_fraction_min", 0.25),
            fraction_max=self._params.get("kelly_fraction_max", 0.50),
        )
        self._vol_sizer = VolatilityAdjustedSizer()
        self._fixed_sizer = FixedFractionalSizer()

    def size(
        self,
        signal: Signal,
        portfolio_value: float,
        price: float,
        volatility: float,
        win_rate: float = 0.55,
    ) -> int:
        """Compute the number of shares to trade for a given signal.

        Parameters
        ----------
        signal:
            The trading signal to size.
        portfolio_value:
            Current portfolio value in dollars.
        price:
            Current share price.
        volatility:
            Annualised or per-bar volatility of the asset.
        win_rate:
            Historical win rate for Kelly calculation.

        Returns
        -------
        int
            Number of whole shares (always >= 0).  Returns 0 for FLAT signals.
        """
        if signal.direction == SignalDirection.FLAT:
            return 0

        if price <= 0 or portfolio_value <= 0:
            return 0

        if self._method == "kelly":
            return self._size_kelly(
                signal=signal,
                portfolio_value=portfolio_value,
                price=price,
                volatility=volatility,
                win_rate=win_rate,
            )
        elif self._method == "volatility":
            return self._size_volatility(
                portfolio_value=portfolio_value,
                price=price,
                volatility=volatility,
            )
        elif self._method == "fixed_fractional":
            return self._size_fixed(
                portfolio_value=portfolio_value,
                price=price,
            )
        else:
            logger.error("position_sizer.unknown_method", method=self._method)
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _size_kelly(
        self,
        signal: Signal,
        portfolio_value: float,
        price: float,
        volatility: float,
        win_rate: float,
    ) -> int:
        # Use volatility as a proxy for avg_loss and signal strength for avg_win
        avg_loss = max(volatility, 0.01)
        avg_win = avg_loss * (1.0 + signal.strength)  # stronger signal => bigger edge

        fraction = self._kelly.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            fraction=self._params.get("kelly_fraction", 0.25),
        )
        if fraction <= 0:
            return 0

        dollar_amount = fraction * portfolio_value
        shares = int(math.floor(dollar_amount / price))
        return max(shares, 0)

    def _size_volatility(
        self,
        portfolio_value: float,
        price: float,
        volatility: float,
    ) -> int:
        target_risk = self._params.get("target_risk", 0.02)
        notional = self._vol_sizer.calculate(
            target_risk=target_risk,
            volatility=volatility,
            portfolio_value=portfolio_value,
        )
        if notional <= 0:
            return 0
        shares = int(math.floor(notional / price))
        return max(shares, 0)

    def _size_fixed(
        self,
        portfolio_value: float,
        price: float,
    ) -> int:
        fraction = self._params.get("fixed_fraction", 0.02)
        shares = self._fixed_sizer.calculate(
            fraction=fraction,
            portfolio_value=portfolio_value,
            price=price,
        )
        return max(shares, 0)
