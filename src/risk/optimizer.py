"""Portfolio optimisation using Black-Litterman with ML model views.

Combines market-equilibrium (CAPM) expected returns with subjective views
(e.g. from ML predictions) to produce optimal allocation weights, and
computes the rebalancing trades required to move from current to target
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RebalanceTrade:
    """A single rebalancing instruction."""

    symbol: str
    current_weight: float
    target_weight: float
    delta_weight: float
    direction: str  # "buy" | "sell" | "hold"


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimiser.

    Blends market-implied equilibrium returns with ML-model views and their
    associated confidence levels to produce posterior expected returns and
    optimal weights.

    Parameters
    ----------
    risk_free_rate:
        Annualised risk-free rate used as the CAPM baseline.
    tau:
        Scaling factor on the covariance matrix reflecting uncertainty in the
        equilibrium prior.  Typical values: 0.01 -- 0.10.
    risk_aversion:
        Market risk-aversion coefficient (delta).  Used to reverse-optimise
        equilibrium returns from market-cap weights.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.risk_aversion = risk_aversion

    # ------------------------------------------------------------------
    # Fit / optimise
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
        views: dict[str, float],
        view_confidences: dict[str, float],
        market_cap_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute optimal allocation weights via Black-Litterman.

        Parameters
        ----------
        returns:
            Historical asset returns with columns as asset symbols.
        views:
            Mapping ``{symbol: expected_return}`` representing subjective /
            model-generated views on absolute expected returns.
        view_confidences:
            Mapping ``{symbol: confidence}`` where ``confidence`` is in
            ``(0, 1]``.  Higher values mean the view is more certain.
        market_cap_weights:
            Optional market-capitalisation weights for computing equilibrium
            returns.  If ``None``, equal weights are assumed.

        Returns
        -------
        dict[str, float]
            Optimised allocation weight per asset, summing to 1.0.
        """
        assets = list(returns.columns)
        n = len(assets)

        if n == 0:
            logger.warning("optimizer.no_assets")
            return {}

        # --- Covariance matrix -----------------------------------------
        sigma = returns.cov().values  # (n, n)

        # --- Equilibrium returns (CAPM / reverse optimisation) ----------
        if market_cap_weights is not None:
            w_mkt = np.array([market_cap_weights.get(a, 1.0 / n) for a in assets])
        else:
            w_mkt = np.full(n, 1.0 / n)
        w_mkt = w_mkt / w_mkt.sum()

        pi = self.risk_aversion * sigma @ w_mkt  # equilibrium excess returns

        logger.debug(
            "optimizer.equilibrium_returns",
            assets=assets,
            pi=[round(float(x), 6) for x in pi],
        )

        # --- Construct view matrices -----------------------------------
        view_assets = [a for a in assets if a in views and a in view_confidences]
        k = len(view_assets)

        if k == 0:
            # No usable views — fall back to equilibrium weights
            logger.info("optimizer.no_views_using_equilibrium")
            weights = self._mean_variance_optimise(pi, sigma)
            return dict(zip(assets, weights))

        # P matrix: each view picks one asset (absolute view)
        P = np.zeros((k, n))
        q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, asset in enumerate(view_assets):
            j = assets.index(asset)
            P[i, j] = 1.0
            q[i] = views[asset]
            # Omega diagonal: lower confidence -> higher variance
            conf = max(min(view_confidences[asset], 1.0), 1e-6)
            omega_diag[i] = (1.0 - conf) / conf * self.tau * sigma[j, j]

        omega = np.diag(omega_diag)

        # --- Black-Litterman posterior ---------------------------------
        tau_sigma = self.tau * sigma
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        pt_omega_inv = P.T @ np.linalg.inv(omega)

        # Posterior precision and mean
        posterior_precision = tau_sigma_inv + pt_omega_inv @ P
        posterior_mean = np.linalg.solve(
            posterior_precision,
            tau_sigma_inv @ pi + pt_omega_inv @ q,
        )

        logger.debug(
            "optimizer.posterior_returns",
            assets=assets,
            mu=[round(float(x), 6) for x in posterior_mean],
        )

        # --- Mean-variance optimise on posterior returns ----------------
        posterior_sigma = sigma + tau_sigma  # combined uncertainty
        weights = self._mean_variance_optimise(posterior_mean, posterior_sigma)

        result = dict(zip(assets, [round(float(w), 6) for w in weights]))
        logger.info("optimizer.weights", weights=result)
        return result

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        threshold: float = 0.02,
    ) -> list[RebalanceTrade]:
        """Compute trades needed to move from *current* to *target* weights.

        Only assets whose weight delta exceeds *threshold* are included.

        Parameters
        ----------
        current_weights:
            Current portfolio allocation.
        target_weights:
            Desired portfolio allocation.
        threshold:
            Minimum absolute weight change to trigger a trade.

        Returns
        -------
        list[RebalanceTrade]
            Ordered list of rebalancing instructions.
        """
        all_symbols = sorted(
            set(current_weights.keys()) | set(target_weights.keys())
        )
        trades: list[RebalanceTrade] = []

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            delta = target - current

            if abs(delta) < threshold:
                continue

            direction = "buy" if delta > 0 else "sell"
            trades.append(
                RebalanceTrade(
                    symbol=symbol,
                    current_weight=round(current, 6),
                    target_weight=round(target, 6),
                    delta_weight=round(delta, 6),
                    direction=direction,
                )
            )

        logger.info(
            "optimizer.rebalance",
            trade_count=len(trades),
            symbols=[t.symbol for t in trades],
        )
        return trades

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_variance_optimise(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """Closed-form mean-variance optimal weights with long-only constraint.

        w* = (1 / delta) * Sigma^{-1} * mu, then clipped to [0, inf) and
        normalised to sum to 1.
        """
        try:
            sigma_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("optimizer.singular_covariance_using_pseudoinverse")
            sigma_inv = np.linalg.pinv(cov_matrix)

        raw_weights = (1.0 / self.risk_aversion) * sigma_inv @ expected_returns

        # Long-only: clip negatives and re-normalise
        raw_weights = np.maximum(raw_weights, 0.0)
        total = raw_weights.sum()
        if total <= 0:
            n = len(raw_weights)
            return np.full(n, 1.0 / n)
        return raw_weights / total
