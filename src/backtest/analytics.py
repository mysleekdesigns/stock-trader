"""Performance analytics for backtesting results.

All computations use numpy for numerical stability and performance.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Trading days per year for annualization.
_TRADING_DAYS = 252


class BacktestAnalytics:
    """Compute standard performance metrics from an equity curve and trade log.

    Parameters
    ----------
    equity_curve:
        Time-series of portfolio equity values (one per bar / day).
    timestamps:
        Corresponding datetime for each equity value.
    trades:
        List of trade dicts.  Each must contain at least ``pnl`` (float)
        and optionally ``entry_time``, ``exit_time`` keys.
    risk_free_rate:
        Annualized risk-free rate for Sharpe / Sortino calculations.
    """

    def __init__(
        self,
        equity_curve: list[float],
        timestamps: list[datetime],
        trades: list[dict],
        risk_free_rate: float = 0.05,
    ) -> None:
        if len(equity_curve) < 2:
            raise ValueError("Equity curve must contain at least 2 data points.")
        if len(equity_curve) != len(timestamps):
            raise ValueError(
                f"equity_curve length ({len(equity_curve)}) != "
                f"timestamps length ({len(timestamps)})"
            )

        self._equity = np.asarray(equity_curve, dtype=np.float64)
        self._timestamps = timestamps
        self._trades = trades
        self._rf = risk_free_rate

        # Daily returns (simple).
        self._returns = np.diff(self._equity) / self._equity[:-1]
        # Daily risk-free rate.
        self._rf_daily = (1.0 + self._rf) ** (1.0 / _TRADING_DAYS) - 1.0
        self._excess_returns = self._returns - self._rf_daily

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Total cumulative return as a fraction (e.g. 0.25 = +25%)."""
        return float((self._equity[-1] / self._equity[0]) - 1.0)

    def annualized_return(self) -> float:
        """Annualized (CAGR) return."""
        n_days = len(self._returns)
        if n_days == 0:
            return 0.0
        total = self._equity[-1] / self._equity[0]
        if total <= 0:
            return -1.0
        return float(total ** (_TRADING_DAYS / n_days) - 1.0)

    def volatility(self) -> float:
        """Annualized volatility of daily returns."""
        if len(self._returns) < 2:
            return 0.0
        return float(np.std(self._returns, ddof=1) * np.sqrt(_TRADING_DAYS))

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (252 trading days)."""
        if len(self._excess_returns) < 2:
            return 0.0
        std = np.std(self._excess_returns, ddof=1)
        if std == 0.0 or np.isnan(std):
            return 0.0
        return float(
            np.mean(self._excess_returns) / std * np.sqrt(_TRADING_DAYS)
        )

    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(self._excess_returns) < 2:
            return 0.0
        downside = self._excess_returns[self._excess_returns < 0.0]
        if len(downside) == 0:
            # No negative returns — infinite Sortino; cap for practicality.
            return float("inf") if np.mean(self._excess_returns) > 0 else 0.0
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std == 0.0:
            return 0.0
        return float(
            np.mean(self._excess_returns) / downside_std * np.sqrt(_TRADING_DAYS)
        )

    def calmar_ratio(self) -> float:
        """Annualized return / max drawdown depth."""
        dd_depth, _ = self.max_drawdown()
        if dd_depth == 0.0:
            return float("inf") if self.annualized_return() > 0 else 0.0
        return self.annualized_return() / dd_depth

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def max_drawdown(self) -> tuple[float, int]:
        """Return (max_drawdown_depth_pct, max_drawdown_duration_days).

        Depth is expressed as a positive fraction (e.g. 0.15 = 15% drawdown).
        Duration is the number of bars from peak to recovery (or end of data).
        """
        peak = self._equity[0]
        max_dd = 0.0
        max_duration = 0
        current_duration = 0

        for val in self._equity[1:]:
            if val >= peak:
                peak = val
                current_duration = 0
            else:
                dd = (peak - val) / peak
                current_duration += 1
                if dd > max_dd:
                    max_dd = dd
                if current_duration > max_duration:
                    max_duration = current_duration

        return float(max_dd), max_duration

    # ------------------------------------------------------------------
    # Trade metrics
    # ------------------------------------------------------------------

    def win_rate(self) -> float:
        """Fraction of trades with positive PnL."""
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.get("pnl", 0) > 0)
        return wins / len(self._trades)

    def profit_factor(self) -> float:
        """Gross profits / gross losses."""
        gross_profit = sum(
            t["pnl"] for t in self._trades if t.get("pnl", 0) > 0
        )
        gross_loss = abs(
            sum(t["pnl"] for t in self._trades if t.get("pnl", 0) < 0)
        )
        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def avg_trade_duration(self) -> float:
        """Average trade duration in days.

        Requires each trade dict to have ``entry_time`` and ``exit_time``
        keys (datetime objects).  Trades missing these keys are skipped.
        """
        durations: list[float] = []
        for t in self._trades:
            entry = t.get("entry_time")
            exit_ = t.get("exit_time")
            if isinstance(entry, datetime) and isinstance(exit_, datetime):
                durations.append((exit_ - entry).total_seconds() / 86400.0)
        if not durations:
            return 0.0
        return float(np.mean(durations))

    # ------------------------------------------------------------------
    # Benchmark-relative metrics
    # ------------------------------------------------------------------

    def alpha(self, benchmark_returns: np.ndarray | list[float]) -> float:
        """Jensen's alpha relative to *benchmark_returns* (daily)."""
        bm = np.asarray(benchmark_returns, dtype=np.float64)
        n = min(len(self._returns), len(bm))
        if n < 2:
            return 0.0
        port = self._returns[:n]
        bm = bm[:n]
        beta_val = self._compute_beta(port, bm)
        ann_port = float(np.mean(port)) * _TRADING_DAYS
        ann_bm = float(np.mean(bm)) * _TRADING_DAYS
        return ann_port - (self._rf + beta_val * (ann_bm - self._rf))

    def beta(self, benchmark_returns: np.ndarray | list[float]) -> float:
        """Portfolio beta relative to *benchmark_returns* (daily)."""
        bm = np.asarray(benchmark_returns, dtype=np.float64)
        n = min(len(self._returns), len(bm))
        if n < 2:
            return 0.0
        return self._compute_beta(self._returns[:n], bm[:n])

    def information_ratio(
        self, benchmark_returns: np.ndarray | list[float],
    ) -> float:
        """Annualized information ratio relative to *benchmark_returns*."""
        bm = np.asarray(benchmark_returns, dtype=np.float64)
        n = min(len(self._returns), len(bm))
        if n < 2:
            return 0.0
        active = self._returns[:n] - bm[:n]
        te = np.std(active, ddof=1)
        if te == 0.0 or np.isnan(te):
            return 0.0
        return float(np.mean(active) / te * np.sqrt(_TRADING_DAYS))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """Return a dict with all non-benchmark metrics."""
        dd_depth, dd_duration = self.max_drawdown()
        return {
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "volatility": self.volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": dd_depth,
            "max_drawdown_duration": dd_duration,
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "avg_trade_duration_days": self.avg_trade_duration(),
            "total_trades": len(self._trades),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_beta(port_returns: np.ndarray, bm_returns: np.ndarray) -> float:
        var = np.var(bm_returns, ddof=1)
        if var == 0.0 or np.isnan(var):
            return 0.0
        cov = np.cov(port_returns, bm_returns, ddof=1)[0, 1]
        return float(cov / var)
