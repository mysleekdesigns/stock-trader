"""Glue between the API and the real :class:`BacktestEngine`.

Loads historical OHLCV data (yfinance), builds the feature pipeline and a
per-symbol strategy dispatcher, runs the event-driven engine, trims the result
to the requested window, and shapes the metrics into the keys the dashboard
expects.  This replaces the simulated random-walk fallback that previously
lived in the API router.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

# Importing these two modules populates the global feature registry as a
# side-effect (technical creates it, price registers onto it).
import src.features.price  # noqa: F401  — registers price/return/zscore features
from src.backtest.analytics import BacktestAnalytics
from src.backtest.data_handler import DataHandler
from src.backtest.engine import BacktestEngine
from src.core.types import TimeFrame
from src.data.providers.yfinance_provider import YFinanceProvider
from src.features.pipeline import FeaturePipeline
from src.features.technical import registry
from src.strategies.base import BaseStrategyABC
from src.strategies.composite import CompositeStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy

if TYPE_CHECKING:
    from src.api.schemas.backtest import BacktestRequest

logger = structlog.get_logger(__name__)


class UnsupportedStrategyError(ValueError):
    """Raised when a requested strategy cannot be run by the real engine."""


class NoDataError(RuntimeError):
    """Raised when no historical data could be loaded for the request."""


# Strategies the real engine can run today.  ``lstm`` / other ML strategies
# need a trained model artifact and are intentionally excluded.
SUPPORTED_STRATEGIES = {"momentum", "mean_reversion", "ensemble"}

# Calendar-day warmup buffer fetched *before* the requested start so that
# look-back indicators (EMA-50, ADX-14, SMA-200, …) are warm by the window start.
_WARMUP_DAYS: dict[TimeFrame, int] = {
    TimeFrame.DAILY: 330,
    TimeFrame.WEEKLY: 1500,
    TimeFrame.HOUR_1: 30,
    TimeFrame.MINUTE_15: 10,
    TimeFrame.MINUTE_5: 7,
    TimeFrame.MINUTE_1: 5,
}

# Intraday timeframes have limited historical depth on yfinance.
_INTRADAY_TIMEFRAMES = frozenset(
    {TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.HOUR_1}
)

# Bars per year per timeframe, used to annualize Sharpe/volatility/return.
# Sub-daily values assume a 24/7 market (correct for crypto such as BTC-USD);
# daily/weekly use equity-market trading-period counts.
_PERIODS_PER_YEAR: dict[TimeFrame, float] = {
    TimeFrame.DAILY: 252.0,
    TimeFrame.WEEKLY: 52.0,
    TimeFrame.HOUR_1: 24 * 365,
    TimeFrame.MINUTE_15: 4 * 24 * 365,
    TimeFrame.MINUTE_5: 12 * 24 * 365,
    TimeFrame.MINUTE_1: 60 * 24 * 365,
}

# Map a strategy's required *feature key* to the registry feature that produces
# it.  Keys not listed here are assumed to match a registry feature 1:1.
#   - ``adx``           is emitted as a column by the ``adx_14`` feature.
#   - ``bollinger_%b``  is the strategy's key; the registry's ``bollinger``
#                       feature emits ``bb_pct_b``, so we register an alias.
_FEATURE_KEY_TO_REGISTRY: dict[str, str] = {
    "adx": "adx_14",
    "bollinger_%b": "bollinger_%b",
}

_FEATURE_LOOKBACK = 250


# ---------------------------------------------------------------------------
# Registry alias: bollinger_%b (the mean-reversion strategy's required key)
# ---------------------------------------------------------------------------

def _register_bollinger_pct_b_alias() -> None:
    if "bollinger_%b" in registry:
        return

    from ta.volatility import BollingerBands

    def _bollinger_pct_b(df: pd.DataFrame) -> pd.Series:
        ind = BollingerBands(close=df["close"], window=20, window_dev=2, fillna=False)
        upper = ind.bollinger_hband()
        lower = ind.bollinger_lband()
        return (df["close"] - lower) / (upper - lower)

    registry.register(
        name="bollinger_%b",
        compute_fn=_bollinger_pct_b,
        group="volatility",
        description="Bollinger %B (mean-reversion strategy alias)",
    )


_register_bollinger_pct_b_alias()


# ---------------------------------------------------------------------------
# Per-symbol strategy dispatcher
# ---------------------------------------------------------------------------

class _PerSymbolStrategy(BaseStrategyABC):
    """Route each symbol's features to a dedicated per-symbol strategy instance.

    The built-in strategies (Momentum, MeanReversion, …) are each bound to a
    single symbol and emit ``Signal(symbol=self.symbol)``.  The engine, however,
    holds one strategy and feeds it every symbol's features.  This wrapper keeps
    one instance per symbol (preserving per-symbol state such as the momentum
    crossover memory) and dispatches by ``features["symbol"]``.
    """

    def __init__(
        self,
        name: str,
        instances: dict[str, BaseStrategyABC],
        trade_start: date | None = None,
    ) -> None:
        super().__init__(name=name)
        self._instances = instances
        self._trade_start = trade_start
        required: set[str] = set()
        for inst in instances.values():
            required.update(inst.get_required_features())
        self._required = sorted(required)

    def get_required_features(self) -> list[str]:
        return list(self._required)

    def generate_signals(self, features: dict[str, Any], timestamp: datetime):
        inst = self._instances.get(features.get("symbol"))
        if inst is None:
            return []
        try:
            signals = inst.generate_signals(features, timestamp)
        except Exception as exc:  # one bad symbol must not abort the whole bar
            logger.debug(
                "backtest_runner.symbol_signal_error",
                symbol=features.get("symbol"),
                error=str(exc),
            )
            return []
        # During the look-back warmup we still run the sub-strategies (so their
        # indicators and crossover state stay continuous) but emit no signals,
        # leaving capital untouched until the requested window begins.
        if self._trade_start is not None and timestamp.date() < self._trade_start:
            return []
        return signals


def _build_strategy(
    name: str,
    symbols: list[str],
    trade_start: date | None = None,
) -> BaseStrategyABC:
    if name == "momentum":
        instances = {s: MomentumStrategy(s) for s in symbols}
    elif name == "mean_reversion":
        instances = {s: MeanReversionStrategy(s) for s in symbols}
    elif name == "ensemble":
        # Confidence-weighted combination of momentum + mean reversion per symbol.
        instances = {
            s: CompositeStrategy([MomentumStrategy(s), MeanReversionStrategy(s)])
            for s in symbols
        }
    else:
        raise UnsupportedStrategyError(
            f"Strategy '{name}' is not runnable by the backtest engine. "
            f"Supported: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
        )
    return _PerSymbolStrategy(name, instances, trade_start=trade_start)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def _load_bars(
    symbols: list[str],
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Fetch bars per symbol and return {symbol: OHLCV DataFrame}."""
    provider = YFinanceProvider()
    bars: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            bar_list = await provider.get_bars(symbol, timeframe, start, end)
        except Exception as exc:
            logger.warning("backtest_runner.fetch_failed", symbol=symbol, error=str(exc))
            continue
        if not bar_list:
            continue

        # Intraday yfinance bars carry a tz-aware (UTC) index; the rest of the
        # engine uses tz-naive datetimes (e.g. Order.created_at via utcnow), so
        # normalize to naive to avoid aware/naive subtraction errors.
        index = pd.DatetimeIndex([b.timestamp for b in bar_list])
        if index.tz is not None:
            index = index.tz_localize(None)

        df = pd.DataFrame(
            {
                "open": [b.open for b in bar_list],
                "high": [b.high for b in bar_list],
                "low": [b.low for b in bar_list],
                "close": [b.close for b in bar_list],
                "volume": [b.volume for b in bar_list],
            },
            index=index,
        )
        df = df[~df.index.duplicated(keep="last")].sort_index()
        bars[symbol] = df
        logger.info("backtest_runner.loaded", symbol=symbol, rows=len(df))

    return bars


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _finite(value: Any, cap: float = 99.99) -> float:
    """Coerce to a JSON-safe finite float (orjson rejects inf/nan)."""
    if value is None:
        return 0.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    if math.isinf(x):
        return cap if x > 0 else -cap
    return x


def _trade_in_window(trade: dict, start: date) -> bool:
    exit_time = trade.get("exit_time")
    if isinstance(exit_time, datetime):
        return exit_time.date() >= start
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_backtest(
    request: BacktestRequest,
) -> tuple[dict[str, Any], list[float], list[dict[str, Any]], list[str]]:
    """Run a real backtest and return ``(metrics, equity_curve, trades, timestamps)``.

    Raises
    ------
    UnsupportedStrategyError
        If the strategy cannot be executed by the engine.
    NoDataError
        If no historical data could be loaded for any requested symbol.
    """
    name = request.strategy.lower().strip()
    if name not in SUPPORTED_STRATEGIES:
        raise UnsupportedStrategyError(
            f"Strategy '{request.strategy}' is not runnable by the backtest "
            f"engine. Supported: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
        )

    symbols = [s.upper() for s in request.symbols]
    timeframe = request.timeframe
    start = request.start_date
    end = request.end_date

    warmup = _WARMUP_DAYS.get(timeframe, 330)
    fetch_start = datetime.combine(start, datetime.min.time()) - timedelta(days=warmup)
    # yfinance treats ``end`` as exclusive, so include the final day.
    fetch_end = datetime.combine(end, datetime.min.time()) + timedelta(days=1)

    bars = await _load_bars(symbols, timeframe, fetch_start, fetch_end)
    if not bars:
        hint = ""
        if timeframe in _INTRADAY_TIMEFRAMES:
            hint = (
                " yfinance only provides intraday data for recent dates "
                "(~60 days for 5m/15m, ~730 days for 1h, ~7 days for 1m), "
                f"but the requested range starts {start.isoformat()}. "
                "Choose a more recent date range."
            )
        raise NoDataError(
            f"No historical {timeframe.value} data returned for {symbols}.{hint}"
        )
    loaded_symbols = sorted(bars.keys())

    data_handler = DataHandler(bars, timeframe)
    strategy = _build_strategy(name, loaded_symbols, trade_start=start)

    registry_features = sorted(
        {_FEATURE_KEY_TO_REGISTRY.get(k, k) for k in strategy.get_required_features()}
    )
    pipeline = FeaturePipeline(registry, feature_names=registry_features, normalize=False)

    engine = BacktestEngine(
        {
            "strategy": strategy,
            "data_handler": data_handler,
            "feature_pipeline": pipeline,
            "feature_lookback": _FEATURE_LOOKBACK,
        }
    )

    result = await engine.run_async(
        initial_capital=request.initial_capital,
        timeframe=timeframe,
    )

    # --- Trim warmup: keep only bars within the requested window. ---
    timestamps = result.timestamps
    equity = result.equity_curve
    window_idx = [i for i, ts in enumerate(timestamps) if ts.date() >= start]
    i0 = window_idx[0] if window_idx else 0
    eq_w = equity[i0:]
    ts_w = timestamps[i0:]
    if len(eq_w) < 2:
        eq_w, ts_w = equity, timestamps

    trades_window = [t for t in result.trades if _trade_in_window(t, start)]

    # --- Metrics over the trimmed window. ---
    metrics_raw: dict[str, Any] = {}
    if len(eq_w) >= 2:
        metrics_raw = BacktestAnalytics(
            equity_curve=eq_w,
            timestamps=ts_w,
            trades=trades_window,
            periods_per_year=_PERIODS_PER_YEAR.get(timeframe, 252.0),
        ).compute_all()

    wins = [t["pnl"] for t in trades_window if t.get("pnl", 0) > 0]
    losses = [t["pnl"] for t in trades_window if t.get("pnl", 0) < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    final_value = eq_w[-1] if eq_w else request.initial_capital

    metrics: dict[str, Any] = {
        "total_return": round(_finite(metrics_raw.get("total_return"), cap=1e6), 4),
        "annualized_return": round(_finite(metrics_raw.get("annualized_return"), cap=1e6), 4),
        "volatility": round(_finite(metrics_raw.get("volatility"), cap=1e6), 4),
        "sharpe_ratio": round(_finite(metrics_raw.get("sharpe_ratio")), 2),
        "sortino_ratio": round(_finite(metrics_raw.get("sortino_ratio")), 2),
        "calmar_ratio": round(_finite(metrics_raw.get("calmar_ratio")), 2),
        "max_drawdown": round(_finite(metrics_raw.get("max_drawdown"), cap=1.0), 4),
        "win_rate": round(_finite(metrics_raw.get("win_rate"), cap=1.0), 4),
        "profit_factor": round(_finite(metrics_raw.get("profit_factor")), 2),
        "trades_count": int(metrics_raw.get("total_trades", len(trades_window))),
        "avg_win": round(_finite(avg_win, cap=1e9), 2),
        "avg_loss": round(_finite(avg_loss, cap=1e9), 2),
        "initial_capital": float(request.initial_capital),
        "final_value": round(_finite(final_value, cap=1e12), 2),
        "bars": len(eq_w),
        "symbols": loaded_symbols,
        "timeframe": timeframe.value,
        "engine": "real",
    }

    # --- Trade log shaped for the dashboard table. ---
    ui_trades: list[dict[str, Any]] = []
    for t in trades_window:
        exit_time = t.get("exit_time")
        ui_trades.append(
            {
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "quantity": round(_finite(t.get("quantity"), cap=1e12), 4),
                "entry_price": round(_finite(t.get("entry_price"), cap=1e12), 2),
                "exit_price": round(_finite(t.get("exit_price"), cap=1e12), 2),
                "pnl": round(_finite(t.get("pnl"), cap=1e12), 2),
                "date": exit_time.isoformat() if isinstance(exit_time, datetime) else None,
            }
        )

    equity_curve = [round(_finite(v, cap=1e12), 2) for v in eq_w]
    # Real per-point timestamps (ISO 8601) aligned 1:1 with equity_curve, so the
    # frontend can plot a correct time axis for any timeframe (incl. intraday).
    equity_timestamps = [t.isoformat() for t in ts_w]

    logger.info(
        "backtest_runner.complete",
        strategy=name,
        symbols=loaded_symbols,
        bars=len(eq_w),
        trades=len(ui_trades),
        total_return=metrics["total_return"],
    )

    return metrics, equity_curve, ui_trades, equity_timestamps
