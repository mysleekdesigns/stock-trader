#!/usr/bin/env python3
"""CLI for running backtests.

Usage examples::

    python -m scripts.run_backtest --symbols SPY QQQ --start 2023-01-01 --end 2024-01-01
    python -m scripts.run_backtest --symbols AAPL --strategy momentum --initial-capital 50000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backtest over historical data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Ticker symbols to backtest (e.g. SPY QQQ AAPL).",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "1h", "1d", "1w"],
        help="Bar timeframe.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        choices=[
            "momentum",
            "mean_reversion",
            "ensemble",
            "donchian_breakout",
            "supertrend",
            "connors_rsi2",
            "macd_trend",
            "dual_momentum",
        ],
        help="Strategy to use.",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.95,
        help=(
            "Fraction of equity deployed per signal at full strength. The app "
            "default is 0.10 (multi-strategy cap); 0.95 gives a single strategy "
            "near-full deployment for standalone evaluation."
        ),
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting portfolio value in USD.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for HTML report output. If omitted, no report is written.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="TimescaleDB connection URL. If provided, data is fetched from DB first.",
    )
    return parser


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        logger.warning("config_file_not_found", path=str(path))
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_configs(config_dir: str) -> dict:
    """Load all YAML configs from the config directory."""
    base = Path(config_dir)
    return {
        "assets": load_yaml(base / "assets.yaml"),
        "strategies": load_yaml(base / "strategies.yaml"),
        "risk_limits": load_yaml(base / "risk_limits.yaml"),
    }


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_bars_yfinance(
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe_str: str,
) -> dict[str, list]:
    """Fetch historical bars from yfinance for each symbol."""
    from src.core.types import TimeFrame
    from src.data.providers.yfinance_provider import YFinanceProvider

    provider = YFinanceProvider()
    timeframe = TimeFrame(timeframe_str)
    results: dict[str, list] = {}

    for symbol in symbols:
        logger.info("fetching_bars", symbol=symbol, source="yfinance")
        try:
            bars = await provider.get_bars(symbol, timeframe, start, end)
            results[symbol] = bars
            logger.info("bars_fetched", symbol=symbol, count=len(bars))
        except Exception:
            logger.exception("fetch_failed", symbol=symbol)

    return results


async def fetch_bars_db(
    db_url: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe_str: str,
) -> dict[str, list]:
    """Fetch historical bars from TimescaleDB."""
    from src.core.types import TimeFrame
    from src.data.storage.timeseries_store import TimeseriesStore

    store = TimeseriesStore(db_url)
    timeframe = TimeFrame(timeframe_str)
    results: dict[str, list] = {}

    try:
        for symbol in symbols:
            logger.info("fetching_bars", symbol=symbol, source="timescaledb")
            bars = await store.get_bars(symbol, timeframe, start, end)
            if bars:
                results[symbol] = bars
                logger.info("bars_fetched", symbol=symbol, count=len(bars))
            else:
                logger.warning("no_bars_in_db", symbol=symbol)
    finally:
        await store.close()

    return results


# ---------------------------------------------------------------------------
# Bars to DataFrame conversion
# ---------------------------------------------------------------------------

def bars_to_dataframes(bars_by_symbol: dict[str, list]) -> dict[str, pd.DataFrame]:
    """Convert lists of Bar objects to pandas DataFrames keyed by symbol."""
    import pandas as pd

    dfs: dict[str, pd.DataFrame] = {}
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        records = [
            {
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        timestamps = [b.timestamp for b in bars]
        df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps))
        df.index.name = "timestamp"
        dfs[symbol] = df
    return dfs


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def create_strategy(strategy_name: str, symbols: list[str]):
    """Build a per-symbol strategy dispatcher for *strategy_name*.

    Delegates to :func:`src.backtest.runner._build_strategy`, which keeps one
    single-symbol strategy instance per symbol (so per-symbol indicator and
    position state stay isolated) and routes each bar's features to the right
    instance by ``features["symbol"]``.
    """
    from src.backtest.runner import UnsupportedStrategyError, _build_strategy

    try:
        return _build_strategy(strategy_name, [s.upper() for s in symbols])
    except UnsupportedStrategyError as exc:
        logger.error("unknown_strategy", name=strategy_name, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_html_report(result, output_path: str) -> None:
    """Generate an HTML report from backtest results."""
    try:
        from src.backtest.report import BacktestReport
        report = BacktestReport(result)
        report.save_html(output_path)
        logger.info("report_saved", path=output_path)
    except ImportError:
        # Fallback: write a minimal HTML report
        html = _minimal_html_report(result)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html)
        logger.info("minimal_report_saved", path=output_path)


def _minimal_html_report(result) -> str:
    """Produce a basic HTML report when BacktestReport is unavailable."""
    metrics = {}
    if hasattr(result, "metrics"):
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
    elif isinstance(result, dict):
        metrics = result.get("metrics", result)

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 60%; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem 1rem; text-align: left; }}
th {{ background: #f4f4f4; }}
</style></head>
<body>
<h1>Backtest Report</h1>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(result) -> None:
    """Print key metrics to the console."""
    metrics = {}
    if hasattr(result, "metrics"):
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
    elif isinstance(result, dict):
        metrics = result.get("metrics", result)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:>12.4f}")
        else:
            print(f"  {key:30s}: {value!s:>12s}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    """Async entry point for the backtest runner."""
    logger.info(
        "backtest.start",
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        strategy=args.strategy,
        initial_capital=args.initial_capital,
    )

    # Parse dates
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    # Fetch data -- try DB first, fall back to yfinance
    bars_by_symbol: dict[str, list] = {}

    if args.db_url:
        logger.info("attempting_db_fetch", url=args.db_url)
        bars_by_symbol = await fetch_bars_db(
            args.db_url, args.symbols, start_dt, end_dt, args.timeframe,
        )
        # Fill in any missing symbols from yfinance
        missing = [s for s in args.symbols if s not in bars_by_symbol]
        if missing:
            logger.info("falling_back_to_yfinance", symbols=missing)
            yf_bars = await fetch_bars_yfinance(missing, start_dt, end_dt, args.timeframe)
            bars_by_symbol.update(yf_bars)
    else:
        bars_by_symbol = await fetch_bars_yfinance(
            args.symbols, start_dt, end_dt, args.timeframe,
        )

    if not bars_by_symbol:
        logger.error("no_data_fetched")
        print("ERROR: No data could be fetched for the requested symbols.")
        return 1

    # Convert to DataFrames
    dfs = bars_to_dataframes(bars_by_symbol)
    logger.info("data_prepared", symbols=list(dfs.keys()))

    # Initialize feature pipeline
    import src.features.price  # noqa: F401 -- registers price features
    from src.backtest.runner import _FEATURE_KEY_TO_REGISTRY
    from src.features.pipeline import FeaturePipeline
    from src.features.technical import registry as tech_registry

    # Create strategy (per-symbol dispatcher).
    strategy = create_strategy(args.strategy, args.symbols)
    if strategy is None:
        logger.warning("strategy_unavailable_running_data_only")

    # Compute only the features the strategy needs.  Computing the *full*
    # registry would drop every row whenever any long-window feature (e.g.
    # sma_200) is still NaN under a short look-back, starving the strategy.
    if strategy is not None:
        required = sorted(
            {_FEATURE_KEY_TO_REGISTRY.get(k, k) for k in strategy.get_required_features()}
        )
        pipeline = FeaturePipeline(tech_registry, feature_names=required, normalize=False)
    else:
        pipeline = FeaturePipeline(registry=tech_registry, normalize=False)

    # Run backtest
    try:
        from src.backtest.data_handler import DataHandler
        from src.backtest.engine import BacktestEngine
        from src.core.types import TimeFrame

        timeframe = TimeFrame(args.timeframe)
        data_handler = DataHandler(bars=dfs, timeframe=timeframe)

        engine = BacktestEngine(config={
            "data_handler": data_handler,
            "strategy": strategy,
            "feature_pipeline": pipeline,
            "feature_lookback": 250,
            "max_position_pct": args.max_position_pct,
        })

        result = await engine.run_async(
            symbols=args.symbols,
            timeframe=timeframe,
            initial_capital=args.initial_capital,
        )
        print_summary(result)

        if args.output:
            generate_html_report(result, args.output)

    except ImportError:
        logger.warning("backtest_engine_not_available_computing_features_only")

        # Compute features as a demo

        for symbol, df in dfs.items():
            logger.info("computing_features", symbol=symbol, rows=len(df))
            featured = pipeline.fit_transform(df)
            logger.info("features_computed", symbol=symbol, columns=list(featured.columns))

        # Build a simple result dict
        result = {
            "metrics": {
                "symbols": ", ".join(dfs.keys()),
                "bars_loaded": sum(len(df) for df in dfs.values()),
                "initial_capital": args.initial_capital,
                "status": "feature_computation_only (engine not yet available)",
            },
        }
        print_summary(result)

        if args.output:
            generate_html_report(result, args.output)

    logger.info("backtest.complete")
    return 0


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    parser = build_parser()
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
