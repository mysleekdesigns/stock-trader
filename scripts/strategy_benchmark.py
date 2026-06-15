#!/usr/bin/env python3
"""Benchmark the rule-based strategies across a basket of stocks and crypto.

Fetches historical daily bars (yfinance), then for every (strategy, symbol)
pair runs the event-driven :class:`BacktestEngine` near-fully invested, trims a
fixed warm-up prefix so every strategy is judged over the same window, and
compares each strategy against a buy-&-hold benchmark for that symbol.

Outputs a JSON blob and a Markdown report.

Usage::

    python -m scripts.strategy_benchmark --out output/strategy_benchmark.json \
        --md output/strategy_benchmark.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime

import pandas as pd
import structlog

import src.features.price  # noqa: F401  -- registers price features
from src.backtest.analytics import BacktestAnalytics
from src.backtest.data_handler import DataHandler
from src.backtest.engine import BacktestEngine
from src.backtest.runner import _FEATURE_KEY_TO_REGISTRY
from src.core.types import TimeFrame
from src.features.pipeline import FeaturePipeline
from src.features.technical import registry
from src.strategies.connors_rsi2 import ConnorsRSI2Strategy
from src.strategies.donchian_breakout import DonchianBreakoutStrategy, DonchianConfig
from src.strategies.dual_momentum import DualMomentumStrategy
from src.strategies.macd_trend import MACDTrendConfig, MACDTrendStrategy
from src.strategies.supertrend import SupertrendConfig, SupertrendStrategy

# Risk-free rate for Sharpe/Sortino. 0.0 keeps low-exposure strategies (which
# sit in cash much of the time) comparable to fully-invested ones rather than
# penalising idle cash against a 5% hurdle.
RISK_FREE = 0.0


def build_strategy(name: str, symbol: str, allow_short: bool):
    """Construct a single-symbol strategy instance for the benchmark.

    Trend strategies are run long/flat by default: the simple backtest
    portfolio has no margin model, so unconstrained shorts on assets that rally
    many-fold (e.g. SOL) produce meaningless sub-zero equity.  Long/flat is also
    how these are realistically run on spot stock/crypto accounts.
    """
    if name == "donchian_breakout":
        return DonchianBreakoutStrategy(symbol, DonchianConfig(allow_short=allow_short))
    if name == "supertrend":
        return SupertrendStrategy(symbol, SupertrendConfig(allow_short=allow_short))
    if name == "macd_trend":
        return MACDTrendStrategy(symbol, MACDTrendConfig(allow_short=allow_short))
    if name == "connors_rsi2":
        return ConnorsRSI2Strategy(symbol)  # long/flat mean reversion by design
    if name == "dual_momentum":
        return DualMomentumStrategy(symbol)  # long/flat by design
    raise ValueError(f"unknown strategy {name}")

STRATEGIES = [
    "donchian_breakout",
    "supertrend",
    "macd_trend",
    "connors_rsi2",
    "dual_momentum",
]

DEFAULT_STOCKS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]
DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD"]


def fetch(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Download adjusted daily OHLCV for *symbol* via yfinance."""
    import yfinance as yf

    df = yf.download(
        symbol, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False,
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def metrics_from_equity(
    equity: list[float], timestamps: list[datetime], trades: list[dict], ppy: float,
) -> dict:
    if len(equity) < 2:
        return {}
    return BacktestAnalytics(
        equity_curve=equity, timestamps=timestamps, trades=trades,
        periods_per_year=ppy, risk_free_rate=RISK_FREE,
    ).compute_all()


def buy_hold(df: pd.DataFrame, start_idx: int, capital: float, ppy: float) -> dict:
    closes = df["close"].to_numpy()[start_idx:]
    ts = df.index.to_pydatetime().tolist()[start_idx:]
    if len(closes) < 2 or closes[0] <= 0:
        return {}
    equity = [float(capital * c / closes[0]) for c in closes]
    return metrics_from_equity(equity, ts, [], ppy)


async def run_one(
    strategy: str, symbol: str, df: pd.DataFrame, capital: float,
    warmup_bars: int, ppy: float, allow_short: bool,
) -> dict:
    strat = build_strategy(strategy, symbol, allow_short)
    required = sorted({_FEATURE_KEY_TO_REGISTRY.get(k, k) for k in strat.get_required_features()})
    pipeline = FeaturePipeline(registry, feature_names=required, normalize=False)
    engine = BacktestEngine({
        "strategy": strat,
        "data_handler": DataHandler({symbol: df}, TimeFrame.DAILY),
        "feature_pipeline": pipeline,
        "feature_lookback": 250,
        "max_position_pct": 0.95,
    })
    result = await engine.run_async(initial_capital=capital)

    # Trim a fixed warm-up prefix so every strategy is judged over one window.
    eq, ts = result.equity_curve, result.timestamps
    i0 = min(warmup_bars, max(0, len(eq) - 2))
    eq_w, ts_w = eq[i0:], ts[i0:]
    cutoff = ts_w[0] if ts_w else None
    trades_w = [
        t for t in result.trades
        if not (cutoff and isinstance(t.get("exit_time"), datetime) and t["exit_time"] < cutoff)
    ]
    m = metrics_from_equity(eq_w, ts_w, trades_w, ppy)
    return {
        "strategy": strategy,
        "symbol": symbol,
        "total_return": m.get("total_return"),
        "annualized_return": m.get("annualized_return"),
        "sharpe_ratio": m.get("sharpe_ratio"),
        "sortino_ratio": m.get("sortino_ratio"),
        "max_drawdown": m.get("max_drawdown"),
        "win_rate": m.get("win_rate"),
        "profit_factor": m.get("profit_factor"),
        "trades": m.get("total_trades", len(trades_w)),
        "final_equity": eq_w[-1] if eq_w else capital,
    }


async def main_async(args: argparse.Namespace) -> None:
    universe = [(s, "stock") for s in args.stocks] + [(c, "crypto") for c in args.crypto]

    # Fetch all data first.
    data: dict[str, dict] = {}
    for symbol, kind in universe:
        start = args.start_crypto if kind == "crypto" else args.start_stocks
        df = fetch(symbol, start, args.end)
        if df is None or len(df) < args.warmup_bars + 30:
            print(f"  ! skip {symbol}: insufficient data ({0 if df is None else len(df)} bars)")
            continue
        data[symbol] = {"df": df, "kind": kind, "ppy": 365.0 if kind == "crypto" else 252.0}
        print(
            f"  fetched {symbol:9s} {len(df):5d} bars  "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )

    results: list[dict] = []
    benchmarks: dict[str, dict] = {}
    for symbol, info in data.items():
        df, ppy = info["df"], info["ppy"]
        i0 = min(args.warmup_bars, max(0, len(df) - 2))
        benchmarks[symbol] = {
            "symbol": symbol, "kind": info["kind"],
            **buy_hold(df, i0, args.capital, ppy),
        }
        for strategy in STRATEGIES:
            r = await run_one(
                strategy, symbol, df, args.capital, args.warmup_bars, ppy, args.allow_short,
            )
            r["kind"] = info["kind"]
            results.append(r)
            print(
                f"  {strategy:18s} {symbol:9s} ret={_f(r['total_return']):>8}  "
                f"sharpe={_f(r['sharpe_ratio']):>6}  maxDD={_f(r['max_drawdown']):>6}  "
                f"win={_f(r['win_rate']):>5}  trades={r['trades']}"
            )

    payload = {
        "generated_at": args.now,
        "config": {
            "stocks": args.stocks, "crypto": args.crypto,
            "start_stocks": args.start_stocks, "start_crypto": args.start_crypto,
            "end": args.end, "capital": args.capital,
            "warmup_bars": args.warmup_bars, "max_position_pct": 0.95,
            "allow_short": args.allow_short, "risk_free_rate": RISK_FREE,
        },
        "results": results,
        "benchmarks": benchmarks,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {args.out}")

    md = render_markdown(payload)
    with open(args.md, "w") as f:
        f.write(md)
    print(f"wrote {args.md}")


def _f(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def render_markdown(payload: dict) -> str:
    results = payload["results"]
    benches = payload["benchmarks"]
    cfg = payload["config"]
    lines: list[str] = []
    lines.append("# Strategy Backtest Benchmark\n")
    lines.append(f"_Generated {payload['generated_at']}_\n")
    lines.append(
        f"- **Stocks** ({cfg['start_stocks']} → {cfg['end']}): {', '.join(cfg['stocks'])}\n"
        f"- **Crypto** ({cfg['start_crypto']} → {cfg['end']}): {', '.join(cfg['crypto'])}\n"
        f"- Capital ${cfg['capital']:,.0f}, near-full deployment "
        f"({cfg['max_position_pct']:.0%}/signal), {cfg['warmup_bars']}-bar warm-up trimmed, "
        f"5 bps slippage + commissions, Sharpe annualised 252 (stocks) / 365 (crypto).\n"
    )

    for kind in ("stock", "crypto"):
        krows = [r for r in results if r["kind"] == kind]
        if not krows:
            continue
        lines.append(f"\n## {kind.capitalize()}s — average across symbols\n")
        lines.append(
            "| Strategy | Avg total return | Avg Sharpe | Avg max DD "
            "| Avg win rate | Avg trades |"
        )
        lines.append("|---|--:|--:|--:|--:|--:|")
        for strat in STRATEGIES:
            srows = [r for r in krows if r["strategy"] == strat]
            lines.append(
                f"| {strat} | {_pct(_avg(srows, 'total_return'))} "
                f"| {_f(_avg(srows, 'sharpe_ratio'), 2)} "
                f"| {_pct(_avg(srows, 'max_drawdown'))} "
                f"| {_pct(_avg(srows, 'win_rate'))} "
                f"| {_f(_avg(srows, 'trades'), 0)} |"
            )
        bench_rows = [b for b in benches.values() if b.get("kind") == kind]
        lines.append(
            f"| _buy & hold_ | {_pct(_avg(bench_rows, 'total_return'))} "
            f"| {_f(_avg(bench_rows, 'sharpe_ratio'), 2)} "
            f"| {_pct(_avg(bench_rows, 'max_drawdown'))} | — | — |"
        )

    lines.append("\n## Per-symbol detail\n")
    lines.append(
        "| Strategy | Symbol | Total return | Sharpe | Sortino "
        "| Max DD | Win rate | Profit factor | Trades |"
    )
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in results:
        lines.append(
            f"| {r['strategy']} | {r['symbol']} | {_pct(r['total_return'])} "
            f"| {_f(r['sharpe_ratio'], 2)} | {_f(r['sortino_ratio'], 2)} "
            f"| {_pct(r['max_drawdown'])} | {_pct(r['win_rate'])} "
            f"| {_f(r['profit_factor'], 2)} | {r['trades']} |"
        )
    lines.append("\n## Buy-&-hold benchmark (same window)\n")
    lines.append("| Symbol | Total return | Sharpe | Max DD |")
    lines.append("|---|--:|--:|--:|")
    for b in benches.values():
        lines.append(
            f"| {b['symbol']} | {_pct(b.get('total_return'))} "
            f"| {_f(b.get('sharpe_ratio'), 2)} | {_pct(b.get('max_drawdown'))} |"
        )
    return "\n".join(lines) + "\n"


def _pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return str(v)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark rule-based strategies across stocks and crypto.",
    )
    p.add_argument("--stocks", nargs="+", default=DEFAULT_STOCKS)
    p.add_argument("--crypto", nargs="+", default=DEFAULT_CRYPTO)
    p.add_argument("--start-stocks", default="2014-01-01")
    p.add_argument("--start-crypto", default="2017-01-01")
    p.add_argument("--end", default="2026-06-12")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--warmup-bars", type=int, default=250)
    p.add_argument(
        "--allow-short", action="store_true",
        help="Let trend strategies short (long/flat only by default).",
    )
    p.add_argument("--out", default="output/strategy_benchmark.json")
    p.add_argument("--md", default="output/strategy_benchmark.md")
    p.add_argument("--now", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # Silence per-bar feature-warmup noise (ATR on <14 rows raises, is caught).
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
