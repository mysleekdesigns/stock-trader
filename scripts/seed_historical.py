#!/usr/bin/env python3
"""Bulk-load historical OHLCV data into TimescaleDB.

Usage examples::

    python -m scripts.seed_historical --symbols SPY QQQ --start 2020-01-01 --end 2024-01-01
    python -m scripts.seed_historical --symbols all --start 2022-01-01 --end 2024-01-01
    python -m scripts.seed_historical --symbols AAPL MSFT --timeframe 1h --start 2024-01-01 --end 2024-06-01
    python -m scripts.seed_historical --provider alpaca --symbols AAPL MSFT --start 2024-01-01 --end 2024-06-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger(__name__)

# Rate-limit delay between yfinance requests (seconds)
DEFAULT_RATE_LIMIT_DELAY = 1.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed historical OHLCV data from Alpaca or yfinance into TimescaleDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help=(
            "Ticker symbols to fetch (e.g. SPY QQQ AAPL). "
            'Use "all" to load every symbol from config/assets.yaml.'
        ),
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
        "--db-url",
        type=str,
        default="postgresql+asyncpg://trader:trader@localhost:5432/trading",
        help="TimescaleDB connection URL.",
    )
    parser.add_argument(
        "--assets-config",
        type=str,
        default="config/assets.yaml",
        help="Path to assets.yaml for resolving 'all' symbols.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="yfinance",
        choices=["yfinance", "alpaca"],
        help="Data provider to fetch bars from.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT_DELAY,
        help="Delay in seconds between API requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Fetch data but do not write to the database (useful for testing).",
    )
    return parser


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def resolve_symbols(raw_symbols: list[str], assets_config_path: str) -> list[str]:
    """Resolve the symbol list, expanding 'all' from config/assets.yaml."""
    if len(raw_symbols) == 1 and raw_symbols[0].lower() == "all":
        path = Path(assets_config_path)
        if not path.exists():
            logger.error("assets_config_not_found", path=str(path))
            print(f"ERROR: Assets config not found at {path}")
            sys.exit(1)

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        symbols: list[str] = []
        asset_classes = config.get("asset_classes", {})

        # Collect equity symbols
        us_equity = asset_classes.get("us_equity", {})
        symbols.extend(us_equity.get("symbols", []))

        # Collect sector ETF symbols
        sectors = asset_classes.get("sectors", {})
        symbols.extend(sectors.keys())

        if not symbols:
            logger.error("no_symbols_found_in_config", path=str(path))
            print("ERROR: No symbols found in assets config.")
            sys.exit(1)

        logger.info("symbols_resolved_from_config", count=len(symbols), symbols=symbols)
        return symbols

    return raw_symbols


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_symbol(
    provider,
    symbol: str,
    timeframe,
    start: datetime,
    end: datetime,
) -> list:
    """Fetch bars for a single symbol, returning an empty list on failure."""
    try:
        bars = await provider.get_bars(symbol, timeframe, start, end)
        logger.info("fetch_complete", symbol=symbol, bars=len(bars))
        return bars
    except Exception:
        logger.exception("fetch_failed", symbol=symbol)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    """Async entry point for the historical data seeder."""
    from src.core.types import TimeFrame

    # Resolve symbols
    symbols = resolve_symbols(args.symbols, args.assets_config)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    timeframe = TimeFrame(args.timeframe)

    logger.info(
        "seed_historical.start",
        symbols=symbols,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        provider=args.provider,
        dry_run=args.dry_run,
    )

    # Initialize provider
    if args.provider == "alpaca":
        from src.data.providers.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider()
        await provider.connect()
    else:
        from src.data.providers.yfinance_provider import YFinanceProvider

        provider = YFinanceProvider()

    # Initialize store (unless dry run)
    store = None
    if not args.dry_run:
        from src.data.storage.timeseries_store import TimeseriesStore

        store = TimeseriesStore(args.db_url)
        try:
            await store.init_db()
            logger.info("database_initialized")
        except Exception:
            logger.exception("database_init_failed")
            print("ERROR: Could not initialize the database. Check your --db-url.")
            return 1

    # Fetch and store for each symbol
    total_bars = 0
    total_stored = 0
    failed_symbols: list[str] = []

    for i, symbol in enumerate(symbols):
        logger.info(
            "processing_symbol",
            symbol=symbol,
            progress=f"{i + 1}/{len(symbols)}",
        )

        bars = await fetch_symbol(provider, symbol, timeframe, start_dt, end_dt)

        if not bars:
            failed_symbols.append(symbol)
            continue

        total_bars += len(bars)

        if store and not args.dry_run:
            try:
                inserted = await store.store_bars(bars)
                total_stored += inserted
                logger.info(
                    "bars_stored",
                    symbol=symbol,
                    fetched=len(bars),
                    inserted=inserted,
                )
            except Exception:
                logger.exception("store_failed", symbol=symbol)
                failed_symbols.append(symbol)
        else:
            logger.info("dry_run_skip_store", symbol=symbol, bars=len(bars))

        # Rate limiting between requests
        if i < len(symbols) - 1:
            logger.debug("rate_limit_pause", delay=args.rate_limit)
            await asyncio.sleep(args.rate_limit)

    # Cleanup
    if args.provider == "alpaca":
        await provider.disconnect()
    if store:
        await store.close()

    # Summary
    print("\n" + "=" * 60)
    print("SEED HISTORICAL DATA SUMMARY")
    print("=" * 60)
    print(f"  Symbols requested:  {len(symbols)}")
    print(f"  Symbols succeeded:  {len(symbols) - len(failed_symbols)}")
    print(f"  Symbols failed:     {len(failed_symbols)}")
    if failed_symbols:
        print(f"  Failed list:        {', '.join(failed_symbols)}")
    print(f"  Total bars fetched: {total_bars}")
    print(f"  Total bars stored:  {total_stored}")
    print(f"  Dry run:            {args.dry_run}")
    print("=" * 60 + "\n")

    logger.info(
        "seed_historical.complete",
        total_bars=total_bars,
        total_stored=total_stored,
        failed=failed_symbols,
    )

    return 0 if not failed_symbols else 1


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
