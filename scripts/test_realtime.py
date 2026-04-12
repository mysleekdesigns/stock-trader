#!/usr/bin/env python3
"""Test Alpaca real-time WebSocket streaming.

Subscribes to bar and trade updates for the given symbols and prints
them to the console. Useful for verifying credentials, connectivity,
and data flow before wiring into the full pipeline.

Usage::

    python -m scripts.test_realtime --symbols AAPL MSFT
    python -m scripts.test_realtime --symbols SPY --trades-only
    python -m scripts.test_realtime --symbols AAPL --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from typing import Any

import structlog

from src.core.types import Bar, TimeFrame
from src.data.providers.alpaca_provider import AlpacaProvider

logger = structlog.get_logger(__name__)


async def on_bar(bar: Bar) -> None:
    """Print each incoming bar."""
    print(
        f"[BAR]  {bar.timestamp}  {bar.symbol:>6s}  "
        f"O={bar.open:.2f}  H={bar.high:.2f}  L={bar.low:.2f}  C={bar.close:.2f}  "
        f"V={bar.volume}"
    )


async def on_trade(trade: dict[str, Any]) -> None:
    """Print each incoming trade."""
    ts = trade.get("timestamp", "")
    print(
        f"[TRADE] {ts}  {trade['symbol']:>6s}  "
        f"P={trade['price']:.2f}  S={trade['size']}"
    )


async def async_main(args: argparse.Namespace) -> int:
    provider = AlpacaProvider()
    await provider.connect()

    print(f"Connected to Alpaca WebSocket stream.")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Duration: {args.duration}s (Ctrl+C to stop early)")
    print("-" * 70)

    if not args.trades_only:
        await provider.subscribe_bars(
            args.symbols, TimeFrame.MINUTE_1, on_bar
        )

    if not args.bars_only:
        await provider.subscribe_trades(args.symbols, on_trade)

    # Run until duration expires or interrupted
    stop = asyncio.Event()

    def _handle_signal() -> None:
        print("\nStopping...")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        await asyncio.wait_for(stop.wait(), timeout=args.duration)
    except asyncio.TimeoutError:
        print(f"\nDuration ({args.duration}s) reached.")

    await provider.disconnect()
    print("Disconnected.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Alpaca real-time WebSocket streaming.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Ticker symbols to subscribe to (e.g. AAPL MSFT SPY).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="How long to stream in seconds (default 300).",
    )
    parser.add_argument(
        "--bars-only",
        action="store_true",
        default=False,
        help="Subscribe to bars only (no trades).",
    )
    parser.add_argument(
        "--trades-only",
        action="store_true",
        default=False,
        help="Subscribe to trades only (no bars).",
    )

    args = parser.parse_args()

    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    sys.exit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
