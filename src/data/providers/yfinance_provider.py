"""Yahoo Finance data provider (historical-only)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal

import structlog
import yfinance as yf

from src.core.exceptions import DataProviderError
from src.core.types import Bar, Quote, TimeFrame
from src.data.providers.base import BarCallback, DataProviderBase, TradeCallback

logger = structlog.get_logger(__name__)

# Map our canonical TimeFrame enum to yfinance interval strings.
_TIMEFRAME_MAP: dict[TimeFrame, str] = {
    TimeFrame.MINUTE_1: "1m",
    TimeFrame.MINUTE_5: "5m",
    TimeFrame.MINUTE_15: "15m",
    TimeFrame.HOUR_1: "1h",
    TimeFrame.DAILY: "1d",
    TimeFrame.WEEKLY: "1wk",
}


class YFinanceProvider(DataProviderBase):
    """Data provider backed by the free Yahoo Finance API.

    This provider supports **historical** data only.  ``subscribe_bars`` and
    ``subscribe_trades`` raise ``NotImplementedError`` because yfinance does not
    offer a real-time WebSocket feed.
    """

    # ── historical data ─────────────────────────────────────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Bar]:
        interval = _TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise DataProviderError(
                f"Unsupported timeframe: {timeframe}",
                details={"provider": "yfinance"},
            )

        log = logger.bind(symbol=symbol, timeframe=timeframe.value, start=str(start), end=str(end))
        log.info("yfinance.get_bars.start")

        try:
            df = await asyncio.to_thread(
                yf.download,
                tickers=symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d") if end else None,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            raise DataProviderError(
                f"yfinance download failed for {symbol}: {exc}",
                details={"provider": "yfinance", "symbol": symbol},
            ) from exc

        if df is None or df.empty:
            log.warning("yfinance.get_bars.empty")
            return []

        # yfinance >= 1.2 returns MultiIndex columns (Price, Ticker) for
        # single-ticker downloads.  Flatten to simple column names.
        if isinstance(df.columns, __import__('pandas').MultiIndex):
            df.columns = df.columns.get_level_values(0)

        bars: list[Bar] = []
        for ts, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    timeframe=timeframe,
                )
            )

        log.info("yfinance.get_bars.done", bar_count=len(bars))
        return bars

    # ── quote ───────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Quote:
        log = logger.bind(symbol=symbol)
        log.info("yfinance.get_quote.start")

        try:
            info: dict = await asyncio.to_thread(lambda: yf.Ticker(symbol).info)
        except Exception as exc:
            raise DataProviderError(
                f"yfinance quote fetch failed for {symbol}: {exc}",
                details={"provider": "yfinance", "symbol": symbol},
            ) from exc

        bid = info.get("bid")
        ask = info.get("ask")
        last = info.get("regularMarketPrice") or info.get("previousClose")

        if bid is None or ask is None:
            if last is None:
                raise DataProviderError(
                    f"No price data available for {symbol}",
                    details={"provider": "yfinance", "symbol": symbol},
                )
            # Approximate bid/ask from last traded price.
            bid = float(last) * 0.999
            ask = float(last) * 1.001

        quote = Quote(
            symbol=symbol,
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            bid_size=int(info.get("bidSize", 0)),
            ask_size=int(info.get("askSize", 0)),
            timestamp=datetime.utcnow(),
        )
        log.info("yfinance.get_quote.done", bid=str(quote.bid), ask=str(quote.ask))
        return quote

    # ── streaming (not supported) ───────────────────────────────────────

    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        callback: BarCallback,
    ) -> None:
        raise NotImplementedError(
            "YFinanceProvider does not support real-time bar subscriptions."
        )

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        raise NotImplementedError(
            "YFinanceProvider does not support real-time trade subscriptions."
        )
