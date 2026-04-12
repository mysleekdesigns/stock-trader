"""Alpaca Markets data provider (historical + real-time)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit

from src.core.exceptions import DataProviderError
from src.core.types import Bar, Quote, TimeFrame
from src.data.providers.base import BarCallback, DataProviderBase, TradeCallback

logger = structlog.get_logger(__name__)

# Map our canonical TimeFrame to Alpaca's TimeFrame objects.
_TIMEFRAME_MAP: dict[TimeFrame, AlpacaTimeFrame] = {
    TimeFrame.MINUTE_1: AlpacaTimeFrame(1, TimeFrameUnit.Minute),
    TimeFrame.MINUTE_5: AlpacaTimeFrame(5, TimeFrameUnit.Minute),
    TimeFrame.MINUTE_15: AlpacaTimeFrame(15, TimeFrameUnit.Minute),
    TimeFrame.HOUR_1: AlpacaTimeFrame(1, TimeFrameUnit.Hour),
    TimeFrame.DAILY: AlpacaTimeFrame(1, TimeFrameUnit.Day),
    TimeFrame.WEEKLY: AlpacaTimeFrame(1, TimeFrameUnit.Week),
}


class AlpacaProvider(DataProviderBase):
    """Data provider backed by the Alpaca Markets API.

    Supports both historical bar/quote retrieval and real-time WebSocket
    streaming for bars and trades.
    """

    def __init__(self) -> None:
        from src.core.config import get_settings

        settings = get_settings()
        self._api_key: str = settings.alpaca_api_key
        self._secret_key: str = settings.alpaca_secret_key

        if not self._api_key or not self._secret_key:
            raise DataProviderError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set",
                details={"provider": "alpaca"},
            )

        self._hist_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        self._stream: StockDataStream | None = None
        self._stream_task: asyncio.Task[None] | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    async def connect(self) -> None:
        await super().connect()
        if self._stream is None:
            self._stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        logger.info("alpaca.connected")

    async def disconnect(self) -> None:
        if self._stream is not None:
            try:
                close_result = self._stream.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
            except Exception:
                logger.warning("alpaca.stream_close_error", exc_info=True)
            self._stream = None
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
            self._stream_task = None
        await super().disconnect()
        logger.info("alpaca.disconnected")

    # ── historical data ─────────────────────────────────────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Bar]:
        alpaca_tf = _TIMEFRAME_MAP.get(timeframe)
        if alpaca_tf is None:
            raise DataProviderError(
                f"Unsupported timeframe: {timeframe}",
                details={"provider": "alpaca"},
            )

        log = logger.bind(symbol=symbol, timeframe=timeframe.value)
        log.info("alpaca.get_bars.start")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=start,
            end=end,
        )

        try:
            bar_set = await asyncio.to_thread(self._hist_client.get_stock_bars, request)
        except Exception as exc:
            raise DataProviderError(
                f"Alpaca bar fetch failed for {symbol}: {exc}",
                details={"provider": "alpaca"},
            ) from exc

        raw_bars = bar_set.get(symbol, []) if hasattr(bar_set, "get") else (bar_set.data.get(symbol, []) if hasattr(bar_set, "data") else [])

        bars: list[Bar] = []
        for b in raw_bars:
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=b.timestamp,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=int(b.volume),
                    timeframe=timeframe,
                    vwap=float(b.vwap) if getattr(b, "vwap", None) is not None else None,
                )
            )

        log.info("alpaca.get_bars.done", bar_count=len(bars))
        return bars

    # ── quote ───────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Quote:
        log = logger.bind(symbol=symbol)
        log.info("alpaca.get_quote.start")

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)

        try:
            quotes_resp = await asyncio.to_thread(
                self._hist_client.get_stock_latest_quote, request
            )
        except Exception as exc:
            raise DataProviderError(
                f"Alpaca quote fetch failed for {symbol}: {exc}",
                details={"provider": "alpaca"},
            ) from exc

        raw: Any = quotes_resp.get(symbol) if isinstance(quotes_resp, dict) else getattr(quotes_resp, symbol, None)
        if raw is None:
            raise DataProviderError(
                f"No quote data returned for {symbol}",
                details={"provider": "alpaca"},
            )

        quote = Quote(
            symbol=symbol,
            bid=Decimal(str(raw.bid_price)),
            ask=Decimal(str(raw.ask_price)),
            bid_size=int(raw.bid_size),
            ask_size=int(raw.ask_size),
            timestamp=raw.timestamp,
        )
        log.info("alpaca.get_quote.done", bid=str(quote.bid), ask=str(quote.ask))
        return quote

    # ── streaming ───────────────────────────────────────────────────────

    def _ensure_stream(self) -> StockDataStream:
        if self._stream is None:
            raise DataProviderError(
                "WebSocket stream not initialised. Call connect() first.",
                details={"provider": "alpaca"},
            )
        return self._stream

    def _start_stream_if_needed(self) -> None:
        """Run the stream in a background task if not already running."""
        stream = self._ensure_stream()
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.get_event_loop().create_task(
                asyncio.to_thread(stream.run)
            )

    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        callback: BarCallback,
    ) -> None:
        stream = self._ensure_stream()
        log = logger.bind(symbols=symbols, timeframe=timeframe.value)
        log.info("alpaca.subscribe_bars")

        async def _on_bar(bar: Any) -> None:
            converted = Bar(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                timeframe=timeframe,
                vwap=float(bar.vwap) if getattr(bar, "vwap", None) is not None else None,
            )
            await callback(converted)

        stream.subscribe_bars(_on_bar, *symbols)
        self._start_stream_if_needed()

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        stream = self._ensure_stream()
        logger.info("alpaca.subscribe_trades", symbols=symbols)

        async def _on_trade(trade: Any) -> None:
            await callback(
                {
                    "symbol": trade.symbol,
                    "price": float(trade.price),
                    "size": int(trade.size),
                    "timestamp": trade.timestamp,
                    "exchange": getattr(trade, "exchange", None),
                    "conditions": getattr(trade, "conditions", None),
                }
            )

        stream.subscribe_trades(_on_trade, *symbols)
        self._start_stream_if_needed()
