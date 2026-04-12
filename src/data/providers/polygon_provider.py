"""Polygon.io data provider (historical + real-time)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from polygon import RESTClient, WebSocketClient
from polygon.websocket.models import WebSocketMessage

from src.core.exceptions import DataProviderError
from src.core.types import Bar, Quote, TimeFrame
from src.data.providers.base import BarCallback, DataProviderBase, TradeCallback

logger = structlog.get_logger(__name__)

# Map our canonical TimeFrame to Polygon multiplier/timespan pairs.
_TIMEFRAME_MAP: dict[TimeFrame, tuple[int, str]] = {
    TimeFrame.MINUTE_1: (1, "minute"),
    TimeFrame.MINUTE_5: (5, "minute"),
    TimeFrame.MINUTE_15: (15, "minute"),
    TimeFrame.HOUR_1: (1, "hour"),
    TimeFrame.DAILY: (1, "day"),
    TimeFrame.WEEKLY: (1, "week"),
}


class PolygonProvider(DataProviderBase):
    """Data provider backed by the Polygon.io API.

    Supports historical bar/quote retrieval through the REST API and real-time
    WebSocket streaming for bars and trades.
    """

    def __init__(self) -> None:
        from src.core.config import get_settings

        settings = get_settings()
        self._api_key: str = settings.polygon_api_key

        if not self._api_key:
            raise DataProviderError(
                "POLYGON_API_KEY must be set",
                details={"provider": "polygon"},
            )

        self._rest_client = RESTClient(api_key=self._api_key)
        self._ws_client: WebSocketClient | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._bar_callback: BarCallback | None = None
        self._trade_callback: TradeCallback | None = None
        self._bar_timeframe: TimeFrame = TimeFrame.MINUTE_1

    # ── lifecycle ───────────────────────────────────────────────────────

    async def connect(self) -> None:
        await super().connect()
        self._ws_client = WebSocketClient(
            api_key=self._api_key,
            subscriptions=[],
        )
        logger.info("polygon.connected")

    async def disconnect(self) -> None:
        if self._ws_client is not None:
            try:
                await asyncio.to_thread(self._ws_client.close)
            except Exception:
                logger.warning("polygon.ws_close_error", exc_info=True)
            self._ws_client = None
        if self._ws_task is not None and not self._ws_task.done():
            self._ws_task.cancel()
            self._ws_task = None
        await super().disconnect()
        logger.info("polygon.disconnected")

    # ── historical data ─────────────────────────────────────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Bar]:
        tf_pair = _TIMEFRAME_MAP.get(timeframe)
        if tf_pair is None:
            raise DataProviderError(
                f"Unsupported timeframe: {timeframe}",
                details={"provider": "polygon"},
            )
        multiplier, timespan = tf_pair

        log = logger.bind(symbol=symbol, timeframe=timeframe.value)
        log.info("polygon.get_bars.start")

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000) if end else int(datetime.utcnow().timestamp() * 1000)

        try:
            aggs = await asyncio.to_thread(
                self._rest_client.get_aggs,
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_ms,
                to=end_ms,
                limit=50000,
            )
        except Exception as exc:
            raise DataProviderError(
                f"Polygon bar fetch failed for {symbol}: {exc}",
                details={"provider": "polygon"},
            ) from exc

        if not aggs:
            log.warning("polygon.get_bars.empty")
            return []

        bars: list[Bar] = []
        for agg in aggs:
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=datetime.utcfromtimestamp(agg.timestamp / 1000),
                    open=float(agg.open),
                    high=float(agg.high),
                    low=float(agg.low),
                    close=float(agg.close),
                    volume=int(agg.volume),
                    timeframe=timeframe,
                    vwap=float(agg.vwap) if getattr(agg, "vwap", None) is not None else None,
                )
            )

        log.info("polygon.get_bars.done", bar_count=len(bars))
        return bars

    # ── quote ───────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Quote:
        log = logger.bind(symbol=symbol)
        log.info("polygon.get_quote.start")

        try:
            quotes = await asyncio.to_thread(
                self._rest_client.get_last_quote, symbol
            )
        except Exception as exc:
            raise DataProviderError(
                f"Polygon quote fetch failed for {symbol}: {exc}",
                details={"provider": "polygon"},
            ) from exc

        if quotes is None:
            raise DataProviderError(
                f"No quote data returned for {symbol}",
                details={"provider": "polygon"},
            )

        # The REST client returns a LastQuote object with bid/ask fields.
        raw = quotes
        quote = Quote(
            symbol=symbol,
            bid=Decimal(str(raw.bid_price)) if hasattr(raw, "bid_price") else Decimal(str(raw.P if hasattr(raw, "P") else 0)),
            ask=Decimal(str(raw.ask_price)) if hasattr(raw, "ask_price") else Decimal(str(raw.p if hasattr(raw, "p") else 0)),
            bid_size=int(raw.bid_size) if hasattr(raw, "bid_size") else int(getattr(raw, "S", 0)),
            ask_size=int(raw.ask_size) if hasattr(raw, "ask_size") else int(getattr(raw, "s", 0)),
            timestamp=datetime.utcnow(),
        )
        log.info("polygon.get_quote.done", bid=str(quote.bid), ask=str(quote.ask))
        return quote

    # ── streaming ───────────────────────────────────────────────────────

    def _ensure_ws(self) -> WebSocketClient:
        if self._ws_client is None:
            raise DataProviderError(
                "WebSocket client not initialised. Call connect() first.",
                details={"provider": "polygon"},
            )
        return self._ws_client

    def _handle_messages(self, messages: list[WebSocketMessage]) -> None:
        """Synchronous callback dispatched by the Polygon WebSocket client."""
        loop = asyncio.get_event_loop()
        for msg in messages:
            event_type = getattr(msg, "ev", None) or getattr(msg, "event_type", None)

            if event_type in ("AM", "A") and self._bar_callback is not None:
                bar = Bar(
                    symbol=getattr(msg, "sym", getattr(msg, "symbol", "")),
                    timestamp=datetime.utcfromtimestamp(
                        getattr(msg, "s", getattr(msg, "start_timestamp", 0)) / 1000
                    ),
                    open=float(getattr(msg, "o", getattr(msg, "open", 0))),
                    high=float(getattr(msg, "h", getattr(msg, "high", 0))),
                    low=float(getattr(msg, "l", getattr(msg, "low", 0))),
                    close=float(getattr(msg, "c", getattr(msg, "close", 0))),
                    volume=int(getattr(msg, "v", getattr(msg, "volume", 0))),
                    timeframe=self._bar_timeframe,
                    vwap=float(getattr(msg, "vw", 0)) if getattr(msg, "vw", None) else None,
                )
                asyncio.run_coroutine_threadsafe(self._bar_callback(bar), loop)

            elif event_type == "T" and self._trade_callback is not None:
                trade: dict[str, Any] = {
                    "symbol": getattr(msg, "sym", getattr(msg, "symbol", "")),
                    "price": float(getattr(msg, "p", getattr(msg, "price", 0))),
                    "size": int(getattr(msg, "s", getattr(msg, "size", 0))),
                    "timestamp": datetime.utcfromtimestamp(
                        getattr(msg, "t", getattr(msg, "timestamp", 0)) / 1e9
                    ),
                    "exchange": getattr(msg, "x", getattr(msg, "exchange", None)),
                    "conditions": getattr(msg, "c", getattr(msg, "conditions", None)),
                }
                asyncio.run_coroutine_threadsafe(self._trade_callback(trade), loop)

    def _start_ws_if_needed(self) -> None:
        ws = self._ensure_ws()
        if self._ws_task is None or self._ws_task.done():
            ws.subscribe(ws.subscriptions)
            self._ws_task = asyncio.get_event_loop().create_task(
                asyncio.to_thread(ws.run, handle_msg=self._handle_messages)
            )

    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        callback: BarCallback,
    ) -> None:
        ws = self._ensure_ws()
        self._bar_callback = callback
        self._bar_timeframe = timeframe
        log = logger.bind(symbols=symbols, timeframe=timeframe.value)
        log.info("polygon.subscribe_bars")

        # Polygon uses "AM.*" channels for aggregate (minute) bars.
        subs = [f"AM.{s}" for s in symbols]
        ws.subscriptions = list(set(getattr(ws, "subscriptions", []) + subs))
        self._start_ws_if_needed()

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        ws = self._ensure_ws()
        self._trade_callback = callback
        logger.info("polygon.subscribe_trades", symbols=symbols)

        subs = [f"T.{s}" for s in symbols]
        ws.subscriptions = list(set(getattr(ws, "subscriptions", []) + subs))
        self._start_ws_if_needed()
