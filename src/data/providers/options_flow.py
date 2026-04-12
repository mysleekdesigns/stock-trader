"""Options flow data provider for unusual activity detection.

Provides :class:`OptionsFlowProvider` which surfaces unusual options activity
(large premium, high volume-to-OI ratios) that may indicate informed trading.
Currently ships with a placeholder implementation; swap in a live feed from
CBOE, Unusual Whales, or a similar vendor.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from src.core.types import Bar, Quote, TimeFrame
from src.data.providers.base import BarCallback, DataProviderBase, TradeCallback

logger = structlog.get_logger(__name__)

# Canonical schema for options flow data.
OPTIONS_FLOW_COLUMNS: list[str] = [
    "timestamp",
    "symbol",
    "strike",
    "expiry",
    "type",         # "call" or "put"
    "volume",
    "open_interest",
    "premium",
    "unusual_flag",
]

OPTIONS_FLOW_DTYPES: dict[str, str] = {
    "timestamp": "datetime64[ns]",
    "symbol": "object",
    "strike": "float64",
    "expiry": "datetime64[ns]",
    "type": "object",
    "volume": "int64",
    "open_interest": "int64",
    "premium": "float64",
    "unusual_flag": "bool",
}


class OptionsFlowProvider(DataProviderBase):
    """Data provider for options flow and unusual activity data.

    This is a **placeholder** implementation that returns empty DataFrames with
    the correct schema.  Replace the body of :meth:`get_unusual_activity` with
    a real vendor integration (e.g. Unusual Whales API, CBOE LiveVol, or
    Polygon options feed) for production use.

    The provider still satisfies the :class:`DataProviderBase` contract but
    raises ``NotImplementedError`` for bar/quote/streaming methods since
    options flow is a fundamentally different data shape.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        logger.info(
            "options_flow_provider.init",
            has_api_key=api_key is not None,
            base_url=base_url,
        )

    # ── Options-specific API ─────────────────────────────────────────

    async def get_unusual_activity(
        self,
        symbol: str,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """Fetch unusual options activity for the given symbol.

        Parameters
        ----------
        symbol
            Ticker symbol (e.g. ``"AAPL"``).
        lookback_days
            Number of calendar days to look back.  Default 5.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``symbol``, ``strike``, ``expiry``,
            ``type`` (call/put), ``volume``, ``open_interest``, ``premium``,
            ``unusual_flag``.
        """
        logger.info(
            "options_flow.get_unusual_activity",
            symbol=symbol,
            lookback_days=lookback_days,
        )

        # ── Placeholder: return empty DataFrame with correct schema ──
        # TODO: Replace with real vendor API call.
        df = pd.DataFrame(columns=OPTIONS_FLOW_COLUMNS)
        for col, dtype in OPTIONS_FLOW_DTYPES.items():
            df[col] = df[col].astype(dtype)

        logger.info(
            "options_flow.get_unusual_activity.done",
            symbol=symbol,
            rows=len(df),
        )
        return df

    # ── DataProviderBase contract (not applicable) ───────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Bar]:
        raise NotImplementedError(
            "OptionsFlowProvider does not serve OHLCV bars. "
            "Use get_unusual_activity() instead."
        )

    async def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError(
            "OptionsFlowProvider does not serve quotes. "
            "Use get_unusual_activity() instead."
        )

    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        callback: BarCallback,
    ) -> None:
        raise NotImplementedError(
            "OptionsFlowProvider does not support real-time bar subscriptions."
        )

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        raise NotImplementedError(
            "OptionsFlowProvider does not support real-time trade subscriptions."
        )
