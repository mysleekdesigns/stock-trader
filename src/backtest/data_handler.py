"""Historical data replay for backtesting.

The :class:`DataHandler` takes pre-loaded OHLCV DataFrames keyed by symbol
and yields ``(timestamp, {symbol: Bar})`` tuples in chronological order,
aligning timestamps across all symbols.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from src.core.types import Bar, TimeFrame

logger = structlog.get_logger(__name__)


class DataHandler:
    """Replay historical bars in chronological order across multiple symbols.

    Parameters
    ----------
    bars:
        Pre-loaded OHLCV data keyed by symbol.  Each DataFrame must have a
        :class:`DatetimeIndex` and columns ``open, high, low, close, volume``.
    timeframe:
        The bar timeframe (used when constructing :class:`Bar` objects).
    """

    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> None:
        self._bars = bars
        self._timeframe = timeframe
        self._symbols = sorted(bars.keys())

        # Build a sorted union of all timestamps across symbols.
        all_timestamps: set[datetime] = set()
        for symbol, df in self._bars.items():
            if df.empty:
                logger.warning("data_handler.empty_dataframe", symbol=symbol)
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"DataFrame for {symbol} must have a DatetimeIndex, "
                    f"got {type(df.index).__name__}"
                )
            all_timestamps.update(df.index.to_pydatetime().tolist())

        self._timestamps: list[datetime] = sorted(all_timestamps)
        self._cursor: int = 0

        logger.info(
            "data_handler.init",
            symbols=self._symbols,
            total_bars=len(self._timestamps),
            start=str(self._timestamps[0]) if self._timestamps else None,
            end=str(self._timestamps[-1]) if self._timestamps else None,
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_timeseries_store(
        cls,
        store: Any,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> DataHandler:
        """Load data from a timeseries store and construct a DataHandler.

        Parameters
        ----------
        store:
            Any object implementing ``get_bars(symbol, timeframe, start, end)``
            that returns a :class:`pd.DataFrame` with a DatetimeIndex.
        symbols:
            List of ticker symbols to load.
        timeframe:
            Bar timeframe.
        start, end:
            Date range (inclusive).
        """
        bars: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = store.get_bars(symbol, timeframe, start, end)
            if df is not None and not df.empty:
                bars[symbol] = df
                logger.info(
                    "data_handler.loaded_from_store",
                    symbol=symbol,
                    rows=len(df),
                )
            else:
                logger.warning("data_handler.no_data_from_store", symbol=symbol)
        return cls(bars=bars, timeframe=timeframe)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[datetime, dict[str, Bar]]]:
        """Yield ``(timestamp, {symbol: Bar})`` tuples in chronological order."""
        self._cursor = 0
        for ts in self._timestamps:
            bar_dict: dict[str, Bar] = {}
            for symbol in self._symbols:
                df = self._bars[symbol]
                if ts in df.index:
                    row = df.loc[ts]
                    bar_dict[symbol] = Bar(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                        timeframe=self._timeframe,
                        vwap=float(row["vwap"]) if "vwap" in row.index else None,
                    )
            self._cursor += 1
            if bar_dict:
                yield ts, bar_dict

    def __len__(self) -> int:
        return len(self._timestamps)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def peek_next(self) -> datetime | None:
        """Return the next timestamp without advancing the cursor, or ``None``."""
        if self._cursor < len(self._timestamps):
            return self._timestamps[self._cursor]
        return None

    def get_history(
        self,
        symbol: str,
        lookback: int,
        current_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        """Return the last *lookback* bars for *symbol* up to *current_timestamp*.

        If *current_timestamp* is ``None``, return the last *lookback* rows
        of the entire dataset.
        """
        if symbol not in self._bars:
            return pd.DataFrame()

        df = self._bars[symbol]
        if current_timestamp is not None:
            df = df.loc[:current_timestamp]
        return df.tail(lookback).copy()

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def timestamps(self) -> list[datetime]:
        return list(self._timestamps)
