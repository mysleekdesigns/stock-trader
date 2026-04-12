"""Tick-to-bar and bar-to-bar aggregation.

:class:`BarAggregator` accumulates raw trades (ticks) into OHLCV bars at
one or more target time-frames, and can also up-sample smaller bars into
larger ones (e.g. 1 m -> 5 m -> 15 m -> 1 h).

Completed bars are emitted to a caller-supplied async callback.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog

from src.core.types import Bar, TimeFrame

logger = structlog.get_logger(__name__)

# Mapping from TimeFrame enum to its duration in seconds.
_TIMEFRAME_SECONDS: dict[TimeFrame, int] = {
    TimeFrame.MINUTE_1: 60,
    TimeFrame.MINUTE_5: 300,
    TimeFrame.MINUTE_15: 900,
    TimeFrame.HOUR_1: 3600,
    TimeFrame.DAILY: 86400,
    TimeFrame.WEEKLY: 604800,
}

# Which smaller time-frame feeds into which larger time-frame.
_UPSAMPLE_MAP: dict[TimeFrame, TimeFrame] = {
    TimeFrame.MINUTE_1: TimeFrame.MINUTE_5,
    TimeFrame.MINUTE_5: TimeFrame.MINUTE_15,
    TimeFrame.MINUTE_15: TimeFrame.HOUR_1,
    TimeFrame.HOUR_1: TimeFrame.DAILY,
}

# Async callback type for emitted bars.
BarEmitCallback = Callable[[Bar], Awaitable[None]]


# ---------------------------------------------------------------------------
# Partial (in-progress) bar state
# ---------------------------------------------------------------------------


@dataclass
class _PartialBar:
    """Mutable accumulator for a bar under construction."""

    symbol: str
    timeframe: TimeFrame
    period_start: datetime
    period_end: datetime
    open: float = 0.0
    high: float = -math.inf
    low: float = math.inf
    close: float = 0.0
    volume: int = 0
    cumulative_pv: float = 0.0  # price * volume (for VWAP)
    tick_count: int = 0

    @property
    def vwap(self) -> float | None:
        if self.volume == 0:
            return None
        return self.cumulative_pv / self.volume

    def to_bar(self) -> Bar:
        return Bar(
            symbol=self.symbol,
            timestamp=self.period_start,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            timeframe=self.timeframe,
            vwap=self.vwap,
        )


# ---------------------------------------------------------------------------
# BarAggregator
# ---------------------------------------------------------------------------


class BarAggregator:
    """Aggregate ticks into bars and smaller bars into larger bars.

    Parameters
    ----------
    output_timeframes:
        The target time-frames for which bars will be emitted.
    callback:
        Async callable invoked with each completed :class:`Bar`.
    """

    def __init__(
        self,
        output_timeframes: list[TimeFrame],
        callback: BarEmitCallback,
    ) -> None:
        self._output_timeframes = sorted(
            output_timeframes,
            key=lambda tf: _TIMEFRAME_SECONDS[tf],
        )
        self._callback = callback

        # (symbol, timeframe) -> current partial bar
        self._bars: dict[tuple[str, TimeFrame], _PartialBar] = {}

        logger.info(
            "bar_aggregator.init",
            timeframes=[tf.value for tf in self._output_timeframes],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_trade(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """Ingest a single trade tick.

        The tick is applied to every output time-frame whose period contains
        *timestamp*.  If the tick falls outside the current partial bar
        period, the old bar is completed and emitted first.
        """
        for tf in self._output_timeframes:
            await self._apply_tick(symbol, tf, price, volume, timestamp)

    async def on_bar(self, bar: Bar) -> None:
        """Ingest a completed bar and up-sample into larger time-frames.

        For example, a 1 m bar is aggregated into 5 m, 15 m, and 1 h bars
        (if those time-frames are configured).
        """
        source_tf = bar.timeframe

        # Walk up the time-frame ladder and merge into each configured target.
        current_tf = source_tf
        while current_tf in _UPSAMPLE_MAP:
            target_tf = _UPSAMPLE_MAP[current_tf]
            if target_tf not in self._output_timeframes:
                current_tf = target_tf
                continue

            await self._merge_bar(bar, target_tf)
            current_tf = target_tf

    async def flush(self, symbol: str | None = None) -> list[Bar]:
        """Emit partial (incomplete) bars and return them.

        Parameters
        ----------
        symbol:
            If provided, only flush bars for this symbol.  Otherwise flush
            all symbols.
        """
        flushed: list[Bar] = []

        keys = list(self._bars.keys())
        for key in keys:
            sym, _tf = key
            if symbol is not None and sym != symbol:
                continue

            partial = self._bars.pop(key)
            if partial.tick_count == 0:
                continue

            completed = partial.to_bar()
            flushed.append(completed)

            logger.info(
                "bar_aggregator.flush",
                symbol=sym,
                timeframe=_tf.value,
                close=completed.close,
                volume=completed.volume,
            )

            await self._callback(completed)

        return flushed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _apply_tick(
        self,
        symbol: str,
        timeframe: TimeFrame,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """Apply a single tick to the partial bar for *(symbol, timeframe)*."""
        key = (symbol, timeframe)
        period_start, period_end = self._get_bar_boundary(timestamp, timeframe)

        partial = self._bars.get(key)

        # If no partial bar exists, or the tick belongs to a new period,
        # close the previous bar and start a new one.
        if partial is None or timestamp >= partial.period_end:
            if partial is not None and partial.tick_count > 0:
                completed = partial.to_bar()
                logger.debug(
                    "bar_aggregator.bar_completed",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    close=completed.close,
                    volume=completed.volume,
                )
                await self._callback(completed)

            partial = _PartialBar(
                symbol=symbol,
                timeframe=timeframe,
                period_start=period_start,
                period_end=period_end,
            )
            self._bars[key] = partial

        # First tick in this bar sets the open.
        if partial.tick_count == 0:
            partial.open = price

        partial.high = max(partial.high, price)
        partial.low = min(partial.low, price)
        partial.close = price
        partial.volume += volume
        partial.cumulative_pv += price * volume
        partial.tick_count += 1

    async def _merge_bar(self, bar: Bar, target_tf: TimeFrame) -> None:
        """Merge a completed bar into the partial bar for *target_tf*."""
        key = (bar.symbol, target_tf)
        period_start, period_end = self._get_bar_boundary(bar.timestamp, target_tf)

        partial = self._bars.get(key)

        if partial is None or bar.timestamp >= partial.period_end:
            # Emit the old partial bar if it has data.
            if partial is not None and partial.tick_count > 0:
                completed = partial.to_bar()
                logger.debug(
                    "bar_aggregator.upsampled_bar_completed",
                    symbol=bar.symbol,
                    timeframe=target_tf.value,
                    close=completed.close,
                    volume=completed.volume,
                )
                await self._callback(completed)

            partial = _PartialBar(
                symbol=bar.symbol,
                timeframe=target_tf,
                period_start=period_start,
                period_end=period_end,
            )
            self._bars[key] = partial

        # Merge OHLCV
        if partial.tick_count == 0:
            partial.open = bar.open
            partial.high = bar.high
            partial.low = bar.low
        else:
            partial.high = max(partial.high, bar.high)
            partial.low = min(partial.low, bar.low)

        partial.close = bar.close
        partial.volume += bar.volume
        if bar.vwap is not None:
            partial.cumulative_pv += bar.vwap * bar.volume
        else:
            # Approximate with typical price.
            partial.cumulative_pv += ((bar.high + bar.low + bar.close) / 3.0) * bar.volume
        partial.tick_count += 1

    @staticmethod
    def _get_bar_boundary(
        timestamp: datetime,
        timeframe: TimeFrame,
    ) -> tuple[datetime, datetime]:
        """Compute the (start, end) boundaries for the bar containing *timestamp*.

        Returns
        -------
        (period_start, period_end) : tuple[datetime, datetime]
            ``period_end`` is exclusive (the first instant of the next bar).
        """
        seconds = _TIMEFRAME_SECONDS[timeframe]

        # Work with UTC epoch seconds for clean arithmetic.
        ts = timestamp.replace(tzinfo=None)
        epoch = datetime(1970, 1, 1)
        total_seconds = (ts - epoch).total_seconds()

        bar_index = int(total_seconds // seconds)
        start_epoch = bar_index * seconds
        end_epoch = start_epoch + seconds

        period_start = datetime(1970, 1, 1) + timedelta(seconds=start_epoch)
        period_end = datetime(1970, 1, 1) + timedelta(seconds=end_epoch)

        return period_start, period_end
