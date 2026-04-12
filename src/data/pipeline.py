"""Data pipeline orchestrating ingest -> normalize -> transform -> store."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import structlog

from src.core.events import EventBus, MarketDataEvent
from src.core.types import Bar, TimeFrame
from src.core.exceptions import DataError
from src.data.providers.base import DataProviderBase

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lightweight store / feature protocols so the pipeline is testable even when
# concrete TimescaleDB / Redis / feature-pipeline classes aren't wired yet.
# ---------------------------------------------------------------------------

try:
    from src.data.storage.timeseries_store import TimeseriesStore
except ImportError:  # pragma: no cover
    TimeseriesStore = Any  # type: ignore[assignment,misc]

try:
    from src.data.storage.feature_store import FeatureStore
except ImportError:  # pragma: no cover
    FeatureStore = Any  # type: ignore[assignment,misc]

try:
    from src.features.registry import FeaturePipeline, FeatureRegistry
except ImportError:  # pragma: no cover
    FeaturePipeline = Any  # type: ignore[assignment,misc]
    FeatureRegistry = Any  # type: ignore[assignment,misc]


# Required OHLCV columns after normalisation
_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
_MAX_FFILL_GAP = 5  # forward-fill gaps smaller than this many rows


class DataValidationError(DataError):
    """Raised when incoming data fails validation checks."""


class DataPipeline:
    """Orchestrates the full data flow: ingest -> normalize -> transform -> store.

    This pipeline:
    1. Fetches raw OHLCV data from configured providers
    2. Normalizes and validates the data
    3. Computes features via the feature pipeline
    4. Stores raw bars in TimescaleDB and features in Redis
    5. Publishes MarketDataEvents to the event bus
    """

    def __init__(
        self,
        providers: list[DataProviderBase],
        timeseries_store: TimeseriesStore,
        feature_store: FeatureStore,
        feature_pipeline: FeaturePipeline,
        event_bus: EventBus,
    ) -> None:
        self._providers = providers
        self._ts_store = timeseries_store
        self._feature_store = feature_store
        self._feature_pipeline = feature_pipeline
        self._event_bus = event_bus

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------

    async def ingest(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Ingest OHLCV data for *symbols* from the first available provider.

        Tries providers in priority order, falling back to the next on failure.
        Returns a dict mapping symbol -> DataFrame of bars.
        """
        if not self._providers:
            raise DataError("No data providers configured")

        results: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            df: pd.DataFrame | None = None
            last_error: Exception | None = None

            for provider in self._providers:
                provider_name = provider.__class__.__name__
                try:
                    bars: list[Bar] = await provider.get_bars(
                        symbol, timeframe, start, end
                    )
                    if not bars:
                        logger.warning(
                            "pipeline.ingest.empty",
                            symbol=symbol,
                            provider=provider_name,
                        )
                        continue

                    df = self._bars_to_dataframe(bars)
                    logger.info(
                        "pipeline.ingest.ok",
                        symbol=symbol,
                        provider=provider_name,
                        rows=len(df),
                    )
                    break  # success — stop trying other providers
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "pipeline.ingest.provider_failed",
                        symbol=symbol,
                        provider=provider_name,
                        error=str(exc),
                    )

            if df is None or df.empty:
                msg = f"All providers failed for {symbol}"
                if last_error is not None:
                    msg = f"{msg}: {last_error}"
                logger.error("pipeline.ingest.all_failed", symbol=symbol)
                raise DataError(msg)

            results[symbol] = df

        return results

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and normalize raw OHLCV data.

        * Ensures required columns exist (open, high, low, close, volume)
        * Sorts by timestamp index
        * Removes duplicate indices
        * Validates: high >= low, open/close within [low, high], volume >= 0
        * Forward-fills small gaps (< 5 consecutive NaN rows) and logs
          warnings for larger ones
        * Casts numeric types
        """
        log = logger.bind(symbol=symbol)

        # --- required columns ---
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns for {symbol}: {missing}"
            )

        # --- ensure numeric types ---
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

        # --- sort & de-dup ---
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                raise DataValidationError(
                    f"DataFrame for {symbol} has no DatetimeIndex or 'timestamp' column"
                )
        df = df.sort_index()
        dup_count = int(df.index.duplicated().sum())
        if dup_count > 0:
            log.warning("pipeline.normalize.duplicates_removed", count=dup_count)
            df = df[~df.index.duplicated(keep="last")]

        # --- validate OHLCV integrity ---
        bad_hl = df["high"] < df["low"]
        if bad_hl.any():
            n_bad = int(bad_hl.sum())
            log.warning("pipeline.normalize.bad_high_low", count=n_bad)
            # Swap high/low where invalid
            df.loc[bad_hl, ["high", "low"]] = df.loc[bad_hl, ["low", "high"]].values

        # Clamp open/close into [low, high]
        df["open"] = df["open"].clip(lower=df["low"], upper=df["high"])
        df["close"] = df["close"].clip(lower=df["low"], upper=df["high"])

        bad_vol = df["volume"] < 0
        if bad_vol.any():
            log.warning(
                "pipeline.normalize.negative_volume",
                count=int(bad_vol.sum()),
            )
            df.loc[bad_vol, "volume"] = 0

        # --- handle gaps (NaN rows) ---
        nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
        if nan_mask.any():
            # Identify runs of NaN rows
            groups = (~nan_mask).cumsum()
            gap_sizes = nan_mask.groupby(groups).sum()
            large_gaps = gap_sizes[gap_sizes >= _MAX_FFILL_GAP]
            if not large_gaps.empty:
                log.warning(
                    "pipeline.normalize.large_gaps",
                    gap_count=len(large_gaps),
                    max_gap_size=int(large_gaps.max()),
                )
            # Forward-fill all small gaps; leave large ones as NaN then drop
            df = df.ffill(limit=_MAX_FFILL_GAP - 1)
            remaining_nans = df[["open", "high", "low", "close"]].isna().any(axis=1)
            if remaining_nans.any():
                drop_count = int(remaining_nans.sum())
                log.warning("pipeline.normalize.dropping_nan_rows", count=drop_count)
                df = df.dropna(subset=["open", "high", "low", "close"])

        log.info("pipeline.normalize.ok", rows=len(df))
        return df

    # ------------------------------------------------------------------
    # 3. Transform (feature computation)
    # ------------------------------------------------------------------

    async def transform(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Compute features for each symbol's normalised data.

        Delegates to the configured :class:`FeaturePipeline`.  If the pipeline
        exposes an async ``compute`` method it is awaited; otherwise the sync
        variant is called in the default executor so we never block the loop.
        """
        results: dict[str, pd.DataFrame] = {}

        for symbol, df in data.items():
            try:
                if hasattr(self._feature_pipeline, "compute"):
                    compute_fn = self._feature_pipeline.compute
                    if asyncio.iscoroutinefunction(compute_fn):
                        feature_df = await compute_fn(df)
                    else:
                        loop = asyncio.get_running_loop()
                        feature_df = await loop.run_in_executor(
                            None, compute_fn, df
                        )
                else:
                    # Fallback: return the data unchanged when feature
                    # pipeline doesn't expose ``compute``.
                    feature_df = df.copy()

                results[symbol] = feature_df
                logger.info(
                    "pipeline.transform.ok",
                    symbol=symbol,
                    feature_count=len(feature_df.columns),
                    rows=len(feature_df),
                )
            except Exception as exc:
                logger.error(
                    "pipeline.transform.failed",
                    symbol=symbol,
                    error=str(exc),
                )
                raise DataError(
                    f"Feature computation failed for {symbol}: {exc}"
                ) from exc

        return results

    # ------------------------------------------------------------------
    # 4. Store
    # ------------------------------------------------------------------

    async def store(
        self,
        raw_data: dict[str, pd.DataFrame],
        feature_data: dict[str, pd.DataFrame],
        timeframe: TimeFrame,
    ) -> None:
        """Persist raw bars to TimescaleDB and latest features to Redis.

        * Converts DataFrames back to :class:`Bar` objects for the timeseries
          store.
        * Writes the most recent feature row per symbol into Redis for
          low-latency access by the strategy layer.
        """
        ts_tasks: list[Any] = []
        feat_tasks: list[Any] = []

        # --- raw bars -> TimescaleDB ---
        for symbol, df in raw_data.items():
            bars = self._dataframe_to_bars(df, symbol, timeframe)
            if hasattr(self._ts_store, "write_bars"):
                write_fn = self._ts_store.write_bars
                if asyncio.iscoroutinefunction(write_fn):
                    ts_tasks.append(write_fn(bars))
                else:
                    ts_tasks.append(
                        asyncio.get_running_loop().run_in_executor(
                            None, write_fn, bars
                        )
                    )

        # --- features -> Redis ---
        for symbol, df in feature_data.items():
            if df.empty:
                continue
            # Store the full feature DataFrame (or latest row, depending on
            # what the feature store supports).
            if hasattr(self._feature_store, "store_features"):
                store_fn = self._feature_store.store_features
                if asyncio.iscoroutinefunction(store_fn):
                    feat_tasks.append(store_fn(symbol, timeframe, df))
                else:
                    feat_tasks.append(
                        asyncio.get_running_loop().run_in_executor(
                            None, store_fn, symbol, timeframe, df
                        )
                    )
            elif hasattr(self._feature_store, "set"):
                # Minimal key-value store: persist the latest feature row as
                # a JSON-serialisable dict.
                latest = df.iloc[-1].to_dict()
                key = f"features:{symbol}:{timeframe.value}"
                set_fn = self._feature_store.set
                if asyncio.iscoroutinefunction(set_fn):
                    feat_tasks.append(set_fn(key, latest))
                else:
                    feat_tasks.append(
                        asyncio.get_running_loop().run_in_executor(
                            None, set_fn, key, latest
                        )
                    )

        # Execute all writes concurrently
        all_tasks = ts_tasks + feat_tasks
        if all_tasks:
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "pipeline.store.write_error",
                        task_index=i,
                        error=str(result),
                    )
                    raise DataError(
                        f"Storage write failed: {result}"
                    ) from result

        logger.info(
            "pipeline.store.ok",
            ts_writes=len(ts_tasks),
            feat_writes=len(feat_tasks),
        )

    # ------------------------------------------------------------------
    # 5. Run (full orchestration)
    # ------------------------------------------------------------------

    async def run(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Execute the full pipeline: ingest -> normalize -> transform -> store.

        Returns the feature DataFrames for each symbol.
        """
        logger.info(
            "pipeline.start",
            symbols=symbols,
            timeframe=timeframe.value,
        )

        # 1. Ingest
        raw_data = await self.ingest(symbols, timeframe, start, end)

        # 2. Normalize
        normalized: dict[str, pd.DataFrame] = {}
        for symbol, df in raw_data.items():
            normalized[symbol] = self._normalize(df, symbol)

        # 3. Transform (compute features)
        feature_data = await self.transform(normalized)

        # 4. Store
        await self.store(normalized, feature_data, timeframe)

        # 5. Publish events
        for symbol in symbols:
            if symbol in feature_data:
                event = MarketDataEvent(
                    symbol=symbol,
                    data_type="bar",
                    payload={
                        "timeframe": timeframe.value,
                        "bar_count": len(feature_data[symbol]),
                    },
                )
                await self._event_bus.publish(event)

        total_bars = sum(len(df) for df in feature_data.values())
        logger.info(
            "pipeline.complete",
            symbols=symbols,
            bars_processed=total_bars,
        )
        return feature_data

    async def run_incremental(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        lookback_days: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Run the pipeline for recent data only (incremental update)."""
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)
        return await self.run(symbols, timeframe, start, end)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
        """Convert a list of :class:`Bar` objects to a DatetimeIndex DataFrame."""
        records = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        df = pd.DataFrame.from_records(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    @staticmethod
    def _dataframe_to_bars(
        df: pd.DataFrame, symbol: str, timeframe: TimeFrame
    ) -> list[Bar]:
        """Convert a normalised DataFrame back to a list of :class:`Bar` objects."""
        bars: list[Bar] = []
        for ts, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    timeframe=timeframe,
                )
            )
        return bars
