"""FINRA dark-pool / off-exchange short-sale volume provider.

FINRA publishes a free daily "Short Sale Volume" file covering all consolidated
(on- and off-exchange) trades reported to its facilities.  Off-exchange volume
reported through FINRA's Trade Reporting Facilities (TRFs) is the standard
public proxy for dark-pool / ATS activity, and the short-volume ratio is a
widely used sentiment/microstructure signal.

Daily file (pipe-delimited)::

    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
    20240115|AAPL|12345678|4567|34567890|B,Q,N,D
    ...

This provider fetches and parses that real data.  Network/parse failures degrade
gracefully to an empty frame with the correct schema (mirroring
:class:`~src.data.alternative.sentiment.SentimentDataProvider`) so downstream
feature engineering and backtests remain runnable offline.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# FINRA consolidated NMS short-sale volume daily file.
_FINRA_DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"

_DARK_POOL_SCHEMA: dict[str, str] = {
    "date": "datetime64[ns]",
    "symbol": "object",
    "short_volume": "int64",
    "short_exempt_volume": "int64",
    "total_volume": "int64",
    "short_volume_ratio": "float64",
    "market": "object",
    "source": "object",
}


class DarkPoolDataProvider:
    """Fetches FINRA daily short-sale (dark-pool proxy) volume.

    Parameters
    ----------
    timeout:
        Per-request HTTP timeout in seconds.
    max_days:
        Safety cap on the number of daily files fetched in a single call.
    base_url:
        Override the FINRA URL template (used for testing).  Must contain a
        ``{date}`` placeholder formatted as ``YYYYMMDD``.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        max_days: int = 120,
        base_url: str = _FINRA_DAILY_URL,
    ) -> None:
        self._timeout = timeout
        self._max_days = max_days
        self._base_url = base_url
        logger.info("dark_pool_data_provider_init", max_days=max_days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_short_volume(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return daily short-sale / dark-pool volume for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol (case-insensitive, e.g. ``"AAPL"``).
        start, end:
            Inclusive date range.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``symbol``, ``short_volume``,
            ``short_exempt_volume``, ``total_volume``, ``short_volume_ratio``,
            ``market``, ``source``.  Empty (with schema) if no data is
            available.
        """
        symbol = symbol.upper()
        start_dt = pd.Timestamp(start).normalize()
        end_dt = pd.Timestamp(end).normalize()

        # FINRA files exist only for trading days; iterating business days keeps
        # the request count bounded and skips weekends.
        days = pd.bdate_range(start=start_dt, end=end_dt, freq="B")
        if len(days) == 0:
            return self._empty_frame()
        if len(days) > self._max_days:
            logger.warning(
                "dark_pool_range_truncated",
                requested=len(days),
                max_days=self._max_days,
            )
            days = days[-self._max_days :]

        frames: list[pd.DataFrame] = []
        for day in days:
            day_df = self._fetch_day(day)
            if day_df is None or day_df.empty:
                continue
            match = day_df[day_df["symbol"] == symbol]
            if not match.empty:
                frames.append(match)

        if not frames:
            logger.info(
                "dark_pool_no_data",
                symbol=symbol,
                start=str(start_dt.date()),
                end=str(end_dt.date()),
            )
            return self._empty_frame()

        df = pd.concat(frames, ignore_index=True)
        df = self._coerce_schema(df)
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(
            "dark_pool_data_retrieved",
            symbol=symbol,
            start=str(start_dt.date()),
            end=str(end_dt.date()),
            n_records=len(df),
        )
        return df

    def get_latest(self, symbol: str, lookback_days: int = 5) -> dict | None:
        """Return the most recent available short-volume record for *symbol*.

        Scans up to ``lookback_days`` business days back from today (FINRA
        publishes with a short lag, so the latest file may be a day or two old).
        Returns ``None`` if nothing is found.
        """
        end = pd.Timestamp.now().normalize()
        start = end - pd.tseries.offsets.BDay(lookback_days)
        df = self.get_short_volume(symbol, start, end)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_day(self, day: pd.Timestamp) -> pd.DataFrame | None:
        """Fetch and parse a single FINRA daily file. Returns ``None`` on error."""
        url = self._base_url.format(date=day.strftime("%Y%m%d"))
        try:
            import httpx

            resp = httpx.get(url, timeout=self._timeout)
            if resp.status_code != 200:
                logger.debug(
                    "dark_pool_day_unavailable",
                    date=str(day.date()),
                    status=resp.status_code,
                )
                return None
            return self._parse(resp.text)
        except ImportError:
            logger.error(
                "dark_pool.missing_dependency",
                message="httpx is not installed. Run: uv add httpx",
            )
            return None
        except Exception as exc:  # network error, timeout, etc.
            logger.debug("dark_pool_fetch_failed", date=str(day.date()), error=str(exc))
            return None

    @staticmethod
    def _parse(text: str) -> pd.DataFrame:
        """Parse a FINRA pipe-delimited short-volume file body."""
        rows: list[dict] = []
        for line in text.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            # Skip the header row and any footer (e.g. trailing record count).
            if parts[0].lower() in {"date", "trade date"} or not parts[0].isdigit():
                continue
            try:
                # Parse the date inside the guard: a digit-only-but-invalid date
                # (e.g. 20240199) must skip just that row, not the whole file.
                date_val = pd.to_datetime(parts[0], format="%Y%m%d")
                short_vol = int(parts[2])
                short_exempt = int(parts[3])
                total_vol = int(parts[4])
            except (ValueError, IndexError):
                continue
            market = parts[5] if len(parts) > 5 else ""
            ratio = (short_vol / total_vol) if total_vol > 0 else 0.0
            rows.append(
                {
                    "date": date_val,
                    "symbol": parts[1].upper(),
                    "short_volume": short_vol,
                    "short_exempt_volume": short_exempt,
                    "total_volume": total_vol,
                    "short_volume_ratio": round(ratio, 6),
                    "market": market,
                    "source": "finra",
                }
            )
        if not rows:
            return pd.DataFrame(columns=list(_DARK_POOL_SCHEMA.keys()))
        return pd.DataFrame(rows)

    @staticmethod
    def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all schema columns exist with the correct dtypes."""
        defaults: dict[str, object] = {
            "int64": 0,
            "float64": 0.0,
            "object": "",
            "datetime64[ns]": pd.NaT,
        }
        for col, dtype in _DARK_POOL_SCHEMA.items():
            if col not in df.columns:
                # Fill with a typed default for existing rows so the subsequent
                # int cast cannot hit IntCastingNaNError.
                df[col] = defaults.get(dtype, "")
            df[col] = df[col].astype(dtype)
        return df[list(_DARK_POOL_SCHEMA.keys())]

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema."""
        df = pd.DataFrame(columns=list(_DARK_POOL_SCHEMA.keys()))
        for col, dtype in _DARK_POOL_SCHEMA.items():
            df[col] = df[col].astype(dtype)
        return df
