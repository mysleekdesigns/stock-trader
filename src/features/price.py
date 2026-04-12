"""Price-derived features: returns, volatility, gaps, momentum, mean-reversion.

All functions operate on a pandas DataFrame with standard OHLCV columns and are
registered with the global :data:`~src.features.technical.registry`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.features.technical import registry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Log returns at multiple horizons
# ---------------------------------------------------------------------------

_RETURN_PERIODS = [1, 5, 10, 21]


def _make_log_return(period: int):
    def _log_return(df: pd.DataFrame) -> pd.Series:
        return np.log(df["close"] / df["close"].shift(period))

    _log_return.__doc__ = f"Log return over {period} periods"
    return _log_return


for _p in _RETURN_PERIODS:
    registry.register(
        name=f"log_return_{_p}",
        compute_fn=_make_log_return(_p),
        group="returns",
        description=f"Log return ({_p} periods)",
    )

# ---------------------------------------------------------------------------
# Realized volatility (annualized rolling std of log returns)
# ---------------------------------------------------------------------------

_VOL_WINDOWS = [5, 10, 21, 63]
_ANNUALIZATION_FACTOR = np.sqrt(252)


def _make_realized_vol(window: int):
    def _realized_vol(df: pd.DataFrame) -> pd.Series:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        return log_ret.rolling(window=window, min_periods=max(2, window // 2)).std() * _ANNUALIZATION_FACTOR

    _realized_vol.__doc__ = f"Realized volatility ({window}-day window, annualized)"
    return _realized_vol


for _w in _VOL_WINDOWS:
    registry.register(
        name=f"realized_vol_{_w}",
        compute_fn=_make_realized_vol(_w),
        dependencies=["log_return_1"],
        group="volatility",
        description=f"Realized volatility ({_w}-day, annualized)",
    )

# ---------------------------------------------------------------------------
# Volume profile
# ---------------------------------------------------------------------------


@registry.feature(
    "volume_profile",
    dependencies=["volume_sma_20"],
    group="volume",
    description="Relative volume and volume z-score",
)
def volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"].astype(float)
    vol_sma = df["volume_sma_20"].astype(float)

    relative_volume = vol / vol_sma.replace(0, np.nan)

    rolling_std = vol.rolling(window=20, min_periods=2).std()
    volume_zscore = (vol - vol_sma) / rolling_std.replace(0, np.nan)

    return pd.DataFrame(
        {
            "relative_volume": relative_volume,
            "volume_zscore": volume_zscore,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Gap features
# ---------------------------------------------------------------------------


@registry.feature("gap_features", group="price", description="Overnight gap features")
def gap_features(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    gap = (df["open"] / prev_close) - 1.0
    gap_direction = np.sign(gap)
    return pd.DataFrame(
        {
            "overnight_gap": gap,
            "gap_direction": gap_direction,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Relative range
# ---------------------------------------------------------------------------


@registry.feature("relative_range", group="price", description="Relative range metrics")
def relative_range(df: pd.DataFrame) -> pd.DataFrame:
    hl_range = df["high"] - df["low"]
    hl_over_close = hl_range / df["close"].replace(0, np.nan)
    close_position = (df["close"] - df["low"]) / hl_range.replace(0, np.nan)
    return pd.DataFrame(
        {
            "hl_range_pct": hl_over_close,
            "close_position_in_range": close_position,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Price momentum (rate of change)
# ---------------------------------------------------------------------------

_MOMENTUM_WINDOWS = [5, 10, 21, 63]


def _make_momentum(window: int):
    def _momentum(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(periods=window)

    _momentum.__doc__ = f"Price rate of change ({window} periods)"
    return _momentum


for _w in _MOMENTUM_WINDOWS:
    registry.register(
        name=f"price_momentum_{_w}",
        compute_fn=_make_momentum(_w),
        group="momentum",
        description=f"Price momentum / ROC ({_w} periods)",
    )

# ---------------------------------------------------------------------------
# Mean-reversion z-score
# ---------------------------------------------------------------------------

_MR_WINDOWS = [10, 20, 50]


def _make_mr_zscore(window: int):
    def _mr_zscore(df: pd.DataFrame) -> pd.Series:
        sma = df["close"].rolling(window=window, min_periods=1).mean()
        std = df["close"].rolling(window=window, min_periods=2).std()
        return (df["close"] - sma) / std.replace(0, np.nan)

    _mr_zscore.__doc__ = f"Mean-reversion z-score (window={window})"
    return _mr_zscore


for _w in _MR_WINDOWS:
    registry.register(
        name=f"mr_zscore_{_w}",
        compute_fn=_make_mr_zscore(_w),
        group="mean_reversion",
        description=f"Mean-reversion z-score (window={_w})",
    )
