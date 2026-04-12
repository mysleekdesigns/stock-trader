"""Cross-asset features registered with the global feature registry.

These features capture market-regime information from VIX, yield curves, and
sector rotation.  Current implementations use placeholder computations that
derive proxy values from standard OHLCV data so that the feature pipeline can
operate without external data feeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.features.technical import registry

logger = structlog.get_logger(__name__)

# ===========================================================================
# VIX features (placeholders derived from realised volatility)
# ===========================================================================


@registry.feature(
    "vix_level",
    group="cross_asset",
    description="Proxy VIX level from 20-day realised volatility",
)
def vix_level(df: pd.DataFrame) -> pd.Series:
    """Annualised 20-day realised volatility as a VIX proxy."""
    if "close" not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window=20, min_periods=2).std() * np.sqrt(252) * 100


@registry.feature(
    "vix_change",
    dependencies=["vix_level"],
    group="cross_asset",
    description="Day-over-day change in proxy VIX level",
)
def vix_change(df: pd.DataFrame) -> pd.Series:
    """Daily change in the VIX proxy."""
    if "vix_level" not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return df["vix_level"].diff()


@registry.feature(
    "vix_term_structure",
    dependencies=["vix_level"],
    group="cross_asset",
    description="VIX term-structure proxy (short vs long realised vol)",
)
def vix_term_structure(df: pd.DataFrame) -> pd.Series:
    """Ratio of short-term (10-day) to long-term (60-day) realised vol.

    Values > 1 indicate backwardation (stressed markets); < 1 contango.
    """
    if "close" not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    short_vol = log_ret.rolling(window=10, min_periods=2).std() * np.sqrt(252)
    long_vol = log_ret.rolling(window=60, min_periods=2).std() * np.sqrt(252)
    return short_vol / long_vol.replace(0, float("nan"))


# ===========================================================================
# Yield curve (placeholder)
# ===========================================================================


@registry.feature(
    "yield_curve_slope",
    group="cross_asset",
    description="Placeholder yield-curve slope proxy from price momentum",
)
def yield_curve_slope(df: pd.DataFrame) -> pd.Series:
    """Placeholder: uses momentum of close prices as a proxy for yield-curve
    slope until a treasury data feed is connected.
    """
    if "close" not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    long_ma = df["close"].rolling(window=60, min_periods=1).mean()
    short_ma = df["close"].rolling(window=10, min_periods=1).mean()
    # Normalise by price level
    return (short_ma - long_ma) / df["close"].replace(0, float("nan"))


# ===========================================================================
# Sector rotation (relative strength)
# ===========================================================================


@registry.feature(
    "sector_rotation_signal",
    group="cross_asset",
    description="Relative strength proxy based on short vs long returns",
)
def sector_rotation_signal(df: pd.DataFrame) -> pd.Series:
    """Relative-strength signal: ratio of 10-day to 60-day cumulative return.

    Values > 1 suggest the asset is outperforming its longer-term trend
    (positive momentum); < 1 suggests underperformance.
    """
    if "close" not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    ret_short = df["close"].pct_change(periods=10)
    ret_long = df["close"].pct_change(periods=60)
    denom = ret_long.replace(0, float("nan"))
    return (1 + ret_short) / (1 + denom)
