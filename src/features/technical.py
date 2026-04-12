"""Technical analysis features computed via the ``ta`` library.

Every public function computes one or more indicators on a DataFrame that must
contain standard OHLCV columns (``open``, ``high``, ``low``, ``close``,
``volume``).  All functions are registered with the global
:class:`FeatureRegistry` at import time.
"""

from __future__ import annotations

import pandas as pd
import structlog
from ta.momentum import (
    RSIIndicator,
    StochRSIIndicator,
    WilliamsRIndicator,
)
from ta.trend import (
    ADXIndicator,
    CCIIndicator,
    EMAIndicator,
    IchimokuIndicator,
    MACD,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from src.features.registry import FeatureRegistry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Global registry instance shared across all feature modules
# ---------------------------------------------------------------------------
registry = FeatureRegistry()

# ===========================================================================
# RSI
# ===========================================================================


@registry.feature("rsi_14", group="momentum", description="RSI 14-period")
def rsi_14(df: pd.DataFrame) -> pd.Series:
    return RSIIndicator(close=df["close"], window=14, fillna=False).rsi()


@registry.feature("rsi_28", group="momentum", description="RSI 28-period")
def rsi_28(df: pd.DataFrame) -> pd.Series:
    return RSIIndicator(close=df["close"], window=28, fillna=False).rsi()


# ===========================================================================
# MACD
# ===========================================================================


@registry.feature("macd", group="trend", description="MACD (12,26,9)")
def macd(df: pd.DataFrame) -> pd.DataFrame:
    ind = MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
        fillna=False,
    )
    return pd.DataFrame(
        {
            "macd_line": ind.macd(),
            "macd_signal": ind.macd_signal(),
            "macd_histogram": ind.macd_diff(),
        },
        index=df.index,
    )


# ===========================================================================
# Bollinger Bands
# ===========================================================================


@registry.feature("bollinger", group="volatility", description="Bollinger Bands (20,2)")
def bollinger(df: pd.DataFrame) -> pd.DataFrame:
    ind = BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2,
        fillna=False,
    )
    upper = ind.bollinger_hband()
    lower = ind.bollinger_lband()
    middle = ind.bollinger_mavg()
    bandwidth = (upper - lower) / middle
    pct_b = (df["close"] - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        },
        index=df.index,
    )


# ===========================================================================
# ATR
# ===========================================================================


@registry.feature("atr_14", group="volatility", description="ATR 14-period")
def atr_14(df: pd.DataFrame) -> pd.Series:
    return AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        fillna=False,
    ).average_true_range()


# ===========================================================================
# OBV
# ===========================================================================


@registry.feature("obv", group="volume", description="On-Balance Volume")
def obv(df: pd.DataFrame) -> pd.Series:
    return OnBalanceVolumeIndicator(
        close=df["close"],
        volume=df["volume"],
        fillna=False,
    ).on_balance_volume()


# ===========================================================================
# VWAP (intraday approximation)
# ===========================================================================


@registry.feature("vwap", group="volume", description="Intraday VWAP approximation")
def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP using typical price * volume.

    For multi-day data this resets each day if the index is a DatetimeIndex;
    otherwise it computes a running VWAP over the entire frame.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)

    if isinstance(df.index, pd.DatetimeIndex):
        date = df.index.date
        cum_tp_vol = (typical * vol).groupby(date).cumsum()
        cum_vol = vol.groupby(date).cumsum()
    else:
        cum_tp_vol = (typical * vol).cumsum()
        cum_vol = vol.cumsum()

    result = cum_tp_vol / cum_vol.replace(0, float("nan"))
    return result


# ===========================================================================
# ADX
# ===========================================================================


@registry.feature("adx_14", group="trend", description="ADX / +DI / -DI (14)")
def adx_14(df: pd.DataFrame) -> pd.DataFrame:
    ind = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        fillna=False,
    )
    return pd.DataFrame(
        {
            "adx": ind.adx(),
            "adx_pos_di": ind.adx_pos(),
            "adx_neg_di": ind.adx_neg(),
        },
        index=df.index,
    )


# ===========================================================================
# Stochastic Oscillator
# ===========================================================================


@registry.feature("stochastic", group="momentum", description="Stochastic (14,3)")
def stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic %K / %D via manual computation (ta's StochasticOscillator)."""
    window_k = 14
    smooth_k = 3

    low_min = df["low"].rolling(window=window_k, min_periods=1).min()
    high_max = df["high"].rolling(window=window_k, min_periods=1).max()
    denom = high_max - low_min
    raw_k = 100.0 * (df["close"] - low_min) / denom.replace(0, float("nan"))
    k = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    d = k.rolling(window=smooth_k, min_periods=1).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)


# ===========================================================================
# CCI
# ===========================================================================


@registry.feature("cci_20", group="momentum", description="CCI 20-period")
def cci_20(df: pd.DataFrame) -> pd.Series:
    return CCIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=20,
        fillna=False,
    ).cci()


# ===========================================================================
# Williams %R
# ===========================================================================


@registry.feature("williams_r_14", group="momentum", description="Williams %R (14)")
def williams_r_14(df: pd.DataFrame) -> pd.Series:
    return WilliamsRIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        lbp=14,
        fillna=False,
    ).williams_r()


# ===========================================================================
# Ichimoku Cloud
# ===========================================================================


@registry.feature("ichimoku", group="trend", description="Ichimoku Cloud")
def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    ind = IchimokuIndicator(
        high=df["high"],
        low=df["low"],
        window1=9,
        window2=26,
        window3=52,
        fillna=False,
    )
    return pd.DataFrame(
        {
            "ichimoku_tenkan": ind.ichimoku_conversion_line(),
            "ichimoku_kijun": ind.ichimoku_base_line(),
            "ichimoku_senkou_a": ind.ichimoku_a(),
            "ichimoku_senkou_b": ind.ichimoku_b(),
        },
        index=df.index,
    )


# ===========================================================================
# Exponential Moving Averages
# ===========================================================================

_EMA_WINDOWS = [10, 20, 50, 200]


def _make_ema(window: int) -> Callable:
    """Factory that creates an EMA feature function for the given window."""

    def _ema(df: pd.DataFrame) -> pd.Series:
        return EMAIndicator(close=df["close"], window=window, fillna=False).ema_indicator()

    _ema.__doc__ = f"EMA {window}-period"
    return _ema


from typing import Callable  # noqa: E402 — needed for factory

for _w in _EMA_WINDOWS:
    registry.register(
        name=f"ema_{_w}",
        compute_fn=_make_ema(_w),
        group="trend",
        description=f"EMA {_w}-period",
    )


# ===========================================================================
# Simple Moving Averages
# ===========================================================================

_SMA_WINDOWS = [10, 20, 50, 200]


def _make_sma(window: int) -> Callable:
    def _sma(df: pd.DataFrame) -> pd.Series:
        return SMAIndicator(close=df["close"], window=window, fillna=False).sma_indicator()

    _sma.__doc__ = f"SMA {window}-period"
    return _sma


for _w in _SMA_WINDOWS:
    registry.register(
        name=f"sma_{_w}",
        compute_fn=_make_sma(_w),
        group="trend",
        description=f"SMA {_w}-period",
    )


# ===========================================================================
# Volume SMA
# ===========================================================================


@registry.feature("volume_sma_20", group="volume", description="Volume SMA 20-period")
def volume_sma_20(df: pd.DataFrame) -> pd.Series:
    return df["volume"].rolling(window=20, min_periods=1).mean()
