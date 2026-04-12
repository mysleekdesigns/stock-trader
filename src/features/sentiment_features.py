"""Sentiment-derived features registered with the global feature registry.

These features expect the input DataFrame to contain a ``sentiment_score``
column (e.g. produced by :class:`SentimentDataProvider`).  If the column is
absent the functions return NaN series gracefully.
"""

from __future__ import annotations

import pandas as pd
import structlog

from src.features.technical import registry

logger = structlog.get_logger(__name__)

_SENTIMENT_COL = "sentiment_score"

# ===========================================================================
# Rolling sentiment average (5, 10, 21-day windows)
# ===========================================================================

_ROLLING_WINDOWS = [5, 10, 21]


def _make_rolling_sentiment(window: int):
    """Factory for rolling sentiment average features."""

    def _compute(df: pd.DataFrame) -> pd.Series:
        if _SENTIMENT_COL not in df.columns:
            logger.debug("sentiment_col_missing", feature=f"sentiment_avg_{window}")
            return pd.Series(float("nan"), index=df.index)
        return df[_SENTIMENT_COL].rolling(window=window, min_periods=1).mean()

    _compute.__doc__ = f"Rolling {window}-day average sentiment score"
    return _compute


for _w in _ROLLING_WINDOWS:
    registry.register(
        name=f"sentiment_avg_{_w}",
        compute_fn=_make_rolling_sentiment(_w),
        group="sentiment",
        description=f"Rolling {_w}-day average sentiment score",
    )


# ===========================================================================
# Sentiment momentum (change in avg sentiment)
# ===========================================================================


@registry.feature(
    "sentiment_momentum",
    dependencies=["sentiment_avg_5", "sentiment_avg_21"],
    group="sentiment",
    description="Momentum: short-window avg minus long-window avg sentiment",
)
def sentiment_momentum(df: pd.DataFrame) -> pd.Series:
    """Difference between 5-day and 21-day rolling sentiment averages."""
    if "sentiment_avg_5" not in df.columns or "sentiment_avg_21" not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return df["sentiment_avg_5"] - df["sentiment_avg_21"]


# ===========================================================================
# Sentiment dispersion (std of sentiment scores)
# ===========================================================================


@registry.feature(
    "sentiment_dispersion",
    group="sentiment",
    description="Rolling 21-day standard deviation of sentiment scores",
)
def sentiment_dispersion(df: pd.DataFrame) -> pd.Series:
    """Rolling 21-day standard deviation of the raw sentiment score."""
    if _SENTIMENT_COL not in df.columns:
        logger.debug("sentiment_col_missing", feature="sentiment_dispersion")
        return pd.Series(float("nan"), index=df.index)
    return df[_SENTIMENT_COL].rolling(window=21, min_periods=2).std()
