"""Sentiment data aggregation provider.

Provides a unified interface for retrieving news sentiment data.  The current
implementation returns synthetic data with the correct schema so that
downstream consumers (feature engineering, backtesting) can be developed and
tested before a live data source is connected.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_SENTIMENT_SCHEMA: dict[str, str] = {
    "timestamp": "datetime64[ns]",
    "symbol": "object",
    "headline": "object",
    "sentiment": "object",
    "score": "float64",
    "source": "object",
}


class SentimentDataProvider:
    """Aggregates news sentiment from one or more upstream sources.

    Currently uses a synthetic placeholder implementation that returns
    random-but-realistic sentiment data matching the production schema.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        logger.info("sentiment_data_provider_init")

    def get_news_sentiment(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return a DataFrame of sentiment scores for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).
        start, end:
            Date range (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``symbol``, ``headline``, ``sentiment``,
            ``score``, ``source``.
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        dates = pd.bdate_range(start=start_dt, end=end_dt, freq="B")
        if len(dates) == 0:
            return self._empty_frame()

        records: list[dict] = []
        sentiments = ["positive", "negative", "neutral"]
        sources = ["reuters", "bloomberg", "seekingalpha", "wsj"]

        for date in dates:
            n_articles = int(self._rng.integers(1, 6))
            for _ in range(n_articles):
                sent = self._rng.choice(sentiments, p=[0.35, 0.25, 0.40])
                score = self._generate_score(sent)
                records.append({
                    "timestamp": date + pd.Timedelta(
                        hours=int(self._rng.integers(6, 20)),
                        minutes=int(self._rng.integers(0, 60)),
                    ),
                    "symbol": symbol,
                    "headline": f"Synthetic headline for {symbol}",
                    "sentiment": sent,
                    "score": round(float(score), 4),
                    "source": str(self._rng.choice(sources)),
                })

        df = pd.DataFrame(records)
        for col, dtype in _SENTIMENT_SCHEMA.items():
            df[col] = df[col].astype(dtype)

        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "sentiment_data_retrieved",
            symbol=symbol,
            start=str(start_dt.date()),
            end=str(end_dt.date()),
            n_records=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_score(self, sentiment: str) -> float:
        """Generate a plausible sentiment score for the given label."""
        if sentiment == "positive":
            return float(self._rng.uniform(0.3, 1.0))
        elif sentiment == "negative":
            return float(self._rng.uniform(-1.0, -0.3))
        return float(self._rng.uniform(-0.2, 0.2))

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema."""
        df = pd.DataFrame(columns=list(_SENTIMENT_SCHEMA.keys()))
        for col, dtype in _SENTIMENT_SCHEMA.items():
            df[col] = df[col].astype(dtype)
        return df
