"""Microstructure features derived from order-book and trade data.

Each feature operates on a :class:`~pandas.DataFrame` that is expected to
contain columns populated by the data feed layer (e.g. ``best_bid``,
``best_ask``, ``bid_volume_1`` .. ``bid_volume_N``, ``ask_volume_1`` ..
``ask_volume_N``, ``trade_price``, ``trade_volume``).

Features are registered with the project-wide :class:`FeatureRegistry` so
that the pipeline can resolve them automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.features.registry import FeatureRegistry

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level registry instance
# ---------------------------------------------------------------------------

microstructure_registry = FeatureRegistry()


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

@microstructure_registry.feature(
    name="bid_ask_spread",
    group="microstructure",
    description="Absolute spread between best ask and best bid.",
)
def bid_ask_spread(df: pd.DataFrame) -> pd.Series:
    """Compute absolute bid-ask spread.

    Expects columns ``best_ask`` and ``best_bid``.
    """
    return df["best_ask"] - df["best_bid"]


@microstructure_registry.feature(
    name="bid_ask_spread_bps",
    dependencies=["bid_ask_spread"],
    group="microstructure",
    description="Bid-ask spread in basis points relative to mid price.",
)
def bid_ask_spread_bps(df: pd.DataFrame) -> pd.Series:
    """Spread normalised by mid-price, expressed in basis points."""
    mid = (df["best_ask"] + df["best_bid"]) / 2.0
    return (df["bid_ask_spread"] / mid) * 10_000


@microstructure_registry.feature(
    name="order_book_imbalance",
    group="microstructure",
    description="Normalised order-book imbalance (bid_vol - ask_vol) / total.",
)
def order_book_imbalance(df: pd.DataFrame) -> pd.Series:
    """Compute order-book imbalance from aggregated bid/ask volume columns.

    Looks for columns matching ``bid_volume`` and ``ask_volume`` (or
    ``bid_volume_1`` .. ``bid_volume_N`` style for multi-level data).
    """
    bid_cols = [c for c in df.columns if c.startswith("bid_volume")]
    ask_cols = [c for c in df.columns if c.startswith("ask_volume")]

    if not bid_cols or not ask_cols:
        logger.warning(
            "microstructure.missing_volume_columns",
            bid_cols=bid_cols,
            ask_cols=ask_cols,
        )
        return pd.Series(0.0, index=df.index, name="order_book_imbalance")

    bid_vol = df[bid_cols].sum(axis=1)
    ask_vol = df[ask_cols].sum(axis=1)
    total = bid_vol + ask_vol

    return ((bid_vol - ask_vol) / total).fillna(0.0)


@microstructure_registry.feature(
    name="vpin",
    group="microstructure",
    description="Volume-synchronized Probability of Informed Trading.",
)
def vpin(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Estimate VPIN from trade-level buy/sell classification.

    Uses the tick rule: trades at or above mid-price are classified as buys.
    Expects columns ``trade_price``, ``trade_volume``, ``best_bid``,
    ``best_ask``.
    """
    mid = (df["best_ask"] + df["best_bid"]) / 2.0
    is_buy = df["trade_price"] >= mid

    buy_vol = df["trade_volume"].where(is_buy, 0.0)
    sell_vol = df["trade_volume"].where(~is_buy, 0.0)

    abs_diff = (buy_vol - sell_vol).abs().rolling(window, min_periods=1).sum()
    total_vol = df["trade_volume"].rolling(window, min_periods=1).sum()

    result = (abs_diff / total_vol).fillna(0.0)
    return result.clip(0.0, 1.0)


@microstructure_registry.feature(
    name="kyle_lambda",
    group="microstructure",
    description="Kyle's lambda: price impact coefficient estimate.",
)
def kyle_lambda(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Estimate Kyle's lambda (price impact per unit of signed order flow).

    Uses rolling OLS: delta_price = lambda * signed_volume + epsilon.

    Expects columns ``trade_price``, ``trade_volume``, ``best_bid``,
    ``best_ask``.
    """
    mid = (df["best_ask"] + df["best_bid"]) / 2.0
    delta_price = mid.diff()

    # Sign volume by tick rule
    is_buy = df["trade_price"] >= mid
    signed_volume = df["trade_volume"].where(is_buy, -df["trade_volume"])

    # Rolling OLS: lambda = cov(dp, sv) / var(sv)
    cov_dp_sv = delta_price.rolling(window, min_periods=max(window // 2, 2)).cov(
        signed_volume
    )
    var_sv = signed_volume.rolling(window, min_periods=max(window // 2, 2)).var()

    result = (cov_dp_sv / var_sv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# Bulk registration helper
# ---------------------------------------------------------------------------

def register_microstructure_features(registry: FeatureRegistry) -> None:
    """Copy all microstructure features into a target registry.

    This allows the main application registry to pick up microstructure
    features without coupling to the module-level registry instance.
    """
    for name in microstructure_registry.feature_names:
        defn = microstructure_registry[name]
        registry.register(
            name=defn.name,
            compute_fn=defn.compute_fn,
            dependencies=defn.dependencies,
            description=defn.description,
            group=defn.group,
        )
    logger.info(
        "microstructure_features_registered",
        count=len(microstructure_registry),
    )
