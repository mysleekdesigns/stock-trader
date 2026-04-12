"""Unit tests for feature computation (technical indicators and price features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.registry import FeatureRegistry
from src.features.technical import registry as global_registry

# Ensure price features are registered on the global registry.
import src.features.price  # noqa: F401


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    """Tests for the RSI indicator."""

    def test_rsi_bounds(self, sample_df):
        """RSI values must always lie in [0, 100]."""
        df = sample_df(n=300)
        result = global_registry.compute_subset(df, ["rsi_14"])
        rsi = result["rsi_14"].dropna()
        assert len(rsi) > 0, "RSI should produce non-NaN values"
        assert rsi.min() >= 0.0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100.0, f"RSI above 100: {rsi.max()}"

    def test_rsi_known_values(self):
        """Verify RSI on a simple synthetic series with known up/down moves."""
        # 14-period RSI: 14 ups of +1 should give RSI close to 100
        n = 30
        closes = [100.0]
        for i in range(1, n):
            # Alternate: 14 ups then 14 downs
            if i <= 14:
                closes.append(closes[-1] + 1.0)
            else:
                closes.append(closes[-1] - 0.5)

        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=pd.date_range("2023-01-01", periods=n, freq="D"),
        )
        result = global_registry.compute_subset(df, ["rsi_14"])
        rsi = result["rsi_14"].dropna()

        # After 14 consecutive ups, RSI should be very high (close to 100)
        assert rsi.iloc[0] > 90.0, f"After 14 ups, RSI should be near 100, got {rsi.iloc[0]}"


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    """Tests for the MACD indicator."""

    def test_macd_signal_crossover(self, sample_df):
        """Verify that MACD line and signal line produce crossover points."""
        df = sample_df(n=300)
        result = global_registry.compute_subset(df, ["macd"])

        macd_line = result["macd_line"].dropna()
        macd_signal = result["macd_signal"].dropna()

        assert len(macd_line) > 0
        assert len(macd_signal) > 0

        # Verify crossovers exist: sign changes in (macd_line - macd_signal)
        common_idx = macd_line.index.intersection(macd_signal.index)
        diff = macd_line.loc[common_idx] - macd_signal.loc[common_idx]
        sign_changes = (diff.shift(1) * diff) < 0
        crossover_count = sign_changes.sum()

        # With 300 bars of random walk data, we should see at least a few crossovers
        assert crossover_count >= 1, "Expected at least one MACD crossover"


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_bands_containment(self, sample_df):
        """Price should be between the bands most of the time."""
        df = sample_df(n=300)
        result = global_registry.compute_subset(df, ["bollinger"])

        upper = result["bb_upper"].dropna()
        lower = result["bb_lower"].dropna()
        close = result["close"].loc[upper.index]

        within_bands = ((close >= lower) & (close <= upper)).mean()

        # Bollinger Bands with 2 std devs should contain ~95% of data
        assert within_bands >= 0.80, (
            f"Expected at least 80% containment, got {within_bands:.2%}"
        )

    def test_bollinger_bands_ordering(self, sample_df):
        """Upper band must be >= middle >= lower band everywhere."""
        df = sample_df(n=300)
        result = global_registry.compute_subset(df, ["bollinger"])

        common = result[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        assert (common["bb_upper"] >= common["bb_middle"]).all()
        assert (common["bb_middle"] >= common["bb_lower"]).all()


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    """Tests for Average True Range."""

    def test_atr_positive(self, sample_df):
        """ATR must always be positive (after the initial warm-up period)."""
        df = sample_df(n=300)
        result = global_registry.compute_subset(df, ["atr_14"])
        atr = result["atr_14"].dropna()
        assert len(atr) > 0
        # ATR can be zero at the very start when high == low == close;
        # after the warm-up window it should be strictly positive.
        atr_after_warmup = atr.iloc[14:]
        assert (atr_after_warmup > 0).all(), "ATR should be positive after warm-up"


# ---------------------------------------------------------------------------
# Price features: log returns
# ---------------------------------------------------------------------------

class TestLogReturns:
    """Tests for log return features."""

    def test_log_returns_calculation(self):
        """Verify log returns match manual computation."""
        closes = [100.0, 105.0, 102.0, 108.0, 106.0]
        n = len(closes)
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=pd.date_range("2023-01-01", periods=n, freq="D"),
        )

        result = global_registry.compute_subset(df, ["log_return_1"])
        log_ret = result["log_return_1"]

        # Manual computation
        expected = np.log(np.array(closes[1:]) / np.array(closes[:-1]))

        actual = log_ret.dropna().values
        np.testing.assert_allclose(actual, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

class TestRealizedVolatility:
    """Tests for realized volatility features."""

    def test_realized_volatility(self, sample_df):
        """Verify annualization: vol_21 should be roughly sqrt(252) * daily std."""
        df = sample_df(n=300, seed=99)
        result = global_registry.compute_subset(df, ["realized_vol_21"])

        vol = result["realized_vol_21"].dropna()
        assert len(vol) > 0

        # Compute manually for comparison
        log_ret = np.log(df["close"] / df["close"].shift(1))
        manual_vol = log_ret.rolling(21, min_periods=11).std() * np.sqrt(252)
        manual_vol = manual_vol.dropna()

        common = vol.index.intersection(manual_vol.index)
        np.testing.assert_allclose(
            vol.loc[common].values,
            manual_vol.loc[common].values,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Feature registry: topological sort & circular dependency
# ---------------------------------------------------------------------------

class TestFeatureRegistry:
    """Tests for FeatureRegistry dependency resolution."""

    def test_feature_registry_toposort(self):
        """Verify that dependencies appear before dependents in the order."""
        reg = FeatureRegistry()
        reg.register("a", lambda df: df["close"], dependencies=[])
        reg.register("b", lambda df: df["close"], dependencies=["a"])
        reg.register("c", lambda df: df["close"], dependencies=["a", "b"])

        order = reg.resolve_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_feature_registry_circular_dependency(self):
        """Circular dependencies must raise ValueError."""
        reg = FeatureRegistry()
        reg.register("x", lambda df: df["close"], dependencies=["y"])
        reg.register("y", lambda df: df["close"], dependencies=["z"])
        reg.register("z", lambda df: df["close"], dependencies=["x"])

        with pytest.raises(ValueError, match="[Cc]ircular"):
            reg.resolve_order()


# ---------------------------------------------------------------------------
# Feature pipeline: fit_transform
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    """Tests for FeaturePipeline fit/transform workflow."""

    def test_feature_pipeline_fit_transform(self, sample_df):
        """Output should have the right shape and no NaN after head trimming."""
        from src.features.pipeline import FeaturePipeline

        df = sample_df(n=300)
        feature_names = ["rsi_14", "atr_14", "log_return_1"]

        pipeline = FeaturePipeline(
            registry=global_registry,
            feature_names=feature_names,
            normalize=True,
        )

        result = pipeline.fit_transform(df)

        # All requested features should be present
        for name in feature_names:
            assert name in result.columns, f"Feature {name} missing from output"

        # After head trimming there should be no NaN in the feature columns
        feature_data = result[feature_names]
        nan_count = feature_data.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after fit_transform"

        # Output should have fewer rows than input (head trimming)
        assert len(result) <= len(df)
        assert len(result) > 0

    def test_feature_pipeline_no_normalize(self, sample_df):
        """Without normalization, raw feature values should be returned."""
        from src.features.pipeline import FeaturePipeline

        df = sample_df(n=300)
        pipeline = FeaturePipeline(
            registry=global_registry,
            feature_names=["rsi_14"],
            normalize=False,
        )
        result = pipeline.fit_transform(df)
        rsi = result["rsi_14"].dropna()

        # Without normalization, RSI should still be in [0, 100]
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0
