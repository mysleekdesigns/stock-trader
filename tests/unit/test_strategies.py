"""Unit tests for trading strategies and signal aggregation/filtering."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.core.types import Signal, SignalDirection
from src.strategies.signal import SignalAggregator, SignalFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    direction: SignalDirection = SignalDirection.LONG,
    strength: float = 0.8,
    confidence: float = 0.7,
    strategy_name: str = "test",
    symbol: str = "SPY",
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        strategy_name=strategy_name,
    )


# ---------------------------------------------------------------------------
# MomentumStrategy tests
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    """Tests for the MomentumStrategy signal generation.

    If MomentumStrategy is not yet implemented, these tests are skipped.
    """

    @pytest.fixture(autouse=True)
    def _try_import(self):
        self._module = pytest.importorskip(
            "src.strategies.momentum",
            reason="MomentumStrategy not yet implemented",
        )
        self.MomentumStrategy = self._module.MomentumStrategy

    def _make_strategy(self, **config_kwargs):
        from src.strategies.momentum import MomentumConfig
        config = MomentumConfig(**config_kwargs) if config_kwargs else MomentumConfig()
        return self.MomentumStrategy(symbol="SPY", config=config)

    def _base_features(self, **overrides) -> dict:
        """Build a base feature dict matching MomentumStrategy's required keys."""
        features = {
            "ema_10": 155.0,
            "ema_50": 150.0,
            # The ``adx_14`` registry feature emits a column named ``adx``
            # (alongside adx_pos_di / adx_neg_di), which is the key the
            # strategy actually consumes.
            "adx": 30.0,
            "atr_14": 2.0,
            "volume_sma_20": 1_000_000,
            "sma_20": 152.0,
            "sma_200": 140.0,
            "rsi_14": 55.0,
            "close": 155.0,
            "volume": 2_000_000,  # relative_volume = 2.0
            "highest_high": None,
            "lowest_low": None,
        }
        features.update(overrides)
        return features

    def test_momentum_long_signal(self):
        """Uptrend with MA crossover should generate a LONG signal."""
        strategy = self._make_strategy()
        # First call: set prev state to fast < slow
        features_before = self._base_features(ema_10=145.0, ema_50=150.0)
        strategy.generate_signals(features_before, datetime.utcnow())

        # Second call: fast crosses above slow -> crossover LONG
        features_after = self._base_features(ema_10=155.0, ema_50=150.0)
        signals = strategy.generate_signals(features_after, datetime.utcnow())
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        assert len(long_signals) > 0, "Expected at least one LONG signal on crossover"

    def test_momentum_short_signal(self):
        """Downtrend crossover should generate a SHORT signal."""
        strategy = self._make_strategy()
        # First call: fast > slow
        features_before = self._base_features(ema_10=155.0, ema_50=150.0)
        strategy.generate_signals(features_before, datetime.utcnow())

        # Second call: fast drops below slow -> crossover SHORT
        features_after = self._base_features(
            ema_10=145.0, ema_50=150.0, close=145.0, rsi_14=40.0,
        )
        signals = strategy.generate_signals(features_after, datetime.utcnow())
        short_signals = [s for s in signals if s.direction == SignalDirection.SHORT]
        assert len(short_signals) > 0, "Expected at least one SHORT signal on crossover"

    def test_momentum_no_signal_low_adx(self):
        """Low ADX (no trend) should produce no directional signal."""
        strategy = self._make_strategy()
        # Set up crossover state
        features_before = self._base_features(ema_10=145.0, ema_50=150.0, adx=15.0)
        strategy.generate_signals(features_before, datetime.utcnow())

        # Crossover happens but ADX is too low
        features_after = self._base_features(ema_10=155.0, ema_50=150.0, adx=15.0)
        signals = strategy.generate_signals(features_after, datetime.utcnow())
        directional = [
            s for s in signals
            if s.direction in (SignalDirection.LONG, SignalDirection.SHORT)
        ]
        assert len(directional) == 0, (
            f"Expected no directional signals with low ADX, got {len(directional)}"
        )

    def test_momentum_volume_filter(self):
        """Low volume should filter out the signal."""
        strategy = self._make_strategy()
        # Set up crossover state
        features_before = self._base_features(
            ema_10=145.0, ema_50=150.0, volume=500_000,
        )
        strategy.generate_signals(features_before, datetime.utcnow())

        # Crossover happens but volume too low
        features_after = self._base_features(
            ema_10=155.0, ema_50=150.0, volume=500_000,  # ratio=0.5
        )
        signals = strategy.generate_signals(features_after, datetime.utcnow())
        directional = [
            s for s in signals
            if s.direction in (SignalDirection.LONG, SignalDirection.SHORT)
        ]
        assert len(directional) == 0, (
            f"Expected no signals with low volume, got {len(directional)}"
        )


# ---------------------------------------------------------------------------
# SignalAggregator: weighted average
# ---------------------------------------------------------------------------

class TestSignalAggregatorWeighted:
    """Tests for weighted-average signal aggregation."""

    def test_weighted_long_dominates(self):
        """When LONG signals outweigh SHORT, result should be LONG."""
        agg = SignalAggregator(symbol="SPY")
        agg.add_signal(_make_signal(SignalDirection.LONG, strength=0.9, confidence=0.8))
        agg.add_signal(_make_signal(SignalDirection.LONG, strength=0.7, confidence=0.6))
        agg.add_signal(_make_signal(SignalDirection.SHORT, strength=0.3, confidence=0.4))

        result = agg.aggregate(method="weighted_average")
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_weighted_short_dominates(self):
        """When SHORT signals outweigh LONG, result should be SHORT."""
        agg = SignalAggregator(symbol="SPY")
        agg.add_signal(_make_signal(SignalDirection.SHORT, strength=0.9, confidence=0.9))
        agg.add_signal(_make_signal(SignalDirection.SHORT, strength=0.8, confidence=0.7))
        agg.add_signal(_make_signal(SignalDirection.LONG, strength=0.2, confidence=0.3))

        result = agg.aggregate(method="weighted_average")
        assert result is not None
        assert result.direction == SignalDirection.SHORT

    def test_weighted_confidence_bounded(self):
        """Aggregated confidence and strength should be in [0, 1]."""
        agg = SignalAggregator(symbol="SPY")
        agg.add_signal(_make_signal(SignalDirection.LONG, strength=1.0, confidence=1.0))
        agg.add_signal(_make_signal(SignalDirection.LONG, strength=1.0, confidence=1.0))

        result = agg.aggregate(method="weighted_average")
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.strength <= 1.0

    def test_weighted_empty_returns_none(self):
        """Empty aggregator should return None."""
        agg = SignalAggregator(symbol="SPY")
        result = agg.aggregate(method="weighted_average")
        assert result is None


# ---------------------------------------------------------------------------
# SignalAggregator: majority vote
# ---------------------------------------------------------------------------

class TestSignalAggregatorMajorityVote:
    """Tests for majority-vote signal aggregation."""

    def test_majority_vote_long(self):
        """3 LONG vs 1 SHORT should yield LONG."""
        agg = SignalAggregator(symbol="SPY")
        agg.add_signal(_make_signal(SignalDirection.LONG, strategy_name="s1"))
        agg.add_signal(_make_signal(SignalDirection.LONG, strategy_name="s2"))
        agg.add_signal(_make_signal(SignalDirection.LONG, strategy_name="s3"))
        agg.add_signal(_make_signal(SignalDirection.SHORT, strategy_name="s4"))

        result = agg.aggregate(method="majority_vote")
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_majority_vote_short(self):
        """3 SHORT vs 1 LONG should yield SHORT."""
        agg = SignalAggregator(symbol="SPY")
        agg.add_signal(_make_signal(SignalDirection.SHORT, strategy_name="s1"))
        agg.add_signal(_make_signal(SignalDirection.SHORT, strategy_name="s2"))
        agg.add_signal(_make_signal(SignalDirection.SHORT, strategy_name="s3"))
        agg.add_signal(_make_signal(SignalDirection.LONG, strategy_name="s4"))

        result = agg.aggregate(method="majority_vote")
        assert result is not None
        assert result.direction == SignalDirection.SHORT

    def test_majority_vote_flat_ignored(self):
        """FLAT signals should not count as votes."""
        agg = SignalAggregator(symbol="SPY")
        agg.add_signal(_make_signal(SignalDirection.FLAT, strategy_name="s1"))
        agg.add_signal(_make_signal(SignalDirection.FLAT, strategy_name="s2"))
        agg.add_signal(_make_signal(SignalDirection.LONG, strategy_name="s3"))

        result = agg.aggregate(method="majority_vote")
        assert result is not None
        assert result.direction == SignalDirection.LONG


# ---------------------------------------------------------------------------
# SignalFilter: confidence gate
# ---------------------------------------------------------------------------

class TestSignalFilter:
    """Tests for signal filtering based on quality gates."""

    def test_filter_low_confidence(self):
        """Low-confidence signal should be filtered out."""
        filt = SignalFilter(min_confidence=0.5)
        signal = _make_signal(confidence=0.3)
        result = filt.filter(signal)
        assert result is None, "Signal with confidence 0.3 should be filtered"

    def test_filter_passes_high_confidence(self):
        """High-confidence signal should pass the filter."""
        filt = SignalFilter(min_confidence=0.5)
        signal = _make_signal(confidence=0.7)
        result = filt.filter(signal)
        assert result is not None, "Signal with confidence 0.7 should pass"

    def test_filter_low_strength(self):
        """Low-strength signal should be filtered out."""
        filt = SignalFilter(min_strength=0.5)
        signal = _make_signal(strength=0.2)
        result = filt.filter(signal)
        assert result is None, "Signal with strength 0.2 should be filtered"

    def test_filter_many(self):
        """filter_many should return only passing signals."""
        filt = SignalFilter(min_confidence=0.5, min_strength=0.3)
        signals = [
            _make_signal(confidence=0.8, strength=0.9),   # pass
            _make_signal(confidence=0.3, strength=0.9),   # fail confidence
            _make_signal(confidence=0.8, strength=0.1),   # fail strength
            _make_signal(confidence=0.6, strength=0.5),   # pass
        ]
        results = filt.filter_many(signals)
        assert len(results) == 2

    def test_signal_filter_confidence_threshold(self):
        """Signals right at the threshold boundary."""
        filt = SignalFilter(min_confidence=0.5)

        # Exactly at threshold: confidence < min_confidence fails
        at_threshold = _make_signal(confidence=0.5)
        result = filt.filter(at_threshold)
        # confidence=0.5 is not < 0.5, so it should pass
        assert result is not None, "Signal at exactly the threshold should pass"

        below = _make_signal(confidence=0.4999)
        result = filt.filter(below)
        assert result is None, "Signal just below threshold should be filtered"
