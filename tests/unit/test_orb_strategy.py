"""Unit tests for the Opening Range Breakout strategy."""

from datetime import datetime, time as dt_time

import pytest

from src.core.types import Bar, SignalDirection, TimeFrame
from src.strategies.opening_range_breakout import ORBConfig, ORBStrategy


def _make_bar(
    symbol: str,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=TimeFrame.MINUTE_1,
    )


class TestORBStrategy:
    """Test the ORB strategy logic."""

    def _build_session_bars(self) -> list[Bar]:
        """Build a realistic session with opening range then breakout."""
        symbol = "TEST"
        bars = []
        base = datetime(2024, 3, 15)

        # Opening range bars (9:30 - 10:00): 30 bars at 1-min each
        or_prices = [
            (100.0, 100.5, 99.8, 100.2),
            (100.2, 100.8, 100.0, 100.6),
            (100.6, 101.0, 100.3, 100.5),  # OR high = 101.0
        ]
        for i, (o, h, l, c) in enumerate(or_prices):
            ts = base.replace(hour=9, minute=30 + i)
            bars.append(_make_bar(symbol, ts, o, h, l, c, 50000))

        # Fill remaining opening range bars
        for i in range(3, 30):
            ts = base.replace(hour=9, minute=30 + i)
            bars.append(_make_bar(symbol, ts, 100.3, 100.7, 100.1, 100.4, 50000))

        return bars

    def test_opening_range_established(self):
        """OR high/low should be tracked during the opening range."""
        strategy = ORBStrategy("TEST")
        bars = self._build_session_bars()

        for bar in bars:
            strategy.on_bar(bar)

        state = strategy.state_snapshot
        assert state["or_high"] == 101.0
        assert state["or_bar_count"] == 30

    def test_no_signal_during_opening_range(self):
        """No signal should fire during the opening range itself."""
        strategy = ORBStrategy("TEST")
        bars = self._build_session_bars()

        for bar in bars:
            result = strategy.on_bar(bar)
            assert result is None

    def test_breakout_signal(self):
        """A breakout above OR high with volume + VWAP should produce a signal."""
        strategy = ORBStrategy("TEST", ORBConfig(volume_multiplier=1.5))
        bars = self._build_session_bars()

        # Process opening range
        for bar in bars:
            strategy.on_bar(bar)

        # First bar after OR to lock in the range
        ts_lock = datetime(2024, 3, 15, 10, 0)
        lock_bar = _make_bar("TEST", ts_lock, 100.5, 100.6, 100.4, 100.5, 40000)
        strategy.on_bar(lock_bar)

        assert strategy.state_snapshot["or_complete"] is True

        # Breakout bar: close > OR high (101.0), volume > 1.5x avg (50000), above VWAP
        ts_break = datetime(2024, 3, 15, 10, 5)
        breakout_bar = _make_bar("TEST", ts_break, 101.0, 101.8, 100.9, 101.5, 100000)
        signal = strategy.on_bar(breakout_bar)

        assert signal is not None
        assert signal.direction == SignalDirection.LONG
        assert signal.symbol == "TEST"
        assert signal.metadata["or_high"] == 101.0
        assert signal.metadata["volume_ratio"] >= 1.5

    def test_no_signal_below_or_high(self):
        """No signal when price is below the OR high."""
        strategy = ORBStrategy("TEST")
        bars = self._build_session_bars()

        for bar in bars:
            strategy.on_bar(bar)

        # Lock
        ts_lock = datetime(2024, 3, 15, 10, 0)
        strategy.on_bar(_make_bar("TEST", ts_lock, 100.5, 100.6, 100.4, 100.5, 40000))

        # Close below OR high
        ts = datetime(2024, 3, 15, 10, 5)
        result = strategy.on_bar(_make_bar("TEST", ts, 100.5, 100.8, 100.3, 100.7, 100000))
        assert result is None

    def test_no_signal_low_volume(self):
        """No signal when volume is below the threshold."""
        strategy = ORBStrategy("TEST", ORBConfig(volume_multiplier=1.5))
        bars = self._build_session_bars()

        for bar in bars:
            strategy.on_bar(bar)

        ts_lock = datetime(2024, 3, 15, 10, 0)
        strategy.on_bar(_make_bar("TEST", ts_lock, 100.5, 100.6, 100.4, 100.5, 40000))

        # Breakout price but low volume
        ts = datetime(2024, 3, 15, 10, 5)
        result = strategy.on_bar(_make_bar("TEST", ts, 101.0, 101.8, 100.9, 101.5, 30000))
        assert result is None

    def test_only_one_signal_per_day(self):
        """Only one breakout signal per session."""
        strategy = ORBStrategy("TEST", ORBConfig(volume_multiplier=1.5))
        bars = self._build_session_bars()

        for bar in bars:
            strategy.on_bar(bar)

        ts_lock = datetime(2024, 3, 15, 10, 0)
        strategy.on_bar(_make_bar("TEST", ts_lock, 100.5, 100.6, 100.4, 100.5, 40000))

        # First breakout
        ts1 = datetime(2024, 3, 15, 10, 5)
        signal1 = strategy.on_bar(_make_bar("TEST", ts1, 101.0, 101.8, 100.9, 101.5, 100000))
        assert signal1 is not None

        # Second breakout attempt — should not fire
        ts2 = datetime(2024, 3, 15, 10, 10)
        signal2 = strategy.on_bar(_make_bar("TEST", ts2, 101.5, 102.0, 101.3, 101.8, 120000))
        assert signal2 is None
        assert strategy.state_snapshot["breached"] is True

    def test_no_signal_after_cutoff(self):
        """No signal after the signal cutoff time."""
        strategy = ORBStrategy("TEST", ORBConfig(signal_cutoff=dt_time(11, 30)))
        bars = self._build_session_bars()

        for bar in bars:
            strategy.on_bar(bar)

        ts_lock = datetime(2024, 3, 15, 10, 0)
        strategy.on_bar(_make_bar("TEST", ts_lock, 100.5, 100.6, 100.4, 100.5, 40000))

        # Breakout after cutoff
        ts = datetime(2024, 3, 15, 11, 35)
        result = strategy.on_bar(_make_bar("TEST", ts, 101.0, 101.8, 100.9, 101.5, 100000))
        assert result is None

    def test_day_reset(self):
        """State should reset on a new trading day."""
        strategy = ORBStrategy("TEST")

        # Day 1 OR
        ts1 = datetime(2024, 3, 15, 9, 30)
        strategy.on_bar(_make_bar("TEST", ts1, 100.0, 101.0, 99.5, 100.5, 50000))

        # Day 2 — should reset
        ts2 = datetime(2024, 3, 18, 9, 30)
        strategy.on_bar(_make_bar("TEST", ts2, 110.0, 111.0, 109.5, 110.5, 60000))

        state = strategy.state_snapshot
        assert state["or_high"] == 111.0
        assert state["or_bar_count"] == 1

    def test_params_property(self):
        """The params property should expose the config."""
        config = ORBConfig(volume_multiplier=2.0, signal_cutoff=dt_time(12, 0))
        strategy = ORBStrategy("AAPL", config)

        params = strategy.params
        assert params["symbol"] == "AAPL"
        assert params["volume_multiplier"] == 2.0
        assert params["signal_cutoff"] == "12:00"

    def test_generate_signals_interface(self):
        """generate_signals should work with the dict-based feature interface."""
        strategy = ORBStrategy("TEST")

        features = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 50000,
        }
        ts = datetime(2024, 3, 15, 9, 30)
        signals = strategy.generate_signals(features, ts)
        assert isinstance(signals, list)
