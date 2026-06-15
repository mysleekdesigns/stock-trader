"""Unit tests for the rule-based strategies and their shared scaffolding.

Covers the incremental indicator helpers (Wilder RSI, EMA, clamp), the
position-state transition FSM (which prevents the engine from pyramiding), and
the entry/exit behaviour of each strategy on deterministic synthetic series.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.core.types import SignalDirection
from src.strategies.connors_rsi2 import ConnorsRSI2Config, ConnorsRSI2Strategy
from src.strategies.donchian_breakout import DonchianBreakoutStrategy, DonchianConfig
from src.strategies.dual_momentum import DualMomentumConfig, DualMomentumStrategy
from src.strategies.macd_trend import MACDTrendConfig, MACDTrendStrategy
from src.strategies.stateful import PositionStateStrategy, _Ema, clamp01, wilder_rsi
from src.strategies.supertrend import SupertrendConfig, SupertrendStrategy

_T0 = datetime(2022, 1, 1)


def _ts(i: int) -> datetime:
    return _T0 + timedelta(days=i)


def _feat(close: float, *, high=None, low=None, atr=None, symbol="TEST", i=0) -> dict:
    """Build a features dict shaped like the engine's per-bar payload."""
    high = close if high is None else high
    low = close if low is None else low
    atr = (0.02 * close) if atr is None else atr
    return {
        "symbol": symbol,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1_000_000,
        "atr_14": atr,
        "timestamp": _ts(i),
    }


def _run(strategy, closes, *, highs=None, lows=None, atr=None):
    """Feed a close series bar-by-bar; return the flat list of emitted signals."""
    out = []
    for i, c in enumerate(closes):
        h = highs[i] if highs else c
        low = lows[i] if lows else c
        f = _feat(c, high=h, low=low, atr=atr, i=i)
        out.extend(strategy.generate_signals(f, _ts(i)))
    return out


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

class TestIndicatorHelpers:
    def test_clamp01(self):
        assert clamp01(-1.0) == 0.0
        assert clamp01(2.0) == 1.0
        assert clamp01(0.5) == 0.5
        assert clamp01(float("nan")) == 0.0

    def test_wilder_rsi_all_gains_is_100(self):
        assert wilder_rsi([10, 11, 12, 13], 2) == 100.0

    def test_wilder_rsi_all_losses_is_0(self):
        assert wilder_rsi([13, 12, 11, 10], 2) == 0.0

    def test_wilder_rsi_insufficient_data_is_none(self):
        # RSI(2) needs 2 deltas, i.e. period+1 = 3 prices.
        assert wilder_rsi([10, 11], 2) is None
        assert wilder_rsi([10, 11, 12], 2) is not None

    def test_wilder_rsi_midrange(self):
        # Alternating up/down keeps RSI near the middle of its range.
        rsi = wilder_rsi([10, 11, 10, 11, 10, 11], 2)
        assert 20.0 < rsi < 80.0

    def test_ema_period_one_tracks_latest(self):
        ema = _Ema(1)  # alpha = 1.0 -> always equals the newest sample
        assert ema.update(5.0) == 5.0
        assert ema.update(10.0) == 10.0

    def test_ema_seeds_then_smooths(self):
        ema = _Ema(10)
        assert ema.update(100.0) == 100.0  # first sample seeds
        v = ema.update(110.0)
        assert 100.0 < v < 110.0  # moves partway toward the new sample


# ---------------------------------------------------------------------------
# Position-state transition FSM
# ---------------------------------------------------------------------------

class TestTransitionFSM:
    def _strat(self):
        # Any concrete subclass exposes the base _transition; use dual momentum.
        return DualMomentumStrategy("TEST")

    def test_no_signal_when_state_unchanged(self):
        s = self._strat()
        assert s._transition(SignalDirection.FLAT, timestamp=_T0) == []

    def test_open_from_flat_emits_single_signal(self):
        s = self._strat()
        sigs = s._transition(SignalDirection.LONG, timestamp=_T0, strength=0.8)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.LONG
        assert s.state == SignalDirection.LONG

    def test_hold_emits_nothing(self):
        s = self._strat()
        s._transition(SignalDirection.LONG, timestamp=_T0)
        assert s._transition(SignalDirection.LONG, timestamp=_T0) == []

    def test_close_to_flat(self):
        s = self._strat()
        s._transition(SignalDirection.LONG, timestamp=_T0)
        sigs = s._transition(SignalDirection.FLAT, timestamp=_T0)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.FLAT
        assert s.state == SignalDirection.FLAT

    def test_reversal_closes_then_opens_in_order(self):
        s = self._strat()
        s._transition(SignalDirection.LONG, timestamp=_T0)
        sigs = s._transition(SignalDirection.SHORT, timestamp=_T0, strength=0.7)
        assert [x.direction for x in sigs] == [SignalDirection.FLAT, SignalDirection.SHORT]
        assert s.state == SignalDirection.SHORT


# ---------------------------------------------------------------------------
# Donchian / Turtle breakout
# ---------------------------------------------------------------------------

class TestDonchianBreakout:
    def test_no_signal_during_channel_warmup(self):
        s = DonchianBreakoutStrategy("TEST", DonchianConfig(entry_period=20, exit_period=10))
        # First 20 bars only fill the channel.
        sigs = _run(s, [100.0] * 20, highs=[101.0] * 20, lows=[99.0] * 20)
        assert sigs == []

    def test_breakout_above_channel_goes_long(self):
        s = DonchianBreakoutStrategy("TEST", DonchianConfig(entry_period=20, exit_period=10))
        closes = [100.0] * 20 + [110.0]
        highs = [101.0] * 20 + [110.0]
        lows = [99.0] * 20 + [109.0]
        sigs = _run(s, closes, highs=highs, lows=lows)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.LONG

    def test_long_exits_on_exit_channel_low(self):
        # allow_short=False so a downside break exits to flat rather than flipping.
        cfg = DonchianConfig(entry_period=20, exit_period=10, atr_stop_mult=99.0, allow_short=False)
        s = DonchianBreakoutStrategy("TEST", cfg)
        # Fill, break out long, hold a few bars, then crash below the exit low.
        closes = [100.0] * 20 + [110.0, 111.0, 112.0, 80.0]
        highs = [101.0] * 20 + [110.0, 111.0, 112.0, 81.0]
        lows = [99.0] * 20 + [109.0, 110.0, 111.0, 79.0]
        sigs = _run(s, closes, highs=highs, lows=lows)
        dirs = [x.direction for x in sigs]
        assert SignalDirection.LONG in dirs
        assert dirs[-1] == SignalDirection.FLAT  # exited

    def test_long_only_when_short_disabled(self):
        cfg = DonchianConfig(entry_period=20, exit_period=10, allow_short=False)
        s = DonchianBreakoutStrategy("TEST", cfg)
        # A downside breakout must NOT produce a SHORT when shorting is off.
        closes = [100.0] * 20 + [80.0]
        highs = [101.0] * 20 + [81.0]
        lows = [99.0] * 20 + [79.0]
        sigs = _run(s, closes, highs=highs, lows=lows)
        assert all(x.direction != SignalDirection.SHORT for x in sigs)


# ---------------------------------------------------------------------------
# Anti-pyramiding: a steady trend must not re-emit entries every bar
# ---------------------------------------------------------------------------

class TestNoPyramiding:
    def test_supertrend_holds_without_re_entering(self):
        s = SupertrendStrategy("TEST", SupertrendConfig(allow_short=False))
        # Smooth, strong uptrend: trend should flip up once and then hold.
        closes = [100.0 * (1.01 ** i) for i in range(80)]
        highs = [c * 1.005 for c in closes]
        lows = [c * 0.995 for c in closes]
        sigs = _run(s, closes, highs=highs, lows=lows)
        opens = [x for x in sigs if x.metadata.get("action") == "open"]
        # At most a couple of opens across 80 trending bars (no per-bar pyramiding).
        assert len(opens) <= 2
        assert all(x.direction == SignalDirection.LONG for x in sigs)


# ---------------------------------------------------------------------------
# Connors RSI-2 (long/flat mean reversion)
# ---------------------------------------------------------------------------

class TestConnorsRSI2:
    def test_buys_oversold_dip_in_uptrend(self):
        cfg = ConnorsRSI2Config(rsi_buy=10.0, trend_sma=200, exit_sma=5)
        s = ConnorsRSI2Strategy("TEST", cfg)
        # 200-bar uptrend (price stays above its 200-SMA), then a sharp 3-bar
        # dip to drive RSI(2) below the buy threshold.
        closes = [100.0 + i for i in range(200)]  # 100 -> 299
        closes += [295.0, 290.0, 285.0]  # dip but still > 200-SMA (~200)
        sigs = _run(s, closes)
        assert any(x.direction == SignalDirection.LONG for x in sigs)
        assert all(x.direction != SignalDirection.SHORT for x in sigs)


# ---------------------------------------------------------------------------
# Dual / absolute momentum (long/flat)
# ---------------------------------------------------------------------------

class TestDualMomentum:
    def test_invests_in_uptrend_then_exits_in_downtrend(self):
        cfg = DualMomentumConfig(lookback=50, trend_sma=100)
        s = DualMomentumStrategy("TEST", cfg)
        up = [100.0 + i for i in range(160)]        # positive momentum, above SMA
        down = [260.0 - 3.0 * i for i in range(60)]  # roll over -> exit
        sigs = _run(s, up + down)
        dirs = [x.direction for x in sigs]
        assert SignalDirection.LONG in dirs
        assert SignalDirection.FLAT in dirs
        assert SignalDirection.SHORT not in dirs


# ---------------------------------------------------------------------------
# MACD trend
# ---------------------------------------------------------------------------

class TestMACDTrend:
    def test_goes_long_in_uptrend(self):
        cfg = MACDTrendConfig(use_trend_filter=False, allow_short=False)
        s = MACDTrendStrategy("TEST", cfg)
        # Down then up so a bullish MACD crossover occurs mid-series.
        closes = [100.0 - i for i in range(40)] + [60.0 + 2.0 * i for i in range(60)]
        sigs = _run(s, closes)
        assert any(x.direction == SignalDirection.LONG for x in sigs)
        assert all(x.direction != SignalDirection.SHORT for x in sigs)
