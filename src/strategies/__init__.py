"""Trading strategies module.

Exports the core building blocks and built-in strategies.
"""

from src.strategies.base import BaseStrategyABC
from src.strategies.connors_rsi2 import ConnorsRSI2Config, ConnorsRSI2Strategy
from src.strategies.donchian_breakout import DonchianBreakoutStrategy, DonchianConfig
from src.strategies.dual_momentum import DualMomentumConfig, DualMomentumStrategy
from src.strategies.macd_trend import MACDTrendConfig, MACDTrendStrategy
from src.strategies.momentum import MomentumConfig, MomentumStrategy
from src.strategies.opening_range_breakout import ORBConfig, ORBStrategy
from src.strategies.signal import SignalAggregator, SignalFilter
from src.strategies.stateful import PositionStateStrategy
from src.strategies.supertrend import SupertrendConfig, SupertrendStrategy

__all__ = [
    "BaseStrategyABC",
    "PositionStateStrategy",
    "MomentumConfig",
    "MomentumStrategy",
    "ORBConfig",
    "ORBStrategy",
    "DonchianConfig",
    "DonchianBreakoutStrategy",
    "SupertrendConfig",
    "SupertrendStrategy",
    "ConnorsRSI2Config",
    "ConnorsRSI2Strategy",
    "MACDTrendConfig",
    "MACDTrendStrategy",
    "DualMomentumConfig",
    "DualMomentumStrategy",
    "SignalAggregator",
    "SignalFilter",
]
