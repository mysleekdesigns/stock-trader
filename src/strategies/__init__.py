"""Trading strategies module.

Exports the core building blocks and built-in strategies.
"""

from src.strategies.base import BaseStrategyABC
from src.strategies.momentum import MomentumConfig, MomentumStrategy
from src.strategies.opening_range_breakout import ORBConfig, ORBStrategy
from src.strategies.signal import SignalAggregator, SignalFilter

__all__ = [
    "BaseStrategyABC",
    "MomentumConfig",
    "MomentumStrategy",
    "ORBConfig",
    "ORBStrategy",
    "SignalAggregator",
    "SignalFilter",
]
