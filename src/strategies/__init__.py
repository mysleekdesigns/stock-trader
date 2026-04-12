"""Trading strategies module.

Exports the core building blocks and the built-in momentum strategy.
"""

from src.strategies.base import BaseStrategyABC
from src.strategies.momentum import MomentumConfig, MomentumStrategy
from src.strategies.signal import SignalAggregator, SignalFilter

__all__ = [
    "BaseStrategyABC",
    "MomentumConfig",
    "MomentumStrategy",
    "SignalAggregator",
    "SignalFilter",
]
