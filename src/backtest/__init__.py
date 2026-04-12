"""Backtesting framework for event-driven strategy evaluation.

Provides a realistic simulation engine with fill modeling, slippage,
commission, and comprehensive performance analytics.
"""

from src.backtest.analytics import BacktestAnalytics
from src.backtest.data_handler import DataHandler
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.fill_simulator import FillResult, FillSimulator
from src.backtest.report import BacktestReport

__all__ = [
    "BacktestAnalytics",
    "BacktestEngine",
    "BacktestResult",
    "BacktestReport",
    "DataHandler",
    "FillResult",
    "FillSimulator",
]
