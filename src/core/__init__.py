"""Core domain primitives for the trading system.

Re-exports key types, events, interfaces, and exceptions so that downstream
modules can do::

    from src.core import Signal, EventBus, BaseStrategy, TradingError
"""

from src.core.events import (
    Event,
    EventBus,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    PortfolioUpdateEvent,
    RiskBreachEvent,
    SignalEvent,
)
from src.core.exceptions import (
    ConfigError,
    DataError,
    DataNotFoundError,
    DataProviderError,
    DataValidationError,
    DrawdownLimitError,
    ExecutionError,
    ExposureLimitError,
    FeatureComputationError,
    InsufficientFundsError,
    ModelError,
    ModelLoadError,
    ModelPredictionError,
    ModelTrainingError,
    OrderCancellationError,
    OrderSubmissionError,
    RiskError,
    RiskLimitExceededError,
    SignalGenerationError,
    StrategyError,
    TradingError,
)
from src.core.interfaces import (
    BasePredictor,
    BaseStrategy,
    BrokerAdapter,
    DataProvider,
)
from src.core.types import (
    AssetClass,
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Signal,
    SignalDirection,
    TimeFrame,
)

__all__ = [
    # types
    "AssetClass",
    "Bar",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
    "Signal",
    "SignalDirection",
    "TimeFrame",
    # events
    "Event",
    "EventBus",
    "FillEvent",
    "MarketDataEvent",
    "OrderEvent",
    "PortfolioUpdateEvent",
    "RiskBreachEvent",
    "SignalEvent",
    # interfaces
    "BasePredictor",
    "BaseStrategy",
    "BrokerAdapter",
    "DataProvider",
    # exceptions
    "ConfigError",
    "DataError",
    "DataNotFoundError",
    "DataProviderError",
    "DataValidationError",
    "DrawdownLimitError",
    "ExecutionError",
    "ExposureLimitError",
    "FeatureComputationError",
    "InsufficientFundsError",
    "ModelError",
    "ModelLoadError",
    "ModelPredictionError",
    "ModelTrainingError",
    "OrderCancellationError",
    "OrderSubmissionError",
    "RiskError",
    "RiskLimitExceededError",
    "SignalGenerationError",
    "StrategyError",
    "TradingError",
]
