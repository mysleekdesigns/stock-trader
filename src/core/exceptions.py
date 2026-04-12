"""Exception hierarchy for the trading system.

Every domain-specific error inherits from ``TradingError`` so callers can
catch broad or narrow categories as needed.  All exceptions carry an
optional *details* dict for structured context.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class TradingError(Exception):
    """Root exception for all trading-system errors."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details: dict[str, Any] = details or {}
        super().__init__(message)


# Alias kept for backward compatibility with external tooling.
TradingBotError = TradingError


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

class DataError(TradingError):
    """Base class for data-layer errors."""


class DataProviderError(DataError):
    """An upstream data provider returned an error or timed out."""


class DataValidationError(DataError):
    """Incoming data failed validation checks."""


class DataNotFoundError(DataError):
    """Requested data does not exist."""


# ---------------------------------------------------------------------------
# Execution errors
# ---------------------------------------------------------------------------

class ExecutionError(TradingError):
    """Base class for order-execution errors."""


class OrderSubmissionError(ExecutionError):
    """Failed to submit an order to the broker."""


class OrderCancellationError(ExecutionError):
    """Failed to cancel an order."""


class InsufficientFundsError(ExecutionError):
    """Account lacks sufficient buying power."""


# Alias kept for backward compatibility with external tooling.
BrokerError = ExecutionError


# ---------------------------------------------------------------------------
# Risk errors
# ---------------------------------------------------------------------------

class RiskError(TradingError):
    """Base class for risk-management errors."""


class RiskLimitExceededError(RiskError):
    """A generic risk limit has been exceeded."""


class DrawdownLimitError(RiskError):
    """Portfolio drawdown exceeds the configured threshold."""


class ExposureLimitError(RiskError):
    """Position or sector exposure exceeds the configured threshold."""


# ---------------------------------------------------------------------------
# Strategy errors
# ---------------------------------------------------------------------------

class StrategyError(TradingError):
    """Base class for strategy errors."""


class SignalGenerationError(StrategyError):
    """Strategy failed to generate signals."""


class FeatureComputationError(StrategyError):
    """A required feature could not be computed."""


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------

class ModelError(TradingError):
    """Base class for ML model errors."""


class ModelTrainingError(ModelError):
    """Model training failed or diverged."""


class ModelPredictionError(ModelError):
    """Model prediction call failed."""


class ModelLoadError(ModelError):
    """Model could not be loaded from disk."""


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------

class ConfigError(TradingError):
    """Raised when configuration is invalid or missing."""
