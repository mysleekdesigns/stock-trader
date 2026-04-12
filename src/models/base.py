"""Abstract base class for all ML predictors in the trading system.

Provides validation utilities, fitted-state tracking, and a consistent
interface that satisfies the ``BasePredictor`` protocol defined in
``src.core.interfaces``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.core.exceptions import ModelError, ModelPredictionError

logger = structlog.get_logger(__name__)


class BaseModelPredictor(ABC):
    """Abstract base implementing the :class:`BasePredictor` protocol.

    Subclasses must implement the six abstract methods.  This base supplies
    common bookkeeping (fitted-state flag, feature-name tracking) and two
    concrete helper methods shared by all models.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.is_fitted: bool = False
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Abstract interface  (mirrors ``BasePredictor`` protocol)
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Any, **kwargs: Any) -> None:
        """Train the model on feature matrix *X* and target(s) *y*."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Return point predictions for *X*."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class-probability estimates for *X*."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefacts to *path*."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model artefacts from *path*."""

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return a mapping of feature-name -> importance score."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the input feature matrix.

        Raises :class:`ModelPredictionError` when *X* has an unexpected shape
        or contains NaN/Inf values that cannot be safely handled.
        """
        if not isinstance(X, pd.DataFrame):
            raise ModelPredictionError(
                "Input must be a pandas DataFrame",
                details={"received_type": type(X).__name__},
            )

        if X.empty:
            raise ModelPredictionError(
                "Input DataFrame is empty",
                details={"shape": X.shape},
            )

        # Check for infinite values — always an error.
        inf_mask = np.isinf(X.select_dtypes(include=[np.number]).values)
        if inf_mask.any():
            inf_cols = X.columns[inf_mask.any(axis=0)].tolist()
            raise ModelPredictionError(
                "Input contains infinite values",
                details={"columns": inf_cols},
            )

        # Warn about NaN values but do not raise — tree models handle NaN.
        nan_count = int(X.isna().sum().sum())
        if nan_count > 0:
            nan_pct = nan_count / (X.shape[0] * X.shape[1]) * 100
            logger.warning(
                "input_contains_nan",
                nan_count=nan_count,
                nan_pct=round(nan_pct, 2),
                model=self.name,
            )

        # Verify feature alignment when model is already fitted.
        if self.is_fitted and self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ModelPredictionError(
                    "Input is missing required features",
                    details={"missing_features": sorted(missing)},
                )
            # Reorder to training order and drop extras.
            X = X[self.feature_names]

        return X

    def _check_fitted(self) -> None:
        """Raise :class:`ModelError` if the model has not been fitted yet."""
        if not self.is_fitted:
            raise ModelError(
                f"Model '{self.name}' has not been fitted. Call fit() first.",
                details={"model": self.name},
            )
