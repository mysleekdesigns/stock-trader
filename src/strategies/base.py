"""Abstract base class for all trading strategies.

Concrete strategies inherit from ``BaseStrategyABC`` and implement the
``generate_signals`` and ``get_required_features`` hooks.  The ABC provides
common plumbing: feature validation, enable/disable toggling, and composite
weighting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import structlog

from src.core.exceptions import FeatureComputationError
from src.core.types import Signal

logger = structlog.get_logger(__name__)


class BaseStrategyABC(ABC):
    """Base class that all strategies must inherit from.

    Parameters
    ----------
    name:
        Human-readable strategy identifier (must be unique within an engine).
    enabled:
        If ``False`` the engine should skip this strategy during signal
        generation.
    weight:
        Relative weight used by :class:`SignalAggregator` when combining
        signals from multiple strategies.  Defaults to 1.0.
    """

    def __init__(
        self,
        name: str,
        *,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        self._name = name
        self.enabled = enabled
        self.weight = weight

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique strategy name."""
        return self._name

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Produce zero or more trading signals from *features*."""
        ...

    @abstractmethod
    def get_required_features(self) -> list[str]:
        """Return the feature keys this strategy needs to operate."""
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate_features(self, features: dict[str, Any]) -> None:
        """Raise :class:`FeatureComputationError` if any required feature is
        missing or ``None``.
        """
        required = set(self.get_required_features())
        present = {k for k, v in features.items() if v is not None}
        missing = required - present

        if missing:
            logger.warning(
                "strategy.missing_features",
                strategy=self.name,
                missing=sorted(missing),
            )
            raise FeatureComputationError(
                f"Strategy '{self.name}' is missing required features: {sorted(missing)}",
                details={"strategy": self.name, "missing": sorted(missing)},
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"enabled={self.enabled} weight={self.weight}>"
        )
