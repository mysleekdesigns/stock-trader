"""Protocol definitions (interfaces) for pluggable system components.

Every concrete adapter/strategy/model must satisfy one of these protocols so
that the rest of the system can depend on abstractions rather than
implementations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from src.core.types import Bar, Order, Position, Quote, Signal, TimeFrame


@runtime_checkable
class DataProvider(Protocol):
    """Provides market data — historical bars, live quotes, and streaming."""

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Bar]: ...

    async def get_quote(self, symbol: str) -> Quote: ...

    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        callback: Callable,
    ) -> None: ...

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: Callable,
    ) -> None: ...


@runtime_checkable
class BasePredictor(Protocol):
    """ML model interface used by the prediction layer."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None: ...

    def predict(self, X: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def get_feature_importance(self) -> dict[str, float]: ...


@runtime_checkable
class BaseStrategy(Protocol):
    """Trading strategy that converts features into signals."""

    name: str

    def generate_signals(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]: ...

    def get_required_features(self) -> list[str]: ...


@runtime_checkable
class BrokerAdapter(Protocol):
    """Broker integration layer for order management and account data."""

    async def submit_order(self, order: Order) -> Order: ...

    async def cancel_order(self, order_id: str) -> bool: ...

    async def get_positions(self) -> list[Position]: ...

    async def get_account(self) -> dict[str, Any]: ...

    async def subscribe_order_updates(self, callback: Callable) -> None: ...
