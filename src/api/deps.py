"""FastAPI dependency injection providers.

All dependencies follow the ``Depends()`` pattern and are designed for use
with FastAPI's dependency injection system.
"""

from __future__ import annotations

from typing import AsyncGenerator

import structlog
from fastapi import Depends, Request

from src.core.config import Settings
from src.core.config import get_settings as _get_settings
from src.core.events import EventBus
from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton holders (populated during app lifespan)
# ---------------------------------------------------------------------------

_event_bus: EventBus | None = None
_portfolio: Portfolio | None = None
_redis_client: object | None = None  # redis.asyncio.Redis at runtime


def set_event_bus(bus: EventBus) -> None:
    """Called at startup to register the global EventBus."""
    global _event_bus  # noqa: PLW0603
    _event_bus = bus


def set_portfolio(portfolio: Portfolio) -> None:
    """Called at startup to register the global Portfolio."""
    global _portfolio  # noqa: PLW0603
    _portfolio = portfolio


def set_redis(client: object) -> None:
    """Called at startup to register the global Redis client."""
    global _redis_client  # noqa: PLW0603
    _redis_client = client


# ---------------------------------------------------------------------------
# Dependency callables
# ---------------------------------------------------------------------------


def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return _get_settings()


def get_event_bus() -> EventBus:
    """Return the application-wide EventBus instance."""
    if _event_bus is None:
        raise RuntimeError("EventBus has not been initialised. Is the app lifespan running?")
    return _event_bus


def get_portfolio() -> Portfolio:
    """Return the live Portfolio singleton."""
    if _portfolio is None:
        raise RuntimeError("Portfolio has not been initialised. Is the app lifespan running?")
    return _portfolio


def get_redis() -> object:
    """Return the async Redis client.

    The return type is ``object`` to avoid a hard import-time dependency on
    ``redis.asyncio`` when Redis is not available (e.g. during tests).
    """
    if _redis_client is None:
        raise RuntimeError("Redis client has not been initialised. Is the app lifespan running?")
    return _redis_client


async def get_db_session() -> AsyncGenerator:
    """Yield an ``AsyncSession`` from the engine pool.

    The session is lazily imported so the module loads even without a running
    database.  The actual ``async_sessionmaker`` is expected to be stored on
    ``app.state.db_session_factory`` by the lifespan handler.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    # Fallback: create a minimal in-memory session factory when no DB is
    # configured.  A real deployment would wire this up during lifespan.
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        settings = get_settings()
        engine = create_async_engine(settings.database_url, echo=False)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    except Exception:
        logger.warning("deps.db_session_unavailable")
        yield None  # type: ignore[misc]
        return

    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()
