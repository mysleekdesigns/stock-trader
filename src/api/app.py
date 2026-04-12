"""FastAPI application factory.

``create_app()`` builds and returns a fully configured FastAPI instance with
routers, middleware, lifespan handlers, and optional static file serving.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from src.api import deps
from src.api.middleware import (
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from src.core.config import get_settings
from src.core.events import EventBus
from src.risk.portfolio import Portfolio

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Alpaca account sync
# ---------------------------------------------------------------------------

async def _sync_portfolio_from_alpaca(settings: Any) -> Portfolio:
    """Pull account balance and positions from Alpaca to initialise the Portfolio.

    Falls back to a default cash-only portfolio if Alpaca is unreachable.
    """
    fallback_cash = Decimal(os.environ.get("INITIAL_CASH", "100000"))

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.warning("app.alpaca_sync_skipped", reason="no credentials")
        return Portfolio(cash=fallback_cash, initial_value=fallback_cash)

    try:
        from src.execution.brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            base_url=settings.alpaca_base_url,
        )
        await broker.connect()

        account = await broker.get_account()
        positions = await broker.get_positions()
        await broker.disconnect()

        cash = Decimal(account.get("cash", str(fallback_cash)))
        equity = Decimal(account.get("equity", str(cash)))

        portfolio = Portfolio(cash=cash, initial_value=equity)

        for pos in positions:
            portfolio.add_position(pos)

        logger.info(
            "app.alpaca_sync_complete",
            cash=str(cash),
            equity=str(equity),
            positions=len(positions),
        )
        return portfolio

    except Exception as exc:
        logger.warning("app.alpaca_sync_failed", error=str(exc))
        return Portfolio(cash=fallback_cash, initial_value=fallback_cash)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle hook.

    Initialises shared singletons (EventBus, Portfolio, Redis) and tears them
    down on shutdown.
    """
    settings = get_settings()
    logger.info("app.startup", database_url=settings.database_url[:30] + "...")

    # -- EventBus --
    event_bus = EventBus()
    deps.set_event_bus(event_bus)

    # -- Portfolio (sync with Alpaca account) --
    portfolio = await _sync_portfolio_from_alpaca(settings)
    deps.set_portfolio(portfolio)

    # -- Redis (best-effort) --
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        await redis_client.ping()
        deps.set_redis(redis_client)
        logger.info("app.redis_connected", url=settings.redis_url)
    except Exception as exc:
        logger.warning("app.redis_unavailable", error=str(exc))
        redis_client = None

    logger.info("app.startup_complete")

    yield  # ---- application runs ----

    # Shutdown
    if redis_client is not None:
        await redis_client.aclose()  # type: ignore[union-attr]
        logger.info("app.redis_disconnected")

    logger.info("app.shutdown_complete")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Stock Trader API",
        description="Algorithmic trading bot dashboard API",
        version="0.1.0",
        lifespan=_lifespan,
        default_response_class=ORJSONResponse,
    )

    # -- CORS --
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Custom middleware (outermost first) --
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_tokens=120, refill_rate=20.0)

    api_key = os.environ.get("API_KEY", "")
    app.add_middleware(APIKeyAuthMiddleware, api_key=api_key or None)

    # -- Routers --
    from src.api.routers import (
        backtest,
        dashboard,
        market_data,
        models,
        orders,
        risk,
        strategies,
        websocket,
    )

    app.include_router(dashboard.router)
    app.include_router(market_data.router)
    app.include_router(strategies.router)
    app.include_router(orders.router)
    app.include_router(backtest.router)
    app.include_router(models.router)
    app.include_router(risk.router)
    app.include_router(websocket.router)

    # -- Health check --
    @app.get("/api/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    # -- Static files (serve frontend build if available) --
    frontend_dist = Path("frontend/dist")
    if frontend_dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
        logger.info("app.static_files_mounted", path=str(frontend_dist))

    return app
