"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings.

    Values are read from environment variables (case-insensitive) and from a
    `.env` file when present.
    """

    # ── Alpaca ──────────────────────────────────────────────────────────
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_BASE_URL",
    )

    # ── Polygon ─────────────────────────────────────────────────────────
    polygon_api_key: str = Field(default="", alias="POLYGON_API_KEY")

    # ── Database ────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/trading",
        alias="DATABASE_URL",
    )

    # ── Redis ───────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
