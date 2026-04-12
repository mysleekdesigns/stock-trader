"""Redis-backed feature store for ML feature vectors.

Stores per-symbol, per-timestamp feature dictionaries as Redis hashes
with configurable TTL.  Also maintains a ``latest:{symbol}`` pointer
that is updated atomically so consumers can always fetch the most
recent feature set without scanning by timestamp.
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
import structlog

from src.core.exceptions import DataError

logger = structlog.get_logger(__name__)

_DEFAULT_TTL_SECONDS: int = 7 * 24 * 60 * 60  # 7 days


class FeatureStore:
    """Async Redis feature store with TTL and latest-pointer semantics."""

    def __init__(self, redis_url: str, default_ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._redis: aioredis.Redis | None = None
        self._log = logger.bind(component="FeatureStore")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        try:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
            )
            # Verify the connection is alive
            await self._redis.ping()
            self._log.info("redis_connected", url=self._redis_url)
        except Exception as exc:
            self._log.error("redis_connect_failed", error=str(exc))
            raise DataError(f"Failed to connect to Redis: {exc}") from exc

    async def disconnect(self) -> None:
        """Close the Redis connection pool gracefully."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self._log.info("redis_disconnected")

    @property
    def _r(self) -> aioredis.Redis:
        if self._redis is None:
            raise DataError("FeatureStore is not connected — call connect() first")
        return self._redis

    # ------------------------------------------------------------------
    # Feature CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_key(symbol: str, timestamp: str) -> str:
        return f"features:{symbol}:{timestamp}"

    @staticmethod
    def _latest_key(symbol: str) -> str:
        return f"latest:{symbol}"

    async def store_features(
        self,
        symbol: str,
        timestamp: str,
        features: dict[str, float],
    ) -> None:
        """Store a feature dictionary as a Redis hash.

        The key ``features:{symbol}:{timestamp}`` is set with the
        default TTL (7 days).
        """
        key = self._feature_key(symbol, timestamp)
        try:
            # Convert all values to strings for Redis hash storage
            str_features = {k: str(v) for k, v in features.items()}
            async with self._r.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping=str_features)  # type: ignore[arg-type]
                pipe.expire(key, self._default_ttl)
                await pipe.execute()
            self._log.debug("features_stored", symbol=symbol, timestamp=timestamp, count=len(features))
        except DataError:
            raise
        except Exception as exc:
            self._log.error("store_features_failed", symbol=symbol, error=str(exc))
            raise DataError(f"Failed to store features: {exc}") from exc

    async def get_features(
        self,
        symbol: str,
        timestamp: str,
    ) -> dict[str, float] | None:
        """Retrieve the feature hash for a given symbol and timestamp."""
        key = self._feature_key(symbol, timestamp)
        try:
            data: dict[str, str] = await self._r.hgetall(key)  # type: ignore[assignment]
            if not data:
                return None
            return {k: float(v) for k, v in data.items()}
        except DataError:
            raise
        except Exception as exc:
            self._log.error("get_features_failed", symbol=symbol, error=str(exc))
            raise DataError(f"Failed to get features: {exc}") from exc

    async def get_latest_features(self, symbol: str) -> dict[str, float] | None:
        """Return the most recently stored features for *symbol*."""
        key = self._latest_key(symbol)
        try:
            data: dict[str, str] = await self._r.hgetall(key)  # type: ignore[assignment]
            if not data:
                return None
            return {k: float(v) for k, v in data.items()}
        except DataError:
            raise
        except Exception as exc:
            self._log.error("get_latest_features_failed", symbol=symbol, error=str(exc))
            raise DataError(f"Failed to get latest features: {exc}") from exc

    async def store_latest_features(
        self,
        symbol: str,
        features: dict[str, float],
    ) -> None:
        """Atomically replace the ``latest:{symbol}`` hash (no TTL)."""
        key = self._latest_key(symbol)
        try:
            str_features = {k: str(v) for k, v in features.items()}
            async with self._r.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                pipe.hset(key, mapping=str_features)  # type: ignore[arg-type]
                await pipe.execute()
            self._log.debug("latest_features_stored", symbol=symbol, count=len(features))
        except DataError:
            raise
        except Exception as exc:
            self._log.error("store_latest_features_failed", symbol=symbol, error=str(exc))
            raise DataError(f"Failed to store latest features: {exc}") from exc

    async def delete_features(self, symbol: str, before: str) -> int:
        """Delete feature keys for *symbol* with timestamps earlier than *before*.

        This performs a SCAN over ``features:{symbol}:*`` and removes keys whose
        embedded timestamp string compares less than *before* (ISO-8601 / sortable
        format assumed).

        Returns the number of keys deleted.
        """
        pattern = f"features:{symbol}:*"
        prefix = f"features:{symbol}:"
        deleted = 0
        try:
            async for key in self._r.scan_iter(match=pattern, count=500):
                ts = key[len(prefix) :]
                if ts < before:
                    await self._r.delete(key)
                    deleted += 1
            self._log.info("features_deleted", symbol=symbol, before=before, deleted=deleted)
            return deleted
        except DataError:
            raise
        except Exception as exc:
            self._log.error("delete_features_failed", symbol=symbol, error=str(exc))
            raise DataError(f"Failed to delete features: {exc}") from exc
