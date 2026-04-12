"""Redis caching layer with orjson serialisation and a ``@cached`` decorator.

Provides a thin async wrapper around Redis suitable for caching expensive
computations such as API responses, computed indicators, and portfolio
summaries.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any, Callable, TypeVar

import orjson
import redis.asyncio as aioredis
import structlog

from src.core.exceptions import DataError

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CacheStore:
    """Async Redis cache with orjson serialisation."""

    def __init__(self, redis_url: str, default_ttl: int = 3600) -> None:
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._redis: aioredis.Redis | None = None
        self._log = logger.bind(component="CacheStore")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        try:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=False,  # orjson produces bytes
            )
            await self._redis.ping()
            self._log.info("cache_connected", url=self._redis_url)
        except Exception as exc:
            self._log.error("cache_connect_failed", error=str(exc))
            raise DataError(f"Cache connection failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Shut down the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self._log.info("cache_disconnected")

    @property
    def _r(self) -> aioredis.Redis:
        if self._redis is None:
            raise DataError("CacheStore is not connected — call connect() first")
        return self._redis

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Retrieve and deserialise a cached value, or ``None`` if missing."""
        try:
            raw: bytes | None = await self._r.get(key)
            if raw is None:
                return None
            return orjson.loads(raw)
        except DataError:
            raise
        except Exception as exc:
            self._log.error("cache_get_failed", key=key, error=str(exc))
            raise DataError(f"Cache get failed for key {key!r}: {exc}") from exc

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Serialise and store *value* with an optional TTL override."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            raw = orjson.dumps(value)
            await self._r.set(key, raw, ex=effective_ttl)
        except DataError:
            raise
        except Exception as exc:
            self._log.error("cache_set_failed", key=key, error=str(exc))
            raise DataError(f"Cache set failed for key {key!r}: {exc}") from exc

    async def delete(self, key: str) -> None:
        """Remove a single key from the cache."""
        try:
            await self._r.delete(key)
        except DataError:
            raise
        except Exception as exc:
            self._log.error("cache_delete_failed", key=key, error=str(exc))
            raise DataError(f"Cache delete failed for key {key!r}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* is present in the cache."""
        try:
            return bool(await self._r.exists(key))
        except DataError:
            raise
        except Exception as exc:
            self._log.error("cache_exists_failed", key=key, error=str(exc))
            raise DataError(f"Cache exists check failed for key {key!r}: {exc}") from exc

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching *pattern* (e.g. ``bars:AAPL:*``).

        Returns the number of keys removed.
        """
        deleted = 0
        try:
            async for key in self._r.scan_iter(match=pattern, count=500):
                await self._r.delete(key)
                deleted += 1
            self._log.info("cache_pattern_cleared", pattern=pattern, deleted=deleted)
            return deleted
        except DataError:
            raise
        except Exception as exc:
            self._log.error("cache_clear_pattern_failed", pattern=pattern, error=str(exc))
            raise DataError(f"Cache clear_pattern failed for {pattern!r}: {exc}") from exc

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def cached(self, ttl: int = 3600) -> Callable[[F], F]:
        """Decorator that caches the return value of an async function.

        The cache key is derived from the function's qualified name and a
        hash of its positional + keyword arguments serialised with orjson.
        """
        store = self

        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Build a deterministic cache key
                key_source = orjson.dumps({"a": args, "k": kwargs})
                key_hash = hashlib.sha256(key_source).hexdigest()[:16]
                cache_key = f"cached:{fn.__qualname__}:{key_hash}"

                # Try cache first
                hit = await store.get(cache_key)
                if hit is not None:
                    return hit

                # Compute and cache
                result = await fn(*args, **kwargs)
                await store.set(cache_key, result, ttl=ttl)
                return result

            return wrapper  # type: ignore[return-value]

        return decorator  # type: ignore[return-value]
