"""Custom ASGI middleware for the trading API.

- RequestLoggingMiddleware: structured request/response logging via structlog
- APIKeyAuthMiddleware: simple header-based API key gate
- RateLimitMiddleware: in-memory token-bucket rate limiter
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Paths that bypass auth and rate limiting.
_PUBLIC_PATHS: set[str] = {"/docs", "/redoc", "/openapi.json", "/api/health"}


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request's method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code if response else 500,
                duration_ms=round(duration_ms, 2),
                client=request.client.host if request.client else None,
            )


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests that lack a valid ``X-API-Key`` header.

    Parameters
    ----------
    api_key:
        Expected key value.  If empty or ``None`` the middleware is a no-op
        (all requests pass) which is useful for local development.
    """

    def __init__(self, app: Any, api_key: str | None = None) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth when no key is configured (dev mode) or for public paths.
        if not self._api_key:
            return await call_next(request)

        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/api/health"):
            return await call_next(request)

        # WebSocket upgrade requests carry the key as a query param.
        if request.scope.get("type") == "websocket":
            key = request.query_params.get("api_key", "")
        else:
            key = request.headers.get("X-API-Key", "")

        if key != self._api_key:
            logger.warning("auth.invalid_api_key", path=path)
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)

        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiting (in-memory token bucket)
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-client-IP token-bucket rate limiter.

    Parameters
    ----------
    max_tokens:
        Bucket capacity (burst size).
    refill_rate:
        Tokens added per second.
    """

    def __init__(
        self,
        app: Any,
        max_tokens: int = 60,
        refill_rate: float = 10.0,
    ) -> None:
        super().__init__(app)
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(max_tokens, refill_rate),
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/api/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        bucket = self._buckets[client_ip]

        if not bucket.consume():
            logger.warning("rate_limit.exceeded", client=client_ip, path=path)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
            )

        return await call_next(request)


class _TokenBucket:
    """Minimal token-bucket implementation."""

    __slots__ = ("_max_tokens", "_refill_rate", "_tokens", "_last_refill")

    def __init__(self, max_tokens: int, refill_rate: float) -> None:
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False
