"""WebSocket endpoint for real-time event streaming.

Clients connect to ``/api/ws`` and send JSON messages to subscribe or
unsubscribe from event types.  The server pushes matching events as they
arrive on the internal EventBus.

Protocol
--------
Client -> Server::

    {"action": "subscribe", "channel": "portfolio_update"}
    {"action": "unsubscribe", "channel": "signal"}

Server -> Client::

    {"channel": "portfolio_update", "data": {...}, "timestamp": "..."}
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.api.deps import get_event_bus
from src.core.events import Event, EventBus

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["websocket"])

# Supported event types that clients may subscribe to.
SUBSCRIBABLE_CHANNELS: set[str] = {
    "market_data",
    "signal",
    "order",
    "fill",
    "risk_breach",
    "portfolio_update",
}


def _serialise(obj: Any) -> Any:
    """Recursively convert dataclass / Decimal / datetime to JSON-safe types."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    return obj


# We need the Enum import for the helper above.
from enum import Enum  # noqa: E402


class _ClientConnection:
    """Tracks a single WebSocket client and its active subscriptions."""

    __slots__ = ("ws", "subscriptions", "_queue")

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.subscriptions: set[str] = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

    async def enqueue(self, message: dict[str, Any]) -> None:
        """Best-effort enqueue; drops the message if the buffer is full."""
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("ws.client_queue_full")

    async def drain(self) -> None:
        """Send queued messages to the client until the connection closes."""
        while True:
            message = await self._queue.get()
            await self.ws.send_json(message)


class _ConnectionManager:
    """Manages all active WebSocket connections and dispatches events."""

    def __init__(self) -> None:
        self._clients: list[_ClientConnection] = []

    async def connect(self, ws: WebSocket) -> _ClientConnection:
        await ws.accept()
        client = _ClientConnection(ws)
        self._clients.append(client)
        logger.info("ws.client_connected", client_count=len(self._clients))
        return client

    def disconnect(self, client: _ClientConnection) -> None:
        if client in self._clients:
            self._clients.remove(client)
        logger.info("ws.client_disconnected", client_count=len(self._clients))

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Push a message to all clients subscribed to *channel*."""
        message = {
            "channel": channel,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for client in list(self._clients):
            if channel in client.subscriptions:
                await client.enqueue(message)


_manager = _ConnectionManager()


def _make_handler(channel: str):
    """Create an EventBus handler that forwards events to WebSocket clients."""

    async def handler(event: Event) -> None:
        try:
            data = _serialise(event)
            await _manager.broadcast(channel, data if isinstance(data, dict) else {"event": data})
        except Exception as exc:
            logger.error("ws.broadcast_error", channel=channel, error=str(exc))

    handler.__qualname__ = f"ws_handler_{channel}"
    return handler


# Map channel name -> handler so we can subscribe/unsubscribe on the EventBus.
_handlers: dict[str, Any] = {ch: _make_handler(ch) for ch in SUBSCRIBABLE_CHANNELS}
_bus_registered = False


async def _ensure_bus_handlers(event_bus: EventBus) -> None:
    """Register EventBus handlers once (idempotent)."""
    global _bus_registered  # noqa: PLW0603
    if _bus_registered:
        return
    for channel, handler in _handlers.items():
        await event_bus.subscribe(channel, handler)
    _bus_registered = True
    logger.info("ws.bus_handlers_registered", channels=list(_handlers.keys()))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Real-time data streaming over WebSocket."""
    # Resolve the event bus manually since WebSocket endpoints don't use
    # Depends() the same way.
    from src.api.deps import get_event_bus as _get_bus

    event_bus = _get_bus()
    await _ensure_bus_handlers(event_bus)

    client = await _manager.connect(ws)

    # Start a background task to drain the send queue.
    drain_task = asyncio.create_task(client.drain())

    try:
        while True:
            raw = await ws.receive_json()
            action = raw.get("action")
            channel = raw.get("channel", "")

            if channel not in SUBSCRIBABLE_CHANNELS:
                await ws.send_json({"error": f"Unknown channel: {channel}"})
                continue

            if action == "subscribe":
                client.subscriptions.add(channel)
                await ws.send_json({"status": "subscribed", "channel": channel})
                logger.debug("ws.subscribe", channel=channel)

            elif action == "unsubscribe":
                client.subscriptions.discard(channel)
                await ws.send_json({"status": "unsubscribed", "channel": channel})
                logger.debug("ws.unsubscribe", channel=channel)

            else:
                await ws.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws.error", error=str(exc))
    finally:
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass
        _manager.disconnect(client)
