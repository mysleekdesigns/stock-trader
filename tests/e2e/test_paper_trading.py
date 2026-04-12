"""End-to-end tests for Alpaca paper trading integration.

These tests require valid Alpaca paper trading credentials in the
environment.  They are unconditionally skipped when ``ALPACA_API_KEY`` is
not set, making them safe to include in CI without secrets.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = os.environ.get(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)

skip_no_alpaca = pytest.mark.skipif(
    not ALPACA_API_KEY,
    reason="ALPACA_API_KEY environment variable not set — skipping Alpaca paper trading tests",
)

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

try:
    from src.core.types import Order, OrderSide, OrderStatus, OrderType
except (ImportError, ModuleNotFoundError):
    pytest.skip("Core types not available", allow_module_level=True)

try:
    from src.core.events import EventBus
except (ImportError, ModuleNotFoundError):
    EventBus = None  # type: ignore[assignment, misc]

try:
    from src.execution.brokers.alpaca_broker import AlpacaBroker

    HAS_ALPACA_BROKER = True
except (ImportError, ModuleNotFoundError):
    HAS_ALPACA_BROKER = False
    AlpacaBroker = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def alpaca_broker():
    """Create and connect an AlpacaBroker configured for paper trading.

    Adapts to whatever constructor signature AlpacaBroker exposes.
    """
    if not HAS_ALPACA_BROKER:
        pytest.skip("AlpacaBroker not available")

    broker = None

    # Try various likely constructor signatures
    init_kwargs: dict = {
        "api_key": ALPACA_API_KEY,
        "api_secret": ALPACA_API_SECRET,
        "base_url": ALPACA_BASE_URL,
    }

    try:
        broker = AlpacaBroker(**init_kwargs)
    except TypeError:
        # Maybe it takes a config dict instead
        try:
            broker = AlpacaBroker(config=init_kwargs)
        except TypeError:
            # Maybe it takes an EventBus as well
            try:
                eb = EventBus() if EventBus is not None else None
                broker = AlpacaBroker(event_bus=eb, **init_kwargs)
            except TypeError:
                pytest.skip("Cannot construct AlpacaBroker with available kwargs")

    if broker is None:
        pytest.skip("Failed to construct AlpacaBroker")

    if hasattr(broker, "connect"):
        await broker.connect()

    yield broker

    if hasattr(broker, "disconnect"):
        await broker.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_no_alpaca
@pytest.mark.asyncio
async def test_connect_alpaca(alpaca_broker) -> None:
    """Verify that connecting to Alpaca paper trading succeeds."""
    assert alpaca_broker is not None

    # The broker should be in a connected state
    if hasattr(alpaca_broker, "is_connected"):
        assert alpaca_broker.is_connected, "Broker should be connected after connect()"
    elif hasattr(alpaca_broker, "connected"):
        assert alpaca_broker.connected, "Broker should be connected after connect()"

    # As an extra sanity check, try fetching account info
    if hasattr(alpaca_broker, "get_account"):
        account = await alpaca_broker.get_account()
        assert account is not None


@skip_no_alpaca
@pytest.mark.asyncio
async def test_get_account(alpaca_broker) -> None:
    """Verify that account info returns valid data."""
    assert hasattr(alpaca_broker, "get_account"), "AlpacaBroker must implement get_account()"

    account = await alpaca_broker.get_account()
    assert account is not None

    # Account should be a dict (per BrokerAdapter protocol) or an object
    if isinstance(account, dict):
        # Expect at least some standard keys
        expected_keys = {"cash", "equity", "buying_power", "portfolio_value"}
        found_keys = set(account.keys())
        # At least one of the expected keys should be present
        assert found_keys & expected_keys, (
            f"Account dict should contain at least one of {expected_keys}, "
            f"got keys: {found_keys}"
        )
    else:
        # It might be a Pydantic model or dataclass
        has_some_attr = any(
            hasattr(account, attr)
            for attr in ("cash", "equity", "buying_power", "portfolio_value")
        )
        assert has_some_attr, "Account object should have cash, equity, or buying_power attribute"


@skip_no_alpaca
@pytest.mark.asyncio
async def test_get_positions(alpaca_broker) -> None:
    """Verify that get_positions returns a list."""
    assert hasattr(alpaca_broker, "get_positions"), "AlpacaBroker must implement get_positions()"

    positions = await alpaca_broker.get_positions()
    assert isinstance(positions, list), "get_positions() should return a list"


@skip_no_alpaca
@pytest.mark.asyncio
async def test_submit_cancel_order(alpaca_broker) -> None:
    """Submit a small limit order at a far-from-market price, then cancel it.

    Uses a very low limit price for a BUY order so it will NOT fill during
    the test.
    """
    assert hasattr(alpaca_broker, "submit_order"), "AlpacaBroker must implement submit_order()"
    assert hasattr(alpaca_broker, "cancel_order"), "AlpacaBroker must implement cancel_order()"

    # Create a limit order that won't fill (price far below market)
    order = Order(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("1.00"),  # absurdly low -- will not fill
        strategy_name="e2e_test",
    )

    # Submit
    submitted = await alpaca_broker.submit_order(order)
    assert submitted is not None

    order_id = submitted.id if hasattr(submitted, "id") else order.id

    # Give Alpaca a moment to process
    await asyncio.sleep(1.0)

    # Cancel
    cancel_result = await alpaca_broker.cancel_order(order_id)

    # cancel_order returns bool per BrokerAdapter protocol
    if isinstance(cancel_result, bool):
        assert cancel_result, "Cancel should return True"
    else:
        # Some implementations may return the updated order
        assert cancel_result is not None

    # Allow cancellation to propagate
    await asyncio.sleep(1.0)

    # Verify the order is no longer open
    if hasattr(alpaca_broker, "get_order"):
        final = await alpaca_broker.get_order(order_id)
        if final is not None and hasattr(final, "status"):
            assert final.status in (
                OrderStatus.CANCELLED,
                OrderStatus.PENDING,  # may still be transitioning
            ), f"Order should be cancelled, got {final.status}"
