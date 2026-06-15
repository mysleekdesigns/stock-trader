"""Unit tests for the IBKR broker adapter (offline / not-connected paths)."""

from __future__ import annotations

import importlib.util
from decimal import Decimal

import pytest

from src.core.exceptions import OrderCancellationError, OrderSubmissionError
from src.core.types import Order, OrderSide, OrderStatus, OrderType
from src.execution.brokers import IBKRBroker

_HAS_IB_INSYNC = importlib.util.find_spec("ib_insync") is not None


def _market_order() -> Order:
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )


def test_init_defaults_and_env(monkeypatch) -> None:
    broker = IBKRBroker()
    assert broker._host == "127.0.0.1"
    assert broker._port == 7497
    assert broker._client_id == 1

    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "7")
    broker2 = IBKRBroker()
    assert broker2._host == "10.0.0.5"
    assert broker2._port == 4002
    assert broker2._client_id == 7


async def test_submit_without_connect_raises() -> None:
    broker = IBKRBroker()
    with pytest.raises(OrderSubmissionError):
        await broker.submit_order(_market_order())


async def test_cancel_without_connect_raises() -> None:
    broker = IBKRBroker()
    with pytest.raises(OrderCancellationError):
        await broker.cancel_order("nonexistent")


async def test_positions_and_account_empty_when_disconnected() -> None:
    broker = IBKRBroker()
    assert await broker.get_positions() == []
    assert await broker.get_account() == {}


async def test_subscribe_registers_callback() -> None:
    broker = IBKRBroker()

    async def _cb(order: Order) -> None:  # pragma: no cover - not invoked here
        pass

    await broker.subscribe_order_updates(_cb)
    assert _cb in broker._update_callbacks


def test_status_mapping() -> None:
    assert IBKRBroker._map_status("Filled") == OrderStatus.FILLED
    assert IBKRBroker._map_status("Submitted") == OrderStatus.SUBMITTED
    assert IBKRBroker._map_status("PreSubmitted") == OrderStatus.SUBMITTED
    assert IBKRBroker._map_status("PendingSubmit") == OrderStatus.PENDING
    assert IBKRBroker._map_status("Cancelled") == OrderStatus.CANCELLED
    assert IBKRBroker._map_status("ApiCancelled") == OrderStatus.CANCELLED
    assert IBKRBroker._map_status("Inactive") == OrderStatus.REJECTED
    # Unknown -> SUBMITTED default
    assert IBKRBroker._map_status("Weird") == OrderStatus.SUBMITTED


def test_order_type_mapping() -> None:
    assert IBKRBroker._map_ib_order_type("MKT") == OrderType.MARKET
    assert IBKRBroker._map_ib_order_type("LMT") == OrderType.LIMIT
    assert IBKRBroker._map_ib_order_type("STP") == OrderType.STOP
    assert IBKRBroker._map_ib_order_type("STP LMT") == OrderType.STOP_LIMIT


async def test_connect_without_dependency_raises_importerror() -> None:
    if _HAS_IB_INSYNC:
        pytest.skip("ib_insync is installed; cannot test missing-dependency path")
    broker = IBKRBroker()
    with pytest.raises(ImportError):
        await broker.connect()


@pytest.mark.skipif(not _HAS_IB_INSYNC, reason="ib_insync not installed")
def test_build_ib_order_requires_prices() -> None:
    broker = IBKRBroker()
    limit_no_price = Order(
        symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("1"), order_type=OrderType.LIMIT
    )
    with pytest.raises(OrderSubmissionError):
        broker._build_ib_order(limit_no_price)


@pytest.mark.skipif(not _HAS_IB_INSYNC, reason="ib_insync not installed")
def test_build_contract_stock_default() -> None:
    broker = IBKRBroker()
    contract = broker._build_contract(_market_order())
    assert contract.symbol == "AAPL"
