"""Execution engine and order management.

Re-exports key classes for convenient access::

    from src.execution import ExecutionEngine, OrderManager
    from src.execution.brokers import BrokerAdapter, AlpacaBroker, SimulatedBroker
"""

from src.execution.engine import ExecutionEngine
from src.execution.order_manager import OrderManager

__all__ = [
    "ExecutionEngine",
    "OrderManager",
]
