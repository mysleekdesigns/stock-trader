"""Main async execution engine — the live trading loop.

The :class:`ExecutionEngine` wires together the event bus, strategies, risk
manager, portfolio, feature pipeline, broker adapter, and data feed into a
single cohesive trading loop.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import structlog

from src.core.events import (
    EventBus,
    FillEvent,
    MarketDataEvent,
    PortfolioUpdateEvent,
    SignalEvent,
)
from src.core.exceptions import (
    ExecutionError,
    FeatureComputationError,
    OrderSubmissionError,
)
from src.core.types import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalDirection,
    TimeFrame,
)
from src.data.providers.base import DataProviderBase
from src.execution.brokers.base import BrokerAdapter
from src.execution.order_manager import OrderManager
from src.features.pipeline import FeaturePipeline
from src.risk.manager import RiskManager
from src.risk.portfolio import Portfolio
from src.strategies.base import BaseStrategyABC

logger = structlog.get_logger(__name__)


class ExecutionEngine:
    """Async trading engine that orchestrates the full signal-to-order pipeline.

    Parameters
    ----------
    event_bus:
        Central pub/sub bus for all domain events.
    strategies:
        List of strategy instances to evaluate on each market data tick.
    risk_manager:
        Pre-trade and portfolio-level risk gate.
    portfolio:
        Live portfolio state tracker.
    feature_pipeline:
        Computes features from raw market data for strategy consumption.
    broker:
        Broker adapter for order submission and account queries.
    data_feed:
        Data provider for subscribing to real-time market data.
    config:
        Engine-level configuration. Supported keys:
        - ``symbols``: list[str] — symbols to trade.
        - ``timeframe``: str — bar timeframe (default "1m").
        - ``order_timeout_seconds``: int — per-order timeout (default 300).
        - ``max_bars_in_memory``: int — rolling window size (default 500).
        - ``default_order_type``: str — "market" or "limit" (default "market").
    """

    def __init__(
        self,
        event_bus: EventBus,
        strategies: list[BaseStrategyABC],
        risk_manager: RiskManager,
        portfolio: Portfolio,
        feature_pipeline: FeaturePipeline,
        broker: BrokerAdapter,
        data_feed: DataProviderBase,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._strategies = strategies
        self._risk_manager = risk_manager
        self._portfolio = portfolio
        self._feature_pipeline = feature_pipeline
        self._broker = broker
        self._data_feed = data_feed
        self._config = config or {}

        # Derived configuration
        self._symbols: list[str] = self._config.get("symbols", [])
        self._timeframe = TimeFrame(self._config.get("timeframe", "1m"))
        self._max_bars = self._config.get("max_bars_in_memory", 500)
        self._default_order_type = OrderType(
            self._config.get("default_order_type", "market")
        )

        # Sub-components
        self._order_manager = OrderManager(
            broker=broker,
            event_bus=event_bus,
            order_timeout_seconds=self._config.get("order_timeout_seconds", 300),
        )

        # State
        self._running = False
        self._bar_buffer: dict[str, list[dict[str, Any]]] = {
            s: [] for s in self._symbols
        }
        self._tasks: list[asyncio.Task[Any]] = []

        logger.info(
            "execution_engine.init",
            symbols=self._symbols,
            timeframe=self._timeframe.value,
            strategy_count=len(self._strategies),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def order_manager(self) -> OrderManager:
        return self._order_manager

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to broker & data feed, subscribe to events, and begin the loop."""
        if self._running:
            logger.warning("execution_engine.already_running")
            return

        logger.info("execution_engine.starting")
        self._running = True

        # Connect broker and data feed
        await self._broker.connect()
        await self._data_feed.connect()

        # Subscribe to broker order updates
        await self._broker.subscribe_order_updates(self._order_manager.on_order_update)

        # Subscribe to event-bus events
        await self._event_bus.subscribe("market_data", self._on_market_data)
        await self._event_bus.subscribe("fill", self._on_fill)

        # Start data feed subscription — bars published as MarketDataEvents
        self._tasks.append(
            asyncio.create_task(
                self._subscribe_data_feed(),
                name="data-feed-subscription",
            )
        )

        logger.info("execution_engine.started")

    async def stop(self) -> None:
        """Gracefully shut down the engine: cancel orders, disconnect, clean up."""
        if not self._running:
            return

        logger.info("execution_engine.stopping")
        self._running = False

        # Cancel open orders
        await self._order_manager.shutdown()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        # Unsubscribe from events
        await self._event_bus.unsubscribe("market_data", self._on_market_data)
        await self._event_bus.unsubscribe("fill", self._on_fill)

        # Disconnect
        await self._data_feed.disconnect()
        await self._broker.disconnect()

        # Flush pending risk events
        await self._risk_manager.flush_events()

        logger.info("execution_engine.stopped")

    # ------------------------------------------------------------------
    # Data feed subscription
    # ------------------------------------------------------------------

    async def _subscribe_data_feed(self) -> None:
        """Subscribe to the data feed and publish bars as MarketDataEvents."""
        try:
            await self._data_feed.subscribe_bars(
                symbols=self._symbols,
                timeframe=self._timeframe,
                callback=self._on_bar_received,
            )
        except asyncio.CancelledError:
            logger.info("execution_engine.data_feed_subscription_cancelled")
        except Exception:
            logger.exception("execution_engine.data_feed_subscription_error")
            self._running = False

    async def _on_bar_received(self, bar: Bar) -> None:
        """Callback invoked by the data feed for each new bar."""
        event = MarketDataEvent(
            symbol=bar.symbol,
            data_type="bar",
            payload={
                "symbol": bar.symbol,
                "timestamp": bar.timestamp.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "timeframe": bar.timeframe.value,
                "vwap": bar.vwap,
            },
        )
        await self._event_bus.publish(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_market_data(self, event: Any) -> None:
        """Process incoming market data: update features, run strategies, submit orders."""
        if not self._running:
            return

        if not isinstance(event, MarketDataEvent):
            return

        if event.data_type != "bar":
            return

        symbol = event.symbol
        payload = event.payload

        log = logger.bind(symbol=symbol)

        try:
            # 1. Buffer the bar
            self._buffer_bar(symbol, payload)

            # 2. Update portfolio prices
            close_price = Decimal(str(payload.get("close", 0)))
            self._portfolio.update_prices({symbol: close_price})

            # 3. Compute features
            features = self._compute_features(symbol)
            if features is None:
                return

            # 4. Run strategies and generate signals
            timestamp = datetime.fromisoformat(
                payload.get("timestamp", datetime.utcnow().isoformat())
            )
            signals = self._run_strategies(features, timestamp)

            # 5. Convert signals to orders, risk-check, and submit
            for signal in signals:
                await self._process_signal(signal)

            # 6. Flush risk events
            await self._risk_manager.flush_events()

            # 7. Publish portfolio update
            await self._publish_portfolio_update()

        except Exception:
            log.exception("execution_engine.market_data_processing_error")

    async def _on_fill(self, event: Any) -> None:
        """Update portfolio state when an order fills."""
        if not isinstance(event, FillEvent):
            return

        log = logger.bind(
            order_id=event.order_id,
            symbol=event.symbol,
        )

        try:
            order = self._order_manager.get_order(event.order_id)
            if order is None:
                log.warning("execution_engine.fill_unknown_order")
                return

            # Update portfolio based on the fill
            self._apply_fill_to_portfolio(
                symbol=event.symbol,
                side=order.side,
                quantity=event.filled_quantity,
                fill_price=event.filled_price,
                commission=event.commission,
            )

            log.info(
                "execution_engine.fill_processed",
                side=order.side.value,
                quantity=str(event.filled_quantity),
                price=str(event.filled_price),
            )

            await self._publish_portfolio_update()

        except Exception:
            log.exception("execution_engine.fill_processing_error")

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def _process_signal(self, signal: Signal) -> None:
        """Convert a signal to an order, risk-check it, and submit."""
        log = logger.bind(
            signal_id=signal.id,
            symbol=signal.symbol,
            direction=signal.direction.value,
        )

        # Publish signal event
        await self._event_bus.publish(SignalEvent(signal=signal))

        # Convert signal to order
        order = self._signal_to_order(signal)
        if order is None:
            log.debug("execution_engine.signal_skipped", reason="no order generated")
            return

        # Risk check
        passed, reason = self._risk_manager.check_order(order, self._portfolio)
        if not passed:
            log.warning(
                "execution_engine.order_rejected_by_risk",
                reason=reason,
            )
            # Try to adjust the order
            adjusted = self._risk_manager.adjust_order_for_risk(order, self._portfolio)
            if adjusted.quantity <= Decimal("0"):
                log.warning("execution_engine.order_adjusted_to_zero")
                return
            order = adjusted
            log.info(
                "execution_engine.order_adjusted",
                adjusted_quantity=str(order.quantity),
            )

        # Submit
        try:
            await self._order_manager.submit_order(order)
        except OrderSubmissionError as exc:
            log.error(
                "execution_engine.order_submission_failed",
                error=str(exc),
            )

    def _signal_to_order(self, signal: Signal) -> Order | None:
        """Map a :class:`Signal` to an :class:`Order`.

        Returns ``None`` for ``FLAT`` signals when there is no existing
        position to close.
        """
        if signal.direction == SignalDirection.FLAT:
            # Close existing position if any
            position = self._portfolio.positions.get(signal.symbol)
            if position is None:
                return None
            side = (
                OrderSide.SELL
                if position.side == OrderSide.BUY
                else OrderSide.BUY
            )
            return Order(
                symbol=signal.symbol,
                side=side,
                quantity=position.quantity,
                order_type=self._default_order_type,
                strategy_name=signal.strategy_name,
                metadata={"signal_id": signal.id},
            )

        side = (
            OrderSide.BUY
            if signal.direction == SignalDirection.LONG
            else OrderSide.SELL
        )

        # Position sizing: use signal strength to scale.  This is a simple
        # approach — a real system would delegate to a position-sizer module.
        equity = self._portfolio.total_value
        if equity <= Decimal("0"):
            return None

        # Target 2% of equity per trade, scaled by signal strength
        target_notional = equity * Decimal(str(0.02 * signal.strength))

        # Get last known price
        close_price = self._get_last_price(signal.symbol)
        if close_price is None or close_price <= Decimal("0"):
            return None

        quantity = (target_notional / close_price).quantize(Decimal("1"))
        if quantity <= Decimal("0"):
            return None

        return Order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=self._default_order_type,
            strategy_name=signal.strategy_name,
            metadata={
                "signal_id": signal.id,
                "signal_strength": signal.strength,
                "signal_confidence": signal.confidence,
            },
        )

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _buffer_bar(self, symbol: str, payload: dict[str, Any]) -> None:
        """Add a bar payload to the rolling buffer."""
        if symbol not in self._bar_buffer:
            self._bar_buffer[symbol] = []

        self._bar_buffer[symbol].append(payload)

        # Trim to max window size
        if len(self._bar_buffer[symbol]) > self._max_bars:
            self._bar_buffer[symbol] = self._bar_buffer[symbol][-self._max_bars:]

    def _compute_features(self, symbol: str) -> dict[str, Any] | None:
        """Compute features for *symbol* using the buffered bars.

        Returns ``None`` if there are insufficient bars or computation fails.
        """
        bars = self._bar_buffer.get(symbol, [])
        if len(bars) < 2:
            return None

        try:
            df = pd.DataFrame(bars)
            result_df = self._feature_pipeline.transform(df)
            if result_df.empty:
                return None
            # Return the most recent row as a dict
            return result_df.iloc[-1].to_dict()
        except FeatureComputationError:
            logger.warning(
                "execution_engine.feature_computation_failed",
                symbol=symbol,
            )
            return None
        except Exception:
            logger.exception(
                "execution_engine.feature_computation_error",
                symbol=symbol,
            )
            return None

    # ------------------------------------------------------------------
    # Strategy execution
    # ------------------------------------------------------------------

    def _run_strategies(
        self,
        features: dict[str, Any],
        timestamp: datetime,
    ) -> list[Signal]:
        """Run all enabled strategies and collect their signals."""
        signals: list[Signal] = []

        for strategy in self._strategies:
            if not strategy.enabled:
                continue

            try:
                strategy.validate_features(features)
                new_signals = strategy.generate_signals(features, timestamp)
                signals.extend(new_signals)

                if new_signals:
                    logger.info(
                        "execution_engine.signals_generated",
                        strategy=strategy.name,
                        count=len(new_signals),
                    )
            except FeatureComputationError:
                logger.warning(
                    "execution_engine.strategy_missing_features",
                    strategy=strategy.name,
                )
            except Exception:
                logger.exception(
                    "execution_engine.strategy_error",
                    strategy=strategy.name,
                )

        return signals

    # ------------------------------------------------------------------
    # Portfolio helpers
    # ------------------------------------------------------------------

    def _apply_fill_to_portfolio(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        fill_price: Decimal,
        commission: Decimal,
    ) -> None:
        """Update portfolio positions and cash after a fill."""
        existing = self._portfolio.positions.get(symbol)

        if existing is None:
            # New position
            self._portfolio.add_position(
                Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                )
            )
        elif existing.side == side:
            # Increase existing position
            total_qty = existing.quantity + quantity
            if total_qty > Decimal("0"):
                new_avg = (
                    existing.avg_entry_price * existing.quantity
                    + fill_price * quantity
                ) / total_qty
                self._portfolio.update_position(
                    symbol,
                    quantity=total_qty,
                    avg_entry_price=new_avg,
                    current_price=fill_price,
                )
        else:
            # Reduce or close position
            if quantity >= existing.quantity:
                self._portfolio.remove_position(symbol)
                remainder = quantity - existing.quantity
                if remainder > Decimal("0"):
                    self._portfolio.add_position(
                        Position(
                            symbol=symbol,
                            side=side,
                            quantity=remainder,
                            avg_entry_price=fill_price,
                            current_price=fill_price,
                        )
                    )
            else:
                self._portfolio.update_position(
                    symbol,
                    quantity=existing.quantity - quantity,
                    current_price=fill_price,
                )

        # Deduct commission from cash
        self._portfolio.cash -= commission

    def _get_last_price(self, symbol: str) -> Decimal | None:
        """Return the most recent close price for *symbol* from the buffer."""
        bars = self._bar_buffer.get(symbol, [])
        if not bars:
            return None
        close = bars[-1].get("close")
        if close is None:
            return None
        return Decimal(str(close))

    async def _publish_portfolio_update(self) -> None:
        """Publish a :class:`PortfolioUpdateEvent` with the current portfolio state."""
        event = PortfolioUpdateEvent(
            positions=list(self._portfolio.positions.values()),
            total_equity=self._portfolio.total_value,
            cash=self._portfolio.cash,
        )
        await self._event_bus.publish(event)
