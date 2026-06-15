"""Event-driven backtesting engine.

Replays historical data through the same event bus used in live trading,
coordinating strategy, risk, fill simulation, and portfolio components.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog

from src.core.events import (
    EventBus,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    SignalEvent,
)
from src.core.types import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Signal,
    SignalDirection,
    TimeFrame,
)
from src.backtest.analytics import BacktestAnalytics
from src.backtest.data_handler import DataHandler
from src.backtest.fill_simulator import FillSimulator

logger = structlog.get_logger(__name__)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class BacktestResult:
    """Container for all outputs produced by a backtest run."""

    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    portfolio_snapshots: list[dict[str, Any]] = field(default_factory=list)


# ======================================================================
# Engine
# ======================================================================

class BacktestEngine:
    """Event-driven backtesting engine.

    Accepts pluggable components for strategy, risk, fills, features, and
    data.  The main loop replays bars chronologically and pushes events
    through the same :class:`EventBus` used in live trading.

    Parameters
    ----------
    config:
        Backtest configuration dict.  Recognized keys:

        - ``strategy``: strategy instance (must implement ``generate_signals``).
        - ``risk_manager``: optional risk-check callable.
        - ``fill_simulator``: :class:`FillSimulator` instance (or config dict).
        - ``feature_pipeline``: :class:`FeaturePipeline` instance (optional).
        - ``data_handler``: :class:`DataHandler` instance (optional; can be
          built later via :meth:`run`).
        - ``portfolio``: Portfolio instance (optional; engine creates a
          simple one if absent).
        - ``event_bus``: optional :class:`EventBus` (defaults to a fresh one).
        - ``fill_config``: dict passed to :class:`FillSimulator` if
          ``fill_simulator`` is not provided directly.
        - ``feature_lookback``: int bars of history to feed the feature
          pipeline (default 100).
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        # Core event bus.
        self._event_bus: EventBus = config.get("event_bus") or EventBus()

        # Components — set in run() if not provided now.
        self._strategy = config.get("strategy")
        self._risk_manager = config.get("risk_manager")
        self._feature_pipeline = config.get("feature_pipeline")
        self._data_handler: DataHandler | None = config.get("data_handler")

        # Fill simulator.
        fs = config.get("fill_simulator")
        if isinstance(fs, FillSimulator):
            self._fill_simulator = fs
        else:
            self._fill_simulator = FillSimulator(
                config.get("fill_config") or (fs if isinstance(fs, dict) else None)
            )

        # Portfolio — will be initialized in run() with initial_capital.
        self._portfolio = config.get("portfolio")

        self._feature_lookback: int = config.get("feature_lookback", 100)

        # Fraction of equity allocated to a single signal at full strength.
        # Defaults to 0.10 (the multi-strategy/multi-symbol risk cap); raise it
        # for single-strategy evaluations that should deploy most of the book.
        self._max_position_pct: float = float(config.get("max_position_pct", 0.1))

        # Bookkeeping.
        self._pending_orders: list[Order] = []
        self._open_positions: dict[str, dict[str, Any]] = {}  # symbol -> info
        self._trades: list[dict] = []
        self._equity_curve: list[float] = []
        self._timestamps: list[datetime] = []
        self._snapshots: list[dict[str, Any]] = []

        logger.info("backtest_engine.init", config_keys=list(config.keys()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """Execute the backtest synchronously (spins up an event loop).

        Parameters
        ----------
        symbols:
            Ticker symbols.  Ignored if a :class:`DataHandler` was passed
            in the config.
        start, end:
            Date range.  Ignored if a DataHandler is already configured.
        timeframe:
            Bar resolution.
        initial_capital:
            Starting cash.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._run_async(symbols, start, end, timeframe, initial_capital),
        ) if self._has_running_loop() is False else asyncio.new_event_loop().run_until_complete(
            self._run_async(symbols, start, end, timeframe, initial_capital),
        )

    async def run_async(
        self,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """Async entry-point for the backtest."""
        return await self._run_async(symbols, start, end, timeframe, initial_capital)

    # ------------------------------------------------------------------
    # Internal main loop
    # ------------------------------------------------------------------

    async def _run_async(
        self,
        symbols: list[str] | None,
        start: datetime | None,
        end: datetime | None,
        timeframe: TimeFrame,
        initial_capital: float,
    ) -> BacktestResult:
        # Reset bookkeeping.
        self._pending_orders.clear()
        self._open_positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        self._timestamps.clear()
        self._snapshots.clear()

        # Initialize portfolio.
        cash = Decimal(str(initial_capital))
        portfolio = self._portfolio
        if portfolio is not None:
            portfolio.cash = cash
        else:
            # Lightweight internal portfolio tracker.
            portfolio = _SimplePortfolio(cash)
        self._portfolio = portfolio

        data_handler = self._data_handler
        if data_handler is None:
            raise ValueError(
                "No DataHandler configured.  Pass 'data_handler' in the "
                "config dict or set it before calling run()."
            )

        bar_count = 0
        total_bars = len(data_handler)

        logger.info(
            "backtest_engine.run_start",
            symbols=data_handler.symbols,
            total_bars=total_bars,
            initial_capital=initial_capital,
        )

        for timestamp, bar_dict in data_handler:
            bar_count += 1

            # 1. Publish market data events.
            await self._publish_market_data(timestamp, bar_dict)

            # 2. Update portfolio prices.
            self._update_portfolio_prices(bar_dict)

            # 3. Process pending orders (fills).
            await self._process_pending_orders(bar_dict)

            # 4. Compute features & generate signals.
            signals = self._generate_signals(timestamp, bar_dict, data_handler)

            # 5. Risk check & create orders.
            orders = self._create_orders_from_signals(signals, bar_dict)

            # 6. Risk filter.
            orders = self._apply_risk_checks(orders)

            # 7. Submit orders (add to pending).
            for order in orders:
                self._pending_orders.append(order)
                await self._event_bus.publish(OrderEvent(order=order))

            # 8. Record state.
            equity = self._get_total_equity(bar_dict)
            self._equity_curve.append(float(equity))
            self._timestamps.append(timestamp)
            self._snapshots.append(
                self._take_snapshot(timestamp, equity, bar_dict),
            )

            if bar_count % 500 == 0:
                logger.info(
                    "backtest_engine.progress",
                    bars_processed=bar_count,
                    total=total_bars,
                    equity=float(equity),
                )

        # Compute analytics.
        logger.info(
            "backtest_engine.run_complete",
            bars_processed=bar_count,
            trades=len(self._trades),
            final_equity=self._equity_curve[-1] if self._equity_curve else 0,
        )

        metrics: dict[str, Any] = {}
        if len(self._equity_curve) >= 2:
            analytics = BacktestAnalytics(
                equity_curve=self._equity_curve,
                timestamps=self._timestamps,
                trades=self._trades,
            )
            metrics = analytics.compute_all()

        return BacktestResult(
            equity_curve=self._equity_curve,
            timestamps=self._timestamps,
            trades=self._trades,
            metrics=metrics,
            portfolio_snapshots=self._snapshots,
        )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def _publish_market_data(
        self,
        timestamp: datetime,
        bar_dict: dict[str, Bar],
    ) -> None:
        for symbol, bar in bar_dict.items():
            event = MarketDataEvent(
                symbol=symbol,
                data_type="bar",
                payload={
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "timestamp": timestamp,
                },
            )
            await self._event_bus.publish(event)

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def _update_portfolio_prices(self, bar_dict: dict[str, Bar]) -> None:
        portfolio = self._portfolio
        if hasattr(portfolio, "update_prices"):
            prices = {s: Decimal(str(b.close)) for s, b in bar_dict.items()}
            portfolio.update_prices(prices)
        elif isinstance(portfolio, _SimplePortfolio):
            for symbol, bar in bar_dict.items():
                portfolio.update_price(symbol, Decimal(str(bar.close)))

    def _get_total_equity(self, bar_dict: dict[str, Bar]) -> Decimal:
        portfolio = self._portfolio
        if hasattr(portfolio, "total_value"):
            tv = portfolio.total_value
            return tv if isinstance(tv, Decimal) else Decimal(str(tv))
        if isinstance(portfolio, _SimplePortfolio):
            return portfolio.equity
        return Decimal("0")

    # ------------------------------------------------------------------
    # Order processing
    # ------------------------------------------------------------------

    async def _process_pending_orders(
        self, bar_dict: dict[str, Bar],
    ) -> None:
        remaining: list[Order] = []
        portfolio = self._portfolio
        equity = float(
            self._equity_curve[-1]
            if self._equity_curve
            else (portfolio.cash if hasattr(portfolio, "cash") else Decimal("100000"))
        )

        for order in self._pending_orders:
            bar = bar_dict.get(order.symbol)
            if bar is None:
                remaining.append(order)
                continue

            fill = self._fill_simulator.simulate_fill(order, bar, equity)
            if fill is None:
                remaining.append(order)
                continue

            # Update order state.
            order.status = OrderStatus.FILLED
            order.filled_quantity = fill.filled_quantity
            order.filled_avg_price = fill.filled_price
            order.updated_at = bar.timestamp

            # Publish fill event.
            fill_event = FillEvent(
                order_id=order.id,
                symbol=order.symbol,
                filled_quantity=fill.filled_quantity,
                filled_price=fill.filled_price,
                commission=fill.commission,
            )
            await self._event_bus.publish(fill_event)

            # Update portfolio.
            self._apply_fill_to_portfolio(order, fill, bar)

        self._pending_orders = remaining

    def _apply_fill_to_portfolio(
        self, order: Order, fill: Any, bar: Bar,
    ) -> None:
        portfolio = self._portfolio
        symbol = order.symbol
        price = fill.filled_price
        qty = fill.filled_quantity
        commission = fill.commission

        if isinstance(portfolio, _SimplePortfolio):
            trade = portfolio.process_fill(order, fill)
            if trade is not None:
                self._trades.append(trade)
        else:
            # Duck-typed external Portfolio.
            cost = price * qty
            if order.side == OrderSide.BUY:
                portfolio.cash -= cost + commission
                if hasattr(portfolio, "add_position"):
                    portfolio.add_position(symbol, qty, price)
            else:
                portfolio.cash += cost - commission
                if hasattr(portfolio, "remove_position"):
                    portfolio.remove_position(symbol, qty, price)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signals(
        self,
        timestamp: datetime,
        bar_dict: dict[str, Bar],
        data_handler: DataHandler,
    ) -> list[Signal]:
        if self._strategy is None:
            return []

        signals: list[Signal] = []
        try:
            # Build features dict for each symbol.
            for symbol, bar in bar_dict.items():
                features: dict[str, Any] = {
                    "symbol": symbol,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "timestamp": timestamp,
                }

                # If a feature pipeline is available, compute features.
                features_ready = False
                if self._feature_pipeline is not None:
                    history = data_handler.get_history(
                        symbol, self._feature_lookback, timestamp,
                    )
                    if not history.empty:
                        try:
                            feature_df = self._feature_pipeline.transform(history)
                            if not feature_df.empty:
                                last_row = feature_df.iloc[-1].to_dict()
                                features.update(last_row)
                                features_ready = True
                        except Exception as exc:
                            logger.debug(
                                "backtest_engine.feature_error",
                                symbol=symbol,
                                error=str(exc),
                            )
                else:
                    features_ready = True

                if not features_ready:
                    continue

                new_signals = self._strategy.generate_signals(features, timestamp)
                signals.extend(new_signals)

        except Exception as exc:
            logger.error(
                "backtest_engine.signal_generation_error",
                error=str(exc),
                exc_info=exc,
            )

        return signals

    # ------------------------------------------------------------------
    # Order creation
    # ------------------------------------------------------------------

    def _create_orders_from_signals(
        self,
        signals: list[Signal],
        bar_dict: dict[str, Bar],
    ) -> list[Order]:
        orders: list[Order] = []
        portfolio = self._portfolio

        for signal in signals:
            if signal.direction == SignalDirection.FLAT:
                # Close existing position.
                if isinstance(portfolio, _SimplePortfolio):
                    pos = portfolio.positions.get(signal.symbol)
                    if pos is not None:
                        side = (
                            OrderSide.SELL
                            if pos["side"] == OrderSide.BUY
                            else OrderSide.BUY
                        )
                        orders.append(
                            Order(
                                symbol=signal.symbol,
                                side=side,
                                quantity=pos["quantity"],
                                order_type=OrderType.MARKET,
                                strategy_name=signal.strategy_name,
                            )
                        )
                continue

            # Size the order based on signal strength and available capital.
            bar = bar_dict.get(signal.symbol)
            if bar is None:
                continue

            equity = float(self._get_total_equity(bar_dict))
            # Allocate a fraction of equity proportional to signal strength,
            # capped at ``max_position_pct`` of the book per signal.
            allocation = equity * signal.strength * self._max_position_pct
            price = bar.close
            if price <= 0 or allocation <= 0:
                continue

            # Fractional quantity — required for high-priced assets such as
            # crypto (e.g. BTC at ~$76k, where a whole unit dwarfs the
            # allocation and int() truncation would yield zero shares).
            qty = (Decimal(str(allocation)) / Decimal(str(price))).quantize(
                Decimal("0.00000001")
            )
            if qty <= 0:
                continue

            side = (
                OrderSide.BUY
                if signal.direction == SignalDirection.LONG
                else OrderSide.SELL
            )

            orders.append(
                Order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    strategy_name=signal.strategy_name,
                )
            )

        return orders

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    def _apply_risk_checks(self, orders: list[Order]) -> list[Order]:
        if self._risk_manager is None:
            return orders

        approved: list[Order] = []
        for order in orders:
            try:
                # Risk manager can be a callable: risk_manager(order, portfolio) -> bool
                passed = self._risk_manager(order, self._portfolio)
                if passed:
                    approved.append(order)
                else:
                    logger.info(
                        "backtest_engine.risk_rejected",
                        symbol=order.symbol,
                        side=order.side.value,
                        qty=str(order.quantity),
                    )
            except Exception as exc:
                logger.error(
                    "backtest_engine.risk_check_error",
                    error=str(exc),
                )
                # Reject on error — conservative.
        return approved

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _take_snapshot(
        self,
        timestamp: datetime,
        equity: Decimal,
        bar_dict: dict[str, Bar],
    ) -> dict[str, Any]:
        portfolio = self._portfolio
        positions: dict = {}
        if isinstance(portfolio, _SimplePortfolio):
            positions = dict(portfolio.positions)
        elif hasattr(portfolio, "positions"):
            positions = dict(portfolio.positions)

        return {
            "timestamp": timestamp,
            "equity": float(equity),
            "cash": float(portfolio.cash) if hasattr(portfolio, "cash") else 0.0,
            "positions": positions,
            "num_trades": len(self._trades),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus


# ======================================================================
# Lightweight internal portfolio tracker
# ======================================================================

class _SimplePortfolio:
    """Minimal portfolio tracker used when no external Portfolio is provided.

    Tracks cash, positions, and generates trade records on close.
    """

    def __init__(self, initial_cash: Decimal) -> None:
        self.cash: Decimal = initial_cash
        self.positions: dict[str, dict[str, Any]] = {}
        self._current_prices: dict[str, Decimal] = {}

    def update_price(self, symbol: str, price: Decimal) -> None:
        self._current_prices[symbol] = price
        if symbol in self.positions:
            self.positions[symbol]["current_price"] = price

    @property
    def equity(self) -> Decimal:
        total = self.cash
        for symbol, pos in self.positions.items():
            price = pos.get("current_price", pos["avg_entry_price"])
            total += pos["quantity"] * price
        return total

    @property
    def total_value(self) -> Decimal:
        return self.equity

    @property
    def drawdown(self) -> float:
        return 0.0  # Tracked externally by the engine.

    @property
    def gross_exposure(self) -> Decimal:
        total = Decimal("0")
        for pos in self.positions.values():
            price = pos.get("current_price", pos["avg_entry_price"])
            total += pos["quantity"] * price
        return total

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        for symbol, price in prices.items():
            self.update_price(symbol, price)

    def process_fill(self, order: Order, fill: Any) -> dict | None:
        """Update position state and return a trade dict if a position is closed."""
        symbol = order.symbol
        price = fill.filled_price
        qty = fill.filled_quantity
        commission = fill.commission

        if order.side == OrderSide.BUY:
            self.cash -= price * qty + commission
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["side"] == OrderSide.BUY:
                    # Adding to long.
                    old_cost = pos["avg_entry_price"] * pos["quantity"]
                    new_cost = price * qty
                    total_qty = pos["quantity"] + qty
                    pos["avg_entry_price"] = (old_cost + new_cost) / total_qty
                    pos["quantity"] = total_qty
                    return None
                else:
                    # Closing short.
                    return self._close_position(symbol, order, fill)
            else:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": OrderSide.BUY,
                    "quantity": qty,
                    "avg_entry_price": price,
                    "current_price": price,
                    "entry_time": order.created_at,
                }
                return None
        else:
            # SELL
            self.cash += price * qty - commission
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["side"] == OrderSide.SELL:
                    # Adding to short.
                    old_cost = pos["avg_entry_price"] * pos["quantity"]
                    new_cost = price * qty
                    total_qty = pos["quantity"] + qty
                    pos["avg_entry_price"] = (old_cost + new_cost) / total_qty
                    pos["quantity"] = total_qty
                    return None
                else:
                    # Closing long.
                    return self._close_position(symbol, order, fill)
            else:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": OrderSide.SELL,
                    "quantity": qty,
                    "avg_entry_price": price,
                    "current_price": price,
                    "entry_time": order.created_at,
                }
                return None

    def _close_position(
        self, symbol: str, order: Order, fill: Any,
    ) -> dict:
        pos = self.positions.pop(symbol)
        entry_price = pos["avg_entry_price"]
        exit_price = fill.filled_price
        qty = min(pos["quantity"], fill.filled_quantity)

        if pos["side"] == OrderSide.BUY:
            pnl = float((exit_price - entry_price) * qty - fill.commission)
        else:
            pnl = float((entry_price - exit_price) * qty - fill.commission)

        return {
            "symbol": symbol,
            "side": pos["side"].value,
            "quantity": float(qty),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": pnl,
            "commission": float(fill.commission),
            "entry_time": pos.get("entry_time"),
            "exit_time": order.updated_at,
        }
