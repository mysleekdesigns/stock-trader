"""TimescaleDB-backed storage for OHLCV bars, orders, strategies, and portfolio snapshots.

Provides async CRUD operations using SQLAlchemy 2.0 ORM with asyncpg driver.
All tables are designed for TimescaleDB hypertables where time-series data is stored.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    and_,
    desc,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.core.exceptions import DataError
from src.core.types import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeFrame,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# ORM Base & Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class OHLCVRecord(Base):
    """OHLCV price bar stored in TimescaleDB hypertable."""

    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "timeframe", name="uq_ohlcv_symbol_ts_tf"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)


class OrderRecord(Base):
    """Persisted order record."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False, default=0)
    filled_avg_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class StrategyRecord(Base):
    """Persistent strategy configuration and state."""

    __tablename__ = "strategies"

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    last_signal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PortfolioSnapshot(Base):
    """Point-in-time snapshot of portfolio value for equity curve tracking."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    total_value: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    positions_value: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


# ---------------------------------------------------------------------------
# TimeseriesStore
# ---------------------------------------------------------------------------

class TimeseriesStore:
    """Async interface for TimescaleDB time-series and order storage."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._log = logger.bind(component="TimeseriesStore")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Create all tables (for development / testing convenience)."""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._log.info("database_tables_created")
        except Exception as exc:
            self._log.error("init_db_failed", error=str(exc))
            raise DataError(f"Failed to initialise database: {exc}") from exc

    # ------------------------------------------------------------------
    # OHLCV Bars
    # ------------------------------------------------------------------

    async def store_bars(self, bars: list[Bar]) -> int:
        """Bulk-insert OHLCV bars using INSERT … ON CONFLICT DO NOTHING.

        Returns the number of rows actually inserted.
        """
        if not bars:
            return 0

        rows = [
            {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "timeframe": bar.timeframe.value if isinstance(bar.timeframe, TimeFrame) else bar.timeframe,
                "vwap": bar.vwap,
            }
            for bar in bars
        ]

        try:
            async with self._session_factory() as session:
                stmt = (
                    pg_insert(OHLCVRecord)
                    .values(rows)
                    .on_conflict_do_nothing(
                        constraint="uq_ohlcv_symbol_ts_tf",
                    )
                )
                result = await session.execute(stmt)
                await session.commit()
                inserted = result.rowcount  # type: ignore[union-attr]
                self._log.info("bars_stored", count=inserted, total=len(bars))
                return inserted
        except Exception as exc:
            self._log.error("store_bars_failed", error=str(exc))
            raise DataError(f"Failed to store bars: {exc}") from exc

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame | str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Retrieve OHLCV bars for *symbol* within ``[start, end]``."""
        tf_value = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(OHLCVRecord)
                    .where(
                        and_(
                            OHLCVRecord.symbol == symbol,
                            OHLCVRecord.timeframe == tf_value,
                            OHLCVRecord.timestamp >= start,
                            OHLCVRecord.timestamp <= end,
                        )
                    )
                    .order_by(OHLCVRecord.timestamp)
                )
                result = await session.execute(stmt)
                records = result.scalars().all()

            return [
                Bar(
                    symbol=r.symbol,
                    timestamp=r.timestamp,
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                    timeframe=TimeFrame(r.timeframe),
                    vwap=r.vwap,
                )
                for r in records
            ]
        except Exception as exc:
            self._log.error("get_bars_failed", symbol=symbol, error=str(exc))
            raise DataError(f"Failed to retrieve bars: {exc}") from exc

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def store_order(self, order: Order) -> None:
        """Insert a new order or update an existing one (upsert by id)."""
        try:
            values = {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side.value if isinstance(order.side, OrderSide) else order.side,
                "quantity": order.quantity,
                "order_type": order.order_type.value if isinstance(order.order_type, OrderType) else order.order_type,
                "status": order.status.value if isinstance(order.status, OrderStatus) else order.status,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "filled_quantity": order.filled_quantity,
                "filled_avg_price": order.filled_avg_price,
                "strategy_name": order.strategy_name,
                "created_at": order.created_at,
                "updated_at": order.updated_at,
            }
            async with self._session_factory() as session:
                stmt = pg_insert(OrderRecord).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "status": stmt.excluded.status,
                        "filled_quantity": stmt.excluded.filled_quantity,
                        "filled_avg_price": stmt.excluded.filled_avg_price,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                await session.execute(stmt)
                await session.commit()
            self._log.info("order_stored", order_id=order.id, status=values["status"])
        except Exception as exc:
            self._log.error("store_order_failed", order_id=order.id, error=str(exc))
            raise DataError(f"Failed to store order: {exc}") from exc

    async def get_orders(
        self,
        symbol: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[OrderRecord]:
        """Retrieve orders with optional symbol/status filters."""
        try:
            async with self._session_factory() as session:
                stmt = select(OrderRecord)
                if symbol is not None:
                    stmt = stmt.where(OrderRecord.symbol == symbol)
                if status is not None:
                    stmt = stmt.where(OrderRecord.status == status)
                stmt = stmt.order_by(desc(OrderRecord.created_at)).limit(limit)
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            self._log.error("get_orders_failed", error=str(exc))
            raise DataError(f"Failed to retrieve orders: {exc}") from exc

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    async def store_portfolio_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Persist a single portfolio snapshot.

        *snapshot* should contain keys matching ``PortfolioSnapshot`` columns.
        """
        try:
            async with self._session_factory() as session:
                record = PortfolioSnapshot(**snapshot)
                session.add(record)
                await session.commit()
            self._log.info("portfolio_snapshot_stored", timestamp=snapshot.get("timestamp"))
        except Exception as exc:
            self._log.error("store_portfolio_snapshot_failed", error=str(exc))
            raise DataError(f"Failed to store portfolio snapshot: {exc}") from exc

    async def get_portfolio_history(
        self,
        start: datetime,
        end: datetime,
    ) -> list[PortfolioSnapshot]:
        """Retrieve portfolio snapshots within ``[start, end]``."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(PortfolioSnapshot)
                    .where(
                        and_(
                            PortfolioSnapshot.timestamp >= start,
                            PortfolioSnapshot.timestamp <= end,
                        )
                    )
                    .order_by(PortfolioSnapshot.timestamp)
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            self._log.error("get_portfolio_history_failed", error=str(exc))
            raise DataError(f"Failed to retrieve portfolio history: {exc}") from exc

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Dispose of the engine connection pool."""
        await self._engine.dispose()
        self._log.info("engine_disposed")
