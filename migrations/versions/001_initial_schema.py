"""001 – Initial schema: ohlcv, orders, strategies, portfolio_snapshots.

Creates the core tables for the trading system and converts time-series
tables (ohlcv, portfolio_snapshots) into TimescaleDB hypertables.

Revision ID: 001_initial
Revises: —
Create Date: 2026-04-12
"""

from alembic import op
import sqlalchemy as sa

# Alembic revision identifiers
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Enable TimescaleDB extension (idempotent)
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # ------------------------------------------------------------------
    # ohlcv
    # ------------------------------------------------------------------
    op.create_table(
        "ohlcv",
        # ``timestamp`` is part of the PK because TimescaleDB requires the
        # partitioning column to belong to every unique/primary key.
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), primary_key=True, nullable=False),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.Integer, nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column("vwap", sa.Float, nullable=True),
        sa.UniqueConstraint("symbol", "timestamp", "timeframe", name="uq_ohlcv_symbol_ts_tf"),
    )

    op.create_index("ix_ohlcv_symbol", "ohlcv", ["symbol"])
    op.create_index("ix_ohlcv_timestamp", "ohlcv", ["timestamp"])
    op.create_index("ix_ohlcv_symbol_timestamp", "ohlcv", ["symbol", "timestamp"])
    op.create_index("ix_ohlcv_symbol_timeframe_timestamp", "ohlcv", ["symbol", "timeframe", "timestamp"])

    # Convert to TimescaleDB hypertable
    op.execute(
        "SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);"
    )

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------
    op.create_table(
        "orders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("limit_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("stop_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("filled_quantity", sa.Numeric(18, 8), nullable=False, server_default="0"),
        sa.Column("filled_avg_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("strategy_name", sa.String(100), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index("ix_orders_symbol", "orders", ["symbol"])
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_strategy_name", "orders", ["strategy_name"])
    op.create_index("ix_orders_created_at", "orders", ["created_at"])

    # ------------------------------------------------------------------
    # strategies
    # ------------------------------------------------------------------
    op.create_table(
        "strategies",
        sa.Column("name", sa.String(100), primary_key=True),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("params", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("last_signal_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ------------------------------------------------------------------
    # portfolio_snapshots
    # ------------------------------------------------------------------
    op.create_table(
        "portfolio_snapshots",
        # ``timestamp`` is part of the PK (TimescaleDB partitioning requirement).
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), primary_key=True, nullable=False),
        sa.Column("total_value", sa.Float, nullable=False),
        sa.Column("cash", sa.Float, nullable=False),
        sa.Column("positions_value", sa.Float, nullable=False),
        sa.Column("unrealized_pnl", sa.Float, nullable=False, server_default="0"),
        sa.Column("realized_pnl", sa.Float, nullable=False, server_default="0"),
        sa.Column("drawdown", sa.Float, nullable=False, server_default="0"),
    )

    op.create_index("ix_portfolio_snapshots_timestamp", "portfolio_snapshots", ["timestamp"])

    # Convert to TimescaleDB hypertable
    op.execute(
        "SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);"
    )


def downgrade() -> None:
    op.drop_table("portfolio_snapshots")
    op.drop_table("strategies")
    op.drop_table("orders")
    op.drop_table("ohlcv")
