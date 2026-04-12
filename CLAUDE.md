# Stock Trader - Project Instructions

## Overview
This is a production-grade algorithmic trading bot and web dashboard. The full spec is in `PRD.md`.

## Tech Stack
- **Backend**: Python 3.12, FastAPI, SQLAlchemy, Alembic
- **ML**: LightGBM, XGBoost, PyTorch (LSTM, TFT), scikit-learn, Stable-Baselines3
- **Data**: TimescaleDB (PostgreSQL), Redis, yfinance, Alpaca, Polygon.io
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, TradingView Lightweight Charts
- **Package Manager**: UV (Python), npm (Node)
- **Infrastructure**: Docker Compose, Alembic migrations

## Development Workflow
- Use UV for all Python dependency management (`uv init`, `uv add`, `uv run`)
- Use `pyproject.toml` for project config - no `requirements.txt`
- Follow the PRD phases in order (Phase 1 -> Phase 2 -> etc.)
- Write tests as you go - don't defer all testing to the end
- Use structlog for all logging
- Use Pydantic for all config and API schemas

## Architecture Rules
- Event-driven architecture using async pub/sub event bus
- All strategies must implement the BaseStrategy protocol
- All models must implement the BasePredictor protocol
- All broker adapters must implement the BrokerAdapter protocol
- Risk checks are mandatory before any order submission
- Walk-forward validation only - never train/test on overlapping data

## Sub-Agent Strategy
When working on this project, use sub-agents aggressively to parallelize:
- Spawn separate agents for independent modules (e.g., data providers, feature engineering, risk management)
- Use worktree isolation for agents that write code to avoid conflicts
- Each phase has independent subsystems - build them in parallel where possible
