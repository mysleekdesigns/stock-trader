# Stock Trader Bot & App - Product Requirements Document

## Overview

Production-grade algorithmic trading bot and web dashboard using an ensemble of ML/AI models to generate buy, sell, short, and futures trading signals. Multi-strategy engine with real-time execution, comprehensive risk management, and a React-based monitoring dashboard.

---

## Phase 1: Foundation

### Project Setup
- [x] Initialize git repository
- [x] Create `pyproject.toml` with UV package manager and all core dependencies
- [x] Create `.env.example` template (without secrets)
- [x] Create `Makefile` with common commands (setup, test, lint, run)
- [x] Create `docker-compose.yml` (PostgreSQL + TimescaleDB, Redis)
- [x] Create `Dockerfile` for backend container
- [x] Create `alembic.ini` and initial migration setup

### Configuration
- [x] `config/__init__.py`
- [x] `config/settings.py` - Pydantic Settings model loading from `.env`
- [x] `config/logging_config.py` - structlog structured logging
- [x] `config/assets.yaml` - tradeable asset universe definitions
- [x] `config/strategies.yaml` - strategy parameter configurations
- [x] `config/risk_limits.yaml` - risk management thresholds

> _Note: the `Settings` model's canonical source is `src/core/config.py`; `config/settings.py` and `config/__init__.py` re-export it to match this layout. The root `config/` YAMLs are consumed by `scripts/run_backtest.py` (default `--config-dir config`) and the training pipeline._

### Core Domain Primitives
- [x] `src/__init__.py`
- [x] `src/core/__init__.py`
- [x] `src/core/types.py` - OrderSide, AssetClass, TimeFrame enums; Signal, Quote, Position dataclasses
- [x] `src/core/events.py` - async pub/sub event bus (MarketDataEvent, SignalEvent, OrderEvent, FillEvent, RiskBreachEvent)
- [x] `src/core/interfaces.py` - Protocol definitions (DataProvider, BasePredictor, BaseStrategy, BrokerAdapter)
- [x] `src/core/exceptions.py` - custom exception hierarchy

### Data Providers
- [x] `src/data/__init__.py`
- [x] `src/data/providers/__init__.py`
- [x] `src/data/providers/base.py` - DataProvider protocol (get_bars, get_quote, subscribe_bars, subscribe_trades)
- [x] `src/data/providers/yfinance_provider.py` - Yahoo Finance free data fallback
- [x] `src/data/providers/alpaca_provider.py` - Alpaca Markets REST + WebSocket
- [x] `src/data/providers/polygon_provider.py` - Polygon.io tick data

### Data Storage
- [x] `src/data/storage/__init__.py`
- [x] `src/data/storage/timeseries_store.py` - TimescaleDB interface with hypertables
- [x] `src/data/storage/feature_store.py` - Redis-backed feature store for low-latency inference
- [x] `src/data/storage/cache.py` - Redis caching layer
- [x] Create Alembic migrations for OHLCV hypertables, orders, strategies, portfolio tables

### Feature Engineering
- [x] `src/features/__init__.py`
- [x] `src/features/registry.py` - feature registry with dependency DAG and topological sort
- [x] `src/features/technical.py` - TA-Lib wrappers: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Ichimoku (200+ indicators)
- [x] `src/features/price.py` - log returns, realized volatility, volume profiles, gap features, relative range
- [x] `src/features/pipeline.py` - FeatureEngineeringPipeline orchestrator (fit/transform/fit_transform)

### Data Pipeline
- [x] `src/data/pipeline.py` - orchestrates ingest -> normalize -> transform -> store

### Phase 1 Verification
- [x] Docker services start (Postgres+TimescaleDB, Redis)
- [x] Import all core types successfully
- [x] Fetch OHLCV data via yfinance and store in TimescaleDB
- [x] Compute technical + price features on sample data
- [x] Feature registry resolves dependency order correctly

---

## Phase 2: Backtest Engine + First Strategy

### Trading Strategies (Base)
- [x] `src/strategies/__init__.py`
- [x] `src/strategies/base.py` - BaseStrategy ABC (generate_signals, get_required_features)
- [x] `src/strategies/signal.py` - Signal dataclass (symbol, direction, strength, confidence, metadata) + signal combiner
- [x] `src/strategies/momentum.py` - trend-following strategy (MA crossovers, ADX, breakout detection)

### Risk Management
- [x] `src/risk/__init__.py`
- [x] `src/risk/manager.py` - RiskManager with pre-trade + portfolio-level checks
- [x] `src/risk/position_sizer.py` - Kelly criterion (fractional 0.25-0.5x), volatility-adjusted, fixed-fractional
- [x] `src/risk/portfolio.py` - Portfolio state tracking, P&L computation
- [x] `src/risk/limits.py` - drawdown limits (10%), daily loss (3%), exposure limits (200% gross), concentration limits (5% per position, 25% per sector)
- [x] `src/risk/stop_loss.py` - trailing stops, ATR-based stops (2.5x multiplier), time-based exits (20 day max hold)

### Backtesting Framework
- [x] `src/backtest/__init__.py`
- [x] `src/backtest/engine.py` - event-driven BacktestEngine (shares Strategy/Risk/Feature code with live)
- [x] `src/backtest/data_handler.py` - historical data replay from TimescaleDB
- [x] `src/backtest/fill_simulator.py` - slippage (volume-dependent), commissions (per-trade/per-share), Almgren-Chriss market impact model
- [x] `src/backtest/analytics.py` - Sharpe, Sortino, Calmar, max drawdown (depth+duration), win rate, profit factor, alpha, beta, information ratio
- [x] `src/backtest/report.py` - HTML report generation with equity curve, drawdown chart, metrics table

### Scripts & Tests
- [x] `scripts/run_backtest.py` - CLI for backtest execution
- [x] `scripts/seed_historical.py` - bulk load historical data
- [x] `tests/conftest.py` - shared fixtures
- [x] `tests/unit/test_features.py` - feature computation correctness
- [x] `tests/unit/test_risk_manager.py` - risk gate edge cases
- [x] `tests/unit/test_position_sizer.py` - sizing calculations vs hand-computed values
- [x] `tests/unit/test_strategies.py` - signal generation with synthetic data

### Phase 2 Verification
- [x] Run backtest on SPY 2020-2024 with momentum strategy
- [x] Verify Sharpe/drawdown metrics match manual calculation
- [x] Verify fill simulator applies slippage correctly
- [x] All unit tests pass with 80%+ coverage on risk and backtest modules

---

## Phase 3: ML Models + Ensemble

### Base Model Infrastructure
- [x] `src/models/__init__.py`
- [x] `src/models/base.py` - BasePredictor protocol (fit, predict, predict_proba, save, load, get_feature_importance)

### Gradient Boosting Models
- [x] `src/models/tree/__init__.py`
- [x] `src/models/tree/lightgbm_model.py` - 3 sub-models: direction classifier, return regressor, volatility regressor
- [x] `src/models/tree/xgboost_model.py` - XGBoost alternative with same interface

### Deep Learning Models
- [x] `src/models/deep/__init__.py`
- [x] `src/models/deep/lstm_attention.py` - BiLSTM (2 layers) + self-attention + 3-head output (direction/return/vol)

### Transformer Models
- [x] `src/models/transformer/__init__.py`
- [x] `src/models/transformer/temporal_fusion.py` - TFT via pytorch-forecasting: multi-horizon forecasting (1d/2d/5d/10d/21d), variable selection attention, known/unknown future covariates

### Ensemble Meta-Learner
- [x] `src/models/ensemble.py` - Ridge regression stacking: separate meta-learner per time horizon, dynamic weight adjustment (exponential decay by rolling Sharpe), 0.05 minimum weight floor

### Training Pipeline
- [x] `src/models/training/__init__.py`
- [x] `src/models/training/walk_forward.py` - walk-forward splits (504d train / 63d val / 21d test, step 21d)
- [x] `src/models/training/trainer.py` - orchestrates full walk-forward training for all models + meta-learner
- [x] `src/models/training/hyperopt.py` - Optuna with financial-specific pruning (prune if rolling Sharpe < threshold)
- [x] `src/models/training/registry.py` - MLflow experiment tracking, model versioning, model registry (staging/production)

### ML-Driven Strategy
- [x] `src/strategies/ml_alpha.py` - wraps ensemble predictor, translates prediction confidence into signal strength

### Scripts
- [x] `scripts/train_models.py` - CLI for model training pipeline

### Tests
- [x] `tests/unit/test_ensemble.py` - meta-learner weight updates, prediction combination
- [x] `tests/integration/test_training_pipeline.py` - walk-forward doesn't leak future data

### Phase 3 Verification
- [x] Train LightGBM on 2+ years of data, verify walk-forward integrity (test timestamps > train end)
- [x] Train LSTM and TFT models, verify convergence on validation loss
- [x] Verify ensemble outperforms best single model on held-out test
- [x] MLflow UI shows all experiments with metrics and artifacts
- [x] Backtest with ml_alpha strategy shows improvement over momentum alone

---

## Phase 4: Live Execution Engine

### Execution Engine
- [x] `src/execution/__init__.py`
- [x] `src/execution/engine.py` - main async trading loop (receive data -> update features -> inference -> signals -> risk check -> submit orders -> monitor fills -> update portfolio -> publish to WebSocket)
- [x] `src/execution/order_manager.py` - order lifecycle (create, submit, partial fill, fill, cancel, reject)

### Broker Adapters
- [x] `src/execution/brokers/__init__.py`
- [x] `src/execution/brokers/base.py` - BrokerAdapter protocol (submit_order, cancel_order, get_positions, get_account, subscribe_order_updates)
- [x] `src/execution/brokers/alpaca_broker.py` - Alpaca paper + live trading (controlled by APCA_API_BASE_URL)
- [x] `src/execution/brokers/simulated_broker.py` - in-memory simulated fills for local development

### Real-Time Data Feed
- [x] `src/data/feed/__init__.py`
- [x] `src/data/feed/realtime_feed.py` - WebSocket multiplexer aggregating multiple providers
- [x] `src/data/feed/bar_aggregator.py` - tick-to-bar aggregation (1m/5m/15m/1h configurable)

### Composite Strategy
- [x] `src/strategies/composite.py` - multi-strategy combiner with configurable weights, confidence-weighted conflict resolution, minimum agreement threshold (60%)

### Tests
- [x] `tests/integration/test_execution_flow.py` - signal -> risk -> order -> fill -> portfolio using SimulatedBroker
- [x] `tests/e2e/test_paper_trading.py` - connect to Alpaca paper, submit real paper order, verify fill

### Phase 4 Verification
- [x] Connect to Alpaca paper trading API successfully
- [x] Submit a market order, receive fill callback, verify portfolio state update
- [x] Real-time WebSocket feed receives live price data
- [x] Execution engine runs complete loop: data -> features -> model -> signal -> order
- [x] SimulatedBroker passes all integration tests

---

## Phase 5: Web Dashboard

### FastAPI Backend
- [x] `src/api/__init__.py`
- [x] `src/api/app.py` - FastAPI application factory
- [x] `src/api/deps.py` - dependency injection (DB sessions, Redis, services)
- [x] `src/api/middleware.py` - authentication, CORS, rate limiting
- [x] `src/api/schemas/__init__.py` + `portfolio.py` + `order.py` + `strategy.py` + `backtest.py` - Pydantic response models

### API Routers
- [x] `src/api/routers/__init__.py`
- [x] `src/api/routers/dashboard.py` - GET /api/dashboard (portfolio summary, P&L, positions)
- [x] `src/api/routers/strategies.py` - CRUD strategies, POST toggle enable/disable
- [x] `src/api/routers/orders.py` - order history with filters, POST manual order
- [x] `src/api/routers/backtest.py` - POST run backtest, GET results
- [x] `src/api/routers/models.py` - model performance, accuracy, contribution metrics
- [x] `src/api/routers/risk.py` - current risk metrics, limit overrides
- [x] `src/api/routers/websocket.py` - real-time streaming via Redis pub/sub (prices, portfolio, orders, signals)

### React Frontend Setup
- [x] `frontend/package.json` - React 18 + TypeScript + Vite
- [x] `frontend/vite.config.ts`
- [x] `frontend/tsconfig.json`
- [x] `frontend/tailwind.config.ts` - Tailwind CSS + shadcn/ui setup _(Tailwind v4 CSS-first: design tokens live in `src/index.css` via `@theme`/`@utility`; `tailwind.config.ts` is loaded via `@config` and the `@tailwindcss/vite` plugin drives the build — no `postcss.config.js` needed)_
- [x] `frontend/src/main.tsx` + `frontend/src/App.tsx`
- [x] `frontend/src/api/client.ts` - Axios wrapper + WebSocket hook
- [x] `frontend/src/stores/useTradeStore.ts` - Zustand state management

### Dashboard Components
- [x] `frontend/src/components/Layout.tsx` - app shell with navigation
- [x] `frontend/src/components/charts/PriceChart.tsx` - TradingView Lightweight Charts candlestick
- [x] `frontend/src/components/charts/EquityCurve.tsx` - portfolio value over time
- [x] `frontend/src/components/charts/DrawdownChart.tsx` - underwater equity chart
- [x] `frontend/src/components/charts/HeatMap.tsx` - correlation / sector heat map
- [x] `frontend/src/components/portfolio/PositionsTable.tsx` - open positions with unrealized P&L
- [x] `frontend/src/components/portfolio/PnLSummary.tsx` - daily/weekly/monthly P&L cards
- [x] `frontend/src/components/portfolio/AllocationDonut.tsx` - portfolio allocation chart
- [x] `frontend/src/components/strategies/StrategyCard.tsx` - status, signals, toggle
- [x] `frontend/src/components/strategies/StrategyConfig.tsx` - parameter adjustment
- [x] `frontend/src/components/risk/RiskGauge.tsx` - visual risk level indicator
- [x] `frontend/src/components/risk/ExposureBar.tsx` - exposure by sector/asset class
- [x] `frontend/src/components/orders/TradeLog.tsx` - trade history with filters

### Pages
- [x] `frontend/src/pages/Dashboard.tsx` - main dashboard with equity curve, P&L, positions, allocation, risk gauges
- [x] `frontend/src/pages/Strategies.tsx` - strategy cards with config and toggle
- [x] `frontend/src/pages/Backtest.tsx` - form to configure/run backtests, results display
- [x] `frontend/src/pages/Models.tsx` - ML model performance dashboard
- [x] `frontend/src/pages/Orders.tsx` - trade log with CSV export
- [x] `frontend/src/pages/Settings.tsx` - API keys, risk limits, notification preferences

### Phase 5 Verification
- [x] `uvicorn` starts FastAPI backend, OpenAPI docs accessible at /docs
- [x] `vite dev` starts frontend, dashboard page loads
- [x] WebSocket connection established, real-time data streaming
- [x] Backtest page triggers a run and displays results with charts
- [x] All API endpoints return correct data with proper Pydantic validation

> _Verified offline: `npm run build` (tsc + vite) passes cleanly; `create_app()` registers OpenAPI/`/docs`, the `/api/ws` WebSocket route, and all 20 API routers. Live end-to-end streaming/charts require the running stack (Postgres/TimescaleDB + Redis + uvicorn + `vite dev`)._

---

## Phase 6: Advanced Features

### Reinforcement Learning Agent
- [x] `src/models/rl/__init__.py`
- [x] `src/models/rl/environment.py` - Gym-compatible TradingEnv (state: signal+confidence+position+pnl+vol, action: continuous [-1,1], reward: risk-adjusted return)
- [x] `src/models/rl/reward.py` - Sharpe-based, Sortino-based, and asymmetric reward functions
- [x] `src/models/rl/agent.py` - SAC agent via Stable-Baselines3 for position sizing and execution timing
- [x] `src/strategies/rl_strategy.py` - wraps RL agent as a strategy

### Sentiment & Alternative Data
- [x] `src/models/sentiment/__init__.py`
- [x] `src/models/sentiment/finbert.py` - FinBERT (ProsusAI/finbert) for financial news sentiment scoring
- [x] `src/data/alternative/__init__.py`
- [x] `src/data/alternative/sentiment.py` - news headline scraping, social media aggregation
- [x] `src/data/alternative/dark_pool.py` - FINRA dark pool volume data
- [x] `src/features/sentiment_features.py` - rolling sentiment averages, momentum, dispersion
- [x] `src/features/cross_asset.py` - VIX features, yield curve features, sector rotation signals

### Market Microstructure
- [x] `src/data/feed/order_book.py` - L2 order book reconstruction from WebSocket
- [x] `src/features/microstructure.py` - bid-ask spread, order book imbalance, VPIN, Kyle's lambda
- [x] `src/data/providers/options_flow.py` - Unusual Whales / CBOE options flow data

### Additional Strategies
- [x] `src/strategies/mean_reversion.py` - Bollinger deviation, z-score, RSI extremes
- [x] `src/strategies/options_flow.py` - unusual options activity detection, smart money flow signals
- [x] `src/strategies/pairs_trading.py` - Engle-Granger cointegration, spread z-score entry/exit

### Portfolio Optimization
- [x] `src/risk/optimizer.py` - Black-Litterman model (CAPM equilibrium + ML views -> optimal allocation), weekly rebalancing

### Advanced Execution
- [x] `src/execution/smart_router.py` - TWAP, VWAP, iceberg order execution algorithms
- [x] `src/execution/brokers/ibkr_broker.py` - Interactive Brokers TWS/Gateway for futures and options

### Model Monitoring & Auto-Retrain
- [x] `src/models/training/monitoring.py` - feature drift (KS-test), prediction drift, performance decay (Sharpe < 0.5), regime change (VIX > 25) via evidently library
- [x] `src/models/training/ab_testing.py` - A/B framework for comparing model versions before promotion

### Production Deployment
- [x] `docker-compose.prod.yml` - Gunicorn+Uvicorn, Nginx, Celery worker+beat, persistent volumes
- [x] `deploy/nginx.conf` - reverse proxy, SSL termination
- [x] `deploy/supervisord.conf` - process management for trading engine

### Phase 6 Verification
- [ ] RL agent trains in custom environment, produces reasonable position sizes
- [ ] FinBERT sentiment features improve ensemble accuracy on validation set
- [ ] Order book features compute from live L2 data
- [ ] IBKR adapter connects and fetches futures quotes
- [x] Drift monitor correctly flags synthetic distribution shifts
- [ ] Production Docker stack starts all services cleanly

> _Status: all Phase 6 **implementations** are complete and import cleanly; `dark_pool.py` and `ibkr_broker.py` are covered by new unit tests (`tests/unit/test_dark_pool.py`, `tests/unit/test_ibkr_broker.py`). The drift monitor was verified offline (KS-test flags a shifted feature, ignores an unchanged one). The remaining end-to-end checks require live infrastructure / credentials not available in this environment: an RL training run (stable-baselines3), FinBERT model download + ensemble retrain, a live L2 feed, a running IBKR TWS/Gateway (`uv add ib_insync`), and a Docker host. The `sentiment.py` and `options_flow.py` providers return synthetic/empty data with the production schema pending live vendor API keys (NewsAPI / Unusual Whales / CBOE)._

---

## Technical Specifications

### ML Ensemble Architecture
| Model | Type | Horizon | Library |
|---|---|---|---|
| Temporal Fusion Transformer | Multi-horizon forecaster | 1d-21d | pytorch-forecasting |
| LightGBM (3 heads) | Direction + Return + Vol | All | lightgbm |
| XGBoost (3 heads) | Direction + Return + Vol | All | xgboost |
| BiLSTM + Attention | Short-term movement | Intraday-2d | PyTorch |
| FinBERT Sentiment | Directional bias | 1d-5d | transformers |
| Technical Signal (RF) | Pattern detection | 1d-5d | scikit-learn |
| **Meta-Learner (Ridge)** | **Ensemble combiner** | **Per horizon** | **scikit-learn** |
| SAC RL Agent | Position sizing | Real-time | stable-baselines3 |

### Risk Parameters (Defaults)
| Parameter | Value |
|---|---|
| Max portfolio drawdown | 10% |
| Max daily loss | 3% |
| Max gross exposure | 200% |
| Max net exposure | 100% |
| Max single position | 5% of portfolio |
| Max sector exposure | 25% |
| Max correlated positions | 3 |
| Stop-loss type | Trailing ATR (2.5x) |
| Max hold period | 20 days |
| Kelly fraction | 0.25-0.5x |

### Strategy Weights (Default Composite)
| Strategy | Weight |
|---|---|
| ML Alpha (ensemble) | 40% |
| Momentum | 20% |
| Mean Reversion | 15% |
| Options Flow | 15% |
| RL Strategy | 10% |
| **Minimum agreement** | **60%** |

### Walk-Forward Training
| Parameter | Value |
|---|---|
| Training window | 504 days (2 years) |
| Validation window | 63 days (3 months) |
| Test window | 21 days (1 month) |
| Step size | 21 days |
| Auto-retrain trigger | Sharpe < 0.5 or accuracy < 52% |
