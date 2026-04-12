# Stock Trader Bot & App - Product Requirements Document

## Overview

Production-grade algorithmic trading bot and web dashboard using an ensemble of ML/AI models to generate buy, sell, short, and futures trading signals. Multi-strategy engine with real-time execution, comprehensive risk management, and a React-based monitoring dashboard.

---

## Phase 1: Foundation

### Project Setup
- [ ] Initialize git repository
- [ ] Create `pyproject.toml` with UV package manager and all core dependencies
- [ ] Create `.env.example` template (without secrets)
- [ ] Create `Makefile` with common commands (setup, test, lint, run)
- [ ] Create `docker-compose.yml` (PostgreSQL + TimescaleDB, Redis)
- [ ] Create `Dockerfile` for backend container
- [ ] Create `alembic.ini` and initial migration setup

### Configuration
- [ ] `config/__init__.py`
- [ ] `config/settings.py` - Pydantic Settings model loading from `.env`
- [ ] `config/logging_config.py` - structlog structured logging
- [ ] `config/assets.yaml` - tradeable asset universe definitions
- [ ] `config/strategies.yaml` - strategy parameter configurations
- [ ] `config/risk_limits.yaml` - risk management thresholds

### Core Domain Primitives
- [ ] `src/__init__.py`
- [ ] `src/core/__init__.py`
- [ ] `src/core/types.py` - OrderSide, AssetClass, TimeFrame enums; Signal, Quote, Position dataclasses
- [ ] `src/core/events.py` - async pub/sub event bus (MarketDataEvent, SignalEvent, OrderEvent, FillEvent, RiskBreachEvent)
- [ ] `src/core/interfaces.py` - Protocol definitions (DataProvider, BasePredictor, BaseStrategy, BrokerAdapter)
- [ ] `src/core/exceptions.py` - custom exception hierarchy

### Data Providers
- [ ] `src/data/__init__.py`
- [ ] `src/data/providers/__init__.py`
- [ ] `src/data/providers/base.py` - DataProvider protocol (get_bars, get_quote, subscribe_bars, subscribe_trades)
- [ ] `src/data/providers/yfinance_provider.py` - Yahoo Finance free data fallback
- [ ] `src/data/providers/alpaca_provider.py` - Alpaca Markets REST + WebSocket
- [ ] `src/data/providers/polygon_provider.py` - Polygon.io tick data

### Data Storage
- [ ] `src/data/storage/__init__.py`
- [ ] `src/data/storage/timeseries_store.py` - TimescaleDB interface with hypertables
- [ ] `src/data/storage/feature_store.py` - Redis-backed feature store for low-latency inference
- [ ] `src/data/storage/cache.py` - Redis caching layer
- [ ] Create Alembic migrations for OHLCV hypertables, orders, strategies, portfolio tables

### Feature Engineering
- [ ] `src/features/__init__.py`
- [ ] `src/features/registry.py` - feature registry with dependency DAG and topological sort
- [ ] `src/features/technical.py` - TA-Lib wrappers: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Ichimoku (200+ indicators)
- [ ] `src/features/price.py` - log returns, realized volatility, volume profiles, gap features, relative range
- [ ] `src/features/pipeline.py` - FeatureEngineeringPipeline orchestrator (fit/transform/fit_transform)

### Data Pipeline
- [ ] `src/data/pipeline.py` - orchestrates ingest -> normalize -> transform -> store

### Phase 1 Verification
- [ ] Docker services start (Postgres+TimescaleDB, Redis)
- [ ] Import all core types successfully
- [ ] Fetch OHLCV data via yfinance and store in TimescaleDB
- [ ] Compute technical + price features on sample data
- [ ] Feature registry resolves dependency order correctly

---

## Phase 2: Backtest Engine + First Strategy

### Trading Strategies (Base)
- [ ] `src/strategies/__init__.py`
- [ ] `src/strategies/base.py` - BaseStrategy ABC (generate_signals, get_required_features)
- [ ] `src/strategies/signal.py` - Signal dataclass (symbol, direction, strength, confidence, metadata) + signal combiner
- [ ] `src/strategies/momentum.py` - trend-following strategy (MA crossovers, ADX, breakout detection)

### Risk Management
- [ ] `src/risk/__init__.py`
- [ ] `src/risk/manager.py` - RiskManager with pre-trade + portfolio-level checks
- [ ] `src/risk/position_sizer.py` - Kelly criterion (fractional 0.25-0.5x), volatility-adjusted, fixed-fractional
- [ ] `src/risk/portfolio.py` - Portfolio state tracking, P&L computation
- [ ] `src/risk/limits.py` - drawdown limits (10%), daily loss (3%), exposure limits (200% gross), concentration limits (5% per position, 25% per sector)
- [ ] `src/risk/stop_loss.py` - trailing stops, ATR-based stops (2.5x multiplier), time-based exits (20 day max hold)

### Backtesting Framework
- [ ] `src/backtest/__init__.py`
- [ ] `src/backtest/engine.py` - event-driven BacktestEngine (shares Strategy/Risk/Feature code with live)
- [ ] `src/backtest/data_handler.py` - historical data replay from TimescaleDB
- [ ] `src/backtest/fill_simulator.py` - slippage (volume-dependent), commissions (per-trade/per-share), Almgren-Chriss market impact model
- [ ] `src/backtest/analytics.py` - Sharpe, Sortino, Calmar, max drawdown (depth+duration), win rate, profit factor, alpha, beta, information ratio
- [ ] `src/backtest/report.py` - HTML report generation with equity curve, drawdown chart, metrics table

### Scripts & Tests
- [ ] `scripts/run_backtest.py` - CLI for backtest execution
- [ ] `scripts/seed_historical.py` - bulk load historical data
- [ ] `tests/conftest.py` - shared fixtures
- [ ] `tests/unit/test_features.py` - feature computation correctness
- [ ] `tests/unit/test_risk_manager.py` - risk gate edge cases
- [ ] `tests/unit/test_position_sizer.py` - sizing calculations vs hand-computed values
- [ ] `tests/unit/test_strategies.py` - signal generation with synthetic data

### Phase 2 Verification
- [ ] Run backtest on SPY 2020-2024 with momentum strategy
- [ ] Verify Sharpe/drawdown metrics match manual calculation
- [ ] Verify fill simulator applies slippage correctly
- [ ] All unit tests pass with 80%+ coverage on risk and backtest modules

---

## Phase 3: ML Models + Ensemble

### Base Model Infrastructure
- [ ] `src/models/__init__.py`
- [ ] `src/models/base.py` - BasePredictor protocol (fit, predict, predict_proba, save, load, get_feature_importance)

### Gradient Boosting Models
- [ ] `src/models/tree/__init__.py`
- [ ] `src/models/tree/lightgbm_model.py` - 3 sub-models: direction classifier, return regressor, volatility regressor
- [ ] `src/models/tree/xgboost_model.py` - XGBoost alternative with same interface

### Deep Learning Models
- [ ] `src/models/deep/__init__.py`
- [ ] `src/models/deep/lstm_attention.py` - BiLSTM (2 layers) + self-attention + 3-head output (direction/return/vol)

### Transformer Models
- [ ] `src/models/transformer/__init__.py`
- [ ] `src/models/transformer/temporal_fusion.py` - TFT via pytorch-forecasting: multi-horizon forecasting (1d/2d/5d/10d/21d), variable selection attention, known/unknown future covariates

### Ensemble Meta-Learner
- [ ] `src/models/ensemble.py` - Ridge regression stacking: separate meta-learner per time horizon, dynamic weight adjustment (exponential decay by rolling Sharpe), 0.05 minimum weight floor

### Training Pipeline
- [ ] `src/models/training/__init__.py`
- [ ] `src/models/training/walk_forward.py` - walk-forward splits (504d train / 63d val / 21d test, step 21d)
- [ ] `src/models/training/trainer.py` - orchestrates full walk-forward training for all models + meta-learner
- [ ] `src/models/training/hyperopt.py` - Optuna with financial-specific pruning (prune if rolling Sharpe < threshold)
- [ ] `src/models/training/registry.py` - MLflow experiment tracking, model versioning, model registry (staging/production)

### ML-Driven Strategy
- [ ] `src/strategies/ml_alpha.py` - wraps ensemble predictor, translates prediction confidence into signal strength

### Scripts
- [ ] `scripts/train_models.py` - CLI for model training pipeline

### Tests
- [ ] `tests/unit/test_ensemble.py` - meta-learner weight updates, prediction combination
- [ ] `tests/integration/test_training_pipeline.py` - walk-forward doesn't leak future data

### Phase 3 Verification
- [ ] Train LightGBM on 2+ years of data, verify walk-forward integrity (test timestamps > train end)
- [ ] Train LSTM and TFT models, verify convergence on validation loss
- [ ] Verify ensemble outperforms best single model on held-out test
- [ ] MLflow UI shows all experiments with metrics and artifacts
- [ ] Backtest with ml_alpha strategy shows improvement over momentum alone

---

## Phase 4: Live Execution Engine

### Execution Engine
- [ ] `src/execution/__init__.py`
- [ ] `src/execution/engine.py` - main async trading loop (receive data -> update features -> inference -> signals -> risk check -> submit orders -> monitor fills -> update portfolio -> publish to WebSocket)
- [ ] `src/execution/order_manager.py` - order lifecycle (create, submit, partial fill, fill, cancel, reject)

### Broker Adapters
- [ ] `src/execution/brokers/__init__.py`
- [ ] `src/execution/brokers/base.py` - BrokerAdapter protocol (submit_order, cancel_order, get_positions, get_account, subscribe_order_updates)
- [ ] `src/execution/brokers/alpaca_broker.py` - Alpaca paper + live trading (controlled by APCA_API_BASE_URL)
- [ ] `src/execution/brokers/simulated_broker.py` - in-memory simulated fills for local development

### Real-Time Data Feed
- [ ] `src/data/feed/__init__.py`
- [ ] `src/data/feed/realtime_feed.py` - WebSocket multiplexer aggregating multiple providers
- [ ] `src/data/feed/bar_aggregator.py` - tick-to-bar aggregation (1m/5m/15m/1h configurable)

### Composite Strategy
- [ ] `src/strategies/composite.py` - multi-strategy combiner with configurable weights, confidence-weighted conflict resolution, minimum agreement threshold (60%)

### Tests
- [ ] `tests/integration/test_execution_flow.py` - signal -> risk -> order -> fill -> portfolio using SimulatedBroker
- [ ] `tests/e2e/test_paper_trading.py` - connect to Alpaca paper, submit real paper order, verify fill

### Phase 4 Verification
- [ ] Connect to Alpaca paper trading API successfully
- [ ] Submit a market order, receive fill callback, verify portfolio state update
- [ ] Real-time WebSocket feed receives live price data
- [ ] Execution engine runs complete loop: data -> features -> model -> signal -> order
- [ ] SimulatedBroker passes all integration tests

---

## Phase 5: Web Dashboard

### FastAPI Backend
- [ ] `src/api/__init__.py`
- [ ] `src/api/app.py` - FastAPI application factory
- [ ] `src/api/deps.py` - dependency injection (DB sessions, Redis, services)
- [ ] `src/api/middleware.py` - authentication, CORS, rate limiting
- [ ] `src/api/schemas/__init__.py` + `portfolio.py` + `order.py` + `strategy.py` + `backtest.py` - Pydantic response models

### API Routers
- [ ] `src/api/routers/__init__.py`
- [ ] `src/api/routers/dashboard.py` - GET /api/dashboard (portfolio summary, P&L, positions)
- [ ] `src/api/routers/strategies.py` - CRUD strategies, POST toggle enable/disable
- [ ] `src/api/routers/orders.py` - order history with filters, POST manual order
- [ ] `src/api/routers/backtest.py` - POST run backtest, GET results
- [ ] `src/api/routers/models.py` - model performance, accuracy, contribution metrics
- [ ] `src/api/routers/risk.py` - current risk metrics, limit overrides
- [ ] `src/api/routers/websocket.py` - real-time streaming via Redis pub/sub (prices, portfolio, orders, signals)

### React Frontend Setup
- [ ] `frontend/package.json` - React 18 + TypeScript + Vite
- [ ] `frontend/vite.config.ts`
- [ ] `frontend/tsconfig.json`
- [ ] `frontend/tailwind.config.ts` - Tailwind CSS + shadcn/ui setup
- [ ] `frontend/src/main.tsx` + `frontend/src/App.tsx`
- [ ] `frontend/src/api/client.ts` - Axios wrapper + WebSocket hook
- [ ] `frontend/src/stores/useTradeStore.ts` - Zustand state management

### Dashboard Components
- [ ] `frontend/src/components/Layout.tsx` - app shell with navigation
- [ ] `frontend/src/components/charts/PriceChart.tsx` - TradingView Lightweight Charts candlestick
- [ ] `frontend/src/components/charts/EquityCurve.tsx` - portfolio value over time
- [ ] `frontend/src/components/charts/DrawdownChart.tsx` - underwater equity chart
- [ ] `frontend/src/components/charts/HeatMap.tsx` - correlation / sector heat map
- [ ] `frontend/src/components/portfolio/PositionsTable.tsx` - open positions with unrealized P&L
- [ ] `frontend/src/components/portfolio/PnLSummary.tsx` - daily/weekly/monthly P&L cards
- [ ] `frontend/src/components/portfolio/AllocationDonut.tsx` - portfolio allocation chart
- [ ] `frontend/src/components/strategies/StrategyCard.tsx` - status, signals, toggle
- [ ] `frontend/src/components/strategies/StrategyConfig.tsx` - parameter adjustment
- [ ] `frontend/src/components/risk/RiskGauge.tsx` - visual risk level indicator
- [ ] `frontend/src/components/risk/ExposureBar.tsx` - exposure by sector/asset class
- [ ] `frontend/src/components/orders/TradeLog.tsx` - trade history with filters

### Pages
- [ ] `frontend/src/pages/Dashboard.tsx` - main dashboard with equity curve, P&L, positions, allocation, risk gauges
- [ ] `frontend/src/pages/Strategies.tsx` - strategy cards with config and toggle
- [ ] `frontend/src/pages/Backtest.tsx` - form to configure/run backtests, results display
- [ ] `frontend/src/pages/Models.tsx` - ML model performance dashboard
- [ ] `frontend/src/pages/Orders.tsx` - trade log with CSV export
- [ ] `frontend/src/pages/Settings.tsx` - API keys, risk limits, notification preferences

### Phase 5 Verification
- [ ] `uvicorn` starts FastAPI backend, OpenAPI docs accessible at /docs
- [ ] `vite dev` starts frontend, dashboard page loads
- [ ] WebSocket connection established, real-time data streaming
- [ ] Backtest page triggers a run and displays results with charts
- [ ] All API endpoints return correct data with proper Pydantic validation

---

## Phase 6: Advanced Features

### Reinforcement Learning Agent
- [ ] `src/models/rl/__init__.py`
- [ ] `src/models/rl/environment.py` - Gym-compatible TradingEnv (state: signal+confidence+position+pnl+vol, action: continuous [-1,1], reward: risk-adjusted return)
- [ ] `src/models/rl/reward.py` - Sharpe-based, Sortino-based, and asymmetric reward functions
- [ ] `src/models/rl/agent.py` - SAC agent via Stable-Baselines3 for position sizing and execution timing
- [ ] `src/strategies/rl_strategy.py` - wraps RL agent as a strategy

### Sentiment & Alternative Data
- [ ] `src/models/sentiment/__init__.py`
- [ ] `src/models/sentiment/finbert.py` - FinBERT (ProsusAI/finbert) for financial news sentiment scoring
- [ ] `src/data/alternative/__init__.py`
- [ ] `src/data/alternative/sentiment.py` - news headline scraping, social media aggregation
- [ ] `src/data/alternative/dark_pool.py` - FINRA dark pool volume data
- [ ] `src/features/sentiment_features.py` - rolling sentiment averages, momentum, dispersion
- [ ] `src/features/cross_asset.py` - VIX features, yield curve features, sector rotation signals

### Market Microstructure
- [ ] `src/data/feed/order_book.py` - L2 order book reconstruction from WebSocket
- [ ] `src/features/microstructure.py` - bid-ask spread, order book imbalance, VPIN, Kyle's lambda
- [ ] `src/data/providers/options_flow.py` - Unusual Whales / CBOE options flow data

### Additional Strategies
- [ ] `src/strategies/mean_reversion.py` - Bollinger deviation, z-score, RSI extremes
- [ ] `src/strategies/options_flow.py` - unusual options activity detection, smart money flow signals
- [ ] `src/strategies/pairs_trading.py` - Engle-Granger cointegration, spread z-score entry/exit

### Portfolio Optimization
- [ ] `src/risk/optimizer.py` - Black-Litterman model (CAPM equilibrium + ML views -> optimal allocation), weekly rebalancing

### Advanced Execution
- [ ] `src/execution/smart_router.py` - TWAP, VWAP, iceberg order execution algorithms
- [ ] `src/execution/brokers/ibkr_broker.py` - Interactive Brokers TWS/Gateway for futures and options

### Model Monitoring & Auto-Retrain
- [ ] `src/models/training/monitoring.py` - feature drift (KS-test), prediction drift, performance decay (Sharpe < 0.5), regime change (VIX > 25) via evidently library
- [ ] `src/models/training/ab_testing.py` - A/B framework for comparing model versions before promotion

### Production Deployment
- [ ] `docker-compose.prod.yml` - Gunicorn+Uvicorn, Nginx, Celery worker+beat, persistent volumes
- [ ] `deploy/nginx.conf` - reverse proxy, SSL termination
- [ ] `deploy/supervisord.conf` - process management for trading engine

### Phase 6 Verification
- [ ] RL agent trains in custom environment, produces reasonable position sizes
- [ ] FinBERT sentiment features improve ensemble accuracy on validation set
- [ ] Order book features compute from live L2 data
- [ ] IBKR adapter connects and fetches futures quotes
- [ ] Drift monitor correctly flags synthetic distribution shifts
- [ ] Production Docker stack starts all services cleanly

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
