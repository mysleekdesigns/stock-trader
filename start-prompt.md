Read PRD.md fully. Then implement Phase 1 (Foundation) using a team of sub-agents working in parallel.

Spawn these agents concurrently:

1. **Project Setup Agent** — Create pyproject.toml with UV, .env.example, Makefile, docker-compose.yml (Postgres+TimescaleDB, Redis), Dockerfile for backend, alembic.ini with initial migration setup.

2. **Config Agent** — Create config/__init__.py, config/settings.py (Pydantic Settings), config/logging_config.py (structlog), config/assets.yaml, config/strategies.yaml, config/risk_limits.yaml.

3. **Core Domain Agent** — Create src/__init__.py, src/core/__init__.py, src/core/types.py (enums + dataclasses), src/core/events.py (async pub/sub event bus), src/core/interfaces.py (Protocol definitions), src/core/exceptions.py.

4. **Data Providers Agent** — Create src/data/__init__.py, src/data/providers/__init__.py, src/data/providers/base.py, src/data/providers/yfinance_provider.py, src/data/providers/alpaca_provider.py, src/data/providers/polygon_provider.py.

5. **Data Storage Agent** — Create src/data/storage/__init__.py, src/data/storage/timeseries_store.py (TimescaleDB), src/data/storage/feature_store.py (Redis), src/data/storage/cache.py, and Alembic migration files.

6. **Feature Engineering Agent** — Create src/features/__init__.py, src/features/registry.py (DAG + toposort), src/features/technical.py (TA-Lib wrappers), src/features/price.py, src/features/pipeline.py.

7. **Data Pipeline Agent** — Create src/data/pipeline.py that orchestrates ingest -> normalize -> transform -> store.

After all agents complete, integrate their work: resolve any import issues, run `uv run python -c "from src.core.types import *; from src.core.events import *; print('Core imports OK')"` to verify, and run any tests.

Then proceed to Phase 2 with the same parallel sub-agent strategy.
