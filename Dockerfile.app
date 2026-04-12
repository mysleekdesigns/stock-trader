# Stock Trader - Application Container
# Multi-stage build for production deployment

# --- Base stage ---
FROM python:3.12-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app

# --- Dependencies stage ---
FROM base AS deps

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# --- Runtime stage ---
FROM base AS runtime

COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
