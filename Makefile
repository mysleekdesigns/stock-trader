.PHONY: setup test lint format run docker-up docker-down migrate seed

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

run:
	uv run uvicorn src.api.app:create_app --factory --reload

docker-up:
	docker compose up -d

docker-down:
	docker compose down

migrate:
	uv run alembic upgrade head

seed:
	uv run python scripts/seed_historical.py
