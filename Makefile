.PHONY: setup data data-check lint typecheck test check

setup:
	uv sync

data:
	uv run python scripts/make_data.py

data-check:
	uv run python scripts/make_data.py --check

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src scripts tests

test:
	uv run pytest -q

check: lint typecheck test
