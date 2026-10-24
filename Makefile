.PHONY: setup data data-check lint typecheck test check run-a-smoke run-a results

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

run-a-smoke:
	uv run python scripts/run_llm.py --split train --limit 20 --purpose smoke

run-a:
	uv run python scripts/run_llm.py --split dev

results:
	uv run python scripts/make_results.py
