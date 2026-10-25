.PHONY: setup data data-check lint typecheck test check run-a-smoke run-a results check-encoders

SEEDS ?= 1 2 3

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

# M2: model B with both candidate encoders, 3 seeds each, dev only.
# Every run is attempted; RESULTS.md is regenerated even if one fails, then make fails.
check-encoders:
	status=0; \
	for config in b_distilroberta b_deberta; do \
		for seed in $(SEEDS); do \
			uv run python scripts/train.py --config configs/$$config.yaml --seed $$seed \
				--purpose encoder_check || { echo "FAILED: $$config seed $$seed"; status=1; }; \
		done; \
	done; \
	uv run python scripts/make_results.py; \
	exit $$status
