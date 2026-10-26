.PHONY: setup data data-check lint typecheck test check run-a-smoke run-a results check-encoders train-b train-c alignment bench serve smoke-api docker-reqs docker-build docker-run fetch-model

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
# M3: final model B, seeds 1-3, deterministic on CPU, checkpoints saved (dev only).
train-b:
	rc=0; \
	for seed in $(SEEDS); do \
		uv run python scripts/train.py --config configs/b_final.yaml --seed $$seed \
			--purpose final || { echo "FAILED: b_final seed $$seed"; rc=1; }; \
	done; \
	uv run python scripts/make_results.py; \
	exit $$rc

# M4: final model C, seeds 1-3, deterministic on CPU, checkpoints saved (dev only).
train-c:
	rc=0; \
	for seed in $(SEEDS); do \
		uv run python scripts/train.py --config configs/c_final.yaml --seed $$seed \
			--purpose final || { echo "FAILED: c_final seed $$seed"; rc=1; }; \
	done; \
	uv run python scripts/make_results.py; \
	exit $$rc

alignment:
	uv run python scripts/check_alignment.py

# M5: CPU speed and cost of B and C (seed 1), interleaved fresh processes, dev headlines.
bench:
	uv run python scripts/bench.py
	uv run python scripts/make_results.py

# M7: serve the chosen model (C seed 1). Weights are mounted, never baked into the image.
SERVED := c_final_s1_20261025T160107Z
SERVED_SHA := $(shell python3 -c "import json; print(json.load(open('results/$(SERVED).json'))['checkpoint']['model_sha256'])")
IMAGE := entity-sentiment-api:m7

serve:
	MODEL_DIR=checkpoints/$(SERVED) MODEL_SHA256=$(SERVED_SHA) HF_HUB_OFFLINE=1 \
		uv run uvicorn entsent.api:app --host 127.0.0.1 --port 8000

smoke-api:
	uv run python scripts/smoke_api.py

docker-reqs:
	uv export --format requirements-txt --no-default-groups --group serve --no-emit-project --no-hashes \
		--no-annotate --no-header | grep -vE '^(torch|nvidia-|triton|cuda-)' \
		> docker/requirements-serve.txt

docker-build:
	docker build -t $(IMAGE) .

# M9: the image contains the served model (downloaded and verified at build time).
docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

# M9: download the served checkpoint from its GitHub Release and verify both SHA-256 values.
fetch-model:
	python3 docker/fetch_model.py \
		"$$(python3 -c "import json; print(json.load(open('deploy/model.json'))['url'])")" \
		"$$(python3 -c "import json; print(json.load(open('deploy/model.json'))['archive_sha256'])")" \
		"$$(python3 -c "import json; print(json.load(open('deploy/model.json'))['model_pt_sha256'])")" \
		checkpoints/$(SERVED)

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
