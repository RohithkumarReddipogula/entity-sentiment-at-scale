# entity-sentiment-at-scale: SPEC

## One-line goal

Fine-tune a small Transformer to predict the sentiment toward each entity in a news sentence, compare two architectures (one model run per entity vs one run per document) and an LLM baseline on accuracy, speed and cost, and serve the best model behind a FastAPI endpoint. Everything measured honestly on a held-out test set.

## Why this project exists

The owner's portfolio shows LLM agents, RAG and evaluation, but no model he trained himself. NLP roles (for example data enrichment teams) ask for exactly that: fine-tuning a pretrained model for token or text classification, and measuring the impact of algorithm changes. This repo is the proof.

Inspiration: Meltwater Engineering, "Doing More With Less: Rethinking Entity-Level Sentiment at Scale" (April 2026). Their v1 ran the model once per entity; their v2 encoded the document once, pooled each entity's mention embeddings and classified them, cutting inference cost by 45.5% and improving accuracy by 3.02%. We reproduce the idea in small on a public dataset and report our own numbers, whatever they are.

## The task

Input: a sentence or headline plus the entities mentioned in it (with character spans).
Output: for each entity, one label from {positive, negative, neutral}.

Example: "Nike beats forecasts while Adidas cuts its outlook." gives Nike = positive, Adidas = negative.

## Dataset (M0 decides)

Pick ONE public dataset with entity-level or aspect-level sentiment labels. Candidates, in order of preference:

1. FinEntity (entity-level sentiment in financial news, Hugging Face)
2. SEntFiN 1.0 (entity-level sentiment in financial news headlines)
3. SemEval-2014 Task 4 aspect-based sentiment (restaurants or laptops)

Rules for M0:
- Verify the license allows public use and redistribution of derived results. Record the license and source URL in DATASET.md. If redistribution of the raw data is not allowed, the repo downloads it with a script instead of committing it.
- Report size, label distribution and how many sentences have two or more entities (this is where the two architectures differ most).
- Create a fixed split: train / dev / test (for example 70/15/15), stratified by label, split by sentence so no sentence appears in two splits. Save the split IDs in the repo. The test split is not touched until M6.

## Models to compare

| ID | Name | How it works |
|---|---|---|
| A | LLM baseline | Zero-shot prompt to Claude Haiku via OpenRouter; returns JSON with one label per entity. Measures what "just use an LLM" gives. |
| B | Per-entity encoder | Fine-tuned encoder. Input = sentence with the target entity wrapped in markers, for example `[E] Nike [/E]`. One forward pass per entity. Classify from the [CLS] token. |
| C | Single-pass encoder | Same base encoder. Encode the sentence once, take the token embeddings covering each entity mention, mean-pool them per entity (average across repeated mentions), classify with a small head. One forward pass per sentence. |

Base encoder: a small model that trains in under about an hour on a free Colab T4 or an Apple Silicon Mac (MPS), for example `distilroberta-base` or `microsoft/deberta-v3-small`. Pick in M2 based on a quick dev-set check, then keep it fixed for B and C so the comparison is fair.

## Metrics (fixed before any results)

- Primary: macro-F1 over the three classes on the test set.
- Secondary: accuracy, per-class precision/recall/F1, macro-F1 on the subset of sentences with 2+ entities, and macro-F1 on the mixed-label subset (sentences whose entities do not all share the same label; 1,233 headlines in SEntFiN 1.0). The two subsets are reported side by side.
- Cost and speed: forward passes per 1,000 sentences, median and p95 latency per sentence on CPU, throughput (sentences per second) on CPU with batching, and for model A the USD cost per 1,000 sentences.
- Uncertainty: 95% bootstrap confidence intervals (10,000 resamples) for macro-F1 of each model, and a paired test (McNemar on per-entity correctness) for B vs C.

Decision rule, written now: C is preferred over B if its macro-F1 is not significantly worse (95% CI of the difference includes zero or favors C) AND it is cheaper per sentence. Otherwise B. The LLM baseline is a reference point, not a candidate for serving.

## Milestones

Each milestone ends with: tests passing, a short entry in RESULTS.md or LOG.md, and a commit the owner approves.

- M0 Dataset: choose dataset, check license, DATASET.md, loader, fixed splits, data tests (no split leakage, all spans valid).
- M1 Baseline A: OpenRouter client (key from .env only), JSON output validation, retries, run on DEV only, record macro-F1 and cost.
- M2 Training setup: shared training script with config files (YAML), seeds fixed, Hugging Face Trainer or plain PyTorch, runs on CPU, MPS or CUDA. Quick dev check to pick the base encoder.
- M3 Model B: per-entity encoder trained, dev results, saved checkpoint.
- M4 Model C: single-pass encoder with mention pooling trained, dev results, saved checkpoint. Unit tests for the span-to-token alignment (this is where bugs hide).
- M5 Speed and cost benchmark: one script that measures passes, latency and throughput for B and C on CPU on the same machine; results table.
- M6 Final test run: run A, B, C once on the untouched test set, bootstrap CIs, McNemar, error analysis (at least 30 wrong predictions read by hand, grouped into patterns). Apply the decision rule.
- M7 Serving: FastAPI endpoint `POST /predict` (sentence + entity spans in, labels and confidences out), Dockerfile, health check, input validation, a small CI-friendly model (or mocked model) for tests.
- M8 Polish: README with the results table, a plain-language "what I learned" section, honest limitations, model card. Optional: push the chosen model to the Hugging Face Hub, deploy the API.

## Engineering rules

- Python 3.12, typed code, ruff for lint, pytest for tests.
- GitHub Actions CI: lint + unit tests on every push. No GPU or API calls in CI; training is never run in CI.
- Every experiment writes a JSON result file with config, seed, git commit, metrics and timestamp to `results/`. RESULTS.md is generated or updated from those files, never typed by hand.
- Reproducible: `make data`, `make train-b`, `make train-c`, `make bench`, `make test-eval` (or equivalent scripts).
- Keep failed or worse runs in the log. Do not delete experiments that did not work.

## Security and working rules (non-negotiable)

- API keys only in `.env`, which is gitignored. Never print, cat or log `.env` or any key.
- Ask the owner before `git commit`, `git push`, `rm`, `sudo`, installing system packages, or touching anything outside the project folder.
- Do not commit datasets whose license forbids redistribution, model weights larger than 50 MB, or any personal data.
- The test split is used exactly once, in M6. If something must change after M6, it is reported as a new, clearly labeled run.

## Definition of done

A public GitHub repo where a reader sees, in the first screen of the README: the task, a results table (A vs B vs C: macro-F1 with CI, passes per 1,000 sentences, CPU latency, cost), the decision made and why, and how to run the API. All numbers come from files in `results/`.
