# entity-sentiment-at-scale

Given a financial news headline and the companies or other entities in it, the task is to say
whether the news is positive, negative or neutral for each entity. I fine-tuned a small encoder
two ways, compared both with a zero-shot LLM on accuracy and cost, and serve the cheaper one
behind an API.

## Results

<!-- BEGIN GENERATED: results -->
Test set: 1,611 headlines, 2,157 entities, used once. B and C are the mean of 3 training seeds.

| Model | Test macro-F1 [95% CI] | 2+ entities | Mixed labels | Forward passes / 1k headlines | CPU s / 1k headlines | Latency median / p95 | Cost / 1k headlines |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: zero-shot LLM (anthropic/claude-haiku-5.5) | 0.837 [0.820, 0.854] | 0.837 | 0.882 | 1,000 API calls | remote | 1,112 / 1,670 ms (network, dev) | $0.081 |
| B: per-entity encoder | 0.883 [0.868, 0.897] | 0.879 | 0.869 | 1,333 | 3.78 | 5.7 / 8.2 ms | not priced (CPU) |
| C: single-pass encoder | 0.882 [0.867, 0.896] | 0.885 | 0.874 | 1,000 | 2.40 | 5.8 / 6.0 ms | not priced (CPU) |
| C seed 1 (the served model) | 0.883 [0.867, 0.898] | 0.885 | 0.876 | | | | |

CPU time and latency: Apple M5, macOS, 4 torch threads; CPU time in batches of 32 headlines, latency one headline at a time, end to end. These numbers depend on the machine (see Limitations).

**Decision: C.** The rule was written before any test result: choose C if its macro-F1 is not significantly worse than B's and it is cheaper. C minus B is -0.0009 with a 95% CI of [-0.0085, +0.0065], which includes zero. C uses 2.40 CPU seconds per 1,000 headlines against B's 3.78, 37% less. So C is served, as C seed 1, the seed with the best dev score.
<!-- END GENERATED: results -->

The full tables, including every seed, both subsets, McNemar tests and the error analysis, are in
[RESULTS.md](RESULTS.md). Every number in this README and in RESULTS.md is generated from the JSON
files in [`results/`](results/) by `make results`.

## Run the API

The API serves model C. The trained weights (317 MB) are not in the git repository. They are
attached to the GitHub Release `model-c-s1-v1`, and [deploy/model.json](deploy/model.json)
records their SHA-256. `make fetch-model` and the Docker build download them and check the
hashes. You can also train them yourself with `make data` and `make train-c` (about 15 minutes
on CPU). Training is deterministic on CPU; I checked that retraining gives bit-identical weights
for model B seed 1.

```bash
make setup                   # Python 3.12 environment with uv
make fetch-model             # download and verify the served weights
make serve                   # local server with the web page on http://127.0.0.1:8000
# or, with Docker (the build downloads and verifies the weights):
make docker-build && make docker-run
```

The page at `/` lets you type a headline and entity names and shows each entity's label and
confidence. It calls `POST /predict_text`, which finds the names in the headline with the same
rules used to prepare the training data.

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "text": "Nike beats forecasts while Adidas cuts its outlook.",
  "entities": [{"name": "Nike", "spans": [[0, 4]]}, {"name": "Adidas", "spans": [[27, 33]]}]
}'
```

Each entity gets a label, a confidence and the three class probabilities. Spans are character
offsets `[start, end)`, and an entity can have several mentions. `GET /health` reports the model
name and the SHA-256 of its weights. Bad input gets a 422 with a reason. The server never logs
request bodies.

## What was compared

The idea comes from Meltwater's engineering post
[Doing More With Less: Rethinking Entity-Level Sentiment at Scale](https://underthehood.meltwater.com/blog/2026/04/23/rethinking-entity-level-sentiment-at-scale/)
(Tawhid, Schulte, Vardhan, Marton and Komban, April 2026). Their first system ran the model once
per entity. Their second encodes the document once and averages the embeddings of each entity's
mentions. They report 45.5% lower inference cost and 3.02% higher accuracy. I rebuilt the idea at
small scale on a public dataset and measured my own numbers.

- **A, zero-shot LLM.** Claude Haiku 5.5 through OpenRouter, with a fixed prompt and strict JSON
  output. No examples, no tuning. It is a reference point, not a candidate for serving.
- **B, per-entity encoder.** distilroberta-base, fine-tuned. The target entity is wrapped in
  `[E] ... [/E]` markers and the model classifies from the first token. One forward pass per
  entity.
- **C, single-pass encoder.** The same encoder and the same training settings. The headline is
  encoded once. For each entity, the token vectors of each mention are averaged, then the
  mentions are averaged, and the same small head classifies the result. One forward pass per
  headline.

Mapping character spans to tokens is where bugs hide, so it has its own tests. A token belongs
to a mention when their character ranges overlap. On train and dev, 99.6% of mentions line up
exactly with token boundaries, and no mention is left without a token.

## Data

[SEntFiN 1.0](https://github.com/pyRis/SEntFiN): 10,753 financial news headlines with 14,405
entity-level labels. 2,847 headlines have two or more entities, and in 1,233 of them the
entities do not all share one label. It is released under the MIT license, Copyright (c) 2021
Ankur Sinha, Satishwar Kedas, Rishu Kumar, Pekka Malo.

> Sinha, A., Kedas, S., Kumar, R., and Malo, P. (2022). SEntFiN 1.0: Entity-aware sentiment
> analysis for financial news. *Journal of the Association for Information Science and
> Technology*, 73(9), 1314-1335. https://doi.org/10.1002/asi.24634 (preprint:
> [arXiv:2305.12257](https://arxiv.org/abs/2305.12257))

The headlines themselves were written by news publishers, so this repository does not include
the raw file. `make data` downloads it at a pinned commit and checks its SHA-256. The repository
keeps only headline IDs for the fixed train, dev and test split (70/15/15), plus derived
numbers. [DATASET.md](DATASET.md) has the details, including every repair the loader makes.

## What I learned

- **Writing the rules down first changed real decisions.** The encoder rule, the cost rule and
  the test protocol were all fixed before the results they decide. With a 1-point threshold I
  would have picked a different encoder. Having the rule on paper made that a choice I made in
  advance, not after looking.
- **One pass per headline was as accurate as one pass per entity, and cheaper.** I expected the
  single-pass model to do better on headlines with several entities. It did not, on dev or on
  test. The gain is in cost, not accuracy.
- **The fine-tuned models beat the LLM mainly on the neutral class.** In the error analysis, the
  LLM was right on half of C's mistakes: finance jargon, implied consequences and comparisons.
  It was almost never right where SEntFiN's idea of "neutral" differs from plain reading. The
  small models learned the dataset's conventions.
- **Check the dtype of what you load.** deberta-v3-small is stored in float16. Loaded next to a
  float32 head, it crashed on Apple MPS with a Metal assertion. Worse, my first speed benchmark
  had measured it in float16 without my noticing. That made it look 7 times slower than
distilroberta-base on CPU. In float32 it is about 2.4 times slower.
- **Speed numbers belong to a machine.** On macOS, one torch thread was the most efficient.
  In a Linux Docker container on the same Mac, the same model was about 12 times slower with one
  thread and about 3 times faster with four. The maths library behind torch differs between
  the two.
- **Deterministic CPU training is worth the extra minutes.** Retraining a final model from its
  seed gave bit-identical weights. That made it easy to show the checkpoints match the code.

## Limitations

- **A threshold picked the encoder.** On dev, deberta-v3-small beat distilroberta-base by 1.47
  points, which is more than seed noise. The rule I set required 2 points because DeBERTa is
  slower on CPU, so distilroberta-base was chosen. A 1-point rule would have chosen DeBERTa.
- **The best epoch was always the last one.** All six final runs peaked at epoch 3 of 3. More
  epochs might help both B and C. I did not try, to avoid tuning on dev.
- **The speed numbers are from macOS on an Apple M5.** They used Apple's Accelerate library. In
  the Linux Docker image the absolute latency is different (about 23 ms per request with 4
  threads), so the image default is 4 threads and 1 worker. B and C were measured under the same
  conditions, so the comparison between them holds. I did not measure x86 Linux.
- **SEntFiN has annotation errors.** Some entity names are cut short ("Alibab", "Australi") or
  were changed by a find-and-replace ("HSBC neutralings" for "HSBC Holdings"; the loader
  repairs these). Some labels look wrong, for example a stock that "rallies 14%" labelled
  negative. Three of the 40 errors I read by hand looked like label errors, and 15 more were
  cases where both readings are defensible.
- **One dataset, one test set, one LLM run.** The headlines are mostly from Indian financial
  media. The LLM baseline ran once and has no temperature control, so a rerun can differ a
  little. Dev scores are slightly optimistic because the best epoch was chosen on dev.
- **No price for B and C.** Their cost is given as CPU time. Turning it into dollars would need
  a cloud price, which I left out.

## Reproduce every number

Each command writes JSON files to `results/`, and `make results` rebuilds RESULTS.md and the
table above from them. Times are for an Apple M5 Mac.

| Numbers | Command | Writes | Time, cost | Reproducible |
|---|---|---|---|---|
| Data, split, dataset stats | `make data` | `data/splits/`, DATASET.md | 1 min | exactly (checked by `make data-check`) |
| LLM baseline on dev (M1) | `make run-a` | `results/a_llm_dev_eval_*.json` | 5 min, about $0.13 | answers vary slightly; cached reruns are free |
| Encoder check (M2) | `make check-encoders` | `results/b_distilroberta_*`, `b_deberta_*` | 35 min (MPS) | close, not bit-exact (MPS) |
| Final B and C (M3, M4) | `make train-b`, `make train-c` | `results/b_final_*`, `c_final_*`, `checkpoints/` | 20 + 15 min (CPU) | bit-exact |
| Span alignment stats | `make alignment` | printed | 1 min | exactly |
| Speed and cost (M5) | `make bench` | `results/bench/` | 10 min | depends on the machine |
| Dev dry run of the test pipeline | `uv run python scripts/evaluate.py --split dev` | `results/dry_run/` | 3 min, $0 (cached) | exactly |
| Test results (M6) | `uv run python scripts/evaluate.py --split test --confirm-one-time-test-run` | `results/test/` | 4 min, about $0.13 | B and C exactly; A slightly |
| All tables | `make results` | RESULTS.md, this README | seconds | exactly |
| Lint, types, tests | `make check` | | 1 min | |

The test command refuses to run when `results/test/` already holds a record, because the test
set is meant to be used once. To recompute it in your own copy, remove that folder first. The
per-entity predictions of every model are in `results/test/predictions/`, so the statistics can
also be checked without retraining.

[LOG.md](LOG.md) records every step, including the runs that failed and the mistakes I made.

## Repository

- `src/entsent/`: data loading, both encoder models, training, metrics, the LLM client, the
  benchmark, the evaluation statistics and the API
- `scripts/`: one script per step above; `configs/`: the training configs
- `tests/`: unit, data and API tests; the ones that need the raw data, downloaded models or
  checkpoints are skipped in CI
- [SPEC.md](SPEC.md): the plan, the metrics and the decision rules, written before the results

The code is under the [MIT license](LICENSE). SEntFiN keeps its own MIT license, and
distilroberta-base its Apache-2.0 license.
