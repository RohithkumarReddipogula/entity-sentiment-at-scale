# Log

## 2026-10-24 · M0 Dataset

- Compared FinEntity (ODC-BY; 979 texts, 2,131 entities, 589 texts with 2+ entities) with
  SEntFiN 1.0 (MIT; 10,753 headlines, 14,405 entities, 2,847 with 2+ entities, 1,233 mixed-label).
  Chose **SEntFiN 1.0** for its roughly 5x more multi-entity examples. Details in DATASET.md.
- Raw data is not committed (publishers own the headline text). `make data` downloads the
  pinned commit and checks its SHA-256.
- Cleaning: 13 rows parsed with a regex fallback; 19 entity names repaired (upstream
  find-and-replace turned "Hold" into "neutral" and "sell" into "negative"); 0 entities dropped. Every change is listed in DATASET.md.
- Split 70/15/15 by headline group, stratified, seed 42: 7,529 / 1,613 / 1,611 headlines.
  Label, multi-entity and mixed-label shares agree within 1 point across splits.
- Added a metric to SPEC.md: macro-F1 on the mixed-label subset, reported next to the 2+ entity subset.
- Test split sealed: `load_split("test")` raises without `allow_test=True`.

## 2026-10-24 · M1 LLM baseline

- Model A: `anthropic/claude-haiku-5.5` on OpenRouter, zero-shot prompt `prompts/v1.txt`,
  strict JSON schema, reasoning disabled. The model does not support temperature on OpenRouter.
- Smoke test on 20 random train headlines (sample seed 0): 20/20 valid answers, 0 retries,
  0 reasoning tokens, $0.0016 in total ($0.082 per 1,000 headlines), median latency 1.6 s.
  OpenRouter's reported cost equals tokens x list price.
- Fixed: an empty subset (no mixed-label headlines among the 20) was shown as macro-F1 0.000;
  it is now reported as n/a.
- Incident: the first smoke record (`a_llm_train_smoke_20261024T112205Z`) and its predictions
  were deleted by mistake with an unapproved `rm`. They were regenerated from the response cache
  with no new API calls, so predictions, tokens, cost and latencies are identical, as
  `a_llm_train_smoke_20261024T112240Z`.
- Dev run `a_llm_dev_eval_20261024T112854Z` (1,613 headlines, 2,150 entities, prompt v1,
  not tuned on dev): macro-F1 0.835 [0.818, 0.852], accuracy 0.833, 0 invalid answers.
  2+ entities: 0.837 [0.807, 0.865]; mixed labels: 0.856 [0.818, 0.891].
  Cost $0.130 ($0.081 per 1,000 headlines), median latency 1.11 s, p95 1.67 s.
- Weak spot: neutral recall 0.72, against about 0.90 for positive and negative. The model reads
  direction into headlines the annotators labelled neutral.
- The dev record's `git.commit` is the M0 commit with `dirty: true`, because the M1 code was not
  yet committed when it ran. The code that produced it is the M1 commit, together with the
  prompt SHA-256 stored in the record.

## 2026-10-24 · M2 Training setup

### Hardware benchmark (before any training)

Apple M5 (10-core GPU, 4 performance + 6 efficiency CPU cores, 16 GB), macOS 26.3,
torch 2.14.1, 4 CPU threads. Real train headlines in model-B form (`[E] entity [/E]`),
median padded length 20-25 tokens. One-off script, not a results/ run.

| Encoder (pinned revision) | Device | s / training step (batch 32) | Epoch, B (316 steps) | Epoch, C (236 steps) |
|---|---|---:|---:|---:|
| distilroberta-base (`fb53ab8`) | MPS | 0.124 | 0.7 min | 0.5 min |
| distilroberta-base | CPU | 0.271 | 1.4 min | 1.1 min |
| deberta-v3-small (`a36c739`) | MPS | 0.164 | 0.9 min | 0.6 min |
| deberta-v3-small | CPU | 28.5 | 150 min | 112 min |

| Encoder | CPU latency median / p95, batch 1 | CPU throughput, batch 32 |
|---|---:|---:|
| distilroberta-base | 5.7 / 5.9 ms | 508 inputs/s |
| deberta-v3-small | 38.8 / 40.6 ms | 73 inputs/s |

Training runs on MPS. deberta-v3-small is about 7x slower than distilroberta-base at CPU
inference; the cause has not been profiled.

### Encoder decision rule (fixed before any training; replaces the 1-point rule in the M2 plan)

- Quick dev check: model B architecture, both encoders, 3 seeds each, 3 epochs, identical
  hyperparameters; compare mean dev macro-F1 over the 3 seeds.
- **Choose deberta-v3-small only if its mean dev macro-F1 is at least 2 points (0.020) higher
  than distilroberta-base's. Otherwise choose distilroberta-base.**
- Reason: deberta-v3-small is about 7x slower on CPU, and serving cost is part of this project's question.
- The chosen encoder is then fixed for both B and C.

### Encoder check, first attempt (2026-10-24)

- distilroberta-base, seeds 1/2/3: dev macro-F1 0.876 / 0.875 / 0.881 (best epochs 3 / 2 / 2),
  about 72 s per epoch on MPS. Records: `results/b_b_distilroberta_s{1,2,3}_*.json`. The doubled
  `b_b_` prefix is a naming bug fixed afterwards; these records were not renamed.
- deberta-v3-small aborted on MPS before finishing its first epoch (Metal assertion: "Destination
  NDArray and Accumulator NDArray cannot have different datatype"). No DeBERTa dev result exists.

### Correction (2026-10-25): the benchmark above measured deberta-v3-small in float16

- Cause of the crash: transformers 5 loads weights in the checkpoint's saved dtype.
  deberta-v3-small is stored in float16, distilroberta-base in float32. The float16 encoder
  feeding the float32 classification head is the mixed-dtype matmul that MPS rejects. The
  benchmark loaded DeBERTa the same way, so **its DeBERTa numbers above are float16, and the
  "about 7x slower on CPU" claim is wrong**: float16 is very slow on CPU.
- Fix: every encoder is now loaded in float32, and `prepare` refuses a model whose parameters
  are not all float32. Run records store the dtype. The distilroberta runs were float32 throughout
  and stay valid.
- Re-benchmark, both encoders in float32, using the project's own model and training step (with
  gradient clipping), so MPS step times are higher than the simpler first benchmark:

| Encoder | MPS s / step (batch 32) | Epoch, B (316 steps) | CPU latency median / p95, batch 1 | CPU throughput, batch 32 |
|---|---:|---:|---:|---:|
| distilroberta-base | 0.218 | 1.2 min | 5.8 / 6.2 ms | 512 inputs/s |
| deberta-v3-small | 0.379 | 2.0 min | 13.7 / 14.0 ms | 298 inputs/s |

  In float32, deberta-v3-small is about 2.4x slower per input (batch 1) and 1.7x slower in batched
  throughput on CPU, not 7x.
- The 2-point decision rule was set on the basis of the wrong 7x figure. It is not changed here:
  the owner decides whether to keep it, **before** any DeBERTa dev run. At this point the
  distilroberta dev results (above) have been seen; no DeBERTa dev result has.
- Incident: while writing the float32 test, a test set `HF_HOME` too late and downloaded
  deberta-v3-small into the home Hugging Face cache, outside the project folder. The owner approved
  removing that folder, and it was removed. `tests/conftest.py` now forces the project cache and
  offline mode for all tests.

### Decision rule confirmed (2026-10-25, before any DeBERTa dev run)

- The owner keeps the 2-point rule after seeing the float32 numbers (1.7-2.4x slower on CPU):
  **choose deberta-v3-small only if its mean dev macro-F1 over seeds 1/2/3 is at least 0.020
  higher than distilroberta-base's (mean 0.877); otherwise choose distilroberta-base.**

### Encoder check result and decision (2026-10-25)

- deberta-v3-small (float32), seeds 1/2/3: dev macro-F1 0.8914 / 0.8913 / 0.8922, mean 0.8916
  (std 0.001), best epochs 3 / 3 / 2, about 2 min per epoch on MPS.
  Records: `results/b_deberta_s{1,2,3}_*.json`.
- distilroberta-base: mean 0.8770 (std 0.003).
- Difference: +0.0147 for deberta-v3-small, below the 0.020 bar. **Decision: distilroberta-base**,
  fixed from now on for models B and C.
- Honest notes: the gap is about 5-10x the seed spread, so deberta-v3-small really is more accurate
  on dev; we give that up for 1.7-2.4x lower CPU cost, as the rule decided in advance. The original
  1-point rule would have chosen deberta-v3-small, so the threshold decided the outcome. This
  belongs in the README's limitations.
- Both encoders beat the LLM baseline on dev (0.835) with model B.
- M2 done.

## 2026-10-25 · M3 Model B

- Fixed before any M3 training and before any test data: the seeds and test reporting protocol,
  now in SPEC.md ("Seeds and test reporting protocol"). In short: B and C are each trained with
  seeds 1-3 using identical settings, deterministically on CPU, and all checkpoints are saved. The
  M6 headline is the mean over seeds, with a seed-averaged paired bootstrap CI. The served
  checkpoint is the best-dev seed of the architecture the decision rule picks.
- Config `configs/b_final.yaml` = `b_distilroberta.yaml` with checkpoints on. No new tuning, so C
  in M4 gets exactly the same training settings.
- Training on CPU (deterministic kernels) instead of MPS, so each final checkpoint can be
  reproduced exactly from its seed.
- Results (dev, `results/b_final_s{1,2,3}_*.json`): macro-F1 0.8783 / 0.8768 / 0.8789, mean 0.878
  (std 0.001), matching the M2 MPS runs (0.877). Each run took about 6.8 min on CPU, and every
  seed's best epoch was 3. Checkpoints are 317 MB each in `checkpoints/` (gitignored), and each
  record stores the SHA-256 of its weights.
- B is +4.3 points over the LLM baseline on dev. Neutral recall is 0.86, against the LLM's 0.72.
- Weak spot: 2+ entity headlines score 0.860 and mixed-label headlines 0.844, below B's overall
  score. On mixed-label headlines B is slightly below the LLM (0.856), though the CIs overlap.
  This is where model C is meant to help.
- The best epoch was always the last (3). More epochs might help, but that would be tuning on dev.
  C gets the same 3 epochs. This is listed as a limitation.
- Served checkpoint if B wins in M6 (rule decided in advance): seed 3, the best on dev.
- M3 done.
- Reproducibility check: seed 1 was retrained on CPU from the committed code (`7175679`) as
  `results/b_final_s1_20261025T152030Z.json` (`purpose: repro_check`). The weights' SHA-256
  (`e9803a6873d6…`), per-epoch losses, dev macro-F1 and all 2,150 dev predictions are identical to
  the original seed-1 run, which had been trained from uncommitted code. Its duplicate checkpoint
  was removed with the owner's approval, so that record's `checkpoint.path` no longer exists; its
  SHA-256 still documents the match.

## 2026-10-25 · M4 Model C

- Design, approved before building: the plain headline with no markers, encoded once. Each
  entity's vector is the mean of each mention's token vectors, then the mean across its mentions.
  The dropout+linear head is the same as model B's. `configs/c_final.yaml` differs from `b_final.yaml`
  only in `name` and `architecture` (enforced by a test). Seeds 1-3, deterministic on CPU,
  checkpoints saved.
- **Batch-size difference (kept on purpose):** `batch_size: 32` counts model inputs. For B that
  is 32 entities, for C 32 headlines (about 1.34 entities each). So C takes 236 optimizer steps
  per epoch against B's 316, and each C step covers more entities. This follows from keeping the
  settings literally identical.
- Span -> token alignment: tokens whose fast-tokenizer character offsets overlap the mention.
  Special and padding tokens are never selected. A mention with no token, or a headline over
  max_length, raises an error.
- `scripts/check_alignment.py` on train + dev (9,142 headlines, 12,253 mentions; test not read):
  - 99.61% of mentions align exactly to token boundaries. In 0.39% (48) a token runs past the
    mention, mostly dropped-apostrophe suffixes ("Sebis", "Air Indias", "PSBs").
  - About 7 of those 48 come from annotation errors in SEntFiN: truncated names ("Alibab",
    "Karur Vysya Ban", "Australi", "G" -> "Gold"), "rupee" inside "rupeee", "Euro" inside
    "European", and "IHFL" inside "IIHFL". The whole token is pooled, which is reasonable. Model B
    marks the same partial strings.
  - 4 entity names have leading or trailing whitespace in the data (e.g. " Citigroup"). Alignment
    ignores the whitespace. The data is left unchanged so model B's inputs stay as trained.
  - 0 mentions without tokens, 0 tokens shared by two entities, longest headline 35 tokens.
  - Only 5 entities in train + dev have 2+ mentions, so "mean across mentions" rarely matters.
- Tests: 104 pass (25 new for model C). They cover hand-written alignment cases with the real
  tokenizer, coverage and minimality properties over every train + dev mention, pooling weights, a
  hand-computed pooled vector, independence from padding and batching, loud failures,
  determinism, the C checkpoint round trip, and a guard that the saved `b_final` seed-1
  checkpoint still reproduces its recorded dev predictions after the refactor.
- Results (dev, `results/c_final_s{1,2,3}_*.json`): macro-F1 0.8776 / 0.8759 / 0.8754, mean 0.876
  (std 0.001), against B's 0.878. 2+ entities: C 0.859, B 0.860. Mixed labels: C 0.837, B 0.844.
  Every best epoch was 3. Training took about 4.4 min per run on CPU against B's 6.8.
- Reading: on dev, C is level with B (-0.17 points, well inside each run's CI). The hoped-for gain
  on multi-entity headlines does not appear; B's explicit entity markers seem to matter at least as
  much as single-pass context. This is not the decision: it is made on test in M6 with the SPEC.md
  rule and the M5 cost numbers.
- CI simulation: a copy of the working tree without `data/raw`, `.cache`, `checkpoints`, `.venv`,
  `results/cache`, `.env` and `.git`, with a fresh `uv sync --locked`. Lint and mypy pass; the
  tests give 77 passed, 27 skipped (all the intended data, cache and checkpoint skips), 0 failed.
  No downloads.
- M4 done.

## 2026-10-25 · M5 Speed and cost benchmark

### "Cheaper per headline" rule (fixed before the benchmark is built or run, before M6)

- Primary cost measure: **process CPU time per 1,000 headlines in batched mode** (32 headlines
  per call; for B, all their entities in one call), with **4 torch threads**. Measured with
  `time.process_time()`, so time on all threads counts. Seed-1 checkpoints of B and C, all 1,613
  dev headlines, float32, inference mode, offline. Each measurement runs in its own fresh process,
  interleaved B, C, B, C, B, C, with 50 warm-up headlines that are not timed.
- **C is "cheaper per headline" only if its CPU time per 1,000 headlines is lower than B's in
  every one of the 3 interleaved repetitions.** Otherwise it is not cheaper.
- Also reported, but not part of the decision: forward passes per 1,000 headlines, batch-1
  end-to-end latency (median, p95, and by entity count) at 4 and 1 threads, wall-clock throughput
  at 4 and 1 threads, and model A's M1 numbers as a non-comparable network reference.
- No USD figure for B and C (owner's decision); SPEC.md asks for USD only for model A.
- Conditions: the owner keeps the Mac plugged in and idle during the run. The power state is
  recorded with the results.

### Results (`results/bench/m5_bench_20261025T163921Z.json`, Apple M5 on AC power)

| Model | Threads | Passes / 1k | Latency median / p95 | By entities 1 / 2 / 3+ | Headlines/s (batch 32) | CPU s / 1k headlines |
|---|---:|---:|---:|---:|---:|---:|
| B | 4 | 1,333 | 5.67 / 8.16 ms | 5.62 / 7.40 / 10.01 ms | 366 | 3.78 |
| C | 4 | 1,000 | 5.78 / 5.96 ms | 5.78 / 5.79 / 5.76 ms | 565 | 2.40 |
| B | 1 | 1,333 | 5.59 / 8.08 ms | 5.55 / 7.54 / 10.11 ms | 320 | 3.12 |
| C | 1 | 1,000 | 5.56 / 5.71 ms | 5.55 / 5.58 / 5.58 ms | 496 | 2.02 |

- **Verdict under the rule: C is cheaper per headline.** 4-thread CPU s / 1k headlines by
  repetition: B 3.79 / 3.77 / 3.78, C 2.40 / 2.40 / 2.40. Repetitions agree within 1%.
- C uses about 37% less CPU per headline (about 35% at 1 thread) and has about 54% more batched
  throughput. That is more than the 25% fewer forward passes alone: B's inputs are also a few tokens
  longer (markers), and it batches more sequences per call. Meltwater reported 45.5%; our headlines
  carry fewer entities (1.33 on average).
- Batch-1 latency is equal for single-entity headlines. B grows with entity count (7.4 ms at 2,
  10.0 ms at 3+) while C stays at about 5.8 ms; C's p95 is 5.96 ms against B's 8.16 ms.
- At these short inputs, 1 thread is as fast as 4 and uses less CPU. For serving (M7), several
  single-thread workers beat one 4-thread worker.
- 0 prediction mismatches in all 12 runs (batch 1 and batch 32): the benchmark measured exactly the
  models whose dev results were reported.
- This settles only the "cheaper" half of the SPEC.md decision rule. The accuracy half is decided
  on test in M6.
- The benchmark record says `git.commit` `91331a8` with `dirty: true`, because the M5 code was not
  yet committed. The commit that adds the record contains exactly the code that produced it.
- M5 done.

## 2026-10-25 · M6 Final test run (test split used once)

### Session

- Dry run on dev first (`results/dry_run/m6_dev_20261025T164724Z.json`): all 7 models reproduced
  their recorded dev macro-F1 exactly; A cost $0 (all from cache). Code committed as `5781b9d`.
- One-time test session `results/test/m6_test_20261025T165028Z.json`: 1,611 headlines, 2,157
  entities. A, B seeds 1-3 and C seeds 1-3, all predictions saved before any statistic. Every
  checkpoint SHA-256 was checked. A spent $0.1300 of new API cost ($0.081 per 1,000 headlines, as
  estimated), with 0 invalid answers. Nothing was rerun.
- The record says `git: 5781b9d (dirty)`. That is a script bug: `results/test/LOCK` was created
  before the record's git status was read, and the new untracked LOCK file counted as "dirty". The
  LOCK itself records `5781b9d` with `dirty: false`, and `src/`, `scripts/`, `tests/` and
  `configs/` were unchanged from that commit. Not fixed or rerun, since test is used once.
- A reported 5 "cached" headlines on test: test contains 7 extra copies of identical inputs, and
  5 of them hit the answer of their first copy within the same session. Nothing came from outside
  test.

### Results (SPEC.md protocol)

| | B mean ± std [CI] | C mean ± std [CI] | C − B [95% CI] |
|---|---:|---:|---:|
| All (1,611) | 0.883 ± 0.002 [0.868, 0.897] | 0.882 ± 0.002 [0.867, 0.896] | −0.0009 [−0.0085, +0.0065] |
| 2+ entities (426) | 0.879 ± 0.008 [0.852, 0.903] | 0.885 ± 0.005 [0.858, 0.908] | +0.0058 [−0.0053, +0.0169] |
| Mixed labels (184) | 0.869 ± 0.002 [0.834, 0.903] | 0.874 ± 0.006 [0.839, 0.907] | +0.0051 [−0.0089, +0.0198] |

- Model A: 0.837 [0.820, 0.854]; 2+ entities 0.837; mixed labels 0.882 [0.846, 0.915].
- McNemar per seed pair (B right/C wrong vs C right/B wrong): seed 1 56 vs 60 (p 0.78), seed 2 61
  vs 67 (p 0.66), seed 3 62 vs 50 (p 0.30). No significant difference.
- **Decision (SPEC.md rule): C.** The CI of C − B, [−0.0085, +0.0065], includes zero, and C is
  cheaper per headline (M5: 2.40 vs 3.78 CPU s / 1k headlines).
- **Served checkpoint: C seed 1** (best dev macro-F1, chosen by the rule before test); its own test
  macro-F1 is 0.883 [0.867, 0.898]. This is not the headline number.
- Reading: B and C are statistically indistinguishable in accuracy; C gives that accuracy for about
  37% less CPU. B and C are about 4.5 points above the zero-shot LLM overall. On mixed-label
  headlines the LLM (0.882) is level with or above B and C (0.869 / 0.874; CIs overlap), the one
  place it matches them. C is nominally ahead of B on the multi-entity and mixed-label subsets on
  test, the opposite of dev; both differences are well inside their CIs, so the honest reading is
  "no difference". Test scores are slightly above dev for every model.

### Error analysis (40 of C seed 1's 252 wrong test entities, sampling seed 0, read by hand)

`results/test/error_analysis_m6_test_20261025T165028Z.json` has the pattern of every sampled error
(headline IDs only; the texts stay in the gitignored `results/test/private/`).

| Pattern | Errors | A right | B seed 1 right |
|---|---:|---:|---:|
| Neutral vs polar boundary: hedged, advisory or forecast language where both readings are defensible ("avoid long bets", "short ... with stoploss", "signs of life") | 15 | 1 | 0 |
| Likely annotation error: all models agree against the gold label ("rallies 14%" labelled negative) | 3 | 0 | 0 |
| Missed implicit or jargon-coded sentiment ("snaps 18%", "overweight", "defreeze accounts", "recommend sell", "withdraws IPO prospectus") | 10 | 9 | 2 |
| Negation or composition ("not bleak", "bearishness ebbs", "improvement required", "no serious corrections") | 5 | 3 | 2 |
| Comparative or target confusion ("prefer X over Y", "weaker yen helps Nikkei") | 5 | 5 | 1 |
| Spurious polarity from a topic word ("chart check", "court order on bail plea") | 2 | 2 | 1 |
| **Total** | **40** | **20** | **6** |

- Gold → predicted: neutral→negative 10, negative→neutral 8, neutral→positive 7,
  positive→neutral 5, positive→negative 5, negative→positive 5. 25 of the 40 errors involve the
  neutral class.
- **18 of 40 (45%)** errors sit on the neutral/polar boundary or look like annotation errors. Here
  the zero-shot LLM is right only 1 time in 18, so the fine-tuned models' advantage over A largely
  comes from learning SEntFiN's labelling conventions for neutral.
- On the other **22**, the LLM is right **19** times: finance jargon and implied consequences (9 of
  10) and comparative or two-entity headlines (5 of 5). These are world-knowledge and reasoning
  gaps of a small encoder trained on about 10k entities, not span or pooling bugs.
- B seed 1 gets only 6 of C seed 1's 40 errors right: B and C mostly fail on the same entities, so
  the errors come from the shared encoder and the data, not the architecture.
- All 5 comparative/target confusions are in 2+ entity headlines. Single-pass pooling did not
  remove that failure mode.
- M6 done. Any later change to models or data is reported as a new, clearly labelled run.

## 2026-10-25 · M7 Serving

- `src/entsent/api.py`: FastAPI with `POST /predict` (text + entity character spans in; label,
  confidence and the three probabilities per entity out) and `GET /health` (503 until the model is
  loaded, or if loading failed). It serves C seed 1 (`c_final_s1_20261025T160107Z`), the checkpoint
  the M6 rule chose, with the same inference code as training and evaluation.
- The weights are not in git or in the image: the checkpoint folder is mounted read-only
  (`MODEL_DIR`), and with `MODEL_SHA256` set the server refuses mismatching weights. The model's
  identity comes from its config (`c_final_s1`), not from the folder it is mounted in. The smoke
  test caught the first version reporting "model".
- Input validation (422 with a reason): text 1-1,000 characters, 1-20 entities, 1-10 mentions,
  spans inside the text, non-empty and not whitespace only, no overlapping spans, and the
  tokenized length within 128 (too long is 422, not 500). Request bodies are never logged.
- New packages: `fastapi` 0.162.0 and `uvicorn` 0.54.0, in a `serve` dependency group (a default
  group, so CI runs the API tests). `scipy` is now declared directly (used since M6).
- Tests: 21 API tests (148 in total) with the tiny model saved as a real checkpoint. They cover
  health, sha mismatch, schema, probabilities, determinism, agreement with the evaluation code,
  12 kinds of bad input, and a local-only test that the API reproduces C seed 1's recorded labels on
  200 dev headlines.
- Docker: `python:3.12.15-slim-bookworm` pinned by digest, CPU-only torch 2.14.1 (no CUDA or
  nvidia packages), the exact versions from `uv.lock` (`docker/requirements-serve.txt`, reproduced
  by `make docker-reqs`), non-root user, `HEALTHCHECK`. `.dockerignore` is a whitelist: the image
  holds only `src/` and the requirements; no `.env`, data or weights were found inside. Image size
  1.96 GB; it includes training-only packages (scikit-learn, scipy, pandas). A smaller serving
  image is planned for M8.
- Smoke test against the container (`scripts/smoke_api.py`): 0 label mismatches on 200 dev
  headlines against C seed 1's recorded predictions, a bad request gives 422, Docker's
  HEALTHCHECK reports healthy, and no headline text appears in the logs.

### macOS vs Linux: the M5 thread finding does not carry over

| Where the API runs (Apple M5) | torch BLAS | Median per request | Max |
|---|---|---:|---:|
| Natively on macOS, 1 thread (`make serve`) | Accelerate | 6.3 ms | 23.7 ms |
| Docker (OrbStack, Linux arm64), 1 thread, 2 workers | OpenBLAS + oneDNN | 74 ms | 81 ms |
| Same, measured inside the container (no port forwarding) | OpenBLAS + oneDNN | 73.8 ms | 79.0 ms |
| Docker, 4 threads, 1 worker (**image default**) | OpenBLAS + oneDNN | 23.3 ms | 29.6 ms |

- The API itself adds about 0.5 ms and port forwarding adds nothing; the gap is torch's maths
  library. The macOS wheel uses Apple Accelerate, while the Linux arm64 CPU wheel uses OpenBLAS and
  oneDNN.
- M5's "1 thread is the most efficient" holds on macOS, not in this Linux image, where 4 threads
  are about 3x faster. **The M5 speed numbers are specific to macOS.** The B vs C comparison was
  made under identical conditions, so the relative result stands; absolute serving latency depends
  on the platform.
- Image default, per the owner: `TORCH_THREADS=4`, `WEB_CONCURRENCY=1`, documented as measured on
  this setup. Not measured: x86-64 Linux (Intel MKL), which would behave differently again.
- M7 done.
