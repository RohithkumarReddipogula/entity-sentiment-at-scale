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

## 2026-10-24 · M2 Training setup (in progress)

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
