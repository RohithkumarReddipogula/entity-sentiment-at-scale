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
