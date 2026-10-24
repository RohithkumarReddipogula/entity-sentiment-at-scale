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
