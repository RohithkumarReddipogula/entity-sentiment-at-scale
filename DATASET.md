# Dataset: SEntFiN 1.0

Entity-level sentiment for financial news headlines: each headline lists the entities it
mentions and one label per entity from {positive, negative, neutral}.

## Source

| | |
|---|---|
| Paper | Sinha, Kedas, Kumar, Malo. "SEntFiN 1.0: Entity-aware sentiment analysis for financial news." *Journal of the Association for Information Science and Technology* (2022). Preprint: [arXiv:2305.12257](https://arxiv.org/abs/2305.12257) |
| Data repository | [github.com/pyRis/SEntFiN](https://github.com/pyRis/SEntFiN) |
| Pinned commit | `eba43610fe3cb57a3e5773cd6c037da1f8992ad5` (2023-05-31) |
| File | `SEntFiN.csv`, 1,118,162 bytes |
| SHA-256 | `570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f` |

## License (checked 2026-10-24)

| Source | What it says |
|---|---|
| `LICENSE` in the data repository | **MIT License**, "Copyright (c) 2021 Ankur Sinha, Satishwar Kedas, Rishu Kumar, Pekka Malo" (the paper's authors). GitHub's API reports `spdx_id: MIT`. |
| arXiv preprint | Paper text under CC BY 4.0. The abstract says "we make publicly available SEntFiN 1.0, a human-annotated dataset". |
| Hugging Face / Kaggle | No official mirror found. |

The MIT license allows use, modification and redistribution, including of derived results, as
long as the copyright notice is kept.

**Caveat:** the headlines themselves were written by news publishers, who hold their copyright.
The MIT license covers the authors' collection and annotations. To stay on the safe side, **this repo
does not commit the raw data**: `make data` downloads the pinned file and checks its SHA-256. The repo
stores only headline IDs (`data/splits/`) and derived statistics.

Alternative considered: FinEntity (ODC-BY 1.0; 979 texts, 2,131 entities, 589 texts with 2+
entities). We rejected it because it has about 5x fewer multi-entity texts, and the B vs C comparison
needs those.

## Cleaning rules

The loader is `src/entsent/data.py`. Every change it makes is listed in the cleaning log below.

1. **Parsing.** The `Decisions` column is a Python dict literal. 13 rows have unescaped quotes
   (e.g. `'Osian's Art Fund'`) and are parsed with a regex fallback.
2. **Repairing names.** Upstream, a find-and-replace turned "Hold" into "neutral" and "sell" into "negative"
   inside entity names (`HSBC neutralings`, `FTSE Rusnegative`). The loader undoes the
   replacement and **keeps the repaired name only if it occurs verbatim in the headline**.
   Otherwise the entity is dropped.
3. **Character spans.** SEntFiN has no offsets, so each entity is matched in its headline:
   case-sensitive, starting at a word boundary (so "Jet" does not match inside "SpiceJet").
   A match may run into a suffix such as "Sebis" or "Nifty50", because headlines often drop apostrophes.
   If no word-boundary match exists, any substring match is used. All mentions are kept.
   A mention that lies inside a longer entity's mention in the same headline is removed
   ("Pharma" inside "Sun Pharma"), so different entities never overlap. An entity with no mention
   left is dropped. A headline with no entity left is dropped.
4. **Duplicates.** Headlines that are identical after lowercasing and whitespace normalization
   form one group. A group is never split across train/dev/test. Some duplicates carry different
   annotations; we keep them as they are.

## Split

- Train / dev / test = 70 / 15 / 15 over headline groups, seed 42 (`scripts/make_data.py`).
- Stratified by: the shared label of single-label headlines × (1 vs 2+ entities), with all
  mixed-label headlines as one extra stratum. This keeps the label mix, the multi-entity share
  and the mixed-label share similar across splits.
- IDs are the `S No.` column of the raw CSV. Files: `data/splits/{train,dev,test}.txt`.
  `data/splits/manifest.json` holds the seed, source hash, file hashes and stats.
- **The test split is sealed until M6.** `load_split("test")` raises unless called with
  `allow_test=True`.

## Statistics

"Mixed-label" means the headline's entities do not all share one label.

<!-- BEGIN GENERATED: stats -->
| | All | Train | Dev | Test |
|---|---:|---:|---:|---:|
| Headlines | 10,753 | 7,529 | 1,613 | 1,611 |
| Entities | 14,405 | 10,098 | 2,150 | 2,157 |
| Positive entities | 5,074 (35.2%) | 3,544 (35.1%) | 760 (35.3%) | 770 (35.7%) |
| Negative entities | 3,814 (26.5%) | 2,662 (26.4%) | 571 (26.6%) | 581 (26.9%) |
| Neutral entities | 5,517 (38.3%) | 3,892 (38.5%) | 819 (38.1%) | 806 (37.4%) |
| Headlines with 2+ entities | 2,847 (26.5%) | 1,996 (26.5%) | 425 (26.3%) | 426 (26.4%) |
| Mixed-label headlines | 1,233 (11.5%) | 865 (11.5%) | 184 (11.4%) | 184 (11.4%) |
<!-- END GENERATED: stats -->

## Cleaning log

<!-- BEGIN GENERATED: cleaning -->
Repaired: 19, dropped: 0, parsed with regex fallback: 13.

| ID | Action | Original | Result | Headline |
|---:|---|---|---|---|
| 3615 | parsed_with_fallback | {'Sebi': 'neutral', 'Osian's Art Fund': 'neutral'} | {'Sebi': 'neutral', "Osian's Art Fund": 'neutral'} | (headline text removed) |
| 3783 | parsed_with_fallback | {'Tata Steel UK': 'negative', 'Moody's': 'neutral'} | {'Tata Steel UK': 'negative', "Moody's": 'neutral'} | (headline text removed) |
| 4909 | parsed_with_fallback | {'Sun Pharma': 'positive', 'Dr Reddy's Labs': 'positive'} | {'Sun Pharma': 'positive', "Dr Reddy's Labs": 'positive'} | (headline text removed) |
| 5014 | parsed_with_fallback | {'Sun Pharma': 'positive', 'Dr. Reddy's': 'positive'} | {'Sun Pharma': 'positive', "Dr. Reddy's": 'positive'} | (headline text removed) |
| 5321 | parsed_with_fallback | {'IOC': 'negative', 'BPCL': 'negative', 'HPCL': 'negative', 'Moody's': 'neutral'} | {'IOC': 'negative', 'BPCL': 'negative', 'HPCL': 'negative', "Moody's": 'neutral'} | (headline text removed) |
| 7226 | parsed_with_fallback | {'BP': 'positive', 'Britain's FTSE': 'positive'} | {'BP': 'positive', "Britain's FTSE": 'positive'} | (headline text removed) |
| 8167 | parsed_with_fallback | {'Nikkei': 'positive', 'BOJ': 'neutral', 'Moody's': 'neutral'} | {'Nikkei': 'positive', 'BOJ': 'neutral', "Moody's": 'neutral'} | (headline text removed) |
| 8202 | parsed_with_fallback | {'Samsung': 'positive', 'Barron's': 'neutral'} | {'Samsung': 'positive', "Barron's": 'neutral'} | (headline text removed) |
| 9624 | parsed_with_fallback | {'Citic Securities': 'negative', 'China's central bank': 'neutral'} | {'Citic Securities': 'negative', "China's central bank": 'neutral'} | (headline text removed) |
| 9824 | parsed_with_fallback | {'Diageo': 'negative', 'BAE': 'negative', 'UK's FTSE': 'negative'} | {'Diageo': 'negative', 'BAE': 'negative', "UK's FTSE": 'negative'} | (headline text removed) |
| 10269 | parsed_with_fallback | {'Sun Pharma': 'negative', 'Dr Reddy's Laboratories': 'negative'} | {'Sun Pharma': 'negative', "Dr Reddy's Laboratories": 'negative'} | (headline text removed) |
| 10355 | parsed_with_fallback | {'Axis Bank': 'positive', 'HDFC Bank': 'positive', 'ICICI Bank': 'positive', 'Moody's': 'neutral'} | {'Axis Bank': 'positive', 'HDFC Bank': 'positive', 'ICICI Bank': 'positive', "Moody's": 'neutral'} | (headline text removed) |
| 10621 | parsed_with_fallback | {'Ranbaxy': 'positive', 'Dr. Reddy's': 'positive', 'Cadila': 'positive', 'HSBC': 'neutral'} | {'Ranbaxy': 'positive', "Dr. Reddy's": 'positive', 'Cadila': 'positive', 'HSBC': 'neutral'} | (headline text removed) |
| 305 | repaired | L&T Finance neutralings | L&T Finance Holdings | (headline text removed) |
| 1087 | repaired | Alibaba Group neutraling Inc | Alibaba Group Holding Inc | (headline text removed) |
| 1385 | repaired | Frank Rusnegative Company | Frank Russell Company | (headline text removed) |
| 2009 | repaired | IIFL neutralings | IIFL Holdings | (headline text removed) |
| 2099 | repaired | Bajaj neutralings | Bajaj Holdings | (headline text removed) |
| 2508 | repaired | Bajaj neutralings | Bajaj Holdings | (headline text removed) |
| 3652 | repaired | Avantha neutralings | Avantha Holdings | (headline text removed) |
| 4179 | repaired | Deccan Chronicle neutralings | Deccan Chronicle Holdings | (headline text removed) |
| 4180 | repaired | Deccan Chronicle neutralings | Deccan Chronicle Holdings | (headline text removed) |
| 4389 | repaired | Deccan Chronicle neutralings Limited | Deccan Chronicle Holdings Limited | (headline text removed) |
| 4550 | repaired | Avantha neutralings | Avantha Holdings | (headline text removed) |
| 5953 | repaired | WNS neutralings | WNS Holdings | (headline text removed) |
| 6656 | repaired | New Crystal Material neutralings | New Crystal Material Holdings | (headline text removed) |
| 6788 | repaired | FTSE Rusnegative | FTSE Russell | (headline text removed) |
| 7563 | repaired | FTSE-Rusnegative | FTSE-Russell | (headline text removed) |
| 8169 | repaired | Nomura neutralings Inc | Nomura Holdings Inc | (headline text removed) |
| 8251 | repaired | United Breweries neutralings | United Breweries Holdings | (headline text removed) |
| 8397 | repaired | Rane neutralings | Rane Holdings | (headline text removed) |
| 9277 | repaired | HSBC neutralings | HSBC Holdings | (headline text removed) |
<!-- END GENERATED: cleaning -->
