# Model card: entity sentiment, single-pass encoder (C seed 1)

## Model

- **What it does:** for each entity mentioned in a financial news headline, it predicts whether
  the news is positive, negative or neutral for that entity.
- **Architecture:** model C in this repository. distilroberta-base encodes the headline once.
  Each entity's vector is the mean of its mention tokens (mean per mention, then across
  mentions), followed by dropout and one linear layer.
- **Base model:** [`distilbert/distilroberta-base`](https://huggingface.co/distilbert/distilroberta-base)
  at revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, Apache-2.0. About 82M parameters.
- **This checkpoint:** run `c_final_s1_20261025T160107Z`, seed 1, float32. Weights SHA-256
  `563bca8e924b6317cc8d4c6e721f2b3df171f447fb18cf63e8e21ea5bbe723ca`.
- **Why this one:** the decision rule in [SPEC.md](SPEC.md) chose model C over model B, and seed 1
  had the best dev score of the three C seeds. Both choices were fixed before the test run.

## Getting the weights

The weights (317 MB) are not in the git repository. They are attached to the GitHub Release
`model-c-s1-v1` as `c_final_s1_20261025T160107Z.tar.gz`. [deploy/model.json](deploy/model.json)
records the SHA-256 of the archive and of `model.pt`, and `make fetch-model` checks both:

```bash
make setup && make fetch-model
```

Training is deterministic on CPU, so they can also be rebuilt exactly:

```bash
make setup && make data && make train-c
shasum -a 256 checkpoints/c_final_s1_*/model.pt   # compare with the SHA-256 above
```

`make train-c` trains seeds 1, 2 and 3; seed 1 is the served model. I checked bit-exact
reproduction on the model B checkpoint of seed 1 (see LOG.md, M3).

## Intended use

- Research and learning: comparing per-entity and single-pass designs, and as a baseline for
  entity-level sentiment on short financial texts.
- Input: one English headline (up to 1,000 characters and 128 tokens) and character spans for 1
  to 20 entities.

Out of scope: trading or investment decisions, long documents, languages other than English,
and text very different from financial news headlines. The model has never seen such text.

## Training data

[SEntFiN 1.0](https://github.com/pyRis/SEntFiN) (MIT license): financial news headlines, mostly
from Indian media, with one label per entity. Fixed split by headline: 7,529 train, 1,613 dev,
1,611 test headlines. Duplicate headlines stay in one split. See [DATASET.md](DATASET.md).

Training: 3 epochs, batch of 32 headlines, AdamW, learning rate 3e-5 with 10% warm-up and linear
decay, weight decay 0.01, dropout 0.1, gradient clipping 1.0, CPU with deterministic kernels. The
epoch with the best dev macro-F1 was kept (epoch 3). The settings are the same as model B's and
were not tuned. Config: [`configs/c_final.yaml`](configs/c_final.yaml).

## Evaluation

On the test split, used once (1,611 headlines, 2,157 entities). These numbers come from
`results/test/m6_test_20261025T165028Z.json`:

| | Macro-F1 [95% CI] |
|---|---:|
| This checkpoint (C seed 1), all headlines | 0.883 [0.867, 0.898] |
| This checkpoint, headlines with 2+ entities | 0.885 [0.858, 0.910] |
| This checkpoint, headlines whose entities have different labels | 0.876 [0.839, 0.909] |
| Model C, mean of seeds 1-3 | 0.882 [0.867, 0.896] |
| Model B (one pass per entity), mean of seeds 1-3 | 0.883 [0.868, 0.897] |
| Zero-shot LLM (Claude Haiku 5.5) | 0.837 [0.820, 0.854] |

The confidence intervals come from a bootstrap over headlines with 10,000 resamples. B and C are
not significantly different. C needs about 37% less CPU per headline (2.40 against 3.78 CPU
seconds per 1,000 headlines on an Apple M5).

## Known errors and limitations

I read 40 of this checkpoint's 252 wrong test predictions by hand:

- 15 sit on the line between neutral and positive or negative, in hedged or advisory language,
  where both readings are defensible.
- 10 miss sentiment that comes from finance jargon or an implied consequence ("snaps 18%",
  "overweight", "recommend sell").
- 5 are negation or composition ("not bleak", "bearishness ebbs").
- 5 give the wrong side of a comparison ("prefer X over Y").
- 3 look like errors in the gold labels.
- 2 react to a topic word in a headline with no sentiment.

The zero-shot LLM got 20 of these 40 right, mostly the jargon and comparison cases. Other
limits:

- One dataset and one domain.
- Truncated or corrupted entity names in the data (the model pools whatever tokens the span
  covers).
- Speed numbers that depend on the machine and the maths library (see the README).

## License and citation

Code: MIT. Base model: Apache-2.0. Data: SEntFiN 1.0, MIT, Copyright (c) 2021 Ankur Sinha,
Satishwar Kedas, Rishu Kumar, Pekka Malo. Please cite the dataset:

> Sinha, A., Kedas, S., Kumar, R., and Malo, P. (2022). SEntFiN 1.0: Entity-aware sentiment
> analysis for financial news. *Journal of the Association for Information Science and
> Technology*, 73(9), 1314-1335. https://doi.org/10.1002/asi.24634
