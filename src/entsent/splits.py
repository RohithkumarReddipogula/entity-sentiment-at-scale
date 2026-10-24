"""Fixed, stratified, group-aware train/dev/test split for SEntFiN 1.0."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TypedDict

from sklearn.model_selection import train_test_split

from entsent.data import LABELS, Example, SplitName, normalize_text

SEED = 42
FRACTIONS: dict[SplitName, float] = {"train": 0.70, "dev": 0.15, "test": 0.15}


class SplitStats(TypedDict):
    headlines: int
    entities: int
    labels: dict[str, int]
    multi_entity_headlines: int
    mixed_label_headlines: int
    entities_in_multi_headlines: int


def stratum(example: Example) -> str:
    """Stratification key: the label (or 'mixed') and whether the headline has 2+ entities."""
    if example.is_mixed:
        return "mixed"
    label = example.entities[0].label
    return f"{label}|{'multi' if example.is_multi else 'single'}"


def make_split(examples: Sequence[Example], seed: int = SEED) -> dict[SplitName, list[int]]:
    """Split by headline group: identical headlines (after normalization) stay together."""
    groups: dict[str, list[Example]] = {}
    for ex in sorted(examples, key=lambda e: e.id):
        groups.setdefault(normalize_text(ex.text), []).append(ex)
    keys = sorted(groups)
    strata = [stratum(groups[k][0]) for k in keys]

    hold = FRACTIONS["dev"] + FRACTIONS["test"]
    train_keys, rest_keys, _, rest_strata = train_test_split(
        keys, strata, test_size=hold, random_state=seed, stratify=strata
    )
    dev_keys, test_keys = train_test_split(
        rest_keys,
        test_size=FRACTIONS["test"] / hold,
        random_state=seed,
        stratify=rest_strata,
    )
    return {
        "train": sorted(ex.id for k in train_keys for ex in groups[k]),
        "dev": sorted(ex.id for k in dev_keys for ex in groups[k]),
        "test": sorted(ex.id for k in test_keys for ex in groups[k]),
    }


def split_stats(examples: Sequence[Example]) -> SplitStats:
    labels = Counter(e.label for ex in examples for e in ex.entities)
    return {
        "headlines": len(examples),
        "entities": sum(labels.values()),
        "labels": {lab: labels[lab] for lab in LABELS},
        "multi_entity_headlines": sum(ex.is_multi for ex in examples),
        "mixed_label_headlines": sum(ex.is_mixed for ex in examples),
        "entities_in_multi_headlines": sum(len(ex.entities) for ex in examples if ex.is_multi),
    }
