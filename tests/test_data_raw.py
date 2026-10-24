"""Data tests on the real SEntFiN file. Skipped when data/raw is absent (e.g. in CI)."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from entsent.data import (
    LABELS,
    RAW_PATH,
    SOURCE_SHA256,
    SPLITS,
    Example,
    load_examples,
    normalize_text,
    read_split_ids,
    sha256_file,
)
from entsent.splits import make_split

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="run `make data` first")


@pytest.fixture(scope="module")
def loaded() -> tuple[list[Example], Counter[str]]:
    examples, events = load_examples(RAW_PATH)
    return examples, Counter(e.action for e in events)


@pytest.fixture(scope="module")
def examples(loaded: tuple[list[Example], Counter[str]]) -> list[Example]:
    return loaded[0]


def test_raw_checksum() -> None:
    assert sha256_file(RAW_PATH) == SOURCE_SHA256


def test_known_counts(loaded: tuple[list[Example], Counter[str]]) -> None:
    examples, actions = loaded
    assert len(examples) == 10_753
    assert sum(len(ex.entities) for ex in examples) == 14_405
    assert sum(ex.is_multi for ex in examples) == 2_847
    assert sum(ex.is_mixed for ex in examples) == 1_233
    assert actions == Counter({"repaired": 19, "parsed_with_fallback": 13})


def test_all_spans_valid(examples: list[Example]) -> None:
    for ex in examples:
        assert ex.entities, ex.id
        for ent in ex.entities:
            assert ent.label in LABELS
            assert ent.spans, (ex.id, ent.name)
            for start, end in ent.spans:
                assert 0 <= start < end <= len(ex.text)
                assert ex.text[start:end] == ent.name, (ex.id, ent.name)


def test_no_corrupted_names_left(examples: list[Example]) -> None:
    for ex in examples:
        for ent in ex.entities:
            assert "neutraling" not in ent.name and "Rusnegative" not in ent.name


def test_spans_of_different_entities_do_not_overlap(examples: list[Example]) -> None:
    for ex in examples:
        spans = sorted(sp for ent in ex.entities for sp in ent.spans)
        for (_, end), (start, _) in pairwise(spans):
            assert end <= start, ex.id


def test_no_headline_text_in_two_splits(examples: list[Example]) -> None:
    by_id = {ex.id: ex for ex in examples}
    seen: dict[str, str] = {}
    for name in SPLITS:
        for i in read_split_ids(name):
            key = normalize_text(by_id[i].text)
            assert seen.setdefault(key, name) == name, f"{i} leaks into {name}"


def test_split_reproduces_from_seed(examples: list[Example]) -> None:
    split = make_split(examples)
    for name in SPLITS:
        assert split[name] == read_split_ids(name)


def test_splits_are_stratified(examples: list[Example]) -> None:
    by_id = {ex.id: ex for ex in examples}

    def shares(ids: list[int]) -> dict[str, float]:
        part = [by_id[i] for i in ids]
        labels = Counter(e.label for ex in part for e in ex.entities)
        n = sum(labels.values())
        out: dict[str, float] = {lab: labels[lab] / n for lab in LABELS}
        out["multi"] = sum(ex.is_multi for ex in part) / len(part)
        out["mixed"] = sum(ex.is_mixed for ex in part) / len(part)
        return out

    overall = shares([ex.id for ex in examples])
    for name in SPLITS:
        for key, value in shares(read_split_ids(name)).items():
            assert abs(value - overall[key]) < 0.02, (name, key)
