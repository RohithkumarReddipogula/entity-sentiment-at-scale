"""Metrics agree with scikit-learn, treat 'invalid' as wrong, and give sane bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from entsent.data import LABELS
from entsent.metrics import bootstrap_macro_f1_ci, confusion, evaluate, scores


def test_matches_sklearn_with_invalid_predictions() -> None:
    rng = np.random.default_rng(1)
    gold = list(rng.choice(LABELS, size=300))
    pred = list(rng.choice([*LABELS, "invalid"], size=300, p=[0.3, 0.3, 0.3, 0.1]))
    s = scores(confusion(gold, pred))
    assert s["macro_f1"] == pytest.approx(
        f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0)
    )
    assert s["accuracy"] == pytest.approx(accuracy_score(gold, pred))
    assert s["n_invalid"] == pred.count("invalid")


def test_invalid_counts_as_wrong() -> None:
    gold = ["positive", "negative", "neutral"]
    assert scores(confusion(gold, gold))["macro_f1"] == 1.0
    worse = scores(confusion(gold, ["positive", "negative", "invalid"]))
    assert worse["accuracy"] == pytest.approx(2 / 3)
    assert worse["per_class"]["neutral"]["recall"] == 0.0


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(2)
    gold = [list(rng.choice(LABELS, size=2)) for _ in range(200)]
    pred = [[g if rng.random() < 0.7 else "neutral" for g in gs] for gs in gold]
    per_headline = np.stack([confusion(g, p) for g, p in zip(gold, pred, strict=True)])
    point = scores(per_headline.sum(axis=0))["macro_f1"]
    lo, hi = bootstrap_macro_f1_ci(per_headline, n_resamples=2_000)
    assert lo < point < hi
    assert 0.02 < hi - lo < 0.3


def test_evaluate_subsets() -> None:
    gold = [["positive"], ["positive", "negative"], ["neutral", "neutral"]]
    pred = [["positive"], ["positive", "positive"], ["neutral", "neutral"]]
    out = evaluate(gold, pred, n_resamples=200)
    assert out["all"]["n_headlines"] == 3
    assert out["multi_entity"]["n_headlines"] == 2
    assert out["mixed_label"]["n_headlines"] == 1
    assert out["mixed_label"]["accuracy"] == 0.5


def test_empty_subset_is_none_not_zero() -> None:
    out = evaluate([["positive"]], [["positive"]], n_resamples=100)
    assert out["multi_entity"] is None and out["mixed_label"] is None
