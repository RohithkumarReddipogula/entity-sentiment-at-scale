"""Entity-level classification metrics with headline-level bootstrap confidence intervals.

Predictions may contain ``invalid`` (no usable answer). It is a fourth prediction column:
always wrong, a false negative for the gold class, and never a false positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from entsent.data import LABELS

PRED_COLUMNS = (*LABELS, "invalid")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0

FloatArray = npt.NDArray[np.float64]


def confusion(gold: Sequence[str], pred: Sequence[str]) -> FloatArray:
    """3 x 4 matrix: rows are gold labels, columns are predictions (labels + invalid)."""
    cm = np.zeros((len(LABELS), len(PRED_COLUMNS)))
    for g, p in zip(gold, pred, strict=True):
        cm[LABELS.index(g), PRED_COLUMNS.index(p)] += 1
    return cm


def _f1_per_class(cm: FloatArray) -> FloatArray:
    """F1 per class for one (3, 4) matrix or a stack (..., 3, 4)."""
    k = len(LABELS)
    tp = np.diagonal(cm[..., :k], axis1=-2, axis2=-1)
    fp = cm[..., :k].sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def scores(cm: FloatArray) -> dict[str, Any]:
    k = len(LABELS)
    tp = np.diagonal(cm[:, :k])
    predicted = cm[:, :k].sum(axis=0)
    support = cm.sum(axis=1)
    f1 = _f1_per_class(cm)
    per_class = {
        label: {
            "precision": float(tp[i] / predicted[i]) if predicted[i] else 0.0,
            "recall": float(tp[i] / support[i]) if support[i] else 0.0,
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
        "n_entities": int(cm.sum()),
        "n_invalid": int(cm[:, k].sum()),
        "per_class": per_class,
    }


def bootstrap_macro_f1_ci(
    per_headline: FloatArray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    chunk: int = 500,
) -> tuple[float, float]:
    """95% percentile CI for macro-F1, resampling headlines (entities in one headline are
    not independent). ``per_headline`` has shape (n_headlines, 3, 4)."""
    rng = np.random.default_rng(seed)
    n = per_headline.shape[0]
    values: list[FloatArray] = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        counts = np.stack([np.bincount(row, minlength=n) for row in idx]).astype(np.float64)
        cms = np.einsum("bh,hij->bij", counts, per_headline)
        values.append(_f1_per_class(cms).mean(axis=-1))
    all_values = np.concatenate(values)
    lo, hi = np.percentile(all_values, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Overall, 2+ entity and mixed-label metrics, each with a bootstrap CI for macro-F1.

    ``gold`` and ``pred`` hold one list of entity labels per headline. An empty subset is
    reported as None rather than as a score of 0.
    """
    per_headline = np.stack([confusion(g, p) for g, p in zip(gold, pred, strict=True)])
    subsets = {
        "all": np.ones(len(gold), dtype=bool),
        "multi_entity": np.array([len(g) >= 2 for g in gold]),
        "mixed_label": np.array([len(set(g)) >= 2 for g in gold]),
    }
    out: dict[str, Any] = {}
    for name, mask in subsets.items():
        if not mask.any():
            out[name] = None
            continue
        sub = per_headline[mask]
        result = scores(sub.sum(axis=0))
        result["n_headlines"] = int(mask.sum())
        lo, hi = bootstrap_macro_f1_ci(sub, n_resamples, seed)
        result["macro_f1_ci95"] = [lo, hi]
        out[name] = result
    return out
