"""M6 statistics, exactly as fixed in SPEC.md ("Seeds and test reporting protocol").

- Per model: macro-F1 with a headline bootstrap CI (metrics.evaluate).
- Per architecture (B, C): mean macro-F1 over seeds 1-3, std, per-seed values, and a 95% CI from
  a paired bootstrap over headlines in which macro-F1 is averaged over the seeds per resample.
- Decision: 95% CI of mean(C) - mean(B) from the same resamples. C is preferred if the CI includes
  zero or favours C (upper bound >= 0) AND C is cheaper per headline (M5). Otherwise B.
- McNemar (exact binomial) on per-entity correctness for each seed pair B s vs C s.
- Every subset (all, 2+ entities, mixed labels) uses the same resample indices.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest

from entsent.metrics import BOOTSTRAP_SEED, N_BOOTSTRAP, FloatArray, _f1_per_class, confusion

SUBSETS = ("all", "multi_entity", "mixed_label")


def subset_masks(gold: Sequence[Sequence[str]]) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(gold), dtype=bool),
        "multi_entity": np.array([len(g) >= 2 for g in gold]),
        "mixed_label": np.array([len(set(g)) >= 2 for g in gold]),
    }


def per_headline_confusions(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]
) -> FloatArray:
    return np.stack([confusion(g, p) for g, p in zip(gold, pred, strict=True)])


def bootstrap_counts(n: int, n_resamples: int, seed: int, chunk: int = 500) -> list[FloatArray]:
    """Resample weights (how often each headline is drawn), shared by every model and subset."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        out.append(np.stack([np.bincount(row, minlength=n) for row in idx]).astype(np.float64))
    return out


def resampled_macro_f1(cms: FloatArray, counts: list[FloatArray]) -> FloatArray:
    """Macro-F1 of one model on every resample. ``cms``: (headlines, 3, 4)."""
    return np.concatenate(
        [_f1_per_class(np.einsum("bh,hij->bij", c, cms)).mean(axis=-1) for c in counts]
    )


def macro_f1(cms: FloatArray) -> float:
    return float(_f1_per_class(cms.sum(axis=0)).mean())


def ci95(values: FloatArray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def architecture_and_decision(
    gold: Sequence[Sequence[str]],
    b_preds: Sequence[Sequence[Sequence[str]]],  # per seed, per headline
    c_preds: Sequence[Sequence[Sequence[str]]],
    c_is_cheaper: bool,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Seed-averaged results for B and C, the C - B difference and the decision, per subset."""
    masks = subset_masks(gold)
    b_cms = [per_headline_confusions(gold, p) for p in b_preds]
    c_cms = [per_headline_confusions(gold, p) for p in c_preds]
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        if not mask.any():
            out[name] = None
            continue
        counts = bootstrap_counts(int(mask.sum()), n_resamples, seed)
        b_seed = [macro_f1(cm[mask]) for cm in b_cms]
        c_seed = [macro_f1(cm[mask]) for cm in c_cms]
        b_boot = np.mean([resampled_macro_f1(cm[mask], counts) for cm in b_cms], axis=0)
        c_boot = np.mean([resampled_macro_f1(cm[mask], counts) for cm in c_cms], axis=0)
        out[name] = {
            "n_headlines": int(mask.sum()),
            "B": _arch_summary(b_seed, b_boot),
            "C": _arch_summary(c_seed, c_boot),
            "C_minus_B": {
                "mean_difference": statistics.mean(c_seed) - statistics.mean(b_seed),
                "ci95": ci95(c_boot - b_boot),
            },
        }
    diff_ci = out["all"]["C_minus_B"]["ci95"]
    not_worse = diff_ci[1] >= 0
    out["decision"] = {
        "rule": "C if the 95% CI of mean(C) - mean(B) on all headlines includes zero or favours "
        "C (upper bound >= 0) AND C is cheaper per headline (M5); otherwise B",
        "c_not_significantly_worse": bool(not_worse),
        "c_is_cheaper": c_is_cheaper,
        "chosen": "C" if (not_worse and c_is_cheaper) else "B",
    }
    return out


def _arch_summary(per_seed: list[float], boot: FloatArray) -> dict[str, Any]:
    return {
        "mean_macro_f1": statistics.mean(per_seed),
        "std": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed,
        "ci95_seed_averaged": ci95(boot),
    }


def mcnemar(
    gold: Sequence[Sequence[str]],
    pred_b: Sequence[Sequence[str]],
    pred_c: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Exact McNemar on per-entity correctness: only discordant entities count."""
    b_only = c_only = both = neither = 0
    for gs, bs, cs in zip(gold, pred_b, pred_c, strict=True):
        for g, b, c in zip(gs, bs, cs, strict=True):
            b_ok, c_ok = b == g, c == g
            b_only += b_ok and not c_ok
            c_only += c_ok and not b_ok
            both += b_ok and c_ok
            neither += not b_ok and not c_ok
    n = b_only + c_only
    p = 1.0 if n == 0 else float(binomtest(min(b_only, c_only), n, 0.5).pvalue)
    return {
        "b_correct_c_wrong": b_only,
        "c_correct_b_wrong": c_only,
        "both_correct": both,
        "both_wrong": neither,
        "p_value_exact": p,
    }


def _f1(m: dict[str, Any] | None) -> str:
    if m is None:
        return "n/a"
    lo, hi = m["macro_f1_ci95"]
    return f"{m['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}]"


def render_report(record: dict[str, Any]) -> list[str]:
    """Markdown for an M6 record (dry run or test), used by the script and RESULTS.md."""
    split = record["split"]
    lines = [
        f"Run `{record['run_id']}` ({record['purpose']}, {split}): {record['n_headlines']:,} "
        f"headlines, {record['n_entities']:,} entities, git `{(record['git']['commit'] or '')[:7]}`"
        f"{' (dirty)' if record['git']['dirty'] else ''}.",
        "",
        "**Per model** (macro-F1 [95% bootstrap CI over headlines])",
        "",
        "| Model | Macro-F1 | Accuracy | 2+ entities | Mixed labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in record["per_model"].items():
        lines.append(
            f"| {name} | {_f1(m['all'])} | {m['all']['accuracy']:.3f} | "
            f"{_f1(m['multi_entity'])} | {_f1(m['mixed_label'])} |"
        )
    lines += [
        "",
        "**B and C: mean over seeds 1-3** (95% CI from a paired bootstrap, macro-F1 averaged "
        "over seeds in each resample)",
        "",
        "| Subset | Headlines | B mean ± std [CI] | C mean ± std [CI] | C - B [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for sub in SUBSETS:
        a = record["architectures"][sub]
        if a is None:
            continue
        cells = [sub, f"{a['n_headlines']:,}"]
        for k in ("B", "C"):
            s = a[k]
            lo, hi = s["ci95_seed_averaged"]
            cells.append(f"{s['mean_macro_f1']:.3f} ± {s['std']:.3f} [{lo:.3f}, {hi:.3f}]")
        d = a["C_minus_B"]
        cells.append(f"{d['mean_difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "**McNemar per seed pair** (per-entity correctness, exact binomial)",
        "",
        "| Pair | B right, C wrong | C right, B wrong | p |",
        "|---|---:|---:|---:|",
    ]
    for pair, m in record["mcnemar"].items():
        lines.append(
            f"| {pair} | {m['b_correct_c_wrong']} | {m['c_correct_b_wrong']} | "
            f"{m['p_value_exact']:.3f} |"
        )
    dec, served = record["architectures"]["decision"], record["served"]
    diff = record["architectures"]["all"]["C_minus_B"]["ci95"]
    lines += [
        "",
        f"**Decision (SPEC.md rule): {dec['chosen']}.** C not significantly worse: "
        f"{'yes' if dec['c_not_significantly_worse'] else 'no'} (CI of C - B "
        f"[{diff[0]:+.4f}, {diff[1]:+.4f}]); C cheaper per headline (M5): "
        f"{'yes' if dec['c_is_cheaper'] else 'no'}.",
        f"**Served checkpoint:** {served['architecture']} seed {served['seed']} (best dev "
        f"macro-F1); its own {split} macro-F1: {_f1(served['metrics']['all'])}. This row is not "
        "the headline number.",
        f"Model A cost on {split}: ${record['models']['A']['cost_usd']:.4f} "
        f"(${record['models']['A']['cost_per_1000_headlines_usd']:.3f} per 1,000 headlines).",
    ]
    return lines


def served_seed(dev_macro_f1_by_seed: dict[int, float]) -> int:
    """Highest dev macro-F1; ties go to the lower seed (SPEC.md)."""
    best = max(dev_macro_f1_by_seed.values())
    return min(s for s, v in dev_macro_f1_by_seed.items() if v == best)
