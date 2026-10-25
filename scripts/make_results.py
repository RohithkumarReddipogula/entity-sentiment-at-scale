"""Regenerate RESULTS.md from the JSON files in results/. Never edit RESULTS.md by hand."""

from __future__ import annotations

import json
import statistics
from typing import Any

from entsent.data import REPO_ROOT
from entsent.evaluation import render_report
from entsent.results import RESULTS_DIR, load_results

RESULTS_MD = REPO_ROOT / "RESULTS.md"


def f1_cell(m: dict[str, Any] | None) -> str:
    if m is None:
        return "n/a"
    lo, hi = m["macro_f1_ci95"]
    return f"{m['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}]"


def row(r: dict[str, Any]) -> str:
    m = r["metrics"]
    cost = r.get("cost", {}).get("per_1000_headlines_usd")
    speed = r.get("speed", {})
    cells = [
        f"`{r['run_id']}`",
        r["model_id"],
        r["purpose"],
        r["split"],
        f"{r['n_headlines']:,} / {r['n_entities']:,}",
        f1_cell(m["all"]),
        f"{m['all']['accuracy']:.3f}",
        f1_cell(m["multi_entity"]),
        f1_cell(m["mixed_label"]),
        str(r.get("invalid_headlines", "")),
        f"{speed['passes_per_1000_headlines']:,.0f}" if speed else "",
        f"{speed['latency_median_s']:.2f} / {speed['latency_p95_s']:.2f}" if speed else "",
        f"${cost:.4f}" if cost is not None else "",
    ]
    return "| " + " | ".join(cells) + " |"


SECTIONS = [
    (
        "final",
        "Final encoder models (dev, seeds 1-3)",
        "Dev scores use best-epoch selection on dev, so they are slightly optimistic. Test "
        "results follow the protocol in SPEC.md and appear only after M6.",
    ),
    (
        "encoder_check",
        "Encoder check (M2, dev, model B)",
        "Rule (LOG.md, fixed before training): choose deberta-v3-small only if its mean dev "
        "macro-F1 is at least 0.020 higher than distilroberta-base; otherwise distilroberta-base.",
    ),
]


def multi_seed(records: list[dict[str, Any]], purpose: str, title: str, note: str) -> list[str]:
    """Mean and spread of dev macro-F1 over seeds, per config, for one purpose."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r["purpose"] == purpose:
            runs.setdefault(r["config"]["name"], []).append(r)
    if not runs:
        return []
    lines = [
        f"## {title}",
        "",
        note,
        "",
        "| Config | Encoder | Seeds | Dev macro-F1 per seed | Mean | Std | "
        "Best epochs | Train time per run (min) |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for name, rs in sorted(runs.items()):
        rs = sorted(rs, key=lambda r: r["seed"]["train"])
        f1 = [r["metrics"]["all"]["macro_f1"] for r in rs]
        minutes = statistics.mean(r["training"]["train_seconds"] for r in rs) / 60
        cells = [
            f"`{name}`",
            rs[0]["config"]["encoder"]["name"],
            ", ".join(str(r["seed"]["train"]) for r in rs),
            ", ".join(f"{v:.3f}" for v in f1),
            f"{statistics.mean(f1):.3f}",
            f"{statistics.stdev(f1):.3f}" if len(f1) > 1 else "",
            ", ".join(str(r["training"]["best_epoch"]) for r in rs),
            f"{minutes:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def speed_section() -> list[str]:
    """M5: the latest benchmark in results/bench/, plus model A's M1 numbers for reference."""
    files = sorted((RESULTS_DIR / "bench").glob("m5_bench_*.json"))
    if not files:
        return []
    bench = json.loads(files[-1].read_text())
    s, d, env = bench["summary"], bench["decision"], bench["environment"]
    lines = [
        "## Speed and cost (M5, CPU, dev headlines)",
        "",
        f"`{bench['run_id']}`: {env['chip']}, torch {env['torch']}, seed-1 checkpoints, "
        f"{bench['config']['reps']} interleaved repetitions (medians shown), "
        "fresh process per measurement.",
        "",
        "| Model | Threads | Passes / 1k headlines | Latency median / p95 (ms) | "
        "Median by entities 1 / 2 / 3+ (ms) | Headlines / s (batch 32) | "
        "CPU s / 1k headlines | Prediction mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in s.items():
        model, threads = key.split("_threads")
        by = m["latency_median_ms_by_entities"]
        cells = [
            model,
            threads,
            f"{m['passes_per_1000_headlines']:,.0f}",
            f"{m['latency_median_ms']:.2f} / {m['latency_p95_ms']:.2f}",
            f"{by['1']:.2f} / {by['2']:.2f} / {by['3+']:.2f}",
            f"{m['headlines_per_s']:.0f}",
            f"{m['cpu_s_per_1000_headlines']:.2f}",
            str(m["mismatches_total"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    a = [r for r in load_results() if r["model_id"] == "A" and r["purpose"] == "eval"]
    if a:
        sp, cost = a[-1]["speed"], a[-1]["cost"]
        lines.append(
            f"| A (reference) | network | {sp['passes_per_1000_headlines']:,.0f} | "
            f"{sp['latency_median_s'] * 1000:.0f} / {sp['latency_p95_s'] * 1000:.0f} | | | | "
            f"${cost['per_1000_headlines_usd']:.3f} / 1k headlines |"
        )
    verdict = "yes" if d["c_is_cheaper"] else "no"
    lines += [
        "",
        f"**C cheaper per headline (LOG.md rule): {verdict}.** 4-thread CPU s / 1k headlines "
        f"by repetition: B {', '.join(f'{v:.2f}' for v in d['b_cpu_s_per_1000_by_rep'])}; "
        f"C {', '.join(f'{v:.2f}' for v in d['c_cpu_s_per_1000_by_rep'])}.",
        "Model A's latency is a network round trip and not comparable; its USD cost is from M1.",
        "",
    ]
    return lines


def test_section() -> list[str]:
    """M6: the one-time test run, rendered from its record (absent until M6)."""
    files = sorted((RESULTS_DIR / "test").glob("m6_test_*.json"))
    if not files:
        return []
    record = json.loads(files[-1].read_text())
    lines = ["## Test set (M6, used once)", "", *render_report(record), ""]
    analysis = RESULTS_DIR / "test" / f"error_analysis_{record['run_id']}.json"
    if analysis.exists():
        ea = json.loads(analysis.read_text())
        lines += [
            f"**Error analysis** ({ea['n_sampled']} of {ea['n_wrong_entities_total']} wrong "
            f"entities of {ea['model']}, sampling seed {ea['sampling_seed']}, read by hand; "
            "patterns in LOG.md)",
            "",
            "| Pattern | Errors | A right | B (same seed) right | In 2+ entity headlines |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, p in ea["patterns"].items():
            lines.append(
                f"| {name.replace('_', ' ')} | {p['count']} | {p['A_right']} | "
                f"{p['B_same_seed_right']} | {p['multi_entity']} |"
            )
        lines += [
            f"| **total** | {ea['n_sampled']} | {ea['A_right_total']} | "
            f"{ea['B_same_seed_right_total']} | |",
            "",
        ]
    return lines


def main() -> None:
    records = load_results()
    header = (
        "| Run | Model | Purpose | Split | Headlines / entities | Macro-F1 [95% CI] "
        "| Accuracy | Macro-F1, 2+ entities | Macro-F1, mixed labels | Invalid "
        "| Passes / 1k | Latency median / p95 (s) | USD / 1k headlines |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [
        "# Results",
        "",
        "Generated by `scripts/make_results.py` from `results/*.json`. Do not edit by hand.",
        "",
        "Smoke runs check plumbing on a few train headlines; they are not evaluations.",
        "LLM latency is a network round trip and is not comparable to local CPU latency.",
        "",
        *test_section(),
        *(line for section in SECTIONS for line in multi_seed(records, *section)),
        *speed_section(),
        "## All runs",
        "",
        header,
        *(row(r) for r in records),
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines))
    print(f"RESULTS.md: {len(records)} runs")


if __name__ == "__main__":
    main()
