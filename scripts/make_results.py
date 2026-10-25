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


README_MD = REPO_ROOT / "README.md"
README_MARKERS = ("<!-- BEGIN GENERATED: results -->", "<!-- END GENERATED: results -->")


def readme_block() -> str:
    """The README's first-screen results table and decision, built only from result files."""
    test = json.loads(sorted((RESULTS_DIR / "test").glob("m6_test_*.json"))[-1].read_text())
    bench = json.loads(sorted((RESULTS_DIR / "bench").glob("m5_bench_*.json"))[-1].read_text())
    m1 = [r for r in load_results() if r["model_id"] == "A" and r["purpose"] == "eval"][-1]
    arch, per, s = test["architectures"], test["per_model"], bench["summary"]
    served = test["served"]
    served_name = f"{served['architecture']}{served['seed']}"

    def ci(value: float, bounds: list[float]) -> str:
        return f"{value:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"

    def arch_row(key: str, label: str) -> str:
        a, b = arch["all"][key], s[f"{key}_threads4"]
        return (
            f"| {label} | {ci(a['mean_macro_f1'], a['ci95_seed_averaged'])} | "
            f"{arch['multi_entity'][key]['mean_macro_f1']:.3f} | "
            f"{arch['mixed_label'][key]['mean_macro_f1']:.3f} | "
            f"{b['passes_per_1000_headlines']:,.0f} | {b['cpu_s_per_1000_headlines']:.2f} | "
            f"{b['latency_median_ms']:.1f} / {b['latency_p95_ms']:.1f} ms | not priced (CPU) |"
        )

    a = per["A"]
    diff = arch["all"]["C_minus_B"]
    cpu_b, cpu_c = (
        s["B_threads4"]["cpu_s_per_1000_headlines"],
        s["C_threads4"]["cpu_s_per_1000_headlines"],
    )
    lines = [
        f"Test set: {test['n_headlines']:,} headlines, {test['n_entities']:,} entities, "
        "used once. B and C are the mean of 3 training seeds.",
        "",
        "| Model | Test macro-F1 [95% CI] | 2+ entities | Mixed labels | Forward passes "
        "/ 1k headlines | CPU s / 1k headlines | Latency median / p95 | Cost / 1k headlines |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| A: zero-shot LLM ({test['models']['A']['llm']}) | "
        f"{ci(a['all']['macro_f1'], a['all']['macro_f1_ci95'])} | "
        f"{a['multi_entity']['macro_f1']:.3f} | {a['mixed_label']['macro_f1']:.3f} | "
        f"{m1['speed']['passes_per_1000_headlines']:,.0f} API calls | remote | "
        f"{m1['speed']['latency_median_s'] * 1000:,.0f} / "
        f"{m1['speed']['latency_p95_s'] * 1000:,.0f} ms (network, dev) | "
        f"${test['models']['A']['cost_per_1000_headlines_usd']:.3f} |",
        arch_row("B", "B: per-entity encoder"),
        arch_row("C", "C: single-pass encoder"),
        f"| {served['architecture']} seed {served['seed']} (the served model) | "
        f"{ci(per[served_name]['all']['macro_f1'], per[served_name]['all']['macro_f1_ci95'])} | "
        f"{per[served_name]['multi_entity']['macro_f1']:.3f} | "
        f"{per[served_name]['mixed_label']['macro_f1']:.3f} | | | | |",
        "",
        f"CPU time and latency: {bench['environment']['chip']}, macOS, 4 torch threads; CPU "
        "time in batches of 32 headlines, latency one headline at a time, end to end. These "
        "numbers depend on the machine (see Limitations).",
        "",
        f"**Decision: {arch['decision']['chosen']}.** The rule was written before any test "
        "result: choose C if its macro-F1 is not significantly worse than B's and it is "
        f"cheaper. C minus B is {diff['mean_difference']:+.4f} with a 95% CI of "
        f"[{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}], which includes zero. C uses "
        f"{cpu_c:.2f} CPU seconds per 1,000 headlines against B's {cpu_b:.2f}, "
        f"{100 * (1 - cpu_c / cpu_b):.0f}% less. So C is served, as "
        f"{served['architecture']} seed {served['seed']}, the seed with the best dev score.",
    ]
    return "\n".join(lines) + "\n"


def update_readme() -> None:
    begin, end = README_MARKERS
    text = README_MD.read_text()
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    README_MD.write_text(f"{head}{begin}\n{readme_block()}{end}{tail}")


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
    if README_MD.exists():
        update_readme()
        print("README.md: results block updated")


if __name__ == "__main__":
    main()
