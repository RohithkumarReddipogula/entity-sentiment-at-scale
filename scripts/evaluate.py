"""M6: evaluate A, B seeds 1-3 and C seeds 1-3 on one split, following SPEC.md exactly.

Usage:
  uv run python scripts/evaluate.py --split dev                      # dry run, repeatable
  uv run python scripts/evaluate.py --split test --confirm-one-time-test-run

Test-split guards: the explicit flag; a clean git tree (the record must name the exact code);
no earlier test record; and a lock file created at the start. A crash before the record is
written can be resumed with --resume: B and C are deterministic on CPU and A's answers are cached,
so a resume recomputes the same predictions rather than taking a second look.

All predictions are computed and saved before any statistic is computed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from entsent.data import REPO_ROOT

os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch

from entsent.bench import verify_checkpoint
from entsent.data import Example, load_split
from entsent.evaluation import (
    architecture_and_decision,
    mcnemar,
    render_report,
    served_seed,
)
from entsent.experiment import load_checkpoint
from entsent.llm_baseline import MODEL, Budget, load_prompt, run
from entsent.metrics import evaluate
from entsent.openrouter import OpenRouterClient, load_api_key
from entsent.results import RESULTS_DIR, git_info, load_results, new_run_id
from entsent.training import build_instances, group_by_headline, make_loader_for, predict

SEEDS = (1, 2, 3)
CONFIGS = {"B": "b_final", "C": "c_final"}
ERROR_SAMPLE_SEED = 0


def final_records() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in load_results():
        for arch, name in CONFIGS.items():
            if r.get("purpose") == "final" and r["config"]["name"] == name:
                key = (arch, r["seed"]["train"])
                if key in out:
                    raise RuntimeError(f"two final records for {key}")
                out[key] = r
    missing = [(a, s) for a in CONFIGS for s in SEEDS if (a, s) not in out]
    if missing:
        raise RuntimeError(f"missing final records: {missing}")
    return out


def m5_c_is_cheaper() -> tuple[bool, str]:
    files = sorted((RESULTS_DIR / "bench").glob("m5_bench_*.json"))
    if not files:
        raise RuntimeError("no M5 benchmark record")
    bench = json.loads(files[-1].read_text())
    return bool(bench["decision"]["c_is_cheaper"]), bench["run_id"]


def guard_test_run(out_dir: Path, resume: bool) -> None:
    if git_info()["dirty"] is not False:
        sys.exit("refusing: the git tree must be clean so the record names the exact code")
    if list(out_dir.glob("m6_test_*.json")):
        sys.exit("refusing: a test record already exists; the test split is used once")
    lock = out_dir / "LOCK"
    if lock.exists() and not resume:
        sys.exit("refusing: LOCK exists (an earlier test session started); see --resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not lock.exists():
        lock.write_text(json.dumps({"started": new_run_id("lock")[1], "git": git_info()}) + "\n")


def predict_model_a(
    examples: list[Example], budget_usd: float
) -> tuple[list[list[str]], dict[str, Any]]:
    system_prompt, prompt_sha = load_prompt("v1")
    budget = Budget(budget_usd)

    async def go() -> Any:
        client = OpenRouterClient(load_api_key())
        try:
            return await run(examples, client, system_prompt, budget, concurrency=8)
        finally:
            await client.aclose()

    results = asyncio.run(go())
    attempts = [a for r in results for a in r.attempts]
    info = {
        "llm": MODEL,
        "prompt_version": "v1",
        "prompt_sha256": prompt_sha,
        "cost_usd": sum(r.cost_usd for r in results),
        "cost_per_1000_headlines_usd": 1000 * sum(r.cost_usd for r in results) / len(results),
        "new_api_spend_usd": budget.spent_usd,
        "cached_headlines": sum(r.cached for r in results),
        "invalid_headlines": sum(not r.valid for r in results),
        "api_calls": len(attempts),
    }
    return [list(r.pred) for r in results], info


def predict_checkpoint(record: dict[str, Any], examples: list[Example]) -> list[list[str]]:
    ckpt = REPO_ROOT / record["checkpoint"]["path"]
    verify_checkpoint(ckpt, record["checkpoint"]["model_sha256"])
    model, tokenizer, cfg = load_checkpoint(ckpt)
    instances = build_instances(cfg, examples)
    loader = make_loader_for(cfg, instances, tokenizer, shuffle=False)
    ids, _, pred = group_by_headline(instances, predict(model, loader, torch.device("cpu")))
    if ids != [ex.id for ex in examples]:
        raise RuntimeError("prediction order does not match the split order")
    return pred


def write_predictions(path: Path, examples: list[Example], pred: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex, ps in zip(examples, pred, strict=True):
            for i, (ent, p) in enumerate(zip(ex.entities, ps, strict=True)):
                f.write(json.dumps({"id": ex.id, "entity": i, "gold": ent.label, "pred": p}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--confirm-one-time-test-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--error-sample", type=int, default=40)
    args = parser.parse_args()

    is_test = args.split == "test"
    out_dir = RESULTS_DIR / ("test" if is_test else "dry_run")
    if is_test:
        if not args.confirm_one_time_test_run:
            sys.exit("refusing: the test split needs --confirm-one-time-test-run")
        guard_test_run(out_dir, args.resume)

    git = git_info()
    examples = load_split(args.split, allow_test=is_test)
    gold = [[e.label for e in ex.entities] for ex in examples]
    records = final_records()
    c_is_cheaper, bench_id = m5_c_is_cheaper()
    run_id, timestamp = new_run_id(f"m6_{args.split}")
    pred_dir = out_dir / "predictions" / run_id

    # 1. Every prediction first, saved before any statistic.
    print(f"{args.split}: {len(examples)} headlines, {sum(map(len, gold))} entities", flush=True)
    preds: dict[str, list[list[str]]] = {}
    preds["A"], a_info = predict_model_a(examples, args.budget_usd)
    print(
        f"A done: cost ${a_info['cost_usd']:.4f}, new spend ${a_info['new_api_spend_usd']:.4f}, "
        f"cached {a_info['cached_headlines']}, invalid {a_info['invalid_headlines']}",
        flush=True,
    )
    for model_id in CONFIGS:
        for s in SEEDS:
            preds[f"{model_id}{s}"] = predict_checkpoint(records[(model_id, s)], examples)
            print(f"{model_id} seed {s} done", flush=True)
    for name, p in preds.items():
        write_predictions(pred_dir / f"{name}.jsonl", examples, p)

    # 2. Statistics, exactly as fixed in SPEC.md.
    per_model = {name: evaluate(gold, p) for name, p in preds.items()}
    arch = architecture_and_decision(
        gold, [preds[f"B{s}"] for s in SEEDS], [preds[f"C{s}"] for s in SEEDS], c_is_cheaper
    )
    pairs = {f"seed{s}": mcnemar(gold, preds[f"B{s}"], preds[f"C{s}"]) for s in SEEDS}
    chosen = arch["decision"]["chosen"]
    dev_f1 = {s: records[(chosen, s)]["metrics"]["all"]["macro_f1"] for s in SEEDS}
    seed = served_seed(dev_f1)
    served_name = f"{chosen}{seed}"

    # 3. Error sample from the served checkpoint (fixed sampling seed).
    wrong = [
        (i, j)
        for i, (gs, ps) in enumerate(zip(gold, preds[served_name], strict=True))
        for j, (g, p) in enumerate(zip(gs, ps, strict=True))
        if g != p
    ]
    sample = sorted(
        random.Random(ERROR_SAMPLE_SEED).sample(wrong, min(args.error_sample, len(wrong)))
    )
    error_rows = [
        {
            "id": examples[i].id,
            "entity_index": j,
            "gold": gold[i][j],
            "served_pred": preds[served_name][i][j],
            "A_pred": preds["A"][i][j],
            "other_arch_same_seed_pred": preds[("C" if chosen == "B" else "B") + str(seed)][i][j],
            "n_entities": len(gold[i]),
        }
        for i, j in sample
    ]
    private = out_dir / "private" / f"{run_id}_errors.jsonl"  # contains headline text: gitignored
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open("w") as f:
        for row, (i, j) in zip(error_rows, sample, strict=True):
            ent = examples[i].entities[j]
            f.write(json.dumps(row | {"text": examples[i].text, "entity": ent.name}) + "\n")

    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "purpose": "test_run" if is_test else "dry_run",
        "split": args.split,
        "n_headlines": len(examples),
        "n_entities": sum(map(len, gold)),
        "git": git,
        "protocol": "SPEC.md, Seeds and test reporting protocol",
        "models": {
            "A": a_info,
            **{
                f"{a}{s}": {
                    "run_id": records[(a, s)]["run_id"],
                    "model_sha256": records[(a, s)]["checkpoint"]["model_sha256"],
                }
                for a in CONFIGS
                for s in SEEDS
            },
        },
        "per_model": per_model,
        "architectures": arch,
        "mcnemar": pairs,
        "m5_bench": bench_id,
        "served": {
            "architecture": chosen,
            "seed": seed,
            "dev_macro_f1_by_seed": dev_f1,
            "metrics": per_model[served_name],
        },
        "error_sample": {
            "model": served_name,
            "n_wrong_entities": len(wrong),
            "sampling_seed": ERROR_SAMPLE_SEED,
            "rows": error_rows,
        },
    }
    if not is_test:  # the dry run must reproduce the recorded dev numbers exactly
        checks = {
            f"{a}{s}": per_model[f"{a}{s}"]["all"]["macro_f1"]
            == records[(a, s)]["metrics"]["all"]["macro_f1"]
            for a in CONFIGS
            for s in SEEDS
        }
        m1 = [r for r in load_results() if r["model_id"] == "A" and r["purpose"] == "eval"][-1]
        checks["A"] = per_model["A"]["all"]["macro_f1"] == m1["metrics"]["all"]["macro_f1"]
        record["dry_run_checks"] = checks

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    print("\n".join(render_report(record)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
