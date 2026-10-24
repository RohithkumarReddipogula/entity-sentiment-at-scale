"""M1: run the zero-shot LLM baseline (model A) on train (smoke test) or dev.

Usage:
  uv run python scripts/run_llm.py --split train --limit 20 --purpose smoke
  uv run python scripts/run_llm.py --split dev

The test split is not available here; it is used once, in M6.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys

import numpy as np

from entsent.data import load_split
from entsent.llm_baseline import (
    INVALID_RETRIES,
    MAX_TOKENS,
    MODEL,
    PRICE_INPUT,
    PRICE_OUTPUT,
    Budget,
    BudgetExceededError,
    HeadlineResult,
    load_prompt,
    run,
)
from entsent.metrics import BOOTSTRAP_SEED, N_BOOTSTRAP, evaluate
from entsent.openrouter import OpenRouterClient, OpenRouterError, load_api_key
from entsent.results import PREDICTIONS_DIR, git_info, new_run_id, write_result


def summarize_cost_speed(results: list[HeadlineResult]) -> dict[str, object]:
    attempts = [a for r in results for a in r.attempts]
    n = len(results)
    reported = sum(r.cost_usd for r in results)
    listed = sum(a.listed_cost_usd for a in attempts)
    latencies = [r.latency_s for r in results]
    return {
        "cost": {
            "total_usd": reported,
            "per_1000_headlines_usd": 1000 * reported / n,
            "listed_price_total_usd": listed,
            "reported_vs_listed_ratio": reported / listed if listed else None,
            "missing_reported_cost_calls": sum(a.cost_usd is None for a in attempts),
            "input_tokens": sum(a.input_tokens for a in attempts),
            "output_tokens": sum(a.output_tokens for a in attempts),
            "reasoning_tokens": sum(a.reasoning_tokens for a in attempts),
        },
        "speed": {
            "api_calls": len(attempts),
            "http_requests": sum(a.http_attempts for a in attempts),
            "passes_per_1000_headlines": 1000 * len(attempts) / n,
            "latency_median_s": statistics.median(latencies),
            "latency_p95_s": float(np.percentile(latencies, 95)),
            "latency_note": "network round trip per headline incl. invalid-answer retries; "
            "not comparable to local CPU latency of models B and C",
        },
        "served_models": sorted({a.model for a in attempts}),
        "cached_headlines": sum(r.cached for r in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--limit", type=int, help="random sample of this many headlines")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--purpose", choices=["smoke", "eval"], default="eval")
    args = parser.parse_args()

    examples = load_split(args.split)
    if args.limit:
        examples = sorted(
            random.Random(args.sample_seed).sample(examples, args.limit), key=lambda e: e.id
        )
    system_prompt, prompt_sha = load_prompt(args.prompt)
    budget = Budget(args.budget_usd)

    async def go() -> list[HeadlineResult]:
        client = OpenRouterClient(load_api_key())
        try:
            return await run(examples, client, system_prompt, budget, concurrency=args.concurrency)
        finally:
            await client.aclose()

    try:
        results = asyncio.run(go())
    except (OpenRouterError, BudgetExceededError) as exc:
        print(f"run aborted: {exc}", file=sys.stderr)
        print("finished headlines are cached; rerunning resumes without paying again")
        return 1

    run_id, timestamp = new_run_id(f"a_llm_{args.split}_{args.purpose}")
    metrics = evaluate([r.gold for r in results], [r.pred for r in results])
    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_id": "A",
        "model_name": "LLM baseline (zero-shot)",
        "purpose": args.purpose,
        "split": args.split,
        "n_headlines": len(results),
        "n_entities": sum(len(r.gold) for r in results),
        "invalid_headlines": sum(not r.valid for r in results),
        "git": git_info(),
        "config": {
            "llm": MODEL,
            "provider": "OpenRouter",
            "prompt_version": args.prompt,
            "prompt_sha256": prompt_sha,
            "reasoning": "disabled",
            "temperature": "not supported by the model on OpenRouter (provider default)",
            "structured_output": "json_schema, strict",
            "max_tokens": MAX_TOKENS,
            "invalid_answer_retries": INVALID_RETRIES,
            "concurrency": args.concurrency,
            "budget_usd": args.budget_usd,
            "limit": args.limit,
            "sample_seed": args.sample_seed if args.limit else None,
            "listed_price_usd_per_m_tokens": {
                "input": PRICE_INPUT * 1e6,
                "output": PRICE_OUTPUT * 1e6,
            },
        },
        "seed": {"bootstrap": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP},
        "metrics": metrics,
        **summarize_cost_speed(results),
    }
    path = write_result(record)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with (PREDICTIONS_DIR / f"{run_id}.jsonl").open("w") as f:
        for r in results:
            for i, (g, p) in enumerate(zip(r.gold, r.pred, strict=True)):
                f.write(json.dumps({"id": r.id, "entity": i, "gold": g, "pred": p}) + "\n")

    m = metrics["all"]
    print(f"wrote {path.relative_to(path.parents[1])}")
    print(
        f"macro-F1 {m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.3f}, "
        f"{m['macro_f1_ci95'][1]:.3f}]  accuracy {m['accuracy']:.3f}  "
        f"invalid headlines {record['invalid_headlines']}"
    )
    for sub in ("multi_entity", "mixed_label"):
        s = metrics[sub]
        if s is None:
            print(f"  {sub}: no headlines in this sample")
        else:
            print(f"  {sub}: macro-F1 {s['macro_f1']:.3f} on {s['n_headlines']} headlines")
    cost, speed = record["cost"], record["speed"]
    print(json.dumps({"cost": cost, "speed": speed, "served": record["served_models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
