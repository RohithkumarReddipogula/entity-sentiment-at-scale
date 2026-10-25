"""M5: CPU speed and cost benchmark for models B and C (seed-1 checkpoints), dev headlines.

Usage: uv run python scripts/bench.py [--reps 3] [--threads 4 1]

The parent launches one fresh child process per (threads, repetition, model), interleaved
B, C, B, C, ... and writes results/bench/m5_bench_<timestamp>.json. The "cheaper" rule is in
LOG.md (M5) and was fixed before this script was written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from entsent.data import REPO_ROOT

os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from entsent.results import RESULTS_DIR, git_info, load_results, new_run_id

BENCH_DIR = RESULTS_DIR / "bench"
MODELS = {"B": "b_final", "C": "c_final"}


def seed1_record(config_name: str) -> dict[str, Any]:
    matches = [
        r
        for r in load_results()
        if r.get("purpose") == "final"
        and r["config"]["name"] == config_name
        and r["seed"]["train"] == 1
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one final seed-1 record for {config_name}, got {len(matches)}"
        )
    return matches[0]


def recorded_predictions(run_id: str) -> list[list[str]]:
    by_id: dict[int, list[str]] = {}
    for line in (RESULTS_DIR / "predictions" / f"{run_id}.jsonl").read_text().splitlines():
        row = json.loads(line)
        by_id.setdefault(row["id"], []).append(row["pred"])
    return list(by_id.values())


def child(model_id: str, threads: int, out: Path) -> None:
    import torch

    from entsent.bench import (
        THROUGHPUT_BATCH,
        WARMUP_HEADLINES,
        count_mismatches,
        latency_summary,
        run_pass,
        throughput_summary,
        verify_checkpoint,
    )
    from entsent.data import load_split
    from entsent.experiment import load_checkpoint

    torch.set_num_threads(threads)
    record = seed1_record(MODELS[model_id])
    ckpt = REPO_ROOT / record["checkpoint"]["path"]
    digest = verify_checkpoint(ckpt, record["checkpoint"]["model_sha256"])

    start = time.perf_counter()
    model, tokenizer, cfg = load_checkpoint(ckpt)
    load_s = time.perf_counter() - start
    dev = load_split("dev")
    recorded = recorded_predictions(record["run_id"])

    run_pass(model, cfg, tokenizer, dev[:WARMUP_HEADLINES], 1)  # warm-up, not timed
    latency = run_pass(model, cfg, tokenizer, dev, 1)
    run_pass(model, cfg, tokenizer, dev[: 2 * THROUGHPUT_BATCH], THROUGHPUT_BATCH)  # warm-up
    batched = run_pass(model, cfg, tokenizer, dev, THROUGHPUT_BATCH)

    out.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "run_id": record["run_id"],
                "threads": threads,
                "torch_threads": torch.get_num_threads(),
                "checkpoint_sha256": digest,
                "load_s": load_s,
                "n_headlines": latency.n_headlines,
                "passes_per_1000_headlines": 1000 * latency.n_passes / latency.n_headlines,
                "latency": latency_summary(latency),
                "throughput": throughput_summary(batched),
                "mismatches": {
                    "latency_pass": count_mismatches(latency.preds, recorded),
                    "batched_pass": count_mismatches(batched.preds, recorded),
                },
            }
        )
    )


def environment() -> dict[str, Any]:
    import torch

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "machine": platform.machine(),
        "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_cores_performance": run("sysctl", "-n", "hw.perflevel0.physicalcpu"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "power": run("pmset", "-g", "batt").splitlines()[0:2],
    }


def median_of(reps: list[dict[str, Any]], *keys: str) -> float:
    values = []
    for r in reps:
        v: Any = r
        for k in keys:
            v = v[k]
        values.append(float(v))
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 1])
    parser.add_argument("--child", choices=list(MODELS))
    parser.add_argument("--child-threads", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.child:
        child(args.child, args.child_threads, args.out)
        return 0

    from entsent.bench import is_cheaper

    git = git_info()
    env = environment()
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for threads in args.threads:
            for rep in range(1, args.reps + 1):
                for model_id in MODELS:  # interleaved: B, C, B, C, ...
                    out = Path(tmp) / f"{model_id}_{threads}_{rep}.json"
                    child_env = os.environ | {
                        "OMP_NUM_THREADS": str(threads),
                        "TRANSFORMERS_VERBOSITY": "error",
                    }
                    cmd = [sys.executable, __file__, "--child", model_id]
                    cmd += ["--child-threads", str(threads), "--out", str(out)]
                    subprocess.run(cmd, env=child_env, check=True)
                    result = json.loads(out.read_text()) | {"rep": rep}
                    runs.append(result)
                    print(
                        f"threads {threads} rep {rep} {model_id}: latency median "
                        f"{result['latency']['median_ms']:.2f} ms, CPU s / 1k headlines "
                        f"{result['throughput']['cpu_s_per_1000_headlines']:.2f}, "
                        f"mismatches {result['mismatches']}",
                        flush=True,
                    )

    summary: dict[str, Any] = {}
    for threads in args.threads:
        for model_id in MODELS:
            reps = [r for r in runs if r["threads"] == threads and r["model_id"] == model_id]
            summary[f"{model_id}_threads{threads}"] = {
                "passes_per_1000_headlines": reps[0]["passes_per_1000_headlines"],
                "latency_median_ms": median_of(reps, "latency", "median_ms"),
                "latency_p95_ms": median_of(reps, "latency", "p95_ms"),
                "latency_median_ms_by_entities": {
                    k: statistics.median(r["latency"]["by_entities"][k]["median_ms"] for r in reps)
                    for k in ("1", "2", "3+")
                },
                "headlines_per_s": median_of(reps, "throughput", "headlines_per_s"),
                "entities_per_s": median_of(reps, "throughput", "entities_per_s"),
                "cpu_s_per_1000_headlines": median_of(
                    reps, "throughput", "cpu_s_per_1000_headlines"
                ),
                "cpu_s_per_1000_headlines_by_rep": [
                    r["throughput"]["cpu_s_per_1000_headlines"] for r in reps
                ],
                "mismatches_total": sum(sum(r["mismatches"].values()) for r in reps),
            }

    b = summary["B_threads4"]["cpu_s_per_1000_headlines_by_rep"]
    c = summary["C_threads4"]["cpu_s_per_1000_headlines_by_rep"]
    run_id, timestamp = new_run_id("m5_bench")
    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "purpose": "bench",
        "split": "dev",
        "git": git,
        "environment": env,
        "config": {
            "reps": args.reps,
            "threads": args.threads,
            "throughput_batch_headlines": 32,
            "warmup_headlines": 50,
            "order": "fresh process per measurement, interleaved B, C, B, C",
        },
        "models": {m: seed1_record(name)["run_id"] for m, name in MODELS.items()},
        "summary": summary,
        "decision": {
            "rule": "C cheaper only if its 4-thread batched CPU time per 1,000 headlines is "
            "lower than B's in every repetition (LOG.md, M5)",
            "b_cpu_s_per_1000_by_rep": b,
            "c_cpu_s_per_1000_by_rep": c,
            "c_is_cheaper": is_cheaper(c, b),
        },
        "runs": runs,
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    print(f"C is cheaper per headline: {record['decision']['c_is_cheaper']}  (B {b}, C {c})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
