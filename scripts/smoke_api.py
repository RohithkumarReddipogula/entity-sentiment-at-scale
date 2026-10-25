"""M7 smoke test against a running server (local `make serve` or the Docker container).

Sends N dev headlines to POST /predict and checks the labels against the recorded predictions of
the served checkpoint. Prints latency. Usage:
  uv run python scripts/smoke_api.py [--url http://127.0.0.1:8000] [--n 100]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

from entsent.data import load_split
from entsent.results import RESULTS_DIR, load_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--wait-s", type=float, default=90)
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait_s
    while True:
        try:
            health = httpx.get(f"{args.url}/health", timeout=2)
            if health.status_code == 200:
                break
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            print("server not healthy in time", file=sys.stderr)
            return 1
        time.sleep(1)
    served = health.json()
    print(f"health: {served}")

    # The served weights' SHA-256 identifies the training run whose dev predictions to compare.
    runs = [
        r
        for r in load_results()
        if r.get("checkpoint", {}).get("model_sha256") == served["sha256"]
        and r["purpose"] == "final"
    ]
    if len(runs) != 1:
        print(f"no unique final run with sha256 {served['sha256']}", file=sys.stderr)
        return 1
    print(f"served weights = training run {runs[0]['run_id']}")
    recorded: dict[int, list[str]] = {}
    lines = (RESULTS_DIR / "predictions" / f"{runs[0]['run_id']}.jsonl").read_text().splitlines()
    for line in lines:
        row = json.loads(line)
        recorded.setdefault(row["id"], []).append(row["pred"])

    mismatches, ms = 0, []
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for ex in load_split("dev")[: args.n]:
            ents = [{"spans": [list(s) for s in e.spans]} for e in ex.entities]
            start = time.perf_counter()
            r = client.post("/predict", json={"text": ex.text, "entities": ents})
            ms.append((time.perf_counter() - start) * 1000)
            r.raise_for_status()
            labels = [e["label"] for e in r.json()["entities"]]
            mismatches += sum(a != b for a, b in zip(labels, recorded[ex.id], strict=True))
        bad = client.post("/predict", json={"text": "x", "entities": [{"spans": [[0, 9]]}]})
    print(f"{args.n} headlines: {mismatches} label mismatches vs recorded dev predictions")
    print(f"HTTP round trip: median {statistics.median(ms):.1f} ms, max {max(ms):.1f} ms")
    print(f"invalid request -> HTTP {bad.status_code}")
    return 0 if mismatches == 0 and bad.status_code == 422 else 1


if __name__ == "__main__":
    raise SystemExit(main())
