"""Experiment records: every run writes one JSON file to results/ (see SPEC.md)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from entsent.data import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def new_run_id(prefix: str) -> tuple[str, str]:
    """Return (run_id, ISO timestamp)."""
    now = datetime.now(UTC)
    return f"{prefix}_{now.strftime('%Y%m%dT%H%M%SZ')}", now.isoformat(timespec="seconds")


def write_result(record: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    path = results_dir / f"{record['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    records = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    return sorted(records, key=lambda r: r["timestamp"])
