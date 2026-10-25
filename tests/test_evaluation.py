"""M6 statistics and the one-time test-run guards. CI-safe: synthetic labels, no models."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from entsent.data import LABELS
from entsent.evaluation import (
    architecture_and_decision,
    macro_f1,
    mcnemar,
    per_headline_confusions,
    render_report,
    served_seed,
)
from entsent.metrics import bootstrap_macro_f1_ci, evaluate, scores

REPO = Path(__file__).resolve().parents[1]


def synthetic(n: int = 300, accuracy: float = 0.8, seed: int = 0) -> tuple[list[list[str]], ...]:
    rng = np.random.default_rng(seed)
    gold = [list(rng.choice(LABELS, size=rng.integers(1, 4))) for _ in range(n)]
    return gold, noisy(gold, accuracy, seed)


def noisy(gold: list[list[str]], accuracy: float, seed: int) -> list[list[str]]:
    """Predictions for ``gold`` that keep each label with probability ``accuracy``."""
    rng = np.random.default_rng(seed + 1000)
    return [[g if rng.random() < accuracy else str(rng.choice(LABELS)) for g in gs] for gs in gold]


def test_macro_f1_matches_metrics_module() -> None:
    gold, pred = synthetic()
    cms = per_headline_confusions(gold, pred)
    assert macro_f1(cms) == pytest.approx(scores(cms.sum(axis=0))["macro_f1"])


def test_seed_average_of_identical_seeds_equals_single_model_bootstrap() -> None:
    gold, pred = synthetic()
    out = architecture_and_decision(gold, [pred] * 3, [pred] * 3, True, n_resamples=2_000)
    b = out["all"]["B"]
    assert b["per_seed"] == [b["mean_macro_f1"]] * 3 and b["std"] == 0.0
    single = bootstrap_macro_f1_ci(per_headline_confusions(gold, pred), n_resamples=2_000)
    assert b["ci95_seed_averaged"] == pytest.approx(list(single))  # same resampling scheme


def test_identical_architectures_give_zero_difference_and_c_if_cheaper() -> None:
    gold, pred = synthetic()
    out = architecture_and_decision(gold, [pred] * 3, [pred] * 3, True, n_resamples=500)
    assert out["all"]["C_minus_B"]["mean_difference"] == 0.0
    assert out["all"]["C_minus_B"]["ci95"] == [0.0, 0.0]
    assert out["decision"]["chosen"] == "C"
    not_cheaper = architecture_and_decision(gold, [pred] * 3, [pred] * 3, False, n_resamples=500)
    assert not_cheaper["decision"]["chosen"] == "B"  # cheaper is required too


def test_clearly_worse_c_is_rejected_even_if_cheaper() -> None:
    gold, good = synthetic(accuracy=0.9)
    bad = noisy(gold, accuracy=0.5, seed=1)
    out = architecture_and_decision(gold, [good] * 3, [bad] * 3, True, n_resamples=500)
    assert out["all"]["C_minus_B"]["ci95"][1] < 0
    assert out["decision"] == out["decision"] | {"c_not_significantly_worse": False, "chosen": "B"}


def test_subsets_are_reported_and_paired() -> None:
    gold, pred = synthetic()
    out = architecture_and_decision(gold, [pred] * 3, [pred] * 3, True, n_resamples=200)
    assert out["multi_entity"]["n_headlines"] == sum(len(g) >= 2 for g in gold)
    assert out["mixed_label"]["n_headlines"] == sum(len(set(g)) >= 2 for g in gold)


def test_mcnemar_counts_and_exact_p_value() -> None:
    gold = [["positive"]] * 12
    b = [["positive"]] * 8 + [["negative"]] * 2 + [["positive"]] * 2  # 10 right
    c = [["negative"]] * 8 + [["positive"]] * 2 + [["positive"]] * 2  # 4 right
    m = mcnemar(gold, b, c)
    assert (m["b_correct_c_wrong"], m["c_correct_b_wrong"]) == (8, 2)
    assert (m["both_correct"], m["both_wrong"]) == (2, 0)
    assert m["p_value_exact"] == pytest.approx(0.109375)  # 2 * P(X <= 2), X ~ Bin(10, 0.5)
    assert mcnemar(gold, b, b)["p_value_exact"] == 1.0


def test_served_seed_is_best_dev_with_ties_to_lower_seed() -> None:
    assert served_seed({1: 0.87, 2: 0.88, 3: 0.86}) == 2
    assert served_seed({1: 0.88, 2: 0.87, 3: 0.88}) == 1


def test_report_renders_every_section() -> None:
    gold, pred = synthetic(n=60)
    arch = architecture_and_decision(gold, [pred] * 3, [pred] * 3, True, n_resamples=100)
    m = evaluate(gold, pred, n_resamples=100)
    record: dict[str, Any] = {
        "run_id": "m6_dev_x",
        "purpose": "dry_run",
        "split": "dev",
        "n_headlines": 60,
        "n_entities": sum(map(len, gold)),
        "git": {"commit": "abc1234", "dirty": False},
        "per_model": {"A": m, "B1": m},
        "architectures": arch,
        "mcnemar": {"seed1": mcnemar(gold, pred, pred)},
        "served": {"architecture": "C", "seed": 1, "metrics": m},
        "models": {"A": {"cost_usd": 0.1, "cost_per_1000_headlines_usd": 0.08}},
    }
    text = "\n".join(render_report(record))
    for needle in ("Per model", "mean over seeds", "McNemar", "Decision", "Served checkpoint"):
        assert needle in text


# --- one-time test-run guards ---------------------------------------------------------------


def evaluate_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_script", REPO / "scripts/evaluate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_test_split_needs_the_explicit_flag() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/evaluate.py"), "--split", "test"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode != 0 and "--confirm-one-time-test-run" in proc.stderr


def test_guard_refuses_dirty_tree_existing_record_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev = evaluate_script()
    monkeypatch.setattr(ev, "git_info", lambda: {"commit": "x", "dirty": True})
    with pytest.raises(SystemExit, match="clean"):
        ev.guard_test_run(tmp_path, resume=False)

    monkeypatch.setattr(ev, "git_info", lambda: {"commit": "x", "dirty": False})
    ev.guard_test_run(tmp_path, resume=False)  # first start: creates the lock
    assert (tmp_path / "LOCK").exists()
    with pytest.raises(SystemExit, match="LOCK"):
        ev.guard_test_run(tmp_path, resume=False)
    ev.guard_test_run(tmp_path, resume=True)  # resume after a crash is allowed

    (tmp_path / "m6_test_done.json").write_text("{}")
    with pytest.raises(SystemExit, match="used once"):
        ev.guard_test_run(tmp_path, resume=True)
