"""M5 benchmark harness, with the tiny model built in code (CI-safe, no downloads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from test_training import DEV, tiny_cfg, tiny_pretrained

from entsent.bench import (
    build_batch,
    count_mismatches,
    is_cheaper,
    latency_summary,
    run_pass,
    throughput_summary,
    verify_checkpoint,
)
from entsent.training import (
    build_instances,
    group_by_headline,
    make_loader_for,
    predict,
    prepare,
)

ARCHITECTURES = ["per_entity", "single_pass"]


def tiny_model(architecture: str) -> tuple[Any, Any, Any]:
    cfg = tiny_cfg(architecture=architecture)
    torch.manual_seed(0)
    tokenizer, encoder = tiny_pretrained(cfg)
    model = prepare(tokenizer, encoder, 0.1, architecture).eval()
    return cfg, tokenizer, model


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_passes_are_counted_per_architecture(architecture: str) -> None:
    cfg, tokenizer, model = tiny_model(architecture)
    result = run_pass(model, cfg, tokenizer, DEV, group_size=1)
    for call, ex in zip(result.calls, DEV, strict=True):
        assert call.n_headlines == 1 and call.n_entities == len(ex.entities)
        # B: one sequence per entity, all of this headline's entities in one call; C: one.
        assert call.n_passes == (len(ex.entities) if architecture == "per_entity" else 1)


def test_b_batch_holds_exactly_the_headlines_entities() -> None:
    cfg, tokenizer, _ = tiny_model("per_entity")
    two_entities = DEV[0]
    batch = build_batch(cfg, [two_entities], tokenizer)
    texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    assert len(texts) == len(two_entities.entities) == 2
    assert "[E] nike [/E]" in texts[0] and "[E] adidas [/E]" in texts[1]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("group_size", [1, 4, 32])
def test_benchmark_predictions_match_the_standard_dev_path(
    architecture: str, group_size: int
) -> None:
    cfg, tokenizer, model = tiny_model(architecture)
    instances = build_instances(cfg, DEV)
    loader = make_loader_for(cfg, instances, tokenizer, shuffle=False)
    _, _, expected = group_by_headline(instances, predict(model, loader, torch.device("cpu")))
    got = run_pass(model, cfg, tokenizer, DEV, group_size)
    assert got.preds == expected
    assert count_mismatches(got.preds, expected) == 0


def test_summaries_are_well_formed() -> None:
    cfg, tokenizer, model = tiny_model("single_pass")
    lat = latency_summary(run_pass(model, cfg, tokenizer, DEV, 1))
    assert 0 < lat["median_ms"] <= lat["p95_ms"]
    assert sum(v["n"] for v in lat["by_entities"].values()) == len(DEV)
    thr = throughput_summary(run_pass(model, cfg, tokenizer, DEV, 32))
    assert thr["headlines_per_s"] > 0 and thr["cpu_s_per_1000_headlines"] > 0
    with pytest.raises(ValueError, match="one headline per call"):
        latency_summary(run_pass(model, cfg, tokenizer, DEV, 4))


def test_mismatches_are_counted_per_entity() -> None:
    assert (
        count_mismatches(
            [["positive", "neutral"], ["negative"]], [["positive", "negative"], ["neutral"]]
        )
        == 2
    )
    with pytest.raises(ValueError):
        count_mismatches([["positive"]], [])


def test_cheaper_rule_needs_every_repetition() -> None:
    assert is_cheaper([1.0, 1.1, 0.9], [1.2, 1.2, 1.2])
    assert not is_cheaper([1.0, 1.3, 0.9], [1.2, 1.2, 1.2])  # one repetition is not cheaper
    assert not is_cheaper([1.2], [1.2])  # equal is not cheaper
    with pytest.raises(ValueError):
        is_cheaper([1.0], [1.0, 1.0])


def test_checkpoint_hash_mismatch_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"weights")
    import hashlib

    good = hashlib.sha256(b"weights").hexdigest()
    assert verify_checkpoint(tmp_path, good) == good
    with pytest.raises(RuntimeError, match="does not match"):
        verify_checkpoint(tmp_path, "0" * 64)
