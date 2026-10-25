"""M5: CPU speed and cost of models B and C, end to end per call.

One call = build the model inputs for a group of headlines (markers for B; offsets and pooling
weights for C) + forward pass + argmax. Model B always batches each headline's own entities
into the same call, which is how it would be served. Forward passes are counted as encoder
input sequences: one per entity for B, one per headline for C.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from entsent.config import TrainConfig
from entsent.data import LABELS, Example, sha256_file
from entsent.single_pass import collate_single_pass, single_pass_instances
from entsent.training import collate_per_entity, model_inputs, per_entity_instances

THROUGHPUT_BATCH = 32
WARMUP_HEADLINES = 50


def build_batch(
    cfg: TrainConfig, examples: Sequence[Example], tokenizer: Any
) -> dict[str, torch.Tensor]:
    """Model inputs for a group of headlines, in headline then entity order."""
    if cfg.architecture == "per_entity":
        return collate_per_entity(per_entity_instances(examples), tokenizer, cfg.max_length)
    return collate_single_pass(single_pass_instances(examples), tokenizer, cfg.max_length)


@dataclass
class CallTiming:
    n_headlines: int
    n_entities: int
    n_passes: int
    wall_s: float
    cpu_s: float


@dataclass
class PassResult:
    """Timings and predictions for one pass over a list of headlines."""

    calls: list[CallTiming] = field(default_factory=list)
    preds: list[list[str]] = field(default_factory=list)  # per headline

    @property
    def wall_s(self) -> float:
        return sum(c.wall_s for c in self.calls)

    @property
    def cpu_s(self) -> float:
        return sum(c.cpu_s for c in self.calls)

    @property
    def n_headlines(self) -> int:
        return sum(c.n_headlines for c in self.calls)

    @property
    def n_passes(self) -> int:
        return sum(c.n_passes for c in self.calls)


def run_pass(
    model: nn.Module,
    cfg: TrainConfig,
    tokenizer: Any,
    examples: Sequence[Example],
    group_size: int,
) -> PassResult:
    """Predict all headlines in groups of ``group_size``, timing each call end to end."""
    model.eval()
    out = PassResult()
    with torch.inference_mode():
        for start in range(0, len(examples), group_size):
            group = examples[start : start + group_size]
            wall0, cpu0 = time.perf_counter(), time.process_time()
            batch = build_batch(cfg, group, tokenizer)
            pred = model(**model_inputs(batch, torch.device("cpu"))).argmax(dim=-1).tolist()
            wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
            n_entities = sum(len(ex.entities) for ex in group)
            out.calls.append(
                CallTiming(len(group), n_entities, int(batch["input_ids"].shape[0]), wall, cpu)
            )
            it = iter(pred)
            out.preds += [[LABELS[next(it)] for _ in ex.entities] for ex in group]
    return out


def latency_summary(result: PassResult) -> dict[str, Any]:
    """Batch-1 latency per headline, overall and by number of entities."""
    if any(c.n_headlines != 1 for c in result.calls):
        raise ValueError("latency needs one headline per call")
    ms = [c.wall_s * 1000 for c in result.calls]
    by_k: dict[str, list[float]] = {"1": [], "2": [], "3+": []}
    for c, v in zip(result.calls, ms, strict=True):
        by_k["1" if c.n_entities == 1 else "2" if c.n_entities == 2 else "3+"].append(v)
    return {
        "median_ms": statistics.median(ms),
        "p95_ms": float(np.percentile(ms, 95)),
        "by_entities": {
            k: {"n": len(v), "median_ms": statistics.median(v) if v else None}
            for k, v in by_k.items()
        },
    }


def throughput_summary(result: PassResult) -> dict[str, Any]:
    n = result.n_headlines
    entities = sum(c.n_entities for c in result.calls)
    return {
        "headlines_per_s": n / result.wall_s,
        "entities_per_s": entities / result.wall_s,
        "cpu_s_per_1000_headlines": 1000 * result.cpu_s / n,
        "wall_s_per_1000_headlines": 1000 * result.wall_s / n,
    }


def count_mismatches(preds: Sequence[Sequence[str]], recorded: Sequence[Sequence[str]]) -> int:
    """Entity predictions that differ from the recorded ones (same headline order)."""
    if len(preds) != len(recorded):
        raise ValueError(f"{len(preds)} headlines predicted, {len(recorded)} recorded")
    return sum(
        p != r for ps, rs in zip(preds, recorded, strict=True) for p, r in zip(ps, rs, strict=True)
    )


def verify_checkpoint(ckpt: Path, expected_sha256: str) -> str:
    """Refuse to benchmark weights that differ from the ones the run record describes."""
    digest = sha256_file(ckpt / "model.pt")
    if digest != expected_sha256:
        raise RuntimeError(f"{ckpt}: weights sha256 {digest} does not match its record")
    return digest


def is_cheaper(c_cpu: Sequence[float], b_cpu: Sequence[float]) -> bool:
    """LOG.md rule: C is cheaper only if its CPU time per 1,000 headlines is lower than B's in
    every interleaved repetition."""
    if len(c_cpu) != len(b_cpu) or not c_cpu:
        raise ValueError("need the same, non-zero number of repetitions for B and C")
    return all(c < b for c, b in zip(c_cpu, b_cpu, strict=True))
