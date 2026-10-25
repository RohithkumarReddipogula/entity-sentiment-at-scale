"""One training run -> results/<run_id>.json, dev predictions, and optionally a checkpoint."""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from entsent.config import TrainConfig
from entsent.data import REPO_ROOT, Example, sha256_file
from entsent.metrics import BOOTSTRAP_SEED, N_BOOTSTRAP, evaluate
from entsent.results import RESULTS_DIR, git_info, new_run_id, write_result
from entsent.single_pass import MentionPoolingClassifier
from entsent.training import DTYPE, EntityClassifier, load_pretrained, train

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
MODEL_IDS = {"per_entity": "B", "single_pass": "C"}
MODEL_NAMES = {"per_entity": "Per-entity encoder", "single_pass": "Single-pass encoder"}


def run_training(
    cfg: TrainConfig,
    train_examples: Sequence[Example],
    dev_examples: Sequence[Example],
    *,
    device: torch.device,
    purpose: str,
    config_path: str,
    results_dir: Path = RESULTS_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    pretrained: Callable[[TrainConfig], tuple[Any, nn.Module]] = load_pretrained,
    n_bootstrap: int = N_BOOTSTRAP,
    log: Callable[[str], None] = print,
) -> Path:
    model_id = MODEL_IDS[cfg.architecture]
    prefix = model_id.lower()
    name = cfg.name if cfg.name.startswith(f"{prefix}_") else f"{prefix}_{cfg.name}"
    run_id, timestamp = new_run_id(f"{name}_s{cfg.seed}")
    git = git_info()
    outcome = train(
        cfg, train_examples, dev_examples, device=device, pretrained=pretrained, log=log
    )
    metrics = evaluate(outcome.dev_gold, outcome.dev_pred, n_resamples=n_bootstrap)

    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_id": MODEL_IDS[cfg.architecture],
        "model_name": MODEL_NAMES[cfg.architecture],
        "purpose": purpose,
        "split": "dev",
        "n_headlines": len(outcome.dev_gold),
        "n_entities": sum(len(g) for g in outcome.dev_gold),
        "git": git,
        "config_path": config_path,
        "config": cfg.model_dump(),
        "seed": {"train": cfg.seed, "bootstrap": BOOTSTRAP_SEED, "n_bootstrap": n_bootstrap},
        "environment": {
            "device": device.type,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "deterministic": device.type != "mps",
            "dtype": str(DTYPE),
        },
        "training": {
            "best_epoch": outcome.best_epoch,
            "epochs": [asdict(e) for e in outcome.epochs],
            "train_seconds": outcome.train_seconds,
            "n_train_instances": outcome.n_train_instances,
            "n_train_targets": outcome.n_train_targets,
            "n_parameters": outcome.n_parameters,
            "selection": "best dev macro-F1 over epochs",
        },
        "metrics": metrics,
    }

    if cfg.save_checkpoint and outcome.best_state is not None:
        ckpt = checkpoint_dir / run_id
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(outcome.best_state, ckpt / "model.pt")
        outcome.tokenizer.save_pretrained(ckpt / "tokenizer")
        outcome.encoder_config.save_pretrained(ckpt / "encoder")
        (ckpt / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))
        record["checkpoint"] = {
            "path": str(ckpt.relative_to(checkpoint_dir.parent)),
            "model_sha256": sha256_file(ckpt / "model.pt"),
            "epoch": outcome.best_epoch,
        }

    path = write_result(record, results_dir)
    predictions = results_dir / "predictions" / f"{run_id}.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("w") as f:
        for hid, gold, pred in zip(
            outcome.dev_ids, outcome.dev_gold, outcome.dev_pred, strict=True
        ):
            for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
                f.write(json.dumps({"id": hid, "entity": i, "gold": g, "pred": p}) + "\n")
    return path


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[EntityClassifier | MentionPoolingClassifier, Any, TrainConfig]:
    """Rebuild a saved model (encoder config + weights + tokenizer); nothing is downloaded."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = TrainConfig.model_validate_json((path / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    encoder_config = AutoConfig.from_pretrained(path / "encoder")
    encoder = AutoModel.from_config(encoder_config).to(DTYPE)
    model_class = EntityClassifier if cfg.architecture == "per_entity" else MentionPoolingClassifier
    model = model_class(encoder, encoder_config.hidden_size, cfg.optim.dropout)
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device or torch.device("cpu")).eval()
    return model, tokenizer, cfg
