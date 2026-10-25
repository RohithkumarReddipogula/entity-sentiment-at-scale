"""Shared training loop for the encoder models (B now; C plugs in at M4).

Plain PyTorch: one loop, one classification head, so B and C differ only in how each
entity's vector is taken from the encoder output.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from entsent.config import TrainConfig
from entsent.data import LABELS, Example
from entsent.metrics import confusion, scores
from entsent.single_pass import (
    HeadlineInstance,
    MentionPoolingClassifier,
    make_single_pass_loader,
    single_pass_instances,
)

MARKER_OPEN = "[E]"
MARKER_CLOSE = "[/E]"


# --- reproducibility and devices -----------------------------------------------------------


def set_seed(seed: int, device: torch.device) -> None:
    """Seed every RNG. CPU and CUDA also use deterministic kernels; MPS has no such mode, so
    MPS runs are seeded but not bitwise reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(device.type != "mps", warn_only=True)


def select_device(preference: str) -> torch.device:
    cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
    if preference == "auto":
        return torch.device("cuda" if cuda else "mps" if mps else "cpu")
    if (preference == "cuda" and not cuda) or (preference == "mps" and not mps):
        raise RuntimeError(f"device {preference!r} requested but not available")
    return torch.device(preference)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# --- model B inputs ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One classification target: an entity in a headline, as model input text."""

    headline_id: int
    entity_index: int
    text: str
    label: int


def mark_entity(text: str, span: tuple[int, int]) -> str:
    """Wrap one mention in markers: "Nike beats" -> "[E] Nike [/E] beats"."""
    start, end = span
    return f"{text[:start]}{MARKER_OPEN} {text[start:end]} {MARKER_CLOSE}{text[end:]}"


def per_entity_instances(examples: Sequence[Example]) -> list[Instance]:
    """Model B: one input per entity, marking its first mention."""
    return [
        Instance(ex.id, i, mark_entity(ex.text, ent.spans[0]), LABELS.index(ent.label))
        for ex in examples
        for i, ent in enumerate(ex.entities)
    ]


AnyInstance = Instance | HeadlineInstance


def build_instances(cfg: TrainConfig, examples: Sequence[Example]) -> list[AnyInstance]:
    if cfg.architecture == "per_entity":
        return list(per_entity_instances(examples))
    return list(single_pass_instances(examples))


def targets(instances: Sequence[AnyInstance]) -> list[tuple[int, int]]:
    """(headline_id, label) per classification target, in the order the model outputs them."""
    out: list[tuple[int, int]] = []
    for inst in instances:
        if isinstance(inst, HeadlineInstance):
            out += [(inst.headline_id, label) for label in inst.labels]
        else:
            out.append((inst.headline_id, inst.label))
    return out


# --- model ----------------------------------------------------------------------------------------


class EntityClassifier(nn.Module):
    """Encoder + dropout + linear head. Model B classifies from the first token (<s>/[CLS])."""

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        first_token = hidden.last_hidden_state[:, 0]
        logits: torch.Tensor = self.head(self.dropout(first_token))
        return logits


DTYPE = torch.float32


def load_pretrained(cfg: TrainConfig) -> tuple[Any, nn.Module]:
    """Tokenizer and encoder at the pinned revision, always in float32.

    transformers 5 otherwise keeps the checkpoint's dtype: deberta-v3-small is stored in
    float16, which made it run in half precision and abort on MPS next to a float32 head.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, revision=cfg.encoder.revision)
    encoder = AutoModel.from_pretrained(
        cfg.encoder.name, revision=cfg.encoder.revision, dtype=DTYPE
    )
    return tokenizer, encoder


def prepare(
    tokenizer: Any, encoder: Any, dropout: float, architecture: str = "per_entity"
) -> EntityClassifier | MentionPoolingClassifier:
    """Add the head (and, for model B, the entity markers). Fails unless all params are float32.

    Model C sees plain headlines, so it gets no marker tokens.
    """
    model: EntityClassifier | MentionPoolingClassifier
    if architecture == "per_entity":
        tokenizer.add_special_tokens({"additional_special_tokens": [MARKER_OPEN, MARKER_CLOSE]})
        encoder.resize_token_embeddings(len(tokenizer))
        model = EntityClassifier(encoder, encoder.config.hidden_size, dropout)
    else:
        model = MentionPoolingClassifier(encoder, encoder.config.hidden_size, dropout)
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {DTYPE}:
        raise TypeError(f"expected all parameters in {DTYPE}, got {sorted(map(str, dtypes))}")
    return model


# --- training loop --------------------------------------------------------------------------------


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    dev_macro_f1: float
    dev_accuracy: float
    train_seconds: float
    eval_seconds: float


@dataclass
class TrainOutcome:
    best_epoch: int
    epochs: list[EpochLog]
    dev_gold: list[list[str]]  # per headline, best epoch
    dev_pred: list[list[str]]
    dev_ids: list[int]
    train_seconds: float
    n_train_instances: int  # model inputs: entities for B, headlines for C
    n_train_targets: int  # entities classified
    n_parameters: int
    tokenizer: Any = field(repr=False)
    encoder_config: Any = field(repr=False)
    best_state: dict[str, torch.Tensor] | None = field(default=None, repr=False)


def make_loader(
    instances: Sequence[Instance],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Instance]:
    return DataLoader(
        cast("Dataset[Instance]", list(instances)),  # a list is a valid map-style dataset
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_per_entity(b, tokenizer, max_length),
        generator=torch.Generator().manual_seed(seed),
    )


def collate_per_entity(
    batch: Sequence[Instance], tokenizer: Any, max_length: int
) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        [b.text for b in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor([b.label for b in batch]),
    }


def make_loader_for(
    cfg: TrainConfig,
    instances: Sequence[AnyInstance],
    tokenizer: Any,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader[Any]:
    """Model B batches entity inputs; model C batches headlines (both use cfg batch_size)."""
    size = batch_size or cfg.optim.batch_size
    if cfg.architecture == "per_entity":
        b_inst = cast("list[Instance]", list(instances))
        return make_loader(b_inst, tokenizer, cfg.max_length, size, shuffle, cfg.seed)
    c_inst = cast("list[HeadlineInstance]", list(instances))
    return make_single_pass_loader(c_inst, tokenizer, cfg.max_length, size, shuffle, cfg.seed)


def model_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if k != "labels"}


def predict(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> list[int]:
    model.eval()
    preds: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(**model_inputs(batch, device))
            preds += logits.argmax(dim=-1).tolist()
    return preds


def group_by_headline(
    instances: Sequence[AnyInstance], preds: Sequence[int]
) -> tuple[list[int], list[list[str]], list[list[str]]]:
    gold: dict[int, list[str]] = {}
    pred: dict[int, list[str]] = {}
    for (headline_id, label), p in zip(targets(instances), preds, strict=True):
        gold.setdefault(headline_id, []).append(LABELS[label])
        pred.setdefault(headline_id, []).append(LABELS[p])
    ids = list(gold)
    return ids, [gold[i] for i in ids], [pred[i] for i in ids]


def train(
    cfg: TrainConfig,
    train_examples: Sequence[Example],
    dev_examples: Sequence[Example],
    *,
    device: torch.device,
    pretrained: Callable[[TrainConfig], tuple[Any, nn.Module]] = load_pretrained,
    log: Callable[[str], None] = print,
) -> TrainOutcome:
    """Train, evaluating dev macro-F1 after each epoch; keep the best epoch."""
    from transformers import get_linear_schedule_with_warmup

    set_seed(cfg.seed, device)
    tokenizer, encoder = pretrained(cfg)
    model = prepare(tokenizer, encoder, cfg.optim.dropout, cfg.architecture).to(device)

    train_inst = build_instances(cfg, train_examples)
    dev_inst = build_instances(cfg, dev_examples)
    o = cfg.optim
    train_loader = make_loader_for(cfg, train_inst, tokenizer, shuffle=True)
    dev_loader = make_loader_for(cfg, dev_inst, tokenizer, shuffle=False)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": o.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=o.lr)
    total_steps = o.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(o.warmup_ratio * total_steps), total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    epochs: list[EpochLog] = []
    best: tuple[float, int, list[int]] = (-1.0, 0, [])
    best_state: dict[str, torch.Tensor] | None = None
    train_seconds = 0.0
    for epoch in range(1, o.epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            logits = model(**model_inputs(batch, device))
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), o.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            n_batches += 1
        synchronize(device)
        epoch_train = time.perf_counter() - start
        train_seconds += epoch_train

        start = time.perf_counter()
        preds = predict(model, dev_loader, device)
        _, gold, pred = group_by_headline(dev_inst, preds)
        s = scores(np.sum([confusion(g, p) for g, p in zip(gold, pred, strict=True)], axis=0))
        epoch_log = EpochLog(
            epoch,
            total_loss / max(n_batches, 1),
            s["macro_f1"],
            s["accuracy"],
            epoch_train,
            time.perf_counter() - start,
        )
        epochs.append(epoch_log)
        log(
            f"epoch {epoch}: loss {epoch_log.train_loss:.4f}  dev macro-F1 "
            f"{epoch_log.dev_macro_f1:.4f}  acc {epoch_log.dev_accuracy:.4f}  "
            f"({epoch_train:.0f}s train)"
        )
        if s["macro_f1"] > best[0]:
            best = (s["macro_f1"], epoch, preds)
            if cfg.save_checkpoint:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ids, gold, pred = group_by_headline(dev_inst, best[2])
    return TrainOutcome(
        best_epoch=best[1],
        epochs=epochs,
        dev_gold=gold,
        dev_pred=pred,
        dev_ids=ids,
        train_seconds=train_seconds,
        n_train_instances=len(train_inst),
        n_train_targets=len(targets(train_inst)),
        n_parameters=sum(p.numel() for p in model.parameters()),
        tokenizer=tokenizer,
        encoder_config=model.encoder.config,
        best_state=best_state,
    )
