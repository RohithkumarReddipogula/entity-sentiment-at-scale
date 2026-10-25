"""Model C: encode each headline once, mean-pool each entity's mention tokens, classify.

Span -> token alignment uses the fast tokenizer's character offsets. A token belongs to a
mention [a, b) if its offsets [s, e) overlap it (s < b and e > a). Special and padding tokens
have empty offsets (s == e) and are never selected. A token that runs past the mention (e.g.
"Sebi" inside "Sebis") is included. A mention with no token, or a headline longer than
max_length, is an error, never silently dropped.

Pooling: mean over each mention's tokens, then mean over the entity's mentions, so repeated
mentions weigh equally whatever their token counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from entsent.data import LABELS, Example

Span = tuple[int, int]


@dataclass(frozen=True)
class HeadlineInstance:
    """One model-C input: a headline and every entity in it, in the headline's order."""

    headline_id: int
    text: str
    labels: tuple[int, ...]
    spans: tuple[tuple[Span, ...], ...]  # per entity: all its mentions


def single_pass_instances(examples: Sequence[Example]) -> list[HeadlineInstance]:
    return [
        HeadlineInstance(
            ex.id,
            ex.text,
            tuple(LABELS.index(e.label) for e in ex.entities),
            tuple(e.spans for e in ex.entities),
        )
        for ex in examples
    ]


def mention_token_indices(offsets: Sequence[Span], span: Span) -> list[int]:
    """Indices of the tokens whose character offsets overlap ``span``."""
    a, b = span
    indices = [i for i, (s, e) in enumerate(offsets) if s < e and s < b and e > a]
    if not indices:
        raise ValueError(f"mention {span} covers no token (offsets {list(offsets)})")
    return indices


def entity_weights(offsets: Sequence[Span], mentions: Sequence[Span]) -> list[float]:
    """Pooling weights over the sequence for one entity; they sum to 1."""
    weights = [0.0] * len(offsets)
    for span in mentions:
        tokens = mention_token_indices(offsets, span)
        for t in tokens:
            weights[t] += 1.0 / (len(tokens) * len(mentions))
    return weights


class MentionPoolingClassifier(nn.Module):
    """Encoder + per-entity mention pooling + the same dropout/linear head as model B."""

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, len(LABELS))

    def pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_weights: torch.Tensor,
        pool_index: torch.Tensor,
    ) -> torch.Tensor:
        """(n_entities, hidden): weighted sum over the tokens of each entity's headline."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        rows = hidden.last_hidden_state[pool_index]  # (n_entities, seq, hidden)
        return torch.einsum("es,esh->eh", pool_weights, rows)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_weights: torch.Tensor,
        pool_index: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(input_ids, attention_mask, pool_weights, pool_index)
        logits: torch.Tensor = self.head(self.dropout(pooled))
        return logits


def collate_single_pass(
    batch: Sequence[HeadlineInstance], tokenizer: Any, max_length: int
) -> dict[str, torch.Tensor]:
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("model C needs a fast tokenizer (character offsets)")
    enc = tokenizer(
        [b.text for b in batch],
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    seq_len = enc["input_ids"].shape[1]
    if seq_len > max_length:
        raise ValueError(f"headline needs {seq_len} tokens, more than max_length {max_length}")
    weights: list[list[float]] = []
    index: list[int] = []
    labels: list[int] = []
    for row, inst in enumerate(batch):
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"][row].tolist()]
        for mentions, label in zip(inst.spans, inst.labels, strict=True):
            weights.append(entity_weights(offsets, mentions))
            index.append(row)
            labels.append(label)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pool_weights": torch.tensor(weights),
        "pool_index": torch.tensor(index),
        "labels": torch.tensor(labels),
    }


def make_single_pass_loader(
    instances: Sequence[HeadlineInstance],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[HeadlineInstance]:
    return DataLoader(
        cast("Dataset[HeadlineInstance]", list(instances)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_single_pass(b, tokenizer, max_length),
        generator=torch.Generator().manual_seed(seed),
    )
