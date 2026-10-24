"""Model A: zero-shot LLM baseline. One API call per headline, one label per entity.

Answers are validated with pydantic; a headline whose answer stays invalid after the
retries gets the label ``invalid`` for every entity, which always counts as wrong.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from entsent.data import LABELS, REPO_ROOT, Example, Label
from entsent.openrouter import Completion, OpenRouterClient

MODEL = "anthropic/claude-haiku-5.5"
# OpenRouter list price for MODEL, USD per token (checked 2026-10-24). Cross-check only:
# the reported cost uses usage.cost from each response.
PRICE_INPUT = 0.10e-6
PRICE_OUTPUT = 0.50e-6
PROMPT_DIR = REPO_ROOT / "prompts"
CACHE_DIR = REPO_ROOT / "results" / "cache" / "llm"
INVALID: Literal["invalid"] = "invalid"
INVALID_RETRIES = 2
MAX_TOKENS = 400

Prediction = Label | Literal["invalid"]


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    label: Label


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[_Item]


def load_prompt(version: str) -> tuple[str, str]:
    """Return (prompt text, sha256 of the text)."""
    text = (PROMPT_DIR / f"{version}.txt").read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()


def user_message(example: Example) -> str:
    lines = [f"Headline: {example.text}", "", "Entities:"]
    lines += [f'{i}. "{ent.name}"' for i, ent in enumerate(example.entities, start=1)]
    return "\n".join(lines)


def response_schema(n_entities: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": list(range(1, n_entities + 1))},
                        "label": {"type": "string", "enum": list(LABELS)},
                    },
                    "required": ["id", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def build_payload(system_prompt: str, example: Example) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message(example)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_sentiment",
                "strict": True,
                "schema": response_schema(len(example.entities)),
            },
        },
        "reasoning": {"enabled": False},
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
        "provider": {"require_parameters": True},
    }


def parse_answer(content: str, n_entities: int) -> list[Label]:
    """Validate a model answer. Raises ValueError unless every entity has exactly one label."""
    try:
        answer = _Answer.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"invalid JSON answer: {exc.error_count()} errors") from exc
    ids = [item.id for item in answer.labels]
    if sorted(ids) != list(range(1, n_entities + 1)):
        raise ValueError(f"expected ids 1..{n_entities} once each, got {ids}")
    by_id = {item.id: item.label for item in answer.labels}
    return [by_id[i] for i in range(1, n_entities + 1)]


@dataclass
class Attempt:
    content: str
    model: str
    generation_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    latency_s: float
    http_attempts: int
    error: str | None = None

    @classmethod
    def from_completion(cls, c: Completion, error: str | None) -> Attempt:
        return cls(
            content=c.content,
            model=c.model,
            generation_id=c.generation_id,
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
            reasoning_tokens=c.reasoning_tokens,
            cost_usd=c.cost_usd,
            latency_s=c.latency_s,
            http_attempts=c.attempts,
            error=error,
        )

    @property
    def listed_cost_usd(self) -> float:
        return self.input_tokens * PRICE_INPUT + self.output_tokens * PRICE_OUTPUT


@dataclass
class HeadlineResult:
    id: int
    gold: list[Label]
    pred: list[Prediction]
    valid: bool
    attempts: list[Attempt] = field(default_factory=list)
    cached: bool = False

    @property
    def cost_usd(self) -> float:
        return sum(
            a.cost_usd if a.cost_usd is not None else a.listed_cost_usd for a in self.attempts
        )

    @property
    def latency_s(self) -> float:
        return sum(a.latency_s for a in self.attempts)


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0

    def charge(self, usd: float) -> None:
        self.spent_usd += usd
        if self.spent_usd > self.limit_usd:
            raise BudgetExceededError(
                f"spent ${self.spent_usd:.4f}, over the ${self.limit_usd:.2f} limit"
            )


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def predict_headline(
    client: OpenRouterClient,
    example: Example,
    system_prompt: str,
    budget: Budget,
    cache_dir: Path | None = CACHE_DIR,
) -> HeadlineResult:
    gold = [e.label for e in example.entities]
    n = len(example.entities)
    payload = build_payload(system_prompt, example)
    cache_path = cache_dir / f"{cache_key(payload)}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        stored = json.loads(cache_path.read_text())
        cached = [Attempt(**a) for a in stored["attempts"]]
        return HeadlineResult(example.id, gold, stored["pred"], stored["valid"], cached, True)

    attempts: list[Attempt] = []
    pred: list[Prediction] = [INVALID] * n
    valid = False
    for _ in range(1 + INVALID_RETRIES):
        completion = await client.complete(payload)
        try:
            labels = parse_answer(completion.content, n)
        except ValueError as exc:
            attempts.append(Attempt.from_completion(completion, str(exc)))
        else:
            attempts.append(Attempt.from_completion(completion, None))
            pred, valid = list(labels), True
        budget.charge(attempts[-1].cost_usd or attempts[-1].listed_cost_usd)
        if valid:
            break

    result = HeadlineResult(example.id, gold, pred, valid, attempts)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"pred": pred, "valid": valid, "attempts": [asdict(a) for a in attempts]}
        cache_path.write_text(json.dumps(record))
    return result


async def run(
    examples: Sequence[Example],
    client: OpenRouterClient,
    system_prompt: str,
    budget: Budget,
    *,
    concurrency: int = 8,
    cache_dir: Path | None = CACHE_DIR,
) -> list[HeadlineResult]:
    gate = asyncio.Semaphore(concurrency)

    async def one(example: Example) -> HeadlineResult:
        async with gate:
            return await predict_headline(client, example, system_prompt, budget, cache_dir)

    return list(await asyncio.gather(*(one(ex) for ex in examples)))
