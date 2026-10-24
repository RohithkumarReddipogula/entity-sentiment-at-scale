"""Minimal async OpenRouter chat client: retries with backoff, token usage and cost.

The API key is read from the environment or ``.env`` and is never logged, printed or put
into exception messages.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

from entsent.data import REPO_ROOT

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """A request failed for good (non-retryable status, or retries exhausted)."""


def load_api_key(env_path: Path = REPO_ROOT / ".env") -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(env_path).get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set in the environment or .env")
    return key


@dataclass(frozen=True)
class Completion:
    content: str
    model: str  # the model OpenRouter actually served
    generation_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float | None  # as reported by OpenRouter
    latency_s: float
    attempts: int


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        max_attempts: int = 5,
        timeout_s: float = 60.0,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._key = api_key
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Title": "entity-sentiment-at-scale",
            },
            timeout=timeout_s,
            transport=transport,
        )

    def __repr__(self) -> str:
        return "OpenRouterClient(api_key=<hidden>)"

    async def aclose(self) -> None:
        await self._http.aclose()

    def _redact(self, text: str, limit: int = 300) -> str:
        return text.replace(self._key, "<hidden>")[:limit]

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._max_delay_s)
        backoff = min(self._max_delay_s, self._base_delay_s * 2.0 ** (attempt - 1))
        return backoff * random.uniform(0.5, 1.0)

    async def complete(self, payload: dict[str, Any]) -> Completion:
        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            retry_after: float | None = None
            start = time.perf_counter()
            try:
                resp = await self._http.post(API_URL, json=payload)
            except httpx.TransportError as exc:
                last_error = type(exc).__name__
            else:
                latency = time.perf_counter() - start
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" not in data:
                        return _parse_completion(data, latency, attempt)
                    last_error = f"error in 200 response: {self._redact(str(data['error']))}"
                elif resp.status_code in RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                    retry_after = _retry_after(resp)
                else:
                    raise OpenRouterError(f"HTTP {resp.status_code}: {self._redact(resp.text)}")
            if attempt < self._max_attempts:
                await self._sleep(self._delay(attempt, retry_after))
        raise OpenRouterError(f"gave up after {self._max_attempts} attempts: {last_error}")


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _parse_completion(data: dict[str, Any], latency: float, attempts: int) -> Completion:
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    return Completion(
        content=data["choices"][0]["message"].get("content") or "",
        model=data.get("model", ""),
        generation_id=data.get("id", ""),
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        cost_usd=float(cost) if cost is not None else None,
        latency_s=latency,
        attempts=attempts,
    )
