"""Model A client and validation, with HTTP mocked. No network or API key needed."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from entsent.data import Entity, Example
from entsent.llm_baseline import (
    INVALID,
    Budget,
    BudgetExceededError,
    HeadlineResult,
    build_payload,
    parse_answer,
    predict_headline,
    run,
    user_message,
)
from entsent.openrouter import OpenRouterClient, OpenRouterError

FAKE_KEY = "sk-or-v1-FAKE-SECRET-0123456789"
EXAMPLE = Example(
    7,
    "Nike beats forecasts while Adidas cuts its outlook.",
    (
        Entity("Nike", "positive", ((0, 4),)),
        Entity("Adidas", "negative", ((27, 33),)),
    ),
)


def ok_body(content: str, cost: float | None = 0.0001) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": 300, "completion_tokens": 30}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1",
        "model": "anthropic/claude-haiku-5.5",
        "choices": [{"message": {"content": content}}],
        "usage": usage,
    }


GOOD = json.dumps({"labels": [{"id": 1, "label": "positive"}, {"id": 2, "label": "negative"}]})


def make_client(responses: list[httpx.Response | Exception]) -> tuple[OpenRouterClient, list[Any]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def no_sleep(_: float) -> None:
        return None

    client = OpenRouterClient(
        FAKE_KEY, max_attempts=3, transport=httpx.MockTransport(handler), sleep=no_sleep
    )
    return client, calls


def predict(
    client: OpenRouterClient, budget: Budget | None = None, cache: Path | None = None
) -> HeadlineResult:
    async def go() -> HeadlineResult:
        try:
            return await predict_headline(
                client, EXAMPLE, "system", budget or Budget(1.0), cache_dir=cache
            )
        finally:
            await client.aclose()

    return asyncio.run(go())


# --- prompt and validation -------------------------------------------------------------


def test_user_message_numbers_entities() -> None:
    assert user_message(EXAMPLE).splitlines()[-2:] == ['1. "Nike"', '2. "Adidas"']


def test_payload_pins_model_schema_and_reasoning() -> None:
    payload = build_payload("system", EXAMPLE)
    assert payload["model"] == "anthropic/claude-haiku-5.5"
    assert payload["reasoning"] == {"enabled": False}
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["labels"]["items"]["properties"]["id"]["enum"] == [1, 2]


def test_parse_answer_orders_by_id() -> None:
    content = json.dumps(
        {"labels": [{"id": 2, "label": "negative"}, {"id": 1, "label": "neutral"}]}
    )
    assert parse_answer(content, 2) == ["neutral", "negative"]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"labels": [{"id": 1, "label": "positive"}]}),  # missing entity
        json.dumps({"labels": [{"id": 1, "label": "positive"}] * 2}),  # duplicate id
        json.dumps({"labels": [{"id": 1, "label": "bullish"}, {"id": 2, "label": "neutral"}]}),
        json.dumps({"labels": [{"id": 1, "label": "positive"}, {"id": 3, "label": "neutral"}]}),
        json.dumps({"labels": [], "extra": 1}),
    ],
)
def test_parse_answer_rejects(content: str) -> None:
    with pytest.raises(ValueError):
        parse_answer(content, 2)


# --- retries -----------------------------------------------------------------------------


def test_retries_transient_errors_then_succeeds() -> None:
    client, calls = make_client(
        [
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.ConnectError("boom"),
            httpx.Response(200, json=ok_body(GOOD)),
        ]
    )
    result = predict(client)
    assert result.valid and result.pred == ["positive", "negative"]
    assert len(calls) == 3
    assert result.attempts[0].http_attempts == 3


def test_gives_up_after_max_attempts() -> None:
    client, _ = make_client([httpx.Response(503)] * 3)
    with pytest.raises(OpenRouterError, match="gave up after 3"):
        predict(client)


def test_non_retryable_status_fails_fast() -> None:
    client, calls = make_client([httpx.Response(401, text="bad key")])
    with pytest.raises(OpenRouterError, match="HTTP 401"):
        predict(client)
    assert len(calls) == 1


def test_invalid_answer_is_retried_then_marked_invalid() -> None:
    client, calls = make_client([httpx.Response(200, json=ok_body("nope"))] * 3)
    result = predict(client)
    assert not result.valid
    assert result.pred == [INVALID, INVALID]
    assert len(calls) == 3  # one call + two invalid-answer retries
    assert result.cost_usd == pytest.approx(0.0003)


def test_invalid_then_valid_answer() -> None:
    client, _ = make_client(
        [httpx.Response(200, json=ok_body("nope")), httpx.Response(200, json=ok_body(GOOD))]
    )
    result = predict(client)
    assert result.valid and len(result.attempts) == 2
    assert result.attempts[0].error and result.attempts[1].error is None


# --- cost, budget, cache -----------------------------------------------------------------


def test_cost_falls_back_to_listed_price_when_not_reported() -> None:
    client, _ = make_client([httpx.Response(200, json=ok_body(GOOD, cost=None))])
    result = predict(client)
    assert result.cost_usd == pytest.approx(300 * 0.10e-6 + 30 * 0.50e-6)


def test_budget_stops_the_run() -> None:
    client, _ = make_client([httpx.Response(200, json=ok_body(GOOD, cost=0.6))] * 5)

    async def go() -> None:
        try:
            await run([EXAMPLE] * 5, client, "system", Budget(1.0), concurrency=1, cache_dir=None)
        finally:
            await client.aclose()

    with pytest.raises(BudgetExceededError):
        asyncio.run(go())


def test_cache_avoids_second_call(tmp_path: Path) -> None:
    client, calls = make_client([httpx.Response(200, json=ok_body(GOOD))])
    first = predict(client, cache=tmp_path)
    client2, calls2 = make_client([])
    second = predict(client2, cache=tmp_path)
    assert len(calls) == 1 and not calls2
    assert second.cached and second.pred == first.pred
    assert second.cost_usd == first.cost_usd


# --- the key never leaks -------------------------------------------------------------------


def test_key_sent_only_in_auth_header() -> None:
    client, calls = make_client([httpx.Response(200, json=ok_body(GOOD))])
    predict(client)
    assert calls[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in calls[0].content.decode()


def test_key_not_in_errors_logs_or_repr(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    echo = f"invalid key {FAKE_KEY}"
    client, _ = make_client([httpx.Response(401, text=echo)])
    assert FAKE_KEY not in repr(client)
    with pytest.raises(OpenRouterError) as info:
        predict(client)
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in caplog.text
