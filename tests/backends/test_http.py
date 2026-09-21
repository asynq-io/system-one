import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr, ValidationError

from system_one import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    Settings,
    SystemOneError,
)
from system_one.backends.http import AsyncHTTPBackend, HTTPBackend, retry_after
from system_one.schemas import SystemOneInput

TYPESAFE = {"base_url": "https://api.typesafe.ai", "path": "/v1/systemone"}
OPENROUTER = {"base_url": "https://openrouter.ai", "path": "/api/alpha/decisions"}

QUESTIONS: dict[str, Any] = {
    "urgent": {"type": "noul", "instructions": "Does this need a human now?"},
    "topic": {
        "type": "choice",
        "instructions": "Which queue?",
        "criteria": ["billing", "technical"],
    },
}

ANSWER_BODY = {
    "model": "jev-latest",
    "usage": {"input_tokens": 120, "output_tokens": 0},
    "answers": {
        "urgent": {"type": "noul", "noul": 0.91},
        "topic": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 1.0, "technical": 0.0},
        },
    },
}


def make_request(model: str = "jev-latest") -> SystemOneInput:
    return SystemOneInput.model_validate(
        {"state": "Double charged.", "model": model, "questions": QUESTIONS}
    )


def backend(*, transport: Any, **overrides: Any) -> HTTPBackend:
    settings = Settings(api_key=SecretStr("secret"), **overrides)
    return HTTPBackend(settings, transport=transport)


def async_backend(*, transport: Any, **overrides: Any) -> AsyncHTTPBackend:
    settings = Settings(api_key=SecretStr("secret"), **overrides)
    return AsyncHTTPBackend(settings, transport=transport)


def json_transport(
    body: dict[str, Any], recorder: list[httpx2.Request] | None = None
) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if recorder is not None:
            recorder.append(request)
        return httpx2.Response(200, json=body)

    return httpx2.MockTransport(handler)


def test_ask_sends_the_contract_body_and_parses_the_response() -> None:
    sent: list[httpx2.Request] = []
    response = backend(transport=json_transport(ANSWER_BODY, sent)).ask(make_request())

    assert json.loads(sent[0].content) == {
        "state": "Double charged.",
        "model": "jev-latest",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Does this need a human now?"},
            "topic": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"billing": None, "technical": None},
            },
        },
    }
    assert sent[0].headers["authorization"] == "Bearer secret"
    assert sent[0].headers["user-agent"].startswith("system-one/")
    assert str(sent[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert response.nouls["urgent"].noul == 0.91
    assert response.choices["topic"].choice == "billing"


def test_both_vendor_configs_produce_the_same_response() -> None:
    urls = []
    answers = []
    for config in (TYPESAFE, OPENROUTER):
        sent: list[httpx2.Request] = []
        client = backend(transport=json_transport(ANSWER_BODY, sent), **config)
        answers.append(client.ask(make_request()).model_dump())
        urls.append(str(sent[0].url))
        assert json.loads(sent[0].content) == json.loads(sent[0].content)

    assert urls == [
        "https://api.typesafe.ai/v1/systemone",
        "https://openrouter.ai/api/alpha/decisions",
    ]
    assert answers[0] == answers[1]


def test_openrouter_shaped_answer_gets_confidence_filled() -> None:
    response = backend(transport=json_transport(ANSWER_BODY), **OPENROUTER).ask(
        make_request()
    )

    assert response.choices["topic"].confidence == 1.0


def test_missing_api_key_is_a_system_one_error() -> None:
    with pytest.raises(SystemOneError, match="SYSTEM_ONE_API_KEY"):
        HTTPBackend(Settings())


def test_unauthorized_is_not_retried() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(401, text="nope")

    with pytest.raises(AuthenticationError) as caught:
        backend(transport=httpx2.MockTransport(handler)).ask(make_request())

    assert caught.value.status == 401
    assert caught.value.body == "nope"
    assert attempts == 1


def test_server_error_is_retried_then_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(500)

    with pytest.raises(APIError) as caught:
        backend(transport=httpx2.MockTransport(handler), max_retries=2).ask(
            make_request()
        )

    assert caught.value.status == 500
    assert attempts == 3
    assert len(slept) == 2


def test_retry_after_ms_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    responses = [
        httpx2.Response(429, headers={"retry-after-ms": "250"}),
        httpx2.Response(200, json=ANSWER_BODY),
    ]

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return responses.pop(0)

    backend(transport=httpx2.MockTransport(handler)).ask(make_request())

    assert slept == [0.25]


def test_timeout_becomes_an_api_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("too slow", request=request)

    with pytest.raises(APITimeoutError):
        backend(transport=httpx2.MockTransport(handler)).ask(make_request())


def test_malformed_body_raises_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        backend(transport=json_transport({"model": "jev-latest"})).ask(make_request())


def test_models_lists_the_account_models() -> None:
    body = {
        "models": [
            {
                "name": "jev-latest",
                "description": "General-purpose system one model.",
                "release_date": "2026-09-15",
            }
        ]
    }
    models = backend(transport=json_transport(body)).models()

    assert [model.name for model in models] == ["jev-latest"]


def test_async_ask_matches_the_sync_result() -> None:
    expected = (
        backend(transport=json_transport(ANSWER_BODY)).ask(make_request()).model_dump()
    )

    async def use() -> dict[str, Any]:
        client = async_backend(transport=json_transport(ANSWER_BODY))
        response = await client.ask(make_request())
        await client.close()
        return response.model_dump()

    assert asyncio.run(use()) == expected


def test_async_server_error_is_retried_then_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("asyncio.sleep", record)
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(503, headers={"retry-after": "1"})

    client = async_backend(transport=httpx2.MockTransport(handler), max_retries=1)

    with pytest.raises(APIError) as caught:
        asyncio.run(client.ask(make_request()))

    assert caught.value.status == 503
    assert caught.value.retry_after == 1.0
    assert attempts == 2
    assert slept == [1.0]


def test_async_timeout_becomes_an_api_timeout_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    client = async_backend(transport=httpx2.MockTransport(handler), max_retries=0)

    with pytest.raises(APITimeoutError):
        asyncio.run(client.ask(make_request()))


def test_connection_failures_become_connection_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    transport = httpx2.MockTransport(handler)

    with pytest.raises(APIConnectionError):
        backend(transport=transport, max_retries=0).ask(make_request())

    with pytest.raises(APIConnectionError):
        asyncio.run(
            async_backend(transport=transport, max_retries=0).ask(make_request())
        )


def test_unparsable_retry_after_is_ignored() -> None:
    assert retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert retry_after({}) is None
    assert retry_after({"retry-after-ms": "bad", "retry-after": "2"}) == 2.0


def test_close_releases_the_sync_client() -> None:
    client = backend(transport=json_transport(ANSWER_BODY))
    client.close()

    with pytest.raises(RuntimeError):
        client.ask(make_request())
