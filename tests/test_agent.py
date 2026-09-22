import asyncio
import importlib.util
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from system_one import (
    AsyncSystemOne,
    OpenRouterConfig,
    Settings,
    SystemOne,
    TypesafeConfig,
)
from system_one.schemas import SystemOneInput, SystemOneOutput

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does this need a human?"}}


class FakeBackend:
    model = "fake-latest"

    def __init__(self) -> None:
        self.requests: list[SystemOneInput] = []
        self.closed = False

    def ask(self, request: SystemOneInput) -> SystemOneOutput:
        self.requests.append(request)
        return SystemOneOutput.model_validate(
            {
                "model": request.model,
                "usage": {"input_tokens": 7, "output_tokens": 0},
                "answers": {
                    name: {"type": "noul", "noul": 0.9} for name in request.questions
                },
            }
        )

    def close(self) -> None:
        self.closed = True


class FakeAsyncBackend(FakeBackend):
    async def ask(self, request: SystemOneInput) -> SystemOneOutput:  # type: ignore[override]
        return super().ask(request)

    async def close(self) -> None:  # type: ignore[override]
        self.closed = True


def test_ask_round_trips_through_the_backend() -> None:
    backend = FakeBackend()
    agent = SystemOne(using=backend)
    response = agent.ask("Customer is furious.", QUESTIONS)

    assert response.nouls["urgent"].noul == 0.9
    assert backend.requests[0].state == "Customer is furious."
    assert backend.requests[0].model == "fake-latest"


def test_async_ask_round_trips_through_the_backend() -> None:
    backend = FakeAsyncBackend()
    agent = AsyncSystemOne(using=backend)
    response = asyncio.run(agent.ask("hi", QUESTIONS))

    assert response.nouls["urgent"].noul == 0.9


def test_per_call_model_overrides_settings() -> None:
    backend = FakeBackend()
    SystemOne(using=backend).ask("hi", QUESTIONS, model="other")

    assert backend.requests[0].model == "other"


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_MODEL", "from-env")
    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "onnx")

    settings = SystemOne(using=FakeBackend()).settings

    assert settings.model == "from-env"
    assert settings.backend == "onnx"


def test_config_selects_the_provider_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from system_one.backends.http import HTTPBackend

    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "secret")  # OpenRouterConfig reads it
    backend = SystemOne(OpenRouterConfig()).backend  # type: ignore[call-arg]

    assert isinstance(backend, HTTPBackend)
    assert backend.base_url == "https://openrouter.ai"
    assert backend.path == "/api/alpha/decisions"
    assert backend.model == "typesafe/jev-latest"


def test_the_model_override_beats_the_config_preset() -> None:
    config = OpenRouterConfig(api_key=SecretStr("secret"))

    assert SystemOne(config, model="x").backend.model == "x"


def test_overrides_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_MODEL", "from-env")

    assert SystemOne(using=FakeBackend(), model="explicit").settings.model == "explicit"


def test_backend_is_reachable_for_backend_specific_calls() -> None:
    backend = FakeBackend()

    assert SystemOne(using=backend).backend is backend


def test_context_manager_closes_the_backend() -> None:
    backend = FakeBackend()
    with SystemOne(using=backend) as agent:
        agent.ask("hi", QUESTIONS)

    assert backend.closed


def test_async_context_manager_closes_the_backend() -> None:
    backend = FakeAsyncBackend()

    async def use() -> None:
        async with AsyncSystemOne(using=backend) as agent:
            await agent.ask("hi", QUESTIONS)

    asyncio.run(use())

    assert backend.closed


def test_hosted_backend_needs_an_api_key() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        SystemOne(TypesafeConfig())  # type: ignore[call-arg]


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="the onnx extra is installed",
)
def test_onnx_backend_reports_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "onnx")
    with pytest.raises(ImportError, match=r"system-one\[onnx\]"):
        SystemOne()


def test_backend_choice_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "onnx")

    assert Settings().backend == "onnx"


def test_stub_backend_answers_every_question_type_at_random() -> None:
    from system_one import StubConfig

    questions: dict[str, dict[str, Any]] = {
        "urgent": {"type": "noul", "instructions": "?"},
        "tone": {"type": "choice", "instructions": "?", "criteria": ["calm", "angry"]},
        "heat": {
            "type": "score",
            "instructions": "?",
            "criteria": ["low", "mid", "high"],
        },
    }
    first = SystemOne(StubConfig(seed=1)).ask("hi", questions)
    again = SystemOne(StubConfig(seed=1)).ask("hi", questions)

    assert first == again
    assert first.model == "stub"
    assert 0.0 <= first.nouls["urgent"].noul <= 1.0
    assert first.choices["tone"].choice in {"calm", "angry"}
    assert 0.0 <= first.scores["heat"].score <= 2.0
    assert first.scores["heat"].confidence is not None


def test_stub_backend_is_selectable_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from system_one.backends.stub import AsyncStubBackend, StubBackend

    monkeypatch.setenv("SYSTEM_ONE_BACKEND", "stub")

    assert isinstance(SystemOne().backend, StubBackend)
    assert isinstance(AsyncSystemOne().backend, AsyncStubBackend)
