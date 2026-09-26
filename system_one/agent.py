"""The one public entry point: `SystemOne.ask(state, questions)`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from system_one.backends import create_async_backend, create_backend
from system_one.schemas import SystemOneInput
from system_one.settings import ONNXConfig, Settings, StubConfig

if TYPE_CHECKING:
    from system_one.backends import AsyncBackend, Backend
    from system_one.schemas import QuestionInput, State, SystemOneOutput
    from system_one.settings import BackendConfig


def _backend_name(config: BackendConfig) -> Literal["http", "onnx", "stub"]:
    if isinstance(config, ONNXConfig):
        return "onnx"
    if isinstance(config, StubConfig):
        return "stub"
    return "http"


class BaseSystemOne:
    """Settings resolution and input building, shared by the sync and async agents.

    `config` is the backend's own configuration — `TypesafeConfig()`,
    `OpenRouterConfig()`, `HTTPConfig(base_url=..., model=...)`, `ONNXConfig()` or
    `StubConfig()` —
    and picks the backend to build; omitted, `SYSTEM_ONE_BACKEND` decides and the
    config is read from the environment. `using` hands over an already-built backend
    instead, which is how tests and custom providers plug in.
    """

    backend: Any  # narrowed to `Backend` / `AsyncBackend` by the two subclasses

    def __init__(self, config: BackendConfig | None = None, **overrides: Any) -> None:
        backend = overrides.pop("backend", None)
        if backend is None and config is not None:
            backend = _backend_name(config)
        self.settings = Settings() if backend is None else Settings(backend=backend)
        self.config = config
        self.overrides = overrides

    def _input(
        self, state: State, questions: QuestionInput, model: str | None
    ) -> SystemOneInput:
        return SystemOneInput.model_validate(
            {
                "state": state,
                "model": model or self.backend.model,
                "questions": questions,
            }
        )


class SystemOne(BaseSystemOne):
    """Ask typed questions about a state and get calibrated answers back."""

    def __init__(
        self,
        config: BackendConfig | None = None,
        *,
        using: Backend | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        self.backend: Backend = (
            using
            if using is not None
            else create_backend(self.settings, config, **self.overrides)
        )

    def ask(
        self, state: State, questions: QuestionInput, *, model: str | None = None
    ) -> SystemOneOutput:
        return self.backend.ask(self._input(state, questions, model))


class AsyncSystemOne(BaseSystemOne):
    """The async counterpart of `SystemOne`: the same `ask`, awaited."""

    def __init__(
        self,
        config: BackendConfig | None = None,
        *,
        using: AsyncBackend | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)

        self.backend: AsyncBackend = (
            using
            if using is not None
            else create_async_backend(self.settings, config, **self.overrides)
        )

    async def ask(
        self, state: State, questions: QuestionInput, *, model: str | None = None
    ) -> SystemOneOutput:
        return await self.backend.ask(self._input(state, questions, model))
