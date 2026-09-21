"""The one public entry point: `SystemOne.ask(state, questions)`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from system_one.backends import create_backend
from system_one.schemas import SystemOneInput
from system_one.settings import Settings

if TYPE_CHECKING:
    from typing_extensions import Self

    from system_one.backends import AsyncBackend, Backend
    from system_one.schemas import QuestionInput, State, SystemOneOutput


class BaseSystemOne:
    """Settings resolution and input building, shared by the sync and async agents."""

    _is_async = False

    def __init__(
        self, backend: Backend | AsyncBackend | str | None = None, **overrides: Any
    ) -> None:
        if isinstance(backend, str):
            overrides["backend"] = backend
            backend = None
        self.settings = Settings(**overrides)
        self.backend = (
            backend
            if backend is not None
            else create_backend(self.settings, is_async=self._is_async)
        )

    def _input(
        self, state: State, questions: QuestionInput, model: str | None
    ) -> SystemOneInput:
        return SystemOneInput.model_validate(
            {
                "state": state,
                "model": model if model is not None else self.settings.model,
                "questions": questions,
            }
        )


class SystemOne(BaseSystemOne):
    """Ask typed questions about a state and get calibrated answers back."""

    backend: Backend

    def ask(
        self, state: State, questions: QuestionInput, *, model: str | None = None
    ) -> SystemOneOutput:
        return self.backend.ask(self._input(state, questions, model))

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncSystemOne(BaseSystemOne):
    """The async counterpart of `SystemOne`: the same `ask`, awaited."""

    _is_async = True
    backend: AsyncBackend

    async def ask(
        self, state: State, questions: QuestionInput, *, model: str | None = None
    ) -> SystemOneOutput:
        return await self.backend.ask(self._input(state, questions, model))

    async def close(self) -> None:
        await self.backend.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
