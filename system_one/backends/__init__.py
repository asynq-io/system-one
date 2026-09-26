"""Backend selection. Vendor imports stay inside the factories so the extras stay optional."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from system_one.errors import SystemOneError
from system_one.settings import BackendConfig, HTTPConfig, ONNXConfig, StubConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from system_one.schemas import SystemOneInput, SystemOneOutput
    from system_one.settings import Settings

Config = TypeVar("Config", bound=BackendConfig)

EXTRA_HINTS = {
    "http": "The http backend needs httpx2: pip install 'system-one[http]'",
    "onnx": "The onnx backend needs onnxruntime, tokenizers and numpy: "
    "pip install 'system-one[onnx]'",
}


class Backend(Protocol):
    """What `SystemOne` needs from a backend."""

    model: str
    """The model to ask when the call names none."""

    def ask(self, request: SystemOneInput) -> SystemOneOutput: ...


class AsyncBackend(Protocol):
    """The same `ask` as `Backend`, awaited."""

    model: str

    async def ask(self, request: SystemOneInput) -> SystemOneOutput: ...


@contextmanager
def _hint(name: str) -> Generator[None]:
    """Turn a missing vendor dependency — and only that — into its install hint."""
    try:
        yield
    except ImportError as exc:
        raise ImportError(EXTRA_HINTS[name]) from exc


def _resolve(
    settings: Settings,
    config: BackendConfig | None,
    default: type[Config],
    overrides: dict[str, Any],
) -> Config:
    """The config to build with: the given one, or one read from the environment."""
    if config is None:
        return default(**overrides)
    if not isinstance(config, default):
        message = (
            f"{type(config).__name__} does not configure the "
            f"{settings.backend!r} backend."
        )
        raise SystemOneError(message)
    return config.model_copy(update=overrides)


def create_backend(
    settings: Settings, config: BackendConfig | None = None, **overrides: Any
) -> Backend:
    """The sync backend named by `settings.backend`; `overrides` patch its config."""
    if settings.backend == "stub":
        from system_one.backends.stub import StubBackend

        return StubBackend(_resolve(settings, config, StubConfig, overrides))
    if settings.backend == "onnx":
        with _hint("onnx"):
            from system_one.backends.onnx import ONNXBackend
        return ONNXBackend(_resolve(settings, config, ONNXConfig, overrides))
    with _hint("http"):
        from system_one.backends.http import HTTPBackend
    return HTTPBackend(_resolve(settings, config, HTTPConfig, overrides))


def create_async_backend(
    settings: Settings, config: BackendConfig | None = None, **overrides: Any
) -> AsyncBackend:
    """The async backend named by `settings.backend`; `overrides` patch its config."""
    if settings.backend == "stub":
        from system_one.backends.stub import AsyncStubBackend

        return AsyncStubBackend(_resolve(settings, config, StubConfig, overrides))
    if settings.backend == "onnx":
        with _hint("onnx"):
            from system_one.backends.onnx import AsyncONNXBackend
        return AsyncONNXBackend(_resolve(settings, config, ONNXConfig, overrides))
    with _hint("http"):
        from system_one.backends.http import AsyncHTTPBackend
    return AsyncHTTPBackend(_resolve(settings, config, HTTPConfig, overrides))
