"""Backend selection. Vendor imports stay inside the factories so the extras stay optional."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    from system_one.schemas import SystemOneInput, SystemOneOutput
    from system_one.settings import Settings

EXTRA_HINTS = {
    "http": "The http backend needs httpx2: pip install 'system-one[http]'",
    "onnx": "The onnx backend needs onnxruntime, tokenizers and numpy: "
    "pip install 'system-one[onnx]'",
}


class Backend(Protocol):
    """What `SystemOne` needs from a backend."""

    model: str
    """The model to ask when neither the call nor `SYSTEM_ONE_MODEL` names one."""

    def ask(self, request: SystemOneInput) -> SystemOneOutput: ...

    def close(self) -> None: ...


class AsyncBackend(Protocol):
    """The same two methods as `Backend`, awaited."""

    model: str

    async def ask(self, request: SystemOneInput) -> SystemOneOutput: ...

    async def close(self) -> None: ...


@contextmanager
def _hint(name: str) -> Iterator[None]:
    """Turn a missing vendor dependency — and only that — into its install hint."""
    try:
        yield
    except ImportError as exc:
        raise ImportError(EXTRA_HINTS[name]) from exc


def create_backend(settings: Settings) -> Backend:
    """The sync backend named by `settings.backend`."""
    if settings.backend == "onnx":
        with _hint("onnx"):
            from system_one.backends.onnx import ONNXBackend
        return ONNXBackend(settings)
    with _hint("http"):
        from system_one.backends.http import HTTPBackend
    return HTTPBackend(settings)


def create_async_backend(settings: Settings) -> AsyncBackend:
    """The async backend named by `settings.backend`."""
    if settings.backend == "onnx":
        with _hint("onnx"):
            from system_one.backends.onnx import AsyncONNXBackend
        return AsyncONNXBackend(settings)
    with _hint("http"):
        from system_one.backends.http import AsyncHTTPBackend
    return AsyncHTTPBackend(settings)
