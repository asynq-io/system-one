"""Backend selection. Vendor imports live inside `create_backend` so the extras stay optional."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from system_one.schemas import SystemOneInput, SystemOneOutput
    from system_one.settings import Settings

EXTRA_HINTS = {
    "http": "The http backend needs httpx2: pip install 'system-one[http]'",
    "onnx": "The onnx backend needs onnxruntime, tokenizers and numpy: "
    "pip install 'system-one[onnx]'",
}


class Backend(Protocol):
    """What `SystemOne` needs from a backend."""

    def ask(self, request: SystemOneInput) -> SystemOneOutput: ...

    def close(self) -> None: ...


class AsyncBackend(Protocol):
    """The same two methods as `Backend`, awaited."""

    async def ask(self, request: SystemOneInput) -> SystemOneOutput: ...

    async def close(self) -> None: ...


def create_backend(
    settings: Settings, *, is_async: bool = False
) -> Backend | AsyncBackend:
    try:
        if settings.backend == "onnx":
            from system_one.backends.onnx import AsyncONNXBackend, ONNXBackend

            return (AsyncONNXBackend if is_async else ONNXBackend)(settings)
        from system_one.backends.http import AsyncHTTPBackend, HTTPBackend

        return (AsyncHTTPBackend if is_async else HTTPBackend)(settings)
    except ImportError as exc:
        raise ImportError(EXTRA_HINTS[settings.backend]) from exc
