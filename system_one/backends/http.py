"""One HTTP backend for every vendor: same JSON body, the config carries the endpoint."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any

import httpx2

from system_one import __version__
from system_one.errors import (
    RETRY_STATUSES,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    SystemOneError,
)
from system_one.schemas import SystemOneOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from system_one.schemas import SystemOneInput
    from system_one.settings import HTTPConfig

logger = logging.getLogger("system_one")

MAX_BACKOFF = 5.0
INITIAL_BACKOFF = 0.5
MAX_ERROR_BODY = 200
AUTH_STATUSES = frozenset({401, 403})


def retry_after(headers: Mapping[str, str]) -> float | None:
    """Read `retry-after-ms` then `retry-after`. HTTP-date forms fall back to backoff."""
    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = headers.get(header)
        if raw is None:
            continue
        try:
            return float(raw) / divisor
        except ValueError:
            continue
    return None


def backoff(attempt: int, after: float | None) -> float:
    if after is not None:
        return min(after, MAX_BACKOFF)
    delay = min(MAX_BACKOFF, INITIAL_BACKOFF * (1 << attempt))
    return delay * (0.5 + secrets.randbelow(500) / 1000)


def api_error(response: httpx2.Response) -> APIError:
    status = response.status_code
    error = AuthenticationError if status in AUTH_STATUSES else APIError
    return error(
        f"System One API returned {status}",
        status=status,
        body=response.text[:MAX_ERROR_BODY],
        retry_after=retry_after(response.headers),
    )


def is_retryable(error: SystemOneError) -> bool:
    if isinstance(error, AuthenticationError):
        return False
    if isinstance(error, APIError):
        return error.status in RETRY_STATUSES
    return True


def transport_error(request: httpx2.Request, exc: httpx2.HTTPError) -> SystemOneError:
    if isinstance(exc, httpx2.TimeoutException):
        return APITimeoutError(f"Request to {request.url} timed out")
    return APIConnectionError(f"Could not reach {request.url}")


class BaseHTTPBackend:
    """Request building, error mapping and retry pacing, shared by both flavours."""

    def __init__(self, config: HTTPConfig, *, transport: Any = None) -> None:
        self.config = config
        self.model = config.resolved_model
        self.base_url = config.base_url.rstrip("/")
        self.path = config.path
        self._transport = transport
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"system-one/{__version__}",
        }
        if config.api_key is not None:
            self._headers["Authorization"] = (
                f"Bearer {config.api_key.get_secret_value()}"
            )

    def _build(self, method: str, path: str, body: str | None = None) -> httpx2.Request:
        url = self.base_url + path
        logger.debug("%s %s", method, url)
        return httpx2.Request(method, url, headers=self._headers, content=body)

    def _ask_request(self, request: SystemOneInput) -> httpx2.Request:
        return self._build(
            "POST", self.path, request.model_dump_json(exclude_none=True)
        )

    def _delay_or_reraise(self, error: SystemOneError, attempt: int) -> float:
        if attempt >= self.config.max_retries or not is_retryable(error):
            raise error
        after = error.retry_after if isinstance(error, APIError) else None
        return backoff(attempt, after)


class HTTPBackend(BaseHTTPBackend):
    """Talks to any vendor implementing the System One contract over HTTP."""

    def __init__(self, config: HTTPConfig, *, transport: Any = None) -> None:
        super().__init__(config, transport=transport)
        self._client = httpx2.Client(timeout=config.timeout, transport=transport)

    def ask(self, request: SystemOneInput) -> SystemOneOutput:
        response = self._send(self._ask_request(request))
        return SystemOneOutput.model_validate_json(response.content)

    def close(self) -> None:
        self._client.close()

    def _send(self, request: httpx2.Request) -> httpx2.Response:
        attempt = 0
        while True:
            try:
                return self._attempt(request)
            except SystemOneError as exc:
                delay = self._delay_or_reraise(exc, attempt)
                attempt += 1
            time.sleep(delay)

    def _attempt(self, request: httpx2.Request) -> httpx2.Response:
        try:
            response = self._client.send(request)
        except httpx2.HTTPError as exc:
            raise transport_error(request, exc) from exc
        if response.is_success:
            return response
        raise api_error(response)


class AsyncHTTPBackend(BaseHTTPBackend):
    """The async counterpart of `HTTPBackend`, sharing its config and retry policy."""

    def __init__(self, config: HTTPConfig, *, transport: Any = None) -> None:
        super().__init__(config, transport=transport)
        self._client = httpx2.AsyncClient(timeout=config.timeout, transport=transport)

    async def ask(self, request: SystemOneInput) -> SystemOneOutput:
        response = await self._send(self._ask_request(request))
        return SystemOneOutput.model_validate_json(response.content)

    async def close(self) -> None:
        await self._client.aclose()

    async def _send(self, request: httpx2.Request) -> httpx2.Response:
        attempt = 0
        while True:
            try:
                return await self._attempt(request)
            except SystemOneError as exc:
                delay = self._delay_or_reraise(exc, attempt)
                attempt += 1
            await asyncio.sleep(delay)

    async def _attempt(self, request: httpx2.Request) -> httpx2.Response:
        try:
            response = await self._client.send(request)
        except httpx2.HTTPError as exc:
            raise transport_error(request, exc) from exc
        if response.is_success:
            return response
        raise api_error(response)
