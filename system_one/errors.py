"""The five exceptions this SDK raises, all rooted in `SystemOneError`."""

from __future__ import annotations

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class SystemOneError(Exception):
    """Base class for every error raised by this SDK."""


class APIError(SystemOneError):
    """A non-2xx response. Inspect `status` rather than catching a per-status subclass."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """A 401 or 403. A configuration problem, so it is never retried."""


class APIConnectionError(SystemOneError, ConnectionError):
    """The request never reached the server."""


class APITimeoutError(APIConnectionError, TimeoutError):
    """The request did not complete within the configured timeout."""
