# Errors

Everything this SDK raises descends from `SystemOneError`, so one `except`
catches the lot:

```
SystemOneError                 # catch this
├── APIError                   # non-2xx; .status .body .retry_after
│   └── AuthenticationError    # 401/403, never retried
└── APIConnectionError
    └── APITimeoutError
```

```python
from system_one import SystemOne, SystemOneError

try:
    response = agent.ask(state, questions)
except SystemOneError:
    fall_back_to_the_rules_engine()
```

## `APIError`

Raised for any non-2xx response. Inspect `status` rather than catching a
per-status subclass:

```python
except APIError as exc:
    exc.status        # the HTTP status
    exc.body          # the response body, when there was one
    exc.retry_after   # seconds, parsed from the Retry-After header
```

`AuthenticationError` is the one subclass — a `401` or `403`. It is a
configuration problem (missing or wrong `SYSTEM_ONE_API_KEY`), so it is never
retried and fails fast.

## `APIConnectionError`

The request never reached the server. `APITimeoutError` is the subclass for
"did not complete within `SYSTEM_ONE_TIMEOUT`". Both also inherit from the
builtins you would expect — `ConnectionError` and `TimeoutError` respectively —
so existing `except ConnectionError` handlers keep working.

## Retries

The HTTP backend retries automatically, up to `SYSTEM_ONE_MAX_RETRIES`
(default `2`) with exponential backoff:

| Retried | Not retried |
| --- | --- |
| `408`, `429`, `500`, `502`, `503`, `504` | every other non-2xx |
| connection errors and timeouts | `401`, `403` |

`Retry-After` is honoured when the server sends it. The exported
`RETRY_STATUSES` frozenset is the exact set, if you need to branch on it.

Set `SYSTEM_ONE_MAX_RETRIES=0` to disable retries — useful when a caller upstream
already has its own budget.

## Validation errors

A malformed response body raises pydantic's `ValidationError`, not a
`SystemOneError`. That is deliberate: it means the provider sent something that
does not match the contract, which is a different problem from the request
failing.
