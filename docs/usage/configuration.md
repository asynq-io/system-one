# Configuration

All settings come from `SYSTEM_ONE_*` environment variables or a `.env` file in
the working directory, and any of them can be overridden per agent:

```python
SystemOne(model="jev-latest", timeout=30)
```

Unknown keys in the environment are ignored, so `SYSTEM_ONE_*` can live
alongside everything else in your `.env`.

| Variable | Default | Notes |
| --- | --- | --- |
| `SYSTEM_ONE_BACKEND` | `http` | `http` or `onnx` |
| `SYSTEM_ONE_MODEL` | — | the backend's own default: `jev-latest` (http), `laya` (onnx) |
| `SYSTEM_ONE_API_KEY` | — | required by the `http` backend |
| `SYSTEM_ONE_BASE_URL` | `https://api.typesafe.ai` | |
| `SYSTEM_ONE_PATH` | `/v1/systemone` | |
| `SYSTEM_ONE_TIMEOUT` | `10.0` | seconds |
| `SYSTEM_ONE_MAX_RETRIES` | `2` | `408`, `429`, `5xx` only |
| `SYSTEM_ONE_ONNX_DIR` | `onnx` | holds `<model>.onnx`, `<model>.json`, `tokenizer/` |

The resolved settings are on the agent:

```python
agent.settings.model
agent.settings.base_url
```

## Three providers, one script

Vendor choice is `base_url` plus `path` — the request body is identical, so
moving between providers is configuration only.

```shell
# typesafe / jev — the defaults
SYSTEM_ONE_API_KEY=…

# OpenRouter — same backend, same body, different endpoint
SYSTEM_ONE_API_KEY=…
SYSTEM_ONE_BASE_URL=https://openrouter.ai
SYSTEM_ONE_PATH=/api/alpha/decisions
SYSTEM_ONE_MODEL=…

# local ONNX — no network, no API key; reads onnx/$SYSTEM_ONE_MODEL.onnx
SYSTEM_ONE_BACKEND=onnx
```

## Picking a backend in code

The first positional argument accepts a backend name, which is shorthand for
the `backend=` override:

```python
SystemOne("onnx")                       # same as SYSTEM_ONE_BACKEND=onnx
SystemOne("http", timeout=5)
```

It also accepts a backend *instance* — see [Backends](backends.md).

## Lifecycle

Agents hold a client (an HTTP connection pool, or a loaded ONNX session), so
close them when you are done. The context manager is the easy way:

=== "Sync"

    ```python
    with SystemOne() as agent:
        ...
    # or, explicitly
    agent = SystemOne()
    try:
        ...
    finally:
        agent.close()
    ```

=== "Async"

    ```python
    async with AsyncSystemOne() as agent:
        ...
    # or, explicitly
    await agent.close()
    ```

A long-lived agent is the intended shape — build one per process, not one per
request.
