# Configuration

`Settings` — which backend to build — comes from `SYSTEM_ONE_*` environment
variables or a `.env` file in the working directory. Everything the backend
itself needs lives in a config object, read from the same environment when you
do not pass one:

```python
SystemOne(model="jev-latest")  # the model override
SystemOne(HTTPConfig(base_url="https://my-host", model="mine", timeout=30))
```

Unknown keys in the environment are ignored, so `SYSTEM_ONE_*` can live
alongside everything else in your `.env`.

| Variable | Default | Notes |
| --- | --- | --- |
| `SYSTEM_ONE_BACKEND` | `http` | `http` or `onnx` |
| `SYSTEM_ONE_MODEL` | — | overrides whatever the config would use |
| `SYSTEM_ONE_BASE_URL` | — | required by plain `http`; the presets set their own |
| `SYSTEM_ONE_PATH` | `/v1/systemone` | |
| `SYSTEM_ONE_API_KEY` | — | required by `TypesafeConfig` / `OpenRouterConfig` |
| `SYSTEM_ONE_TIMEOUT` | `10.0` | seconds |
| `SYSTEM_ONE_MAX_RETRIES` | `2` | `408`, `429`, `5xx` only |
| `SYSTEM_ONE_ONNX_DIR` | `onnx` | holds `<model>.onnx`, `<model>.json`, `tokenizer/` |

The resolved settings are on the agent:

```python
agent.settings.backend
agent.backend.base_url
```

## Three providers, one script

Vendor choice is a config class — the request body is identical, so moving
between providers is one constructor argument.

```python
from system_one import (
    HTTPConfig,
    ONNXConfig,
    OpenRouterConfig,
    SystemOne,
    TypesafeConfig,
)

SystemOne(TypesafeConfig())  # hosted jev, $SYSTEM_ONE_API_KEY
SystemOne(OpenRouterConfig())  # same body, different endpoint
SystemOne(HTTPConfig(base_url="https://my-host", model="mine"))
SystemOne(ONNXConfig())  # local graph, no network, no key
```

`SYSTEM_ONE_BACKEND=onnx` picks the local backend without naming a config, and
`SystemOne(model="x")` overrides whichever model the config would have used.

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
