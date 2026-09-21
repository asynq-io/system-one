# Backends

A backend is the thing that actually answers. `SystemOne` owns one and exposes
it as `agent.backend`; which one you get is `SYSTEM_ONE_BACKEND`.

| Backend | Extra | What it does |
| --- | --- | --- |
| `http` (default) | `system-one[http]` | Calls a hosted provider over HTTP. |
| `onnx` | `system-one[onnx]` | Runs the model in-process. No network, no API key. |

Vendor imports live inside the backend factory, so the extras stay genuinely
optional: installing the core package pulls neither `httpx2` nor `onnxruntime`.
Selecting a backend whose extra is missing raises `ImportError` with the pip
command that fixes it.

## The protocol

Both backends satisfy the same two-method protocol —
[`Backend`][system_one.backends.Backend], or
[`AsyncBackend`][system_one.backends.AsyncBackend] for the awaited version:

```python
def ask(self, request: SystemOneInput) -> SystemOneOutput: ...
def close(self) -> None: ...
```

That is the whole contract. Anything implementing it can be handed to an agent.

## HTTP

```shell
SYSTEM_ONE_API_KEY=…
SYSTEM_ONE_BASE_URL=https://api.typesafe.ai   # default
SYSTEM_ONE_PATH=/v1/systemone                 # default
```

Retries are built in: `408`, `429` and `5xx` are retried up to
`SYSTEM_ONE_MAX_RETRIES` times with exponential backoff, honouring the
`Retry-After` header when the server sends one. `401` and `403` are never
retried — see [Errors](errors.md).

Backend-specific calls stay reachable through `agent.backend`:

```python
for model in agent.backend.models():
    print(model.name, model.release_date, model.description)
```

!!! warning
    `models()` exists on the HTTP backend only. Reach for `agent.backend` when
    you have decided which backend you are on.

## ONNX

```shell
SYSTEM_ONE_BACKEND=onnx
SYSTEM_ONE_ONNX_DIR=onnx     # default
```

The directory must hold three things for the configured model:

```
onnx/
├── laya.onnx            # the exported graph
├── laya.json            # model config: prompt layout, temperature buckets
└── tokenizer/
    └── tokenizer.json
```

Sessions are cached per `(directory, model)` pair, so a second agent — or a
second model on the same agent — does not reload gigabytes of weights. A
missing graph raises `SystemOneError` naming the path it looked for.

Those artifacts are produced once from a
[laya](https://huggingface.co/convaiinnovations/laya) checkpoint — see
[Running a local model](local-model.md).

The async ONNX backend runs the same synchronous session off the event loop, so
`await agent.ask(...)` does not block it.

## Injecting your own

Pass any object satisfying the protocol as the first argument. Tests use this to
avoid the network entirely:

```python
class FakeBackend:
    def ask(self, request):
        return SystemOneOutput(
            model=request.model,
            usage=Usage(input_tokens=0, output_tokens=0),
            answers={"urgent": NoulAnswer(noul=0.9)},
        )

    def close(self):
        pass


agent = SystemOne(using=FakeBackend())
```

Settings are still resolved when you inject a backend, so `agent.settings.model`
still supplies the default model for `ask`.
