# Backends

A backend is the thing that actually answers. `SystemOne` owns one and exposes
it as `agent.backend`; which one you get is the config you pass, or
`SYSTEM_ONE_BACKEND` when you pass none.

| Backend | Extra | What it does |
| --- | --- | --- |
| `http` (default) | `system-one[http]` | Calls a hosted provider over HTTP. |
| `onnx` | `system-one[onnx]` | Runs the model in-process. No network, no API key. |
| `stub` | none | Answers at random. For wiring tests and demos only. |

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

One class per vendor, each an `HTTPConfig` with its endpoint filled in:

```python
SystemOne(TypesafeConfig())  # https://api.typesafe.ai/v1/systemone
SystemOne(OpenRouterConfig())  # https://openrouter.ai/api/alpha/decisions
SystemOne(HTTPConfig(base_url="https://my-host", model="mine"))
```

The presets require `SYSTEM_ONE_API_KEY` (or `api_key=`); a plain `HTTPConfig`
sends no `Authorization` header without one. Every field also reads its
`SYSTEM_ONE_*` variable, so `SystemOne()` alone needs `SYSTEM_ONE_BASE_URL` and
`SYSTEM_ONE_MODEL`.

Retries are built in: `408`, `429` and `5xx` are retried up to
`SYSTEM_ONE_MAX_RETRIES` times with exponential backoff, honouring the
`Retry-After` header when the server sends one. `401` and `403` are never
retried — see [Errors](errors.md).

Backend-specific calls stay reachable through `agent.backend`.

## ONNX

```python
SystemOne(ONNXConfig(onnx_dir="onnx", model="laya"))  # both are the defaults
```

or `SYSTEM_ONE_BACKEND=onnx` with `SYSTEM_ONE_ONNX_DIR` / `SYSTEM_ONE_MODEL`.

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

## Stub

```python
SystemOne(StubConfig(seed=1))  # `seed` is optional; set it for reproducible output
```

or `SYSTEM_ONE_BACKEND=stub`. Every question gets a uniformly random answer of the
right shape — a `noul` probability, a choice drawn from its labels with a
distribution, an expected score with its legend — and zero usage. Nothing about
the state is read, so never use it for decisions.

## Injecting your own

Pass any object satisfying the protocol as `using=`. Tests use this to avoid the
network entirely:

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

An injected backend owns its model: `ask` falls back to `backend.model`, not
`SYSTEM_ONE_MODEL`.
