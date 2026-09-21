![Tests](https://github.com/asynq-io/system-one/workflows/Tests/badge.svg)
![Build](https://github.com/asynq-io/system-one/workflows/Publish/badge.svg)
![License](https://img.shields.io/github/license/asynq-io/system-one)
![Python](https://img.shields.io/pypi/pyversions/system-one)
![Format](https://img.shields.io/pypi/format/system-one)
![PyPi](https://img.shields.io/pypi/v/system-one)
![Mypy](https://img.shields.io/badge/mypy-checked-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

# system-one

A vendor-neutral SDK for System One models. One contract — `ask(state, questions)
-> answers` — over three question primitives (`noul`, `choice`, `score`).
Switching providers is an environment variable, not a code change.

These are decision models, not chat models: no messages, no streaming, no
temperature.

## Installation

```shell
pip install "system-one[http]"   # hosted providers
pip install "system-one[onnx]"   # local, in-process
```

## Usage

```python
from system_one import SystemOne

with SystemOne() as agent:
    response = agent.ask(
        "Customer is furious about a double charge.",
        {
            "urgent": {"type": "noul", "instructions": "Does this need a human now?"},
            "topic": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": ["billing", "technical", "other"],
            },
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["minor", "normal", "major", "critical"],
            },
        },
    )

print(response.nouls["urgent"].noul, response.nouls["urgent"].confidence)
print(response.choices["topic"].choice, response.choices["topic"].probabilities)
print(response.scores["severity"].score)
```

`AsyncSystemOne` has the same shape and the same method names, awaited:
`await agent.ask(...)`, `await agent.close()`, `async with`.

## Configuration

All settings come from `SYSTEM_ONE_*` environment variables or a `.env` file,
and any of them can be set in code instead: `SystemOne(HTTPConfig(..., timeout=30))`.

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

Vendor choice is a config class, not a string:

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

Backend-specific calls stay reachable through `agent.backend`. Tests can inject
one directly: `SystemOne(using=FakeBackend())`.

## Running locally

The `onnx` backend runs a decision model in-process — no network, no API key,
no torch at runtime. The weights come from
[laya](https://github.com/receptron/laya)
([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)),
in the layout the backend reads:

```shell
pip install "system-one[onnx,hub]"
system-one fetch --out-dir onnx   # --variant fp32|int8|fp16, --name sets the base name
```

That writes `onnx/laya.onnx` (≈1.6 GB fp32), `onnx/laya.json` and
`onnx/tokenizer/` — no torch, no tracing. To use another laya variant or your
own model, write a spec and run
`system-one export examples/export/laya-english.yaml`; that needs
`system-one[export]`. Unset, `SYSTEM_ONE_MODEL` is whatever the backend serves —
`laya` here — so a local run only sets it for a differently named graph. Nothing
leaves the machine:

```shell
SYSTEM_ONE_BACKEND=onnx
SYSTEM_ONE_ONNX_DIR=onnx   # default
SYSTEM_ONE_MODEL=laya      # the file base name in that directory
```

```python
with SystemOne(ONNXConfig(model="laya")) as agent:
    response = agent.ask(
        "The site is down.",
        {"outage": {"type": "noul", "instructions": "Is there an outage?"}},
    )
```

Full walkthrough — fetch vs export, variants, several models in one directory,
verification, troubleshooting: [docs/usage/local-model.md](docs/usage/local-model.md).

## Errors

```
SystemOneError                 # catch this
├── APIError                   # non-2xx; .status .body .retry_after
│   └── AuthenticationError    # 401/403, never retried
└── APIConnectionError
    └── APITimeoutError
```

A malformed response body raises pydantic's `ValidationError`.

## Development

```shell
uv sync --all-extras
uv run pytest
uv run pre-commit run --all-files
```

Tests needing the ONNX artifacts skip themselves when `onnx/laya.onnx` is
absent (`SYSTEM_ONE_ONNX_DIR` and `SYSTEM_ONE_MODEL` override where they look).
`scripts/check_onnx_parity.py` checks the graph against the reference `laya`
implementation and needs the `export` extra plus the real weights.
