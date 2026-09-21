# System One

A vendor-neutral SDK for System One models. One contract —
`ask(state, questions) -> answers` — over three question primitives
(`noul`, `choice`, `score`). Switching providers is an environment variable,
not a code change.

These are decision models, not chat models: no messages, no streaming, no
temperature. You hand the model a state and a set of typed questions; it hands
back a calibrated answer per question.

## Installation

```shell
pip install "system-one[http]"   # hosted providers
pip install "system-one[onnx]"   # local, in-process
pip install "system-one[mcp]"    # expose an agent over MCP
pip install "system-one[all]"    # all of the above
```

The core install pulls only `pydantic` and `pydantic-settings`; every backend
lives behind an extra.

## Quickstart

=== "Sync"

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

=== "Async"

    ```python
    from system_one import AsyncSystemOne

    async with AsyncSystemOne() as agent:
        response = await agent.ask(
            "Customer is furious about a double charge.",
            {"urgent": {"type": "noul", "instructions": "Does this need a human now?"}},
        )
    ```

[`AsyncSystemOne`][system_one.agent.AsyncSystemOne] has the same shape and the
same method names as [`SystemOne`][system_one.agent.SystemOne], awaited:
`await agent.ask(...)`, `await agent.close()`, `async with`.

## Where to go next

- [Questions](usage/questions.md) — the three primitives and how to write them.
- [Answers](usage/answers.md) — what comes back, and what the numbers mean.
- [Configuration](usage/configuration.md) — `SYSTEM_ONE_*` and per-agent overrides.
- [Backends](usage/backends.md) — hosted HTTP vs local ONNX.
- [Local model](usage/local-model.md) — fetch or export laya as ONNX and run offline.
- [Errors](usage/errors.md) — the exception tree and retry behaviour.
- [MCP server](usage/mcp.md) — expose an agent as MCP tools.
