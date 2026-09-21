# MCP server

`system-one[mcp]` ships a server that exposes an env-configured agent as MCP
tools, so an MCP client can ask System One questions without writing any Python.

```shell
pip install "system-one[mcp]"
```

The agent it builds is a plain `AsyncSystemOne` with no config argument, so it
is configured entirely by `SYSTEM_ONE_*` [environment
variables](configuration.md): `SYSTEM_ONE_BASE_URL` and `SYSTEM_ONE_MODEL` for
the HTTP backend, or `SYSTEM_ONE_BACKEND=onnx` for a fully local server.

## Running it

```shell
system-one-mcp --config questions.yaml
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--config` | `$SYSTEM_ONE_MCP_CONFIG` | YAML file defining the tools |
| `--generic` | off | also expose the untyped `ask` tool |
| `--transport` | `stdio` | `stdio`, `http` or `sse` |

With no `--config`, the server exposes only the generic `ask` tool.

Under the `fastmcp` CLI, which owns the transport itself, point it at the
`server` factory and pass the flags after `--`:

```shell
fastmcp run system_one/mcp.py:server -- --config questions.yaml
```

## Defining tools

Each top-level key in the YAML becomes one tool that asks its whole batch of
questions in a single call. `questions` is required and uses the same shapes as
[Questions](questions.md); `description` is optional and defaults to the
questions' instructions joined together.

```yaml
moderation:
  description: Moderate a user message before publishing.
  questions:
    is_spam:
      type: noul
      instructions: Is this message spam?
    severity:
      type: score
      instructions: How severe is the policy violation?
      criteria: [none, mild, severe]

routing:
  questions:
    department:
      type: choice
      instructions: Which department should handle this ticket?
      criteria: [billing, technical, sales]
```

That file yields two tools, `moderation` and `routing`, each taking a single
`state` argument. This example lives at `examples/questions.yaml` in the repo.

The config is validated at startup, so a typo'd question type fails the server
launch rather than the first call.

!!! tip "Prefer named tools over the generic one"
    A tool with fixed questions gives the client a narrow, well-described
    action. The generic `ask` tool makes the client invent the questions, which
    is more rope than most clients need — reach for it when exploring, and pin
    the questions in YAML once you know what you are asking.

## The generic tool

With `--generic` (or no config at all), the server also exposes:

```
ask(state, questions) -> SystemOneOutput
```

which is the Python API verbatim, with `questions` validated against the same
schemas.

## Client configuration

```json
{
  "mcpServers": {
    "system-one": {
      "command": "system-one-mcp",
      "args": ["--config", "/path/to/questions.yaml"],
      "env": {
        "SYSTEM_ONE_BASE_URL": "https://api.typesafe.ai",
        "SYSTEM_ONE_MODEL": "jev-latest",
        "SYSTEM_ONE_API_KEY": "…"
      }
    }
  }
}
```
