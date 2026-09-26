"""Expose an env-configured System One agent over MCP."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from fastmcp import FastMCP
from fastmcp.tools import Tool
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from system_one.agent import AsyncSystemOne
from system_one.schemas import Question, State, SystemOneOutput


class ToolSpec(BaseModel):
    """One YAML group: a batch of questions exposed as a single MCP tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: Mapping[str, Question] = Field(min_length=1)
    description: str | None = None


CONFIG_ADAPTER = TypeAdapter(dict[str, ToolSpec])


def load_config(path: Path) -> dict[str, ToolSpec]:
    """Read and validate the YAML tool config; invalid input raises at startup."""
    return CONFIG_ADAPTER.validate_python(yaml.safe_load(path.read_text()))


def _group_tool(agent: AsyncSystemOne, name: str, spec: ToolSpec) -> Tool:
    async def run(state: State) -> SystemOneOutput:
        return await agent.ask(state, spec.questions)

    description = spec.description or "; ".join(
        question.instructions for question in spec.questions.values()
    )
    return Tool.from_function(run, name=name, description=description)


def _generic_tool(agent: AsyncSystemOne) -> Tool:
    async def ask(state: State, questions: dict[str, Question]) -> SystemOneOutput:
        """Ask typed questions (noul / choice / score) about a JSON state."""
        return await agent.ask(state, questions)

    return Tool.from_function(ask)


def create_server(
    config: Path | None = None,
    *,
    generic: bool = False,
    agent: AsyncSystemOne | None = None,
) -> FastMCP:
    """Build the MCP server; `agent` is the injection seam used by the tests."""
    agent = agent if agent is not None else AsyncSystemOne()
    tools = [
        _group_tool(agent, name, spec)
        for name, spec in (load_config(config) if config else {}).items()
    ]
    if generic or config is None:
        tools.append(_generic_tool(agent))
    return FastMCP("system-one", tools=tools)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="system-one-mcp")
    parser.add_argument(
        "--config", type=Path, default=os.getenv("SYSTEM_ONE_MCP_CONFIG")
    )
    parser.add_argument("--generic", action="store_true")
    parser.add_argument(
        "--transport", default="stdio", choices=("stdio", "http", "sse")
    )
    return parser


def server() -> FastMCP:
    """Factory for `fastmcp run system_one/mcp.py:server -- --config questions.yaml`.

    The CLI owns the transport and puts everything after `--` on `sys.argv`, so this
    reads only its own flags and ignores the CLI's.
    """
    args, _ = _parser().parse_known_args()
    return create_server(args.config, generic=args.generic)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the server directly, without the fastmcp CLI."""
    args = _parser().parse_args(argv)
    create_server(args.config, generic=args.generic).run(transport=args.transport)


if __name__ == "__main__":
    main()
