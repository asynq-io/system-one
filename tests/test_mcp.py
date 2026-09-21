import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from system_one import AsyncSystemOne
from system_one.schemas import SystemOneInput, SystemOneOutput

pytest.importorskip("fastmcp")

from fastmcp import Client, FastMCP

from system_one.mcp import create_server, server

CONFIG = """
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
"""


class FakeAsyncBackend:
    model = "fake-latest"

    def __init__(self) -> None:
        self.requests: list[SystemOneInput] = []
        self.closed = False

    async def ask(self, request: SystemOneInput) -> SystemOneOutput:
        self.requests.append(request)
        return SystemOneOutput.model_validate(
            {
                "model": request.model,
                "usage": {"input_tokens": 7, "output_tokens": 0},
                "answers": {
                    name: {"type": "noul", "noul": 0.9} for name in request.questions
                },
            }
        )

    async def close(self) -> None:
        self.closed = True


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(CONFIG)
    return path


def tool_names(server: FastMCP) -> set[str]:
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_yaml_groups_become_tools(tmp_path: Path) -> None:
    server = create_server(
        write_config(tmp_path), agent=AsyncSystemOne(using=FakeAsyncBackend())
    )
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == {"moderation", "routing"}
    assert (
        tools["moderation"].description == "Moderate a user message before publishing."
    )
    assert tools["routing"].description == (
        "Which department should handle this ticket?"
    )


@pytest.mark.timeout(15)  # the first in-memory client pulls in jsonschema
def test_group_tool_sends_its_configured_questions(tmp_path: Path) -> None:
    backend = FakeAsyncBackend()
    server = create_server(write_config(tmp_path), agent=AsyncSystemOne(using=backend))

    async def call() -> Any:
        async with Client(server) as client:
            return await client.call_tool("moderation", {"state": "buy now!!!"})

    result = asyncio.run(call())

    request = backend.requests[0]
    assert request.state == "buy now!!!"
    assert set(request.questions) == {"is_spam", "severity"}
    assert request.questions["severity"].criteria == ["none", "mild", "severe"]
    assert result.data["answers"]["is_spam"]["noul"] == 0.9


def test_generic_tool_only_without_config_or_with_flag(tmp_path: Path) -> None:
    config = write_config(tmp_path)

    assert tool_names(
        create_server(agent=AsyncSystemOne(using=FakeAsyncBackend()))
    ) == {"ask"}
    assert "ask" not in tool_names(
        create_server(config, agent=AsyncSystemOne(using=FakeAsyncBackend()))
    )
    assert tool_names(
        create_server(
            config, generic=True, agent=AsyncSystemOne(using=FakeAsyncBackend())
        )
    ) == {"moderation", "routing", "ask"}


def test_factory_reads_its_config_flag_from_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYSTEM_ONE_BASE_URL", "https://example.test")
    monkeypatch.setenv("SYSTEM_ONE_MODEL", "fake-latest")
    monkeypatch.setattr(
        sys, "argv", ["fastmcp", "run", "--config", str(write_config(tmp_path))]
    )

    assert tool_names(server()) == {"moderation", "routing"}
