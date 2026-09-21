"""What can be exported and what can be fetched — the model catalogue.

Kept free of torch and huggingface_hub so the CLI can read a spec, and report a
bad one, without either extra installed.
"""

from __future__ import annotations

import hashlib
import json
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_FILE = "rl_agent_config.json"
DIGEST_BLOCK = 1 << 20
EXPORTER = f"system-one {version('system-one')}"

# An ONNX repo carries no calibration file, so the published reference checkpoint's
# root values live here.
FETCH_CALIBRATION: dict[str, Any] = {
    "max_len": 512,
    "head_max_len": 192,
    "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
    "temperature_by_options": {
        "choice:2": 1.9063563346862793,
        "choice:3-5": 1.7601518630981445,
        "choice:6-10": 1.0000158548355103,
        "choice:11+": 0.10058280825614929,
        "score:3-5": 1.2514300346374512,
        "noul:2": 1.983399510383606,
    },
}
CALIBRATION_KEYS = tuple(FETCH_CALIBRATION)

# `fetch` serves exactly this repo: the remote paths and the calibration above are the
# published reference checkpoint's, so a `--repo` flag would only let callers mislabel
# someone else's graph.
FETCH_REPO = "rarha/laya-onnx"
DEFAULT_NAME = "laya"
FETCH_VARIANTS = {
    "fp32": ("laya.onnx", "laya.onnx.data"),
    "int8": ("int8/laya_int8.onnx",),
    "fp16": (
        "fp16_onlygpu_unverified/laya_fp16.onnx",
        "fp16_onlygpu_unverified/laya_fp16.onnx.data",
    ),
}


def sha256_file(path: Path) -> str:
    """Digest of one file, read in blocks so a multi-gigabyte graph stays cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(partial(handle.read, DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


class Source(BaseModel):
    """The optional `source` block of `<name>.json`: what produced the graph beside it.

    Only `graph_sha256` is checked at load time; the rest answers "which weights
    produced this number" after the fact. Fields left `None` are dropped on write —
    `weights_sha256` for a downloaded graph, `variant` for an export.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str
    graph_sha256: str
    exporter: str = EXPORTER
    subfolder: str = ""
    weights_sha256: str | None = None
    opset: int | None = None
    variant: str | None = None
    calibration: str | None = None


def source_block(graph: Path, **fields: Any) -> dict[str, Any]:
    """A `Source` for `graph` as plain JSON, ready to merge into the calibration."""
    source = Source(graph_sha256=sha256_file(graph), **fields)
    return source.model_dump(exclude_none=True)


def _default_patterns(data: dict[str, Any]) -> list[str]:
    """HF `allow_patterns` narrow enough to skip the sibling checkpoints in one repo."""
    subfolder = data.get("subfolder", "")
    prefix = f"{subfolder}/" if subfolder else ""
    names = (
        data["config_file"],
        data["weights"],
        f"{data['tokenizer_dir']}/*",
        "encoder/*",
    )
    return [prefix + name for name in names]


class ExportSpec(BaseModel):
    """One export: where the checkpoint comes from and how to build it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    repo: str | None = None
    revision: str | None = None
    subfolder: str = ""
    path: Path | None = None
    builder: str = "system_one.export:build_decision_model"
    config_file: str = CONFIG_FILE
    weights: str = "model.safetensors"
    tokenizer_dir: str = "tokenizer"
    patterns: list[str] = Field(default_factory=_default_patterns)
    calibration: tuple[str, ...] = CALIBRATION_KEYS
    opset: int = 18

    @model_validator(mode="after")
    def _check_source(self) -> ExportSpec:
        if (self.repo is None) == (self.path is None):
            raise ValueError("set exactly one of 'repo' or 'path'")
        return self


def load_spec(path: Path) -> ExportSpec:
    """Read and validate an export spec; YAML is a JSON superset, so one loader does."""
    text = path.read_text()
    try:
        import yaml
    except ImportError:
        return ExportSpec.model_validate(json.loads(text))
    return ExportSpec.model_validate(yaml.safe_load(text))


def fetch_plan(variant: str, name: str = DEFAULT_NAME) -> list[tuple[str, str]]:
    """Remote-to-local file mapping for `fetch`, kept pure so it is testable offline."""
    plan = [
        (remote, f"{name}.onnx.data" if remote.endswith(".data") else f"{name}.onnx")
        for remote in FETCH_VARIANTS[variant]
    ]
    plan.append(("tokenizer.json", "tokenizer/tokenizer.json"))
    return plan
