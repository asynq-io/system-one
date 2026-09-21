"""`system-one` command line: fetch the published graph, export a checkpoint, run MCP."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

ONNX_REPO = "rarha/laya-onnx"
CONFIG_FILE = "rl_agent_config.json"

# An ONNX repo carries no calibration file, so laya's published root values live here.
LAYA_CALIBRATION: dict[str, Any] = {
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
CALIBRATION_KEYS = tuple(LAYA_CALIBRATION)

VARIANT_GRAPHS = {
    "fp32": ("laya.onnx", "laya.onnx.data"),
    "int8": ("int8/laya_int8.onnx",),
    "fp16": (
        "fp16_onlygpu_unverified/laya_fp16.onnx",
        "fp16_onlygpu_unverified/laya_fp16.onnx.data",
    ),
}

EXTRA_HINTS = {
    "export": "system-one export needs torch and the ONNX tooling: "
    "pip install 'system-one[export]'",
    "fetch": "system-one fetch needs huggingface_hub: pip install 'system-one[hub]'",
}


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
    builder: str = "system_one.export:build_laya"
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


def fetch_plan(variant: str) -> list[tuple[str, str]]:
    """Remote-to-local file mapping for `fetch`, kept pure so it is testable offline."""
    plan = [
        (remote, "laya.onnx.data" if remote.endswith(".data") else "laya.onnx")
        for remote in VARIANT_GRAPHS[variant]
    ]
    plan.append(("tokenizer.json", "tokenizer/tokenizer.json"))
    return plan


def fetch(
    out_dir: Path,
    variant: str = "fp32",
    revision: str = "main",
    repo: str = ONNX_REPO,
) -> None:
    """Download an already-exported laya graph into the layout `load_model` reads."""
    from huggingface_hub import hf_hub_download

    for remote, local in fetch_plan(variant):
        target = out_dir / local
        target.parent.mkdir(parents=True, exist_ok=True)
        source = hf_hub_download(repo, remote, revision=revision)  # nosec B615
        shutil.copyfile(source, target)
        print(f"{remote} -> {target}")

    calibration = out_dir / "laya.json"
    calibration.write_text(json.dumps(LAYA_CALIBRATION, indent=1))
    print(f"wrote {calibration}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="system-one")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export", help="trace a checkpoint into one ONNX graph"
    )
    export.add_argument("spec", type=Path, help="export spec, YAML or JSON")
    export.add_argument("-o", "--out-dir", type=Path, default=Path("onnx"))

    download = commands.add_parser("fetch", help="download the published laya graph")
    download.add_argument("--variant", choices=sorted(VARIANT_GRAPHS), default="fp32")
    download.add_argument("-o", "--out-dir", type=Path, default=Path("onnx"))
    download.add_argument("--revision", default="main")
    download.add_argument(
        "--repo", default=ONNX_REPO, help="Hugging Face repo holding the ONNX graph"
    )

    commands.add_parser(
        "mcp", add_help=False, help="run the MCP server (see system-one-mcp --help)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command == "mcp":
        from system_one.mcp import main as mcp_main

        mcp_main(passthrough)
        return
    if passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")

    try:
        if args.command == "fetch":
            fetch(args.out_dir, args.variant, args.revision, args.repo)
        else:
            from system_one.export import run_export

            run_export(load_spec(args.spec), args.out_dir)
    except ImportError as exc:
        raise ImportError(EXTRA_HINTS[args.command]) from exc


if __name__ == "__main__":
    main()
