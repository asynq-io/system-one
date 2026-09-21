"""`system-one` command line: fetch the published graph, export a checkpoint, run MCP."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from system_one.catalog import (
    DEFAULT_NAME,
    FETCH_CALIBRATION,
    FETCH_REPO,
    FETCH_VARIANTS,
    fetch_plan,
    load_spec,
    source_block,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

EXTRA_HINTS = {
    "export": "system-one export needs torch and the ONNX tooling: "
    "pip install 'system-one[export]'",
    "fetch": "system-one fetch needs huggingface_hub: pip install 'system-one[hub]'",
}


def fetch(
    out_dir: Path,
    variant: str = "fp32",
    revision: str = "main",
    name: str = DEFAULT_NAME,
) -> None:
    """Download the published reference graph into the layout `load_model` reads."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(EXTRA_HINTS["fetch"]) from exc

    resolved = revision
    for remote, local in fetch_plan(variant, name):
        target = out_dir / local
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(
            hf_hub_download(FETCH_REPO, remote, revision=revision)  # nosec B615
        )
        shutil.copyfile(source, target)
        if "snapshots" in source.parts:
            resolved = source.parts[source.parts.index("snapshots") + 1]
        print(f"{remote} -> {target}")

    calibration = out_dir / f"{name}.json"
    calibration.write_text(
        json.dumps(
            {
                **FETCH_CALIBRATION,
                "source": source_block(
                    out_dir / f"{name}.onnx",
                    repo=FETCH_REPO,
                    revision=resolved,
                    variant=variant,
                    calibration="system_one.catalog:FETCH_CALIBRATION",
                ),
            },
            indent=1,
        )
    )
    print(f"wrote {calibration}")


def run_export(spec_path: Path, out_dir: Path) -> None:
    """Validate the spec first, then pull in torch to trace it."""
    spec = load_spec(spec_path)
    try:
        from system_one.export import run_export as trace
    except ImportError as exc:
        raise ImportError(EXTRA_HINTS["export"]) from exc
    trace(spec, out_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="system-one")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export", help="trace a checkpoint into one ONNX graph"
    )
    export.add_argument("spec", type=Path, help="export spec, YAML or JSON")
    export.add_argument("-o", "--out-dir", type=Path, default=Path("onnx"))

    download = commands.add_parser(
        "fetch", help=f"download the published reference graph from {FETCH_REPO}"
    )
    download.add_argument("--variant", choices=sorted(FETCH_VARIANTS), default="fp32")
    download.add_argument("-o", "--out-dir", type=Path, default=Path("onnx"))
    download.add_argument("--revision", default="main")
    download.add_argument("--name", default=DEFAULT_NAME)

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

    if args.command == "fetch":
        fetch(args.out_dir, args.variant, args.revision, args.name)
    else:
        run_export(args.spec, args.out_dir)


if __name__ == "__main__":
    main()
