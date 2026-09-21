import builtins
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from pydantic import ValidationError

from system_one.cli import (
    LAYA_CALIBRATION,
    ExportSpec,
    fetch,
    fetch_plan,
    load_spec,
    main,
)

YAML_SPEC = """
name: laya-multi
repo: convaiinnovations/laya
subfolder: multilingual
"""
JSON_SPEC = {
    "name": "laya-multi",
    "repo": "convaiinnovations/laya",
    "subfolder": "multilingual",
}


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_yaml_and_json_specs_are_equivalent(tmp_path: Path) -> None:
    from_yaml = load_spec(write(tmp_path, "spec.yaml", YAML_SPEC))
    from_json = load_spec(write(tmp_path, "spec.json", json.dumps(JSON_SPEC)))
    assert from_yaml == from_json
    assert from_yaml.name == "laya-multi"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="varient"):
        load_spec(
            write(tmp_path, "spec.yaml", "name: laya\nrepo: x\nvarient: english\n")
        )


@pytest.mark.parametrize(
    "fields",
    [{}, {"repo": "convaiinnovations/laya", "path": "checkpoints/laya"}],
)
def test_exactly_one_source_is_required(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ExportSpec.model_validate({"name": "laya", **fields})


def test_default_patterns_carry_the_subfolder() -> None:
    spec = ExportSpec(
        name="laya", repo="convaiinnovations/laya", subfolder="multilingual"
    )
    assert spec.patterns == [
        "multilingual/rl_agent_config.json",
        "multilingual/model.safetensors",
        "multilingual/tokenizer/*",
        "multilingual/encoder/*",
    ]
    root = ExportSpec(name="laya", repo="convaiinnovations/laya")
    assert root.patterns == [
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/*",
        "encoder/*",
    ]


def test_explicit_patterns_win() -> None:
    spec = ExportSpec(
        name="laya", path=Path("checkpoints/laya"), patterns=["*.safetensors"]
    )
    assert spec.patterns == ["*.safetensors"]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (
            "fp32",
            [("laya.onnx", "laya.onnx"), ("laya.onnx.data", "laya.onnx.data")],
        ),
        ("int8", [("int8/laya_int8.onnx", "laya.onnx")]),
        (
            "fp16",
            [
                ("fp16_onlygpu_unverified/laya_fp16.onnx", "laya.onnx"),
                ("fp16_onlygpu_unverified/laya_fp16.onnx.data", "laya.onnx.data"),
            ],
        ),
    ],
)
def test_fetch_plan(variant: str, expected: list[tuple[str, str]]) -> None:
    plan = fetch_plan(variant)
    assert plan == [*expected, ("tokenizer.json", "tokenizer/tokenizer.json")]


def test_export_without_the_extra_names_the_pip_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = write(tmp_path, "spec.yaml", "name: laya\nrepo: convaiinnovations/laya\n")
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "system_one.export":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"system-one\[export\]"):
        main(["export", str(spec), "-o", str(tmp_path / "out")])


def test_fetch_without_the_extra_names_the_pip_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "huggingface_hub":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"system-one\[hub\]"):
        main(["fetch", "-o", str(tmp_path / "out")])


def test_mcp_subcommand_dispatches(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("fastmcp")
    with pytest.raises(SystemExit) as exit_info:
        main(["mcp", "--help"])
    assert exit_info.value.code == 0
    assert "system-one-mcp" in capsys.readouterr().out


def test_fetch_writes_the_bundled_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(repo_id: str, filename: str, revision: str = "main") -> str:
        blob = tmp_path / "remote" / filename
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(f"{repo_id}:{filename}@{revision}")
        return str(blob)

    hub = ModuleType("huggingface_hub")
    cast("Any", hub).hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    out_dir = tmp_path / "onnx"
    fetch(out_dir, "int8", repo="me/laya-onnx")

    assert (
        out_dir / "laya.onnx"
    ).read_text() == "me/laya-onnx:int8/laya_int8.onnx@main"
    assert (out_dir / "tokenizer" / "tokenizer.json").exists()
    assert json.loads((out_dir / "laya.json").read_text()) == LAYA_CALIBRATION
