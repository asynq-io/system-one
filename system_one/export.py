"""Trace a decision model (encoder + decision head) into a single ONNX graph.

Inputs : input_ids [B,L] i64, attention_mask [B,L] i64, marker_pos [B,K] i64,
         marker_mask [B,K] bool, qtype [B] i64
Outputs: logits [B,K] f32 (uncalibrated; masked slots = -1e4), act_probs [B,2] f32

The signature is fixed: it is the contract `system_one.backends.onnx` decodes.
Everything else — where the checkpoint lives, how it is built — comes from the spec.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    from system_one.cli import ExportSpec


class DecisionGraph(torch.nn.Module):
    """Wraps the decision model so the action head's softmax lives inside the graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, act = self.model(
            input_ids, attention_mask, marker_pos, marker_mask, qtype
        )
        return logits, torch.softmax(act.float(), -1)


def build_laya(config: dict[str, Any], model_dir: Path) -> torch.nn.Module:
    """Default builder: the laya reference implementation."""
    from laya.common import build_model

    model = build_model(config, encoder_dir=str(model_dir / "encoder"))
    model.encoder.config.reference_compile = False
    return cast("torch.nn.Module", model)


def resolve_source(spec: ExportSpec) -> Path:
    if spec.path is not None:
        return spec.path
    assert spec.repo is not None
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(  # nosec B615
            spec.repo, revision=spec.revision, allow_patterns=spec.patterns
        )
    )
    return snapshot / spec.subfolder if spec.subfolder else snapshot


def load_graph(
    spec: ExportSpec, model_dir: Path
) -> tuple[DecisionGraph, dict[str, Any]]:
    config: dict[str, Any] = json.loads((model_dir / spec.config_file).read_text())
    module_name, _, function_name = spec.builder.partition(":")
    build = getattr(importlib.import_module(module_name), function_name)
    model = build(config, model_dir)
    model.load_state_dict(load_file(model_dir / spec.weights), strict=True)
    model.eval()
    return DecisionGraph(model), config


def sample_inputs() -> tuple[torch.Tensor, ...]:
    attention_mask = torch.ones(2, 40, dtype=torch.long)
    attention_mask[1, 30:] = 0
    return (
        torch.randint(5, 1000, (2, 40)),
        attention_mask,
        torch.tensor([[3, 9, 15, 21], [3, 9, 0, 0]]),
        torch.tensor([[True, True, True, True], [True, True, False, False]]),
        torch.tensor([0, 2]),
    )


def export(
    graph: DecisionGraph,
    example: tuple[torch.Tensor, ...],
    target: Path,
    opset: int = 18,
) -> None:
    batch = torch.export.Dim("batch")
    seq = torch.export.Dim("seq", min=8)
    options = torch.export.Dim("options", min=2)
    # Grad must stay enabled: nn.TransformerEncoderLayer's fused fast path is taken only under
    # no_grad and is not exportable.
    program = torch.onnx.export(
        graph,
        example,
        opset_version=opset,
        dynamo=True,
        optimize=True,
        input_names=[
            "input_ids",
            "attention_mask",
            "marker_pos",
            "marker_mask",
            "qtype",
        ],
        output_names=["logits", "act_probs"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
            "marker_pos": {0: batch, 1: options},
            "marker_mask": {0: batch, 1: options},
            "qtype": {0: batch},
        },
    )
    assert program is not None
    program.save(str(target), external_data=True)


def report_parity(
    graph: DecisionGraph, example: tuple[torch.Tensor, ...], target: Path
) -> None:
    with torch.no_grad():
        reference_logits, reference_act = graph(*example)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    logits, act = session.run(
        None, dict(zip(names, (t.numpy() for t in example), strict=True))
    )
    print(f"max |dlogits| = {np.abs(logits - reference_logits.numpy()).max():.3e}")
    print(f"max |dact|    = {np.abs(act - reference_act.numpy()).max():.3e}")


def run_export(spec: ExportSpec, out_dir: Path) -> None:
    """Export `spec` into `<out_dir>/<name>.onnx`, `<name>.json` and `tokenizer/`."""
    model_dir = resolve_source(spec)
    graph, config = load_graph(spec, model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = out_dir / f"{spec.name}.onnx"
    example = sample_inputs()
    export(graph, example, target, spec.opset)

    shutil.copytree(
        model_dir / spec.tokenizer_dir, out_dir / "tokenizer", dirs_exist_ok=True
    )
    calibration = {key: config[key] for key in spec.calibration}
    (out_dir / f"{spec.name}.json").write_text(json.dumps(calibration, indent=1))

    report_parity(graph, example, target)
    written = sum(f.stat().st_size for f in out_dir.glob(f"{spec.name}.onnx*"))
    print(f"wrote {target} ({written / 1e6:.0f} MB)")
