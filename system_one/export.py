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

from system_one.catalog import sha256_file, source_block

if TYPE_CHECKING:
    from system_one.catalog import ExportSpec


class DecisionModel(torch.nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        d: int = cast("Any", encoder).config.hidden_size
        layer = torch.nn.TransformerEncoderLayer(
            d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.head = (
            torch.nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = torch.nn.Embedding(3, d)
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(d),
            torch.nn.Linear(d, d),
            torch.nn.GELU(),
            torch.nn.Linear(d, 1),
        )
        self.act_head = torch.nn.Sequential(
            torch.nn.Linear(d + 4, 256), torch.nn.GELU(), torch.nn.Linear(256, n_act)
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        markers = torch.gather(h, 1, idx)
        logits = self.scorer(markers).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack(
            [top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], -1
        )
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits


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


def build_decision_model(config: dict[str, Any], model_dir: Path) -> torch.nn.Module:
    """Default builder: an encoder from `transformers` under the decision head above."""
    from transformers import AutoConfig, AutoModel

    loader = cast("Any", AutoModel)
    encoder_dir = model_dir / "encoder"
    if encoder_dir.exists():
        encoder_config = AutoConfig.from_pretrained(str(encoder_dir))  # nosec B615
        encoder = loader.from_config(encoder_config, attn_implementation="sdpa")
    else:
        encoder = loader.from_pretrained(  # nosec B615
            config["encoder"], attn_implementation="sdpa"
        )
    encoder.config.reference_compile = False
    return DecisionModel(
        encoder, config.get("head_layers", 2), len(config.get("act_costs", {})) + 1
    )


def resolve_source(spec: ExportSpec) -> tuple[Path, str]:
    """The directory to read the checkpoint from, and the revision it resolved to."""
    if spec.path is not None:
        return spec.path, spec.revision or "local"
    assert spec.repo is not None
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(  # nosec B615
            spec.repo, revision=spec.revision, allow_patterns=spec.patterns
        )
    )
    model_dir = snapshot / spec.subfolder if spec.subfolder else snapshot
    return model_dir, snapshot.name


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
    model_dir, revision = resolve_source(spec)
    graph, config = load_graph(spec, model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = out_dir / f"{spec.name}.onnx"
    example = sample_inputs()
    export(graph, example, target, spec.opset)

    shutil.copytree(
        model_dir / spec.tokenizer_dir, out_dir / "tokenizer", dirs_exist_ok=True
    )
    calibration = {key: config[key] for key in spec.calibration}
    calibration["source"] = source_block(
        target,
        repo=spec.repo or str(spec.path),
        revision=revision,
        subfolder=spec.subfolder,
        weights_sha256=sha256_file(model_dir / spec.weights),
        opset=spec.opset,
    )
    (out_dir / f"{spec.name}.json").write_text(json.dumps(calibration, indent=1))

    report_parity(graph, example, target)
    written = sum(f.stat().st_size for f in out_dir.glob(f"{spec.name}.onnx*"))
    print(f"wrote {target} ({written / 1e6:.0f} MB)")
