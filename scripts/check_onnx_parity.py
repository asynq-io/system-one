"""Compare the ONNX graph against the reference implementation on real questions.

Works on any directory written by `system-one fetch` or `system-one export`.

Usage:  uv run scripts/check_onnx_parity.py [onnx_dir] [--model laya]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from laya import Agent, triage_questions
from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from transformers import AutoTokenizer

LONG_STATE = (
    "The customer writes: our production deployment failed after the upgrade. " * 40
)
MANY_OPTIONS = {f"k{i:02d}": f"option number {i}" for i in range(18)}

CASES: dict[str, tuple[str | dict[str, Any], dict[str, Any]]] = {
    "triage preset": (
        "I cancelled two weeks ago and still have not seen the refund on my card. This is urgent.",
        triage_questions(),
    ),
    "json state": (
        {
            "subject": "API returning 503",
            "body": "Our integration has been down since 09:00 UTC.",
        },
        triage_questions(),
    ),
    "single question, tiny state": (
        "hi",
        {"a": {"type": "noul", "instructions": "Is this a greeting?"}},
    ),
    "long state, fills max_len": (
        LONG_STATE,
        {"a": {"type": "noul", "instructions": "Is there an outage?"}},
    ),
    "18-option choice": (
        "Refund please",
        {"a": {"type": "choice", "instructions": "Pick one", "criteria": MANY_OPTIONS}},
    ),
    "2-option choice": (
        "Refund please",
        {
            "a": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"yes": None, "no": None},
            }
        },
    ),
    "mixed batch of five": (
        LONG_STATE,
        {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "money", "support": "bugs", "sales": "buying"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "medium", "high", "critical"],
            },
            "outage": {"type": "noul", "instructions": "Is there an outage?"},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "bucket": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": MANY_OPTIONS,
            },
        },
    ),
}


def run_onnx(
    session: ort.InferenceSession,
    tokenizer: Any,
    config: dict[str, Any],
    state: str | dict[str, Any],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items = []
    for qid, definition in questions.items():
        question = Agent._to_internal(definition)
        sequence, markers = build_sequence(
            tokenizer, state, question, config["max_len"], config["head_max_len"]
        )
        if len(markers) != len(render_options(question)):
            raise ValueError(f"question {qid!r} options exceed head_max_len")
        items.append(
            {"ids": sequence, "markers": markers, "qtype": QTYPES[question["t"]]}
        )

    batch = collate_items([items], tokenizer.pad_token_id)
    logits, act = session.run(
        None,
        {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
            "marker_pos": batch["marker_pos"].numpy(),
            "marker_mask": batch["marker_mask"].numpy(),
            "qtype": batch["qtype"].numpy(),
        },
    )
    return postprocess(logits, act, items, questions, config)


def postprocess(
    logits: np.ndarray[Any, Any],
    act: np.ndarray[Any, Any],
    items: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for row, (qid, definition) in enumerate(questions.items()):
        question = Agent._to_internal(definition)
        count = len(items[row]["markers"])
        qtype = QTYPES[question["t"]]
        scale = config["temperature_by_options"].get(
            temp_bucket(qtype, count), config["temperature"][qtype]
        )
        centred = logits[row, :count] / max(1e-3, float(scale))
        probabilities = np.exp(centred - centred.max())
        probabilities /= probabilities.sum()

        confidence = round(confidence_from_probs(probabilities, count), 4)
        action = {"act_probability": round(float(act[row, 0]), 4)}

        if question["t"] == "choice":
            labels = list(question["crit"].keys())
            answers[qid] = {
                "type": "choice",
                "choice": labels[int(probabilities.argmax())],
                "probabilities": {
                    k: round(float(v), 4)
                    for k, v in zip(labels, probabilities, strict=True)
                },
                "confidence": confidence,
                "action": action,
            }
        elif question["t"] == "score":
            answers[qid] = {
                "type": "score",
                "score": round(float((np.arange(count) * probabilities).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(question["crit"])},
                "probabilities": {
                    str(i): round(float(v), 4) for i, v in enumerate(probabilities)
                },
                "confidence": confidence,
                "action": action,
            }
        else:
            true_probability = float(probabilities[1])
            answers[qid] = {
                "type": "noul",
                "noul": round(true_probability, 4),
                "confidence": round(max(true_probability, 1.0 - true_probability), 4),
                "action": action,
            }
    return answers


def max_numeric_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        return max((max_numeric_delta(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return abs(float(left) - float(right))
    if left != right:
        raise AssertionError(f"mismatch: {left!r} != {right!r}")
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_dir", nargs="?", type=Path, default=Path("onnx"))
    parser.add_argument("--model", default="laya")
    args = parser.parse_args()

    config = json.loads((args.onnx_dir / f"{args.model}.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - local directory
        str(args.onnx_dir / "tokenizer")
    )
    session = ort.InferenceSession(
        str(args.onnx_dir / f"{args.model}.onnx"), providers=["CPUExecutionProvider"]
    )
    agent = Agent(device="cpu")

    worst = 0.0
    for name, (state, questions) in CASES.items():
        expected = agent.system_one(state, questions)["answers"]
        actual = run_onnx(session, tokenizer, config, state, questions)
        delta = max_numeric_delta(expected, actual)
        worst = max(worst, delta)
        print(f"{name:<28} max |delta| = {delta:.4f}")

    assert worst < 5e-4, (
        f"ONNX answers diverge from the reference implementation by {worst}"
    )
    print(f"OK: worst answer delta {worst:.2e}")


if __name__ == "__main__":
    main()
