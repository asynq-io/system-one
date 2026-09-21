"""Compare the shipped ONNX backend against the reference implementation.

This drives `SystemOne(backend="onnx")` exactly as a user would, so it guards the
whole pure-numpy pipeline in `system_one.backends.onnx` — the rendering, the
tokenisation, the collation, the temperature lookup and the answer construction —
not just the traced graph.

Works on any directory written by `system-one fetch` or `system-one export`.

Usage:  uv run scripts/check_onnx_parity.py [onnx_dir] [--model laya]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from laya import Agent, triage_questions

from system_one import SystemOne

# 373 tokens: the longest state that still fits every case below. The 18-option
# `bucket` question spends 192 of `max_len` on its head and leaves room for 377, and
# the SDK now raises on an overlong state where the reference truncates it — so a
# longer state here would compare a refusal against a silently cut sequence.
LONG_STATE = (
    "The customer writes: our production deployment failed after the upgrade. " * 31
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
    "long state, near max_len": (
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


def max_numeric_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        missing = set(left) - set(right)
        if missing:
            raise AssertionError(f"missing keys: {sorted(missing)}")
        return max((max_numeric_delta(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return abs(float(left) - float(right))
    if left != right:
        raise AssertionError(f"mismatch: {left!r} != {right!r}")
    return 0.0


def comparable(answers: dict[str, Any]) -> dict[str, Any]:
    """Drop what the SDK contract defines differently from the reference.

    `action` the SDK does not report at all. `confidence` it derives on its own
    scale — the probability of the reported label for a choice, `1 - 2 * sd / (k - 1)`
    for a score — where the reference always uses normalised entropy. Nothing is lost
    by skipping it: the SDK derives it from `probabilities` / `noul`, both compared
    exactly here, so the formula itself is `tests/test_schemas.py`'s job.
    """
    dropped = ("action", "confidence")
    return {
        qid: {k: v for k, v in answer.items() if k not in dropped}
        for qid, answer in answers.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_dir", nargs="?", type=Path, default=Path("onnx"))
    parser.add_argument("--model", default="laya")
    args = parser.parse_args()

    agent = Agent(device="cpu")
    worst = 0.0
    with SystemOne(backend="onnx", onnx_dir=args.onnx_dir, model=args.model) as sdk:
        for name, (state, questions) in CASES.items():
            expected = comparable(agent.system_one(state, questions)["answers"])
            actual = sdk.ask(state, questions).model_dump(mode="json")["answers"]
            delta = max_numeric_delta(expected, actual)
            worst = max(worst, delta)
            print(f"{name:<28} max |delta| = {delta:.4f}")

    assert worst < 5e-4, (
        f"ONNX answers diverge from the reference implementation by {worst}"
    )
    print(f"OK: worst answer delta {worst:.2e}")


if __name__ == "__main__":
    main()
