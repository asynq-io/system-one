"""Runs only where the ONNX artifacts are checked out. CI installs no extras and skips these."""

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from system_one import Settings, SystemOne, SystemOneError
from system_one.backends.onnx import (
    AsyncONNXBackend,
    ONNXBackend,
    build_items,
    load_config,
    load_model,
)
from system_one.schemas import SystemOneInput

ONNX_DIR = Path(os.getenv("SYSTEM_ONE_ONNX_DIR", "onnx"))
MODEL = os.getenv("SYSTEM_ONE_MODEL", "laya")

pytestmark = [
    pytest.mark.skipif(
        not (ONNX_DIR / f"{MODEL}.onnx").exists(), reason="onnx artifacts absent"
    ),
    pytest.mark.timeout(120),
]

# 321 tokens, and it must stay under `max_len` minus the longest head below: the
# reference truncates an overlong state, `build_items` raises on one, so a longer
# state here would stop the two being comparable at all.
LONG_STATE = "Our production deployment failed after the upgrade. " * 40
MANY_OPTIONS = {f"k{index:02d}": f"option number {index}" for index in range(18)}

QUESTIONS: dict[str, Any] = {
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
    "bucket": {"type": "choice", "instructions": "Pick one", "criteria": MANY_OPTIONS},
}


@pytest.fixture(scope="module")
def backend() -> ONNXBackend:
    return ONNXBackend(Settings(backend="onnx", onnx_dir=ONNX_DIR, model=MODEL))


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("laya") is None,
    reason="the reference implementation is not installed",
)
@pytest.mark.parametrize(
    "state", ["hi", LONG_STATE, {"subject": "API returning 503", "body": "Down."}]
)
def test_tokenizers_match_the_transformers_reference(state: Any) -> None:
    from laya import Agent as LayaAgent
    from laya.common import QTYPES, build_sequence, render_options
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    config = load_config(ONNX_DIR, MODEL)
    reference = AutoTokenizer.from_pretrained(str(ONNX_DIR / "tokenizer"))
    tokenizer = Tokenizer.from_file(str(ONNX_DIR / "tokenizer" / "tokenizer.json"))
    request = SystemOneInput.model_validate(
        {"state": state, "model": MODEL, "questions": QUESTIONS}
    )

    items = build_items(tokenizer, config, request)

    for item, definition in zip(items, QUESTIONS.values(), strict=True):
        internal = LayaAgent._to_internal(definition)
        expected_ids, expected_markers = build_sequence(
            reference, state, internal, config["max_len"], config["head_max_len"]
        )
        assert item.ids == expected_ids
        assert item.markers == expected_markers
        assert item.qtype == QTYPES[internal["t"]]
        assert len(item.markers) == len(render_options(internal))


def test_ask_answers_every_question(backend: ONNXBackend) -> None:
    response = backend.ask(
        SystemOneInput.model_validate(
            {
                "state": "Refund never arrived and the API is down.",
                "model": MODEL,
                "questions": QUESTIONS,
            }
        )
    )

    assert set(response.answers) == set(QUESTIONS)
    assert response.choices["team"].choice in {"billing", "support", "sales"}
    assert 0.0 <= response.scores["urgency"].score <= 3.0
    assert 0.0 <= response.nouls["outage"].noul <= 1.0
    assert response.usage.input_tokens > 0

    for choice in response.choices.values():
        assert choice.probabilities is not None
        assert sum(choice.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
        assert choice.confidence is not None
        assert 0.0 <= choice.confidence <= 1.0


def test_async_ask_matches_sync(backend: ONNXBackend) -> None:
    request = SystemOneInput.model_validate(
        {"state": "hi", "model": MODEL, "questions": {"a": QUESTIONS["outage"]}}
    )
    async_backend = AsyncONNXBackend(backend.settings)

    assert (
        asyncio.run(async_backend.ask(request)).model_dump()
        == backend.ask(request).model_dump()
    )


def test_agent_selects_the_onnx_backend_from_settings() -> None:
    with SystemOne(backend="onnx", onnx_dir=ONNX_DIR, model=MODEL) as agent:
        response = agent.ask("The site is down.", {"a": QUESTIONS["outage"]})

    assert isinstance(agent.backend, ONNXBackend)
    assert 0.0 <= response.nouls["a"].noul <= 1.0


def test_a_model_is_loaded_once_and_then_served_from_the_cache() -> None:
    assert load_model(ONNX_DIR, MODEL) is load_model(ONNX_DIR, MODEL)


def test_missing_graph_names_the_model_file_and_the_variable(tmp_path: Path) -> None:
    with pytest.raises(SystemOneError, match=r"other\.onnx"):
        ONNXBackend(Settings(backend="onnx", onnx_dir=tmp_path, model="other"))

    # No SYSTEM_ONE_MODEL: the onnx backend fills in the published graph's name.
    with pytest.raises(SystemOneError, match=r"laya\.onnx.*SYSTEM_ONE_ONNX_DIR"):
        ONNXBackend(Settings(backend="onnx", onnx_dir=tmp_path, model=None))
