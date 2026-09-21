"""Local backend: a pure-numpy reimplementation of the reference pipeline around the graph."""

from __future__ import annotations

import asyncio
import json  # not orjson: the reference pipeline's separators are load-bearing
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeAlias

import numpy as np
import onnxruntime
import orjson
from tokenizers import Tokenizer as HFTokenizer

from system_one.catalog import DEFAULT_NAME, sha256_file
from system_one.errors import SystemOneError
from system_one.schemas import (
    ROUNDING,
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    State,
    SystemOneOutput,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from system_one.schemas import SystemOneInput
    from system_one.settings import Settings

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {index: name for name, index in QTYPES.items()}
OPTION_BUCKETS = ((2, "2"), (5, "3-5"), (10, "6-10"))
LARGE_BUCKET = "11+"

MASK_TEXT = "[MASK]"
SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]", "[MASK]")
MAX_OPTION_IDS = 48
MIN_OPTION_BUDGET = 16
MIN_OPTION_IDS = 4
MIN_HEAD_IDS = 8
MIN_TEMPERATURE = 1e-3


class Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...


class Tokenizer(Protocol):
    def encode(self, sequence: str, *, add_special_tokens: bool = ...) -> Encoding: ...

    def token_to_id(self, token: str) -> int | None: ...


class Specials(NamedTuple):
    cls_id: int
    sep_id: int
    pad_id: int
    mask_id: int


class Item(NamedTuple):
    ids: list[int]
    markers: list[int]
    qtype: int


class Prompt(NamedTuple):
    ids: list[int]
    markers: list[int]
    options: int
    state_ids: int
    room: int


class Model(NamedTuple):
    session: Any
    tokenizer: Tokenizer
    config: dict[str, Any]
    pad_id: int
    max_options: int | None


def load_config(onnx_dir: Path, model: str) -> dict[str, Any]:
    config: dict[str, Any] = orjson.loads((onnx_dir / f"{model}.json").read_bytes())
    return config


def check_graph_digest(graph: Path, config: dict[str, Any]) -> None:
    """Refuse a calibration that was written for a different graph.

    Hashing is cheap — the weights live in the `.onnx.data` sidecar, so this reads
    only the few megabytes of graph proto. A hand-placed graph carries no `source`
    block at all and is left alone.
    """
    expected = config.get("source", {}).get("graph_sha256")
    if expected and expected != sha256_file(graph):
        message = (
            f"{graph.name} does not match the graph_sha256 recorded in "
            f"{graph.stem}.json, so the calibration and the graph came from "
            f"different places. Re-run `system-one fetch` or `system-one export` "
            f"to write both together."
        )
        raise SystemOneError(message)


@lru_cache(maxsize=2)
def load_model(onnx_dir: Path, model: str) -> Model:
    """Load `<model>.onnx` from `onnx_dir`, once per `(directory, model)` pair.

    Sessions hold gigabytes of weights, so the cache is what makes a second agent —
    or a second model on the same agent — cheap.
    """
    graph = onnx_dir / f"{model}.onnx"
    tokenizer_file = onnx_dir / "tokenizer" / "tokenizer.json"
    for artifact in (graph, onnx_dir / f"{model}.json", tokenizer_file):
        if not artifact.exists():
            message = (
                f"No ONNX artifact at {artifact}. Point SYSTEM_ONE_ONNX_DIR at the "
                f"directory holding {model}.onnx, {model}.json and tokenizer/, and "
                f"SYSTEM_ONE_MODEL at {model!r}; `system-one fetch` writes that layout."
            )
            raise SystemOneError(message)
    config = load_config(onnx_dir, model)
    check_graph_digest(graph, config)
    tokenizer: Tokenizer = HFTokenizer.from_file(str(tokenizer_file))
    # CPU only: the CoreML EP cannot build this graph, and fixed shapes cost ~11x.
    # See docs/usage/local-model.md, "Apple Silicon".
    session = onnxruntime.InferenceSession(
        str(graph), providers=["CPUExecutionProvider"]
    )
    marker_pos = session.get_inputs()[2]
    max_options = next(
        (dim for dim in marker_pos.shape[1:] if isinstance(dim, int)), None
    )
    return Model(
        session,
        tokenizer,
        config,
        load_specials(tokenizer).pad_id,
        max_options,
    )


def load_specials(tokenizer: Tokenizer) -> Specials:
    ids = []
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            message = f"The tokenizer has no {token} token."
            raise SystemOneError(message)
        ids.append(token_id)
    return Specials(*ids)


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text.replace(MASK_TEXT, " "), add_special_tokens=False).ids


def serialize_state(state: State) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question: Question) -> list[str]:
    """Option texts in label-index order. A `noul` is always `[false, true]`."""
    match question:
        case Choice():
            return [
                label if value in (None, "") else f"{label}: {render_criterion(value)}"
                for label, value in question.criteria.items()
            ]
        case Score():
            return [
                f"level {index}: {render_criterion(value)}"
                for index, value in enumerate(question.criteria)
            ]
        case Noul():
            criteria = question.criteria
            outcomes = (
                (
                    "false",
                    criteria.false if criteria else None,
                    "no, the statement does not hold",
                ),
                (
                    "true",
                    criteria.true if criteria else None,
                    "yes, the statement holds",
                ),
            )
            return [
                f"{label}: {default if value in (None, '') else render_criterion(value)}"
                for label, value, default in outcomes
            ]


def build_sequence(
    tokenizer: Tokenizer,
    specials: Specials,
    state: State,
    question: Question,
    config: dict[str, Any],
) -> Prompt:
    """`[CLS] <type> question: <ins> [SEP] (MASK opt)... [SEP] state [SEP]`.

    Returns the ids plus the numbers that say whether everything fitted: a marker
    shortfall means the head overflowed, `state_ids > room` means the state did.
    """
    max_len, head_max_len = config["max_len"], config["head_max_len"]
    head_ids = encode(tokenizer, f"{question.type} question: {question.instructions}")
    options = render_options(question)
    # ponytail: options are capped at MAX_OPTION_IDS tokens each and truncated to an
    # equal share when they do not fit; author-controlled text, so silent unlike the
    # state. Raise instead if authors start hitting it.
    option_ids = [
        [specials.mask_id, *encode(tokenizer, " " + option)[:MAX_OPTION_IDS]]
        for option in options
    ]
    budget = head_max_len - sum(len(ids) for ids in option_ids)
    if budget < MIN_OPTION_BUDGET:
        per_option = max(
            MIN_OPTION_IDS,
            (head_max_len - MIN_OPTION_BUDGET) // max(1, len(option_ids)),
        )
        option_ids = [ids[:per_option] for ids in option_ids]
        budget = head_max_len - sum(len(ids) for ids in option_ids)

    ids = [specials.cls_id, *head_ids[: max(MIN_HEAD_IDS, budget)], specials.sep_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(specials.sep_id)

    room = max(0, max_len - len(ids) - 1)
    state_ids = encode(tokenizer, serialize_state(state))
    ids += [*state_ids[:room], specials.sep_id]
    kept = [marker for marker in markers if marker < max_len]
    return Prompt(ids[:max_len], kept, len(options), len(state_ids), room)


def build_items(
    tokenizer: Tokenizer,
    config: dict[str, Any],
    request: SystemOneInput,
    max_options: int | None = None,
) -> list[Item]:
    specials = load_specials(tokenizer)
    max_len, head_max_len = config["max_len"], config["head_max_len"]
    items = []
    for name, question in request.questions.items():
        prompt = build_sequence(tokenizer, specials, request.state, question, config)
        if max_options is not None and prompt.options > max_options:
            message = (
                f"Question {name!r} has {prompt.options} options but the graph caps "
                f"marker_pos at {max_options}. Export a graph with a dynamic options "
                f"dimension, or ask fewer criteria at a time."
            )
            raise SystemOneError(message)
        if len(prompt.markers) != prompt.options:
            message = (
                f"Question {name!r} has options exceeding head_max_len={head_max_len}."
            )
            raise SystemOneError(message)
        if prompt.state_ids > prompt.room:
            message = (
                f"Question {name!r}: state is {prompt.state_ids} tokens but "
                f"max_len={max_len} leaves room for {prompt.room}. Shorten the state "
                f"or split it across asks."
            )
            raise SystemOneError(message)
        items.append(Item(prompt.ids, prompt.markers, QTYPES[question.type]))
    return items


def collate(items: Sequence[Item], pad_id: int) -> dict[str, Array]:
    """Right-pad into one batch. `marker_pos` pads with 0, which is why `marker_mask` exists."""
    rows = len(items)
    length = max(len(item.ids) for item in items)
    options = max(len(item.markers) for item in items)
    input_ids = np.full((rows, length), pad_id, dtype=np.int64)
    attention_mask = np.zeros((rows, length), dtype=np.int64)
    marker_pos = np.zeros((rows, options), dtype=np.int64)
    marker_mask = np.zeros((rows, options), dtype=np.bool_)

    for row, item in enumerate(items):
        input_ids[row, : len(item.ids)] = item.ids
        attention_mask[row, : len(item.ids)] = 1
        marker_pos[row, : len(item.markers)] = item.markers
        marker_mask[row, : len(item.markers)] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": np.array([item.qtype for item in items], dtype=np.int64),
    }


def temp_bucket(qtype: int, options: int) -> str:
    size = next(
        (label for limit, label in OPTION_BUCKETS if options <= limit), LARGE_BUCKET
    )
    return f"{QTYPE_NAMES[qtype]}:{size}"


def softmax(logits: Array, scale: float) -> Array:
    centred = logits / max(MIN_TEMPERATURE, scale)
    weights = np.exp(centred - centred.max())
    probabilities: Array = weights / weights.sum()
    return probabilities


def build_answer(question: Question, probabilities: Array) -> Answer:
    match question:
        case Choice():
            labels = list(question.criteria)
            return ChoiceAnswer(
                choice=labels[int(probabilities.argmax())],
                probabilities={
                    label: round(float(value), ROUNDING)
                    for label, value in zip(labels, probabilities, strict=True)
                },
            )
        case Score():
            levels = np.arange(len(probabilities))
            return ScoreAnswer(
                score=round(float((levels * probabilities).sum()), ROUNDING),
                legend=dict(enumerate(question.criteria)),
                probabilities={
                    index: round(float(value), ROUNDING)
                    for index, value in enumerate(probabilities)
                },
            )
        case Noul():
            return NoulAnswer(noul=round(float(probabilities[1]), ROUNDING))


def postprocess(
    logits: Array,
    items: Sequence[Item],
    request: SystemOneInput,
    config: dict[str, Any],
) -> SystemOneOutput:
    answers: dict[str, Answer] = {}
    for row, (name, question) in enumerate(request.questions.items()):
        options = len(items[row].markers)
        qtype = QTYPES[question.type]
        scale = config["temperature_by_options"].get(
            temp_bucket(qtype, options), config["temperature"][qtype]
        )
        probabilities = softmax(logits[row, :options], float(scale))
        answers[name] = build_answer(question, probabilities)
    return SystemOneOutput(
        model=request.model,
        usage=Usage(input_tokens=sum(len(item.ids) for item in items), output_tokens=0),
        answers=answers,
    )


class ONNXBackend:
    """Runs the exported graph in-process. No torch, no transformers, no network."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.model or DEFAULT_NAME
        load_model(settings.onnx_dir, self.model)

    def ask(self, request: SystemOneInput) -> SystemOneOutput:
        model = load_model(self.settings.onnx_dir, request.model)
        items = build_items(model.tokenizer, model.config, request, model.max_options)
        logits = model.session.run(None, collate(items, model.pad_id))[0]
        return postprocess(logits, items, request, model.config)

    def close(self) -> None:
        """Sessions are shared through `load_model`, so there is nothing to release."""


class AsyncONNXBackend:
    """The sync backend on a worker thread — ONNX Runtime releases the GIL anyway."""

    def __init__(self, settings: Settings) -> None:
        self.backend = ONNXBackend(settings)
        self.model = self.backend.model

    async def ask(self, request: SystemOneInput) -> SystemOneOutput:
        return await asyncio.to_thread(self.backend.ask, request)

    async def close(self) -> None:
        """Sessions are shared through `load_model`, so there is nothing to release."""
