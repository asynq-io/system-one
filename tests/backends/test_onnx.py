import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
import onnxruntime
import pytest

from system_one import SystemOneError
from system_one.backends import onnx as onnx_backend
from system_one.backends.onnx import (
    Item,
    build_items,
    build_sequence,
    collate,
    load_model,
    load_specials,
    postprocess,
    render_options,
    softmax,
    temp_bucket,
)
from system_one.schemas import (
    Choice,
    Noul,
    SystemOneInput,
    choice_confidence,
    score_confidence,
)

CONFIG: dict[str, Any] = {
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

CLS, SEP, PAD, MASK = 1, 2, 3, 4
FIRST_WORD_ID = 10


class StubEncoding(NamedTuple):
    ids: list[int]


class StubTokenizer:
    """One id per whitespace-separated word, so token counts are hand-countable."""

    specials: ClassVar[dict[str, int]] = {
        "[CLS]": CLS,
        "[SEP]": SEP,
        "[PAD]": PAD,
        "[MASK]": MASK,
    }

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> StubEncoding:
        return StubEncoding([FIRST_WORD_ID + i for i, _ in enumerate(sequence.split())])

    def token_to_id(self, token: str) -> int | None:
        return self.specials.get(token)


def make_request(questions: dict[str, Any], state: Any = "hi there") -> SystemOneInput:
    return SystemOneInput.model_validate(
        {"state": state, "model": "jev-latest", "questions": questions}
    )


def test_noul_options_are_false_then_true() -> None:
    assert render_options(Noul(instructions="Down?")) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]


def test_markers_point_at_the_mask_before_each_option() -> None:
    tokenizer = StubTokenizer()
    ids, markers, *_ = build_sequence(
        tokenizer,
        load_specials(tokenizer),
        "hi there",
        Noul(instructions="Is it urgent?"),
        CONFIG,
    )

    assert markers == [7, 15]
    assert ids[0] == CLS
    assert ids[6] == SEP
    assert [ids[marker] for marker in markers] == [MASK, MASK]
    assert ids[-1] == SEP


def test_tight_option_budget_truncates_every_option_equally() -> None:
    tokenizer = StubTokenizer()
    question = {
        "type": "choice",
        "instructions": "Which one",
        "criteria": {f"label{index} a b c d": None for index in range(3)},
    }
    request = make_request({"q": question})
    config = {**CONFIG, "head_max_len": 20}

    ids, markers, *_ = build_sequence(
        tokenizer,
        load_specials(tokenizer),
        request.state,
        request.questions["q"],
        config,
    )

    assert [markers[1] - markers[0], markers[2] - markers[1]] == [4, 4]
    assert [ids[marker] for marker in markers] == [MASK] * 3


def test_options_pushed_past_the_sequence_limit_are_rejected() -> None:
    request = make_request(
        {
            "q": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": ["a", "b", "c"],
            }
        }
    )

    with pytest.raises(SystemOneError, match="head_max_len"):
        build_items(StubTokenizer(), {**CONFIG, "max_len": 8}, request)


@pytest.mark.parametrize(
    ("qtype", "options", "expected"),
    [
        (0, 2, "choice:2"),
        (0, 5, "choice:3-5"),
        (0, 6, "choice:6-10"),
        (0, 10, "choice:6-10"),
        (0, 11, "choice:11+"),
        (1, 3, "score:3-5"),
        (2, 2, "noul:2"),
    ],
)
def test_temp_bucket_boundaries(qtype: int, options: int, expected: str) -> None:
    assert temp_bucket(qtype, options) == expected


def test_collate_right_pads_and_masks_the_shorter_row() -> None:
    batch = collate([Item([5, 6, 7], [1, 2], 0), Item([8], [0], 2)], PAD)

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, PAD, PAD]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["marker_pos"].tolist() == [[1, 2], [0, 0]]
    assert batch["marker_mask"].tolist() == [[True, True], [True, False]]
    assert batch["qtype"].tolist() == [0, 2]


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(1.0, [0.7311, 0.2689]), (2.0, [0.6225, 0.3775]), (0.0, [1.0, 0.0])],
)
def test_softmax_divides_by_the_temperature(
    scale: float, expected: list[float]
) -> None:
    probabilities = softmax(np.array([1.0, 0.0]), scale)

    assert [round(float(value), 4) for value in probabilities] == expected


def test_postprocess_scores_the_expectation_of_a_uniform_distribution() -> None:
    request = make_request(
        {
            "level": {
                "type": "score",
                "instructions": "How bad?",
                "criteria": ["low", "medium", "high"],
            },
            "down": {"type": "noul", "instructions": "Is it down?"},
        }
    )
    items = [Item([1, 2, 3], [1, 2, 3], 1), Item([1, 2], [1, 2], 2)]
    logits = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1e4]])

    response = postprocess(logits, items, request, CONFIG)
    score = response.scores["level"]

    assert score.score == 1.0
    assert score.confidence == 0.1835
    assert score.probabilities == {0: 0.3333, 1: 0.3333, 2: 0.3333}
    assert score.legend == {0: "low", 1: "medium", 2: "high"}
    assert response.nouls["down"].noul == 0.5
    assert response.nouls["down"].confidence == 0.5
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 0
    assert response.model == "jev-latest"


def test_postprocess_picks_the_argmax_label() -> None:
    request = make_request(
        {
            "queue": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": ["billing", "technical"],
            }
        }
    )
    items = [Item([1, 2], [1, 2], 0)]
    logits = np.array([[0.0, 5.0]])

    choice = postprocess(logits, items, request, CONFIG).choices["queue"]

    assert choice.choice == "technical"
    assert choice.probabilities is not None
    assert sum(choice.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert choice.confidence is not None
    assert choice.confidence > 0.5


def test_build_items_carries_the_question_type() -> None:
    request = make_request(
        {
            "a": {"type": "noul", "instructions": "Down?"},
            "b": {
                "type": "score",
                "instructions": "How bad?",
                "criteria": ["low", "high"],
            },
        }
    )
    items = build_items(StubTokenizer(), CONFIG, request)

    assert [item.qtype for item in items] == [2, 1]
    assert [len(item.markers) for item in items] == [2, 2]


def test_json_state_is_serialized_before_tokenizing() -> None:
    tokenizer = StubTokenizer()
    ids, *_ = build_sequence(
        tokenizer,
        load_specials(tokenizer),
        {"subject": "API down"},
        Noul(instructions="Down?"),
        CONFIG,
    )

    assert ids[-1] == SEP
    assert len(ids) > 0


def test_structured_criteria_render_as_compact_json() -> None:
    options = render_options(
        Choice.model_validate(
            {
                "instructions": "Which?",
                "criteria": {
                    "calm": {"desc": "relaxed"},
                    "loud": ["a", "b"],
                    "n": None,
                },
            }
        )
    )

    assert options == [
        'calm: {"desc": "relaxed"}',
        'loud: ["a", "b"]',
        "n",
    ]


def test_a_tokenizer_without_the_mask_token_is_rejected() -> None:
    class Incomplete(StubTokenizer):
        specials: ClassVar[dict[str, int]] = {"[CLS]": CLS}

    with pytest.raises(SystemOneError, match=r"\[SEP\]"):
        load_specials(Incomplete())


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([1.0], 1.0),
        ([0.8, 0.2], 0.8),
        ([0.5, 0.25, 0.25], 0.5),
    ],
)
def test_choice_confidence_is_the_probability_of_the_reported_label(
    probabilities: list[float], expected: float
) -> None:
    assert choice_confidence(probabilities) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([1.0], 1.0),
        ([0.0, 1.0, 0.0], 1.0),
        ([0.5, 0.5, 0.0], 0.5),
        ([0.5, 0.0, 0.5], 0.0),
    ],
)
def test_score_confidence_tracks_dispersion_around_the_expectation(
    probabilities: list[float], expected: float
) -> None:
    assert score_confidence(probabilities) == pytest.approx(expected)


def test_score_confidence_separates_orderings_entropy_would_conflate() -> None:
    adjacent = score_confidence([0.5, 0.5, 0.0])
    extremes = score_confidence([0.5, 0.0, 0.5])

    assert adjacent > extremes


@pytest.mark.parametrize(
    "probabilities",
    [[], [2.1, 1.9, 1.8], [-0.4, -0.5, -0.6], [0.3, 0.3]],
)
def test_confidence_rejects_anything_that_is_not_a_distribution(
    probabilities: list[float],
) -> None:
    with pytest.raises(ValueError, match="probabilities must"):
        choice_confidence(probabilities)
    with pytest.raises(ValueError, match="probabilities must"):
        score_confidence(probabilities)


LONG_WORDS = "word " * 600


def test_an_overlong_state_is_rejected_instead_of_truncated() -> None:
    request = make_request(
        {"topic": {"type": "noul", "instructions": "Down?"}}, LONG_WORDS
    )

    with pytest.raises(SystemOneError, match=r"'topic': state is 600 tokens"):
        build_items(StubTokenizer(), CONFIG, request)


def test_a_state_that_fits_is_accepted_whole() -> None:
    request = make_request(
        {"topic": {"type": "noul", "instructions": "Down?"}}, "word " * 400
    )

    ids = build_items(StubTokenizer(), CONFIG, request)[0].ids

    assert ids[-401:] == [*range(FIRST_WORD_ID, FIRST_WORD_ID + 400), SEP]


def test_options_beyond_the_graphs_ceiling_are_rejected() -> None:
    request = make_request(
        {
            "queue": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": ["a", "b", "c"],
            }
        }
    )

    with pytest.raises(SystemOneError, match="caps marker_pos at 2"):
        build_items(StubTokenizer(), CONFIG, request, max_options=2)


def test_a_dynamic_graph_imposes_no_option_ceiling() -> None:
    request = make_request(
        {
            "queue": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": ["a", "b", "c"],
            }
        }
    )

    assert len(build_items(StubTokenizer(), CONFIG, request, None)[0].markers) == 3


class StubInput(NamedTuple):
    name: str
    shape: list[Any]


class StubSession:
    """Enough of an `InferenceSession` for `load_model` to read `marker_pos`."""

    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path

    def get_inputs(self) -> list[StubInput]:
        return [
            StubInput("input_ids", ["batch", "seq"]),
            StubInput("attention_mask", ["batch", "seq"]),
            StubInput("marker_pos", ["batch", 2]),
        ]


class StubLoader:
    @staticmethod
    def from_file(_: str) -> StubTokenizer:
        return StubTokenizer()


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")
    (tmp_path / "m.onnx").write_text("graph bytes")
    (tmp_path / "m.json").write_text(json.dumps(CONFIG))
    monkeypatch.setattr(onnx_backend, "HFTokenizer", StubLoader)
    monkeypatch.setattr(onnxruntime, "InferenceSession", StubSession)
    return tmp_path


@pytest.mark.parametrize("missing", ["m.onnx", "m.json", "tokenizer/tokenizer.json"])
def test_every_missing_artifact_names_itself_and_the_env_vars(
    artifacts: Path, missing: str
) -> None:
    (artifacts / missing).unlink()

    with pytest.raises(SystemOneError, match=r"No ONNX artifact at .*SYSTEM_ONE_"):
        load_model(artifacts, "m")


def test_a_calibration_written_for_another_graph_is_rejected(artifacts: Path) -> None:
    config = {**CONFIG, "source": {"graph_sha256": "0" * 64}}
    (artifacts / "m.json").write_text(json.dumps(config))

    with pytest.raises(SystemOneError, match="graph_sha256"):
        load_model(artifacts, "m")


def test_a_matching_graph_digest_loads_and_carries_the_option_ceiling(
    artifacts: Path,
) -> None:
    digest = hashlib.sha256((artifacts / "m.onnx").read_bytes()).hexdigest()
    config = {**CONFIG, "source": {"graph_sha256": digest}}
    (artifacts / "m.json").write_text(json.dumps(config))

    model = load_model(artifacts, "m")

    assert model.max_options == 2
    assert model.pad_id == PAD


def test_a_calibration_without_a_source_block_loads_unchanged(artifacts: Path) -> None:
    assert load_model(artifacts, "m").config == CONFIG
