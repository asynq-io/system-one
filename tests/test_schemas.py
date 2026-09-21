import pytest
from pydantic import TypeAdapter, ValidationError

from system_one.schemas import (
    Answer,
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneInput,
    SystemOneOutput,
)

question_adapter: TypeAdapter[Question] = TypeAdapter(Question)
answer_adapter: TypeAdapter[Answer] = TypeAdapter(Answer)


def test_question_dispatches_on_type() -> None:
    noul = question_adapter.validate_python(
        {"type": "noul", "instructions": "Is it urgent?"}
    )
    choice = question_adapter.validate_python(
        {"type": "choice", "instructions": "Which?", "criteria": {"a": "first"}}
    )
    score = question_adapter.validate_python(
        {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]}
    )
    assert (noul.type, choice.type, score.type) == ("noul", "choice", "score")


def test_unknown_question_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        question_adapter.validate_python({"type": "noul", "instruction": "typo"})


def test_instructions_are_required() -> None:
    with pytest.raises(ValidationError):
        question_adapter.validate_python({"type": "noul"})


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "score", "instructions": "How bad?", "criteria": []},
        {"type": "choice", "instructions": "Which?", "criteria": {}},
    ],
)
def test_empty_criteria_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        question_adapter.validate_python(payload)


def test_empty_questions_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneInput(state="hi", model="jev-latest", questions={})


def test_bare_label_list_becomes_a_mapping() -> None:
    choice = Choice.model_validate({"instructions": "Which?", "criteria": ["a", "b"]})
    assert choice.criteria == {"a": None, "b": None}


def test_noul_confidence_is_computed() -> None:
    assert NoulAnswer(noul=0.2).confidence == 0.8


def test_score_keys_are_coerced_to_integers() -> None:
    answer = ScoreAnswer.model_validate(
        {"score": 1.5, "legend": {"0": "low"}, "probabilities": {"0": 0.5, "1": 0.5}}
    )
    assert answer.legend == {0: "low"}
    assert answer.probabilities == {0: 0.5, 1: 0.5}


def test_confidence_is_filled_from_probabilities() -> None:
    answer = ChoiceAnswer.model_validate(
        {"choice": "a", "probabilities": {"a": 0.5, "b": 0.5}}
    )
    assert answer.confidence == 0.5
    certain = ChoiceAnswer.model_validate(
        {"choice": "a", "probabilities": {"a": 1.0, "b": 0.0}}
    )
    assert certain.confidence == 1.0


def test_score_confidence_is_filled_from_probabilities() -> None:
    answer = ScoreAnswer.model_validate(
        {"score": 1.0, "probabilities": {0: 0.0, 1: 1.0, 2: 0.0}}
    )
    assert answer.confidence == 1.0
    torn = ScoreAnswer.model_validate(
        {"score": 1.0, "probabilities": {0: 0.5, 1: 0.0, 2: 0.5}}
    )
    assert torn.confidence == 0.0


@pytest.mark.parametrize(
    "probabilities",
    [
        {"a": 2.1, "b": 1.9, "c": 1.8},
        {"a": -0.4, "b": -0.5},
        {"a": 0.3, "b": 0.3},
        {},
    ],
)
def test_unusable_probabilities_leave_confidence_unset(
    probabilities: dict[str, float],
) -> None:
    answer = ChoiceAnswer.model_validate(
        {"choice": "a", "probabilities": probabilities}
    )

    assert answer.confidence is None


def test_reported_confidence_is_never_overwritten() -> None:
    answer = ChoiceAnswer.model_validate(
        {"choice": "a", "confidence": 0.42, "probabilities": {"a": 0.9, "b": 0.1}}
    )

    assert answer.confidence == 0.42


def test_confidence_stays_none_without_probabilities() -> None:
    assert ChoiceAnswer(choice="a").confidence is None


def test_serializer_keeps_nested_none_and_drops_unset_optionals() -> None:
    dumped = Choice(instructions="Which?", criteria={"calm": None}).model_dump()
    assert dumped == {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"calm": None},
    }


def test_response_partitions_answers_by_type() -> None:
    response = SystemOneOutput.model_validate(
        {
            "model": "jev-latest",
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "answers": {
                "a": {"type": "noul", "noul": 0.9},
                "b": {"type": "choice", "choice": "x", "confidence": 0.5},
                "c": {"type": "score", "score": 1.0, "confidence": 0.5},
            },
        }
    )
    assert set(response.nouls) == {"a"}
    assert set(response.choices) == {"b"}
    assert set(response.scores) == {"c"}


def test_response_ignores_unknown_fields() -> None:
    response = SystemOneOutput.model_validate(
        {
            "model": "jev-latest",
            "usage": {"input_tokens": 1, "output_tokens": 0, "surprise": 1},
            "answers": {},
            "surprise": True,
        }
    )
    assert response.usage.input_tokens == 1


def test_request_serializes_to_the_wire_body() -> None:
    request = SystemOneInput.model_validate(
        {
            "state": {"subject": "outage"},
            "model": "jev-latest",
            "questions": {"a": {"type": "noul", "instructions": "Down?"}},
        }
    )
    assert request.model_dump() == {
        "state": {"subject": "outage"},
        "model": "jev-latest",
        "questions": {"a": {"type": "noul", "instructions": "Down?"}},
    }


def test_answer_dispatches_on_type() -> None:
    assert isinstance(
        answer_adapter.validate_python({"type": "score", "score": 2.0}), ScoreAnswer
    )
    assert isinstance(
        answer_adapter.validate_python({"type": "noul", "noul": 0.1}), NoulAnswer
    )


def test_score_criteria_rejects_a_bare_string() -> None:
    with pytest.raises(ValidationError):
        Score(instructions="How bad?", criteria="low")
