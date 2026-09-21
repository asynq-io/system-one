"""The System One decision contract: questions in, answers out."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

JSONValue: TypeAlias = str | Mapping[str, Any] | Sequence[Any]
State: TypeAlias = JSONValue


ROUNDING = 4
PROBABILITY_TOLERANCE = 0.01
EMPTY_DISTRIBUTION = "probabilities must not be empty"


def distribution(probabilities: Iterable[float]) -> list[float]:
    """Read a probability vector, rejecting anything that is not one.

    Confidence derived from logits, top-k remnants or any other unnormalized vector
    is meaningless, so callers get a `ValueError` instead of a plausible number.
    """
    values = [float(probability) for probability in probabilities]
    if not values:
        raise ValueError(EMPTY_DISTRIBUTION)
    if any(not 0.0 <= value <= 1.0 for value in values):
        message = f"probabilities must lie in [0, 1], got {values}"
        raise ValueError(message)
    total = math.fsum(values)
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        message = f"probabilities must sum to 1, got {total}"
        raise ValueError(message)
    return values


def choice_confidence(probabilities: Iterable[float]) -> float:
    """Confidence in the selected choice: the probability of the reported label."""
    return max(distribution(probabilities))


def score_confidence(probabilities: Iterable[float]) -> float:
    """Confidence in the expected score: `1 - 2 * sd / (k - 1)`.

    The score is an expectation over ordered levels, so its reliability is how
    tightly the mass sits around it. Entropy cannot see order and would rate a
    distribution split between the end levels — whose expectation lands in a
    valley no level claims — the same as one split between neighbours.
    """
    values = distribution(probabilities)
    if len(values) == 1:
        return 1.0
    expected = math.fsum(level * value for level, value in enumerate(values))
    variance = math.fsum(
        value * (level - expected) ** 2 for level, value in enumerate(values)
    )
    return max(0.0, 1.0 - 2.0 * math.sqrt(variance) / (len(values) - 1))


class BaseSchema(BaseModel):
    """Frozen, strict-input model whose unset optional fields stay off the wire.

    The wrap serializer drops only top-level ``None`` field values, so a user-supplied
    ``criteria={"calm": None}`` survives while an unset ``instructions`` does not;
    ``exclude_none`` would drop both and ``exclude_unset`` would drop the ``type``
    discriminator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_serializer(mode="wrap")
    def _omit_none(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


class BaseOutput(BaseSchema):
    """Response-side model that tolerates fields a newer server adds."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class NoulCriteria(BaseSchema):
    """Optional descriptions of the two `noul` outcomes."""

    true: JSONValue | None = None
    false: JSONValue | None = None


class Noul(BaseSchema):
    """A yes/no question."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: NoulCriteria | None = None


class Choice(BaseSchema):
    """A pick-one question over labelled options."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Mapping[str, JSONValue | None] = Field(min_length=1)

    @field_validator("criteria", mode="before")
    @classmethod
    def _labels_without_descriptions(cls, value: Any) -> Any:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return dict.fromkeys(value)
        return value


class Score(BaseSchema):
    """An ordinal question whose criteria index is the score."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: Sequence[JSONValue] = Field(min_length=1)


Question: TypeAlias = Annotated[Noul | Choice | Score, Field(discriminator="type")]
QuestionInput: TypeAlias = Mapping[str, Question | Mapping[str, Any]]


class NoulAnswer(BaseOutput):
    """A yes/no answer, where `noul` is the probability the statement holds."""

    type: Literal["noul"] = "noul"
    noul: float

    @property
    def confidence(self) -> float:
        """Probability of the reported outcome — `choice_confidence` over two levels.

        The hosted API does not send this; it is derived so that every answer type
        reports confidence on the same scale.
        """
        return round(max(self.noul, 1.0 - self.noul), ROUNDING)


class ConfidentAnswer(BaseOutput):
    """An answer whose confidence is derived from its probabilities when omitted.

    The hosted API always reports `confidence`; this fills it in for providers that
    do not. Probabilities it cannot trust leave `confidence` unset, because an
    absent confidence is honest while a fabricated `1.0` is not.
    """

    confidence: float | None = None

    _confidence_of: ClassVar[staticmethod[[Iterable[float]], float]]

    @model_validator(mode="before")
    @classmethod
    def _confidence_from_probabilities(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or data.get("confidence") is not None:
            return data
        probabilities = data.get("probabilities")
        if not isinstance(probabilities, Mapping):
            return data
        try:
            confidence = cls._confidence_of(probabilities.values())
        except (ValueError, TypeError):
            return data
        return {**data, "confidence": round(confidence, ROUNDING)}


class ChoiceAnswer(ConfidentAnswer):
    """The selected label and, when reported, the distribution it came from."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] | None = None

    _confidence_of = staticmethod(choice_confidence)


class ScoreAnswer(ConfidentAnswer):
    """A probability-weighted expected score, not an argmax."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[int, JSONValue] | None = None
    probabilities: dict[int, float] | None = None

    _confidence_of = staticmethod(score_confidence)


Answer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")
]


class Usage(BaseOutput):
    """Token counts, and cost where the provider reports one."""

    input_tokens: int
    output_tokens: int
    cost: float | None = None


class ModelInfo(BaseOutput):
    """A model available to the account."""

    name: str
    description: str
    release_date: str


class SystemOneInput(BaseSchema):
    """The single normalized input shared by every backend."""

    state: State
    model: str
    questions: Mapping[str, Question] = Field(min_length=1)


class SystemOneOutput(BaseOutput):
    """Answers keyed by question name, with model and usage metadata."""

    model: str
    usage: Usage
    answers: dict[str, Answer] = Field(default_factory=dict)
    id: str | None = None

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, NoulAnswer)
        }

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }
