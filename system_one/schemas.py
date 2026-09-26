"""The System One decision contract: questions in, answers out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import NotRequired, Self, TypedDict

from .utils import (
    choice_confidence,
    confidence_over,
    round_confidence,
    score_confidence,
)

JSONValue: TypeAlias = str | Mapping[str, Any] | Sequence[Any]
State: TypeAlias = JSONValue


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        from_attributes=True,
    )


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


class NoulCriteriaDict(TypedDict, total=False):
    """Dict form of `NoulCriteria`."""

    true: JSONValue | None
    false: JSONValue | None


class NoulDict(TypedDict):
    """Dict form of `Noul`."""

    type: Literal["noul"]
    instructions: str
    criteria: NotRequired[NoulCriteria | NoulCriteriaDict | None]


class ChoiceDict(TypedDict):
    """Dict form of `Choice`; bare labels stand for options without descriptions."""

    type: Literal["choice"]
    instructions: str
    criteria: Mapping[str, JSONValue | None] | Sequence[str]


class ScoreDict(TypedDict):
    """Dict form of `Score`."""

    type: Literal["score"]
    instructions: str
    criteria: Sequence[JSONValue]


Question: TypeAlias = Annotated[Noul | Choice | Score, Field(discriminator="type")]
QuestionDict: TypeAlias = NoulDict | ChoiceDict | ScoreDict
QuestionInput: TypeAlias = Mapping[str, Question | QuestionDict]


class ConfidentAnswer(BaseSchema):
    """An answer that reports how much to trust itself on one scale for every type.

    The hosted API always sends `confidence`; `_derive` fills it in for providers that
    do not. A payload it cannot trust leaves `confidence` unset, because an absent
    confidence is honest while a fabricated `1.0` is not.
    """

    confidence: float | None = None

    def _derive(self) -> float | None:
        """Confidence implied by the answer, or `None` if it cannot be derived."""
        raise NotImplementedError

    @model_validator(mode="after")
    def _fill_confidence(self) -> Self:
        if self.confidence is None:
            confidence = self._derive()
            if confidence is not None:
                self.confidence = round_confidence(confidence)
        return self


class NoulAnswer(ConfidentAnswer):
    """A yes/no answer, where `noul` is the probability the statement holds."""

    type: Literal["noul"] = "noul"
    noul: float

    def _derive(self) -> float | None:
        return max(self.noul, 1.0 - self.noul)


class ChoiceAnswer(ConfidentAnswer):
    """The selected label and, when reported, the distribution it came from."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] | None = None

    def _derive(self) -> float | None:
        return confidence_over(self.probabilities, choice_confidence)


class ScoreAnswer(ConfidentAnswer):
    """A probability-weighted expected score, not an argmax."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[int, JSONValue] | None = None
    probabilities: dict[int, float] | None = None

    def _derive(self) -> float | None:
        return confidence_over(self.probabilities, score_confidence)


Answer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")
]
AnswerT = TypeVar("AnswerT", NoulAnswer, ChoiceAnswer, ScoreAnswer)


class Usage(BaseSchema):
    """Token counts, and cost where the provider reports one."""

    input_tokens: int
    output_tokens: int
    cost: float | None = None


class SystemOneInput(BaseSchema):
    """The single normalized input shared by every backend."""

    state: State
    model: str
    questions: Mapping[str, Question] = Field(min_length=1)


class SystemOneOutput(BaseSchema):
    """Answers keyed by question name, with model and usage metadata."""

    model: str
    usage: Usage
    answers: dict[str, Answer] = Field(default_factory=dict)
    id: str | None = None

    def _of_type(self, kind: type[AnswerT]) -> dict[str, AnswerT]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, kind)
        }

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return self._of_type(NoulAnswer)

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return self._of_type(ChoiceAnswer)

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return self._of_type(ScoreAnswer)
