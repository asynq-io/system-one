"""A backend that answers at random. For wiring tests and demos, never for decisions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from system_one.schemas import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneOutput,
    Usage,
)

if TYPE_CHECKING:
    from system_one.schemas import SystemOneInput
    from system_one.settings import StubConfig

ROUNDING = 4


class _BaseStubBackend:
    model: str = "stub"

    def __init__(self, config: StubConfig) -> None:
        self.config = config
        self._random = random.Random(config.seed)  # noqa: S311  # nosec B311

    def _distribution(self, size: int) -> list[float]:
        weights = [self._random.random() for _ in range(size)]
        total = sum(weights)
        return [round(weight / total, ROUNDING) for weight in weights]

    def _answer(self, question: Question) -> Answer:
        match question:
            case Noul():
                return NoulAnswer(noul=round(self._random.random(), ROUNDING))
            case Choice():
                labels = list(question.criteria)
                probabilities = dict(
                    zip(labels, self._distribution(len(labels)), strict=True)
                )
                return ChoiceAnswer(
                    choice=max(probabilities, key=probabilities.__getitem__),
                    probabilities=probabilities,
                )
            case Score():
                by_level = dict(enumerate(self._distribution(len(question.criteria))))
                return ScoreAnswer(
                    score=round(
                        sum(level * value for level, value in by_level.items()),
                        ROUNDING,
                    ),
                    legend=dict(enumerate(question.criteria)),
                    probabilities=by_level,
                )

    def _ask(self, request: SystemOneInput) -> SystemOneOutput:
        return SystemOneOutput(
            model=request.model,
            usage=Usage(input_tokens=0, output_tokens=0),
            answers={
                name: self._answer(question)
                for name, question in request.questions.items()
            },
        )


class StubBackend(_BaseStubBackend):
    """Uniformly random answers of the right shape, reproducible with `seed`."""

    def ask(self, request: SystemOneInput) -> SystemOneOutput:
        return self._ask(request)

    def close(self) -> None:
        """Nothing to release."""


class AsyncStubBackend(_BaseStubBackend):
    """`StubBackend` behind the async protocol; nothing here blocks."""

    async def ask(self, request: SystemOneInput) -> SystemOneOutput:
        return self._ask(request)

    async def close(self) -> None:
        """Nothing to release."""
