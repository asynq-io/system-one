"""Probability arithmetic behind answer confidence."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

ROUNDING = 4
PROBABILITY_TOLERANCE = 0.01
EMPTY_DISTRIBUTION = "probabilities must not be empty"


def round_confidence(confidence: float) -> float:
    return round(confidence, ROUNDING)


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


def confidence_over(
    probabilities: Mapping[Any, float] | None,
    measure: Callable[[Iterable[float]], float],
) -> float | None:
    """`measure` over a probability mapping, or `None` when it is not one we trust."""
    if probabilities is None:
        return None
    try:
        return measure(probabilities.values())
    except ValueError:
        return None
