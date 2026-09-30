"""Probability arithmetic and criterion rendering."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

ROUNDING = 4
PROBABILITY_TOLERANCE = 0.02
EMPTY_DISTRIBUTION = "probabilities must not be empty"


def round_confidence(confidence: float) -> float:
    return round(confidence, ROUNDING)


def distribution(probabilities: Iterable[float]) -> list[float]:
    """Read a probability vector, rejecting anything that is not one.

    Confidence derived from logits, top-k remnants or any other unnormalized vector
    is meaningless, so callers get a `ValueError` instead of a plausible number.
    Vectors within tolerance are renormalized, absorbing wire rounding.
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
    return [value / total for value in values]


def choice_confidence(probabilities: Iterable[float]) -> float:
    """Confidence in the selected choice: `(pmax - 1/k) / (1 - 1/k)`.

    0 at uniform, 1 at certainty, whatever the number of options.
    """
    values = distribution(probabilities)
    if len(values) == 1:
        return 1.0
    floor = 1.0 / len(values)
    return (max(values) - floor) / (1.0 - floor)


def score_confidence(probabilities: Iterable[float]) -> float:
    """Confidence in the score: `1 - E|i - mode| / D`, with `D` that of a uniform.

    The score is an expectation over ordered levels, so its reliability is how
    tightly the mass sits around the most likely level. Entropy cannot see order
    and would rate a distribution split between the end levels — whose expectation
    lands in a valley no level claims — the same as one split between neighbours.
    """
    values = distribution(probabilities)
    levels = len(values)
    if levels == 1:
        return 1.0
    mode = max(range(levels), key=values.__getitem__)
    centre = (levels - 1) / 2
    uniform_spread = math.fsum(abs(level - centre) for level in range(levels)) / levels
    spread = math.fsum(value * abs(level - mode) for level, value in enumerate(values))
    return max(0.0, 1.0 - spread / uniform_spread)


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


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
