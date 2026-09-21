"""A vendor-neutral SDK for System One models: `ask(state, questions) -> answers`."""

from importlib.metadata import version

from system_one.agent import AsyncSystemOne, SystemOne
from system_one.errors import (
    RETRY_STATUSES,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    SystemOneError,
)
from system_one.schemas import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
    SystemOneInput,
    SystemOneOutput,
    Usage,
)
from system_one.settings import (
    HTTPConfig,
    ONNXConfig,
    OpenRouterConfig,
    Settings,
    TypesafeConfig,
)

__version__ = version("system-one")

__all__ = [
    "RETRY_STATUSES",
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "Answer",
    "AsyncSystemOne",
    "AuthenticationError",
    "Choice",
    "ChoiceAnswer",
    "HTTPConfig",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "ONNXConfig",
    "OpenRouterConfig",
    "Question",
    "Score",
    "ScoreAnswer",
    "Settings",
    "SystemOne",
    "SystemOneError",
    "SystemOneInput",
    "SystemOneOutput",
    "TypesafeConfig",
    "Usage",
    "__version__",
]
