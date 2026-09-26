"""Configuration read from `SYSTEM_ONE_*` environment variables or a `.env` file.

`Settings` picks the backend; a `*Config` carries everything that backend needs.
The hosted presets are `HTTPConfig` subclasses, so a vendor is a type, not a string.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from system_one.catalog import DEFAULT_NAME
from system_one.errors import SystemOneError


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYSTEM_ONE_", env_file=".env", extra="ignore"
    )


class Settings(_BaseSettings):
    """Which backend to build; everything else lives on its config."""

    backend: Literal["http", "onnx", "stub"] = "http"


class BackendConfig(_BaseSettings):
    """What every backend's config shares: the env source and the model to ask."""

    default_model: ClassVar[str | None] = None
    """What an unset `model` falls back to; `None` means it has to be named."""

    model: str | None = None

    @property
    def resolved_model(self) -> str:
        """`model`, else the class's `default_model`."""
        model = self.model or self.default_model
        if model is None:
            message = f"{type(self).__name__} needs a model: set SYSTEM_ONE_MODEL"
            raise SystemOneError(message)
        return model


class HTTPConfig(BackendConfig):
    """Any endpoint speaking the System One contract."""

    requires_api_key: ClassVar[bool] = False

    base_url: str
    path: str = "/v1/systemone"
    api_key: SecretStr | None = None
    timeout: float = 10.0
    max_retries: int = 2

    @model_validator(mode="after")
    def _check_api_key(self) -> HTTPConfig:
        if self.requires_api_key and self.api_key is None:
            message = "api_key is required: set SYSTEM_ONE_API_KEY"
            raise ValueError(message)
        return self


class TypesafeConfig(HTTPConfig):
    """Typesafe's hosted jev, preset. Needs `SYSTEM_ONE_API_KEY`."""

    requires_api_key = True
    default_model = "jev-latest"
    base_url: str = "https://api.typesafe.ai"


class OpenRouterConfig(HTTPConfig):
    """OpenRouter's decisions endpoint, preset. Needs `SYSTEM_ONE_API_KEY`."""

    requires_api_key = True
    default_model = "typesafe/jev-latest"
    base_url: str = "https://openrouter.ai"
    path: str = "/api/alpha/decisions"


class ONNXConfig(BackendConfig):
    """Where the local graph lives, and which one to load."""

    default_model = DEFAULT_NAME
    onnx_dir: Path = Path("onnx")


class StubConfig(BackendConfig):
    """Random answers of the right shape; `seed` makes them reproducible."""

    default_model = "stub"
    seed: int | None = None
