"""Configuration read from `SYSTEM_ONE_*` environment variables or a `.env` file.

`Settings` picks the backend; a `*Config` carries everything that backend needs.
The hosted presets are `HTTPConfig` subclasses, so a vendor is a type, not a string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from system_one.catalog import DEFAULT_NAME

ENV = SettingsConfigDict(env_prefix="SYSTEM_ONE_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Which backend to build, and the model that overrides the config's own."""

    model_config = ENV

    backend: Literal["http", "onnx"] = "http"
    model: str | None = None


class HTTPConfig(BaseSettings):
    """Any endpoint speaking the System One contract."""

    model_config = ENV

    base_url: str
    path: str = "/v1/systemone"
    model: str
    api_key: SecretStr | None = None
    timeout: float = 10.0
    max_retries: int = 2


class TypesafeConfig(HTTPConfig):
    """Typesafe's hosted jev, preset. Needs `SYSTEM_ONE_API_KEY`."""

    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"
    api_key: SecretStr


class OpenRouterConfig(HTTPConfig):
    """OpenRouter's decisions endpoint, preset. Needs `SYSTEM_ONE_API_KEY`."""

    base_url: str = "https://openrouter.ai"
    path: str = "/api/alpha/decisions"
    model: str = "typesafe/jev-latest"
    api_key: SecretStr


class ONNXConfig(BaseSettings):
    """Where the local graph lives, and which one to load."""

    model_config = ENV

    onnx_dir: Path = Path("onnx")
    model: str = DEFAULT_NAME


BackendConfig: TypeAlias = HTTPConfig | ONNXConfig
