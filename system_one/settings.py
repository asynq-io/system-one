"""Configuration read from `SYSTEM_ONE_*` environment variables or a `.env` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything both backends need. Vendor choice is `base_url` plus `path`."""

    model_config = SettingsConfigDict(
        env_prefix="SYSTEM_ONE_", env_file=".env", extra="ignore"
    )

    backend: Literal["http", "onnx"] = "http"
    # None means "whatever the chosen backend serves": `jev-latest` hosted, `laya` local.
    model: str | None = None

    api_key: SecretStr | None = None
    base_url: str = "https://api.typesafe.ai"
    path: str = "/v1/systemone"
    timeout: float = 10.0
    max_retries: int = 2

    onnx_dir: Path = Path("onnx")
