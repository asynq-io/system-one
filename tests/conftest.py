import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in [key for key in os.environ if key.startswith("SYSTEM_ONE_")]:
        monkeypatch.delenv(name, raising=False)
    # Settings reads a relative `.env`, so run where the repo's own one is invisible.
    monkeypatch.chdir(tmp_path)
