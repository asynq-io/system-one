import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [key for key in os.environ if key.startswith("SYSTEM_ONE_")]:
        monkeypatch.delenv(name, raising=False)
