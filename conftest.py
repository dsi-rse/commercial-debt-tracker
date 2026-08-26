"""Configure pytest."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from cdt import settings


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    """Disable warnings-as-workflow errors; our upstream dependencies raise warnings."""
    if exitstatus == 5:  # noqa: PLR2004
        session.exitstatus = 0


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default artifact root at a per-test directory.

    Commands that fall back to ``settings.DATA_DIR`` (and now take the
    pipeline-writer lease there, #88) must never touch the developer's real
    data directory — or a live local pipeline's lock — from a test.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "default-data-dir")


@pytest.fixture
def propagate_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[logging.Logger], None]:
    """Let ``caplog`` observe a cdt module logger.

    The shared ``get_logger`` sets ``propagate = False`` and attaches its own
    handler, so records never reach the root handler ``caplog`` installs. Call
    this with a module's ``LOGGER`` to restore propagation for one test.
    """

    def _propagate(logger: logging.Logger) -> None:
        monkeypatch.setattr(logger, "propagate", True)

    return _propagate
