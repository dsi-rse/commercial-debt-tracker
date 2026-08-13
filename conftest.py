"""Configure pytest."""

import logging
from collections.abc import Callable

import pytest


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    """Disable warnings-as-workflow errors; our upstream dependencies raise warnings."""
    if exitstatus == 5:  # noqa: PLR2004
        session.exitstatus = 0


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
