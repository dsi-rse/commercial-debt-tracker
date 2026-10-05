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
def _isolated_s3_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep one test's AWS profile out of every later test in the session.

    ``configure_s3_profile`` writes process-global state and is now called
    unconditionally from ``cli.main`` and ``orchestrator.main``, both of which
    the suite invokes for real dozens of times. ``orchestrator``'s flag
    defaults to ``os.environ.get("AWS_PROFILE", "")``, so a developer or CI
    runner with AWS_PROFILE exported would otherwise leak a real profile —
    and real credentials — into the rest of the run.

    Here rather than in ``tests/test_storage.py``, where the reset used to
    live, because the leak crosses files by definition. The ``boto3.Session``
    stub stays local to that module: this only has to guarantee isolation, not
    fake AWS for everyone.
    """
    from cdt import storage

    monkeypatch.setattr(storage, "_S3_CLIENTS", {})
    monkeypatch.setattr(storage, "_BOTO3_SESSIONS", {})
    monkeypatch.setattr(storage, "_CONFIGURED_S3_PROFILE", "")


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
