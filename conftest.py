"""Configure pytest."""

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

    ``configure_s3_profile`` writes process-global state and is called
    unconditionally from ``cli.main``, which the suite invokes for real dozens
    of times; without this reset one test's profile reaches every later test.
    """
    from cdt.storage import objects as storage_objects

    monkeypatch.setattr(storage_objects, "_S3_CLIENTS", {})
    monkeypatch.setattr(storage_objects, "_BOTO3_SESSIONS", {})
    monkeypatch.setattr(storage_objects, "_CONFIGURED_S3_PROFILE", "")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default artifact root at a per-test directory.

    Commands that fall back to ``settings.DATA_DIR``, and take the
    pipeline-writer lease there, must never touch the developer's real data
    directory — or a live local pipeline's lock — from a test. The CLI's
    environment defaults (``cdt.cli.ENVIRONMENT_DEFAULTS``) are cleared for the
    same reason: ``ARTIFACT_ROOT`` comes before DATA_DIR, and a developer's
    exported ``GENRES`` or ``CDT_DEFAULT_CIK_FILE`` would change what a test runs.
    """
    from cdt.cli import ENVIRONMENT_DEFAULTS

    for variable in ENVIRONMENT_DEFAULTS.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "default-data-dir")
