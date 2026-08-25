"""Tests for the advisory pipeline-writer lease."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import cdt.lease as lease_module
from cdt.lease import acquire_lease, lease_path, release_lease, renew_lease


def test_acquire_blocks_second_acquirer(tmp_path: Path) -> None:
    """While a live lease is held, a second acquire returns None."""
    first = acquire_lease(tmp_path, "writer")
    assert first is not None

    assert acquire_lease(tmp_path, "writer") is None


def test_release_frees_lease_immediately(tmp_path: Path) -> None:
    """After release, the next acquire succeeds without waiting for the TTL."""
    first = acquire_lease(tmp_path, "writer")
    assert first is not None
    release_lease(first)

    second = acquire_lease(tmp_path, "writer")
    assert second is not None
    assert second.holder != first.holder


def test_expired_lease_is_stolen(tmp_path: Path) -> None:
    """A lease past its TTL is taken over by the next acquirer."""
    stale = acquire_lease(tmp_path, "writer", ttl_seconds=0)
    assert stale is not None

    stolen = acquire_lease(tmp_path, "writer")
    assert stolen is not None
    assert stolen.holder != stale.holder


def test_corrupt_lease_is_stolen(tmp_path: Path) -> None:
    """A lock file without a parseable expiry is treated as expired."""
    path = Path(lease_path(tmp_path, "writer"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"holder": "old", "expires_at": "not a date"}))

    assert acquire_lease(tmp_path, "writer") is not None


def test_release_of_stolen_lease_is_noop(tmp_path: Path) -> None:
    """Releasing a lease another holder stole must not free their lease."""
    stale = acquire_lease(tmp_path, "writer", ttl_seconds=0)
    assert stale is not None
    stolen = acquire_lease(tmp_path, "writer")
    assert stolen is not None

    release_lease(stale)

    payload = json.loads(Path(stolen.path).read_text())
    assert payload["holder"] == stolen.holder
    assert datetime.fromisoformat(payload["expires_at"]) > datetime.now(UTC)


def test_release_tolerates_missing_lock_file(tmp_path: Path) -> None:
    """Releasing after the lock object disappeared is a no-op, not an error."""
    lease = acquire_lease(tmp_path, "writer")
    assert lease is not None
    Path(lease.path).unlink()

    release_lease(lease)


def test_ttl_written_into_payload(tmp_path: Path) -> None:
    """The persisted expiry reflects the requested TTL."""
    before = datetime.now(UTC)
    lease = acquire_lease(tmp_path, "writer", ttl_seconds=3600)
    assert lease is not None

    payload = json.loads(Path(lease.path).read_text())
    expiry = datetime.fromisoformat(payload["expires_at"])
    assert before + timedelta(minutes=59) < expiry < before + timedelta(minutes=61)


def test_normal_handoff_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """Consecutive released-then-reacquired ticks must stay quiet.

    Regression test for #32: releasing marks the lock expired rather than
    deleting it, so every tick after the first takes the compare-and-swap path.
    Warning there produced ~24 false alarms a day on the hourly poller and made a
    real crash-recovery steal indistinguishable from routine operation.
    """
    propagate_logger(lease_module.LOGGER)

    with caplog.at_level(logging.DEBUG, logger=lease_module.LOGGER.name):
        for _ in range(3):
            lease = acquire_lease(tmp_path, "writer")
            assert lease is not None
            release_lease(lease)

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert "Acquired released lease" in caplog.text


def test_crashed_holder_still_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """Taking over from a holder that never released is still a WARNING."""
    propagate_logger(lease_module.LOGGER)
    crashed = acquire_lease(tmp_path, "writer", ttl_seconds=0)
    assert crashed is not None  # never released: simulates a died-mid-tick run

    with caplog.at_level(logging.DEBUG, logger=lease_module.LOGGER.name):
        stolen = acquire_lease(tmp_path, "writer")

    assert stolen is not None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "expired without being released" in warnings[0].getMessage()
    assert crashed.holder in warnings[0].getMessage()


def test_corrupt_lease_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """Unreadable lock state is reported, not silently taken over."""
    propagate_logger(lease_module.LOGGER)
    path = Path(lease_path(tmp_path, "writer"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"holder": "old", "expires_at": "not a date"}))

    with caplog.at_level(logging.DEBUG, logger=lease_module.LOGGER.name):
        assert acquire_lease(tmp_path, "writer") is not None

    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_corrupt_lock_file_is_stealable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """A truncated lock file reads as corrupt-and-stealable, not a crash."""
    propagate_logger(lease_module.LOGGER)
    path = Path(lease_path(tmp_path, "writer"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"holder": "old", "expires_at": "2099')  # killed mid-write

    with caplog.at_level(logging.DEBUG, logger=lease_module.LOGGER.name):
        lease = acquire_lease(tmp_path, "writer")

    assert lease is not None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_renew_lease_extends_expiry(tmp_path: Path) -> None:
    """Renewal pushes the expiry forward while the holder still owns the lease."""
    lease = acquire_lease(tmp_path, "writer", ttl_seconds=60)
    assert lease is not None
    before = lease.expires_at

    assert renew_lease(lease, ttl_seconds=7200) is True
    assert lease.expires_at > before

    # A second acquirer still cannot take the (now longer-lived) lease.
    assert acquire_lease(tmp_path, "writer") is None


def test_renew_lease_fails_after_steal(tmp_path: Path) -> None:
    """A holder whose lease was stolen learns it from the renewal result."""
    original = acquire_lease(tmp_path, "writer", ttl_seconds=0)
    assert original is not None
    thief = acquire_lease(tmp_path, "writer")
    assert thief is not None

    assert renew_lease(original) is False
    assert renew_lease(thief) is True
