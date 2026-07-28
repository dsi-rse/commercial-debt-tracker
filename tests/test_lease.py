"""Tests for the advisory pipeline-writer lease."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cdt.lease import acquire_lease, lease_path, release_lease


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
