"""Advisory single-writer lease over artifact storage.

EventBridge fires the poll schedule hourly whether or not the previous tick
finished, and both the ``daily`` and ``poll`` orchestrator modes rewrite the
match/final snapshots. This module serializes those writers with a small lease
object under ``{artifact_root}/locks/<name>.json``, acquired with a conditional
create (S3 ``If-None-Match: *`` / local exclusive create) and stolen after its
TTL expires with a conditional replace (S3 ``If-Match`` compare-and-swap), so
two racing acquirers can never both win.

Losing the lease is not an error: callers skip their turn and the next
scheduled run picks the work up. The TTL only matters after a crash — releases
happen in a ``finally`` — so it is sized generously above a normal tick.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from cdt.shared import get_logger
from cdt.storage import (
    ArtifactPath,
    artifact_exists,
    join_artifact_path,
    read_json_artifact_versioned,
    replace_json_artifact_if_match,
    write_json_artifact_if_absent,
)

LOGGER = get_logger(__name__)

# A poll tick is normally minutes; the TTL only gates recovery after a crash.
DEFAULT_LEASE_TTL_SECONDS = 2 * 60 * 60
# One lease serializes every writer of extract job state and match/final
# snapshots: poll ticks (hourly schedule + EventBridge retries), daily's
# match/finalize, and the admin reset command.
PIPELINE_WRITER_LEASE = "pipeline-writer"
_EXPIRED = "1970-01-01T00:00:00+00:00"


@dataclass
class Lease:
    """A held lease; pass back to ``release_lease`` when done."""

    path: str
    holder: str
    expires_at: str


def lease_path(artifact_root: ArtifactPath, name: str) -> str:
    """Return the lock-object path for one lease name."""
    return join_artifact_path(str(artifact_root), "locks", f"{name}.json")


def _payload(holder: str, now: datetime, ttl_seconds: int) -> dict[str, object]:
    return {
        "holder": holder,
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }


def _current_expiry(payload: object) -> datetime | None:
    """Parse a lease payload's expiry; None means corrupt (treat as expired)."""
    if not isinstance(payload, dict):
        return None
    try:
        return datetime.fromisoformat(str(payload["expires_at"]))
    except (KeyError, ValueError):
        return None


def acquire_lease(
    artifact_root: ArtifactPath,
    name: str,
    *,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> Lease | None:
    """Acquire the named lease, stealing it only if expired; None if held."""
    path = lease_path(artifact_root, name)
    holder = uuid.uuid4().hex
    now = datetime.now(UTC)
    payload = _payload(holder, now, ttl_seconds)
    if write_json_artifact_if_absent(path, payload):
        return Lease(path=path, holder=holder, expires_at=str(payload["expires_at"]))

    current, version = read_json_artifact_versioned(path)
    expiry = _current_expiry(current)
    if expiry is not None and expiry > now:
        return None
    # Expired (or corrupt): compare-and-swap so only one stealer wins.
    if not replace_json_artifact_if_match(path, payload, version=version):
        return None
    previous = current.get("holder") if isinstance(current, dict) else None
    LOGGER.warning("Stole expired lease %s from holder %s", name, previous)
    return Lease(path=path, holder=holder, expires_at=str(payload["expires_at"]))


def release_lease(lease: Lease) -> None:
    """Mark a held lease expired so the next acquirer takes over immediately.

    Best-effort: if the lease was stolen (holder changed) or storage read/write
    races, the release is a no-op and the TTL governs instead.
    """
    if not artifact_exists(lease.path):
        return
    current, version = read_json_artifact_versioned(lease.path)
    if not isinstance(current, dict) or current.get("holder") != lease.holder:
        return
    released = dict(cast(dict[str, object], current))
    released["expires_at"] = _EXPIRED
    replace_json_artifact_if_match(lease.path, released, version=version)
