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

Releasing marks the lock object expired rather than deleting it, so the steady
state is a lease that exists and is free. Two different situations therefore
reach the same compare-and-swap: a normal handoff from a holder that released,
and a rescue from a holder that died still holding it. Only the second is worth a
warning, and they are told apart by the exact ``_EXPIRED`` stamp a release writes
(see ``_log_takeover``).
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


def _was_released(payload: object) -> bool:
    """True when the previous holder released cleanly rather than dying.

    ``release_lease`` stamps exactly ``_EXPIRED``, a value no live lease can hold,
    so it distinguishes a normal handoff from a TTL rescue. A corrupt payload
    counts as not-released: unreadable lock state is worth reporting.
    """
    return isinstance(payload, dict) and str(payload.get("expires_at")) == _EXPIRED


def _log_takeover(name: str, previous: object) -> None:
    """Log a lease takeover at a level matching what actually happened."""
    holder = previous.get("holder") if isinstance(previous, dict) else None
    if _was_released(previous):
        # The overwhelmingly common path: every tick after the first takes over a
        # lease its predecessor released on the way out. Logging that at WARNING
        # would bury the case below under ~24 false alarms a day.
        LOGGER.debug("Acquired released lease %s (previous holder %s)", name, holder)
        return
    LOGGER.warning(
        "Stole lease %s from holder %s: it expired without being released, so that "
        "run likely died mid-tick — check for a lost run.",
        name,
        holder,
    )


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
    # Expired, released, or corrupt: compare-and-swap so only one taker wins.
    if not replace_json_artifact_if_match(path, payload, version=version):
        return None
    _log_takeover(name, current)
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
