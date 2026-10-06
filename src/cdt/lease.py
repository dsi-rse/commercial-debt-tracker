"""Advisory single-writer lease over artifact storage.

Serializes pipeline writers (overlapping poll ticks, daily runs) with a lease
object at ``{artifact_root}/locks/<name>.json``. It is acquired by conditional
create (S3 ``If-None-Match: *`` / local exclusive create) and, once expired,
taken over by conditional replace (S3 ``If-Match``), so two racing acquirers
never both win.

Failing to acquire is not an error: the caller skips its turn and the next
scheduled run picks the work up. Releasing stamps the object with the
``_EXPIRED`` sentinel rather than deleting it, which lets a takeover tell a
normal handoff from a rescue of a holder that died (see ``_log_takeover``).
The TTL only matters after such a crash.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from cdt.shared import get_logger
from cdt.storage.objects import (
    ArtifactPath,
    artifact_exists,
    join_artifact_path,
    read_json_artifact_versioned,
    replace_json_artifact_if_match,
    write_json_artifact_if_absent,
)

LOGGER = get_logger(__name__)

# Sized well above a normal tick (minutes); it only gates recovery after a crash.
DEFAULT_LEASE_TTL_SECONDS = 2 * 60 * 60
# The one lease every writer of extract job state and match/final snapshots
# holds: poll ticks, daily's match/finalize, and the admin reset command.
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
    """Return whether the previous holder released cleanly rather than dying.

    True only for the exact ``_EXPIRED`` stamp ``release_lease`` writes, a value
    no live lease can hold. A corrupt payload counts as not released.
    """
    return isinstance(payload, dict) and str(payload.get("expires_at")) == _EXPIRED


def _log_takeover(name: str, previous: object) -> None:
    """Log a lease takeover at a level matching what actually happened."""
    holder = previous.get("holder") if isinstance(previous, dict) else None
    if _was_released(previous):
        # The normal handoff on every tick; WARNING would bury the case below.
        LOGGER.debug("Acquired released lease %s (previous holder %s)", name, holder)
        return
    # "Stole lease" feeds a CloudWatch metric-filter alarm; keep the literal in
    # sync with pulumi/infra/alerts.py.
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
    """Acquire the named lease, taking it over only if expired or corrupt; None if held."""
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


def renew_lease(lease: Lease, *, ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS) -> bool:
    """Extend a held lease's expiry; False if it was stolen or storage raced.

    Call at phase boundaries of work that can outlast the TTL. False means the
    caller no longer holds the lease and must treat the tick as lost.
    """
    current, version = read_json_artifact_versioned(lease.path)
    if not isinstance(current, dict) or current.get("holder") != lease.holder:
        return False
    renewed = dict(cast(dict[str, object], current))
    renewed["expires_at"] = (
        datetime.now(UTC) + timedelta(seconds=ttl_seconds)
    ).isoformat()
    if not replace_json_artifact_if_match(lease.path, renewed, version=version):
        return False
    lease.expires_at = str(renewed["expires_at"])
    return True


class LeaseLostError(RuntimeError):
    """A phase-boundary renewal found the lease no longer held.

    The holder must stop writing immediately: another run owns the datasets and
    snapshots now, and continuing would interleave two writers.
    """


def renewer(lease: Lease) -> Callable[[], None]:
    """Return a renewal callback that raises LeaseLostError instead of a bool.

    For long phases that take a renewal hook across module boundaries, where
    a False return could be silently discarded.
    """

    def renew() -> None:
        if not renew_lease(lease):
            msg = (
                f"Lease {lease.path} is no longer held by {lease.holder}; "
                "another run owns the pipeline outputs now."
            )
            raise LeaseLostError(msg)

    return renew


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
