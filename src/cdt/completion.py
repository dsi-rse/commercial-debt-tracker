"""The per-stage completion registry: what each stage has processed, at which source version."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Self

from cdt.datasets import (
    PARTITION_PATTERN,
    dataset_root,
    match_date_shard_partition,
    resolve_artifact_root,
)
from cdt.shared import get_logger
from cdt.storage.objects import (
    ArtifactPath,
    artifact_exists,
    join_artifact_path,
    list_artifacts,
    list_artifacts_with_versions,
    normalize_artifact_path,
    read_json_artifact,
    read_json_artifact_versioned,
    replace_json_artifact_if_match,
    write_json_artifact_if_absent,
)
from cdt.storage.tables import is_orphaned_temp_artifact

LOGGER = get_logger(__name__)


def completion_registry_root(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the prefix ``runs/<stage>/completed`` holding a registry's shards.

    The shard objects under it are ``completion_registry_shard_path``.
    """
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "runs",
        stage_name,
        "completed",
    )


# Registry shards are keyed by the source partition's year-month; why:
# docs/decisions/storage-and-completion.md.
_REGISTRY_SHARD_DATE_CHARS = len("YYYY-MM")
# A key with no parseable partition date gets its own shard rather than being
# dropped, so its completion state still round-trips.
_UNDATED_REGISTRY_SHARD = "unknown"


def _registry_shard_label(key: str) -> str:
    """Return the registry shard one entry's key belongs in."""
    partition = match_date_shard_partition(key)
    if partition is None:
        return _UNDATED_REGISTRY_SHARD
    return partition["date"][:_REGISTRY_SHARD_DATE_CHARS]


def completion_registry_shard_path(
    stage_name: str,
    shard_label: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the path of one date-prefix shard of a stage's registry."""
    return join_artifact_path(
        completion_registry_root(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ),
        f"date={shard_label}.json",
    )


@dataclass
class CompletedPartition:
    """One source partition's completion record.

    ``fingerprint`` is the source object's version at processing time (S3 ETag;
    size+mtime locally); None when unknown. ``item_ids`` (extract only) are the
    content-terminal rows (SUCCESS or FAILED-on-validation), which
    re-processing the partition skips.
    """

    fingerprint: str | None = None
    item_ids: frozenset[str] = frozenset()
    # False when some relevant rows are not yet content-terminal: the partition
    # stays pending and only rows missing from item_ids are processed next time.
    complete: bool = True


class CompletionRegistry(dict[str, CompletedPartition]):
    """A loaded completion registry that remembers which keys the run changed.

    ``dirty`` holds the keys assigned since load; ``save_completion_registry``
    writes only those, overlaid on the freshest persisted state, so concurrent
    writers (daily, poll and manual runs) do not revert each other. Change
    entries by assignment (``registry[key] = entry``), never by mutating a
    loaded ``CompletedPartition`` in place.
    """

    def __init__(self: Self, *args: object, **kwargs: object) -> None:
        """Initialize from loaded entries, which start out clean."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.dirty: set[str] = set()

    def __setitem__(self: Self, key: str, value: CompletedPartition) -> None:
        """Record the entry and remember it as changed by this run."""
        super().__setitem__(key, value)
        self.dirty.add(key)

    def setdefault(
        self: Self, key: str, default: CompletedPartition | None = None
    ) -> CompletedPartition:
        """Insert-if-absent through __setitem__ so insertions count as changes."""
        if key not in self:
            self[key] = default if default is not None else CompletedPartition()
        return self[key]


def load_completed_partitions(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> set[str]:
    """Load completed source partition paths for one stage, ignoring fingerprints."""
    return {
        path
        for path, entry in load_completion_registry(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ).items()
        if entry.complete
    }


def load_completion_registry(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> CompletionRegistry:
    """Load one stage's registry from every ``date=`` shard, with no keys dirty.

    Only ``date=*.json`` objects are read. Shards are read concurrently and merged in sorted path order.
    Keys come back as whole paths under the artifact root. No shards returns
    an empty registry.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    entries: dict[str, CompletedPartition] = {}
    shard_paths = [
        path
        for path in list_artifacts(
            completion_registry_root(
                stage_name, artifact_root=resolved_root, data_dir=data_dir
            ),
            suffix=".json",
        )
        if path.rsplit("/", 1)[-1].startswith("date=")
    ]
    for payload in _read_registry_shards(shard_paths):
        entries.update(_registry_entries(payload, resolved_root))
    return CompletionRegistry(entries)


# Capped at botocore's default ``max_pool_connections`` (10): the GETs share one
# client, and extra threads only queue and log "connection pool is full". Raise
# both together, with ``S3_CLIENT_CONFIG`` in storage/objects.py.
_REGISTRY_LOAD_CONCURRENCY = 10


def _read_registry_shards(shard_paths: list[str]) -> list[object]:
    """Read registry shard objects concurrently, in ``shard_paths`` order.

    Returns a list, not an iterator, so a failed read raises before the caller
    has merged anything.
    """
    if not shard_paths:
        return []
    if len(shard_paths) == 1:
        # The common local/test shape: skip the pool.
        return [read_json_artifact(shard_paths[0])]
    with ThreadPoolExecutor(
        max_workers=min(_REGISTRY_LOAD_CONCURRENCY, len(shard_paths))
    ) as pool:
        return list(pool.map(read_json_artifact, shard_paths))


# Payload version. Keys are stored root-relative when the remainder is a bare
# canonical partition path, and whole otherwise. Shape tells them apart safely:
# a whole path (``/srv/cdt/documents/...``, ``s3://bucket/documents/...``) has a
# slash in its dataset segment, which ``[a-z\-]+`` cannot ``fullmatch``.
_REGISTRY_VERSION = 3


# Key prefixes are derived once per object, not per key (the per-key cost is in
# docs/decisions/storage-and-completion.md). The probe is a bare canonical
# partition path, the only shape a prefix is applied to, so slicing it off a
# join yields exactly what ``join_artifact_path`` prepends on any root,
# including ``.`` and ``""``, where ``pathlib`` collapses the join.
_REGISTRY_PREFIX_PROBE = "documents/date=2000-01-01/shard=0000/part-0000.parquet"


def _registry_join_prefix(artifact_root: str) -> str:
    """Return the string that re-homes a relativized key under one root."""
    joined = join_artifact_path(artifact_root, _REGISTRY_PREFIX_PROBE)
    return joined[: -len(_REGISTRY_PREFIX_PROBE)]


def _registry_strip_prefix(artifact_root: str) -> str:
    """Return the prefix a key must carry for the root to be strippable."""
    return f"{normalize_artifact_path(artifact_root).rstrip('/')}/"


def _strip_registry_root(key: str, strip_prefix: str) -> str:
    """Strip a precomputed root prefix off one registry key."""
    if not key.startswith(strip_prefix):
        return key
    relative = key[len(strip_prefix) :]
    return relative if PARTITION_PATTERN.fullmatch(relative) else key


def _prepend_registry_root(stored: str, join_prefix: str) -> str:
    """Reattach a precomputed root prefix to one persisted registry key."""
    if not PARTITION_PATTERN.fullmatch(stored):
        return stored
    return join_prefix + stored


def _relative_registry_key(key: str, artifact_root: str) -> str:
    """Return one registry key with the artifact root stripped off.

    Stripped only when the remainder is a bare canonical partition path, so
    this is the exact inverse of ``_absolute_registry_key``; any other key is
    returned whole. The one-key form of ``_strip_registry_root``.
    """
    return _strip_registry_root(key, _registry_strip_prefix(artifact_root))


def _absolute_registry_key(stored: str, artifact_root: str) -> str:
    """Return one persisted key with the artifact root reattached.

    A stored key is root-relative exactly when it is a bare canonical
    date/shard partition path -- ``fullmatch``, not the suffix ``search`` the
    parsers use, so a whole path or an S3 URI is not mistaken for one. Anything
    else is returned as stored.
    """
    return _prepend_registry_root(stored, _registry_join_prefix(artifact_root))


def _registry_entries(
    payload: object, artifact_root: str
) -> dict[str, CompletedPartition]:
    """Parse one persisted registry object into whole-path-keyed entries."""
    join_prefix = _registry_join_prefix(artifact_root)
    return {
        _prepend_registry_root(key, join_prefix): entry
        for key, entry in _parse_registry_payload(payload).items()
    }


def _parse_registry_payload(payload: object) -> dict[str, CompletedPartition]:
    """Parse one persisted registry shard payload; junk reads as empty."""
    if not isinstance(payload, dict):
        return {}
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        return {}
    registry: dict[str, CompletedPartition] = {}
    for key, entry in partitions.items():
        if not str(key).strip() or not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        item_ids = entry.get("item_ids", [])
        registry[str(key)] = CompletedPartition(
            fingerprint=str(fingerprint) if fingerprint else None,
            item_ids=frozenset(
                str(item) for item in item_ids if isinstance(item_ids, list)
            ),
            complete=bool(entry.get("complete", True)),
        )
    return registry


def _registry_payload(
    stage_name: str,
    shard_label: str,
    registry: dict[str, CompletedPartition],
    *,
    artifact_root: str,
) -> dict[str, object]:
    """Build the persisted v3 payload for one date shard of a registry.

    Sorted on the relativized key alone: ``CompletedPartition`` is unordered,
    so a tuple sort would raise TypeError if two keys relativized alike.
    """
    strip_prefix = _registry_strip_prefix(artifact_root)
    return {
        "stage": stage_name,
        "version": _REGISTRY_VERSION,
        "date_prefix": shard_label,
        "partitions": {
            path: {
                "fingerprint": entry.fingerprint,
                **({"item_ids": sorted(entry.item_ids)} if entry.item_ids else {}),
                **({} if entry.complete else {"complete": False}),
            }
            for path, entry in sorted(
                (
                    (_strip_registry_root(key, strip_prefix), entry)
                    for key, entry in registry.items()
                ),
                key=itemgetter(0),
            )
        },
    }


# Losing this many consecutive compare-and-swap races means writers are churning
# the registry far faster than any supported schedule; give up loudly.
_REGISTRY_CAS_ATTEMPTS = 8


def save_completion_registry(
    stage_name: str,
    registry: dict[str, CompletedPartition],
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Persist one stage's changed registry entries; return the registry root.

    Writes only the changed entries (``CompletionRegistry.dirty``, or every key
    of a plain dict), grouped by date shard. Each shard is compare-and-swapped
    on its own: the entries are overlaid on that shard's freshest persisted
    state, so concurrent writers keep each other's entries, and a lost race
    re-reads and retries. Shards commit independently, and a committed shard's
    keys leave the dirty set, so an interrupted save keeps the shards already
    written. Raises RuntimeError when a shard loses ``_REGISTRY_CAS_ATTEMPTS``
    races in a row.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    changed = sorted(
        registry.dirty if isinstance(registry, CompletionRegistry) else registry
    )
    by_shard: dict[str, dict[str, CompletedPartition]] = {}
    for key in changed:
        by_shard.setdefault(_registry_shard_label(key), {})[key] = registry[key]
    for shard_label, shard_entries in sorted(by_shard.items()):
        _save_registry_shard(
            stage_name,
            shard_label,
            shard_entries,
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        # Committed, so the next batch boundary must not re-send these keys;
        # otherwise every save rewrites every shard the run has touched and the
        # cost grows quadratically with run length. Cleared per shard because a
        # failed swap raises, leaving only that shard's keys dirty.
        if isinstance(registry, CompletionRegistry):
            registry.dirty -= shard_entries.keys()
    return completion_registry_root(
        stage_name, artifact_root=resolved_root, data_dir=data_dir
    )


def _save_registry_shard(
    stage_name: str,
    shard_label: str,
    entries: dict[str, CompletedPartition],
    *,
    artifact_root: str,
    data_dir: Path | None,
) -> str:
    """Compare-and-swap ``entries`` into one date shard of a stage's registry.

    The merge runs on whole-path keys and the payload strips the root on
    write, so each key has one spelling in the stored shard. Returns the shard
    path; raises RuntimeError after ``_REGISTRY_CAS_ATTEMPTS`` lost races.
    """
    path = completion_registry_shard_path(
        stage_name, shard_label, artifact_root=artifact_root, data_dir=data_dir
    )
    for _ in range(_REGISTRY_CAS_ATTEMPTS):
        if not artifact_exists(path):
            if write_json_artifact_if_absent(
                path,
                _registry_payload(
                    stage_name,
                    shard_label,
                    entries,
                    artifact_root=artifact_root,
                ),
            ):
                return path
            continue
        try:
            payload, version = read_json_artifact_versioned(path)
        except FileNotFoundError:
            continue
        merged = _registry_entries(payload, artifact_root)
        merged.update(entries)
        if replace_json_artifact_if_match(
            path,
            _registry_payload(
                stage_name, shard_label, merged, artifact_root=artifact_root
            ),
            version=version,
        ):
            return path
    msg = (
        f"Lost {_REGISTRY_CAS_ATTEMPTS} compare-and-swap races persisting the "
        f"{stage_name!r} completion registry shard at {path}"
    )
    raise RuntimeError(msg)


#: How often a long stage saves its completion registry, on top of one save at
#: the end: what an interruption can lose at most. A save is a few S3 requests;
#: at this interval its cost is negligible next to the work it protects.
CHECKPOINT_INTERVAL_SECONDS = 5 * 60


def pending_source_partitions(
    stage_name: str,
    source_dataset: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    force: bool = False,
) -> tuple[list[tuple[str, str]], dict[str, CompletedPartition]]:
    """Select source partitions a whole-partition stage still has to process.

    Returns ``([(source_path, fingerprint), ...], registry)`` in path order. A
    partition is pending when ``force`` is set, it has no completion entry, or
    its source fingerprint changed (ingest merges late rows into partition
    files in place). With ``force`` the registry starts empty. The caller
    persists the returned registry via ``save_completion_registry``. Raises
    ValueError on a non-canonical parquet file other than an orphaned temp
    file.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    registry = (
        # Not a plain ``{}``: save_completion_registry treats every key of a
        # plain dict as changed and would re-send the whole run each save.
        CompletionRegistry()
        if force
        else load_completion_registry(
            stage_name, artifact_root=resolved_root, data_dir=data_dir
        )
    )
    # Same stray-file contract as iter_date_shard_partitions.
    fingerprints: dict[str, str] = {}
    for path, version in list_artifacts_with_versions(
        dataset_root(source_dataset, artifact_root=resolved_root, data_dir=data_dir),
        suffix=".parquet",
    ).items():
        if match_date_shard_partition(path) is not None:
            fingerprints[path] = version
            continue
        if is_orphaned_temp_artifact(path):
            LOGGER.warning("Skipping orphaned temp partition file: %s", path)
            continue
        msg = (
            f"Non-canonical parquet file in dataset {source_dataset!r}: {path}. "
            "Re-partition or remove it before running stages."
        )
        raise ValueError(msg)
    pending: list[tuple[str, str]] = []
    for source_path in sorted(fingerprints):
        fingerprint = fingerprints[source_path]
        entry = registry.get(source_path)
        if force or entry is None or entry.fingerprint != fingerprint:
            pending.append((source_path, fingerprint))
    return pending, registry
