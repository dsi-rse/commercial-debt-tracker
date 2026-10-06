"""Dataset path, partition and completion-registry helpers for CDT pipelines.

Prefixes are named ``*_root`` and single objects ``*_path``.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from operator import itemgetter
from pathlib import Path
from typing import Self, cast

from cdt import settings
from cdt.shared import get_logger
from cdt.storage import (
    ArtifactPath,
    artifact_exists,
    is_orphaned_temp_artifact,
    join_artifact_path,
    list_artifacts,
    list_artifacts_with_versions,
    normalize_artifact_path,
    read_json_artifact,
    read_json_artifact_versioned,
    replace_json_artifact_if_match,
    write_json_artifact,
    write_json_artifact_if_absent,
)

LOGGER = get_logger(__name__)

# The 6-K triage stage's output dataset. Named here, not beside its writer,
# because the extractor must name it and cannot import ``cdt.sixk`` (which
# imports ``cdt.extractor.core``); this module is the leaf both import.
SIXK_SNIPPET_DATASET_NAME = "sixk-snippets"
MATCH_SHARDS = 64
PARTITION_PATTERN = re.compile(
    r"(?P<dataset>[a-z\-]+)/date=(?P<date>\d{4}-\d{2}-\d{2})/shard=(?P<shard>\d{4})/part-0000\.parquet$"
)
CIK_PARTITION_PATTERN = re.compile(
    r"(?P<dataset>[a-z\-]+)/cik_shard=(?P<cik_shard>\d{4})/part-0000\.parquet$"
)


def default_artifact_root(data_dir: Path | None = None) -> str:
    """Return the default artifact root for local development."""
    return str(data_dir or settings.DATA_DIR)


def resolve_artifact_root(
    artifact_root: ArtifactPath | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Resolve the configured artifact root to a normalized string path."""
    return normalize_artifact_path(artifact_root or default_artifact_root(data_dir))


def dataset_root(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the root for one canonical dataset."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir), dataset_name
    )


def run_manifest_path(
    stage_name: str,
    run_id: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the path for one stage run manifest."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "runs",
        stage_name,
        f"run_id={run_id}.json",
    )


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


def completion_registry_path(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Deprecated alias for ``completion_registry_root``; do not add callers.

    Removal is tracked on #227.
    """
    return completion_registry_root(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
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

    Only ``date=*.json`` objects are read; on S3 the shard prefix also matches
    the unsupported legacy ``runs/<stage>/completed-partitions.json``, which is
    ignored. Shards are read concurrently and merged in sorted path order.
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
# both together, with ``S3_CLIENT_CONFIG`` in storage.py.
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


def failure_registry_path(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the path for one stage failure registry."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "failures",
        stage_name,
        "failures.json",
    )


def load_row_failures(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Load one stage's row-level failure registry, keyed by row id."""
    path = failure_registry_path(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
    )
    if not artifact_exists(path):
        return {}
    payload = read_json_artifact(path)
    if not isinstance(payload, dict):
        return {}
    failures = payload.get("failures", {})
    if not isinstance(failures, dict):
        return {}
    return {
        str(key): cast(dict[str, object], value)
        for key, value in failures.items()
        if isinstance(value, dict)
    }


def save_row_failures(
    stage_name: str,
    failures: dict[str, dict[str, object]],
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Persist one stage's row-level failure registry.

    Unlike the completion registry this is diagnostic, not control flow: nothing
    reads it to decide what to process. It exists so rows that a run dropped are
    recoverable as a work-list instead of only appearing in an audit log.
    """
    return write_json_artifact(
        failure_registry_path(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ),
        {
            "stage": stage_name,
            "failure_count": len(failures),
            "failures": {key: failures[key] for key in sorted(failures)},
        },
    )


def extractor_run_path(
    run_id: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the path for one extractor full audit JSONL artifact."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "extractor-runs",
        f"run_id={run_id}",
        "full.jsonl",
    )


def date_shard_partition_path(
    dataset_name: str,
    *,
    partition_date: str,
    shard: str,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return one canonical date/shard partition file path."""
    return join_artifact_path(
        dataset_root(dataset_name, artifact_root=artifact_root, data_dir=data_dir),
        f"date={partition_date}",
        f"shard={shard}",
        "part-0000.parquet",
    )


def cik_shard_partition_path(
    dataset_name: str,
    *,
    cik_shard: str,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return one canonical cik-shard partition file path."""
    return join_artifact_path(
        dataset_root(dataset_name, artifact_root=artifact_root, data_dir=data_dir),
        f"cik_shard={cik_shard}",
        "part-0000.parquet",
    )


def match_date_shard_partition(path: ArtifactPath) -> dict[str, str] | None:
    """Match a canonical date/shard partition path, or None when non-canonical.

    The single matcher behind both the strict parser and the dataset scan, so
    the two can never drift on what counts as a canonical layout.
    """
    match = PARTITION_PATTERN.search(normalize_artifact_path(path))
    return match.groupdict() if match else None


def parse_date_shard_partition(path: ArtifactPath) -> dict[str, str]:
    """Parse a canonical date/shard partition path."""
    partition = match_date_shard_partition(path)
    if partition is None:
        normalized = normalize_artifact_path(path)
        raise ValueError(f"Unrecognized date/shard partition path: {normalized}")
    return partition


def iter_date_shard_partitions(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[str]:
    """List canonical date/shard partitions, optionally filtered by date window.

    ``start_date`` and ``end_date`` are inclusive; None leaves that side open.
    Orphaned temp files are skipped with a warning. Raises ValueError on any
    other non-canonical parquet file.
    """
    paths = list_artifacts(
        dataset_root(dataset_name, artifact_root=artifact_root, data_dir=data_dir),
        suffix=".parquet",
    )
    filtered: list[str] = []
    for path in paths:
        partition = match_date_shard_partition(path)
        if partition is None:
            # An orphaned tempfile is junk to skip. Any other non-canonical
            # parquet is real data laid out wrong; skipping it would run the
            # pipeline on nothing while ingest counts its rows as ingested.
            if is_orphaned_temp_artifact(path):
                LOGGER.warning("Skipping orphaned temp partition file: %s", path)
                continue
            msg = (
                f"Non-canonical parquet file in dataset {dataset_name!r}: {path}. "
                "Re-partition or remove it before running stages."
            )
            raise ValueError(msg)
        partition_date = date.fromisoformat(partition["date"])
        if start_date is not None and partition_date < start_date:
            continue
        if end_date is not None and partition_date > end_date:
            continue
        filtered.append(path)
    return filtered


def existing_date_shard_partition_ids(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> set[tuple[str, str]]:
    """Return the ``(date, shard)`` ids of every written partition in a dataset.

    One LIST answers "does the output partition exist?" for any number of
    candidates, instead of one HEAD per target.
    """
    return {
        (partition["date"], partition["shard"])
        for partition in (
            parse_date_shard_partition(path)
            for path in iter_date_shard_partitions(
                dataset_name, artifact_root=artifact_root, data_dir=data_dir
            )
        )
    }


def shard_label(value: str, shard_count: int) -> str:
    """Return the canonical shard label for one key.

    The single source for the shard contract (crc32, modulo, four-digit label)
    that PARTITION_PATTERN and every partition directory name depend on. Any
    change strands existing partitions.
    """
    return f"{zlib_crc32(value) % shard_count:04d}"


CIK_DIGITS = 10


def normalize_cik(cik: object) -> str:
    """Return SEC's canonical 10-digit zero-padded CIK string.

    Non-numeric input is returned stripped rather than padded, so a malformed
    value stays visibly malformed.
    """
    text = str(cik).strip()
    return text.zfill(CIK_DIGITS) if text.isdigit() else text


def shard_for_cik(cik: str) -> str:
    """Return the canonical cik-shard partition for one CIK.

    Hashes the unpadded form (``707605``), which existing partitions are keyed
    on, so padded and unpadded spellings of one CIK land in the same shard.
    """
    return shard_label(str(cik).lstrip("0") or "0", MATCH_SHARDS)


def zlib_crc32(value: str) -> int:
    """Return a stable non-cryptographic integer hash."""
    from zlib import crc32

    return int(crc32(value.encode("utf-8")))
