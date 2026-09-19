"""Dataset path and partition helpers for file-native CDT pipelines."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
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

# The 6-K triage stage's output dataset. Named here rather than beside its
# writer, as the other dataset names are, because the extractor must name it to
# claim work from it and cannot import `cdt.sixk`: `cdt.sixk.triage` imports
# `normalize_reasoning_effort` from `cdt.extractor.core`, so the dependency runs
# the other way. This module is the leaf both sides already import.
SIXK_SNIPPET_DATASET_NAME = "sixk-snippets"
ITEMIZE_CLASSIFY_EXTRACT_SHARDS = 8
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


def completion_registry_path(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the prefix holding one stage's completion registry shards.

    A prefix, not a single object, since #191. The registry used to be one
    ``runs/<stage>/completed-partitions.json`` that every batch boundary read
    *and* rewrote whole, under compare-and-swap. Measured at full corpus scale
    with this module's own functions -- 8,533 business days x ~52 occupied
    shards = 440,000 entries -- that object serializes to 56.8 MB (129 B per
    entry; the real registry on ``data/genwindow-eval-apr`` measures 146 B per
    entry over 1,640 entries), one compare-and-swap cycle moves 113.5 MB and
    costs 3.2 s of JSON, and a single itemize pass does 4,400 of them: 499 GB
    moved and 3.9 h of pure JSON CPU before any work happens. Sharded by the
    source partition's year-month, a cycle touches only the one or two months
    its 100-partition chunk covers and moves half a megabyte.

    Callers that want the pre-#191 object -- read-compat and migration -- want
    ``legacy_completion_registry_path`` instead.
    """
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "runs",
        stage_name,
        "completed",
    )


def legacy_completion_registry_path(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the pre-#191 single-object registry path, for read-compat.

    Still read on every load, because an artifact root written before #191 has
    all of its completion state here and nowhere else. Reading it as empty is
    not a slow path, it is the #107 failure mode: ``force=False`` reported 0
    pending partitions and ``force=True`` reported 20,046, because the registry
    the run consulted was not the registry the corpus had.
    """
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "runs",
        stage_name,
        "completed-partitions.json",
    )


# The registry is sharded by its source partition's year-month. Stages walk
# pending partitions in sorted path order, which is date order, so one
# 100-partition chunk covers one or two consecutive dates and therefore one or
# two shards. That locality is the whole mechanism: it is what turns a 56.8 MB
# rewrite per batch boundary into a ~0.2 MB one (#191).
_REGISTRY_SHARD_DATE_CHARS = len("YYYY-MM")
# A key with no parseable partition date still has to round-trip. Dropping it
# would silently lose completion state -- the same class of bug as reading a
# legacy registry as empty -- so it gets a named shard of its own.
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
        completion_registry_path(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ),
        f"date={shard_label}.json",
    )


@dataclass
class CompletedPartition:
    """One source partition's completion record (registry v2).

    ``fingerprint`` is the source object's version at processing time (S3 ETag;
    size+mtime locally); None on entries migrated from the v1 path list, which
    read as "complete as recorded, reprocess if the source ever changes".
    ``item_ids`` (extract only) are the content-terminal rows — SUCCESS or
    FAILED-on-validation — so re-processing a partition is row-level and never
    re-pays rows that already have a real outcome (#49, #62).
    """

    fingerprint: str | None = None
    item_ids: frozenset[str] = frozenset()
    # False when some relevant rows are not yet content-terminal (an aborted or
    # infra-interrupted pass): the partition stays pending and only the rows
    # missing from item_ids are processed next time.
    complete: bool = True


class CompletionRegistry(dict[str, CompletedPartition]):
    """A loaded completion registry that remembers which keys the run changed.

    Registries have concurrent writers with no other serialization guarantee
    (a scheduled daily run, an hourly poll tick, and manual CLI stage runs can
    overlap, #88), so a save must overlay only the entries this run actually
    wrote onto the freshest persisted state — overlaying the whole loaded dict
    would revert another writer's updates to entries this run merely read.
    Entries must therefore be changed by assignment (``registry[key] = entry``),
    never by mutating a loaded ``CompletedPartition`` in place.
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
    """Load completed source partition paths for one stage (v1-compatible view)."""
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
    """Load one stage's registry from every date shard, plus the legacy object.

    The pre-#191 single object is read first and the date shards overlay it, so
    an entry a run re-wrote after the split wins over the copy the legacy object
    still carries. Loading is the one place that reads everything; it happens
    once or twice per run, against the 4,400 saves per itemize pass that #191
    is about.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    entries: dict[str, CompletedPartition] = {}
    legacy_path = legacy_completion_registry_path(
        stage_name, artifact_root=resolved_root, data_dir=data_dir
    )
    if artifact_exists(legacy_path):
        entries.update(
            _registry_entries(read_json_artifact(legacy_path), resolved_root)
        )
    for shard_path in list_artifacts(
        completion_registry_path(
            stage_name, artifact_root=resolved_root, data_dir=data_dir
        ),
        suffix=".json",
    ):
        entries.update(_registry_entries(read_json_artifact(shard_path), resolved_root))
    return CompletionRegistry(entries)


# v3 persists keys with the artifact root stripped off; v1 and v2 keys carry
# whatever whole path the writing run resolved. Both forms are read by shape
# rather than by version, which is safe because the shapes cannot collide: a
# relativized key is a *bare* canonical partition path, and any whole path --
# `/srv/cdt/documents/date=...`, `s3://bucket/documents/date=...`, or the
# relative `data/genwindow-eval-apr/documents/date=...` the real registry on
# that root holds -- has a dataset segment with a slash in it, which the
# pattern's `[a-z\-]+` cannot match under `fullmatch`. So no v2 key is ever
# mistaken for a v3 one and given a second root.
_REGISTRY_VERSION = 3


def _relative_registry_key(key: str, artifact_root: str) -> str:
    """Return one registry key with the artifact root stripped off.

    Keys used to carry the whole path, root included. That made the registry
    larger than it needs to be -- 129 B per entry against 105 B, measured over
    440,000 full-corpus-shaped entries -- and, less obviously, made an artifact
    root non-portable: copy a root and every key keeps a prefix that no longer
    exists, so the copy's registry matches nothing and the corpus reads as
    entirely unprocessed. That is #107's shape arriving by way of `cp -r`.

    Relativized only when the result reads back through
    ``_absolute_registry_key``, so the two are exactly inverse and a key no
    reader could reattach a root to is stored whole instead.
    """
    prefix = f"{normalize_artifact_path(artifact_root).rstrip('/')}/"
    if not key.startswith(prefix):
        return key
    relative = key[len(prefix) :]
    return relative if PARTITION_PATTERN.fullmatch(relative) else key


def _absolute_registry_key(stored: str, artifact_root: str) -> str:
    """Return one persisted key with the artifact root reattached.

    A stored key is root-relative exactly when it is a bare canonical
    date/shard partition path -- ``fullmatch``, not the suffix ``search`` the
    parsers use, so a whole path or an S3 URI is not mistaken for one. Anything
    else, v1 and v2 keys included, is returned as stored.
    """
    if not PARTITION_PATTERN.fullmatch(stored):
        return stored
    return join_artifact_path(artifact_root, stored)


def _registry_entries(
    payload: object, artifact_root: str
) -> dict[str, CompletedPartition]:
    """Parse one persisted registry object into whole-path-keyed entries."""
    return {
        _absolute_registry_key(key, artifact_root): entry
        for key, entry in _parse_registry_payload(payload).items()
    }


def _parse_registry_payload(payload: object) -> dict[str, CompletedPartition]:
    """Parse a persisted registry payload (v3, v2 or v1); junk reads as empty."""
    if not isinstance(payload, dict):
        return {}
    partitions = payload.get("partitions")
    if isinstance(partitions, dict):
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
    values = payload.get("source_partitions", [])
    if not isinstance(values, list):
        return {}
    return {str(value): CompletedPartition() for value in values if str(value).strip()}


def _registry_payload(
    stage_name: str,
    shard_label: str,
    registry: dict[str, CompletedPartition],
    *,
    artifact_root: str,
) -> dict[str, object]:
    """Build the persisted v3 payload for one date shard of a registry."""
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
                (_relative_registry_key(key, artifact_root), entry)
                for key, entry in registry.items()
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
    """Persist one stage's completion registry via compare-and-swap merge.

    Registries have concurrent writers (daily vs poll vs manual CLI runs, #88),
    and a blind overwrite loses every entry another writer persisted since this
    run loaded its snapshot — including entries whose loss silently strands or
    fake-completes partitions. Only the entries this run changed (the
    ``CompletionRegistry.dirty`` set; every entry for a plain dict, which force
    runs and tests pass) are overlaid on the freshest persisted state, and a
    lost race re-reads and retries.

    Since #191 the changed entries are grouped by date shard and each shard is
    compare-and-swapped on its own, so a save reads and rewrites only the
    months it touched instead of the whole corpus. Nothing else moves: the
    payload format, the merge rule and the dirty-set semantics are the ones
    above, applied per object rather than to one object. Because the dirty set
    already records the write set, this is path derivation, not new
    bookkeeping.

    Each shard's compare-and-swap is independent, which is what keeps the
    per-batch save of #111 durable: a save interrupted after three of five
    shards leaves those three persisted, and every persisted entry is a
    partition whose output was already written.

    A save is also where an existing root's pre-#191 single object is folded
    into date shards, because a save is the only point that holds the writer
    lease. Every stage saves unconditionally at the end of its run, so the
    migration lands on the first run after the upgrade even when nothing was
    pending.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    changed = sorted(
        registry.dirty if isinstance(registry, CompletionRegistry) else registry
    )
    adopted, legacy_version = _legacy_registry_to_migrate(
        stage_name, artifact_root=resolved_root, data_dir=data_dir
    )
    by_shard: dict[str, dict[str, CompletedPartition]] = {}
    for key in changed:
        by_shard.setdefault(_registry_shard_label(key), {})[key] = registry[key]
    adopted_by_shard: dict[str, dict[str, CompletedPartition]] = {}
    for key, entry in adopted.items():
        adopted_by_shard.setdefault(_registry_shard_label(key), {})[key] = entry
    for shard_label in sorted(by_shard.keys() | adopted_by_shard.keys()):
        _save_registry_shard(
            stage_name,
            shard_label,
            by_shard.get(shard_label, {}),
            adopted=adopted_by_shard.get(shard_label, {}),
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
    if adopted:
        # Only after every shard's swap succeeded -- _save_registry_shard
        # raises rather than returning on exhaustion, so the marker below can
        # never be written over state that did not make it into a shard.
        _retire_legacy_registry(
            stage_name,
            legacy_version,
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
    return completion_registry_path(
        stage_name, artifact_root=resolved_root, data_dir=data_dir
    )


def _legacy_registry_to_migrate(
    stage_name: str,
    *,
    artifact_root: str,
    data_dir: Path | None,
) -> tuple[dict[str, CompletedPartition], str]:
    """Return the pre-#191 object's entries and the version token to retire it.

    Read on every save, not once behind a flag, because there is nowhere to
    keep a flag that a concurrent writer would also see (#88). The steady-state
    cost after migration is one HeadObject plus one GET of a ~200-byte marker
    per save, against the 56.8 MB this whole change is removing.
    """
    path = legacy_completion_registry_path(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
    )
    if not artifact_exists(path):
        return {}, ""
    try:
        payload, version = read_json_artifact_versioned(path)
    except FileNotFoundError:
        return {}, ""
    return _registry_entries(payload, artifact_root), version


def _retire_legacy_registry(
    stage_name: str,
    version: str,
    *,
    artifact_root: ArtifactPath | None,
    data_dir: Path | None,
) -> bool:
    """Replace a migrated pre-#191 object with an empty forwarding marker.

    Compare-and-swapped on the version the migration read, so a writer that
    added entries to the object in the meantime is not clobbered. A lost swap
    is harmless and needs no retry: the object stays as it was, every entry it
    held is already in a shard, and the next save migrates it again --
    idempotently, because an adopted entry never overwrites the shard's copy.

    Cleared rather than deleted so the object survives as an operator-visible
    record of where the state went. The migration is still one-way: code from
    before #191 reads the marker as an empty registry, which is the #107
    failure mode, so a rollback has to rebuild the object from the shards.
    """
    return replace_json_artifact_if_match(
        legacy_completion_registry_path(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ),
        {
            "stage": stage_name,
            "version": 2,
            "partitions": {},
            "migrated_to": completion_registry_path(
                stage_name, artifact_root=artifact_root, data_dir=data_dir
            ),
        },
        version=version,
    )


def _save_registry_shard(
    stage_name: str,
    shard_label: str,
    entries: dict[str, CompletedPartition],
    *,
    adopted: dict[str, CompletedPartition],
    artifact_root: str,
    data_dir: Path | None,
) -> str:
    """Compare-and-swap ``entries`` into one date shard of a stage's registry.

    ``adopted`` entries come from the pre-#191 single object and are inserted
    only where the shard has no entry for the key: anything already in the
    shard was written after the split and is therefore newer than the copy the
    legacy object still carries.

    The merge runs on whole-path keys and the payload strips the root on the
    way out, so a shard written at an older key convention is normalized on
    its next write rather than accumulating both spellings of a key.
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
                    {**adopted, **entries},
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
        for key, entry in adopted.items():
            merged.setdefault(key, entry)
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


def save_completed_partitions(
    stage_name: str,
    source_partitions: set[str],
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Persist completed partitions path-only (v1-compatible writer).

    Merges into the v2 registry without fingerprints; callers migrating to
    row-aware completion should use save_completion_registry directly.
    """
    registry = load_completion_registry(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
    )
    for path in source_partitions:
        registry.setdefault(path, CompletedPartition())
    return save_completion_registry(
        stage_name, registry, artifact_root=artifact_root, data_dir=data_dir
    )


def pending_source_partitions(
    stage_name: str,
    source_dataset: str,
    target_dataset: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    force: bool = False,
) -> tuple[list[tuple[str, str]], dict[str, CompletedPartition]]:
    """Select source partitions a whole-partition stage still has to process.

    Returns ``([(source_path, fingerprint), ...], registry)``. A partition is
    pending when it has no completion entry or its source fingerprint changed —
    ingest merges late-arriving rows into partition files in place, and a
    path-level "target exists" skip silently strands those rows forever (#62).
    Legacy entries (v1 lists, or targets that predate the registry) are stamped
    with the current fingerprint in the returned registry so future growth is
    detectable; the caller persists it via save_completion_registry.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    registry = (
        {}
        if force
        else load_completion_registry(
            stage_name, artifact_root=resolved_root, data_dir=data_dir
        )
    )
    # Same stray contract as iter_date_shard_partitions: an orphaned tempfile
    # is junk to skip, but any other non-canonical parquet is real data laid
    # out wrong — silently dropping it here would run the stage on nothing
    # while ingest keeps counting the file's rows as ingested.
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
    existing_target_ids = (
        set()
        if force
        else existing_date_shard_partition_ids(
            target_dataset, artifact_root=resolved_root, data_dir=data_dir
        )
    )
    pending: list[tuple[str, str]] = []
    for source_path in sorted(fingerprints):
        partition = parse_date_shard_partition(source_path)
        fingerprint = fingerprints[source_path]
        entry = registry.get(source_path)
        if force:
            pending.append((source_path, fingerprint))
            continue
        if entry is None:
            if (partition["date"], partition["shard"]) in existing_target_ids:
                registry[source_path] = CompletedPartition(fingerprint=fingerprint)
                continue
            pending.append((source_path, fingerprint))
            continue
        if entry.fingerprint == fingerprint:
            continue
        if entry.fingerprint is None:
            # Reassign rather than mutate so the change lands in the registry's
            # dirty set and survives the compare-and-swap merge on save (#88).
            registry[source_path] = replace(entry, fingerprint=fingerprint)
            continue
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


def parse_cik_shard_partition(path: ArtifactPath) -> dict[str, str]:
    """Parse a canonical cik-shard partition path."""
    normalized = normalize_artifact_path(path)
    match = CIK_PARTITION_PATTERN.search(normalized)
    if match is None:
        raise ValueError(f"Unrecognized cik-shard partition path: {normalized}")
    return match.groupdict()


def iter_date_shard_partitions(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[str]:
    """List canonical date/shard partitions, optionally filtered by date window."""
    paths = list_artifacts(
        dataset_root(dataset_name, artifact_root=artifact_root, data_dir=data_dir),
        suffix=".parquet",
    )
    filtered: list[str] = []
    for path in paths:
        partition = match_date_shard_partition(path)
        if partition is None:
            # A tempfile orphaned by a crash mid-write_table is junk and must
            # not brick every stage until someone deletes it by hand (#68). Any
            # other non-canonical parquet is real data laid out wrong (e.g. a
            # pre-migration flat file); skipping it would silently run the
            # pipeline on nothing while ingest keeps counting its rows as
            # already ingested, so fail loudly instead.
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

    One LIST of the dataset answers "does the output partition exist?" for any
    number of source partitions. The per-partition alternative — a HeadObject on
    each candidate target — costs one sequential round-trip per partition ever
    written and dominates stage runtime once the dataset is large (#83).
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


def iter_cik_shard_partitions(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> list[str]:
    """List canonical cik-shard partitions for one dataset."""
    return [
        path
        for path in list_artifacts(
            dataset_root(dataset_name, artifact_root=artifact_root, data_dir=data_dir),
            suffix=".parquet",
        )
        if CIK_PARTITION_PATTERN.search(path)
    ]


def shard_label(value: str, shard_count: int) -> str:
    """Return the canonical shard label for one key.

    The single source for the shard contract — crc32, modulo, four-digit label —
    that PARTITION_PATTERN and every partition directory name depend on. Any
    change here strands existing partitions (#61), so change it nowhere else.
    """
    return f"{zlib_crc32(value) % shard_count:04d}"


def shard_for_accession(accession_number: str) -> str:
    """Return the canonical date/shard partition for one accession."""
    return shard_label(accession_number, ITEMIZE_CLASSIFY_EXTRACT_SHARDS)


CIK_DIGITS = 10


def normalize_cik(cik: object) -> str:
    """Return SEC's canonical 10-digit zero-padded CIK string.

    Published rows carry this form so they join against anything keyed on SEC's
    canonical CIKs (#153). Non-numeric input is returned stripped rather than
    padded, so a malformed manifest value stays visibly malformed.
    """
    text = str(cik).strip()
    return text.zfill(CIK_DIGITS) if text.isdigit() else text


def shard_for_cik(cik: str) -> str:
    """Return the canonical cik-shard partition for one CIK.

    Hashes the unpadded form: partitions written before CIKs were zero-padded
    (#153) hashed bare strings like ``707605``, and per `shard_label`'s
    contract a changed input strands them. Padded and unpadded spellings of one
    CIK therefore always land in the same shard.
    """
    return shard_label(str(cik).lstrip("0") or "0", MATCH_SHARDS)


def zlib_crc32(value: str) -> int:
    """Return a stable non-cryptographic integer hash."""
    from zlib import crc32

    return int(crc32(value.encode("utf-8")))


def unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    """Return de-duplicated values while preserving first-seen order."""
    return tuple(dict.fromkeys(values))
