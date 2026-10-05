"""Dataset path and partition helpers for file-native CDT pipelines."""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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

# The 6-K triage stage's output dataset. Named here rather than beside its
# writer, as the other dataset names are, because the extractor must name it to
# claim work from it and cannot import `cdt.sixk`: `cdt.sixk.triage` imports
# `normalize_reasoning_effort` from `cdt.extractor.core`, so the dependency runs
# the other way. This module is the leaf both sides already import.
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

    Named ``_root``, not ``_path``, because this module splits the two without
    exception and #220 broke that (#227): prefixes are ``*_root``
    (``dataset_root``, ``items_root``, ``mentions_root``, ``batches_root``,
    ``mirror_root``) and single objects are ``*_path`` (``run_manifest_path``,
    ``failure_registry_path``, ``active_job_path``, ``final_pointer_path``).
    With ``completion_registry_shard_path`` for the objects underneath it, this
    is the coherent pair.
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

    Kept only because four stage modules still import this name, and all four
    live in files owned by branches running in parallel with this one, where a
    rename would be a pure textual conflict for no behavioural gain. Retiring
    it -- the four call sites, plus the five run manifests whose
    ``"completion_registry"`` key now names a directory while its neighbours
    ``"audit_path"`` and ``"failure_registry"`` still name files -- is tracked
    on #227 and should land once those branches merge.
    """
    return completion_registry_root(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
    )


# The registry is sharded by its source partition's year-month. Stages walk
# pending partitions in sorted path order, which is date order, so one
# 100-partition chunk covers one or two consecutive dates and therefore one or
# two shards. That locality is the whole mechanism: it is what turns a 56.8 MB
# rewrite per batch boundary into a ~0.2 MB one (#191).
_REGISTRY_SHARD_DATE_CHARS = len("YYYY-MM")
# A key with no parseable partition date still has to round-trip. Dropping it
# would silently lose completion state, so it gets a named shard of its own.
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
    size+mtime locally); None on entries saved path-only through
    ``save_completed_partitions``, which read as "complete as recorded,
    reprocess if the source ever changes".
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
    """Load one stage's registry from every date shard.

    Loading is the one place that reads everything; it happens once or twice
    per run, against the 4,400 saves per itemize pass that #191 is about.

    The pre-#191 single ``runs/<stage>/completed-partitions.json`` object is
    not read: nothing in that format is kept during beta, so there is no
    migration path from it and no rollback path back to it. On S3 the shard
    prefix ``runs/<stage>/completed`` also matches that object, so the listing
    keeps only ``date=`` files rather than everything under the prefix.

    The shard reads are issued concurrently because sharding turned one GET
    into one per occupied month, and a serial loop over them is #110's
    pathology in a new place (#227). Measured at full-corpus shape -- 393
    shards -- a load issues one LIST and 393 GETs; serially, at
    the 70 ms round trip #110 measured on this same 1-vCPU stack, that is 27.5 s
    per load and a pipeline run does five of them, growing by one shard every
    month forever. Note that #191's own text claimed the old design "cannot be
    fixed by parallelising reads, because it is one object"; that stopped being
    true at 393 objects.

    Overlay order is preserved regardless of completion order: ``map`` yields
    in submission order, and ``list_artifacts`` already returns sorted paths,
    so the shards still merge in exactly the sequence the serial loop used.
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


# Capped at botocore's default `max_pool_connections` (10): the GETs all share
# one cached client, and more threads than connections just queue inside the
# pool while logging "connection pool is full" on every overflow. Raising both
# together belongs with `S3_CLIENT_CONFIG` in storage.py, not here.
_REGISTRY_LOAD_CONCURRENCY = 10


def _read_registry_shards(shard_paths: list[str]) -> list[object]:
    """Read registry shard objects concurrently, in ``shard_paths`` order.

    Materialized rather than streamed so the pool is joined before the caller
    merges anything: a read that raises surfaces from here, instead of halfway
    through an overlay that has already mutated the registry being built.
    """
    if not shard_paths:
        return []
    if len(shard_paths) == 1:
        # The overwhelmingly common local/test shape; skip the pool entirely so
        # a one-shard load costs no thread setup.
        return [read_json_artifact(shard_paths[0])]
    with ThreadPoolExecutor(
        max_workers=min(_REGISTRY_LOAD_CONCURRENCY, len(shard_paths))
    ) as pool:
        return list(pool.map(read_json_artifact, shard_paths))


# Keys are persisted with the artifact root stripped off when the remainder is a
# bare canonical partition path, and whole otherwise. The two are told apart by
# shape, which is safe because they cannot collide: any whole path --
# `/srv/cdt/documents/date=...`, `s3://bucket/documents/date=...` -- has a
# dataset segment with a slash in it, which the pattern's `[a-z\-]+` cannot
# match under `fullmatch`. So no whole key is ever given a second root.
_REGISTRY_VERSION = 3


# Both key prefixes are derived once per object read or written rather than once
# per key (#227). Per key, `join_artifact_path` built a `pathlib.Path` and
# `_relative_registry_key` rebuilt a normalized prefix and an f-string, inside
# comprehensions that run over every entry the registry holds. Measured on a
# full-corpus-shaped registry of 440,000 entries (local storage, best of three),
# that made the load 2.32 s against 1.06 s for the single object it replaced,
# and made `_absolute_registry_key` 56% of the whole save under cProfile.
# Hoisted, the same load is 1.08 s: level with the pre-#191 object on a short
# root, and faster under a long absolute root (1.18 s against 1.27 s), because
# the per-key cost no longer grows with the root's length.
#
# A bare canonical partition path, which is the only shape either prefix is ever
# applied to (`PARTITION_PATTERN.fullmatch` guarantees it, with no `.` or `..`
# segment and no doubled separator), so the probe below derives the join prefix
# by construction and cannot disagree with `join_artifact_path` on any root --
# including `.` and `""`, where `pathlib` collapses the join rather than
# prefixing it.
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

    Keys used to carry the whole path, root included. That made the registry
    larger than it needs to be -- 129 B per entry against 105 B, measured over
    440,000 full-corpus-shaped entries -- and, less obviously, made an artifact
    root non-portable: copy a root and every key keeps a prefix that no longer
    exists, so the copy's registry matches nothing and the corpus reads as
    entirely unprocessed. That is #107's shape arriving by way of `cp -r`.

    Relativized only when the result reads back through
    ``_absolute_registry_key``, so the two are exactly inverse and a key no
    reader could reattach a root to is stored whole instead. The hot paths call
    ``_strip_registry_root`` with the prefix hoisted out of their loop; this is
    the one-key spelling, and the pair the inverse property is asserted on.
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

    Sorted on the relativized key alone (``itemgetter(0)``), not on the whole
    tuple: a tuple sort falls through to comparing two ``CompletedPartition``
    dataclasses whenever two keys relativize alike, and they are unordered, so
    it raises ``TypeError``. Not reachable through ``save_completion_registry``
    today -- ``_registry_entries`` absolutizes every stored key first -- but
    crashing on a key collision is a bad trade for a free sort key, and
    ``json.dumps(sort_keys=True)`` orders the written bytes anyway.
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
    per-batch save of #111 durable *for the stages that save per batch*: a save
    interrupted after three of five shards leaves those three persisted, and
    every persisted entry is a partition whose output was already written.
    Itemize, classify and 6-K triage do save at every batch boundary. Extract
    does not, and it is the stage where this matters most (#227):
    ``extract_pending_items`` accepts ``batch_size`` but never chunks its
    partition loop -- the value only reaches the run manifest -- so its sole
    save is after the loop, an interruption still discards the whole run's
    registry progress, and that one save is also the only one that can span
    ~400 shards. Still strictly better than the pre-#191 object, which lost
    100% of its progress on the same interruption; just not the per-batch
    durability #111 asked for.
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
        # The swap above committed these keys, so they are part of the
        # freshest persisted state now and the next boundary must not re-send
        # them. Without this the dirty set is the whole run's write set rather
        # than the batch's, so batch k rewrites every shard batches 1..k-1
        # touched and the cost is quadratic in the run's length again -- the
        # thing #191 exists to remove. Cleared per shard rather than once at
        # the end because an exhausted swap raises: the shards that did land
        # stay clean and only the failed shard's keys are retried.
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


def save_completed_partitions(
    stage_name: str,
    source_partitions: set[str],
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Persist completed partitions path-only, without fingerprints.

    Callers that need row-aware completion should use
    ``save_completion_registry`` directly.
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
    Entries with no fingerprint (saved path-only, or targets that predate the
    registry) are stamped with the current fingerprint in the returned registry so future growth is
    detectable; the caller persists it via save_completion_registry.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    registry = (
        # Not a plain ``{}``: save_completion_registry treats every key of one
        # as changed, which re-sends the whole run at every batch boundary.
        CompletionRegistry()
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
