"""Dataset path and partition helpers for file-native CDT pipelines."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from cdt import settings
from cdt.shared import get_logger
from cdt.storage import (
    ArtifactPath,
    artifact_exists,
    is_orphaned_temp_artifact,
    join_artifact_path,
    list_artifacts,
    normalize_artifact_path,
    read_json_artifact,
    write_json_artifact,
)

LOGGER = get_logger(__name__)

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
    """Return the path for one stage completion registry."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root, data_dir=data_dir),
        "runs",
        stage_name,
        "completed-partitions.json",
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


def load_completed_partitions(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> set[str]:
    """Load completed source partition paths for one stage (v1-compatible view)."""
    return set(
        load_completion_registry(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        )
    )


def load_completion_registry(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> dict[str, CompletedPartition]:
    """Load one stage's completion registry, migrating v1 path lists in memory."""
    path = completion_registry_path(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
    )
    if not artifact_exists(path):
        return {}
    payload = read_json_artifact(path)
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
            )
        return registry
    values = payload.get("source_partitions", [])
    if not isinstance(values, list):
        return {}
    return {str(value): CompletedPartition() for value in values if str(value).strip()}


def save_completion_registry(
    stage_name: str,
    registry: dict[str, CompletedPartition],
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Persist one stage's completion registry in the v2 format."""
    return write_json_artifact(
        completion_registry_path(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ),
        {
            "stage": stage_name,
            "version": 2,
            "partitions": {
                path: {
                    "fingerprint": entry.fingerprint,
                    **({"item_ids": sorted(entry.item_ids)} if entry.item_ids else {}),
                }
                for path, entry in sorted(registry.items())
            },
        },
    )


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


def shard_for_cik(cik: str) -> str:
    """Return the canonical cik-shard partition for one CIK."""
    return shard_label(cik, MATCH_SHARDS)


def zlib_crc32(value: str) -> int:
    """Return a stable non-cryptographic integer hash."""
    from zlib import crc32

    return int(crc32(value.encode("utf-8")))


def unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    """Return de-duplicated values while preserving first-seen order."""
    return tuple(dict.fromkeys(values))
