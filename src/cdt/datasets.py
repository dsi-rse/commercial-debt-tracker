"""Dataset path and partition helpers for file-native CDT pipelines."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from cdt import settings
from cdt.storage import (
    ArtifactPath,
    artifact_exists,
    join_artifact_path,
    list_artifacts,
    normalize_artifact_path,
    read_json_artifact,
    write_json_artifact,
)

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


def load_completed_partitions(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> set[str]:
    """Load completed source partitions for one stage."""
    path = completion_registry_path(
        stage_name, artifact_root=artifact_root, data_dir=data_dir
    )
    if not artifact_exists(path):
        return set()
    payload = read_json_artifact(path)
    if not isinstance(payload, dict):
        return set()
    values = payload.get("source_partitions", [])
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value).strip()}


def save_completed_partitions(
    stage_name: str,
    source_partitions: set[str],
    *,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
) -> str:
    """Persist completed source partitions for one stage."""
    return write_json_artifact(
        completion_registry_path(
            stage_name, artifact_root=artifact_root, data_dir=data_dir
        ),
        {
            "stage": stage_name,
            "source_partitions": sorted(source_partitions),
        },
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


def parse_date_shard_partition(path: ArtifactPath) -> dict[str, str]:
    """Parse a canonical date/shard partition path."""
    normalized = normalize_artifact_path(path)
    match = PARTITION_PATTERN.search(normalized)
    if match is None:
        raise ValueError(f"Unrecognized date/shard partition path: {normalized}")
    return match.groupdict()


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
        partition = parse_date_shard_partition(path)
        partition_date = date.fromisoformat(partition["date"])
        if start_date is not None and partition_date < start_date:
            continue
        if end_date is not None and partition_date > end_date:
            continue
        filtered.append(path)
    return filtered


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


def shard_for_accession(accession_number: str) -> str:
    """Return the canonical date/shard partition for one accession."""
    return f"{zlib_crc32(accession_number) % ITEMIZE_CLASSIFY_EXTRACT_SHARDS:04d}"


def shard_for_cik(cik: str) -> str:
    """Return the canonical cik-shard partition for one CIK."""
    return f"{zlib_crc32(cik) % MATCH_SHARDS:04d}"


def zlib_crc32(value: str) -> int:
    """Return a stable non-cryptographic integer hash."""
    from zlib import crc32

    return int(crc32(value.encode("utf-8")))


def unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    """Return de-duplicated values while preserving first-seen order."""
    return tuple(dict.fromkeys(values))
