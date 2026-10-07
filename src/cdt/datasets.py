"""Dataset paths, partition naming, shard assignment and filing genres.

Prefixes are named ``*_root`` and single objects ``*_path``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from cdt import settings
from cdt.shared import get_logger
from cdt.storage.objects import (
    ArtifactPath,
    artifact_exists,
    join_artifact_path,
    list_artifacts,
    normalize_artifact_path,
    read_json_artifact,
    write_json_artifact,
)
from cdt.storage.tables import is_orphaned_temp_artifact

LOGGER = get_logger(__name__)

# The 6-K triage stage's output dataset. Named here, not beside its writer,
# because the extractor must name it and cannot import ``cdt.classifier.sixk``
# (which imports ``cdt.extractor``); this module is the leaf both import.
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


#: Filing genres: 8-K runs ingest → itemize → classify, 6-K runs ingest →
#: triage; both converge at extract.
GENRE_8K = "8-K"
GENRE_6K = "6-K"

CDT_FORM_TYPE = "8-K"
DEFAULT_FORM_TYPES: tuple[str, ...] = (CDT_FORM_TYPE,)
# The 6-K genre's forms.
SIXK_FORM_TYPES: tuple[str, ...] = ("6-K", "6-K/A")

DOCUMENT_DATASET_NAME = "documents"
# The 6-K genre's own documents dataset (see IngestConfig.dataset_name).
SIXK_DOCUMENT_DATASET_NAME = "documents-sixk"
ITEM_DATASET_NAME = "items"
CLASSIFICATION_DATASET_NAME = "classifications"


@dataclass(frozen=True)
class Genre:
    """One filing genre: its SEC forms and the datasets its upstream stages write.

    ``document_dataset`` is what ingest writes, ``item_dataset`` the rows the
    published ``items`` table takes from this genre, and ``classified_dataset``
    the rows the extractor reads. For 6-K the last two are the same dataset.
    ``inlines_bodies`` is whether ingest may store document bodies in the
    partition (``--download``); a 6-K row instead points at the mirrored
    submission, so every read does not pay for every body.
    """

    name: str
    form_types: tuple[str, ...]
    document_dataset: str
    item_dataset: str
    classified_dataset: str
    inlines_bodies: bool


#: Every genre, in the order a run prepares them.
GENRES: dict[str, Genre] = {
    GENRE_8K: Genre(
        name=GENRE_8K,
        form_types=DEFAULT_FORM_TYPES,
        document_dataset=DOCUMENT_DATASET_NAME,
        item_dataset=ITEM_DATASET_NAME,
        classified_dataset=CLASSIFICATION_DATASET_NAME,
        inlines_bodies=True,
    ),
    GENRE_6K: Genre(
        name=GENRE_6K,
        form_types=SIXK_FORM_TYPES,
        document_dataset=SIXK_DOCUMENT_DATASET_NAME,
        item_dataset=SIXK_SNIPPET_DATASET_NAME,
        classified_dataset=SIXK_SNIPPET_DATASET_NAME,
        inlines_bodies=False,
    ),
}
