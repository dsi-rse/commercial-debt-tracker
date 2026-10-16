"""Dataset paths, partition naming, shard assignment and filing genres.

Prefixes are named ``*_root`` and single objects ``*_path``. Two partition
layouts exist: ``documents`` datasets are ``date=YYYY-MM-DD/shard=NNNN``, and
every dataset derived from them is ``month=YYYY-MM``
(:func:`target_month`). The matcher's outputs are ``cik_shard=NNNN``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
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

LOGGER = get_logger(__name__)

# The 6-K classify stage's output dataset. Named here, not beside its writer,
# because the extractor must name it and cannot import ``cdt.classifier.sixk``
# (which imports ``cdt.extractor``); this module is the leaf both import.
SIXK_SNIPPET_DATASET_NAME = "sixk-snippets"
# The 6-K segment stage's output: window spans, no text.
SIXK_WINDOW_DATASET_NAME = "sixk-windows"
MATCH_SHARDS = 64
PARTITION_PATTERN = re.compile(
    r"(?P<dataset>[a-z\-]+)/date=(?P<date>\d{4}-\d{2}-\d{2})/shard=(?P<shard>\d{4})/part-0000\.parquet$"
)
MONTH_PARTITION_PATTERN = re.compile(
    r"(?P<dataset>[a-z\-]+)/month=(?P<month>\d{4}-\d{2})/part-0000\.parquet$"
)


def default_artifact_root() -> str:
    """Return the default artifact root for local development: ``settings.DATA_DIR``."""
    return str(settings.DATA_DIR)


def resolve_artifact_root(
    artifact_root: ArtifactPath | None = None,
) -> str:
    """Resolve the configured artifact root to a normalized string path."""
    return normalize_artifact_path(artifact_root or default_artifact_root())


def dataset_root(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
) -> str:
    """Return the root for one canonical dataset."""
    return join_artifact_path(resolve_artifact_root(artifact_root), dataset_name)


def run_manifest_path(
    stage_name: str,
    run_id: str,
    *,
    artifact_root: ArtifactPath | None = None,
) -> str:
    """Return the path for one stage run manifest."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root),
        "runs",
        stage_name,
        f"run_id={run_id}.json",
    )


def failure_registry_path(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
) -> str:
    """Return the path for one stage failure registry."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root),
        "failures",
        stage_name,
        "failures.json",
    )


def load_row_failures(
    stage_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
) -> dict[str, dict[str, object]]:
    """Load one stage's row-level failure registry, keyed by row id."""
    path = failure_registry_path(stage_name, artifact_root=artifact_root)
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
) -> str:
    """Persist one stage's row-level failure registry.

    Unlike the completion registry, no plain run reads it to decide what to
    process. It is the work-list of rows a run could not finish: extract's for
    review, and a classify stage's for ``cdt classify --retry-failures``
    (:func:`cdt.classifier.core.retry_failed_classifications`).
    """
    return write_json_artifact(
        failure_registry_path(stage_name, artifact_root=artifact_root),
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
) -> str:
    """Return the path for one extractor full audit JSONL artifact."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root),
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
) -> str:
    """Return one canonical date/shard partition file path."""
    return join_artifact_path(
        dataset_root(dataset_name, artifact_root=artifact_root),
        f"date={partition_date}",
        f"shard={shard}",
        "part-0000.parquet",
    )


def target_month(partition_date: str) -> str:
    """Return the ``month=`` partition a row dated ``partition_date`` belongs in."""
    return str(partition_date)[: len("YYYY-MM")]


def month_partition_path(
    dataset_name: str,
    *,
    month: str,
    artifact_root: ArtifactPath | None = None,
) -> str:
    """Return one canonical month partition file path."""
    return join_artifact_path(
        dataset_root(dataset_name, artifact_root=artifact_root),
        f"month={month}",
        "part-0000.parquet",
    )


def source_partition_month(partition: dict[str, str]) -> str:
    """Return the target month of a parsed source partition of either layout."""
    if "month" in partition:
        return partition["month"]
    return target_month(partition["date"])


def cik_shard_partition_path(
    dataset_name: str,
    *,
    cik_shard: str,
    artifact_root: ArtifactPath | None = None,
) -> str:
    """Return one canonical cik-shard partition file path."""
    return join_artifact_path(
        dataset_root(dataset_name, artifact_root=artifact_root),
        f"cik_shard={cik_shard}",
        "part-0000.parquet",
    )


def match_date_shard_partition(path: ArtifactPath) -> dict[str, str] | None:
    """Match a canonical date/shard partition path, or None when non-canonical."""
    match = PARTITION_PATTERN.search(normalize_artifact_path(path))
    return match.groupdict() if match else None


def match_source_partition(path: ArtifactPath) -> dict[str, str] | None:
    """Match a canonical partition path of either layout, or None.

    Returns ``{"date", "shard"}`` for a date/shard partition and
    ``{"month"}`` for a month partition, each with ``dataset``.
    """
    normalized = normalize_artifact_path(path)
    for pattern in (PARTITION_PATTERN, MONTH_PARTITION_PATTERN):
        match = pattern.search(normalized)
        if match:
            return match.groupdict()
    return None


def is_bare_partition_path(path: str) -> bool:
    """Return whether ``path`` is exactly ``<dataset>/<partition>/part-0000.parquet``.

    True for either layout, with nothing before the dataset name.
    """
    return bool(
        PARTITION_PATTERN.fullmatch(path) or MONTH_PARTITION_PATTERN.fullmatch(path)
    )


def parse_source_partition(path: ArtifactPath) -> dict[str, str]:
    """Parse a canonical partition path of either layout; raise ValueError if neither."""
    partition = match_source_partition(path)
    if partition is None:
        normalized = normalize_artifact_path(path)
        raise ValueError(f"Unrecognized partition path: {normalized}")
    return partition


def iter_date_shard_partitions(
    dataset_name: str,
    *,
    artifact_root: ArtifactPath | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[str]:
    """List canonical date/shard partitions, optionally filtered by date window.

    ``start_date`` and ``end_date`` are inclusive; None leaves that side open.
    Raises ValueError on a non-canonical parquet file.
    """
    paths = list_artifacts(
        dataset_root(dataset_name, artifact_root=artifact_root),
        suffix=".parquet",
    )
    filtered: list[str] = []
    for path in paths:
        partition = match_date_shard_partition(path)
        if partition is None:
            # A non-canonical parquet is real data laid out wrong; skipping it
            # would run the pipeline on nothing while ingest counts its rows as
            # ingested.
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


DOCUMENT_PARTITION_SHARDS = 64


def document_shard(accession_number: str) -> str:
    """Return the ``documents`` shard an accession is stored in."""
    return shard_label(str(accession_number), DOCUMENT_PARTITION_SHARDS)


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


#: Filing genres. Each runs ingest → segment → classify (8-K: items and the
#: item classifier; 6-K: window spans and the two-stage triage); both converge
#: at extract.
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

    ``document_dataset`` is what ingest writes, ``segment_dataset`` what segment
    writes, ``item_dataset`` the rows the published ``items`` table takes from
    this genre, and ``classified_dataset`` the rows the extractor reads. For 8-K
    the segments are the items; for 6-K the published items are the classified
    snippets.
    ``inlines_bodies`` is whether ingest may store document bodies in the
    partition (``--download``); a 6-K row instead points at the mirrored
    submission, so every read does not pay for every body.
    """

    name: str
    form_types: tuple[str, ...]
    document_dataset: str
    segment_dataset: str
    item_dataset: str
    classified_dataset: str
    inlines_bodies: bool


#: Every genre, in the order a run prepares them.
GENRES: dict[str, Genre] = {
    GENRE_8K: Genre(
        name=GENRE_8K,
        form_types=DEFAULT_FORM_TYPES,
        document_dataset=DOCUMENT_DATASET_NAME,
        segment_dataset=ITEM_DATASET_NAME,
        item_dataset=ITEM_DATASET_NAME,
        classified_dataset=CLASSIFICATION_DATASET_NAME,
        inlines_bodies=True,
    ),
    GENRE_6K: Genre(
        name=GENRE_6K,
        form_types=SIXK_FORM_TYPES,
        document_dataset=SIXK_DOCUMENT_DATASET_NAME,
        segment_dataset=SIXK_WINDOW_DATASET_NAME,
        item_dataset=SIXK_SNIPPET_DATASET_NAME,
        classified_dataset=SIXK_SNIPPET_DATASET_NAME,
        inlines_bodies=False,
    ),
}
