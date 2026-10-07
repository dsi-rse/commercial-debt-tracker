"""Acquire SEC filings from scraper-managed S3 storage into document partitions, for any genre."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self, cast

import pandas as pd

from cdt.datasets import (
    DEFAULT_FORM_TYPES,
    DOCUMENT_DATASET_NAME,
    dataset_root,
    default_artifact_root,
    failure_registry_path,
    iter_date_shard_partitions,
    normalize_cik,
    run_manifest_path,
    shard_label,
)
from cdt.shared import FailureClassifier, FailureRegistry, get_logger
from cdt.storage.objects import (
    get_object_bytes,
    join_artifact_path,
    normalize_artifact_path,
    parse_s3_uri,
    write_json_artifact,
)
from cdt.storage.objects import s3_client as storage_s3_client
from cdt.storage.tables import (
    count_partition_rows,
    read_partitions,
    read_table,
    write_partition_table,
)

LOGGER = get_logger(__name__)
DOCUMENT_COLUMNS = [
    "accession_number",
    "cik",
    "company_name",
    "url",
    "text",
    "date",
    "resource_uri",
    # Provenance, not keys: the SEC form ("8-K", "6-K/A", ...) and a
    # DocumentSource value.
    "form_type",
    "source",
]
# The SEC scraper's output bucket. In dev CDT writes to it too, by prefix: the
# scraper owns `sec/`, CDT owns `processors/cdt/` and `database/cdt/`.
DEFAULT_BUCKET = "idi-dev-ftm2j-shared-processor-storage"
DEFAULT_AWS_PROFILE = ""
DEFAULT_S3_PREFIX = "sec"


DEFAULT_BATCH_SIZE = 100
PROGRESS_DAY_INTERVAL = 30
# {prefix...}/{date}/{form}/{cik}/{accession}/manifest.json — the CIK is
# counted from the end so a multi-segment --s3-prefix cannot shift it.
MANIFEST_KEY_CIK_INDEX_FROM_END = -3
MIN_MANIFEST_KEY_PARTS = 5
DEFAULT_OUTPUT_PREFIX = "processors/cdt"
DOCUMENT_PARTITION_SHARDS = 64


class IngestFailureType(StrEnum):
    """Permanent ingest failure types."""

    MANIFEST_READ_FAILED = "manifest_read_failed"
    INVALID_MANIFEST = "invalid_manifest"
    DOCUMENT_NOT_FOUND = "document_not_found"
    DOCUMENT_DOWNLOAD_FAILED = "document_download_failed"
    MALFORMED_DOCUMENT = "malformed_document"


class IngestFailureClassifier(FailureClassifier):
    """Treat ingest source failures as permanent by default."""

    @property
    def do_not_retry(self: Self) -> frozenset[IngestFailureType]:
        """Return non-retryable failure types."""
        return frozenset(
            {
                IngestFailureType.INVALID_MANIFEST,
                IngestFailureType.DOCUMENT_NOT_FOUND,
                # Re-reading the same object returns the same bytes.
                IngestFailureType.MALFORMED_DOCUMENT,
            }
        )

    def classify_from_response(
        self: Self, response: dict, **kwargs: object
    ) -> IngestFailureType:
        """Satisfy the shared classifier interface."""
        del response, kwargs
        return IngestFailureType.MANIFEST_READ_FAILED


class ReadableBody(Protocol):
    """Readable response body returned by S3."""

    def read(self: Self) -> bytes:
        """Read body bytes."""


class S3Paginator(Protocol):
    """Paginator protocol for S3 object listing."""

    def paginate(self: Self, Bucket: str, Prefix: str) -> Iterable[dict[str, object]]:  # noqa: N803
        """Paginate S3 list-object responses."""


class S3Client(Protocol):
    """Subset of the S3 client API used by this module."""

    def get_paginator(self: Self, name: str) -> S3Paginator:
        """Return an S3 paginator."""

    def get_object(self: Self, Bucket: str, Key: str) -> dict[str, ReadableBody]:  # noqa: N803
        """Return an S3 object body."""


class DocumentSource(StrEnum):
    """How a document row was acquired, recorded per row in ``source``."""

    S3_MANIFEST = "s3-manifest"


@dataclass(frozen=True)
class DocumentCandidate:
    """A manifest-backed SEC document candidate."""

    accession_number: str
    cik: str
    company_name: str
    url: str
    resource_uri: str
    date: str
    form_type: str = ""
    source: str = DocumentSource.S3_MANIFEST


class DocumentCandidateSource(Protocol):
    """A source of document candidates for one ingest run.

    ``failures`` is read after iteration and added to the run's failure count:
    the candidates the source could not acquire, which never reach ingest.
    """

    def __iter__(self: Self) -> Iterator[DocumentCandidate]:
        """Yield the candidates this source acquired."""

    @property
    def failures(self: Self) -> int:
        """Return the number of candidates the source could not acquire."""


@dataclass(frozen=True)
class ListCandidateSource:
    """A source over an already-materialized candidate list."""

    candidates: list[DocumentCandidate]

    def __iter__(self: Self) -> Iterator[DocumentCandidate]:
        """Yield the listed candidates."""
        return iter(self.candidates)

    @property
    def failures(self: Self) -> int:
        """Return zero; whoever built the list recorded its failures."""
        return 0


@dataclass(frozen=True)
class ScrapedDocument:
    """A document entry from a filing manifest."""

    seq: str
    description: str
    filename: str
    type: str
    s3_key: str
    url: str


@dataclass(frozen=True)
class ScrapedFiling:
    """A scraper manifest for one SEC filing."""

    cik: str
    accession_number: str
    form_type: str
    filing_date: date
    last_scraped_at: str
    index_url: str
    company_name: str
    report_date: str
    failure_reason: str
    documents: tuple[ScrapedDocument, ...]


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for one ingest run."""

    mode: str
    bucket: str
    cik_file: Path
    start_date: date
    end_date: date
    data_dir: Path | None = None
    output_root: str | None = None
    force: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    download: bool = False
    failure_file: str | Path | None = None
    aws_profile: str = DEFAULT_AWS_PROFILE
    s3_prefix: str = DEFAULT_S3_PREFIX
    form_types: tuple[str, ...] = DEFAULT_FORM_TYPES
    # Each genre gets its own documents dataset: downstream stages select work
    # by source-partition fingerprint, so mixing forms would make one genre's
    # backfill re-pend the other's partitions.
    dataset_name: str = DOCUMENT_DATASET_NAME


@dataclass(frozen=True)
class IngestRunResult:
    """Summary of a completed ingest run."""

    mode: str
    start_date: date
    end_date: date
    ciks_count: int
    candidates_seen: int
    skipped_existing: int
    downloaded: int
    failures: int
    total_rows: int
    output_root: str
    documents_root: str
    document_partitions: tuple[str, ...]
    failure_file: str
    run_manifest: str
    form_types: tuple[str, ...] = DEFAULT_FORM_TYPES
    dataset_name: str = DOCUMENT_DATASET_NAME


def documents_root(
    output_root: str | None = None,
    *,
    data_dir: Path | None = None,
    dataset_name: str = DOCUMENT_DATASET_NAME,
) -> str:
    """Return the root URI for one canonical document dataset's partitions."""
    return dataset_root(dataset_name, artifact_root=output_root, data_dir=data_dir)


def normalize_accession_number(accession_number: str) -> str:
    """Normalize an SEC accession number for use as a stable key."""
    return accession_number.replace("-", "")


def run_ingest_pipeline(
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    s3_client: S3Client | None = None,
    candidate_source: Callable[[FailureRegistry], DocumentCandidateSource],
    return_documents: bool = False,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Ingest one date window into the config's documents dataset.

    Skips accessions already stored in the window (unless ``force``), merges
    new rows into date/shard partitions in batches, and writes a run manifest.

    Args:
        config: The run's configuration.
        ciks: CIKs to keep; None keeps every filer.
        s3_client: Client to use; None builds one from ``config.aws_profile``
            when first needed.
        candidate_source: Factory, given this run's failure registry, for the
            source of this genre's candidates (``acquire_eightk_documents`` and
            the 6-K scraper each pass their own).
        return_documents: Read back and return the window's documents. When
            False the frame is empty and only ``total_rows`` counts them.

    Returns:
        The documents frame (see ``return_documents``) and the run summary.

    Raises:
        ValueError: If ``config.batch_size`` is not positive.
    """
    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    # Built on demand: a run that never touches S3 must not need a profile.
    resolved_client: list[S3Client] = [s3_client] if s3_client is not None else []

    def client() -> S3Client:
        if not resolved_client:
            resolved_client.append(storage_s3_client(config.aws_profile))
        return resolved_client[0]

    normalized_ciks = _normalize_ciks(ciks)
    output_root = config.output_root or default_artifact_root(config.data_dir)
    documents_dataset_root = documents_root(
        output_root, data_dir=config.data_dir, dataset_name=config.dataset_name
    )
    failure_file = config.failure_file or failure_registry_path(
        "ingest", artifact_root=output_root, data_dir=config.data_dir
    )
    if not str(failure_file).startswith("s3://"):
        Path(str(failure_file)).parent.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_manifest = run_manifest_path(
        "ingest", run_id, artifact_root=output_root, data_dir=config.data_dir
    )
    failure_registry = FailureRegistry(
        str(failure_file),
        IngestFailureClassifier(),
    )

    LOGGER.info(
        "Starting ingest: mode=%s bucket=%s forms=%s dataset=%s start_date=%s "
        "end_date=%s batch_size=%s download=%s",
        config.mode,
        config.bucket,
        ",".join(config.form_types),
        config.dataset_name,
        config.start_date,
        config.end_date,
        config.batch_size,
        config.download,
    )

    existing_accessions = (
        set()
        if config.force
        else _existing_accessions(
            config.dataset_name,
            output_root=output_root,
            start_date=config.start_date,
            end_date=config.end_date,
        )
    )
    seen_accessions: set[str] = set()
    pending_rows: list[dict[str, str]] = []
    candidates_seen = 0
    skipped_existing = 0
    downloaded = 0
    failures = 0
    document_partitions_written: set[str] = set()
    flush_count = 0

    def flush_pending_rows() -> None:
        nonlocal pending_rows, flush_count
        if not pending_rows:
            return
        flush_count += 1
        written = _write_document_partitions(
            documents_dataset_root,
            pd.DataFrame(pending_rows, columns=DOCUMENT_COLUMNS),
        )
        document_partitions_written.update(written)
        LOGGER.info(
            "Ingest batch complete: batch=%s rows=%s partitions_written=%s total_candidates=%s total_downloaded=%s total_failures=%s",
            flush_count,
            len(pending_rows),
            len(written),
            candidates_seen,
            downloaded,
            failures,
        )
        pending_rows = []

    source: DocumentCandidateSource = candidate_source(failure_registry)
    for candidate in source:
        candidates_seen += 1
        if candidate.accession_number in seen_accessions:
            continue
        seen_accessions.add(candidate.accession_number)

        if candidate.accession_number in existing_accessions:
            skipped_existing += 1
            continue

        row = {
            "accession_number": candidate.accession_number,
            "cik": candidate.cik,
            "company_name": candidate.company_name,
            "url": candidate.url,
            "date": candidate.date,
            "resource_uri": candidate.resource_uri,
            "text": "",
            "form_type": candidate.form_type,
            "source": candidate.source,
        }
        if config.download:
            try:
                row["text"] = _download_candidate(client(), candidate)
            except Exception:
                LOGGER.exception(
                    "Failed to download candidate: accession=%s resource=%s",
                    candidate.accession_number,
                    candidate.resource_uri,
                )
                failure_registry.add(
                    _failure_key_for_candidate(candidate),
                    IngestFailureType.DOCUMENT_DOWNLOAD_FAILED,
                )
                failures += 1
                continue
            downloaded += 1
            if config.force:
                # The registered failure did not reproduce; drop it.
                failure_registry.discard(_failure_key_for_candidate(candidate))

        pending_rows.append(row)
        if len(pending_rows) >= config.batch_size:
            flush_pending_rows()

    flush_pending_rows()
    failures += source.failures
    failure_registry.flush()

    # Read back the window's partitions plus any this run wrote outside it. No
    # row-level date filter is needed: partitions are keyed on the row's date.
    read_paths = sorted(
        set(
            iter_date_shard_partitions(
                config.dataset_name,
                artifact_root=output_root,
                start_date=config.start_date,
                end_date=config.end_date,
            )
        )
        | document_partitions_written
    )
    # Footer row counts only, unless the caller asked for the documents.
    if return_documents:
        documents = read_partitions(read_paths, columns=DOCUMENT_COLUMNS)
        total_rows = len(documents)
    else:
        documents = pd.DataFrame(columns=DOCUMENT_COLUMNS)
        total_rows = count_partition_rows(read_paths)
    write_json_artifact(
        run_manifest,
        {
            "run_id": run_id,
            "mode": config.mode,
            "bucket": config.bucket,
            "form_types": list(config.form_types),
            "dataset_name": config.dataset_name,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "ciks_count": len(normalized_ciks or set()),
            "candidates_seen": candidates_seen,
            "skipped_existing": skipped_existing,
            "downloaded": downloaded,
            "failures": failures,
            "output_root": output_root,
            "documents_root": documents_dataset_root,
            "document_partitions": sorted(document_partitions_written),
            "failure_file": str(failure_file),
        },
    )
    LOGGER.info(
        "Document acquisition complete: candidates=%s downloaded=%s skipped_existing=%s failures=%s rows=%s documents_root=%s",
        candidates_seen,
        downloaded,
        skipped_existing,
        failures,
        total_rows,
        documents_dataset_root,
    )
    return documents.reindex(columns=DOCUMENT_COLUMNS), IngestRunResult(
        mode=config.mode,
        start_date=config.start_date,
        end_date=config.end_date,
        ciks_count=len(normalized_ciks or set()),
        candidates_seen=candidates_seen,
        skipped_existing=skipped_existing,
        downloaded=downloaded,
        failures=failures,
        total_rows=total_rows,
        output_root=output_root,
        documents_root=documents_dataset_root,
        document_partitions=tuple(sorted(document_partitions_written)),
        failure_file=failure_file,
        run_manifest=run_manifest,
        form_types=config.form_types,
        dataset_name=config.dataset_name,
    )


def iter_manifest_keys_for_date_range(
    s3_client: S3Client,
    bucket: str,
    form_types: str | Sequence[str],
    start_date: date,
    end_date: date,
    *,
    ciks: set[str] | None = None,
    s3_prefix: str = DEFAULT_S3_PREFIX,
) -> Iterator[str]:
    """Yield manifest keys for the given forms over an inclusive date range.

    Only the scan: no document selection and no failure-registry lookup.
    """
    return _iter_manifest_keys(
        s3_client,
        bucket,
        form_types,
        start_date,
        end_date,
        ciks=_normalize_ciks(ciks),
        s3_prefix=s3_prefix,
    )


def iter_filings(
    s3_client: S3Client,
    bucket: str,
    form_types: str | Sequence[str],
    start_date: date,
    end_date: date,
    *,
    include_failures: bool = False,
    ciks: set[str] | None = None,
) -> Iterator[ScrapedFiling]:
    """Yield filing manifests for exact form type prefixes over an inclusive date range."""
    for manifest_key in _iter_manifest_keys(
        s3_client,
        bucket,
        form_types,
        start_date,
        end_date,
        ciks=_normalize_ciks(ciks),
    ):
        try:
            manifest = _read_json_object(s3_client, bucket, manifest_key)
        except Exception:
            LOGGER.exception("Failed to read manifest: key=%s", manifest_key)
            raise
        filing = _filing_from_manifest(manifest)
        if filing.failure_reason and not include_failures:
            LOGGER.info("Skipping failed manifest %s", manifest_key)
            continue
        yield filing


def s3_uri(bucket: str, key: str) -> str:
    """Build a canonical S3 URI from bucket and key."""
    return f"s3://{bucket}/{key.lstrip('/')}"


def normalize_s3_uri(bucket: str, key_or_uri: str) -> str:
    """Return a canonical S3 URI for a manifest-provided key or URI."""
    if key_or_uri.startswith("s3://"):
        return key_or_uri
    return s3_uri(bucket, key_or_uri)


def filing_from_manifest_key(
    s3_client: S3Client,
    bucket: str,
    manifest_key: str,
    *,
    failure_registry: FailureRegistry | None = None,
) -> ScrapedFiling | None:
    """Read one manifest into a filing.

    Returns None for an invalid manifest (recorded in ``failure_registry`` as
    permanent when given), and for an unreadable manifest or one the scraper
    marked failed (neither recorded, so the next run retries them).
    """
    try:
        manifest = _read_json_object(s3_client, bucket, manifest_key)
    except Exception:
        LOGGER.exception("Failed to read manifest: key=%s", manifest_key)
        _record_failure(
            failure_registry,
            _failure_key(bucket, manifest_key),
            IngestFailureType.MANIFEST_READ_FAILED,
        )
        return None

    try:
        filing = _filing_from_manifest(manifest)
    except Exception:
        LOGGER.exception("Invalid manifest: key=%s", manifest_key)
        _record_failure(
            failure_registry,
            _failure_key(bucket, manifest_key),
            IngestFailureType.INVALID_MANIFEST,
        )
        return None

    if filing.failure_reason:
        LOGGER.info("Skipping failed upstream manifest %s", manifest_key)
        return None
    return filing


def _download_candidate(s3_client: S3Client, candidate: DocumentCandidate) -> str:
    bucket, key = parse_s3_uri(candidate.resource_uri)
    body = get_object_bytes(s3_client, bucket, key)
    return decode_document_bytes(body)


def _record_failure(
    failure_registry: FailureRegistry | None,
    key: tuple[str, str],
    failure_type: IngestFailureType,
) -> None:
    """Persist the failure when a registry is configured."""
    if failure_registry is None:
        return
    failure_registry.add(key, failure_type)


def decode_document_bytes(body: bytes) -> str:
    """Decode plain-text or gzip-compressed SEC document bytes."""
    if body.startswith(b"\x1f\x8b"):
        body = gzip.decompress(body)
    return body.decode("utf-8", errors="replace")


def _read_json_object(s3_client: S3Client, bucket: str, key: str) -> dict[str, object]:
    body = get_object_bytes(s3_client, bucket, key)
    return cast(dict[str, object], json.loads(body.decode("utf-8")))


def _filing_from_manifest(manifest: dict[str, object]) -> ScrapedFiling:
    documents = tuple(
        _document_from_manifest(document)
        for document in cast(list[dict[str, object]], manifest.get("documents", []))
    )
    return ScrapedFiling(
        cik=normalize_cik(manifest.get("cik", "")),
        accession_number=str(manifest.get("accession_number", "")),
        form_type=str(manifest.get("form_type", "")),
        filing_date=date.fromisoformat(str(manifest["filing_date"])),
        last_scraped_at=str(manifest.get("last_scraped_at", "")),
        index_url=str(manifest.get("index_url", "")),
        company_name=str(manifest.get("company_name", "")),
        report_date=str(manifest.get("report_date", "")),
        failure_reason=str(manifest.get("failure_reason", "")),
        documents=documents,
    )


def _document_from_manifest(document: dict[str, object]) -> ScrapedDocument:
    return ScrapedDocument(
        seq=str(document.get("seq", "")),
        description=str(document.get("description", "")),
        filename=str(document.get("filename", "")),
        type=str(document.get("type", "")),
        s3_key=str(document.get("s3_key", "")),
        url=str(document.get("url", "")),
    )


def _iter_manifest_keys(
    s3_client: S3Client,
    bucket: str,
    form_types: str | Sequence[str],
    start_date: date,
    end_date: date,
    *,
    ciks: set[str] | None = None,
    s3_prefix: str = DEFAULT_S3_PREFIX,
) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for day_index, day in enumerate(_days_in_range(start_date, end_date), start=1):
        if day_index == 1 or day_index % PROGRESS_DAY_INTERVAL == 0:
            LOGGER.info("Scanning S3 manifest prefixes through %s", day)
        for form_type in _normalize_form_types(form_types):
            prefix = f"{s3_prefix}/{day.isoformat()}/{form_type}/"
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                contents = cast(list[dict[str, str]], page.get("Contents", []))
                yield from (
                    obj["Key"]
                    for obj in contents
                    if obj["Key"].endswith("/manifest.json")
                    and _key_matches_ciks(obj["Key"], ciks)
                )


def _normalize_form_types(form_types: str | Sequence[str]) -> tuple[str, ...]:
    values = [form_types] if isinstance(form_types, str) else form_types
    return tuple(form_type.replace("/", "_") for form_type in values)


def _normalize_ciks(ciks: set[str] | None) -> set[str] | None:
    if ciks is None:
        return None
    return {str(cik).lstrip("0") for cik in ciks}


def _key_matches_ciks(key: str, ciks: set[str] | None) -> bool:
    if ciks is None:
        return True
    parts = key.split("/")
    if len(parts) < MIN_MANIFEST_KEY_PARTS:
        return False
    return parts[MANIFEST_KEY_CIK_INDEX_FROM_END] in ciks


def _failure_key(bucket: str, key: str) -> tuple[str, str]:
    return (bucket, key)


def _failure_key_for_candidate(candidate: DocumentCandidate) -> tuple[str, str]:
    return parse_s3_uri(candidate.resource_uri)


def _days_in_range(start: date, end: date) -> Iterator[date]:
    if end < start:
        msg = f"end_date {end.isoformat()} is before start_date {start.isoformat()}"
        raise ValueError(msg)
    current = start
    while current <= end:
        yield current
        current = current.fromordinal(current.toordinal() + 1)


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _document_shard(accession_number: str) -> str:
    """Return the document shard for one accession, stable across processes.

    Per-partition dedup relies on an accession always landing in the same shard.
    """
    return shard_label(accession_number, DOCUMENT_PARTITION_SHARDS)


def _partition_path(dataset_root: str, partition: dict[str, str]) -> str:
    partition_root = normalize_artifact_path(dataset_root).rstrip("/")
    for key, value in partition.items():
        partition_root = join_artifact_path(partition_root, f"{key}={value}")
    return join_artifact_path(partition_root, "part-0000.parquet")


def _existing_accessions(
    dataset_name: str,
    *,
    output_root: str,
    start_date: date,
    end_date: date,
) -> set[str]:
    """Return already-ingested accessions stored under this run's date window.

    A skip-list optimization, not the uniqueness guarantee: that is the
    per-partition dedup in ``_write_document_partitions``. See
    docs/decisions/pipeline-ingest-and-publish.md for why the window suffices.
    """
    paths = list(
        iter_date_shard_partitions(
            dataset_name,
            artifact_root=output_root,
            start_date=start_date,
            end_date=end_date,
        )
    )
    # One parallel scan, not a read per partition (measured in the decisions doc).
    table = read_partitions(paths, columns=["accession_number"])
    if table.empty or "accession_number" not in table:
        return set()
    return set(table["accession_number"].dropna().astype(str))


def _write_document_partitions(
    documents_dataset_root: str,
    table: pd.DataFrame,
) -> set[str]:
    written_paths: set[str] = set()
    if table.empty:
        return written_paths

    grouped = table.groupby("date", sort=True)
    for date_value, date_group in grouped:
        for shard, shard_group in date_group.assign(
            shard=date_group["accession_number"].map(
                lambda value: _document_shard(str(value))
            )
        ).groupby("shard", sort=True):
            partition = {"date": str(date_value), "shard": str(shard)}
            path = _partition_path(documents_dataset_root, partition)
            # Read the one known file directly; listing would cost a LIST per group.
            existing = read_table(path, columns=DOCUMENT_COLUMNS)
            merged = pd.concat(
                [
                    existing.reindex(columns=DOCUMENT_COLUMNS),
                    shard_group.drop(columns=["shard"]).reindex(
                        columns=DOCUMENT_COLUMNS
                    ),
                ],
                ignore_index=True,
            )
            merged = merged.drop_duplicates(
                subset=["accession_number"],
                keep="last",
            ).sort_values(
                by=["date", "accession_number"],
                kind="stable",
            )
            write_partition_table(
                documents_dataset_root,
                partition=partition,
                table=merged.reindex(columns=DOCUMENT_COLUMNS),
            )
            written_paths.add(path)
    return written_paths
