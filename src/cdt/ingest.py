"""Acquire SEC filings from scraper-managed S3 storage."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self, cast

import boto3
import pandas as pd

from cdt import settings
from cdt.shared import FailureClassifier, FailureRegistry, get_logger
from cdt.storage import (
    join_artifact_path,
    normalize_artifact_path,
    parse_s3_uri,
    read_dataset,
    write_json_artifact,
    write_partition_table,
)

LOGGER = get_logger(__name__)
DOCUMENT_COLUMNS = ["accession_number", "cik", "url", "text", "date", "resource_uri"]
DEFAULT_BUCKET = "idi-dev-processor-s3"
DEFAULT_AWS_PROFILE = ""
DEFAULT_S3_PREFIX = "sec"
CDT_FORM_TYPE = "8-K"
CDT_DOCUMENT_TYPE = "COMPLETE SUBMISSION TEXT FILE"
CDT_DOCUMENT_DESCRIPTION = "COMPLETE SUBMISSION TEXT FILE"
DEFAULT_BATCH_SIZE = 100
PROGRESS_DAY_INTERVAL = 30
MIN_MANIFEST_KEY_PARTS = 5
MANIFEST_KEY_CIK_INDEX = 3
DEFAULT_OUTPUT_PREFIX = "cdt/dev"
DOCUMENT_DATASET_NAME = "documents"
RUN_DATASET_NAME = "runs"
FAILURE_DATASET_NAME = "failures"
DOCUMENT_PARTITION_SHARDS = 64


class IngestFailureType(StrEnum):
    """Permanent ingest failure types."""

    MANIFEST_READ_FAILED = "manifest_read_failed"
    INVALID_MANIFEST = "invalid_manifest"
    DOCUMENT_NOT_FOUND = "document_not_found"
    DOCUMENT_DOWNLOAD_FAILED = "document_download_failed"


class IngestFailureClassifier(FailureClassifier):
    """Treat ingest source failures as permanent by default."""

    @property
    def do_not_retry(self: Self) -> frozenset[IngestFailureType]:
        """Return non-retryable failure types."""
        return frozenset(
            {
                IngestFailureType.INVALID_MANIFEST,
                IngestFailureType.DOCUMENT_NOT_FOUND,
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


@dataclass(frozen=True)
class DocumentCandidate:
    """A manifest-backed SEC document candidate."""

    accession_number: str
    cik: str
    url: str
    resource_uri: str
    date: str


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

    @property
    def database_path(self: Self) -> str:
        """Compatibility shim for legacy callers expecting a DB path."""
        return self.run_manifest

    @property
    def documents_path(self: Self) -> str:
        """Compatibility shim for legacy callers expecting a documents directory."""
        return self.documents_root


def default_output_root(data_dir: Path | None = None) -> str:
    """Return the default local artifact root."""
    return str(data_dir or settings.DATA_DIR)


def documents_root(
    output_root: str | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the root URI for canonical document dataset partitions."""
    return join_artifact_path(
        output_root or default_output_root(data_dir), DOCUMENT_DATASET_NAME
    )


def default_failure_file(
    output_root: str | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the default ingest failure registry path."""
    root = output_root or default_output_root(data_dir)
    return join_artifact_path(root, FAILURE_DATASET_NAME, "ingest", "failures.json")


def default_run_manifest_path(
    run_id: str,
    *,
    output_root: str | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return the default run-manifest location for one ingest execution."""
    root = output_root or default_output_root(data_dir)
    return join_artifact_path(root, RUN_DATASET_NAME, "ingest", f"run_id={run_id}.json")


def normalize_accession_number(accession_number: str) -> str:
    """Normalize an SEC accession number for use as a stable key."""
    return accession_number.replace("-", "")


def acquire_documents(
    bucket: str,
    year: int,
    ciks: set[str] | None = None,
    *,
    data_dir: Path | None = None,
    s3_client: S3Client | None = None,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    download: bool = False,
) -> pd.DataFrame:
    """Acquire matching 8-K documents and update canonical document partitions."""
    documents, _ = run_ingest_pipeline(
        IngestConfig(
            mode="historical",
            bucket=bucket,
            cik_file=Path(),
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
            data_dir=data_dir,
            output_root=default_output_root(data_dir),
            force=force,
            batch_size=batch_size,
            download=download,
        ),
        ciks=ciks,
        s3_client=s3_client,
    )
    return documents


def acquire_documents_for_date_range(
    bucket: str,
    start_date: date,
    end_date: date,
    ciks: set[str] | None = None,
    *,
    data_dir: Path | None = None,
    s3_client: S3Client | None = None,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    download: bool = False,
) -> pd.DataFrame:
    """Acquire matching 8-K documents for a date range and update partitions."""
    documents, _ = run_ingest_pipeline(
        IngestConfig(
            mode="historical",
            bucket=bucket,
            cik_file=Path(),
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            output_root=default_output_root(data_dir),
            force=force,
            batch_size=batch_size,
            download=download,
        ),
        ciks=ciks,
        s3_client=s3_client,
    )
    return documents


def run_ingest_pipeline(
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    s3_client: S3Client | None = None,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Run ingest using an orchestrator-style config object."""
    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    client = s3_client or default_s3_client(config.aws_profile)
    normalized_ciks = _normalize_ciks(ciks)
    output_root = config.output_root or default_output_root(config.data_dir)
    documents_dataset_root = documents_root(output_root, data_dir=config.data_dir)
    failure_file = config.failure_file or default_failure_file(
        output_root,
        data_dir=config.data_dir,
    )
    if not str(failure_file).startswith("s3://"):
        Path(str(failure_file)).parent.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_manifest = default_run_manifest_path(
        run_id,
        output_root=output_root,
        data_dir=config.data_dir,
    )
    failure_registry = FailureRegistry(
        str(failure_file),
        IngestFailureClassifier(),
    )

    LOGGER.info(
        "Starting ingest: mode=%s bucket=%s start_date=%s end_date=%s batch_size=%s download=%s",
        config.mode,
        config.bucket,
        config.start_date,
        config.end_date,
        config.batch_size,
        config.download,
    )

    existing_accessions = (
        set() if config.force else _existing_accessions(documents_dataset_root)
    )
    seen_accessions: set[str] = set()
    pending_rows: list[dict[str, str]] = []
    candidates_seen = 0
    skipped_existing = 0
    downloaded = 0
    failures = 0
    document_partitions_written: set[str] = set()

    def flush_pending_rows() -> None:
        nonlocal pending_rows
        if not pending_rows:
            return
        written = _write_document_partitions(
            documents_dataset_root,
            pd.DataFrame(pending_rows, columns=DOCUMENT_COLUMNS),
        )
        document_partitions_written.update(written)
        pending_rows = []

    for candidate in iter_document_candidates_for_date_range(
        client,
        config.bucket,
        config.start_date,
        config.end_date,
        normalized_ciks,
        failure_registry=failure_registry,
        s3_prefix=config.s3_prefix,
    ):
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
            "url": candidate.url,
            "date": candidate.date,
            "resource_uri": candidate.resource_uri,
            "text": "",
        }
        if config.download:
            try:
                row["text"] = _download_candidate(client, candidate)
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

        pending_rows.append(row)
        if len(pending_rows) >= config.batch_size:
            flush_pending_rows()

    flush_pending_rows()
    failure_registry.flush()

    updated = read_dataset(documents_dataset_root, columns=DOCUMENT_COLUMNS)
    filtered_updated = updated.loc[
        updated["date"].between(
            config.start_date.isoformat(),
            config.end_date.isoformat(),
        )
    ].reset_index(drop=True)
    write_json_artifact(
        run_manifest,
        {
            "run_id": run_id,
            "mode": config.mode,
            "bucket": config.bucket,
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
        len(filtered_updated),
        documents_dataset_root,
    )
    return filtered_updated.reindex(columns=DOCUMENT_COLUMNS), IngestRunResult(
        mode=config.mode,
        start_date=config.start_date,
        end_date=config.end_date,
        ciks_count=len(normalized_ciks or set()),
        candidates_seen=candidates_seen,
        skipped_existing=skipped_existing,
        downloaded=downloaded,
        failures=failures,
        total_rows=len(filtered_updated),
        output_root=output_root,
        documents_root=documents_dataset_root,
        document_partitions=tuple(sorted(document_partitions_written)),
        failure_file=failure_file,
        run_manifest=run_manifest,
    )


def iter_document_candidates(
    s3_client: S3Client,
    bucket: str,
    year: int,
    ciks: set[str] | None = None,
) -> list[DocumentCandidate]:
    """Return manifest-backed document candidates for a year and optional CIKs."""
    return iter_document_candidates_for_date_range(
        s3_client,
        bucket,
        date(year, 1, 1),
        date(year, 12, 31),
        ciks,
    )


def iter_document_candidates_for_date_range(
    s3_client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    ciks: set[str] | None = None,
    *,
    failure_registry: FailureRegistry | None = None,
    s3_prefix: str = DEFAULT_S3_PREFIX,
) -> list[DocumentCandidate]:
    """Return manifest-backed document candidates for a date range."""
    candidates: list[DocumentCandidate] = []
    for manifest_key in _iter_manifest_keys(
        s3_client,
        bucket,
        CDT_FORM_TYPE,
        start_date,
        end_date,
        ciks=_normalize_ciks(ciks),
        s3_prefix=s3_prefix,
    ):
        key = _failure_key(bucket, manifest_key)
        if failure_registry is not None and key in failure_registry:
            LOGGER.info(
                "Skipping known ingest failure: bucket=%s key=%s", bucket, manifest_key
            )
            continue
        candidate = _candidate_from_manifest_key(
            s3_client,
            bucket,
            manifest_key,
            failure_registry=failure_registry,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def iter_manifest_keys(s3_client: S3Client, bucket: str, year: int) -> list[str]:
    """List 8-K manifest keys for every day in a filing year."""
    return list(
        _iter_manifest_keys(
            s3_client,
            bucket,
            CDT_FORM_TYPE,
            date(year, 1, 1),
            date(year, 12, 31),
        )
    )


def iter_filings(
    s3_client: S3Client,
    bucket: str,
    form_types: str | list[str],
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


def default_s3_client(profile_name: str = DEFAULT_AWS_PROFILE) -> S3Client:
    """Return the default S3 client for the analysis account profile."""
    session = (
        boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    )
    return cast(S3Client, session.client("s3"))


def s3_uri(bucket: str, key: str) -> str:
    """Build a canonical S3 URI from bucket and key."""
    return f"s3://{bucket}/{key.lstrip('/')}"


def normalize_s3_uri(bucket: str, key_or_uri: str) -> str:
    """Return a canonical S3 URI for a manifest-provided key or URI."""
    if key_or_uri.startswith("s3://"):
        return key_or_uri
    return s3_uri(bucket, key_or_uri)


def _candidate_from_filing(
    filing: ScrapedFiling,
    *,
    bucket: str,
) -> DocumentCandidate | None:
    document = next(
        (document for document in filing.documents if _is_cdt_document(document)),
        None,
    )
    if document is None:
        return None
    return DocumentCandidate(
        accession_number=normalize_accession_number(filing.accession_number),
        cik=filing.cik,
        url=document.url,
        resource_uri=normalize_s3_uri(bucket, document.s3_key),
        date=filing.filing_date.isoformat(),
    )


def _candidate_from_manifest_key(
    s3_client: S3Client,
    bucket: str,
    manifest_key: str,
    *,
    failure_registry: FailureRegistry | None = None,
) -> DocumentCandidate | None:
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

    candidate = _candidate_from_filing(filing, bucket=bucket)
    if candidate is None:
        LOGGER.warning("Manifest missing target CDT document: key=%s", manifest_key)
        _record_failure(
            failure_registry,
            _failure_key(bucket, manifest_key),
            IngestFailureType.DOCUMENT_NOT_FOUND,
        )
        return None
    return candidate


def _is_cdt_document(document: ScrapedDocument) -> bool:
    return (
        document.type.upper() == CDT_DOCUMENT_TYPE
        or document.description.upper() == CDT_DOCUMENT_DESCRIPTION
    )


def _download_candidate(s3_client: S3Client, candidate: DocumentCandidate) -> str:
    bucket, key = parse_s3_uri(candidate.resource_uri)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
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
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return cast(dict[str, object], json.loads(body.decode("utf-8")))


def _filing_from_manifest(manifest: dict[str, object]) -> ScrapedFiling:
    documents = tuple(
        _document_from_manifest(document)
        for document in cast(list[dict[str, object]], manifest.get("documents", []))
    )
    return ScrapedFiling(
        cik=str(manifest.get("cik", "")).lstrip("0"),
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
    form_types: str | list[str],
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


def _normalize_form_types(form_types: str | list[str]) -> tuple[str, ...]:
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
    return parts[MANIFEST_KEY_CIK_INDEX] in ciks


def _failure_key(bucket: str, key: str) -> tuple[str, str]:
    return (bucket, key)


def _failure_key_for_candidate(candidate: DocumentCandidate) -> tuple[str, str]:
    return parse_s3_uri(candidate.resource_uri)


def _days_in_year(year: int) -> list[date]:
    return list(_days_in_range(date(year, 1, 1), date(year, 12, 31)))


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


def _partition_for_row(row: dict[str, str]) -> dict[str, str]:
    accession_number = row["accession_number"]
    shard = f"{hash(accession_number) % DOCUMENT_PARTITION_SHARDS:04d}"
    return {
        "date": row["date"],
        "shard": shard,
    }


def _partition_path(dataset_root: str, partition: dict[str, str]) -> str:
    partition_root = normalize_artifact_path(dataset_root).rstrip("/")
    for key, value in partition.items():
        partition_root = join_artifact_path(partition_root, f"{key}={value}")
    return join_artifact_path(partition_root, "part-0000.parquet")


def _existing_accessions(documents_dataset_root: str) -> set[str]:
    table = read_dataset(documents_dataset_root, columns=DOCUMENT_COLUMNS)
    if table.empty or "accession_number" not in table:
        return set()
    return set(table["accession_number"].astype(str))


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
                lambda value: f"{hash(str(value)) % DOCUMENT_PARTITION_SHARDS:04d}"
            )
        ).groupby("shard", sort=True):
            partition = {"date": str(date_value), "shard": str(shard)}
            path = _partition_path(documents_dataset_root, partition)
            existing = read_dataset(
                documents_dataset_root,
                columns=DOCUMENT_COLUMNS,
                partition_filter=partition,
            )
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
