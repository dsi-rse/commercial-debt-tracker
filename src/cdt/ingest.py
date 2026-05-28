"""Acquire SEC filings from scraper-managed S3 storage."""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, Self, cast
from urllib.parse import urlparse

import boto3
import pandas as pd

from cdt import settings
from cdt.database import (
    cdt_db_path,
    connect_cdt_db,
    read_document_accessions,
    read_documents,
    upsert_documents,
)
from cdt.storage import write_parquet_batch

LOGGER = logging.getLogger(__name__)
DOCUMENT_COLUMNS = ["accession_number", "cik", "url", "text", "date"]
DOCUMENT_INDEX_COLUMNS = [*DOCUMENT_COLUMNS, "resource_uri", "batch_path", "status"]
DEFAULT_BUCKET = "idi-dev-processor-s3"
DEFAULT_AWS_PROFILE = "idi-analysis"
DEFAULT_S3_PREFIX = "sec"
CDT_FORM_TYPE = "8-K"
CDT_DOCUMENT_TYPE = "COMPLETE SUBMISSION TEXT FILE"
CDT_DOCUMENT_DESCRIPTION = "COMPLETE SUBMISSION TEXT FILE"
DEFAULT_BATCH_SIZE = 100
PROGRESS_DAY_INTERVAL = 30
MIN_MANIFEST_KEY_PARTS = 5
MANIFEST_KEY_CIK_INDEX = 3
DOCUMENT_BATCH_PREFIX = "document-batch"


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


def documents_path(data_dir: Path | None = None) -> Path:
    """Return the directory for document batch artifacts."""
    return (data_dir or settings.DATA_DIR) / "documents"


def documents_db_path(data_dir: Path | None = None) -> Path:
    """Return the shared CDT SQLite database path."""
    return cdt_db_path(data_dir)


def document_batches_path(data_dir: Path | None = None) -> Path:
    """Return the directory for append-only document parquet batches."""
    return documents_path(data_dir)


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
    """Acquire matching 8-K documents and update the canonical documents table."""
    return acquire_documents_for_date_range(
        bucket,
        date(year, 1, 1),
        date(year, 12, 31),
        ciks,
        data_dir=data_dir,
        s3_client=s3_client,
        force=force,
        batch_size=batch_size,
        download=download,
    )


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
    """Acquire matching 8-K documents for a date range and update the table."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    client = s3_client or default_s3_client()
    db_path = documents_db_path(data_dir)
    conn = connect_document_db(db_path)
    normalized_ciks = _normalize_ciks(ciks)

    LOGGER.info(
        "Starting SEC document acquisition for start_date=%s end_date=%s bucket=%s",
        start_date,
        end_date,
        bucket,
    )

    existing_accessions = set() if force else read_document_accessions(conn)
    seen_accessions: set[str] = set()
    pending_index_rows: list[dict[str, object]] = []
    pending_download_rows: list[dict[str, str]] = []
    candidates_seen = 0
    skipped_existing = 0
    downloaded = 0
    batches_written = 0

    def flush_pending_rows() -> None:
        nonlocal pending_index_rows, pending_download_rows, batches_written
        if not pending_index_rows:
            return
        batch_path: str | None = None
        if download:
            batch_path = str(
                write_parquet_batch(
                    document_batches_path(data_dir),
                    DOCUMENT_BATCH_PREFIX,
                    pd.DataFrame(pending_download_rows, columns=DOCUMENT_COLUMNS),
                )
            )
        upsert_documents(
            conn,
            pending_index_rows,
            batch_path=batch_path,
            status="downloaded" if download else "indexed",
        )
        batches_written += 1
        pending_index_rows = []
        pending_download_rows = []

    for candidate in iter_document_candidates_for_date_range(
        client,
        bucket,
        start_date,
        end_date,
        normalized_ciks,
    ):
        candidates_seen += 1
        if candidate.accession_number in seen_accessions:
            continue
        seen_accessions.add(candidate.accession_number)

        if candidate.accession_number in existing_accessions:
            skipped_existing += 1
            continue

        pending_index_rows.append(
            {
                "accession_number": candidate.accession_number,
                "cik": candidate.cik,
                "url": candidate.url,
                "resource_uri": candidate.resource_uri,
                "date": candidate.date,
            }
        )
        if download:
            pending_download_rows.append(_download_candidate(client, candidate))
            downloaded += 1
        if len(pending_index_rows) >= batch_size:
            flush_pending_rows()

    flush_pending_rows()

    updated = read_documents_from_index(conn)
    LOGGER.info(
        "Document acquisition complete: candidates=%s downloaded=%s "
        "skipped_existing=%s rows=%s path=%s",
        candidates_seen,
        downloaded,
        skipped_existing,
        len(updated),
        db_path,
    )
    conn.close()
    return updated


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
) -> list[DocumentCandidate]:
    """Return manifest-backed document candidates for a date range."""
    candidates: list[DocumentCandidate] = []
    for filing in iter_filings(
        s3_client,
        bucket,
        CDT_FORM_TYPE,
        start_date,
        end_date,
        ciks=ciks,
    ):
        candidate = _candidate_from_filing(filing)
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


def default_s3_client() -> S3Client:
    """Return the default S3 client for the analysis account profile."""
    return cast(S3Client, boto3.Session(profile_name=DEFAULT_AWS_PROFILE).client("s3"))


def _candidate_from_filing(filing: ScrapedFiling) -> DocumentCandidate | None:
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
        resource_uri=document.s3_key,
        date=filing.filing_date.isoformat(),
    )


def _is_cdt_document(document: ScrapedDocument) -> bool:
    return (
        document.type.upper() == CDT_DOCUMENT_TYPE
        or document.description.upper() == CDT_DOCUMENT_DESCRIPTION
    )


def _download_candidate(
    s3_client: S3Client, candidate: DocumentCandidate
) -> dict[str, str]:
    bucket, key = parse_s3_uri(candidate.resource_uri)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    text = decode_document_bytes(body)
    return {
        "accession_number": candidate.accession_number,
        "cik": candidate.cik,
        "url": candidate.url,
        "text": text,
        "date": candidate.date,
    }


def decode_document_bytes(body: bytes) -> str:
    """Decode plain-text or gzip-compressed SEC document bytes."""
    if body.startswith(b"\x1f\x8b"):
        body = gzip.decompress(body)
    return body.decode("utf-8", errors="replace")


def connect_document_db(path: Path) -> sqlite3.Connection:
    """Connect to the shared CDT SQLite database and initialize its schema."""
    return connect_cdt_db(path)


def write_document_batch(
    conn: sqlite3.Connection,
    batches_path: Path,
    rows: list[dict[str, str]],
    *,
    force: bool,
) -> pd.DataFrame:
    """Write one append-only parquet batch and update the SQLite index."""
    del force
    if not rows:
        return read_documents_from_index(conn)

    table = pd.DataFrame(rows, columns=DOCUMENT_COLUMNS)
    batch_path = write_parquet_batch(
        batches_path,
        DOCUMENT_BATCH_PREFIX,
        table,
    )
    index_rows = [
        {
            "accession_number": str(row["accession_number"]),
            "cik": str(row["cik"]),
            "url": str(row["url"]),
            "resource_uri": str(row.get("resource_uri", row["url"])),
            "date": str(row["date"]),
        }
        for row in rows
    ]
    upsert_documents(
        conn,
        index_rows,
        batch_path=str(batch_path),
        status="downloaded",
    )
    return read_documents_from_index(conn)


def read_documents_from_index(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read indexed documents from the shared database."""
    records = read_documents(
        conn,
        statuses=("indexed", "downloaded", "itemized"),
    )
    if not records:
        return pd.DataFrame(columns=DOCUMENT_INDEX_COLUMNS)
    table = pd.DataFrame(records)
    table["text"] = ""
    return table.reindex(columns=DOCUMENT_INDEX_COLUMNS)


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
) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for day_index, day in enumerate(_days_in_range(start_date, end_date), start=1):
        if day_index == 1 or day_index % PROGRESS_DAY_INTERVAL == 0:
            LOGGER.info("Scanning S3 manifest prefixes through %s", day)
        for form_type in _normalize_form_types(form_types):
            prefix = f"{DEFAULT_S3_PREFIX}/{day.isoformat()}/{form_type}/"
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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        msg = f"Expected an s3:// URI, got {uri!r}"
        raise ValueError(msg)
    return parsed.netloc, parsed.path.lstrip("/")


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
