"""Acquire SEC filings from scraper-managed S3 storage."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, Self, cast
from urllib.parse import urlparse

import boto3
import pandas as pd

from cdt import settings
from cdt.storage import append_new_rows, read_table

LOGGER = logging.getLogger(__name__)
DOCUMENT_COLUMNS = ["accession_number", "cik", "url", "text", "date"]
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
    s3_uri: str
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
    """Return the canonical documents table path."""
    return (data_dir or settings.DATA_DIR) / "documents" / "documents.parquet"


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
) -> pd.DataFrame:
    """Acquire matching 8-K documents for a date range and update the table."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    client = s3_client or default_s3_client()
    path = documents_path(data_dir)
    existing = read_table(path, DOCUMENT_COLUMNS)
    normalized_ciks = _normalize_ciks(ciks)

    LOGGER.info(
        "Starting SEC document acquisition for start_date=%s end_date=%s bucket=%s",
        start_date,
        end_date,
        bucket,
    )

    existing_accessions = (
        set(existing["accession_number"]) if not existing.empty and not force else set()
    )
    seen_accessions: set[str] = set()
    pending_rows: list[dict[str, str]] = []
    candidates_seen = 0
    skipped_existing = 0
    downloaded = 0
    batches_written = 0

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

        pending_rows.append(_download_candidate(client, candidate))
        downloaded += 1
        if len(pending_rows) >= batch_size:
            batches_written += 1
            _append_document_batch(path, pending_rows, force=force)
            LOGGER.info(
                "Wrote batch %s: downloaded=%s candidates=%s skipped_existing=%s",
                batches_written,
                downloaded,
                candidates_seen,
                skipped_existing,
            )
            pending_rows = []

    if pending_rows:
        batches_written += 1
        _append_document_batch(path, pending_rows, force=force)
        LOGGER.info(
            "Wrote final batch %s: downloaded=%s candidates=%s skipped_existing=%s",
            batches_written,
            downloaded,
            candidates_seen,
            skipped_existing,
        )

    if not path.exists():
        _append_document_batch(path, [], force=force)

    updated = read_table(path, DOCUMENT_COLUMNS)
    LOGGER.info(
        "Document acquisition complete: candidates=%s downloaded=%s "
        "skipped_existing=%s rows=%s path=%s",
        candidates_seen,
        downloaded,
        skipped_existing,
        len(updated),
        path,
    )
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
        manifest = _read_json_object(s3_client, bucket, manifest_key)
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
        s3_uri=document.s3_key,
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
    bucket, key = parse_s3_uri(candidate.s3_uri)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    text = body.decode("utf-8", errors="replace")
    return {
        "accession_number": candidate.accession_number,
        "cik": candidate.cik,
        "url": candidate.url,
        "text": text,
        "date": candidate.date,
    }


def _append_document_batch(
    path: Path,
    rows: list[dict[str, str]],
    *,
    force: bool,
) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=DOCUMENT_COLUMNS)
    replace_keys = (
        {row["accession_number"] for row in rows}
        if force and rows
        else None
    )
    return append_new_rows(
        path,
        table,
        ["accession_number"],
        DOCUMENT_COLUMNS,
        replace_keys=replace_keys,
        replace_key_column="accession_number" if replace_keys is not None else None,
    )


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
        current += timedelta(days=1)
