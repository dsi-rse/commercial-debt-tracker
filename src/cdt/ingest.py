"""Acquire SEC 8-K documents from scraper-managed S3 storage."""

from __future__ import annotations

import json
import logging
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
FORM_TYPE = "8-K"
CIK_PART_INDEX = 2


class ReadableBody(Protocol):
    """Readable response body returned by S3."""

    def read(self: Self) -> bytes:
        """Read body bytes."""


class S3Paginator(Protocol):
    """Paginator protocol for S3 object listing."""

    def paginate(self: Self, Bucket: str, Prefix: str) -> list[dict[str, object]]:  # noqa: N803
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
) -> pd.DataFrame:
    """Acquire matching 8-K documents and update the canonical documents table."""
    client = s3_client or cast(S3Client, boto3.client("s3"))
    path = documents_path(data_dir)
    existing = read_table(path, DOCUMENT_COLUMNS)
    normalized_ciks = (
        {str(cik).lstrip("0") for cik in ciks} if ciks is not None else None
    )

    LOGGER.info("Starting SEC document acquisition for year=%s bucket=%s", year, bucket)
    candidates = list(iter_document_candidates(client, bucket, year, normalized_ciks))
    requested_accessions = {candidate.accession_number for candidate in candidates}
    if force and not existing.empty:
        existing = existing.loc[
            ~existing["accession_number"].isin(requested_accessions)
        ]

    existing_accessions = (
        set(existing["accession_number"]) if not existing.empty else set()
    )
    missing_candidates = [
        candidate
        for candidate in candidates
        if candidate.accession_number not in existing_accessions
    ]
    LOGGER.info(
        "Found %s candidate documents; downloading %s missing documents",
        len(candidates),
        len(missing_candidates),
    )

    new_rows = [
        _download_candidate(client, candidate) for candidate in missing_candidates
    ]
    new_table = pd.DataFrame(new_rows, columns=DOCUMENT_COLUMNS)
    updated = append_new_rows(
        path,
        new_table,
        ["accession_number"],
        DOCUMENT_COLUMNS,
        replace_keys=requested_accessions if force else None,
        replace_key_column="accession_number" if force else None,
    )
    LOGGER.info("Document acquisition complete: %s rows at %s", len(updated), path)
    return updated


def iter_document_candidates(
    s3_client: S3Client,
    bucket: str,
    year: int,
    ciks: set[str] | None = None,
) -> list[DocumentCandidate]:
    """Return manifest-backed document candidates for a year and optional CIKs."""
    candidates: list[DocumentCandidate] = []
    for manifest_key in iter_manifest_keys(s3_client, bucket, year):
        key_parts = manifest_key.split("/")
        key_cik = key_parts[CIK_PART_INDEX] if len(key_parts) > CIK_PART_INDEX else ""
        if ciks is not None and key_cik not in ciks:
            continue
        manifest = _read_json_object(s3_client, bucket, manifest_key)
        if manifest.get("failure_reason"):
            LOGGER.info("Skipping failed manifest %s", manifest_key)
            continue
        candidate = _candidate_from_manifest(manifest)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def iter_manifest_keys(s3_client: S3Client, bucket: str, year: int) -> list[str]:
    """List 8-K manifest keys for every day in a filing year."""
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for day in _days_in_year(year):
        prefix = f"{day.isoformat()}/{FORM_TYPE}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            contents = cast(list[dict[str, str]], page.get("Contents", []))
            keys.extend(
                obj["Key"] for obj in contents if obj["Key"].endswith("/manifest.json")
            )
    return keys


def _candidate_from_manifest(manifest: dict[str, object]) -> DocumentCandidate | None:
    documents = manifest.get("documents", [])
    if not documents:
        return None
    document = cast(dict[str, object], documents[0])
    return DocumentCandidate(
        accession_number=normalize_accession_number(str(manifest["accession_number"])),
        cik=str(manifest["cik"]).lstrip("0"),
        url=str(document.get("url", "")),
        s3_uri=str(document["s3_key"]),
        date=str(manifest["filing_date"]),
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


def _read_json_object(s3_client: S3Client, bucket: str, key: str) -> dict[str, object]:
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return cast(dict[str, object], json.loads(body.decode("utf-8")))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        msg = f"Expected an s3:// URI, got {uri!r}"
        raise ValueError(msg)
    return parsed.netloc, parsed.path.lstrip("/")


def _days_in_year(year: int) -> list[date]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days
