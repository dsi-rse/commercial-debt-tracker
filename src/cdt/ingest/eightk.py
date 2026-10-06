"""The 8-K candidate source: each filing's complete submission text file, from the scraper's manifests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from cdt.datasets import (
    DEFAULT_FORM_TYPES,
    DOCUMENT_DATASET_NAME,
    default_artifact_root,
)
from cdt.ingest.core import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_S3_PREFIX,
    DocumentCandidate,
    DocumentCandidateSource,
    DocumentSource,
    IngestConfig,
    IngestFailureType,
    IngestRunResult,
    ListCandidateSource,
    S3Client,
    ScrapedDocument,
    ScrapedFiling,
    _failure_key,
    _iter_manifest_keys,
    _normalize_ciks,
    _record_failure,
    filing_from_manifest_key,
    normalize_accession_number,
    normalize_s3_uri,
    run_ingest_pipeline,
)
from cdt.shared import FailureRegistry, get_logger
from cdt.storage.objects import s3_client as storage_s3_client

LOGGER = get_logger(__name__)

CDT_DOCUMENT_TYPE = "COMPLETE SUBMISSION TEXT FILE"
CDT_DOCUMENT_DESCRIPTION = "COMPLETE SUBMISSION TEXT FILE"


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
    form_types: tuple[str, ...] = DEFAULT_FORM_TYPES,
    dataset_name: str = DOCUMENT_DATASET_NAME,
) -> pd.DataFrame:
    """Acquire matching documents for a year and update document partitions."""
    documents, _ = acquire_eightk_documents(
        IngestConfig(
            mode="historical",
            bucket=bucket,
            cik_file=Path(),
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
            data_dir=data_dir,
            output_root=default_artifact_root(data_dir),
            force=force,
            batch_size=batch_size,
            download=download,
            form_types=form_types,
            dataset_name=dataset_name,
        ),
        ciks=ciks,
        s3_client=s3_client,
        return_documents=True,
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
    form_types: tuple[str, ...] = DEFAULT_FORM_TYPES,
    dataset_name: str = DOCUMENT_DATASET_NAME,
) -> pd.DataFrame:
    """Acquire matching documents for a date range and update partitions."""
    documents, _ = acquire_eightk_documents(
        IngestConfig(
            mode="historical",
            bucket=bucket,
            cik_file=Path(),
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            output_root=default_artifact_root(data_dir),
            force=force,
            batch_size=batch_size,
            download=download,
            form_types=form_types,
            dataset_name=dataset_name,
        ),
        ciks=ciks,
        s3_client=s3_client,
        return_documents=True,
    )
    return documents


def acquire_eightk_documents(
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    s3_client: S3Client | None = None,
    return_documents: bool = False,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Ingest the 8-K complete submissions the scraper's manifests list.

    Candidates come from ``iter_document_candidates_for_date_range`` over the
    config's window, form types and S3 prefix. ``config.force`` also retries
    filings the failure registry marks permanent. Returns what
    ``run_ingest_pipeline`` returns.
    """

    def manifest_source(failure_registry: FailureRegistry) -> DocumentCandidateSource:
        return ListCandidateSource(
            iter_document_candidates_for_date_range(
                s3_client or storage_s3_client(config.aws_profile),
                config.bucket,
                config.start_date,
                config.end_date,
                _normalize_ciks(ciks),
                failure_registry=failure_registry,
                s3_prefix=config.s3_prefix,
                form_types=config.form_types,
                # --force retries even permanently registered failures; new
                # failures are still recorded.
                retry_registered_failures=config.force,
            )
        )

    return run_ingest_pipeline(
        config,
        ciks=ciks,
        s3_client=s3_client,
        candidate_source=manifest_source,
        return_documents=return_documents,
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
    retry_registered_failures: bool = False,
    form_types: str | Sequence[str] = DEFAULT_FORM_TYPES,
) -> list[DocumentCandidate]:
    """Return manifest-backed document candidates for an inclusive date range.

    Args:
        s3_client: Client for the scraper bucket.
        bucket: The scraper bucket.
        start_date: First filing date scanned.
        end_date: Last filing date scanned.
        ciks: CIKs to keep; None keeps every filer.
        failure_registry: Where new failures are recorded and known ones looked
            up; None records nothing.
        s3_prefix: The scraper's key prefix.
        retry_registered_failures: Re-attempt manifests the registry marks
            failed, discarding the entry when one now succeeds.
        form_types: SEC form names ("8-K", "6-K/A").
    """
    candidates: list[DocumentCandidate] = []
    for manifest_key in _iter_manifest_keys(
        s3_client,
        bucket,
        form_types,
        start_date,
        end_date,
        ciks=_normalize_ciks(ciks),
        s3_prefix=s3_prefix,
    ):
        key = _failure_key(bucket, manifest_key)
        if (
            not retry_registered_failures
            and failure_registry is not None
            and key in failure_registry
        ):
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
            if (
                retry_registered_failures
                and failure_registry is not None
                and key in failure_registry
            ):
                # The registered failure did not reproduce; drop it.
                failure_registry.discard(key)
            candidates.append(candidate)
    return candidates


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
        company_name=filing.company_name,
        url=document.url,
        resource_uri=normalize_s3_uri(bucket, document.s3_key),
        date=filing.filing_date.isoformat(),
        # The manifest's form_type, not the key prefix, which spells "/" as "_".
        form_type=filing.form_type,
        source=DocumentSource.S3_MANIFEST,
    )


def _candidate_from_manifest_key(
    s3_client: S3Client,
    bucket: str,
    manifest_key: str,
    *,
    failure_registry: FailureRegistry | None = None,
) -> DocumentCandidate | None:
    filing = filing_from_manifest_key(
        s3_client, bucket, manifest_key, failure_registry=failure_registry
    )
    if filing is None:
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
