"""Acquire Form 6-K filings from the scraper's bucket, the 6-K path's only source.

The scraper stores a 6-K as one object per document (body, then exhibits),
with no whole-submission object, so this source assembles the submission by
concatenating the stored ``<DOCUMENT>`` blocks in sequence order and mirrors it
(:mod:`cdt.sixk.mirror`). The result is EDGAR's complete submission minus the
``<SEC-HEADER>`` preamble; see ``docs/sixk-two-stage-triage.md`` for how that
was checked.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Self

import pandas as pd

from cdt.datasets import default_artifact_root
from cdt.ingest import (
    SIXK_FORM_TYPES,
    DocumentCandidate,
    DocumentSource,
    IngestConfig,
    IngestFailureType,
    IngestRunResult,
    S3Client,
    ScrapedFiling,
    decode_document_bytes,
    filing_from_manifest_key,
    iter_manifest_keys_for_date_range,
    normalize_accession_number,
    run_ingest_pipeline,
)
from cdt.shared import FailureRegistry, get_logger
from cdt.sixk.mirror import mirror_path
from cdt.storage import (
    artifact_exists,
    get_object_bytes,
    parse_s3_uri,
    write_bytes_artifact,
)
from cdt.storage import (
    s3_client as storage_s3_client,
)

LOGGER = get_logger(__name__)


def submission_url(cik: str, accession_number: str) -> str:
    """Return the EDGAR complete-submission text file URL for one filing.

    ``accession_number`` must be the dashed form; the dash-stripped form names
    a file that does not exist. The URL is recorded, never fetched.
    """
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"
        f"{accession_number.replace('-', '')}/{accession_number}.txt"
    )


#: The marker every stored document must begin with for assembly to be
#: faithful; checked by :func:`assemble_submission`.
DOCUMENT_MARKER = "<DOCUMENT>"


class MalformedSubmissionError(RuntimeError):
    """A stored document is not a dissemination-format ``<DOCUMENT>`` block."""


def assemble_submission(filing: ScrapedFiling, documents: list[str]) -> str:
    """Return one complete submission from a filing's stored documents.

    ``documents`` are the decoded objects in submission order, concatenated
    verbatim with no separator.

    Raises:
        MalformedSubmissionError: If a document does not begin with
            :data:`DOCUMENT_MARKER`.
    """
    for index, document in enumerate(documents):
        if not document.lstrip().startswith(DOCUMENT_MARKER):
            msg = (
                f"Document {index} of {filing.accession_number} does not begin "
                f"with {DOCUMENT_MARKER}; the scraper's copies are stored in "
                "dissemination format and assembling one that is not would "
                "change the prose the triage stage reads."
            )
            raise MalformedSubmissionError(msg)
    return "".join(documents)


def documents_in_sequence(filing: ScrapedFiling) -> list[str]:
    """Return the filing's document S3 URIs in submission order.

    Sorted by the manifest's ``seq``, the order a snippet's ``document_index``
    counts in. A document with no integer ``seq`` sorts last; one with no S3 key
    is omitted.
    """
    ordered = sorted(filing.documents, key=lambda document: _sequence(document.seq))
    return [document.s3_key for document in ordered if document.s3_key]


def _sequence(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1 << 31


@dataclass
class ScraperDocumentSource:
    """6-K candidates assembled from the scraper's per-document objects.

    Iterating mirrors each filing's assembled submission and yields its
    candidate. A filing whose mirror exists is yielded without re-reading its
    documents unless ``config.force``; a filing that fails is recorded in
    ``failure_registry`` and skipped.
    """

    config: IngestConfig
    s3_client: S3Client
    failure_registry: FailureRegistry | None = None
    ciks: set[str] | None = None
    _failures: int = field(default=0, init=False)

    @property
    def failures(self: Self) -> int:
        """Return the number of filings that could not be acquired."""
        return self._failures

    def __iter__(self: Self) -> Iterator[DocumentCandidate]:
        """Assemble, mirror and yield every matching filing in the range."""
        artifact_root = self.config.output_root or default_artifact_root(
            self.config.data_dir
        )
        indexed = 0
        mirrored = 0
        reused = 0
        for manifest_key in iter_manifest_keys_for_date_range(
            self.s3_client,
            self.config.bucket,
            self.config.form_types,
            self.config.start_date,
            self.config.end_date,
            ciks=self.ciks,
            s3_prefix=self.config.s3_prefix,
        ):
            key = (self.config.bucket, manifest_key)
            if not self.config.force and self._is_registered_failure(key):
                LOGGER.info("Skipping known ingest failure: key=%s", manifest_key)
                continue
            filing = filing_from_manifest_key(
                self.s3_client,
                self.config.bucket,
                manifest_key,
                failure_registry=self.failure_registry,
            )
            if filing is None:
                # Recorded by the shared reader; not a failure of this run.
                continue
            indexed += 1
            target = mirror_path(
                artifact_root,
                filing_date=filing.filing_date.isoformat(),
                accession_number=normalize_accession_number(filing.accession_number),
            )
            if artifact_exists(target) and not self.config.force:
                reused += 1
            else:
                if not self._mirror(filing, target, key):
                    continue
                mirrored += 1
            yield self._candidate(filing, target)
        LOGGER.info(
            "Scraper 6-K acquisition complete: filings=%s assembled=%s "
            "already_mirrored=%s failures=%s",
            indexed,
            mirrored,
            reused,
            self._failures,
        )

    def _mirror(
        self: Self, filing: ScrapedFiling, target: str, key: tuple[str, str]
    ) -> bool:
        """Assemble one submission into the mirror; False when it could not be."""
        uris = documents_in_sequence(filing)
        if not uris:
            LOGGER.warning(
                "Manifest lists no documents: accession=%s", filing.accession_number
            )
            self._record(key, IngestFailureType.DOCUMENT_NOT_FOUND)
            return False
        documents = []
        for uri in uris:
            try:
                bucket, object_key = parse_s3_uri(uri)
                documents.append(
                    decode_document_bytes(
                        get_object_bytes(self.s3_client, bucket, object_key)
                    )
                )
            except Exception:
                LOGGER.exception("Failed to read scraped 6-K document: %s", uri)
                self._record(key, IngestFailureType.DOCUMENT_DOWNLOAD_FAILED)
                return False
        try:
            submission = assemble_submission(filing, documents)
        except MalformedSubmissionError as error:
            LOGGER.error("%s", error)
            self._record(key, IngestFailureType.MALFORMED_DOCUMENT)
            return False
        write_bytes_artifact(target, gzip.compress(submission.encode("utf-8")))
        if self.config.force and self.failure_registry is not None:
            # The registered failure did not reproduce, so stop skipping it.
            self.failure_registry.discard(key)
        return True

    def _candidate(self: Self, filing: ScrapedFiling, target: str) -> DocumentCandidate:
        return DocumentCandidate(
            accession_number=normalize_accession_number(filing.accession_number),
            # Kept 10-digit padded, as the 8-K candidate records it.
            cik=filing.cik,
            company_name=filing.company_name,
            # Dashed accession: the stored dash-stripped form 404s on sec.gov.
            url=submission_url(filing.cik, filing.accession_number),
            resource_uri=target,
            date=filing.filing_date.isoformat(),
            form_type=filing.form_type,
            source=DocumentSource.S3_MANIFEST,
        )

    def _is_registered_failure(self: Self, key: tuple[str, str]) -> bool:
        return self.failure_registry is not None and key in self.failure_registry

    def _record(self: Self, key: tuple[str, str], failure: IngestFailureType) -> None:
        self._failures += 1
        if self.failure_registry is not None:
            self.failure_registry.add(key, failure)


def acquire_scraped_sixk_documents(
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    s3_client: S3Client | None = None,
    return_documents: bool = False,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Acquire 6-K filings from the scraper into the config's documents dataset.

    Runs :func:`cdt.ingest.run_ingest_pipeline` with
    :class:`ScraperDocumentSource` as the candidate source. The frame is empty
    unless ``return_documents``.

    Raises:
        ValueError: If ``config.download`` is set or ``config.form_types`` is
            empty.
    """
    if config.download:
        msg = (
            "download=True is not supported for 6-K acquisition: the assembled "
            "submission is mirrored under the artifact root and resolved from "
            "resource_uri by the stage that reads it, exactly as an 8-K body is "
            "read from the scraper's copy."
        )
        raise ValueError(msg)
    if not config.form_types:
        msg = f"form_types must not be empty; expected one of {SIXK_FORM_TYPES}"
        raise ValueError(msg)
    client = s3_client or storage_s3_client(config.aws_profile)
    return run_ingest_pipeline(
        config,
        ciks=ciks,
        s3_client=client,
        return_documents=return_documents,
        candidate_source=lambda registry: ScraperDocumentSource(
            config=config,
            s3_client=client,
            failure_registry=registry,
            ciks=ciks,
        ),
    )
