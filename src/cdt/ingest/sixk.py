"""Acquire Form 6-K filings from the scraper's bucket, the 6-K path's only source.

The scraper stores a 6-K as one object per document (body, then exhibits),
with no whole-submission object, so this source assembles the submission by
concatenating the stored ``<DOCUMENT>`` blocks in sequence order and mirrors it
(:mod:`cdt.ingest.mirror`). The result is EDGAR's complete submission minus the
``<SEC-HEADER>`` preamble and minus any binary attachment the scraper stored
as the raw file rather than as a ``<DOCUMENT>`` block (a courtesy PDF, an
image), which no downstream stage reads; see ``docs/sixk-two-stage-triage.md``
for how the assembly was checked.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Self

import pandas as pd

from cdt.datasets import SIXK_FORM_TYPES, default_artifact_root
from cdt.ingest.core import (
    DocumentCandidate,
    DocumentSource,
    IngestConfig,
    IngestFailures,
    IngestFailureType,
    IngestRunResult,
    S3Client,
    ScrapedDocument,
    ScrapedFiling,
    decode_document_bytes,
    filing_from_manifest_key,
    iter_manifest_keys_for_date_range,
    normalize_accession_number,
    run_ingest_pipeline,
)
from cdt.ingest.mirror import mirror_path
from cdt.shared import get_logger
from cdt.storage.objects import (
    artifact_exists,
    get_object_bytes,
    parse_s3_uri,
    write_bytes_artifact,
)
from cdt.storage.objects import s3_client as storage_s3_client

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


#: The marker a stored document begins with when it is a dissemination-format
#: block; :func:`assemble_submission` keeps exactly those documents.
DOCUMENT_MARKER = "<DOCUMENT>"


#: Filename suffixes of the documents the scraper stores as ``<DOCUMENT>``
#: blocks. One of these stored bare is a change in how the source stores text,
#: not an attachment; :func:`assemble_submission` fails the filing on it.
TEXT_DOCUMENT_SUFFIXES = frozenset({".htm", ".html", ".txt"})


class MalformedSubmissionError(RuntimeError):
    """A text document is stored without its ``<DOCUMENT>`` wrapper, or no document has one."""


def is_dissemination_document(document: str) -> bool:
    """Return whether a decoded stored object is a ``<DOCUMENT>`` block."""
    return document.lstrip().startswith(DOCUMENT_MARKER)


def names_text_document(document: ScrapedDocument) -> bool:
    """Return whether the manifest names ``document`` as a text document.

    By filename suffix (:data:`TEXT_DOCUMENT_SUFFIXES`), falling back to the S3
    key when the manifest carries no filename. The manifest's ``type`` cannot
    tell: a courtesy PDF and the body it accompanies are both typed ``6-K``.
    """
    name = document.filename or document.s3_key
    return PurePosixPath(name).suffix.lower() in TEXT_DOCUMENT_SUFFIXES


def assemble_submission(
    filing: ScrapedFiling, documents: list[tuple[ScrapedDocument, str]]
) -> str:
    """Return one complete submission from a filing's stored documents.

    ``documents`` pairs each manifest entry with its decoded object, in
    submission order. The ``<DOCUMENT>`` blocks are concatenated verbatim with
    no separator. An object that is not a block and is not named as a text
    document is a binary attachment (a courtesy PDF, an image) no stage reads,
    so it is left out. The blocks :func:`cdt.segmenter.sixk.prose_documents`
    keeps are what a snippet's ``document_index`` counts.

    Raises:
        MalformedSubmissionError: If an object named as a text document is not
            a ``<DOCUMENT>`` block, since assembling it would change the prose
            the triage stage reads, or if no object is a block.
    """
    kept: list[str] = []
    left_out: list[str] = []
    for entry, text in documents:
        if is_dissemination_document(text):
            kept.append(text)
        elif names_text_document(entry):
            msg = (
                f"Document {entry.filename!r} (seq {entry.seq}) of "
                f"{filing.accession_number} is named as text but does not begin "
                f"with {DOCUMENT_MARKER}; the scraper stores text documents in "
                "dissemination format and assembling one that is not would "
                "change the prose the triage stage reads."
            )
            raise MalformedSubmissionError(msg)
        else:
            left_out.append(entry.filename or entry.s3_key)
    if not kept:
        msg = (
            f"None of the {len(documents)} stored documents of "
            f"{filing.accession_number} begins with {DOCUMENT_MARKER}; there is "
            "no dissemination-format text to assemble."
        )
        raise MalformedSubmissionError(msg)
    if left_out:
        LOGGER.info(
            "Left binary attachments out of the assembly: accession=%s "
            "left_out=%s total=%d",
            filing.accession_number,
            ",".join(left_out),
            len(documents),
        )
    return "".join(kept)


def documents_in_sequence(filing: ScrapedFiling) -> list[ScrapedDocument]:
    """Return the filing's manifest documents that have an S3 key, in submission order.

    Sorted by the manifest's ``seq``, the order a snippet's ``document_index``
    counts in. A document with no integer ``seq`` sorts last; one with no S3 key
    is omitted.
    """
    ordered = sorted(filing.documents, key=lambda document: _sequence(document.seq))
    return [document for document in ordered if document.s3_key]


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
    ``failures`` and skipped.
    """

    config: IngestConfig
    s3_client: S3Client
    failures: IngestFailures = field(default_factory=IngestFailures)
    ciks: set[str] | None = None
    #: Called per scanned day and per manifest, to extend the writer lease.
    renew: Callable[[], None] | None = None

    def __iter__(self: Self) -> Iterator[DocumentCandidate]:
        """Assemble, mirror and yield every matching filing in the range."""
        artifact_root = self.config.output_root or default_artifact_root()
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
            renew=self.renew,
        ):
            if self.renew is not None:
                self.renew()
            key = (self.config.bucket, manifest_key)
            if self.failures.is_skipped(key):
                LOGGER.info("Skipping known ingest failure: key=%s", manifest_key)
                continue
            filing = filing_from_manifest_key(
                self.s3_client,
                self.config.bucket,
                manifest_key,
                failures=self.failures,
            )
            if filing is None:
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
            self.failures.count,
        )

    def _mirror(
        self: Self, filing: ScrapedFiling, target: str, key: tuple[str, str]
    ) -> bool:
        """Assemble one submission into the mirror; False when it could not be."""
        entries = documents_in_sequence(filing)
        if not entries:
            LOGGER.warning(
                "Manifest lists no documents: accession=%s", filing.accession_number
            )
            self.failures.record(key, IngestFailureType.DOCUMENT_NOT_FOUND)
            return False
        documents: list[tuple[ScrapedDocument, str]] = []
        for entry in entries:
            try:
                bucket, object_key = parse_s3_uri(entry.s3_key)
                text = decode_document_bytes(
                    get_object_bytes(self.s3_client, bucket, object_key)
                )
                documents.append((entry, text))
            except Exception:
                LOGGER.exception(
                    "Failed to read scraped 6-K document: %s", entry.s3_key
                )
                self.failures.record(key, IngestFailureType.DOCUMENT_DOWNLOAD_FAILED)
                return False
        try:
            submission = assemble_submission(filing, documents)
        except MalformedSubmissionError as error:
            LOGGER.error("%s", error)
            self.failures.record(key, IngestFailureType.MALFORMED_DOCUMENT)
            return False
        write_bytes_artifact(target, gzip.compress(submission.encode("utf-8")))
        self.failures.clear(key)
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


def acquire_scraped_sixk_documents(
    config: IngestConfig,
    *,
    ciks: set[str] | None = None,
    s3_client: S3Client | None = None,
    return_documents: bool = False,
    renew: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Acquire 6-K filings from the scraper into the config's documents dataset.

    Runs :func:`cdt.ingest.core.run_ingest_pipeline` with
    :class:`ScraperDocumentSource` as the candidate source. The frame is empty
    unless ``return_documents``. ``renew`` is called per manifest and per
    candidate (pass a :func:`cdt.lease.throttled` hook).

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
        candidate_source=lambda failures: ScraperDocumentSource(
            config=config,
            s3_client=client,
            failures=failures,
            ciks=ciks,
            renew=renew,
        ),
        renew=renew,
    )
