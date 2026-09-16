"""Acquire Form 6-K filings from the scraper's bucket, like every other form.

The scraper carries 6-K over its whole history — every filing-date partition
from 2016-01-04 onward has them — so this is the 6-K path's only source, as the
scraper's bucket is the 8-K path's only source. An earlier direct-EDGAR
acquisition existed for the window when the bucket held no 6-K at all; it was
removed once the bucket carried them, because a second way to acquire one genre
is a second failure taxonomy, a second throttling policy and a second thing to
keep true for no remaining benefit.

It cannot reuse the 8-K path's candidate scan, for one reason: what the scraper
stores per filing differs by form. An 8-K filing is one object, the complete
submission text file, and the 8-K candidate names it through ``resource_uri``.
A 6-K filing is one object *per document* (the 6-K body, then its exhibits) and
no whole-submission object exists to name, while the triage stage reads one
submission per row — it splits a filing into its prose documents itself.

So this source assembles the submission the scraper did not store, and mirrors
it under CDT's own prefix. Assembly is a concatenation and nothing more: each
stored object is already the document's dissemination-format ``<DOCUMENT>``
block, header lines included, so joining them in sequence order reproduces the
submission EDGAR itself serves, minus the ``<SEC-HEADER>`` preamble that
:func:`cdt.sixk.documents.prose_documents` discards anyway. Checked against
EDGAR on 23 real filings spanning 2016 to 2026, including a 6-K/A: every
flattened prose document came out byte-identical, which is what carries the
triage stage's measured behaviour over from the corpus it was scored on.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Self

import pandas as pd

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
    default_output_root,
    default_s3_client,
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

LOGGER = get_logger(__name__)


def submission_url(cik: str, accession_number: str) -> str:
    """Return the complete-submission text file URL for one filing.

    Not fetched — recorded. It names, publicly, the submission a row's
    assembled text *is*, which is what an 8-K row's ``url`` names for the
    object the scraper stored. ``accession_number`` is the manifest's dashed
    form; the directory segment is the same digits without dashes.
    """
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"
        f"{accession_number.replace('-', '')}/{accession_number}.txt"
    )


#: The marker every stored document begins with. Checked rather than assumed:
#: assembly is only faithful because the scraper keeps the ``<DOCUMENT>``
#: wrapper, and a source that quietly stopped doing so would produce prose the
#: window stage reads differently, with nothing downstream able to tell.
DOCUMENT_MARKER = "<DOCUMENT>"


class MalformedSubmissionError(RuntimeError):
    """A stored document is not a dissemination-format ``<DOCUMENT>`` block."""


def assemble_submission(filing: ScrapedFiling, documents: list[str]) -> str:
    """Return one complete submission from a filing's stored documents.

    ``documents`` are the decoded objects in the order they should appear.
    Concatenated verbatim — no separator, no re-encoding — because each is
    already a complete ``<DOCUMENT>`` block and the extractor later quotes this
    text as evidence.
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

    Sorted by the manifest's own ``seq``, which is the submission's document
    order and therefore the order a snippet's ``document_index`` counts in.
    A document with no usable sequence sorts last rather than failing the
    filing: its position is unknown, and dropping the filing over it would lose
    prose that is still readable.
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

    Acquisition happens during iteration: a candidate exists only once its
    submission has been mirrored, and a filing already mirrored is yielded
    without re-reading its documents. That makes the mirror the resume ledger,
    so re-running a range costs one existence check per filing.
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
        artifact_root = self.config.output_root or default_output_root(
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
                # Already recorded and counted by the shared reader; a manifest
                # the scraper marked failed is not a failure of this run.
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
        # UTF-8 because that is what decode_document_bytes produced: the stored
        # objects are decoded once here, and re-encoding to anything else would
        # change the text the extractor quotes as evidence.
        write_bytes_artifact(target, gzip.compress(submission.encode("utf-8")))
        if self.config.force and self.failure_registry is not None:
            # The registered failure did not reproduce, so stop skipping it.
            self.failure_registry.discard(key)
        return True

    def _candidate(self: Self, filing: ScrapedFiling, target: str) -> DocumentCandidate:
        return DocumentCandidate(
            accession_number=normalize_accession_number(filing.accession_number),
            # Exactly what the 8-K candidate records: the manifest reader's
            # canonical 10-digit padded form (#153). Stripping it here instead
            # would publish one issuer's CIK in two spellings across the two
            # genres. Sharding would survive that — `shard_for_cik` hashes the
            # unpadded form deliberately — but a published column that reads
            # differently per genre is the inconsistency this path exists to
            # avoid.
            cik=filing.cik,
            company_name=filing.company_name,
            # The submission this row's text is, named where it is public —
            # what an 8-K row's `url` means for the object the scraper stored.
            # The scraper's own per-document URLs cannot be one value. Built
            # from the manifest's dashed accession, the spelling sec.gov serves
            # it under; the stored, dash-stripped one names a file that 404s.
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
) -> tuple[pd.DataFrame, IngestRunResult]:
    """Acquire 6-K filings from the scraper into the config's documents dataset.

    Shares every stage of ingest except where filings come from, so the run
    manifest, accession dedup and partition layout are the 8-K path's.
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
    client = s3_client or default_s3_client(config.aws_profile)
    return run_ingest_pipeline(
        config,
        ciks=ciks,
        s3_client=client,
        candidate_source=lambda registry: ScraperDocumentSource(
            config=config,
            s3_client=client,
            failure_registry=registry,
            ciks=ciks,
        ),
    )
