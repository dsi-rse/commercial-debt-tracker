"""Tests for scraper-backed acquisition of Form 6-K filings."""

from __future__ import annotations

import gzip
import json
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Self

import pytest

from cdt.ingest import (
    DOCUMENT_COLUMNS,
    SIXK_DOCUMENT_DATASET_NAME,
    SIXK_FORM_TYPES,
    IngestConfig,
    IngestFailureClassifier,
    IngestFailureType,
    ScrapedDocument,
    ScrapedFiling,
)
from cdt.ingest import documents_root as ingest_documents_root
from cdt.sixk.documents import prose_documents
from cdt.sixk.mirror import mirror_path
from cdt.sixk.scraper import (
    MalformedSubmissionError,
    acquire_scraped_sixk_documents,
    assemble_submission,
    documents_in_sequence,
)
from cdt.storage import read_dataset, read_json_artifact

BUCKET = "scraper-bucket"
HARMONY_ACCESSION = "0001628280-26-060803"
HARMONY_STORED = "000162828026060803"
VALE_ACCESSION = "0001292814-26-002379"
VALE_STORED = "000129281426002379"
EXPECTED_SIXK_ROWS = 2
EXPECTED_PROSE_DOCUMENTS = 2


def _document(document_type: str, sequence: str, filename: str, body: str) -> str:
    """Return one document the way the scraper stores it: a <DOCUMENT> block."""
    return (
        "<DOCUMENT>\n"
        f"<TYPE>{document_type}\n"
        f"<SEQUENCE>{sequence}\n"
        f"<FILENAME>{filename}\n"
        f"<DESCRIPTION>{document_type}\n"
        "<TEXT>\n"
        f"<html><body><p>{body}</p></body></html>\n"
        "</TEXT>\n"
        "</DOCUMENT>\n"
    )


HARMONY_BODY = _document("6-K", "1", "body.htm", "The Company issued notes.")
HARMONY_EXHIBIT = _document("EX-99.1", "2", "exhibit.htm", "Press release text.")
VALE_BODY = _document("6-K/A", "1", "amended.htm", "The amended report.")


class FakePaginator:
    """Small paginator fake for S3 list calls."""

    def __init__(self: Self, objects: dict[tuple[str, str], bytes]) -> None:
        """Initialize the paginator fake."""
        self.objects = objects

    def paginate(self: Self, Bucket: str, Prefix: str) -> list[dict[str, object]]:  # noqa: N803
        """Return pages with keys matching the requested bucket and prefix."""
        contents = [
            {"Key": key}
            for bucket, key in sorted(self.objects)
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3Client:
    """Small S3 fake recording which objects a run read."""

    def __init__(self: Self, objects: dict[tuple[str, str], bytes]) -> None:
        """Initialize the fake with bucket/key object bytes."""
        self.objects = objects
        self.reads: list[str] = []
        self.missing: set[tuple[str, str]] = set()

    def get_paginator(self: Self, name: str) -> FakePaginator:
        """Return a fake list-objects paginator."""
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)

    def get_object(self: Self, Bucket: str, Key: str) -> dict[str, BytesIO]:  # noqa: N803
        """Return fake object bytes, recording the read."""
        self.reads.append(Key)
        if (Bucket, Key) in self.missing:
            msg = f"NoSuchKey: {Key}"
            raise RuntimeError(msg)
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}


def _manifest(
    *,
    cik: str,
    accession_number: str,
    form_type: str,
    filing_date: str,
    documents: list[dict[str, str]],
    failure_reason: str = "",
) -> bytes:
    return json.dumps(
        {
            "cik": cik,
            "accession_number": accession_number,
            "form_type": form_type,
            "filing_date": filing_date,
            "last_scraped_at": "2026-09-15T06:16:33+00:00",
            "index_url": "https://sec.example/index.htm",
            "company_name": "Example Foreign Issuer",
            "report_date": filing_date,
            "failure_reason": failure_reason,
            "documents": documents,
        }
    ).encode()


def _document_entry(
    *,
    key: str,
    document_type: str,
    sequence: str,
    filename: str,
) -> dict[str, str]:
    return {
        "seq": sequence,
        "description": document_type,
        "filename": filename,
        "type": document_type,
        "s3_key": f"s3://{BUCKET}/{key}",
        "url": f"https://sec.example/{filename}",
    }


def _harmony_keys() -> tuple[str, str, str]:
    root = f"sec/2026-09-08/6-K/1023514/{HARMONY_STORED}"
    return f"{root}/manifest.json", f"{root}/body.htm", f"{root}/exhibit.htm"


def _vale_keys() -> tuple[str, str]:
    root = f"sec/2026-09-09/6-K_A/1292814/{VALE_STORED}"
    return f"{root}/manifest.json", f"{root}/amended.htm"


def _objects(
    *, gzip_bodies: bool = True, exhibit_first_in_manifest: bool = False
) -> dict[tuple[str, str], bytes]:
    manifest_key, body_key, exhibit_key = _harmony_keys()
    vale_manifest_key, vale_body_key = _vale_keys()

    def stored(text: str) -> bytes:
        raw = text.encode("utf-8")
        # The scraper gzips its copies; decode_document_bytes sniffs the magic.
        return gzip.compress(raw) if gzip_bodies else raw

    body_entry = _document_entry(
        key=body_key, document_type="6-K", sequence="1", filename="body.htm"
    )
    exhibit_entry = _document_entry(
        key=exhibit_key, document_type="EX-99.1", sequence="2", filename="exhibit.htm"
    )
    documents = (
        [exhibit_entry, body_entry]
        if exhibit_first_in_manifest
        else [body_entry, exhibit_entry]
    )
    return {
        (BUCKET, manifest_key): _manifest(
            cik="0001023514",
            accession_number=HARMONY_ACCESSION,
            form_type="6-K",
            filing_date="2026-09-08",
            documents=documents,
        ),
        (BUCKET, body_key): stored(HARMONY_BODY),
        (BUCKET, exhibit_key): stored(HARMONY_EXHIBIT),
        (BUCKET, vale_manifest_key): _manifest(
            cik="1292814",
            accession_number=VALE_ACCESSION,
            form_type="6-K/A",
            filing_date="2026-09-09",
            documents=[
                _document_entry(
                    key=vale_body_key,
                    document_type="6-K/A",
                    sequence="1",
                    filename="amended.htm",
                )
            ],
        ),
        (BUCKET, vale_body_key): stored(VALE_BODY),
    }


def _config(tmp_path: Path, **overrides: object) -> IngestConfig:
    defaults: dict[str, object] = {
        "mode": "historical",
        "bucket": BUCKET,
        "cik_file": Path(),
        "start_date": date(2026, 9, 8),
        "end_date": date(2026, 9, 9),
        "data_dir": tmp_path,
        "output_root": str(tmp_path),
        "form_types": SIXK_FORM_TYPES,
        "dataset_name": SIXK_DOCUMENT_DATASET_NAME,
    }
    defaults.update(overrides)
    return IngestConfig(**defaults)  # type: ignore[arg-type]


def test_acquire_writes_six_k_rows_pointing_at_assembled_submissions(
    tmp_path: Path,
) -> None:
    """A run assembles each filing's documents into one mirrored submission."""
    client = FakeS3Client(_objects())

    table, result = acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    assert table["accession_number"].to_list() == [HARMONY_STORED, VALE_STORED]
    assert table["form_type"].to_list() == ["6-K", "6-K/A"]
    # The scraper path's own provenance value, the one the 8-K path records.
    assert table["source"].to_list() == ["s3-manifest", "s3-manifest"]
    # Zero-stripped, as the matcher shards on this value.
    assert table["cik"].to_list() == ["1023514", "1292814"]
    # Bodies stay out of the partition: the row points at the mirror.
    assert table["text"].to_list() == ["", ""]
    assert result.failures == 0
    assert result.dataset_name == SIXK_DOCUMENT_DATASET_NAME

    mirror = mirror_path(
        str(tmp_path), filing_date="2026-09-08", accession_number=HARMONY_STORED
    )
    assert table.loc[0, "resource_uri"] == mirror
    assert gzip.decompress(Path(mirror).read_bytes()).decode() == (
        HARMONY_BODY + HARMONY_EXHIBIT
    )
    # The dashed accession, which is the spelling EDGAR serves the submission
    # under; the row's own dash-stripped one names a file that 404s.
    assert table.loc[0, "url"] == (
        "https://www.sec.gov/Archives/edgar/data/1023514/"
        f"{HARMONY_STORED}/{HARMONY_ACCESSION}.txt"
    )
    assert (
        len(
            read_dataset(
                ingest_documents_root(
                    str(tmp_path), dataset_name=SIXK_DOCUMENT_DATASET_NAME
                ),
                columns=DOCUMENT_COLUMNS,
            )
        )
        == EXPECTED_SIXK_ROWS
    )


def test_mirrored_submission_splits_back_into_its_prose_documents(
    tmp_path: Path,
) -> None:
    """The assembled submission is what the triage stage expects to read.

    The point of assembling rather than storing documents separately: the stage
    splits one submission into prose documents itself, and a document's index in
    that split is part of a snippet's identity.
    """
    client = FakeS3Client(_objects())

    acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    mirror = mirror_path(
        str(tmp_path), filing_date="2026-09-08", accession_number=HARMONY_STORED
    )
    documents = prose_documents(gzip.decompress(Path(mirror).read_bytes()).decode())
    assert len(documents) == EXPECTED_PROSE_DOCUMENTS
    assert [document.document_type for document in documents] == ["6-K", "EX-99.1"]
    assert "The Company issued notes." in documents[0].text
    assert "Press release text." in documents[1].text


def test_documents_are_assembled_in_sequence_order_not_manifest_order(
    tmp_path: Path,
) -> None:
    """Submission order is the manifest's seq, which document_index counts in."""
    client = FakeS3Client(_objects(exhibit_first_in_manifest=True))

    acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    mirror = mirror_path(
        str(tmp_path), filing_date="2026-09-08", accession_number=HARMONY_STORED
    )
    assert gzip.decompress(Path(mirror).read_bytes()).decode() == (
        HARMONY_BODY + HARMONY_EXHIBIT
    )


def test_plain_text_documents_are_read_too(tmp_path: Path) -> None:
    """Whether the scraper gzipped its copy is not this path's business."""
    client = FakeS3Client(_objects(gzip_bodies=False))

    table, result = acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    assert result.failures == 0
    assert len(table) == EXPECTED_SIXK_ROWS


def test_already_mirrored_filings_are_not_reassembled(tmp_path: Path) -> None:
    """The mirror is the resume ledger, so a second run reads no documents.

    This is also what makes the cutover from EDGAR free: a filing EDGAR already
    mirrored is yielded from the same path without being re-acquired here.
    """
    objects = _objects()
    first_client = FakeS3Client(objects)
    acquire_scraped_sixk_documents(_config(tmp_path), s3_client=first_client)

    second_client = FakeS3Client(objects)
    table, result = acquire_scraped_sixk_documents(
        _config(tmp_path, force=True), s3_client=second_client
    )

    # force=True re-reads manifests and re-writes rows, so the rows are proof
    # the run happened; without it every accession is skipped as existing.
    assert len(table) == EXPECTED_SIXK_ROWS
    assert result.failures == 0
    document_reads = [key for key in second_client.reads if key.endswith(".htm")]
    assert document_reads != []

    third_client = FakeS3Client(objects)
    acquire_scraped_sixk_documents(_config(tmp_path), s3_client=third_client)
    assert [key for key in third_client.reads if key.endswith(".htm")] == []


def test_ciks_filter_which_filings_are_acquired(tmp_path: Path) -> None:
    """A CIK not asked for is not acquired, zero-padded or not."""
    client = FakeS3Client(_objects())

    table, _ = acquire_scraped_sixk_documents(
        _config(tmp_path), ciks={"0001023514"}, s3_client=client
    )

    assert table["accession_number"].to_list() == [HARMONY_STORED]


def test_malformed_document_fails_the_filing_permanently(tmp_path: Path) -> None:
    """A document that is not a <DOCUMENT> block is a recorded, permanent failure.

    Assembly is only faithful because the scraper stores documents in
    dissemination format. Concatenating one that is not would hand the window
    stage prose nothing downstream could tell was different, so the filing fails
    instead.
    """
    objects = _objects()
    _, body_key, _ = _harmony_keys()
    objects[(BUCKET, body_key)] = gzip.compress(
        b"<html><body><p>bare document, no wrapper</p></body></html>"
    )
    client = FakeS3Client(objects)

    table, result = acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    assert table["accession_number"].to_list() == [VALE_STORED]
    assert result.failures == 1
    registry = read_json_artifact(result.failure_file)
    persisted = json.dumps(registry)
    assert HARMONY_STORED in persisted
    assert IngestFailureType.MALFORMED_DOCUMENT.value in persisted


def test_missing_document_object_fails_the_filing(tmp_path: Path) -> None:
    """An unreadable document object fails its filing and no other."""
    objects = _objects()
    _, body_key, _ = _harmony_keys()
    client = FakeS3Client(objects)
    client.missing.add((BUCKET, body_key))

    table, result = acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    assert table["accession_number"].to_list() == [VALE_STORED]
    assert result.failures == 1


def test_manifest_without_documents_fails_the_filing(tmp_path: Path) -> None:
    """A filing with nothing to assemble is a failure, not an empty submission."""
    objects = _objects()
    manifest_key, _, _ = _harmony_keys()
    objects[(BUCKET, manifest_key)] = _manifest(
        cik="1023514",
        accession_number=HARMONY_ACCESSION,
        form_type="6-K",
        filing_date="2026-09-08",
        documents=[],
    )
    client = FakeS3Client(objects)

    table, result = acquire_scraped_sixk_documents(_config(tmp_path), s3_client=client)

    assert table["accession_number"].to_list() == [VALE_STORED]
    assert result.failures == 1


def test_registered_failures_are_skipped_until_forced(tmp_path: Path) -> None:
    """A recorded permanent failure stops the next run re-reading the filing."""
    objects = _objects()
    _, body_key, _ = _harmony_keys()
    objects[(BUCKET, body_key)] = gzip.compress(b"no wrapper")
    first_client = FakeS3Client(objects)
    acquire_scraped_sixk_documents(_config(tmp_path), s3_client=first_client)

    second_client = FakeS3Client(objects)
    _, result = acquire_scraped_sixk_documents(
        _config(tmp_path), s3_client=second_client
    )

    manifest_key, _, _ = _harmony_keys()
    assert manifest_key not in second_client.reads
    assert result.failures == 0

    # --force retries it, which is the remedy when the scraper re-uploads a
    # filing it stored badly (#67).
    forced_client = FakeS3Client(objects)
    _, forced = acquire_scraped_sixk_documents(
        _config(tmp_path, force=True), s3_client=forced_client
    )
    assert manifest_key in forced_client.reads
    assert forced.failures == 1


def test_download_is_rejected(tmp_path: Path) -> None:
    """Bodies belong in the mirror, not in the parquet partitions."""
    with pytest.raises(ValueError, match="download=True is not supported"):
        acquire_scraped_sixk_documents(
            _config(tmp_path, download=True), s3_client=FakeS3Client({})
        )


def test_empty_form_types_is_rejected(tmp_path: Path) -> None:
    """An empty form list would scan nothing and report a clean run."""
    with pytest.raises(ValueError, match="form_types must not be empty"):
        acquire_scraped_sixk_documents(
            _config(tmp_path, form_types=()), s3_client=FakeS3Client({})
        )


def test_assemble_submission_rejects_a_document_without_its_wrapper() -> None:
    """The check is on the unit under test, not only on a run."""
    with pytest.raises(MalformedSubmissionError, match="does not begin with"):
        assemble_submission(_filing(), [HARMONY_BODY, "<html>bare</html>"])


def test_documents_with_unusable_sequence_sort_last() -> None:
    """A document whose seq cannot be read keeps the filing, at the end."""
    filing = _filing()
    assert documents_in_sequence(filing) == [
        f"s3://{BUCKET}/second.htm",
        f"s3://{BUCKET}/first.htm",
    ]


def _filing() -> ScrapedFiling:
    """Return a filing whose first manifest document has no usable sequence."""
    return ScrapedFiling(
        cik="1023514",
        accession_number=HARMONY_ACCESSION,
        form_type="6-K",
        filing_date=date(2026, 9, 8),
        last_scraped_at="",
        index_url="",
        company_name="Example Foreign Issuer",
        report_date="2026-09-08",
        failure_reason="",
        documents=(
            ScrapedDocument(
                seq="",
                description="EX-99.1",
                filename="first.htm",
                type="EX-99.1",
                s3_key=f"s3://{BUCKET}/first.htm",
                url="",
            ),
            ScrapedDocument(
                seq="1",
                description="6-K",
                filename="second.htm",
                type="6-K",
                s3_key=f"s3://{BUCKET}/second.htm",
                url="",
            ),
        ),
    )


def test_malformed_documents_are_classified_permanent() -> None:
    """Re-reading the same object returns the same bytes, so do not retry it."""
    assert (
        IngestFailureType.MALFORMED_DOCUMENT in IngestFailureClassifier().do_not_retry
    )
