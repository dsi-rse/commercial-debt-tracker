"""Tests for S3-backed SEC document acquisition."""

from __future__ import annotations

import gzip
import json
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Self

from cdt.ingest import (
    IngestConfig,
    acquire_documents,
    acquire_documents_for_date_range,
    default_failure_file,
    documents_root,
    iter_filings,
    normalize_accession_number,
    run_ingest_pipeline,
)
from cdt.storage import list_artifacts, read_dataset, read_table

EXPECTED_PARTITION_FILES = 3


class FakePaginator:
    """Small paginator fake for S3 list calls."""

    def __init__(self: Self, keys: list[str]) -> None:
        """Initialize the paginator fake."""
        self.keys = keys

    def paginate(self: Self, Bucket: str, Prefix: str) -> list[dict[str, object]]:  # noqa: N803
        """Return pages with keys matching the requested prefix."""
        del Bucket
        contents = [{"Key": key} for key in self.keys if key.startswith(Prefix)]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3Client:
    """Small S3 fake for ingestion tests."""

    def __init__(self: Self, objects: dict[tuple[str, str], bytes]) -> None:
        """Initialize the fake with bucket/key object bytes."""
        self.objects = objects
        self.downloads: list[tuple[str, str]] = []
        self.manifest_reads: list[tuple[str, str]] = []

    def get_paginator(self: Self, name: str) -> FakePaginator:
        """Return a fake list-objects paginator."""
        assert name == "list_objects_v2"
        keys = [key for bucket, key in self.objects if bucket == "sec-bucket"]
        return FakePaginator(keys)

    def get_object(self: Self, Bucket: str, Key: str) -> dict[str, BytesIO]:  # noqa: N803
        """Return fake object bytes."""
        if Key.endswith("manifest.json"):
            self.manifest_reads.append((Bucket, Key))
        else:
            self.downloads.append((Bucket, Key))
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}


def test_normalize_accession_number_strips_dashes() -> None:
    """Accession numbers are stored without SEC dashes."""
    assert normalize_accession_number("0001140361-26-006577") == "000114036126006577"


def test_acquire_documents_indexes_resources_without_downloading(
    tmp_path: Path,
) -> None:
    """Acquisition records resource URIs without fetching document bodies."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0001140361-26-006577",
                    "form_type": "8-K",
                    "filing_date": "2024-01-02",
                    "failure_reason": "",
                    "documents": [
                        {
                            "type": "EX-99",
                            "s3_key": "s3://sec-bucket/sec/2024-01-02/8-K/320193/000114036126006577/exhibit.htm",
                            "url": "https://sec.example/exhibit.htm",
                        },
                        {
                            "description": "Complete submission text file",
                            "filename": "full.txt",
                            "type": "",
                            "s3_key": "s3://sec-bucket/sec/2024-01-02/8-K/320193/000114036126006577/full.txt",
                            "url": "https://sec.example/full.txt",
                        },
                    ],
                }
            ).encode(),
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000114036126006577/full.txt",
            ): b"complete submission",
            (
                "sec-bucket",
                "sec/2024-01-03/8-K/789019/000000000024000001/manifest.json",
            ): json.dumps(
                {
                    "cik": "789019",
                    "accession_number": "0000000000-24-000001",
                    "form_type": "8-K",
                    "filing_date": "2024-01-03",
                    "failure_reason": "",
                    "documents": [
                        {
                            "type": "COMPLETE SUBMISSION TEXT FILE",
                            "s3_key": "s3://sec-bucket/sec/2024-01-03/8-K/789019/000000000024000001/full.txt",
                            "url": "https://sec.example/other.txt",
                        }
                    ],
                }
            ).encode(),
            (
                "sec-bucket",
                "sec/2024-01-04/8-K/320193/000000000024000002/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0000000000-24-000002",
                    "form_type": "8-K",
                    "filing_date": "2024-01-04",
                    "failure_reason": "api_error",
                    "documents": [],
                }
            ).encode(),
            (
                "sec-bucket",
                "sec/2023-01-02/8-K/320193/000000000023000001/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0000000000-23-000001",
                    "form_type": "8-K",
                    "filing_date": "2023-01-02",
                    "failure_reason": "",
                    "documents": [],
                }
            ).encode(),
        }
    )

    first = acquire_documents(
        "sec-bucket",
        2024,
        {"320193"},
        data_dir=tmp_path,
        s3_client=client,
    )
    second = acquire_documents(
        "sec-bucket",
        2024,
        {"320193"},
        data_dir=tmp_path,
        s3_client=client,
    )

    assert first["accession_number"].to_list() == ["000114036126006577"]
    assert first["company_name"].to_list() == [""]
    assert first["resource_uri"].to_list() == [
        "s3://sec-bucket/sec/2024-01-02/8-K/320193/000114036126006577/full.txt"
    ]
    assert first["text"].to_list() == [""]
    assert len(second) == 1
    assert client.downloads == []
    assert (
        "sec-bucket",
        "sec/2024-01-03/8-K/789019/000000000024000001/manifest.json",
    ) not in client.manifest_reads
    documents = read_dataset(documents_root(data_dir=tmp_path), columns=first.columns)
    assert documents["accession_number"].to_list() == ["000114036126006577"]
    assert documents["company_name"].to_list() == [""]
    assert (
        len(list_artifacts(documents_root(data_dir=tmp_path), suffix=".parquet")) == 1
    )


def test_acquire_documents_writes_downloads_in_batches_when_requested(
    tmp_path: Path,
) -> None:
    """Downloaded document bodies are flushed according to the batch size."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000000000024000001/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000001",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-02/8-K/320193/000000000024000001/full.txt",
            ),
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000000000024000001/full.txt",
            ): b"first",
            (
                "sec-bucket",
                "sec/2024-01-03/8-K/320193/000000000024000002/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000002",
                "8-K",
                "2024-01-03",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-03/8-K/320193/000000000024000002/full.txt",
            ),
            (
                "sec-bucket",
                "sec/2024-01-03/8-K/320193/000000000024000002/full.txt",
            ): b"second",
            (
                "sec-bucket",
                "sec/2024-01-04/8-K/320193/000000000024000003/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000003",
                "8-K",
                "2024-01-04",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-04/8-K/320193/000000000024000003/full.txt",
            ),
            (
                "sec-bucket",
                "sec/2024-01-04/8-K/320193/000000000024000003/full.txt",
            ): b"third",
        }
    )
    table = acquire_documents_for_date_range(
        "sec-bucket",
        date(2024, 1, 2),
        date(2024, 1, 4),
        {"320193"},
        data_dir=tmp_path,
        s3_client=client,
        batch_size=2,
        download=True,
    )

    assert table["accession_number"].to_list() == [
        "000000000024000001",
        "000000000024000002",
        "000000000024000003",
    ]
    assert table["text"].to_list() == ["first", "second", "third"]
    assert client.downloads == [
        ("sec-bucket", "sec/2024-01-02/8-K/320193/000000000024000001/full.txt"),
        ("sec-bucket", "sec/2024-01-03/8-K/320193/000000000024000002/full.txt"),
        ("sec-bucket", "sec/2024-01-04/8-K/320193/000000000024000003/full.txt"),
    ]
    assert (
        len(list_artifacts(documents_root(data_dir=tmp_path), suffix=".parquet"))
        == EXPECTED_PARTITION_FILES
    )


def test_acquire_documents_normalizes_bare_s3_keys(tmp_path: Path) -> None:
    """Manifest document keys without an s3:// prefix are canonicalized."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000000000024000001/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000001",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="sec/2024-01-02/8-K/320193/000000000024000001/full.txt",
            ),
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000000000024000001/full.txt",
            ): b"first",
        }
    )

    table = acquire_documents_for_date_range(
        "sec-bucket",
        date(2024, 1, 2),
        date(2024, 1, 2),
        {"320193"},
        data_dir=tmp_path,
        s3_client=client,
        download=True,
    )

    assert table["resource_uri"].to_list() == [
        "s3://sec-bucket/sec/2024-01-02/8-K/320193/000000000024000001/full.txt"
    ]
    assert client.downloads == [
        ("sec-bucket", "sec/2024-01-02/8-K/320193/000000000024000001/full.txt")
    ]


def test_acquire_documents_decompresses_gzip_downloads(tmp_path: Path) -> None:
    """Downloaded SEC document bodies should be decompressed before storage."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000000000024000001/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000001",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-02/8-K/320193/000000000024000001/full.txt",
            ),
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000000000024000001/full.txt",
            ): gzip.compress(b"Item 8.01 Other Events.\nDownloaded text.\n"),
        }
    )

    table = acquire_documents_for_date_range(
        "sec-bucket",
        date(2024, 1, 2),
        date(2024, 1, 2),
        {"320193"},
        data_dir=tmp_path,
        s3_client=client,
        download=True,
    )

    partition_path = list_artifacts(
        documents_root(data_dir=tmp_path),
        suffix=".parquet",
    )[0]
    downloaded = read_table(partition_path)

    assert table["text"].to_list() == ["Item 8.01 Other Events.\nDownloaded text.\n"]
    assert downloaded["text"].to_list() == [
        "Item 8.01 Other Events.\nDownloaded text.\n"
    ]


def test_ingest_records_missing_document_failures(tmp_path: Path) -> None:
    """Missing complete-submission documents are persisted in the failure registry."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0001140361-26-006577",
                    "form_type": "8-K",
                    "filing_date": "2024-01-02",
                    "failure_reason": "",
                    "documents": [
                        {
                            "type": "EX-99",
                            "description": "Exhibit 99",
                            "filename": "ex99.htm",
                            "s3_key": "s3://sec-bucket/sec/2024-01-02/8-K/320193/000114036126006577/ex99.htm",
                            "url": "https://sec.example/ex99.htm",
                        }
                    ],
                }
            ).encode()
        }
    )

    config = IngestConfig(
        mode="historical",
        bucket="sec-bucket",
        cik_file=tmp_path / "ciks.txt",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        data_dir=tmp_path,
        failure_file=tmp_path / "failures" / "ingest_failures.json",
    )
    first, result = run_ingest_pipeline(config, ciks={"320193"}, s3_client=client)
    second, _ = run_ingest_pipeline(config, ciks={"320193"}, s3_client=client)

    assert first.empty
    assert second.empty
    assert result.failures == 0
    assert client.manifest_reads == [
        ("sec-bucket", "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json")
    ]
    failure_json = json.loads(Path(result.failure_file).read_text(encoding="utf-8"))
    assert failure_json["entries"] == [
        ["sec-bucket", "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json"]
    ]


def test_ingest_records_download_failures(tmp_path: Path) -> None:
    """Download failures are written to the failure registry."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0001140361-26-006577",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-02/8-K/320193/000114036126006577/missing.txt",
            )
        }
    )

    _, result = run_ingest_pipeline(
        IngestConfig(
            mode="historical",
            bucket="sec-bucket",
            cik_file=tmp_path / "ciks.txt",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            data_dir=tmp_path,
            failure_file=default_failure_file(tmp_path),
            download=True,
        ),
        ciks={"320193"},
        s3_client=client,
    )

    assert result.failures == 1
    failure_json = json.loads(Path(result.failure_file).read_text(encoding="utf-8"))
    assert failure_json["entries"] == []


def test_iter_filings_yields_manifest_objects_for_form_type_list() -> None:
    """Manifest iteration unions exact form type prefixes over the date range."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0001140361-26-006577",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
            ),
            (
                "sec-bucket",
                "sec/2024-01-03/13F-HR/1000045/000100004524000001/manifest.json",
            ): _manifest_bytes(
                "1000045",
                "0001000045-24-000001",
                "13F-HR",
                "2024-01-03",
                "INFORMATION TABLE",
            ),
            (
                "sec-bucket",
                "sec/2024-01-04/8-K/320193/000114036124000001/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0001140361-24-000001",
                "8-K",
                "2024-01-04",
                "COMPLETE SUBMISSION TEXT FILE",
            ),
        }
    )

    filings = list(
        iter_filings(
            client,
            "sec-bucket",
            ["8-K", "13F-HR"],
            date(2024, 1, 2),
            date(2024, 1, 3),
        )
    )

    assert [filing.form_type for filing in filings] == ["8-K", "13F-HR"]
    assert filings[0].filing_date == date(2024, 1, 2)
    assert filings[0].documents[0].type == "COMPLETE SUBMISSION TEXT FILE"


def test_iter_filings_skips_failures_unless_requested() -> None:
    """Failure manifests are available only when explicitly requested."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0001140361-26-006577",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                failure_reason="api_error",
            )
        }
    )

    skipped = list(
        iter_filings(
            client,
            "sec-bucket",
            "8-K",
            date(2024, 1, 2),
            date(2024, 1, 2),
        )
    )
    included = list(
        iter_filings(
            client,
            "sec-bucket",
            "8-K",
            date(2024, 1, 2),
            date(2024, 1, 2),
            include_failures=True,
        )
    )

    assert skipped == []
    assert included[0].failure_reason == "api_error"


def test_iter_filings_normalizes_amended_form_prefix() -> None:
    """SEC form slashes are normalized to scraper S3 prefixes."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/10-K_A/320193/000114036126006577/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0001140361-26-006577",
                "10-K/A",
                "2024-01-02",
                "EX-21.1",
            )
        }
    )

    filings = list(
        iter_filings(
            client,
            "sec-bucket",
            "10-K/A",
            date(2024, 1, 2),
            date(2024, 1, 2),
        )
    )

    assert filings[0].form_type == "10-K/A"


def _manifest_bytes(
    cik: str,
    accession_number: str,
    form_type: str,
    filing_date: str,
    document_type: str,
    *,
    failure_reason: str = "",
    s3_key: str = "s3://sec-bucket/sec/2024-01-02/8-K/320193/000114036126006577/document.htm",
) -> bytes:
    return json.dumps(
        {
            "cik": cik,
            "accession_number": accession_number,
            "form_type": form_type,
            "filing_date": filing_date,
            "last_scraped_at": "2026-04-30T12:00:00+00:00",
            "index_url": "https://sec.example/index.htm",
            "company_name": "Example Inc.",
            "report_date": filing_date,
            "failure_reason": failure_reason,
            "documents": [
                {
                    "seq": "1",
                    "description": document_type,
                    "filename": "document.htm",
                    "type": document_type,
                    "s3_key": s3_key,
                    "url": "https://sec.example/document.htm",
                }
            ],
        }
    ).encode()


def test_document_shard_is_stable_across_processes() -> None:
    """Shard assignment must not depend on the per-process hash salt (#61)."""
    from cdt.ingest import DOCUMENT_PARTITION_SHARDS, _document_shard

    # crc32 is deterministic: pin exact values so any change to the scheme
    # (which would strand existing partitions) fails loudly.
    assert _document_shard("0001437749-26-027029") == _document_shard(
        "0001437749-26-027029"
    )
    shard = int(_document_shard("0001437749-26-027029"))
    assert 0 <= shard < DOCUMENT_PARTITION_SHARDS
    import subprocess
    import sys

    out = subprocess.run(  # noqa: S603 — spawns sys.executable with a fixed literal
        [
            sys.executable,
            "-c",
            "from cdt.ingest import _document_shard;"
            "print(_document_shard('0001437749-26-027029'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == _document_shard("0001437749-26-027029")
