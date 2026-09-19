"""Tests for S3-backed SEC document acquisition."""

from __future__ import annotations

import gzip
import json
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Self

import pandas as pd
import pytest

from cdt.datasets import parse_date_shard_partition
from cdt.ingest import (
    DOCUMENT_COLUMNS,
    SIXK_DOCUMENT_DATASET_NAME,
    IngestConfig,
    _document_shard,
    _partition_path,
    acquire_documents,
    acquire_documents_for_date_range,
    default_failure_file,
    default_output_root,
    documents_root,
    iter_filings,
    normalize_accession_number,
    run_ingest_pipeline,
)
from cdt.storage import list_artifacts, read_dataset, read_table, write_table

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


def test_force_reingest_repairs_pre_crc32_shard_duplicates(tmp_path: Path) -> None:
    """A copy stored under a pre-#61 salted shard must not survive a --force run."""
    import pandas as pd

    from cdt.ingest import DOCUMENT_COLUMNS, _document_shard
    from cdt.storage import write_partition_table

    accession = "000114036126006577"
    canonical_shard = _document_shard(accession)
    stale_shard = f"{(int(canonical_shard) + 1) % 64:04d}"
    write_partition_table(
        documents_root(data_dir=tmp_path),
        partition={"date": "2024-01-02", "shard": stale_shard},
        table=pd.DataFrame(
            [
                {
                    "accession_number": accession,
                    "cik": "320193",
                    "company_name": "Example Inc.",
                    "url": "https://sec.example/full.txt",
                    "text": "stale copy",
                    "date": "2024-01-02",
                    "resource_uri": None,
                }
            ],
            columns=DOCUMENT_COLUMNS,
        ),
    )
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
            )
        }
    )

    table, _ = run_ingest_pipeline(
        IngestConfig(
            mode="historical",
            bucket="sec-bucket",
            cik_file=tmp_path / "ciks.txt",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            data_dir=tmp_path,
            force=True,
        ),
        ciks={"320193"},
        s3_client=client,
    )

    assert table["accession_number"].to_list() == [accession]
    partition_files = list_artifacts(
        documents_root(data_dir=tmp_path), suffix=".parquet"
    )
    assert len(partition_files) == 1
    assert partition_files[0].endswith(
        f"date=2024-01-02/shard={canonical_shard}/part-0000.parquet"
    )
    documents = read_dataset(documents_root(data_dir=tmp_path))
    assert documents["accession_number"].to_list() == [accession]


def test_repair_document_shards_moves_rows_without_canonical_copy(
    tmp_path: Path,
) -> None:
    """A row that only exists under the wrong shard is moved, never dropped."""
    import pandas as pd

    from cdt.ingest import DOCUMENT_COLUMNS, _document_shard, repair_document_shards
    from cdt.storage import write_partition_table

    accession = "000114036126006577"
    canonical_shard = _document_shard(accession)
    stale_shard = f"{(int(canonical_shard) + 1) % 64:04d}"
    write_partition_table(
        documents_root(data_dir=tmp_path),
        partition={"date": "2024-01-02", "shard": stale_shard},
        table=pd.DataFrame(
            [
                {
                    "accession_number": accession,
                    "cik": "320193",
                    "company_name": "Example Inc.",
                    "url": "https://sec.example/full.txt",
                    "text": "only copy",
                    "date": "2024-01-02",
                    "resource_uri": None,
                }
            ],
            columns=DOCUMENT_COLUMNS,
        ),
    )

    removed = repair_document_shards(documents_root(data_dir=tmp_path))

    assert removed == 1
    documents = read_dataset(documents_root(data_dir=tmp_path))
    assert documents["accession_number"].to_list() == [accession]
    assert documents["text"].to_list() == ["only copy"]
    partition_files = list_artifacts(
        documents_root(data_dir=tmp_path), suffix=".parquet"
    )
    assert len(partition_files) == 1
    assert partition_files[0].endswith(
        f"date=2024-01-02/shard={canonical_shard}/part-0000.parquet"
    )


def test_force_retries_registered_permanent_failures(tmp_path: Path) -> None:
    """--force must be able to unpoison a filing the registry marked permanent (#67)."""
    from cdt.ingest import (
        IngestFailureClassifier,
        iter_document_candidates_for_date_range,
    )
    from cdt.shared import FailureRegistry

    manifest_key = "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json"
    client = FakeS3Client(
        {
            ("sec-bucket", manifest_key): _manifest_bytes(
                "320193",
                "0001140361-26-006577",
                "8-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
            )
        }
    )
    registry = FailureRegistry(
        str(tmp_path / "failures.json"), IngestFailureClassifier()
    )
    from cdt.ingest import IngestFailureType

    registry.add(("sec-bucket", manifest_key), IngestFailureType.DOCUMENT_NOT_FOUND)

    skipped = iter_document_candidates_for_date_range(
        client,
        "sec-bucket",
        date(2024, 1, 2),
        date(2024, 1, 2),
        failure_registry=registry,
    )
    retried = iter_document_candidates_for_date_range(
        client,
        "sec-bucket",
        date(2024, 1, 2),
        date(2024, 1, 2),
        failure_registry=registry,
        retry_registered_failures=True,
    )

    assert skipped == []
    assert [c.accession_number for c in retried] == ["000114036126006577"]
    # A successful retry unpoisons the filing: the entry is discarded so
    # normal (non-force) runs stop skipping it (#67).
    assert ("sec-bucket", manifest_key) not in registry
    registry.flush()
    persisted = json.loads((tmp_path / "failures.json").read_text(encoding="utf-8"))
    assert persisted["entries"] == []


def test_key_matches_ciks_with_multi_segment_prefix() -> None:
    """The CIK segment is found from the key's end, not a fixed index (#73)."""
    from cdt.ingest import _key_matches_ciks

    single = "sec/2024-01-02/8-K/320193/000114036126006577/manifest.json"
    multi = "edgar/8k/2024-01-02/8-K/320193/000114036126006577/manifest.json"

    assert _key_matches_ciks(single, {"320193"})
    assert _key_matches_ciks(multi, {"320193"})
    assert not _key_matches_ciks(multi, {"999999"})


def test_filing_from_manifest_pads_the_cik() -> None:
    """#153: the manifest reader is where padding enters the pipeline.

    It previously stripped leading zeros instead, and reverting it to
    `.lstrip("0")` left the suite green because nothing asserted the CIK here.
    """
    from cdt.ingest import _filing_from_manifest

    filing = _filing_from_manifest(
        {
            "cik": "320193",
            "accession_number": "0000320193-24-000001",
            "form_type": "8-K",
            "filing_date": "2024-01-02",
            "company_name": "Example Inc.",
            "documents": [],
        }
    )
    assert filing.cik == "0000320193"

    already_padded = _filing_from_manifest(
        {
            "cik": "0000707605",
            "accession_number": "0000707605-24-000001",
            "form_type": "8-K",
            "filing_date": "2024-01-02",
            "company_name": "Padded Co",
            "documents": [],
        }
    )
    assert already_padded.cik == "0000707605"


def test_ingest_routes_configured_form_types_to_their_own_dataset(
    tmp_path: Path,
) -> None:
    """A 6-K run reads 6-K prefixes and writes the 6-K documents dataset."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "sec/2024-01-02/6-K/320193/000000000024000001/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000001",
                "6-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-02/6-K/320193/000000000024000001/full.txt",
            ),
            (
                "sec-bucket",
                "sec/2024-01-03/6-K_A/320193/000000000024000002/manifest.json",
            ): _manifest_bytes(
                "320193",
                "0000000000-24-000002",
                "6-K/A",
                "2024-01-03",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-03/6-K_A/320193/000000000024000002/full.txt",
            ),
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
        }
    )

    table, result = run_ingest_pipeline(
        IngestConfig(
            mode="historical",
            bucket="sec-bucket",
            cik_file=Path(),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4),
            data_dir=tmp_path,
            output_root=str(tmp_path),
            form_types=("6-K", "6-K/A"),
            dataset_name=SIXK_DOCUMENT_DATASET_NAME,
        ),
        ciks={"320193"},
        s3_client=client,
    )

    # The 8-K manifest sits in the same bucket and date range and is not read:
    # form types select S3 prefixes, so a non-requested form costs no LIST hit.
    assert table["accession_number"].to_list() == [
        "000000000024000001",
        "000000000024000002",
    ]
    assert table["form_type"].to_list() == ["6-K", "6-K/A"]
    assert table["source"].to_list() == ["s3-manifest", "s3-manifest"]
    assert result.form_types == ("6-K", "6-K/A")
    assert result.dataset_name == SIXK_DOCUMENT_DATASET_NAME
    assert result.documents_root.endswith(SIXK_DOCUMENT_DATASET_NAME)
    assert (
        "sec-bucket",
        "sec/2024-01-04/8-K/320193/000000000024000003/manifest.json",
    ) not in client.manifest_reads

    sixk_root = documents_root(str(tmp_path), dataset_name=SIXK_DOCUMENT_DATASET_NAME)
    assert len(list_artifacts(sixk_root, suffix=".parquet")) == 2
    assert list_artifacts(documents_root(str(tmp_path)), suffix=".parquet") == []


def test_ingesting_six_k_leaves_the_eight_k_partitions_byte_identical(
    tmp_path: Path,
) -> None:
    """The genres share no partition, so a 6-K run cannot make 8-K work pending.

    Every downstream stage selects work by source-partition fingerprint (#62).
    Merging 6-K rows into the 8-K partitions would change those fingerprints and
    make the whole 8-K corpus pending again, so this asserts the stronger
    property the split dataset buys: the bytes do not move.
    """
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
                "sec/2024-01-02/6-K/789019/000000000024000002/manifest.json",
            ): _manifest_bytes(
                "789019",
                "0000000000-24-000002",
                "6-K",
                "2024-01-02",
                "COMPLETE SUBMISSION TEXT FILE",
                s3_key="s3://sec-bucket/sec/2024-01-02/6-K/789019/000000000024000002/full.txt",
            ),
        }
    )
    common = {
        "mode": "historical",
        "bucket": "sec-bucket",
        "cik_file": Path(),
        "start_date": date(2024, 1, 2),
        "end_date": date(2024, 1, 2),
        "data_dir": tmp_path,
        "output_root": str(tmp_path),
    }

    run_ingest_pipeline(
        IngestConfig(**common),
        ciks={"320193"},
        s3_client=client,
    )
    eightk_paths = list_artifacts(documents_root(str(tmp_path)), suffix=".parquet")
    before = {path: Path(path).read_bytes() for path in eightk_paths}
    assert before

    run_ingest_pipeline(
        IngestConfig(
            **common,
            form_types=("6-K",),
            dataset_name=SIXK_DOCUMENT_DATASET_NAME,
        ),
        ciks={"789019"},
        s3_client=client,
    )

    after = {
        path: Path(path).read_bytes()
        for path in list_artifacts(documents_root(str(tmp_path)), suffix=".parquet")
    }
    assert after == before
    sixk = read_dataset(
        documents_root(str(tmp_path), dataset_name=SIXK_DOCUMENT_DATASET_NAME),
        columns=DOCUMENT_COLUMNS,
    )
    assert sixk["accession_number"].to_list() == ["000000000024000002"]


def test_partitions_written_before_the_provenance_columns_stay_readable(
    tmp_path: Path,
) -> None:
    """A partition predating form_type/source reads back with them as null.

    The two columns are provenance, not keys, so the migration is a read-time
    reindex rather than a rewrite: an 8-K partition written by an earlier
    release must still load, and a null form_type is an 8-K.
    """
    legacy_columns = [
        column for column in DOCUMENT_COLUMNS if column not in {"form_type", "source"}
    ]
    legacy_path = _partition_path(
        documents_root(str(tmp_path)),
        {"date": "2023-01-02", "shard": _document_shard("000000000023000001")},
    )
    write_table(
        legacy_path,
        pd.DataFrame(
            [
                {
                    "accession_number": "000000000023000001",
                    "cik": "320193",
                    "company_name": "Example Inc.",
                    "url": "https://sec.example/legacy.txt",
                    "text": "",
                    "date": "2023-01-02",
                    "resource_uri": "s3://sec-bucket/legacy.txt",
                }
            ],
            columns=legacy_columns,
        ),
    )

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
        }
    )
    run_ingest_pipeline(
        IngestConfig(
            mode="historical",
            bucket="sec-bucket",
            cik_file=Path(),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            data_dir=tmp_path,
            output_root=str(tmp_path),
        ),
        ciks={"320193"},
        s3_client=client,
    )

    documents = read_dataset(
        documents_root(str(tmp_path)), columns=DOCUMENT_COLUMNS
    ).set_index("accession_number")
    assert documents.loc["000000000023000001", "form_type"] is None or pd.isna(
        documents.loc["000000000023000001", "form_type"]
    )
    assert documents.loc["000000000024000001", "form_type"] == "8-K"
    assert documents.loc["000000000024000001", "source"] == "s3-manifest"


def test_sixk_document_partitions_are_canonical_date_shard_partitions() -> None:
    """The 6-K dataset takes part in the ordinary partition contract.

    Every stage locates work by joining a dataset root and then reading each
    path's date and shard; the dataset segment itself is matched but never
    consumed. So this pins what the new dataset actually needs — that its
    partitions parse — rather than anything about its name.
    """
    partition = parse_date_shard_partition(
        f"/artifacts/{SIXK_DOCUMENT_DATASET_NAME}/date=2024-01-02/shard=0001/part-0000.parquet"
    )
    assert partition["date"] == "2024-01-02"
    assert partition["shard"] == "0001"


def _document_row(accession: str, filing_date: str) -> dict[str, object]:
    return {
        "accession_number": accession,
        "cik": "320193",
        "company_name": "Example Inc.",
        "url": "https://sec.example/full.txt",
        "text": "stored",
        "date": filing_date,
        "resource_uri": None,
        "form_type": "8-K",
        "source": "s3-manifest",
    }


def _store_document(data_dir: Path, accession: str, filing_date: str) -> None:
    from cdt.storage import write_partition_table

    write_partition_table(
        documents_root(data_dir=data_dir),
        partition={"date": filing_date, "shard": _document_shard(accession)},
        table=pd.DataFrame(
            [_document_row(accession, filing_date)], columns=DOCUMENT_COLUMNS
        ),
    )


def test_existing_accessions_reads_only_the_windowed_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dedup scan must not move the whole corpus to build a set of strings (#190).

    ``documents`` is ~100% the ``text`` column at 1.345 MB/row (measured on
    data/genwindow-eval-apr: 12.2 GB over 1,640 partitions), so a whole-corpus
    scan was the single most expensive thing an otherwise no-op ingest did. The
    file count read is pinned, not only the returned set: a version that read
    every partition and then filtered would return the same set and be exactly
    the bug.
    """
    from cdt import ingest

    for day, accession in (
        ("2024-01-01", "000000000024000001"),
        ("2024-01-05", "000000000024000005"),
        ("2024-01-06", "000000000024000006"),
        ("2024-01-09", "000000000024000009"),
    ):
        _store_document(tmp_path, accession, day)

    read_paths: list[str] = []
    original = ingest.read_table

    def recording_read_table(path: object, columns: object = None) -> pd.DataFrame:
        read_paths.append(str(path))
        return original(path, columns)

    monkeypatch.setattr(ingest, "read_table", recording_read_table)

    accessions = ingest._existing_accessions(
        "documents",
        output_root=default_output_root(tmp_path),
        start_date=date(2024, 1, 5),
        end_date=date(2024, 1, 6),
    )

    assert accessions == {"000000000024000005", "000000000024000006"}
    assert len(read_paths) == 2
    assert all(
        "date=2024-01-05" in path or "date=2024-01-06" in path for path in read_paths
    )


def test_existing_accessions_projects_away_the_document_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the key column is requested: the text column is the entire cost (#190)."""
    from cdt import ingest

    _store_document(tmp_path, "000000000024000005", "2024-01-05")
    requested: list[object] = []
    original = ingest.read_table

    def recording_read_table(path: object, columns: object = None) -> pd.DataFrame:
        requested.append(columns)
        return original(path, columns)

    monkeypatch.setattr(ingest, "read_table", recording_read_table)

    ingest._existing_accessions(
        "documents",
        output_root=default_output_root(tmp_path),
        start_date=date(2024, 1, 5),
        end_date=date(2024, 1, 5),
    )

    assert requested == [["accession_number"]]


def test_reingest_inside_the_window_still_skips_the_download(tmp_path: Path) -> None:
    """Windowing the dedup scan must not cost a re-download (#190).

    The window is drawn from the same dates the candidates come from — a
    candidate's ``date`` is its manifest's ``filing_date``, which is the day
    prefix the manifest was listed under — so an already-ingested accession is
    inside it.
    """
    accession = "000114036126006577"
    _store_document(tmp_path, accession, "2024-01-02")
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
                "sec/2024-01-02/8-K/320193/000114036126006577/document.htm",
            ): b"body",
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
            download=True,
        ),
        ciks={"320193"},
        s3_client=client,
    )

    assert result.skipped_existing == 1
    assert client.downloads == []
    documents = read_dataset(documents_root(data_dir=tmp_path))
    assert documents["accession_number"].to_list() == [accession]
    assert documents["text"].to_list() == ["stored"]
