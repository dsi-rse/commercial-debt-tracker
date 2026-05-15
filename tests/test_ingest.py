"""Tests for S3-backed SEC document acquisition."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Self

from cdt.ingest import acquire_documents, normalize_accession_number


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

    def get_paginator(self: Self, name: str) -> FakePaginator:
        """Return a fake list-objects paginator."""
        assert name == "list_objects_v2"
        keys = [key for bucket, key in self.objects if bucket == "sec-bucket"]
        return FakePaginator(keys)

    def get_object(self: Self, Bucket: str, Key: str) -> dict[str, BytesIO]:  # noqa: N803
        """Return fake object bytes."""
        if not Key.endswith("manifest.json"):
            self.downloads.append((Bucket, Key))
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}


def test_normalize_accession_number_strips_dashes() -> None:
    """Accession numbers are stored without SEC dashes."""
    assert normalize_accession_number("0001140361-26-006577") == "000114036126006577"


def test_acquire_documents_filters_and_skips_existing(tmp_path: Path) -> None:
    """Acquisition filters manifests and downloads only missing documents."""
    client = FakeS3Client(
        {
            (
                "sec-bucket",
                "2024-01-02/8-K/320193/000114036126006577/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0001140361-26-006577",
                    "filing_date": "2024-01-02",
                    "failure_reason": "",
                    "documents": [
                        {
                            "s3_key": "s3://sec-bucket/2024-01-02/8-K/320193/000114036126006577/full.txt",
                            "url": "https://sec.example/full.txt",
                        }
                    ],
                }
            ).encode(),
            (
                "sec-bucket",
                "2024-01-02/8-K/320193/000114036126006577/full.txt",
            ): b"complete submission",
            (
                "sec-bucket",
                "2024-01-03/8-K/789019/000000000024000001/manifest.json",
            ): json.dumps(
                {
                    "cik": "789019",
                    "accession_number": "0000000000-24-000001",
                    "filing_date": "2024-01-03",
                    "failure_reason": "",
                    "documents": [
                        {
                            "s3_key": "s3://sec-bucket/2024-01-03/8-K/789019/000000000024000001/full.txt",
                            "url": "https://sec.example/other.txt",
                        }
                    ],
                }
            ).encode(),
            (
                "sec-bucket",
                "2024-01-04/8-K/320193/000000000024000002/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0000000000-24-000002",
                    "filing_date": "2024-01-04",
                    "failure_reason": "api_error",
                    "documents": [],
                }
            ).encode(),
            (
                "sec-bucket",
                "2023-01-02/8-K/320193/000000000023000001/manifest.json",
            ): json.dumps(
                {
                    "cik": "320193",
                    "accession_number": "0000000000-23-000001",
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
    assert first["text"].to_list() == ["complete submission"]
    assert len(second) == 1
    assert client.downloads == [
        ("sec-bucket", "2024-01-02/8-K/320193/000114036126006577/full.txt")
    ]
