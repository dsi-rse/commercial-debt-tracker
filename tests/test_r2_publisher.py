"""Tests for Cloudflare R2 snapshot publishing."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from botocore.exceptions import ClientError

from cdt.r2_publisher import (
    R2PublishConfig,
    build_dashboard_snapshot,
    sync_snapshot,
)
from cdt.storage import write_partition_table


class FakePaginator:
    """Simple paginator over in-memory object keys."""

    def __init__(self: FakePaginator, objects: dict[str, bytes]) -> None:
        """Store visible object state."""
        self._objects = objects

    def paginate(
        self: FakePaginator,
        *,
        Bucket: str,
        Prefix: str,
    ) -> list[dict[str, object]]:  # noqa: N803
        """Return a single page containing matching keys."""
        del Bucket
        return [
            {
                "Contents": [
                    {"Key": key}
                    for key in sorted(self._objects)
                    if key.startswith(Prefix)
                ]
            }
        ]


class FakeR2Client:
    """Minimal S3-compatible client for change-detection tests."""

    def __init__(
        self: FakeR2Client,
        objects: dict[str, bytes] | None = None,
    ) -> None:
        """Initialize the fake object store."""
        self.objects = dict(objects or {})
        self.put_calls: list[str] = []
        self.delete_calls: list[str] = []

    def get_paginator(self: FakeR2Client, name: str) -> FakePaginator:
        """Return the list-objects paginator."""
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)

    def get_object(
        self: FakeR2Client,
        *,
        Bucket: str,
        Key: str,
    ) -> dict[str, object]:  # noqa: N803
        """Return one object body or raise a missing-key error."""
        del Bucket
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(
        self: FakeR2Client,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
    ) -> None:  # noqa: N803
        """Persist one object body."""
        del Bucket
        self.objects[Key] = Body
        self.put_calls.append(Key)

    def delete_object(
        self: FakeR2Client,
        *,
        Bucket: str,
        Key: str,
    ) -> None:  # noqa: N803
        """Delete one object key."""
        del Bucket
        self.objects.pop(Key, None)
        self.delete_calls.append(Key)


def test_build_dashboard_snapshot_matches_contract_shape(tmp_path: Path) -> None:
    """Snapshot builder should emit contract-shaped JSON objects."""
    write_partition_table(
        tmp_path / "items",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=pd.DataFrame(
            [
                {
                    "item_id": "item-1",
                    "item": "1.01",
                    "accession_number": "0001",
                    "cik": "0000320193",
                    "url": "https://sec.example/1",
                    "text": "Borrower entered a Senior Notes agreement for $500 million.",
                    "date": "2024-01-02",
                    "resource_uri": "sec://0001",
                    "item_information": "ENTRY INTO A MATERIAL DEFINITIVE AGREEMENT",
                    "extraction_status": "ok",
                    "duplicate_resolution": "selected",
                    "section_heading": "Item 1.01",
                    "start_line": 1,
                    "end_line": 2,
                    "section_char_count": 61,
                },
                {
                    "item_id": "item-2",
                    "item": "1.01",
                    "accession_number": "0002",
                    "cik": "0000320193",
                    "url": "https://sec.example/2",
                    "text": "Borrower amended the Senior Notes agreement with Acme Bank.",
                    "date": "2024-01-03",
                    "resource_uri": "sec://0002",
                    "item_information": "ENTRY INTO A MATERIAL DEFINITIVE AGREEMENT",
                    "extraction_status": "ok",
                    "duplicate_resolution": "selected",
                    "section_heading": "Item 1.01",
                    "start_line": 1,
                    "end_line": 2,
                    "section_char_count": 62,
                },
            ]
        ),
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=pd.DataFrame(
            [
                {
                    "debt_instrument_mention_id": "dim::base",
                    "item_id": "item-1",
                    "accession_number": "0001",
                    "cik": "0000320193",
                    "date": "2024-01-02",
                    "raw_id": "i-1",
                    "name": "Senior Notes",
                    "start_date": "2024-01-15",
                    "end_date": "2031-01-15",
                    "amount": "500000000",
                    "amendment_of": None,
                    "split_of": None,
                    "lenders_json": json.dumps(
                        [
                            {
                                "tag_ids": ["tag-l-1"],
                                "mentions": [
                                    {
                                        "tag_id": "tag-l-1",
                                        "type": "organization",
                                        "text": "Acme Bank",
                                        "char_start": 47,
                                        "char_end": 56,
                                    }
                                ],
                            }
                        ],
                        sort_keys=True,
                    ),
                    "other_interested_parties_json": "[]",
                    "name_json": json.dumps(
                        {
                            "tag_ids": ["tag-n-1"],
                            "mentions": [
                                {
                                    "tag_id": "tag-n-1",
                                    "type": "debt_instrument",
                                    "text": "Senior Notes",
                                    "char_start": 19,
                                    "char_end": 31,
                                }
                            ],
                        },
                        sort_keys=True,
                    ),
                    "start_date_json": json.dumps(
                        {
                            "tag_ids": ["tag-s-1"],
                            "mentions": [
                                {
                                    "tag_id": "tag-s-1",
                                    "type": "date",
                                    "text": "January 15, 2024",
                                    "char_start": 0,
                                    "char_end": 16,
                                }
                            ],
                            "normalized_date": "2024-01-15",
                        },
                        sort_keys=True,
                    ),
                    "end_date_json": json.dumps(
                        {
                            "tag_ids": ["tag-e-1"],
                            "mentions": [
                                {
                                    "tag_id": "tag-e-1",
                                    "type": "date",
                                    "text": "January 15, 2031",
                                    "char_start": 0,
                                    "char_end": 16,
                                }
                            ],
                            "normalized_date": "2031-01-15",
                        },
                        sort_keys=True,
                    ),
                    "amount_json": json.dumps(
                        {
                            "tag_ids": ["tag-a-1"],
                            "mentions": [
                                {
                                    "tag_id": "tag-a-1",
                                    "type": "amount",
                                    "text": "$500 million",
                                    "char_start": 46,
                                    "char_end": 58,
                                }
                            ],
                            "normalized_amount": "500000000",
                            "currency": "USD",
                        },
                        sort_keys=True,
                    ),
                },
                {
                    "debt_instrument_mention_id": "dim::amendment",
                    "item_id": "item-2",
                    "accession_number": "0002",
                    "cik": "0000320193",
                    "date": "2024-01-03",
                    "raw_id": "i-1",
                    "name": "Senior Notes",
                    "start_date": "2024-01-15",
                    "end_date": "2032-01-15",
                    "amount": "500000000",
                    "amendment_of": "dim::base",
                    "split_of": None,
                    "lenders_json": "[]",
                    "other_interested_parties_json": "[]",
                    "name_json": "{}",
                    "start_date_json": "{}",
                    "end_date_json": "{}",
                    "amount_json": "{}",
                },
            ]
        ),
    )
    write_partition_table(
        tmp_path / "mention-matches",
        partition={"cik_shard": "0001"},
        table=pd.DataFrame(
            [
                {
                    "debt_instrument_mention_id": "dim::base",
                    "debt_instrument_id": "dim::base",
                    "matcher_status": "singleton",
                },
                {
                    "debt_instrument_mention_id": "dim::amendment",
                    "debt_instrument_id": "dim::amendment",
                    "matcher_status": "matched",
                },
            ]
        ),
    )
    write_partition_table(
        tmp_path / "debt-instruments",
        partition={"cik_shard": "0001"},
        table=pd.DataFrame(
            [
                {
                    "debt_instrument_id": "dim::base",
                    "cik": "0000320193",
                    "seed_debt_instrument_mention_id": "dim::base",
                    "amendment_of_debt_instrument_id": None,
                    "split_of_debt_instrument_id": None,
                    "name": "Senior Notes",
                    "start_date": "2024-01-15",
                    "end_date": "2031-01-15",
                    "amount": "500000000",
                    "direct_mentions_json": '["dim::base"]',
                    "lenders_json": json.dumps(
                        [
                            {
                                "tag_ids": ["tag-l-1"],
                                "mentions": [{"text": "Acme Bank"}],
                            }
                        ],
                        sort_keys=True,
                    ),
                    "other_interested_parties_json": "[]",
                    "possibly_related_json": '["dim::amendment"]',
                },
                {
                    "debt_instrument_id": "dim::amendment",
                    "cik": "0000320193",
                    "seed_debt_instrument_mention_id": "dim::amendment",
                    "amendment_of_debt_instrument_id": "dim::base",
                    "split_of_debt_instrument_id": None,
                    "name": "Senior Notes",
                    "start_date": "2024-01-15",
                    "end_date": "2032-01-15",
                    "amount": "500000000",
                    "direct_mentions_json": '["dim::amendment"]',
                    "lenders_json": "[]",
                    "other_interested_parties_json": "[]",
                    "possibly_related_json": "[]",
                },
            ]
        ),
    )

    snapshot = build_dashboard_snapshot(
        artifact_root=tmp_path,
        generated_at=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )

    assert set(snapshot) == {
        "index.json",
        "companies/0000320193.json",
        "debt-instruments/dim%3A%3Abase.json",
        "debt-instruments/dim%3A%3Aamendment.json",
    }

    index_payload = json.loads(snapshot["index.json"])
    company_payload = json.loads(snapshot["companies/0000320193.json"])
    instrument_payload = json.loads(snapshot["debt-instruments/dim%3A%3Abase.json"])

    assert index_payload["generatedAt"] == "2026-05-31T12:00:00+00:00"
    assert index_payload["instruments"][0]["id"] == "dim::amendment"
    assert company_payload["cik"] == "0000320193"
    assert company_payload["debtInstruments"][0]["id"] == "dim::amendment"
    assert instrument_payload["instrument"]["seedMentionId"] == "dim::base"
    assert instrument_payload["instrument"]["lenders"] == ["Acme Bank"]
    assert instrument_payload["mentions"][0]["matcherStatus"] == "singleton"
    assert instrument_payload["mentions"][0]["summary"]["amount"] == "500000000"
    assert (
        instrument_payload["mentions"][0]["highlightMap"]["amount"][0]["text"]
        == "$500 million"
    )
    assert instrument_payload["properties"][1]["key"] == "amount"
    assert instrument_payload["properties"][1]["mentionId"] == "dim::base"
    assert instrument_payload["relatedInstruments"] == [
        {
            "id": "dim::amendment",
            "name": "Senior Notes",
            "relationship": "Amended by",
        },
        {
            "id": "dim::amendment",
            "name": "Senior Notes",
            "relationship": "Possibly related",
        },
    ]


def test_sync_snapshot_only_writes_changed_objects() -> None:
    """R2 sync should upload only changed objects and delete stale keys."""
    config = R2PublishConfig(
        account_id="acct",
        bucket_name="bucket",
        access_key_id="key",
        secret_access_key="test-secret",  # noqa: S106
    )
    snapshot = {
        "debt-instruments/a.json": b'{"id":"a"}',
        "companies/1.json": b'{"cik":"1"}',
        "index.json": b'{"generatedAt":"2026-05-31T12:00:00+00:00","instruments":[]}',
    }
    client = FakeR2Client(
        {
            "generated/index.json": snapshot["index.json"],
            "generated/companies/1.json": b'{"cik":"old"}',
            "generated/stale.json": b"stale",
        }
    )

    result = sync_snapshot(snapshot=snapshot, config=config, r2_client=client)

    assert result.object_count == 3
    assert result.uploaded_count == 2
    assert result.skipped_count == 1
    assert result.deleted_count == 1
    assert client.put_calls == [
        "generated/debt-instruments/a.json",
        "generated/companies/1.json",
    ]
    assert client.delete_calls == ["generated/stale.json"]
