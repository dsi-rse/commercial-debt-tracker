"""Tests for the matcher stage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cdt.database import cdt_db_path, connect_cdt_db, upsert_documents, upsert_items
from cdt.matcher import match_pending_mentions


def seed_document_and_item(
    tmp_path: Path,
    *,
    accession_number: str,
    cik: str,
    item_id: str,
    date: str,
) -> None:
    """Create one extracted item row with its backing document."""
    resource_path = tmp_path / f"{accession_number}.txt"
    resource_path.write_text("seed", encoding="utf-8")
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        upsert_documents(
            conn,
            [
                {
                    "accession_number": accession_number,
                    "cik": cik,
                    "url": f"https://sec.example/{accession_number}",
                    "resource_uri": str(resource_path),
                    "date": date,
                }
            ],
            batch_path=None,
            status="itemized",
        )
        upsert_items(
            conn,
            [
                {
                    "item_id": item_id,
                    "accession_number": accession_number,
                    "item": "8.01",
                }
            ],
            batch_path=str(tmp_path / "items" / "seed.parquet"),
            status="extracted",
        )
        conn.execute(
            """
            UPDATE items
            SET status = 'extracted',
                relevance = 1,
                extracted_at = '2024-01-03T00:00:00+00:00'
            WHERE item_id = ?
            """,
            (item_id,),
        )
        conn.commit()
    finally:
        conn.close()


def insert_mention(
    tmp_path: Path,
    *,
    mention_id: str,
    item_id: str,
    raw_id: str,
    name: str,
    start_date: str,
    amount: str,
    lenders: list[str],
    amendment_of: str | None = None,
    split_of: str | None = None,
) -> None:
    """Insert one instrument mention row."""
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        lender_payload = json.dumps(
            [
                {
                    "tag_ids": [f"tag-{index}"],
                    "mentions": [{"tag_id": f"tag-{index}", "text": lender}],
                }
                for index, lender in enumerate(lenders, start=1)
            ],
            sort_keys=True,
        )
        conn.execute(
            """
            INSERT INTO instrument_mentions (
                instrument_mention_id,
                item_id,
                raw_id,
                name,
                start_date,
                end_date,
                amount,
                amendment_of,
                split_of,
                lenders_json,
                other_interested_parties_json,
                mention_corefs_json,
                start_date_corefs_json,
                end_date_corefs_json,
                amount_corefs_json,
                instrument_mention_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mention_id,
                item_id,
                raw_id,
                name,
                start_date,
                None,
                amount,
                amendment_of,
                split_of,
                lender_payload,
                "[]",
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "2024-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_connect_cdt_db_creates_matcher_schema(tmp_path: Path) -> None:
    """Connecting should create matcher tables and columns."""
    db_path = cdt_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE items (
                item_id TEXT PRIMARY KEY,
                accession_number TEXT NOT NULL,
                item TEXT NOT NULL,
                batch_path TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE instrument_mentions (
                instrument_mention_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                raw_id TEXT NOT NULL,
                name TEXT,
                start_date TEXT,
                end_date TEXT,
                amount TEXT,
                amendment_of TEXT,
                split_of TEXT,
                lenders_json TEXT NOT NULL,
                other_interested_parties_json TEXT NOT NULL,
                mention_corefs_json TEXT NOT NULL,
                start_date_corefs_json TEXT NOT NULL,
                end_date_corefs_json TEXT NOT NULL,
                amount_corefs_json TEXT NOT NULL,
                instrument_mention_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    conn = connect_cdt_db(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        mention_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(instrument_mentions)")
        }
    finally:
        conn.close()

    assert "debt_instruments" in tables
    assert {
        "debt_instrument_id",
        "matcher_status",
        "matched_at",
        "potential_matches_json",
    } <= mention_columns


def test_matcher_groups_exact_same_cik_mentions_and_skips_reruns(
    tmp_path: Path,
) -> None:
    """Exact amount/date plus strong lender similarity should auto-merge once."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="320193",
        item_id="0001-8-01",
        date="2024-01-02",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="320193",
        item_id="0002-8-01",
        date="2024-01-03",
    )
    insert_mention(
        tmp_path,
        mention_id="0001-8-01--i-1",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Original Loan",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["JPMorgan Chase Bank, N.A."],
    )
    insert_mention(
        tmp_path,
        mention_id="0002-8-01--i-1",
        item_id="0002-8-01",
        raw_id="i-1",
        name="Term Loan",
        start_date="January 1, 2024",
        amount="$100,000,000",
        lenders=["JPMorgan Chase Bank National Association"],
    )

    first = match_pending_mentions(data_dir=tmp_path, batch_size=1)
    second = match_pending_mentions(data_dir=tmp_path, batch_size=1)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        rows = conn.execute(
            """
            SELECT instrument_mention_id, debt_instrument_id, matcher_status
            FROM instrument_mentions
            ORDER BY instrument_mention_id
            """
        ).fetchall()
        instruments = conn.execute(
            "SELECT debt_instrument_id FROM debt_instruments ORDER BY debt_instrument_id"
        ).fetchall()
    finally:
        conn.close()

    assert len(first["instrument_mentions"]) == 2
    assert second["instrument_mentions"].empty
    assert rows == [
        ("0001-8-01--i-1", "di::0001-8-01--i-1", "singleton"),
        ("0002-8-01--i-1", "di::0001-8-01--i-1", "matched"),
    ]
    assert instruments == [("di::0001-8-01--i-1",)]


def test_matcher_does_not_cross_match_different_ciks(tmp_path: Path) -> None:
    """Equivalent terms across issuers should remain separate instruments."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="320193",
        item_id="0001-8-01",
        date="2024-01-02",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="789019",
        item_id="0002-8-01",
        date="2024-01-03",
    )
    for item_id in ("0001-8-01", "0002-8-01"):
        insert_mention(
            tmp_path,
            mention_id=f"{item_id}--i-1",
            item_id=item_id,
            raw_id="i-1",
            name="Revolver",
            start_date="2024-01-01",
            amount="$50 million",
            lenders=["Bank of America"],
        )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM debt_instruments").fetchone()[0]
        statuses = conn.execute(
            "SELECT matcher_status FROM instrument_mentions ORDER BY instrument_mention_id"
        ).fetchall()
    finally:
        conn.close()

    assert count == 2
    assert statuses == [("singleton",), ("singleton",)]


def test_matcher_records_loose_candidates_without_forcing_merge(tmp_path: Path) -> None:
    """Weak lender similarity should keep a singleton and store potential matches."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="320193",
        item_id="0001-8-01",
        date="2024-01-02",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="320193",
        item_id="0002-8-01",
        date="2024-01-03",
    )
    insert_mention(
        tmp_path,
        mention_id="0001-8-01--i-1",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Credit Facility",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Citibank"],
    )
    insert_mention(
        tmp_path,
        mention_id="0002-8-01--i-1",
        item_id="0002-8-01",
        raw_id="i-1",
        name="Credit Facility",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Citizens Bank"],
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        row = conn.execute(
            """
            SELECT matcher_status, potential_matches_json
            FROM instrument_mentions
            WHERE instrument_mention_id = '0002-8-01--i-1'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == "ambiguous"
    assert json.loads(row[1]) == [
        {
            "amount_match": True,
            "debt_instrument_id": "di::0001-8-01--i-1",
            "lender_similarity": 0.7619,
            "match_via": "self->self",
            "start_date_match": True,
        }
    ]


def test_matcher_uses_amendment_bridge_for_exact_terms(tmp_path: Path) -> None:
    """Amendment lineage should provide an alternate match surface."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="320193",
        item_id="0001-8-01",
        date="2024-01-02",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="320193",
        item_id="0002-8-01",
        date="2024-01-03",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0003",
        cik="320193",
        item_id="0003-8-01",
        date="2024-01-04",
    )
    insert_mention(
        tmp_path,
        mention_id="0001-8-01--i-1",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Original Loan",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Acme Bank"],
    )
    insert_mention(
        tmp_path,
        mention_id="0002-8-01--i-1",
        item_id="0002-8-01",
        raw_id="i-1",
        name="Amended Loan",
        start_date="2024-02-01",
        amount="$125 million",
        lenders=["Acme Bank"],
        amendment_of="0001-8-01--i-1",
    )
    insert_mention(
        tmp_path,
        mention_id="0003-8-01--i-1",
        item_id="0003-8-01",
        raw_id="i-1",
        name="Facility Reference",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Acme Bank"],
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        row = conn.execute(
            """
            SELECT debt_instrument_id, matcher_status
            FROM instrument_mentions
            WHERE instrument_mention_id = '0003-8-01--i-1'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == ("di::0001-8-01--i-1", "matched")
