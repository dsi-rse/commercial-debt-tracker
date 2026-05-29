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
    end_date: str | None = None,
    amount: str,
    lenders: list[str],
    other_interested_parties: list[str] | None = None,
    amendment_of: str | None = None,
    split_of: str | None = None,
) -> None:
    """Insert one debt instrument mention row."""
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        lender_payload = json.dumps(
            [
                {
                    "tag_ids": [f"tag-l-{index}"],
                    "mentions": [{"tag_id": f"tag-l-{index}", "text": lender}],
                }
                for index, lender in enumerate(lenders, start=1)
            ],
            sort_keys=True,
        )
        party_payload = json.dumps(
            [
                {
                    "tag_ids": [f"tag-p-{index}"],
                    "mentions": [{"tag_id": f"tag-p-{index}", "text": party}],
                }
                for index, party in enumerate(other_interested_parties or [], start=1)
            ],
            sort_keys=True,
        )
        conn.execute(
            """
            INSERT INTO debt_instrument_mentions (
                debt_instrument_mention_id,
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
                name_json,
                start_date_json,
                end_date_json,
                amount_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mention_id,
                item_id,
                raw_id,
                name,
                start_date,
                end_date,
                amount,
                amendment_of,
                split_of,
                lender_payload,
                party_payload,
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
    """Connecting should create matcher tables and active view."""
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
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        mention_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(debt_instrument_mentions)")
        }
        instrument_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(debt_instrument)")
        }
    finally:
        conn.close()

    assert "debt_instrument_mentions" in tables
    assert "debt_instrument" in tables
    assert "active_debt_instruments" in tables
    assert "debt_instrument_mention_id" in mention_columns
    assert "direct_mentions_json" in instrument_columns
    assert "mentions_json" not in instrument_columns


def test_connect_cdt_db_migrates_legacy_mention_columns(tmp_path: Path) -> None:
    """Connecting should migrate legacy mention column names in place."""
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
            CREATE TABLE debt_instrument_mentions (
                debt_instrument_mention_id TEXT PRIMARY KEY,
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
                debt_instrument_id TEXT,
                matcher_status TEXT,
                matched_at TEXT,
                potential_matches_json TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO debt_instrument_mentions (
                debt_instrument_mention_id,
                item_id,
                raw_id,
                name,
                lenders_json,
                other_interested_parties_json,
                mention_corefs_json,
                start_date_corefs_json,
                end_date_corefs_json,
                amount_corefs_json,
                instrument_mention_json,
                potential_matches_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-1",
                "item-1",
                "i-1",
                "Legacy Loan",
                "[]",
                "[]",
                '{"tag_ids":["tag-1"],"mentions":[]}',
                '{"tag_ids":["tag-2"],"mentions":[]}',
                '{"tag_ids":["tag-3"],"mentions":[]}',
                '{"tag_ids":["tag-4"],"mentions":[]}',
                '{"legacy":true}',
                '[{"candidate":"old"}]',
                "2024-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    conn = connect_cdt_db(db_path)
    try:
        mention_columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(debt_instrument_mentions)")
        ]
        row = conn.execute(
            """
            SELECT name_json, start_date_json, end_date_json, amount_json
            FROM debt_instrument_mentions
            WHERE debt_instrument_mention_id = 'legacy-1'
            """
        ).fetchone()
    finally:
        conn.close()

    assert "name_json" in mention_columns
    assert "start_date_json" in mention_columns
    assert "end_date_json" in mention_columns
    assert "amount_json" in mention_columns
    assert "mention_corefs_json" not in mention_columns
    assert "instrument_mention_json" not in mention_columns
    assert "potential_matches_json" not in mention_columns
    assert row == (
        '{"tag_ids":["tag-1"],"mentions":[]}',
        '{"tag_ids":["tag-2"],"mentions":[]}',
        '{"tag_ids":["tag-3"],"mentions":[]}',
        '{"tag_ids":["tag-4"],"mentions":[]}',
    )


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
        mention_id="m-0001",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Original Loan",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["JPMorgan Chase Bank, N.A."],
    )
    insert_mention(
        tmp_path,
        mention_id="m-0002",
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
        mention_rows = conn.execute(
            """
            SELECT debt_instrument_mention_id, debt_instrument_id, matcher_status
            FROM debt_instrument_mentions
            ORDER BY debt_instrument_mention_id
            """
        ).fetchall()
        instrument_rows = conn.execute(
            """
            SELECT debt_instrument_id, seed_debt_instrument_mention_id, direct_mentions_json
            FROM debt_instrument
            """
        ).fetchall()
    finally:
        conn.close()

    assert len(first["debt_instrument_mentions"]) == 2
    assert second["debt_instrument_mentions"].empty
    assert mention_rows == [
        ("m-0001", "m-0001", "singleton"),
        ("m-0002", "m-0001", "matched"),
    ]
    assert instrument_rows == [("m-0001", "m-0001", '["m-0001", "m-0002"]')]


def test_matcher_records_loose_candidates_without_forcing_merge(tmp_path: Path) -> None:
    """Weak lender similarity should keep a singleton marked ambiguous."""
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
        mention_id="m-0001",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Credit Facility",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Citibank"],
    )
    insert_mention(
        tmp_path,
        mention_id="m-0002",
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
            SELECT matcher_status, debt_instrument_id
            FROM debt_instrument_mentions
            WHERE debt_instrument_mention_id = 'm-0002'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == "ambiguous"
    assert row[1] == "m-0002"


def test_matcher_falls_back_to_exact_name_when_lender_evidence_is_missing_or_generic(
    tmp_path: Path,
) -> None:
    """Missing or generic lender evidence should still merge exact same note mentions."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="1461755",
        item_id="0001-8-01",
        date="2020-08-21",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="1461755",
        item_id="0002-8-01",
        date="2020-12-04",
    )
    insert_mention(
        tmp_path,
        mention_id="dim::9e36b5bc2daf2c6f893d9d21",
        item_id="0001-8-01",
        raw_id="i-1",
        name="5.50% Fixed to Floating Rate Subordinated Notes due 2030",
        start_date="August 20, 2020",
        end_date="September 1, 2025",
        amount="$75.0 million",
        lenders=["Purchasers"],
    )
    insert_mention(
        tmp_path,
        mention_id="dim::615353591b21b1b773f7806b",
        item_id="0002-8-01",
        raw_id="i-1",
        name="5.5% fixed to floating rate subordinated notes due 2030",
        start_date="2020-08-20",
        amount="$75 million",
        lenders=[],
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        rows = conn.execute(
            """
            SELECT debt_instrument_mention_id, debt_instrument_id, matcher_status
            FROM debt_instrument_mentions
            ORDER BY debt_instrument_mention_id
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        (
            "dim::615353591b21b1b773f7806b",
            "dim::9e36b5bc2daf2c6f893d9d21",
            "matched",
        ),
        (
            "dim::9e36b5bc2daf2c6f893d9d21",
            "dim::9e36b5bc2daf2c6f893d9d21",
            "singleton",
        ),
    ]


def test_matcher_does_not_fallback_merge_distinct_series_names(tmp_path: Path) -> None:
    """Exact amount and start date should not override distinct normalized names."""
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
        mention_id="m-a2",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Class A-2 Asset Backed Notes",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=[],
    )
    insert_mention(
        tmp_path,
        mention_id="m-a3",
        item_id="0002-8-01",
        raw_id="i-1",
        name="Class A-3 Asset Backed Notes",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Purchasers"],
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        rows = conn.execute(
            """
            SELECT debt_instrument_mention_id, debt_instrument_id, matcher_status
            FROM debt_instrument_mentions
            ORDER BY debt_instrument_mention_id
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        ("m-a2", "m-a2", "singleton"),
        ("m-a3", "m-a3", "singleton"),
    ]


def test_matcher_does_not_fallback_merge_conflicting_end_dates(tmp_path: Path) -> None:
    """Fallback should reject rows when both end dates are present and disagree."""
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
        mention_id="m-2031",
        item_id="0001-8-01",
        raw_id="i-1",
        name="6.125% Senior Unsecured Notes due 2031",
        start_date="2024-01-01",
        end_date="2031-09-01",
        amount="$100 million",
        lenders=[],
    )
    insert_mention(
        tmp_path,
        mention_id="m-2031-bad",
        item_id="0002-8-01",
        raw_id="i-1",
        name="6.125% Senior Unsecured Notes due 2031",
        start_date="2024-01-01",
        end_date="2034-09-01",
        amount="$100 million",
        lenders=["Holders"],
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        rows = conn.execute(
            """
            SELECT debt_instrument_mention_id, debt_instrument_id, matcher_status
            FROM debt_instrument_mentions
            ORDER BY debt_instrument_mention_id
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        ("m-2031", "m-2031", "singleton"),
        ("m-2031-bad", "m-2031-bad", "singleton"),
    ]


def test_matcher_creates_active_view_rollup_for_split(tmp_path: Path) -> None:
    """Active view should include ancestor mentions on both split branches."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="320193",
        item_id="0001-8-01",
        date="2024-01-01",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="320193",
        item_id="0002-8-01",
        date="2024-01-02",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0003",
        cik="320193",
        item_id="0003-8-01",
        date="2024-01-03",
    )
    insert_mention(
        tmp_path,
        mention_id="m-root",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Original Facility",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Acme Bank"],
    )
    insert_mention(
        tmp_path,
        mention_id="m-left",
        item_id="0002-8-01",
        raw_id="i-1",
        name="Left Facility",
        start_date="2024-02-01",
        amount="$55 million",
        lenders=["Acme Bank"],
        split_of="m-root",
    )
    insert_mention(
        tmp_path,
        mention_id="m-right",
        item_id="0003-8-01",
        raw_id="i-1",
        name="Right Facility",
        start_date="2024-03-01",
        amount="$45 million",
        lenders=["Acme Bank"],
        split_of="m-root",
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        instrument_rows = conn.execute(
            """
            SELECT debt_instrument_id, split_of_debt_instrument_id
            FROM debt_instrument
            ORDER BY debt_instrument_id
            """
        ).fetchall()
        active_rows = conn.execute(
            """
            SELECT debt_instrument_id, mentions_json
            FROM active_debt_instruments
            ORDER BY debt_instrument_id
            """
        ).fetchall()
    finally:
        conn.close()

    assert instrument_rows == [
        ("m-left", "m-root"),
        ("m-right", "m-root"),
        ("m-root", None),
    ]
    assert active_rows == [
        ("m-left", '["m-left","m-root"]'),
        ("m-right", '["m-right","m-root"]'),
    ]


def test_matcher_resolves_current_properties_and_related_mentions(
    tmp_path: Path,
) -> None:
    """Debt instrument rows should use newest non-null fields and advisory related mentions."""
    seed_document_and_item(
        tmp_path,
        accession_number="0001",
        cik="320193",
        item_id="0001-8-01",
        date="2024-01-01",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0002",
        cik="320193",
        item_id="0002-8-01",
        date="2024-01-02",
    )
    seed_document_and_item(
        tmp_path,
        accession_number="0003",
        cik="320193",
        item_id="0003-8-01",
        date="2024-01-03",
    )
    insert_mention(
        tmp_path,
        mention_id="m-root",
        item_id="0001-8-01",
        raw_id="i-1",
        name="Original Facility",
        start_date="2024-01-01",
        amount="$100 million",
        lenders=["Acme Bank"],
        other_interested_parties=["Party One"],
    )
    insert_mention(
        tmp_path,
        mention_id="m-amended",
        item_id="0002-8-01",
        raw_id="i-1",
        name="Amended Facility",
        start_date="2024-01-01",
        amount="$125 million",
        lenders=["Acme Bank, N.A."],
        amendment_of="m-root",
    )
    insert_mention(
        tmp_path,
        mention_id="m-related",
        item_id="0003-8-01",
        raw_id="i-1",
        name="Possible Sidecar",
        start_date="2026-01-01",
        amount="$33 million",
        lenders=["Acme Bank National Association"],
    )

    match_pending_mentions(data_dir=tmp_path)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        row = conn.execute(
            """
            SELECT
                debt_instrument_id,
                amendment_of_debt_instrument_id,
                name,
                amount,
                other_interested_parties_json,
                possibly_related_json
            FROM debt_instrument
            WHERE debt_instrument_id = 'm-amended'
            """
        ).fetchone()
        active_ids = conn.execute(
            "SELECT debt_instrument_id FROM active_debt_instruments ORDER BY debt_instrument_id"
        ).fetchall()
    finally:
        conn.close()

    assert row == (
        "m-amended",
        "m-root",
        "Amended Facility",
        "$125 million",
        '[{"mentions": [{"tag_id": "tag-p-1", "text": "Party One"}], "tag_ids": ["tag-p-1"]}]',
        '["m-related"]',
    )
    assert active_ids == [("m-amended",), ("m-related",)]
