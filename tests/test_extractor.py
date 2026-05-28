"""Tests for the LLM-backed extractor stage."""

# ruff: noqa: ANN101, D102, D107

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from cdt.database import cdt_db_path, connect_cdt_db, upsert_items
from cdt.extractor import (
    INSTRUMENT_MENTION_COLUMNS,
    extract_pending_items,
    extract_tables,
)
from cdt.itemizer.core import ITEM_COLUMNS


class FakeChatClient:
    """Deterministic stand-in for the OpenRouter client."""

    def __init__(
        self,
        responses: dict[str, list[str] | str],
        *,
        error_stage: str | None = None,
    ) -> None:
        self.responses = responses
        self.error_stage = error_stage
        self.calls: list[str] = []
        self.stage_attempts: dict[str, int] = {}

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        del model, reasoning_effort
        stage = infer_stage(messages[0]["content"])
        self.calls.append(stage)
        self.stage_attempts[stage] = self.stage_attempts.get(stage, 0) + 1
        if stage == self.error_stage:
            raise RuntimeError("simulated provider error")
        responses = self.responses[stage]
        if isinstance(responses, str):
            return responses
        return responses[self.stage_attempts[stage] - 1]


def infer_stage(system_prompt: str) -> str:
    """Infer the stage name from the prompt text."""
    if "legal document analysis" in system_prompt:
        return "ner"
    if "lineage relationships" in system_prompt:
        return "instrument_relation"
    return "instrument_ie"


def build_item_rows() -> pd.DataFrame:
    """Create one item row for in-memory extraction tests."""
    return pd.DataFrame(
        [
            {
                "item_id": "000114036126006577-8-01",
                "item": "8.01",
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/item",
                "text": "The Original Loan was amended and replaced by the New Loan.",
                "date": "2024-01-02",
                "relevance": True,
            }
        ]
    )


def build_batch_table() -> pd.DataFrame:
    """Create one item batch parquet table."""
    return pd.DataFrame(
        [
            {
                "item_id": "000114036126006577-8-01",
                "item": "8.01",
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/item",
                "text": "The Original Loan was amended and replaced by the New Loan.",
                "date": "2024-01-02",
                "item_information": "Other Events",
                "extraction_status": "ok",
                "duplicate_resolution": None,
                "section_heading": "Item 8.01",
                "start_line": 1,
                "end_line": 2,
                "section_char_count": 63,
            }
        ],
        columns=ITEM_COLUMNS,
    )


def successful_client() -> FakeChatClient:
    """Return a fully successful fake chat client."""
    return FakeChatClient(
        {
            "ner": (
                "<body>The <debt_instrument>Original Loan</debt_instrument> was amended "
                "and replaced by the <debt_instrument>New Loan</debt_instrument>.</body>"
            ),
            "instrument_ie": json.dumps(
                [
                    {"name": ["tag-1"]},
                    {"name": ["tag-2"]},
                ]
            ),
            "instrument_relation": json.dumps(
                [{"from": "i-2", "to": "i-1", "type": "amendment_of"}]
            ),
        }
    )


def single_mention_client() -> FakeChatClient:
    """Return a fake chat client whose IE stage finds one mention only."""
    return FakeChatClient(
        {
            "ner": "<body>The <debt_instrument>Term Loan</debt_instrument> was issued.</body>",
            "instrument_ie": json.dumps([{"name": ["tag-1"]}]),
            "instrument_relation": "[]",
        }
    )


def seed_pending_item_db(tmp_path: Path) -> None:
    """Create one classified relevant pending item row and its backing parquet batch."""
    batch_path = tmp_path / "items" / "item-batch-000001.parquet"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    build_batch_table().to_parquet(batch_path, index=False)
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        upsert_items(
            conn,
            [
                {
                    "item_id": "000114036126006577-8-01",
                    "accession_number": "000114036126006577",
                    "item": "8.01",
                }
            ],
            batch_path=str(batch_path),
            status="classified",
        )
        conn.execute(
            """
            UPDATE items
            SET status = 'classified',
                label = 'relevant',
                relevance = 1,
                classification_score = 0.99,
                classified_at = '2024-01-03T00:00:00+00:00'
            WHERE item_id = ?
            """,
            ("000114036126006577-8-01",),
        )
        conn.commit()
    finally:
        conn.close()


def test_extract_tables_returns_instrument_mentions(tmp_path: Path) -> None:
    """In-memory extraction should return instrument mention rows."""
    tables = extract_tables(
        build_item_rows(),
        data_dir=tmp_path,
        client=successful_client(),
    )

    mentions = tables["instrument_mentions"]
    assert mentions.columns.tolist() == INSTRUMENT_MENTION_COLUMNS
    assert mentions["instrument_mention_id"].to_list() == [
        "000114036126006577-8-01--i-1",
        "000114036126006577-8-01--i-2",
    ]
    assert mentions["amendment_of"].to_list() == [None, "000114036126006577-8-01--i-1"]
    assert (tmp_path / "extractor_runs").exists()


def test_extract_tables_skips_relation_for_single_mention(tmp_path: Path) -> None:
    """Relation stage should be skipped when only one mention cluster is found."""
    client = single_mention_client()
    item_rows = build_item_rows()
    item_rows.loc[0, "text"] = "The Term Loan was issued."

    tables = extract_tables(item_rows, data_dir=tmp_path, client=client)

    mentions = tables["instrument_mentions"]
    assert len(mentions) == 1
    assert "instrument_relation" not in client.calls


def test_extract_tables_retries_after_validation_failure(tmp_path: Path) -> None:
    """A validation failure should retry and then succeed."""
    client = FakeChatClient(
        {
            "ner": (
                "<body>The <debt_instrument>Original Loan</debt_instrument> was amended "
                "and replaced by the <debt_instrument>New Loan</debt_instrument>.</body>"
            ),
            "instrument_ie": [
                "not json",
                json.dumps([{"name": ["tag-1"]}, {"name": ["tag-2"]}]),
            ],
            "instrument_relation": json.dumps(
                [{"from": "i-2", "to": "i-1", "type": "amendment_of"}]
            ),
        }
    )

    mentions = extract_tables(build_item_rows(), data_dir=tmp_path, client=client)[
        "instrument_mentions"
    ]

    assert len(mentions) == 2
    assert client.stage_attempts["instrument_ie"] == 2


def test_extract_tables_retries_after_malformed_name_cluster(tmp_path: Path) -> None:
    """Malformed name clusters should fail validation and retry, not crash."""
    client = FakeChatClient(
        {
            "ner": (
                "<body>The <debt_instrument>Original Loan</debt_instrument> was amended "
                "and replaced by the <debt_instrument>New Loan</debt_instrument>.</body>"
            ),
            "instrument_ie": [
                json.dumps([{"name": [["tag-1"]]}, {"name": ["tag-2"]}]),
                json.dumps([{"name": ["tag-1"]}, {"name": ["tag-2"]}]),
            ],
            "instrument_relation": json.dumps(
                [{"from": "i-2", "to": "i-1", "type": "amendment_of"}]
            ),
        }
    )

    mentions = extract_tables(build_item_rows(), data_dir=tmp_path, client=client)[
        "instrument_mentions"
    ]

    assert len(mentions) == 2
    assert client.stage_attempts["instrument_ie"] == 2


def test_extract_pending_items_marks_failures(tmp_path: Path) -> None:
    """Retry exhaustion should mark the item row extraction_failed."""
    seed_pending_item_db(tmp_path)
    client = FakeChatClient(
        {
            "ner": (
                "<body>The <debt_instrument>Original Loan</debt_instrument> was amended "
                "and replaced by the <debt_instrument>New Loan</debt_instrument>.</body>"
            ),
            "instrument_ie": "not json",
            "instrument_relation": "[]",
        }
    )

    mentions = extract_pending_items(
        data_dir=tmp_path,
        batch_size=1,
        max_attempts=2,
        client=client,
    )

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        row = conn.execute(
            "SELECT status, extractor_error FROM items WHERE item_id = ?",
            ("000114036126006577-8-01",),
        ).fetchone()
        mention_count = conn.execute(
            "SELECT COUNT(*) FROM instrument_mentions WHERE item_id = ?",
            ("000114036126006577-8-01",),
        ).fetchone()
    finally:
        conn.close()

    assert mentions.empty
    assert row[0] == "extraction_failed"
    assert "not valid JSON" in row[1]
    assert mention_count[0] == 0


def test_extract_pending_items_updates_sqlite_idempotently(tmp_path: Path) -> None:
    """SQLite-backed extraction should persist mention rows and skip reruns."""
    seed_pending_item_db(tmp_path)
    client = successful_client()

    first = extract_pending_items(data_dir=tmp_path, batch_size=1, client=client)
    second = extract_pending_items(data_dir=tmp_path, batch_size=1, client=client)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        item_row = conn.execute(
            """
            SELECT status, extractor_model, extractor_reasoning, extractor_run_path
            FROM items
            WHERE item_id = ?
            """,
            ("000114036126006577-8-01",),
        ).fetchone()
        mention_rows = conn.execute(
            """
            SELECT instrument_mention_id, amendment_of
            FROM instrument_mentions
            WHERE item_id = ?
            ORDER BY instrument_mention_id
            """,
            ("000114036126006577-8-01",),
        ).fetchall()
    finally:
        conn.close()

    assert len(first) == 2
    assert second.empty
    assert item_row[0] == "extracted"
    assert item_row[1] == "openai/gpt-5.4"
    assert item_row[2] == "none"
    assert Path(item_row[3]).joinpath("full.jsonl").exists()
    assert mention_rows == [
        ("000114036126006577-8-01--i-1", None),
        ("000114036126006577-8-01--i-2", "000114036126006577-8-01--i-1"),
    ]


def test_extract_pending_items_force_replaces_existing_mentions(tmp_path: Path) -> None:
    """Forced reruns should replace stale mention rows for the item."""
    seed_pending_item_db(tmp_path)
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
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
                "stale-mention",
                "000114036126006577-8-01",
                "i-99",
                "Old Mention",
                None,
                None,
                None,
                None,
                None,
                "[]",
                "[]",
                "{}",
                "{}",
                "{}",
                "{}",
                "{}",
                "2024-01-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            UPDATE items
            SET status = 'extracted',
                extracted_at = '2024-01-01T00:00:00+00:00',
                extractor_model = 'old/model',
                extractor_reasoning = 'low',
                extractor_run_path = 'old-run'
            WHERE item_id = ?
            """,
            ("000114036126006577-8-01",),
        )
        conn.commit()
    finally:
        conn.close()

    extract_pending_items(
        data_dir=tmp_path,
        batch_size=1,
        force=True,
        client=successful_client(),
    )

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        rows = conn.execute(
            """
            SELECT instrument_mention_id
            FROM instrument_mentions
            WHERE item_id = ?
            ORDER BY instrument_mention_id
            """,
            ("000114036126006577-8-01",),
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        ("000114036126006577-8-01--i-1",),
        ("000114036126006577-8-01--i-2",),
    ]


def test_connect_cdt_db_creates_instrument_mentions_table(tmp_path: Path) -> None:
    """Connecting should create the extractor output table."""
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
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()

    assert "instrument_mentions" in tables
