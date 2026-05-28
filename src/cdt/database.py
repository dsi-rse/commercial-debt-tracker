"""Shared SQLite state for CDT pipeline stages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from cdt import settings

CDT_DB_FILENAME = "cdt.sqlite"
DOCUMENT_STATUSES = ("indexed", "downloaded", "itemized")
ITEM_STATUSES = ("itemized", "classified", "extracted", "extraction_failed")
ITEM_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "TEXT"),
    ("relevance", "INTEGER"),
    ("classification_score", "REAL"),
    ("classified_at", "TEXT"),
    ("extracted_at", "TEXT"),
    ("extractor_model", "TEXT"),
    ("extractor_reasoning", "TEXT"),
    ("extractor_run_path", "TEXT"),
    ("extractor_error", "TEXT"),
)
INSTRUMENT_MENTION_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("debt_instrument_id", "TEXT"),
    ("matcher_status", "TEXT"),
    ("matched_at", "TEXT"),
    ("potential_matches_json", "TEXT"),
)
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    accession_number TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    url TEXT NOT NULL,
    resource_uri TEXT NOT NULL,
    date TEXT NOT NULL,
    batch_path TEXT,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    accession_number TEXT NOT NULL,
    item TEXT NOT NULL,
    batch_path TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS instrument_mentions (
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
    debt_instrument_id TEXT,
    matcher_status TEXT,
    matched_at TEXT,
    potential_matches_json TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(item_id) REFERENCES items(item_id)
);
CREATE TABLE IF NOT EXISTS debt_instruments (
    debt_instrument_id TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    created_from_mention_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_cik ON documents(cik);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_items_accession_number ON items(accession_number);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_instrument_mentions_item_id ON instrument_mentions(item_id);
CREATE INDEX IF NOT EXISTS idx_debt_instruments_cik ON debt_instruments(cik);
"""


def cdt_db_path(data_dir: Path | None = None) -> Path:
    """Return the canonical CDT SQLite path."""
    return (data_dir or settings.DATA_DIR) / CDT_DB_FILENAME


def connect_cdt_db(path: Path) -> sqlite3.Connection:
    """Connect to the shared CDT SQLite database and initialize its schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SQLITE_SCHEMA)
    ensure_item_columns(conn)
    ensure_instrument_mention_columns(conn)
    ensure_matcher_indexes(conn)
    conn.commit()
    return conn


def timestamp() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def upsert_documents(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, object]],
    *,
    batch_path: str | None,
    status: str,
) -> None:
    """Insert or update document index rows."""
    updated_at = timestamp()
    payload = [
        (
            str(row["accession_number"]),
            str(row["cik"]),
            str(row["url"]),
            str(row["resource_uri"]),
            str(row["date"]),
            batch_path,
            status,
            updated_at,
        )
        for row in rows
    ]
    if not payload:
        return
    conn.executemany(
        """
        INSERT INTO documents (
            accession_number,
            cik,
            url,
            resource_uri,
            date,
            batch_path,
            status,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(accession_number) DO UPDATE SET
            cik = excluded.cik,
            url = excluded.url,
            resource_uri = excluded.resource_uri,
            date = excluded.date,
            batch_path = excluded.batch_path,
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        payload,
    )
    conn.commit()


def upsert_items(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, object]],
    *,
    batch_path: str,
    status: str,
) -> None:
    """Insert or update item index rows."""
    updated_at = timestamp()
    payload = [
        (
            str(row["item_id"]),
            str(row["accession_number"]),
            str(row["item"]),
            batch_path,
            status,
            updated_at,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        for row in rows
    ]
    if not payload:
        return
    conn.executemany(
        """
        INSERT INTO items (
            item_id,
            accession_number,
            item,
            batch_path,
            status,
            updated_at,
            label,
            relevance,
            classification_score,
            classified_at,
            extracted_at,
            extractor_model,
            extractor_reasoning,
            extractor_run_path,
            extractor_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            accession_number = excluded.accession_number,
            item = excluded.item,
            batch_path = excluded.batch_path,
            status = excluded.status,
            updated_at = excluded.updated_at,
            label = NULL,
            relevance = NULL,
            classification_score = NULL,
            classified_at = NULL,
            extracted_at = NULL,
            extractor_model = NULL,
            extractor_reasoning = NULL,
            extractor_run_path = NULL,
            extractor_error = NULL
        """,
        payload,
    )
    conn.commit()


def read_document_accessions(
    conn: sqlite3.Connection,
    *,
    statuses: Sequence[str] | None = None,
) -> set[str]:
    """Read document accessions from the shared database."""
    query = ["SELECT accession_number FROM documents"]
    params: list[object] = []
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        query.append(f"WHERE status IN ({placeholders})")
        params.extend(statuses)
    rows = conn.execute("\n".join(query), params)
    return {str(row[0]) for row in rows}


def read_documents(
    conn: sqlite3.Connection,
    *,
    statuses: Sequence[str] | None = None,
    exclude_accessions: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Read document rows from the shared database."""
    query = [
        "SELECT accession_number, cik, url, resource_uri, date, batch_path, status",
        "FROM documents",
    ]
    params: list[object] = []
    where: list[str] = []
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        where.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if exclude_accessions:
        excluded = list(exclude_accessions)
        placeholders = ", ".join("?" for _ in excluded)
        where.append(f"accession_number NOT IN ({placeholders})")
        params.extend(excluded)
    if where:
        query.append("WHERE " + " AND ".join(where))
    query.append("ORDER BY date, accession_number")
    if limit is not None:
        query.append("LIMIT ?")
        params.append(limit)
    rows = conn.execute("\n".join(query), params)
    return [
        {
            "accession_number": row[0],
            "cik": row[1],
            "url": row[2],
            "resource_uri": row[3],
            "date": row[4],
            "batch_path": row[5],
            "status": row[6],
        }
        for row in rows
    ]


def read_item_accessions(
    conn: sqlite3.Connection,
    *,
    statuses: Sequence[str] | None = None,
) -> set[str]:
    """Read accession numbers from the item index table."""
    query = ["SELECT accession_number FROM items"]
    params: list[object] = []
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        query.append(f"WHERE status IN ({placeholders})")
        params.extend(statuses)
    rows = conn.execute("\n".join(query), params)
    return {str(row[0]) for row in rows}


def read_items(
    conn: sqlite3.Connection,
    *,
    statuses: Sequence[str] | None = None,
    exclude_item_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Read item index rows from the shared database."""
    query = [
        "SELECT",
        "    item_id,",
        "    accession_number,",
        "    item,",
        "    batch_path,",
        "    status,",
        "    updated_at,",
        "    label,",
        "    relevance,",
        "    classification_score,",
        "    classified_at,",
        "    extracted_at,",
        "    extractor_model,",
        "    extractor_reasoning,",
        "    extractor_run_path,",
        "    extractor_error",
        "FROM items",
    ]
    params: list[object] = []
    where: list[str] = []
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        where.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if exclude_item_ids:
        excluded = list(exclude_item_ids)
        placeholders = ", ".join("?" for _ in excluded)
        where.append(f"item_id NOT IN ({placeholders})")
        params.extend(excluded)
    if where:
        query.append("WHERE " + " AND ".join(where))
    query.append("ORDER BY accession_number, item_id")
    if limit is not None:
        query.append("LIMIT ?")
        params.append(limit)
    rows = conn.execute("\n".join(query), params)
    return [
        {
            "item_id": row[0],
            "accession_number": row[1],
            "item": row[2],
            "batch_path": row[3],
            "status": row[4],
            "updated_at": row[5],
            "label": row[6],
            "relevance": row[7],
            "classification_score": row[8],
            "classified_at": row[9],
            "extracted_at": row[10],
            "extractor_model": row[11],
            "extractor_reasoning": row[12],
            "extractor_run_path": row[13],
            "extractor_error": row[14],
        }
        for row in rows
    ]


def read_document_resource_uri(
    conn: sqlite3.Connection, accession_number: str
) -> str | None:
    """Return the resource URI stored for one accession, if any."""
    row = conn.execute(
        "SELECT resource_uri FROM documents WHERE accession_number = ?",
        (accession_number,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def mark_documents_itemized(
    conn: sqlite3.Connection, accessions: Iterable[str]
) -> None:
    """Mark source documents as itemized."""
    accessions = list(accessions)
    if not accessions:
        return
    updated_at = timestamp()
    conn.executemany(
        """
        UPDATE documents
        SET status = 'itemized',
            updated_at = ?
        WHERE accession_number = ?
        """,
        [(updated_at, accession) for accession in accessions],
    )
    conn.commit()


def mark_items_classified(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, object]],
) -> None:
    """Persist classification outputs for item rows."""
    classified_at = timestamp()
    payload = [
        (
            "classified",
            str(row["label"]),
            int(bool(row["relevance"])),
            float(row["classification_score"]),
            classified_at,
            classified_at,
            str(row["item_id"]),
        )
        for row in rows
    ]
    if not payload:
        return
    conn.executemany(
        """
        UPDATE items
        SET status = ?,
            label = ?,
            relevance = ?,
            classification_score = ?,
            classified_at = ?,
            updated_at = ?,
            extracted_at = NULL,
            extractor_model = NULL,
            extractor_reasoning = NULL,
            extractor_run_path = NULL,
            extractor_error = NULL
        WHERE item_id = ?
        """,
        payload,
    )
    conn.commit()


def mark_items_extracted(
    conn: sqlite3.Connection,
    item_ids: Iterable[str],
    *,
    extractor_model: str,
    extractor_reasoning: str,
    extractor_run_path: str,
) -> None:
    """Persist successful extraction metadata for item rows."""
    item_ids = list(item_ids)
    if not item_ids:
        return
    extracted_at = timestamp()
    conn.executemany(
        """
        UPDATE items
        SET status = 'extracted',
            extracted_at = ?,
            extractor_model = ?,
            extractor_reasoning = ?,
            extractor_run_path = ?,
            extractor_error = NULL,
            updated_at = ?
        WHERE item_id = ?
        """,
        [
            (
                extracted_at,
                extractor_model,
                extractor_reasoning,
                extractor_run_path,
                extracted_at,
                item_id,
            )
            for item_id in item_ids
        ],
    )
    conn.commit()


def mark_items_extraction_failed(
    conn: sqlite3.Connection,
    failures: Iterable[dict[str, str]],
    *,
    extractor_model: str,
    extractor_reasoning: str,
    extractor_run_path: str,
) -> None:
    """Persist extraction failures for item rows."""
    failed_at = timestamp()
    payload = [
        (
            failed_at,
            extractor_model,
            extractor_reasoning,
            extractor_run_path,
            failure["extractor_error"],
            failed_at,
            failure["item_id"],
        )
        for failure in failures
    ]
    if not payload:
        return
    conn.executemany(
        """
        UPDATE items
        SET status = 'extraction_failed',
            extracted_at = ?,
            extractor_model = ?,
            extractor_reasoning = ?,
            extractor_run_path = ?,
            extractor_error = ?,
            updated_at = ?
        WHERE item_id = ?
        """,
        payload,
    )
    conn.commit()


def replace_instrument_mentions(
    conn: sqlite3.Connection,
    item_id: str,
    rows: Iterable[dict[str, object]],
) -> None:
    """Replace all extracted instrument mentions for one item."""
    updated_at = timestamp()
    conn.execute("DELETE FROM instrument_mentions WHERE item_id = ?", (item_id,))
    payload = [
        (
            str(row["instrument_mention_id"]),
            item_id,
            str(row["raw_id"]),
            row.get("name"),
            row.get("start_date"),
            row.get("end_date"),
            row.get("amount"),
            row.get("amendment_of"),
            row.get("split_of"),
            str(row["lenders_json"]),
            str(row["other_interested_parties_json"]),
            str(row["mention_corefs_json"]),
            str(row["start_date_corefs_json"]),
            str(row["end_date_corefs_json"]),
            str(row["amount_corefs_json"]),
            str(row["instrument_mention_json"]),
            None,
            None,
            None,
            None,
            updated_at,
        )
        for row in rows
    ]
    if payload:
        conn.executemany(
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
                debt_instrument_id,
                matcher_status,
                matched_at,
                potential_matches_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
    conn.commit()


def ensure_item_columns(conn: sqlite3.Connection) -> None:
    """Add optional classifier columns to the items table when missing."""
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(items)")}
    for column_name, column_type in ITEM_EXTRA_COLUMNS:
        if column_name in existing_columns:
            continue
        conn.execute(f"ALTER TABLE items ADD COLUMN {column_name} {column_type}")


def ensure_instrument_mention_columns(conn: sqlite3.Connection) -> None:
    """Add optional matcher columns to the instrument_mentions table when missing."""
    existing_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(instrument_mentions)")
    }
    for column_name, column_type in INSTRUMENT_MENTION_EXTRA_COLUMNS:
        if column_name in existing_columns:
            continue
        conn.execute(
            f"ALTER TABLE instrument_mentions ADD COLUMN {column_name} {column_type}"
        )


def ensure_matcher_indexes(conn: sqlite3.Connection) -> None:
    """Create matcher indexes after migration-added columns are present."""
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_instrument_mentions_matcher_status
        ON instrument_mentions(matcher_status);
        CREATE INDEX IF NOT EXISTS idx_instrument_mentions_debt_instrument_id
        ON instrument_mentions(debt_instrument_id);
        """
    )


def read_matcher_mentions(
    conn: sqlite3.Connection,
    *,
    pending_only: bool = True,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Read instrument mentions with issuer and filing context for matcher processing."""
    query = [
        "SELECT",
        "    mention.instrument_mention_id,",
        "    mention.item_id,",
        "    mention.raw_id,",
        "    mention.name,",
        "    mention.start_date,",
        "    mention.end_date,",
        "    mention.amount,",
        "    mention.amendment_of,",
        "    mention.split_of,",
        "    mention.lenders_json,",
        "    mention.other_interested_parties_json,",
        "    mention.mention_corefs_json,",
        "    mention.start_date_corefs_json,",
        "    mention.end_date_corefs_json,",
        "    mention.amount_corefs_json,",
        "    mention.instrument_mention_json,",
        "    mention.debt_instrument_id,",
        "    mention.matcher_status,",
        "    mention.matched_at,",
        "    mention.potential_matches_json,",
        "    item.accession_number,",
        "    document.cik,",
        "    document.date",
        "FROM instrument_mentions AS mention",
        "JOIN items AS item ON item.item_id = mention.item_id",
        "JOIN documents AS document ON document.accession_number = item.accession_number",
    ]
    params: list[object] = []
    if pending_only:
        query.append("WHERE mention.matcher_status IS NULL")
    query.append(
        "ORDER BY document.date, item.accession_number, mention.item_id, mention.raw_id"
    )
    if limit is not None:
        query.append("LIMIT ?")
        params.append(limit)
    rows = conn.execute("\n".join(query), params)
    return [
        {
            "instrument_mention_id": row[0],
            "item_id": row[1],
            "raw_id": row[2],
            "name": row[3],
            "start_date": row[4],
            "end_date": row[5],
            "amount": row[6],
            "amendment_of": row[7],
            "split_of": row[8],
            "lenders_json": row[9],
            "other_interested_parties_json": row[10],
            "mention_corefs_json": row[11],
            "start_date_corefs_json": row[12],
            "end_date_corefs_json": row[13],
            "amount_corefs_json": row[14],
            "instrument_mention_json": row[15],
            "debt_instrument_id": row[16],
            "matcher_status": row[17],
            "matched_at": row[18],
            "potential_matches_json": row[19],
            "accession_number": row[20],
            "cik": row[21],
            "date": row[22],
        }
        for row in rows
    ]


def clear_matcher_assignments(
    conn: sqlite3.Connection,
    mention_ids: Iterable[str] | None = None,
) -> None:
    """Clear matcher outputs for one set of mentions or all mentions."""
    updated_at = timestamp()
    mention_list = list(mention_ids) if mention_ids is not None else None
    if mention_list is None:
        conn.execute(
            """
            UPDATE instrument_mentions
            SET debt_instrument_id = NULL,
                matcher_status = NULL,
                matched_at = NULL,
                potential_matches_json = NULL,
                updated_at = ?
            """,
            (updated_at,),
        )
    elif mention_list:
        conn.executemany(
            """
            UPDATE instrument_mentions
            SET debt_instrument_id = NULL,
                matcher_status = NULL,
                matched_at = NULL,
                potential_matches_json = NULL,
                updated_at = ?
            WHERE instrument_mention_id = ?
            """,
            [(updated_at, mention_id) for mention_id in mention_list],
        )
    conn.commit()


def replace_debt_instruments(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, str]],
) -> None:
    """Replace all debt instruments with a deterministic payload."""
    updated_at = timestamp()
    conn.execute("DELETE FROM debt_instruments")
    payload = [
        (
            row["debt_instrument_id"],
            row["cik"],
            row["created_from_mention_id"],
            updated_at,
            updated_at,
        )
        for row in rows
    ]
    if payload:
        conn.executemany(
            """
            INSERT INTO debt_instruments (
                debt_instrument_id,
                cik,
                created_from_mention_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
    conn.commit()


def update_matcher_results(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, object]],
) -> None:
    """Persist matcher outputs for instrument mentions."""
    matched_at = timestamp()
    payload = [
        (
            row.get("debt_instrument_id"),
            row["matcher_status"],
            matched_at,
            row["potential_matches_json"],
            matched_at,
            row["instrument_mention_id"],
        )
        for row in rows
    ]
    if not payload:
        return
    conn.executemany(
        """
        UPDATE instrument_mentions
        SET debt_instrument_id = ?,
            matcher_status = ?,
            matched_at = ?,
            potential_matches_json = ?,
            updated_at = ?
        WHERE instrument_mention_id = ?
        """,
        payload,
    )
    conn.commit()
