"""Shared SQLite state for CDT pipeline stages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from cdt import settings

CDT_DB_FILENAME = "cdt.sqlite"
DOCUMENT_STATUSES = ("indexed", "downloaded", "itemized")
ITEM_STATUSES = ("itemized",)
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
CREATE INDEX IF NOT EXISTS idx_documents_cik ON documents(cik);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_items_accession_number ON items(accession_number);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
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
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            accession_number = excluded.accession_number,
            item = excluded.item,
            batch_path = excluded.batch_path,
            status = excluded.status,
            updated_at = excluded.updated_at
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
