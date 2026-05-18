"""Itemizer stage for SEC 8-K documents."""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from cdt import settings
from cdt.database import (
    cdt_db_path,
    connect_cdt_db,
    mark_documents_itemized,
    read_documents,
    read_item_accessions,
    upsert_items,
)
from cdt.ingest import DOCUMENT_COLUMNS, default_s3_client
from cdt.itemizer.extract import DocumentText, ItemSection, extract_items_from_document
from cdt.storage import write_parquet_batch

LOGGER = logging.getLogger(__name__)
POTENTIALLY_RELEVANT_ITEM_NUMBERS = (
    "1.01",
    "1.02",
    "2.03",
    "2.04",
    "7.01",
    "8.01",
)
ITEM_METADATA_COLUMNS = [
    "item_information",
    "extraction_status",
    "duplicate_resolution",
    "section_heading",
    "start_line",
    "end_line",
    "section_char_count",
]
ITEM_COLUMNS = ["item_id", "item", *DOCUMENT_COLUMNS, *ITEM_METADATA_COLUMNS]
ITEM_INTEGER_COLUMNS = [
    "start_line",
    "end_line",
    "section_char_count",
]
ITEM_BATCH_PREFIX = "item-batch"


def items_path(data_dir: Path | None = None) -> Path:
    """Return the directory for item batch artifacts."""
    return (data_dir or settings.DATA_DIR) / "items"


def item_id_for(accession_number: str, item_number: str) -> str:
    """Build a deterministic item identifier."""
    return f"{accession_number}-{item_number.replace('.', '-')}"


def itemize_documents(
    documents: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    force: bool = False,
    s3_client: object | None = None,
    item_numbers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Extract and persist item sections from complete 8-K documents."""
    if documents.empty:
        return pd.DataFrame(columns=ITEM_COLUMNS)

    selected_item_numbers = normalize_item_numbers(item_numbers)
    conn = connect_cdt_db(cdt_db_path(data_dir))
    try:
        existing_accessions = (
            read_item_accessions(conn, statuses=("itemized",)) if not force else set()
        )
        documents_to_process = documents.loc[
            ~documents["accession_number"].isin(existing_accessions)
        ]
        LOGGER.info(
            "Starting itemizer: %s documents requested, %s documents to process",
            len(documents),
            len(documents_to_process),
        )
        if documents_to_process.empty:
            return pd.DataFrame(columns=ITEM_COLUMNS)

        resource_uri_map = _resource_uri_map(conn)
        resolved_s3_client = _ensure_s3_client(
            s3_client,
            documents_to_process.to_dict("records"),
            resource_uri_map,
        )
        rows: list[dict[str, object]] = []
        saved_item_counts: Counter[str] = Counter()
        irrelevant_count = 0
        for document in documents_to_process.to_dict("records"):
            sections = itemize_document_record(
                document,
                resource_uri_map=resource_uri_map,
                data_dir=data_dir,
                s3_client=resolved_s3_client,
                item_numbers=None,
            )
            relevant_sections = [
                section
                for section in sections
                if section.item_number in selected_item_numbers
            ]
            irrelevant_count += len(sections) - len(relevant_sections)
            saved_item_counts.update(
                section.item_number
                for section in relevant_sections
                if section.item_number
            )
            rows.extend(item_row(section) for section in relevant_sections)

        table = normalize_item_table(pd.DataFrame(rows, columns=ITEM_COLUMNS))
        mark_documents_itemized(conn, documents_to_process["accession_number"])
        if table.empty:
            _log_itemizer_summary(
                total_saved=0,
                saved_item_counts=saved_item_counts,
                selected_item_numbers=selected_item_numbers,
                irrelevant_count=irrelevant_count,
                batch_path=None,
            )
            return table

        batch_path = write_parquet_batch(items_path(data_dir), ITEM_BATCH_PREFIX, table)
        upsert_items(
            conn,
            table.to_dict("records"),
            batch_path=str(batch_path),
            status="itemized",
        )
        _log_itemizer_summary(
            total_saved=len(table),
            saved_item_counts=saved_item_counts,
            selected_item_numbers=selected_item_numbers,
            irrelevant_count=irrelevant_count,
            batch_path=batch_path,
        )
        return table
    finally:
        conn.close()


def itemize_pending_documents(
    *,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    s3_client: object | None = None,
    item_numbers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Itemize source documents tracked in the shared CDT SQLite database."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    selected_item_numbers = normalize_item_numbers(item_numbers)
    conn = connect_cdt_db(cdt_db_path(data_dir))
    processed_accessions: set[str] = set()
    processed_frames: list[pd.DataFrame] = []
    total_documents = 0
    shared_s3_client = s3_client
    try:
        while True:
            index_rows = read_documents(
                conn,
                statuses=("indexed", "downloaded", "itemized")
                if force
                else ("indexed", "downloaded"),
                exclude_accessions=processed_accessions,
                limit=batch_size,
            )
            if not index_rows:
                break
            documents = pd.DataFrame(index_rows)
            shared_s3_client = _ensure_s3_client(
                shared_s3_client,
                documents.to_dict("records"),
                None,
            )
            items = itemize_documents(
                documents,
                data_dir=data_dir,
                force=force,
                s3_client=shared_s3_client,
                item_numbers=selected_item_numbers,
            )
            accessions = set(documents["accession_number"])
            processed_accessions.update(accessions)
            total_documents += len(documents)
            if items.empty:
                continue
            processed_frames.append(items)
    finally:
        conn.close()

    LOGGER.info("Itemized %s source documents", total_documents)
    if not processed_frames:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(columns=ITEM_COLUMNS)


def normalize_item_table(table: pd.DataFrame) -> pd.DataFrame:
    """Coerce item table columns to Parquet-friendly dtypes."""
    if table.empty:
        return table.reindex(columns=ITEM_COLUMNS)

    normalized = table.reindex(columns=ITEM_COLUMNS).copy()
    for column in ITEM_INTEGER_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(
            "Int64"
        )
    return normalized


def itemize_document_record(
    document: dict[str, object],
    *,
    resource_uri_map: dict[str, str] | None = None,
    data_dir: Path | None = None,
    s3_client: object | None = None,
    item_numbers: tuple[str, ...] | None = None,
) -> list[ItemSection]:
    """Extract item sections from one document record."""
    text = _document_text_for_record(
        document,
        resource_uri_map=resource_uri_map or {},
        data_dir=data_dir,
        s3_client=s3_client,
    )
    sections = extract_items_from_document(
        DocumentText(
            accession_number=str(document["accession_number"]),
            cik=str(document["cik"]),
            url=str(document["url"]),
            text=text,
            date=str(document["date"]),
        )
    )
    if item_numbers is None:
        return sections

    selected_item_numbers = normalize_item_numbers(item_numbers)
    return [
        section for section in sections if section.item_number in selected_item_numbers
    ]


def item_row(section: ItemSection) -> dict[str, object]:
    """Convert an extracted item section to a persisted item row."""
    return {
        "item_id": item_id_for(section.accession_number, section.item_number),
        "item": section.item_number,
        "accession_number": section.accession_number,
        "cik": section.cik,
        "url": section.url,
        "text": section.section_text,
        "date": section.date,
        "item_information": section.item_information,
        "extraction_status": section.extraction_status,
        "duplicate_resolution": section.duplicate_resolution,
        "section_heading": section.section_heading,
        "start_line": section.start_line,
        "end_line": section.end_line,
        "section_char_count": section.section_char_count,
    }


def _resource_uri_map(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["accession_number"]): str(row["resource_uri"])
        for row in read_documents(
            conn,
            statuses=("indexed", "downloaded", "itemized"),
        )
    }


def _document_text_for_record(
    document: dict[str, object],
    *,
    resource_uri_map: dict[str, str],
    data_dir: Path | None,
    s3_client: object | None,
) -> str:
    text = document.get("text")
    if isinstance(text, str) and text.strip():
        return text

    resource_uri = document.get("resource_uri")
    if not isinstance(resource_uri, str) or not resource_uri.strip():
        resource_uri = resource_uri_map.get(str(document["accession_number"]))
    if not resource_uri:
        msg = f"no text or resource URI available for accession {document['accession_number']}"
        raise ValueError(msg)
    return _load_resource_text(
        str(resource_uri), data_dir=data_dir, s3_client=s3_client
    )


def _load_resource_text(
    resource_uri: str,
    *,
    data_dir: Path | None,
    s3_client: object | None,
) -> str:
    if resource_uri.startswith("s3://"):
        if s3_client is None:
            msg = "expected an initialized S3 client for s3:// resources"
            raise ValueError(msg)
        bucket, key = _parse_s3_uri(resource_uri)
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8", errors="replace")

    path = Path(resource_uri)
    if not path.is_absolute() and data_dir is not None:
        path = (data_dir / path).resolve()
    return path.read_text(encoding="utf-8")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        msg = f"Expected an s3:// URI, got {uri!r}"
        raise ValueError(msg)
    return parsed.netloc, parsed.path.lstrip("/")


def _ensure_s3_client(
    s3_client: object | None,
    documents: list[dict[str, object]],
    resource_uri_map: dict[str, str] | None,
) -> object | None:
    if s3_client is not None:
        return s3_client
    for document in documents:
        resource_uri = document.get("resource_uri")
        if not isinstance(resource_uri, str) or not resource_uri.strip():
            resource_uri = (resource_uri_map or {}).get(
                str(document["accession_number"])
            )
        if isinstance(resource_uri, str) and resource_uri.startswith("s3://"):
            return default_s3_client()
    return None


def normalize_item_numbers(item_numbers: tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize configured item numbers or return the default relevant set."""
    values = item_numbers or POTENTIALLY_RELEVANT_ITEM_NUMBERS
    normalized = []
    for value in values:
        stripped = value.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(dict.fromkeys(normalized))


def _log_itemizer_summary(
    *,
    total_saved: int,
    saved_item_counts: Counter[str],
    selected_item_numbers: tuple[str, ...],
    irrelevant_count: int,
    batch_path: Path | None,
) -> None:
    """Log the saved relevant-item counts and discarded irrelevant count."""
    item_summary = ", ".join(
        f"{item_number}={saved_item_counts.get(item_number, 0)}"
        for item_number in selected_item_numbers
    )
    LOGGER.info(
        "Itemizer complete: total_saved=%s items=[%s] irrelevant_not_saved=%s%s",
        total_saved,
        item_summary,
        irrelevant_count,
        f" batch_path={batch_path}" if batch_path is not None else "",
    )
