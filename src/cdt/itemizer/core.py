"""Stub itemizer stage for SEC 8-K documents."""

import logging
from pathlib import Path

import pandas as pd

from cdt import settings
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.storage import append_new_rows, read_table

LOGGER = logging.getLogger(__name__)
ITEM_COLUMNS = ["item_id", "item", *DOCUMENT_COLUMNS]
STUB_ITEM_NUMBER = "0.00"


def items_path(data_dir: Path | None = None) -> Path:
    """Return the canonical items table path."""
    return (data_dir or settings.DATA_DIR) / "items" / "items.parquet"


def item_id_for(accession_number: str, item_number: str) -> str:
    """Build a deterministic item identifier."""
    return f"{accession_number}-{item_number.replace('.', '-')}"


def itemize_documents(
    documents: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Stub itemizer that creates one placeholder item per document."""
    path = items_path(data_dir)
    existing = read_table(path, ITEM_COLUMNS)
    requested_accessions = (
        set(documents["accession_number"]) if not documents.empty else set()
    )
    existing_accessions = (
        set(existing["accession_number"]) if not existing.empty and not force else set()
    )
    documents_to_process = documents.loc[
        ~documents["accession_number"].isin(existing_accessions)
    ]

    LOGGER.info(
        "Starting itemizer: %s documents requested, %s documents to process",
        len(documents),
        len(documents_to_process),
    )
    rows = []
    for document in documents_to_process.to_dict("records"):
        rows.append(
            {
                "item_id": item_id_for(
                    str(document["accession_number"]), STUB_ITEM_NUMBER
                ),
                "item": STUB_ITEM_NUMBER,
                "accession_number": document["accession_number"],
                "cik": document["cik"],
                "url": document["url"],
                "text": document["text"],
                "date": document["date"],
            }
        )
    updated = append_new_rows(
        path,
        pd.DataFrame(rows, columns=ITEM_COLUMNS),
        ["item_id"],
        ITEM_COLUMNS,
        replace_keys=requested_accessions if force else None,
        replace_key_column="accession_number" if force else None,
    )
    LOGGER.info("Itemizer complete: %s rows at %s", len(updated), path)
    return updated
