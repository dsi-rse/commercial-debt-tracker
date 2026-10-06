"""The items dataset both genres share: columns, paths, table normalization and document text loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cdt.datasets import dataset_root
from cdt.ingest.core import decode_document_bytes
from cdt.shared import get_logger
from cdt.storage.objects import get_object_bytes, parse_s3_uri
from cdt.storage.objects import s3_client as storage_s3_client

LOGGER = get_logger(__name__)

ITEM_METADATA_COLUMNS = [
    "item_information",
    "extraction_status",
    "duplicate_resolution",
    "section_heading",
    "start_line",
    "end_line",
    "section_char_count",
]
# The document columns an item row copies. Pinned, not derived from
# ingest.DOCUMENT_COLUMNS: four datasets and the published items table take
# their schema from this list.
ITEM_DOCUMENT_COLUMNS = [
    "accession_number",
    "cik",
    "company_name",
    "url",
    "text",
    "date",
    "resource_uri",
]
ITEM_COLUMNS = ["item_id", "item", *ITEM_DOCUMENT_COLUMNS, *ITEM_METADATA_COLUMNS]
ITEM_INTEGER_COLUMNS = [
    "start_line",
    "end_line",
    "section_char_count",
]
ITEM_DATASET_NAME = "items"


def items_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical items dataset root."""
    return dataset_root(
        ITEM_DATASET_NAME, artifact_root=artifact_root, data_dir=data_dir
    )


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


def document_text_for_record(
    document: dict[str, object],
    *,
    data_dir: Path | None = None,
    s3_client: object | None = None,
) -> str:
    """Return one documents row's text: its inline ``text``, else its resource.

    A relative local ``resource_uri`` resolves against ``data_dir``.

    Raises:
        ValueError: If the row has neither text nor a resource URI, or the
            resource is on S3 and ``s3_client`` is None.
    """
    text = document.get("text")
    if isinstance(text, str) and text.strip():
        return text

    resource_uri = document.get("resource_uri")
    if not isinstance(resource_uri, str) or not resource_uri.strip():
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
        bucket, key = parse_s3_uri(resource_uri)
        body = get_object_bytes(s3_client, bucket, key)
        return decode_document_bytes(body)

    path = Path(resource_uri)
    if not path.is_absolute() and data_dir is not None:
        path = (data_dir / path).resolve()
    return decode_document_bytes(path.read_bytes())


def ensure_s3_client(
    s3_client: object | None,
    documents: list[dict[str, object]],
) -> object | None:
    """Return ``s3_client``, else a new client if some row's resource is on S3, else None."""
    if s3_client is not None:
        return s3_client
    for document in documents:
        resource_uri = document.get("resource_uri")
        if isinstance(resource_uri, str) and resource_uri.startswith("s3://"):
            return storage_s3_client()
    return None
