"""Itemizer stage for SEC 8-K documents."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd

from cdt.datasets import (
    completion_registry_path,
    dataset_root,
    date_shard_partition_path,
    iter_date_shard_partitions,
    load_completed_partitions,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
    save_completed_partitions,
)
from cdt.ingest import DOCUMENT_COLUMNS, decode_document_bytes, default_s3_client
from cdt.itemizer.extract import DocumentText, ItemSection, extract_items_from_document
from cdt.shared import get_logger
from cdt.storage import (
    artifact_exists,
    parse_s3_uri,
    read_table,
    write_json_artifact,
    write_partition_table,
)

LOGGER = get_logger(__name__)
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
    """Extract relevant item sections from complete 8-K documents."""
    del force
    if documents.empty:
        return pd.DataFrame(columns=ITEM_COLUMNS)

    selected_item_numbers = normalize_item_numbers(item_numbers)
    resolved_s3_client = _ensure_s3_client(s3_client, documents.to_dict("records"))
    rows: list[dict[str, object]] = []
    saved_item_counts: Counter[str] = Counter()
    irrelevant_count = 0
    for document in documents.to_dict("records"):
        sections = itemize_document_record(
            document,
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
            section.item_number for section in relevant_sections if section.item_number
        )
        rows.extend(item_row(section) for section in relevant_sections)

    table = normalize_item_table(pd.DataFrame(rows, columns=ITEM_COLUMNS))
    _log_itemizer_summary(
        total_saved=len(table),
        saved_item_counts=saved_item_counts,
        selected_item_numbers=selected_item_numbers,
        irrelevant_count=irrelevant_count,
    )
    return table


def itemize_pending_documents(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    s3_client: object | None = None,
    item_numbers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Itemize canonical document partitions into canonical item partitions."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    selected_item_numbers = normalize_item_numbers(item_numbers)
    processed_frames: list[pd.DataFrame] = []
    processed_partitions: list[str] = []
    completed_document_paths = (
        set()
        if force
        else load_completed_partitions(
            "itemize", artifact_root=resolved_root, data_dir=data_dir
        )
    )
    visited_document_paths: set[str] = set()
    total_documents = 0
    empty_partitions = 0
    shared_s3_client = s3_client
    pending_document_paths: list[str] = []

    for document_path in iter_date_shard_partitions(
        "documents",
        artifact_root=resolved_root,
        data_dir=data_dir,
    ):
        partition = parse_date_shard_partition(document_path)
        target_path = date_shard_partition_path(
            ITEM_DATASET_NAME,
            partition_date=partition["date"],
            shard=partition["shard"],
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        if not force and artifact_exists(target_path):
            continue
        if not force and document_path in completed_document_paths:
            continue
        pending_document_paths.append(document_path)

    total_partitions = len(pending_document_paths)
    for chunk_start in range(0, total_partitions, batch_size):
        chunk_paths = pending_document_paths[chunk_start : chunk_start + batch_size]
        for partition_index, document_path in enumerate(
            chunk_paths, start=chunk_start + 1
        ):
            partition = parse_date_shard_partition(document_path)
            partition_label = f"date={partition['date']} shard={partition['shard']}"
            partition_start = perf_counter()
            visited_document_paths.add(document_path)
            documents = read_table(document_path, DOCUMENT_COLUMNS).reindex(
                columns=DOCUMENT_COLUMNS
            )
            total_documents += len(documents)
            shared_s3_client = _ensure_s3_client(
                shared_s3_client,
                documents.to_dict("records"),
            )
            items = itemize_documents(
                documents,
                data_dir=data_dir,
                s3_client=shared_s3_client,
                item_numbers=selected_item_numbers,
            )
            if items.empty:
                empty_partitions += 1
            else:
                write_partition_table(
                    items_root(resolved_root, data_dir=data_dir),
                    partition={"date": partition["date"], "shard": partition["shard"]},
                    table=items.reindex(columns=ITEM_COLUMNS),
                )
                processed_frames.append(items)
                processed_partitions.append(
                    date_shard_partition_path(
                        ITEM_DATASET_NAME,
                        partition_date=partition["date"],
                        shard=partition["shard"],
                        artifact_root=resolved_root,
                        data_dir=data_dir,
                    )
                )
            LOGGER.info(
                "Itemize partition complete: %s progress=%s/%s documents=%s items=%s wrote_output=%s elapsed=%.1fs",
                partition_label,
                partition_index,
                total_partitions,
                len(documents),
                len(items),
                not items.empty,
                perf_counter() - partition_start,
            )

    updated_completed_paths = completed_document_paths | visited_document_paths
    save_completed_partitions(
        "itemize",
        updated_completed_paths,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )

    manifest = {
        "artifact_root": resolved_root,
        "stage": "itemize",
        "batch_size": batch_size,
        "force": force,
        "item_numbers": list(selected_item_numbers),
        "documents_processed": total_documents,
        "partitions_visited": sorted(visited_document_paths),
        "partitions_written": processed_partitions,
        "empty_partitions_skipped_from_write": empty_partitions,
        "completion_registry": completion_registry_path(
            "itemize", artifact_root=resolved_root, data_dir=data_dir
        ),
    }
    write_json_artifact(
        run_manifest_path(
            "itemize",
            "latest",
            artifact_root=resolved_root,
            data_dir=data_dir,
        ),
        manifest,
    )
    if not processed_frames:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    LOGGER.info("Itemized %s source documents", total_documents)
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
    data_dir: Path | None = None,
    s3_client: object | None = None,
    item_numbers: tuple[str, ...] | None = None,
) -> list[ItemSection]:
    """Extract item sections from one document record."""
    text = _document_text_for_record(
        document,
        data_dir=data_dir,
        s3_client=s3_client,
    )
    sections = extract_items_from_document(
        DocumentText(
            accession_number=str(document["accession_number"]),
            cik=str(document["cik"]),
            company_name=str(document.get("company_name") or ""),
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
        "company_name": section.company_name,
        "url": section.url,
        "text": section.section_text,
        "date": section.date,
        "resource_uri": None,
        "item_information": section.item_information,
        "extraction_status": section.extraction_status,
        "duplicate_resolution": section.duplicate_resolution,
        "section_heading": section.section_heading,
        "start_line": section.start_line,
        "end_line": section.end_line,
        "section_char_count": section.section_char_count,
    }


def _document_text_for_record(
    document: dict[str, object],
    *,
    data_dir: Path | None,
    s3_client: object | None,
) -> str:
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
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return decode_document_bytes(body)

    path = Path(resource_uri)
    if not path.is_absolute() and data_dir is not None:
        path = (data_dir / path).resolve()
    return decode_document_bytes(path.read_bytes())


def _ensure_s3_client(
    s3_client: object | None,
    documents: list[dict[str, object]],
) -> object | None:
    if s3_client is not None:
        return s3_client
    for document in documents:
        resource_uri = document.get("resource_uri")
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
) -> None:
    """Log the saved relevant-item counts and discarded irrelevant count."""
    item_summary = ", ".join(
        f"{item_number}={saved_item_counts.get(item_number, 0)}"
        for item_number in selected_item_numbers
    )
    LOGGER.info(
        "Itemizer complete: total_saved=%s items=[%s] irrelevant_not_saved=%s",
        total_saved,
        item_summary,
        irrelevant_count,
    )
