"""Tests for pipeline stage persistence and stub behavior."""

from pathlib import Path

import pandas as pd

from cdt.classifier import classify_items
from cdt.extractor import extract_tables
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import item_id_for, itemize_documents


def test_itemizer_creates_stable_item_ids_and_is_idempotent(tmp_path: Path) -> None:
    """The stub itemizer creates one deterministic item row per document."""
    documents = pd.DataFrame(
        [
            {
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/full.txt",
                "text": "complete submission",
                "date": "2024-01-02",
            }
        ],
        columns=DOCUMENT_COLUMNS,
    )

    first = itemize_documents(documents, data_dir=tmp_path)
    second = itemize_documents(documents, data_dir=tmp_path)

    assert first["item_id"].to_list() == ["000114036126006577-0-00"]
    assert first["item"].to_list() == ["0.00"]
    assert len(second) == 1
    assert item_id_for("000114036126006577", "2.03") == "000114036126006577-2-03"


def test_classifier_adds_relevance_and_is_idempotent(tmp_path: Path) -> None:
    """The stub classifier marks items as not relevant and avoids duplicates."""
    items = pd.DataFrame(
        [
            {
                "item_id": "000114036126006577-0-00",
                "item": "0.00",
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/full.txt",
                "text": "complete submission",
                "date": "2024-01-02",
            }
        ]
    )

    first = classify_items(items, data_dir=tmp_path)
    second = classify_items(items, data_dir=tmp_path)

    assert first["relevance"].to_list() == [False]
    assert len(second) == 1


def test_extractor_returns_empty_tables_and_writes_metadata(tmp_path: Path) -> None:
    """The stub extractor records successful completion."""
    classified_items = pd.DataFrame(
        [{"item_id": "000114036126006577-0-00", "relevance": False}]
    )

    tables = extract_tables(classified_items, data_dir=tmp_path)

    assert tables == {}
    assert (tmp_path / "extracted_tables" / "_SUCCESS.json").exists()
