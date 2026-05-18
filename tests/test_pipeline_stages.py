"""Tests for pipeline stage persistence and stub behavior."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cdt.itemizer.core
from cdt.classifier import classify_items
from cdt.classifier import core as classifier_core
from cdt.database import cdt_db_path, connect_cdt_db, upsert_documents
from cdt.extractor import extract_tables
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import item_id_for, itemize_documents, itemize_pending_documents


def test_itemizer_creates_stable_item_ids_and_is_idempotent(tmp_path: Path) -> None:
    """The stub itemizer creates one deterministic item row per document."""
    documents = pd.DataFrame(
        [
            {
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/full.txt",
                "text": """
ITEM INFORMATION: Other Events
<DOCUMENT>
<TYPE>8-K
<TEXT>
Item 8.01 Other Events.
This is the extracted event text.
Item 9.01 Financial Statements and Exhibits.
Exhibit list.
</TEXT>
</DOCUMENT>
""",
                "date": "2024-01-02",
            }
        ],
        columns=DOCUMENT_COLUMNS,
    )

    first = itemize_documents(documents, data_dir=tmp_path)
    second = itemize_documents(documents, data_dir=tmp_path)

    assert first["item_id"].to_list() == ["000114036126006577-8-01"]
    assert first["item"].to_list() == ["8.01"]
    assert first["extraction_status"].to_list() == ["ok"]
    assert "extracted event text" in first["text"].to_list()[0]
    assert "Exhibit list" not in first["text"].to_list()[0]
    assert len(second) == 0
    assert len(list((tmp_path / "items").glob("item-batch-*.parquet"))) == 1
    assert item_id_for("000114036126006577", "2.03") == "000114036126006577-2-03"


def test_itemizer_handles_mixed_integer_and_empty_section_fields(
    tmp_path: Path,
) -> None:
    """Itemization should write rows even when some sections lack line numbers."""
    documents = pd.DataFrame(
        [
            {
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/full.txt",
                "text": """
ITEM INFORMATION: Other Events
ITEM INFORMATION: Not A Real Item
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body>
<p>Item 8.01 Other Events.</p>
<p>One extracted section.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p>
<p>Exhibit text.</p>
</body></html>
</TEXT>
</DOCUMENT>
""",
                "date": "2024-01-02",
            }
        ],
        columns=DOCUMENT_COLUMNS,
    )

    items = itemize_documents(documents, data_dir=tmp_path)

    assert len(items) == 2
    assert items["item"].to_list() == ["8.01", ""]
    assert items["start_line"].dtype.name == "Int64"
    assert items["end_line"].dtype.name == "Int64"
    assert items["section_char_count"].dtype.name == "Int64"
    assert len(list((tmp_path / "items").glob("item-batch-*.parquet"))) == 1


def test_itemize_pending_documents_processes_downloaded_database_rows(
    tmp_path: Path,
) -> None:
    """Database-backed itemization loads downloaded docs and marks them itemized."""
    resource_file = tmp_path / "source.txt"
    resource_file.write_text(
        """
ITEM INFORMATION: Other Events
Item 8.01 Other Events.
Database-backed item text.
SIGNATURES
""",
        encoding="utf-8",
    )
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        upsert_documents(
            conn,
            [
                {
                    "accession_number": "000114036126006577",
                    "cik": "320193",
                    "url": "https://sec.example/full.txt",
                    "resource_uri": str(resource_file),
                    "date": "2024-01-02",
                }
            ],
            batch_path=None,
            status="indexed",
        )
    finally:
        conn.close()

    first = itemize_pending_documents(data_dir=tmp_path, batch_size=1)
    second = itemize_pending_documents(data_dir=tmp_path, batch_size=1)
    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        statuses = [
            row[0]
            for row in conn.execute(
                "SELECT status FROM documents ORDER BY accession_number"
            )
        ]
    finally:
        conn.close()

    assert first["item_id"].to_list() == ["000114036126006577-8-01"]
    assert "Database-backed item text" in first["text"].to_list()[0]
    assert second.empty
    assert statuses == ["itemized"]
    assert len(list((tmp_path / "items").glob("item-batch-*.parquet"))) == 1


def test_itemize_documents_reuses_one_s3_client_per_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3-backed itemization should reuse one default client per invocation."""

    class FakeS3Client:
        def __init__(self: "FakeS3Client") -> None:
            self.requests: list[tuple[str, str]] = []

        def get_object(
            self: "FakeS3Client", Bucket: str, Key: str
        ) -> dict[str, BytesIO]:  # noqa: N803
            self.requests.append((Bucket, Key))
            return {
                "Body": BytesIO(
                    b"""
ITEM INFORMATION: Other Events
Item 8.01 Other Events.
Shared client text.
SIGNATURES
"""
                )
            }

    created_clients: list[FakeS3Client] = []

    def fake_default_s3_client() -> FakeS3Client:
        client = FakeS3Client()
        created_clients.append(client)
        return client

    monkeypatch.setattr(cdt.itemizer.core, "default_s3_client", fake_default_s3_client)

    documents = pd.DataFrame(
        [
            {
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/full-1.txt",
                "text": "",
                "date": "2024-01-02",
                "resource_uri": "s3://sec-bucket/doc-1.txt",
            },
            {
                "accession_number": "000114036126006578",
                "cik": "320193",
                "url": "https://sec.example/full-2.txt",
                "text": "",
                "date": "2024-01-03",
                "resource_uri": "s3://sec-bucket/doc-2.txt",
            },
        ]
    )

    items = itemize_documents(documents, data_dir=tmp_path)

    assert len(items) == 2
    assert len(created_clients) == 1
    assert created_clients[0].requests == [
        ("sec-bucket", "doc-1.txt"),
        ("sec-bucket", "doc-2.txt"),
    ]


def test_classifier_adds_binary_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classifier adds binary label, score, and relevance fields."""
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

    class FakeModel:
        def decision_function(self: "FakeModel", texts: list[str]) -> np.ndarray:
            return np.asarray([1.0 if "submission" in text else -1.0 for text in texts])

    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    first = classify_items(items, data_dir=tmp_path)

    assert first["label"].to_list() == ["relevant"]
    assert first["relevance"].to_list() == [True]
    assert first["classification_score"].to_list()[0] > 0.5


def test_extractor_returns_empty_tables_and_writes_metadata(tmp_path: Path) -> None:
    """The stub extractor records successful completion."""
    classified_items = pd.DataFrame(
        [{"item_id": "000114036126006577-0-00", "relevance": False}]
    )

    tables = extract_tables(classified_items, data_dir=tmp_path)

    assert tables == {}
    assert (tmp_path / "extracted_tables" / "_SUCCESS.json").exists()
