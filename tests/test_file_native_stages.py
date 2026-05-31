"""File-native stage tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cdt.classifier import classifications_root, classify_pending_items
from cdt.classifier import core as classifier_core
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.core import ExtractionRowState
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import itemize_pending_documents, items_root
from cdt.matcher import (
    debt_instruments_root,
    match_pending_mentions,
    mention_matches_root,
)
from cdt.storage import read_dataset, write_partition_table


class FakeModel:
    """Classifier stub returning a fixed relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a single strong-positive score."""
        del texts
        return [2.0]


def seed_document_partition(tmp_path: Path) -> str:
    """Write one canonical document partition."""
    table = pd.DataFrame(
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
</TEXT>
</DOCUMENT>
""",
                "date": "2024-01-02",
                "resource_uri": None,
            }
        ],
        columns=DOCUMENT_COLUMNS,
    )
    return write_partition_table(
        tmp_path / "documents",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=table,
    )


def test_itemize_pending_documents_writes_canonical_partitions(tmp_path: Path) -> None:
    """Itemization should consume document partitions and write item partitions."""
    seed_document_partition(tmp_path)

    items = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)

    written = read_dataset(items_root(tmp_path))
    assert len(items) == 1
    assert written["item_id"].to_list() == ["000114036126006577-8-01"]


def test_classify_pending_items_writes_canonical_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification should consume item partitions and write classification partitions."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    classified = classify_pending_items(artifact_root=tmp_path, batch_size=5)

    written = read_dataset(classifications_root(tmp_path))
    assert len(classified) == 1
    assert written["label"].to_list() == ["relevant"]
    assert written["relevance"].to_list() == [True]


def test_extract_pending_items_writes_mentions_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction should consume classifications and write mentions plus audit log."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

    async def fake_run_extraction_workflow(
        **kwargs: object,
    ) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {
                "debt_instrument_mention_id": "m-1",
                "item_id": item_row["item_id"],
                "accession_number": item_row["accession_number"],
                "cik": item_row["cik"],
                "date": item_row["date"],
                "raw_id": "i-1",
                "name": "Term Loan",
                "start_date": "2024-01-01",
                "end_date": None,
                "amount": "$100 million",
                "amendment_of": None,
                "split_of": None,
                "lenders_json": "[]",
                "other_interested_parties_json": "[]",
                "name_json": "{}",
                "start_date_json": "{}",
                "end_date_json": "{}",
                "amount_json": "{}",
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.core.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    mentions = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )

    written = read_dataset(mentions_root(tmp_path))
    assert len(mentions) == 1
    assert written["debt_instrument_mention_id"].to_list()
    audit_files = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit_files) == 1


def test_match_pending_mentions_writes_match_datasets(tmp_path: Path) -> None:
    """Matcher should consume mention dataset and write match outputs."""
    mention_rows = pd.DataFrame(
        [
            {
                "debt_instrument_mention_id": "m-1",
                "item_id": "item-1",
                "accession_number": "0001",
                "cik": "320193",
                "date": "2024-01-02",
                "raw_id": "i-1",
                "name": "Term Loan",
                "start_date": "2024-01-01",
                "end_date": None,
                "amount": "$100 million",
                "amendment_of": None,
                "split_of": None,
                "lenders_json": '[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]',
                "other_interested_parties_json": "[]",
                "name_json": "{}",
                "start_date_json": "{}",
                "end_date_json": "{}",
                "amount_json": "{}",
            }
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows,
    )

    tables = match_pending_mentions(artifact_root=tmp_path, batch_size=5)

    written_matches = read_dataset(mention_matches_root(tmp_path))
    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert len(tables["debt_instrument_mentions"]) == 1
    assert written_matches["matcher_status"].to_list() == ["singleton"]
    assert written_instruments["debt_instrument_id"].to_list() == ["m-1"]
