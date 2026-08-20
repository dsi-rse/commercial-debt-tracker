"""File-native stage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cdt.classifier import classifications_root, classify_pending_items
from cdt.classifier import core as classifier_core
from cdt.datasets import shard_for_accession
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.core import (
    ExtractionRowState,
    InstrumentIEStage,
    InstrumentRelationStage,
    is_rate_like_amount_text,
)
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import core as itemizer_core
from cdt.itemizer import itemize_pending_documents, items_root
from cdt.matcher import (
    debt_instruments_root,
    match_pending_mentions,
    mention_matches_root,
)
from cdt.matcher.core import coerce_optional_text, match_tables
from cdt.storage import artifact_exists, read_dataset, write_partition_table


class FakeModel:
    """Classifier stub returning a fixed relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a single strong-positive score."""
        del texts
        return [2.0]


def test_shard_for_accession_uses_eight_date_shards() -> None:
    """Date-partitioned stages should only use shards 0000 through 0007."""
    shards = {shard_for_accession(str(index)) for index in range(200)}
    assert shards == {f"{index:04d}" for index in range(8)}


def seed_document_partition(tmp_path: Path) -> str:
    """Write one canonical document partition."""
    table = pd.DataFrame(
        [
            {
                "accession_number": "000114036126006577",
                "cik": "320193",
                "company_name": "Example Inc.",
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


def seed_document_partitions(tmp_path: Path) -> list[str]:
    """Write multiple canonical document partitions."""
    partitions: list[tuple[str, str, str, str]] = [
        (
            "000114036126006577",
            "320193",
            "2024-01-02",
            "0001",
        ),
        (
            "000078901925000010",
            "789019",
            "2024-01-03",
            "0002",
        ),
    ]
    paths: list[str] = []
    for accession_number, cik, filing_date, shard in partitions:
        table = pd.DataFrame(
            [
                {
                    "accession_number": accession_number,
                    "cik": cik,
                    "company_name": "Example Inc.",
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
                    "date": filing_date,
                    "resource_uri": None,
                }
            ],
            columns=DOCUMENT_COLUMNS,
        )
        paths.append(
            write_partition_table(
                tmp_path / "documents",
                partition={"date": filing_date, "shard": shard},
                table=table,
            )
        )
    return paths


def build_mention_row(
    *,
    mention_id: str,
    item_id: str,
    accession_number: str,
    cik: str,
    date: str,
    name: str,
    start_date: str,
    amount: str,
    lenders_json: str = "[]",
) -> dict[str, object]:
    """Return one canonical mention row for matcher tests."""
    return {
        "debt_instrument_mention_id": mention_id,
        "item_id": item_id,
        "accession_number": accession_number,
        "cik": cik,
        "company_name": "Example Inc.",
        "date": date,
        "raw_id": "i-1",
        "name": name,
        "start_date": start_date,
        "end_date": None,
        "amount": amount,
        "amendment_of": None,
        "retired_of": None,
        "split_of": None,
        "lenders_json": lenders_json,
        "other_interested_parties_json": "[]",
        "name_json": "{}",
        "start_date_json": "{}",
        "end_date_json": "{}",
        "amount_json": "{}",
    }


def test_itemize_pending_documents_writes_canonical_partitions(tmp_path: Path) -> None:
    """Itemization should consume document partitions and write item partitions."""
    seed_document_partition(tmp_path)

    items = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)

    written = read_dataset(items_root(tmp_path))
    assert len(items) == 1
    assert written["item_id"].to_list() == ["000114036126006577-8-01"]


def test_itemize_pending_documents_drains_all_partitions(tmp_path: Path) -> None:
    """Itemization should process all pending partitions across chunks."""
    seed_document_partitions(tmp_path)

    items = itemize_pending_documents(artifact_root=tmp_path, batch_size=1)

    written = read_dataset(items_root(tmp_path))
    assert len(items) == 2
    assert sorted(written["item_id"].to_list()) == [
        "000078901925000010-8-01",
        "000114036126006577-8-01",
    ]


def test_itemize_pending_documents_skips_empty_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty itemize results should not write parquet and should not rerun."""
    seed_document_partition(tmp_path)
    calls = 0

    def fake_itemize_documents(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        del args, kwargs
        calls += 1
        return pd.DataFrame(columns=itemizer_core.ITEM_COLUMNS)

    monkeypatch.setattr(itemizer_core, "itemize_documents", fake_itemize_documents)

    first = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    second = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)

    assert first.empty
    assert second.empty
    assert calls == 1
    assert not artifact_exists(
        items_root(tmp_path) + "/date=2024-01-02/shard=0001/part-0000.parquet"
    )


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


def test_classify_pending_items_drains_all_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification should process all pending partitions across chunks."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    classified = classify_pending_items(artifact_root=tmp_path, batch_size=1)

    written = read_dataset(classifications_root(tmp_path))
    assert len(classified) == 2
    assert written["label"].to_list().count("relevant") == 2


def test_classify_pending_items_skips_empty_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty classifier results should not write parquet and should not rerun."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    calls = 0

    def fake_classify_items(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        del args, kwargs
        calls += 1
        return pd.DataFrame(columns=classifier_core.CLASSIFIED_ITEM_COLUMNS)

    monkeypatch.setattr(classifier_core, "classify_items", fake_classify_items)

    first = classify_pending_items(artifact_root=tmp_path, batch_size=5)
    second = classify_pending_items(artifact_root=tmp_path, batch_size=5)

    assert first.empty
    assert second.empty
    assert calls == 1
    assert not artifact_exists(
        classifications_root(tmp_path) + "/date=2024-01-02/shard=0001/part-0000.parquet"
    )


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
                "retired_of": None,
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


def test_extract_pending_items_drains_all_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction should process all pending partitions across chunks."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=1)

    async def fake_run_extraction_workflow(
        **kwargs: object,
    ) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {
                "debt_instrument_mention_id": f"m-{item_row['accession_number']}",
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
                "retired_of": None,
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
        batch_size=1,
        client=None,
    )

    written = read_dataset(mentions_root(tmp_path))
    assert len(mentions) == 2
    assert sorted(written["debt_instrument_mention_id"].to_list()) == [
        "m-000078901925000010",
        "m-000114036126006577",
    ]
    audit_files = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit_files) == 1


def test_extract_pending_items_skips_empty_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty extraction results should not write parquet and should not rerun."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)
    calls = 0

    async def fake_run_extraction_workflow(
        **kwargs: object,
    ) -> ExtractionRowState:
        nonlocal calls
        item_row = kwargs["item_row"]
        calls += 1
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.core.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    first = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )
    second = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )

    assert first.empty
    assert second.empty
    assert calls == 1
    assert not artifact_exists(
        mentions_root(tmp_path) + "/date=2024-01-02/shard=0001/part-0000.parquet"
    )


def test_instrument_ie_validate_allows_shared_evidence_and_skipped_collective_tags() -> (
    None
):
    """Shared evidence and skipped collective labels should validate successfully."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, the Company issued a
<debt_instrument id="tag-i-1">Senior Subordinated Convertible Promissory Note</debt_instrument>
(the <debt_instrument id="tag-i-2">Initial Exchange Note</debt_instrument>)
in an aggregate principal amount of <amount id="tag-a-1">$5.5 million</amount>
to <organization id="tag-o-1">EGT 11 LLC</organization>.
On <date id="tag-d-2">March 20, 2025</date>, the Company issued a
<debt_instrument id="tag-i-1b">Senior Subordinated Convertible Promissory Note</debt_instrument>
(the <debt_instrument id="tag-i-3">Subsequent Exchange Note</debt_instrument>)
in an aggregate principal amount of <amount id="tag-a-2">$269,000</amount>
to <organization id="tag-o-1b">EGT 11 LLC</organization>.
Together, the <debt_instrument id="tag-i-4">Exchange Notes</debt_instrument> were outstanding.
</body>
""".strip()
    response = """
[
  {
    "name": ["tag-i-1", "tag-i-2"],
    "start_date": {"evidence": ["tag-d-1"], "normalized_date": "2025-03-17"},
    "amount": {"evidence": ["tag-a-1"], "normalized_amount": "5500000", "currency": "USD"},
    "lenders": [["tag-o-1"]],
    "other_interested_parties": []
  },
  {
    "name": ["tag-i-1b", "tag-i-3"],
    "start_date": {"evidence": ["tag-d-2"], "normalized_date": "2025-03-20"},
    "amount": {"evidence": ["tag-a-2"], "normalized_amount": "269000", "currency": "USD"},
    "lenders": [["tag-o-1b"]],
    "other_interested_parties": []
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert failures == []


def test_instrument_ie_validate_rejects_conflicting_start_dates() -> None:
    """One extracted object cannot carry two distinct start dates."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The <debt_instrument id="tag-i-1">Senior Subordinated Convertible Promissory Note</debt_instrument>
was issued on <date id="tag-d-1">March 17, 2025</date> and <date id="tag-d-2">March 20, 2025</date>.
</body>
""".strip()
    response = """
[
  {
    "name": ["tag-i-1"],
    "start_date": {"evidence": ["tag-d-1", "tag-d-2"], "normalized_date": "2025-03-17"}
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert any("multiple distinct normalized values" in failure for failure in failures)


RATE_AMOUNT_XML = """
<body>
<debt_instrument id="tag-i-1">ABR Loan</debt_instrument> borrowings bear interest at
<amount id="tag-a-rate">0.875% per annum</amount> and the facility provides for
<amount id="tag-a-principal">$500.0 million</amount> of commitments.
</body>
""".strip()


def rate_amount_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state with both a rate and a principal amount."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_AMOUNT_XML
    return row_state


def test_is_rate_like_amount_text_separates_rates_from_principal() -> None:
    """Rate detection should key on percentages without currency or scale words."""
    assert is_rate_like_amount_text("0.875% per annum") is True
    assert is_rate_like_amount_text("5.75%") is True
    assert is_rate_like_amount_text("175 basis points") is True
    assert is_rate_like_amount_text("$500.0 million") is False
    assert is_rate_like_amount_text("$500,000,000 (100% of principal)") is False
    assert is_rate_like_amount_text("100% of the outstanding 30.0 million") is False
    assert is_rate_like_amount_text(None) is False


def test_instrument_ie_validate_rejects_rate_only_amount_evidence() -> None:
    """An amount citing only a rate should fail validation and retry."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-rate"],
                    "normalized_amount": "0.875",
                    "currency": None,
                },
            }
        ]
    )

    failures = InstrumentIEStage().validate(rate_amount_row_state(), response)

    assert any(
        "describes an interest rate, margin, or fee" in failure for failure in failures
    )


def test_instrument_ie_validate_accepts_principal_amount_evidence() -> None:
    """A principal amount should still validate."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-principal"],
                    "normalized_amount": "500000000",
                    "currency": "USD",
                },
            }
        ]
    )

    assert InstrumentIEStage().validate(rate_amount_row_state(), response) == []


def test_instrument_ie_postprocess_drops_rate_amount() -> None:
    """A rate that slips past validation should not be stored as an amount."""
    row_state = rate_amount_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-rate"],
                    "normalized_amount": "0.875",
                    "currency": "USD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amount_json"]))
    assert mention["amount"] is None
    assert payload["normalized_amount"] is None
    assert payload["currency"] is None
    # Evidence is preserved so the dropped value stays auditable.
    assert payload["tag_ids"] == ["tag-a-rate"]


def test_instrument_relation_stage_accepts_retired_of() -> None:
    """Relation validation and postprocessing should support retired_of."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_relation",
    )
    row_state.debt_instrument_mentions = [
        {
            "debt_instrument_mention_id": "m-1",
            "raw_id": "i-1",
        },
        {
            "debt_instrument_mention_id": "m-2",
            "raw_id": "i-2",
        },
    ]
    response = '[{"from": "i-2", "to": "i-1", "type": "retired_of"}]'

    failures = InstrumentRelationStage().validate(row_state, response)
    assert failures == []

    row_state.stage_responses["instrument_relation"] = response
    InstrumentRelationStage().postprocess(row_state)

    assert row_state.debt_instrument_mentions[1]["retired_of"] == "m-1"


def test_match_pending_mentions_writes_match_datasets(tmp_path: Path) -> None:
    """Matcher should consume mention dataset and write match outputs."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                lenders_json='[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]',
            )
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
    assert written_matches["edge_type"].to_list() == ["member"]
    assert written_instruments["debt_instrument_id"].to_list() == ["m-1"]


def test_match_pending_mentions_drains_all_shards(tmp_path: Path) -> None:
    """Matcher should process all shard groups across chunks."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                lenders_json='[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]',
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="789019",
                date="2024-01-03",
                name="Revolving Credit Facility",
                start_date="2024-01-01",
                amount="$250 million",
                lenders_json='[{"mentions": [{"text": "Contoso Bank"}], "tag_ids": ["tag-l-2"]}]',
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows.iloc[[0]],
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-03", "shard": "0002"},
        table=mention_rows.iloc[[1]],
    )

    tables = match_pending_mentions(artifact_root=tmp_path, batch_size=1)

    written_matches = read_dataset(mention_matches_root(tmp_path))
    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert len(tables["debt_instrument_mentions"]) == 2
    assert sorted(written_matches["debt_instrument_mention_id"].to_list()) == [
        "m-1",
        "m-2",
    ]
    assert sorted(written_instruments["debt_instrument_id"].to_list()) == [
        "m-1",
        "m-2",
    ]


def test_match_pending_mentions_force_rebuilds_existing_memberships(
    tmp_path: Path,
) -> None:
    """Force reruns should discard stale member assignments for the shard."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="Alpha Loan",
                start_date="2024-01-01",
                amount="$100 million",
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2024-01-02",
                name="Beta Facility",
                start_date="2024-01-01",
                amount="$100 million",
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-01", "shard": "0001"},
        table=mention_rows.iloc[[0]],
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows.iloc[[1]],
    )

    first = match_pending_mentions(
        artifact_root=tmp_path,
        batch_size=5,
        strong_match_threshold=0.75,
        loose_match_threshold=0.75,
    )
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in first["debt_instrument_mentions"]
        .query("edge_type == 'member'")
        .to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-1"}

    second = match_pending_mentions(
        artifact_root=tmp_path,
        batch_size=5,
        force=True,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    written_matches = read_dataset(mention_matches_root(tmp_path)).sort_values(
        ["debt_instrument_mention_id", "edge_type", "debt_instrument_id"]
    )
    written_instruments = read_dataset(debt_instruments_root(tmp_path)).sort_values(
        "debt_instrument_id"
    )
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in second["debt_instrument_mentions"]
        .query("edge_type == 'member'")
        .to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-2"}
    assert written_matches["edge_type"].to_list() == ["member", "member", "related"]
    assert written_matches["debt_instrument_id"].to_list() == ["m-1", "m-2", "m-1"]
    assert written_instruments["debt_instrument_id"].to_list() == ["m-1", "m-2"]


def test_match_tables_supports_incremental_batches_against_existing_clusters() -> None:
    """Delta batches should match against existing clusters without full history."""
    existing_mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="Alpha Loan",
                start_date="2024-01-01",
                amount="$100 million",
                lenders_json='[{"mentions": [{"text": "Acme Bank"}]}]',
            )
        ]
    )
    existing_tables = match_tables(existing_mentions)

    new_mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2024-01-02",
                name="Alpha Loan",
                start_date="2024-01-01",
                amount="$100 million",
                lenders_json='[{"mentions": [{"text": "Acme Bank"}]}]',
            )
        ]
    )
    tables = match_tables(
        new_mentions,
        existing_edges=existing_tables["debt_instrument_mentions"],
        existing_instruments=existing_tables["debt_instrument"],
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-1"}
    assert tables["debt_instrument"]["debt_instrument_id"].to_list() == ["m-1"]
    assert tables["debt_instrument"]["name"].to_list() == ["Alpha Loan"]
    assert tables["debt_instrument"]["company_name"].to_list() == ["Example Inc."]


def test_coerce_optional_text_treats_pandas_nan_as_missing() -> None:
    """Matcher text coercion should drop pandas null sentinels."""
    assert coerce_optional_text(pd.NA) is None
    assert coerce_optional_text(float("nan")) is None


def test_match_tables_does_not_emit_literal_nan_company_names() -> None:
    """Matched instruments should keep missing filer names as null, not 'nan'."""
    mention = build_mention_row(
        mention_id="m-1",
        item_id="item-1",
        accession_number="0001",
        cik="320193",
        date="2024-01-01",
        name="Alpha Loan",
        start_date="2024-01-01",
        amount="$100 million",
        lenders_json='[{"mentions": [{"text": "Acme Bank"}]}]',
    )
    mention["company_name"] = pd.NA

    mentions = pd.DataFrame([mention])

    tables = match_tables(mentions)

    assert tables["debt_instrument"]["company_name"].to_list() == [None]


def test_match_tables_retired_of_keeps_separate_clusters_and_updates_parent_end_date() -> (
    None
):
    """Retirement lineage should not collapse into one cluster and should end-date the parent."""
    mentions = pd.DataFrame(
        [
            {
                **build_mention_row(
                    mention_id="m-1",
                    item_id="item-1",
                    accession_number="0001",
                    cik="320193",
                    date="2024-01-01",
                    name="Term Loan",
                    start_date="2024-01-01",
                    amount="$100 million",
                    lenders_json='[{"mentions": [{"text": "Acme Bank"}]}]',
                ),
                "end_date": None,
            },
            {
                **build_mention_row(
                    mention_id="m-2",
                    item_id="item-2",
                    accession_number="0002",
                    cik="320193",
                    date="2024-03-01",
                    name="Term Loan",
                    start_date="2024-01-01",
                    amount="$100 million",
                    lenders_json='[{"mentions": [{"text": "Acme Bank"}]}]',
                ),
                "end_date": "2024-03-01",
                "retired_of": "m-1",
            },
        ]
    )

    tables = match_tables(
        mentions,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-2"}

    instruments = {
        row["debt_instrument_id"]: row
        for row in tables["debt_instrument"].to_dict("records")
    }
    assert instruments["m-2"]["retired_of_debt_instrument_id"] == "m-1"
    assert instruments["m-1"]["end_date"] == "2024-03-01"
