"""File-native stage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cdt.classifier import classifications_root, classify_pending_items
from cdt.classifier import core as classifier_core
from cdt.datasets import (
    completion_registry_path,
    existing_date_shard_partition_ids,
    load_row_failures,
    shard_for_accession,
)
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.core import (
    ExtractionRowState,
    InstrumentIEStage,
    InstrumentRelationStage,
    is_rate_like_amount_text,
    load_prompt,
    normalized_maturity_from_text,
)
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import core as itemizer_core
from cdt.itemizer import itemize_pending_documents, items_root
from cdt.matcher import (
    debt_instruments_root,
    match_pending_mentions,
    mention_matches_root,
)
from cdt.matcher.core import (
    coerce_optional_text,
    company_names_by_cik,
    lender_signature,
    match_tables,
)
from cdt.pipeline import normalize_snapshot_text
from cdt.storage import (
    artifact_exists,
    coerce_dataset_text,
    read_dataset,
    read_json_artifact,
    read_table,
    write_partition_table,
)


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
    lenders_known_incomplete: bool = False,
    company_name: str | None = "Example Inc.",
) -> dict[str, object]:
    """Return one canonical mention row for matcher tests."""
    return {
        "debt_instrument_mention_id": mention_id,
        "item_id": item_id,
        "accession_number": accession_number,
        "cik": cik,
        "company_name": company_name,
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
        "lenders_known_incomplete": lenders_known_incomplete,
        "other_interested_parties_json": "[]",
        "name_json": "{}",
        "start_date_json": "{}",
        "end_date_json": "{}",
        "amount_json": "{}",
    }


def test_coerce_dataset_text_treats_placeholder_values_as_missing() -> None:
    """Parquet placeholders must never survive as real text values."""
    assert coerce_dataset_text(float("nan")) is None
    assert coerce_dataset_text(None) is None
    assert coerce_dataset_text("nan") is None
    assert coerce_dataset_text("N/A") is None
    assert coerce_dataset_text("  ") is None
    assert (
        coerce_dataset_text(" Appreciate Holdings, Inc. ")
        == "Appreciate Holdings, Inc."
    )
    assert coerce_dataset_text("Nantucket Bank") == "Nantucket Bank"


def test_itemize_document_record_blanks_missing_company_name() -> None:
    """A missing document company name must not become the literal text 'nan'."""
    document = {
        "accession_number": "000114036126006577",
        "cik": "1821075",
        "company_name": float("nan"),
        "url": "https://sec.example/full.txt",
        "date": "2026-01-02",
        "text": """
ITEM INFORMATION: Other Events
<DOCUMENT>
<TYPE>8-K
<TEXT>
Item 8.01 Other Events.
The Company issued a promissory note.
</TEXT>
</DOCUMENT>
""".strip(),
    }

    sections = itemizer_core.itemize_document_record(document)

    assert sections
    assert {section.company_name for section in sections} == {""}


def test_normalize_snapshot_text_nulls_placeholder_strings() -> None:
    """Dashboard-facing snapshots must not carry literal placeholder text."""
    table = pd.DataFrame(
        [
            {"company_name": "nan", "name": "convertible debentures", "amount": 1.5},
            {"company_name": "Versigent PLC", "name": "None", "amount": 2.5},
        ]
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["company_name"].to_list() == [None, "Versigent PLC"]
    assert normalized["name"].to_list() == ["convertible debentures", None]
    assert normalized["amount"].to_list() == [1.5, 2.5]


def test_normalize_snapshot_text_keeps_non_text_values_typed() -> None:
    """Booleans in an object column must not be published as text."""
    # Partitions written before lenders_known_incomplete existed leave an object
    # column holding booleans and nulls side by side.
    table = pd.DataFrame(
        {
            "lenders_known_incomplete": [True, None, False],
            "company_name": ["Acme Inc.", "nan", "Contoso Ltd."],
        }
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["lenders_known_incomplete"].to_list() == [True, None, False]
    assert normalized["company_name"].to_list() == [
        "Acme Inc.",
        None,
        "Contoso Ltd.",
    ]


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
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
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
                "lenders_known_incomplete": False,
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
                "lenders_known_incomplete": True,
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
    "lenders": [{"tag_ids": ["tag-o-1"], "kind": "named"}],
    "other_interested_parties": []
  },
  {
    "name": ["tag-i-1b", "tag-i-3"],
    "start_date": {"evidence": ["tag-d-2"], "normalized_date": "2025-03-20"},
    "amount": {"evidence": ["tag-a-2"], "normalized_amount": "269000", "currency": "USD"},
    "lenders": [{"tag_ids": ["tag-o-1b"], "kind": "named"}],
    "other_interested_parties": []
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert failures == []


PARTY_ROLE_XML = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, <organization id="tag-o-borrower">Example Inc.</organization>
entered into a <debt_instrument id="tag-i-1">Term Loan</debt_instrument> with
<organization id="tag-o-named">JPMorgan Chase Bank, N.A.</organization> and
<organization id="tag-o-collective">the other lenders party thereto</organization>, with
<organization id="tag-o-agent">Wells Fargo Bank, National Association</organization> as administrative agent.
</body>
""".strip()


def party_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state seeded with party-role tagged XML."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    return row_state


def instrument_ie_mention(response: str) -> dict[str, object]:
    """Run instrument_ie postprocessing on one response and return its mention."""
    row_state = party_row_state()
    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert len(row_state.debt_instrument_mentions) == 1
    return row_state.debt_instrument_mentions[0]


def test_instrument_ie_validate_accepts_party_kinds_and_roles() -> None:
    """Annotated lender and other-party clusters should validate."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "lenders": [
                    {"tag_ids": ["tag-o-named"], "kind": "named"},
                    {"tag_ids": ["tag-o-collective"], "kind": "collective"},
                ],
                "lenders_known_incomplete": True,
                "other_interested_parties": [
                    {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    {"tag_ids": ["tag-o-borrower"], "role": "borrower"},
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(party_row_state(), response) == []


def test_instrument_ie_validate_rejects_unannotated_party_clusters() -> None:
    """Bare tag-id lists and unknown annotations should fail validation."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "lenders": [["tag-o-named"]],
                "other_interested_parties": [
                    {"tag_ids": ["tag-o-agent"], "role": "servicer"}
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "'lenders'[0] must be an object with 'tag_ids' and 'kind' keys." in failure
        for failure in failures
    )
    assert any("'role' must be one of" in failure for failure in failures)


def test_instrument_ie_validate_rejects_non_boolean_lenders_known_incomplete() -> None:
    """The lenders_known_incomplete flag must be boolean when present."""
    response = json.dumps([{"name": ["tag-i-1"], "lenders_known_incomplete": "yes"}])

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "'lenders_known_incomplete' must be true or false" in failure
        for failure in failures
    )


def test_instrument_ie_postprocess_keeps_named_lenders_and_flags_incompleteness() -> (
    None
):
    """A named lender plus a collective phrase should keep only the named lender."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [
                        {"tag_ids": ["tag-o-named"], "kind": "named"},
                        {"tag_ids": ["tag-o-collective"], "kind": "collective"},
                    ],
                    "lenders_known_incomplete": True,
                    "other_interested_parties": [
                        {"tag_ids": ["tag-o-agent"], "role": "agent"}
                    ],
                }
            ]
        )
    )

    lenders = json.loads(str(mention["lenders_json"]))
    other_parties = json.loads(str(mention["other_interested_parties_json"]))
    assert [cluster["tag_ids"] for cluster in lenders] == [["tag-o-named"]]
    assert mention["lenders_known_incomplete"] is True
    # Persisted clusters keep the plain tag_ids/mentions shape: the model's kind and
    # role labels decide what is stored and are not themselves stored.
    assert set(lenders[0]) == {"tag_ids", "mentions"}
    assert [cluster["tag_ids"] for cluster in other_parties] == [["tag-o-agent"]]
    assert set(other_parties[0]) == {"tag_ids", "mentions"}


def test_instrument_ie_postprocess_leaves_named_only_lenders_unflagged() -> None:
    """A lender list with only named clusters is not known to be incomplete."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [{"tag_ids": ["tag-o-named"], "kind": "named"}],
                }
            ]
        )
    )

    assert mention["lenders_known_incomplete"] is False
    assert len(json.loads(str(mention["lenders_json"]))) == 1


def test_instrument_ie_postprocess_honors_declared_incompleteness() -> None:
    """A model-declared flag survives even when every cluster is named."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [{"tag_ids": ["tag-o-named"], "kind": "named"}],
                    "lenders_known_incomplete": True,
                }
            ]
        )
    )

    assert mention["lenders_known_incomplete"] is True
    assert len(json.loads(str(mention["lenders_json"]))) == 1


def test_instrument_ie_postprocess_drops_collective_only_lenders() -> None:
    """A collective-only lender list carries no lenders and is flagged."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [
                        {"tag_ids": ["tag-o-collective"], "kind": "collective"}
                    ],
                }
            ]
        )
    )

    assert json.loads(str(mention["lenders_json"])) == []
    assert mention["lenders_known_incomplete"] is True


def test_instrument_ie_postprocess_excludes_the_borrower_from_other_parties() -> None:
    """The filer itself is not persisted as an interested party."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "other_interested_parties": [
                        {"tag_ids": ["tag-o-borrower"], "role": "borrower"},
                        {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    ],
                }
            ]
        )
    )

    other_parties = json.loads(str(mention["other_interested_parties_json"]))
    assert [cluster["tag_ids"] for cluster in other_parties] == [["tag-o-agent"]]


def test_lender_signature_prefers_the_named_party_over_an_alias() -> None:
    """A defined-term alias in the cluster must not hide the party it names."""
    payload = json.dumps([{"mentions": [{"text": "Purchasers"}, {"text": "Oaktree"}]}])

    assert lender_signature(payload) == "oaktree"


def test_lender_signature_uses_stored_lender_clusters() -> None:
    """Lender signatures come from the persisted named clusters."""
    payload = json.dumps([{"mentions": [{"text": "Acme Bank"}]}])

    assert lender_signature(payload) == "acme bank"


def test_match_pending_mentions_carries_lender_incompleteness(tmp_path: Path) -> None:
    """Matcher output should carry mention-level lender incompleteness forward."""
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
                lenders_json=(
                    '[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]'
                ),
                lenders_known_incomplete=True,
            )
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows,
    )

    match_pending_mentions(artifact_root=tmp_path, batch_size=5)

    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert written_instruments["lenders_known_incomplete"].to_list() == [True]


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


MATURITY_IN_NAME_XML = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, the Company issued
<debt_instrument id="tag-i-1">3.875% senior notes due 2028</debt_instrument>
in an aggregate principal amount of <amount id="tag-a-1">$500 million</amount>.
</body>
""".strip()


def maturity_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state whose instrument name carries a maturity."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = MATURITY_IN_NAME_XML
    return row_state


def maturity_mention(response: str) -> dict[str, object]:
    """Run instrument_ie postprocessing on one response and return its mention."""
    row_state = maturity_row_state()
    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert len(row_state.debt_instrument_mentions) == 1
    return row_state.debt_instrument_mentions[0]


def test_normalized_maturity_from_text_parses_due_phrases() -> None:
    """Maturity parsing should cover year-only, full-date, and ambiguous names."""
    assert normalized_maturity_from_text("3.875% senior notes due 2028") == "2028-12-31"
    assert (
        normalized_maturity_from_text("senior notes due October 1, 2028")
        == "2028-10-01"
    )
    assert normalized_maturity_from_text("notes due in 2030") == "2030-12-31"
    assert normalized_maturity_from_text("notes due 2028 and notes due 2031") is None
    assert normalized_maturity_from_text("3.875% senior notes") is None
    assert normalized_maturity_from_text("Series 2025-B Notes") is None


def test_instrument_ie_validate_accepts_name_span_as_end_date_evidence() -> None:
    """The instrument name span is valid end_date evidence when maturity is embedded."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "end_date": {
                    "evidence": ["tag-i-1"],
                    "normalized_date": "2028-12-31",
                    "derived_from_name": True,
                },
            }
        ]
    )

    assert InstrumentIEStage().validate(maturity_row_state(), response) == []


def test_instrument_ie_validate_still_rejects_name_span_as_start_date_evidence() -> (
    None
):
    """Only end_date may cite a debt_instrument tag."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "start_date": {
                    "evidence": ["tag-i-1"],
                    "normalized_date": "2028-12-31",
                },
            }
        ]
    )

    failures = InstrumentIEStage().validate(maturity_row_state(), response)

    assert any("expected date" in failure for failure in failures)


def test_instrument_ie_postprocess_keeps_name_derived_end_date() -> None:
    """A maturity cited from the name span should survive normalization."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "end_date": {
                        "evidence": ["tag-i-1"],
                        "normalized_date": "2028-12-31",
                    },
                }
            ]
        )
    )

    assert mention["end_date"] == "2028-12-31"
    payload = json.loads(str(mention["end_date_json"]))
    assert payload["tag_ids"] == ["tag-i-1"]


def test_instrument_ie_postprocess_backfills_end_date_from_name() -> None:
    """A missing end_date should fall back to the maturity in the instrument name."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "start_date": {
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2025-03-17",
                    },
                }
            ]
        )
    )

    assert mention["end_date"] == "2028-12-31"
    payload = json.loads(str(mention["end_date_json"]))
    # A maturity read from the name has no citable date tag of its own.
    assert payload["tag_ids"] == []


def test_instrument_ie_postprocess_leaves_end_date_null_without_maturity() -> None:
    """Names without a maturity phrase should not gain an invented end date."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The Company issued <debt_instrument id="tag-i-1">3.875% senior notes</debt_instrument>
on <date id="tag-d-1">March 17, 2025</date>.
</body>
""".strip()
    row_state.stage_responses["instrument_ie"] = json.dumps([{"name": ["tag-i-1"]}])

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["end_date"] is None
    assert json.loads(str(mention["end_date_json"]))["normalized_date"] is None


def test_instrument_ie_postprocess_drops_end_date_that_contradicts_evidence() -> None:
    """A normalized date that does not match its evidence text is still dropped."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "end_date": {
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2028-12-31",
                    },
                }
            ]
        )
    )

    payload = json.loads(str(mention["end_date_json"]))
    assert payload["tag_ids"] == ["tag-d-1"]
    # The cited date tag says March 17, 2025, so the model value is rejected and the
    # name maturity fills the gap instead.
    assert mention["end_date"] == "2028-12-31"


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


def test_instrument_ie_postprocess_normalizes_wrapped_names() -> None:
    """A name wrapped across lines is stored collapsed, verbatim in name_json."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        "<body>The Company entered into a "
        '<debt_instrument id="tag-i-1">revolving credit\nfacility</debt_instrument> '
        "and issued "
        '<debt_instrument id="tag-i-2">4.85% Remarketable\tSenior\xa0Notes '
        "due 2032</debt_instrument>.</body>"
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"]}, {"name": ["tag-i-2"]}]
    )

    InstrumentIEStage().postprocess(row_state)

    names = [mention["name"] for mention in row_state.debt_instrument_mentions]
    assert names == [
        "revolving credit facility",
        "4.85% Remarketable Senior Notes due 2032",
    ]
    # Provenance keeps the verbatim span, since the char offsets index into it.
    verbatim = [
        json.loads(str(mention["name_json"]))["mentions"][0]["text"]
        for mention in row_state.debt_instrument_mentions
    ]
    assert verbatim == [
        "revolving credit\nfacility",
        "4.85% Remarketable\tSenior\xa0Notes due 2032",
    ]


def test_instrument_ie_postprocess_drops_duplicate_identical_mentions() -> None:
    """Objects that differ in no extracted property must not duplicate a mention row."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The trust issued
<debt_instrument id="tag-i-1">Class A-1 Notes, Class A-2 Notes, and Class A-3 Notes</debt_instrument>.
</body>
""".strip()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"]}, {"name": ["tag-i-1"]}, {"name": ["tag-i-1"]}]
    )

    InstrumentIEStage().postprocess(row_state)

    mentions = row_state.debt_instrument_mentions
    assert len(mentions) == 1
    assert mentions[0]["raw_id"] == "i-1"


def test_instrument_ie_prompt_requires_one_object_per_class() -> None:
    """The IE prompt must keep telling the model to split multi-class offerings."""
    prompt = load_prompt("instrument_ie")

    assert "one object per class, tranche, or series" in prompt
    assert "Class A-1" in prompt


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


def test_company_names_by_cik_takes_the_newest_known_name() -> None:
    """CIK name resolution should ignore missing values and prefer newer filings."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="2078008",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                company_name=None,
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="2078008",
                date="2024-02-02",
                name="Revolver",
                start_date="2024-02-01",
                amount="$50 million",
                company_name="Versigent PLC",
            ),
            build_mention_row(
                mention_id="m-3",
                item_id="item-3",
                accession_number="0003",
                cik="320193",
                date="2024-03-02",
                name="Senior Notes",
                start_date="2024-03-01",
                amount="$1 billion",
            ),
        ]
    )

    assert company_names_by_cik(mention_rows) == {
        "2078008": "Versigent PLC",
        "320193": "Example Inc.",
    }


def test_match_tables_backfills_company_name_from_cik() -> None:
    """An instrument seeded by a mention without display metadata is still named."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="2078008",
                date="2024-01-02",
                name="6.125% senior unsecured notes due 2031",
                start_date="2024-01-01",
                amount="$400 million",
                company_name=None,
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="2078008",
                date="2024-02-02",
                name="Revolving Credit Facility",
                start_date="2024-02-01",
                amount="$50 million",
                company_name="Versigent PLC",
            ),
        ]
    )

    tables = match_tables(mention_rows)

    instruments = tables["debt_instrument"].set_index("debt_instrument_id")
    assert len(instruments) == 2
    assert instruments.loc["m-1", "company_name"] == "Versigent PLC"
    assert instruments.loc["m-2", "company_name"] == "Versigent PLC"


def test_coerce_optional_text_treats_nan_like_text_as_missing() -> None:
    """Literal placeholder strings must never reach a dashboard-facing column."""
    assert coerce_optional_text("nan") is None
    assert coerce_optional_text("NaN") is None
    assert coerce_optional_text("None") is None
    assert coerce_optional_text("N/A") is None
    assert coerce_optional_text("  ") is None
    assert coerce_optional_text("Nantucket Bank") == "Nantucket Bank"


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


def test_retry_includes_prior_response_as_assistant_turn() -> None:
    """Retry conversations must include the failed output the retry references."""
    row_state = ExtractionRowState(item_row={"item_id": "item-1"}, stage_name="ner")
    row_state.add_messages([{"role": "user", "content": "tag this filing"}])
    row_state.add_response("<bad-xml>")
    row_state.add_validation(["unclosed tag"])
    row_state.retry("Your previous NER output failed validation: unclosed tag")
    assert row_state.current_attempt.messages == [
        {"role": "user", "content": "tag this filing"},
        {"role": "assistant", "content": "<bad-xml>"},
        {
            "role": "user",
            "content": "Your previous NER output failed validation: unclosed tag",
        },
    ]


def test_extract_failures_are_recorded_and_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped row lands in the failure registry; a later success clears it."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

    async def failing_workflow(**kwargs: object) -> ExtractionRowState:
        row_state = ExtractionRowState(
            item_row=kwargs["item_row"], stage_name="instrument_ie"
        )
        row_state.add_validation(["instrument_ie did not return valid JSON"])
        row_state.finish("ERROR")
        return row_state

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", failing_workflow)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # The partition is registered complete even though the row produced nothing,
    # which is exactly why the failure has to be recorded somewhere durable.
    assert (
        "extract"
        in read_json_artifact(
            completion_registry_path("extract", artifact_root=tmp_path)
        )["stage"]
    )
    failures = load_row_failures("extract", artifact_root=tmp_path)
    assert len(failures) == 1
    entry = next(iter(failures.values()))
    assert entry["backend"] == "live"
    assert entry["state"] == "ERROR"
    assert entry["error"] == "instrument_ie did not return valid JSON"
    assert entry["date"] and entry["shard"]

    async def succeeding_workflow(**kwargs: object) -> ExtractionRowState:
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
                "start_date": None,
                "end_date": None,
                "amount": None,
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
        "cdt.extractor.core.run_extraction_workflow", succeeding_workflow
    )
    extract_pending_items(artifact_root=tmp_path, batch_size=5, force=True, client=None)

    assert load_row_failures("extract", artifact_root=tmp_path) == {}


def test_existing_date_shard_partition_ids_lists_written_partitions(
    tmp_path: Path,
) -> None:
    """The one-LIST partition-id set matches exactly what was written (#83)."""
    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-03-04", "shard": "0001"},
        table=table,
    )

    ids = existing_date_shard_partition_ids("items", artifact_root=str(root))

    assert ids == {("2026-01-02", "0007"), ("2026-03-04", "0001")}
    assert (
        existing_date_shard_partition_ids("mentions", artifact_root=str(root)) == set()
    )


def test_iter_date_shard_partitions_skips_orphaned_tempfiles(tmp_path: Path) -> None:
    """A tempfile left by a crash between create and rename must not brick the scan (#68)."""
    from cdt.datasets import iter_date_shard_partitions

    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (
        root / "items" / "date=2026-01-02" / "shard=0007" / "tmpabc123.parquet"
    ).write_bytes(b"")

    paths = iter_date_shard_partitions("items", artifact_root=str(root))

    assert len(paths) == 1
    assert paths[0].endswith("date=2026-01-02/shard=0007/part-0000.parquet")


def test_iter_date_shard_partitions_raises_on_non_canonical_data(
    tmp_path: Path,
) -> None:
    """Real data laid out wrong must fail loudly, not silently empty the run.

    Skipping a pre-migration flat file would let every stage process nothing
    and exit 0 while ingest keeps counting the flat file's rows as ingested —
    those filings would be invisible to the pipeline forever.
    """
    from cdt.datasets import iter_date_shard_partitions

    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (root / "items" / "items.parquet").write_bytes(b"")

    with pytest.raises(ValueError, match="Non-canonical parquet file"):
        iter_date_shard_partitions("items", artifact_root=str(root))


def test_read_dataset_skips_orphaned_tempfiles(tmp_path: Path) -> None:
    """Every read path, not just the partition scan, must survive an orphan.

    ingest's existing-accession scan, its per-partition merge, the matcher, and
    pipeline finalize all read through read_dataset; a zero-byte tmp*.parquet
    orphan previously made each of them raise ArrowInvalid.
    """
    root = tmp_path / "artifacts" / "items"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (root / "date=2026-01-02" / "shard=0007" / "tmpabc123.parquet").write_bytes(b"")

    read = read_dataset(str(root))

    assert read["item_id"].to_list() == ["a"]


def test_pending_source_partitions_skips_orphans_and_raises_on_flat_files(
    tmp_path: Path,
) -> None:
    """Fingerprint work selection follows the same stray contract as the scan.

    Silently dropping a mis-laid-out real file here would run the stage on
    nothing while ingest keeps counting the file's rows as ingested.
    """
    from cdt.datasets import pending_source_partitions

    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (
        root / "items" / "date=2026-01-02" / "shard=0007" / "tmpabc123.parquet"
    ).write_bytes(b"")

    pending, _ = pending_source_partitions(
        "classify", "items", "classifications", artifact_root=str(root)
    )

    assert len(pending) == 1
    assert pending[0][0].endswith("date=2026-01-02/shard=0007/part-0000.parquet")

    (root / "items" / "items.parquet").write_bytes(b"")

    with pytest.raises(ValueError, match="Non-canonical parquet file"):
        pending_source_partitions(
            "classify", "items", "classifications", artifact_root=str(root)
        )


def test_completion_registry_saves_merge_concurrent_updates(tmp_path: Path) -> None:
    """Overlapping writers must not lose each other's registry entries (#88).

    A lost entry silently strands a partition (or fake-completes it with empty
    item_ids), so saves overlay only the entries a run changed onto the freshest
    persisted state instead of overwriting the file with a stale snapshot.
    """
    from cdt.datasets import (
        CompletedPartition,
        load_completion_registry,
        save_completion_registry,
    )

    save_completion_registry(
        "itemize", {"P": CompletedPartition(fingerprint="f1")}, artifact_root=tmp_path
    )
    writer_a = load_completion_registry("itemize", artifact_root=tmp_path)
    writer_b = load_completion_registry("itemize", artifact_root=tmp_path)

    writer_b["P"] = CompletedPartition(fingerprint="f2")
    writer_b["Q"] = CompletedPartition(fingerprint="q1")
    save_completion_registry("itemize", writer_b, artifact_root=tmp_path)

    # A loaded P at f1 but never touched it; its save must not revert B's f2.
    writer_a["R"] = CompletedPartition(fingerprint="r1")
    save_completion_registry("itemize", writer_a, artifact_root=tmp_path)

    final = load_completion_registry("itemize", artifact_root=tmp_path)
    assert set(final) == {"P", "Q", "R"}
    assert final["P"].fingerprint == "f2"
    assert final["Q"].fingerprint == "q1"
    assert final["R"].fingerprint == "r1"


def test_pending_source_partitions_stamps_survive_concurrent_saves(
    tmp_path: Path,
) -> None:
    """Legacy-entry stamping counts as a change and survives the merge (#88)."""
    from cdt.datasets import (
        CompletedPartition,
        load_completion_registry,
        pending_source_partitions,
        save_completion_registry,
    )

    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    source_path = write_partition_table(
        str(tmp_path / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    # A v1-migrated entry: complete but fingerprint-less.
    save_completion_registry(
        "classify", {source_path: CompletedPartition()}, artifact_root=tmp_path
    )

    pending, registry = pending_source_partitions(
        "classify", "items", "classifications", artifact_root=str(tmp_path)
    )
    assert pending == []
    save_completion_registry("classify", registry, artifact_root=tmp_path)

    final = load_completion_registry("classify", artifact_root=tmp_path)
    assert final[source_path].fingerprint is not None


def _seed_classifications(
    tmp_path: Path, item_ids: list[str], *, date: str = "2024-01-02"
) -> None:
    """Write one classifications partition with the given relevant items."""
    from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS

    rows = []
    for item_id in item_ids:
        row: dict[str, object] = {column: None for column in CLASSIFIED_ITEM_COLUMNS}
        row.update(
            {
                "item_id": item_id,
                "accession_number": item_id.split("-")[0],
                "cik": "320193",
                "date": date,
                "item": "8.01",
                "text": f"text for {item_id}",
                "relevance": True,
            }
        )
        rows.append(row)
    write_partition_table(
        str(classifications_root(tmp_path)),
        partition={"date": date, "shard": "0001"},
        table=pd.DataFrame(rows, columns=CLASSIFIED_ITEM_COLUMNS),
    )


def _fake_success_workflow(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the extraction workflow to succeed with one mention per row."""
    calls: list[str] = []

    async def fake_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        calls.append(str(item_row["item_id"]))
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {"item_id": str(item_row["item_id"]), "name": "Term Loan"}
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", fake_workflow)
    return calls


def test_late_arriving_rows_extract_after_partition_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows merged into an already-processed partition are picked up (#62)."""
    _seed_classifications(tmp_path, ["a-8-01"])
    calls = _fake_success_workflow(monkeypatch)

    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)
    assert calls == ["a-8-01"]

    # Ingest-style in-place merge: the partition object grows a new row.
    _seed_classifications(tmp_path, ["a-8-01", "b-8-01"])
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # Only the new row is paid for, and the target holds both rows' mentions.
    assert calls == ["a-8-01", "b-8-01"]
    written = read_dataset(mentions_root(tmp_path))
    assert sorted(written["item_id"]) == ["a-8-01", "b-8-01"]


def test_infrastructure_error_aborts_and_preserves_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider failure stops the run; terminal rows are never re-paid (#49)."""
    from cdt.datasets import load_completion_registry
    from cdt.extractor.core import InfrastructureError

    _seed_classifications(tmp_path, ["a-8-01", "b-8-01"])
    calls: list[str] = []

    async def failing_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        calls.append(str(item_row["item_id"]))
        if str(item_row["item_id"]) == "b-8-01":
            raise InfrastructureError("PaymentRequiredResponseError: 402")
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {"item_id": str(item_row["item_id"]), "name": "Term Loan"}
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", failing_workflow)
    with pytest.raises(InfrastructureError):
        extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    registry = load_completion_registry("extract", artifact_root=tmp_path)
    (entry,) = registry.values()
    assert not entry.complete
    assert entry.item_ids == frozenset({"a-8-01"})
    # The finished row's mentions survived the abort.
    assert sorted(read_dataset(mentions_root(tmp_path))["item_id"]) == ["a-8-01"]

    # Recovery: a healthy run pays only for the row that never got a verdict.
    recovery_calls = _fake_success_workflow(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)
    assert recovery_calls == ["b-8-01"]
    registry = load_completion_registry("extract", artifact_root=tmp_path)
    (entry,) = registry.values()
    assert entry.complete
    assert sorted(read_dataset(mentions_root(tmp_path))["item_id"]) == [
        "a-8-01",
        "b-8-01",
    ]


def test_infrastructure_error_classification() -> None:
    """Status- and name-shaped provider errors classify as infrastructure."""
    from cdt.extractor.core import is_infrastructure_error

    class PaymentRequiredResponseError(Exception):
        pass

    class WithStatus(Exception):
        status_code = 503

    assert is_infrastructure_error(PaymentRequiredResponseError())
    assert is_infrastructure_error(WithStatus())
    assert is_infrastructure_error(ConnectionResetError())
    assert not is_infrastructure_error(ValueError("bad xml"))


def test_grown_document_partition_reitemizes_and_reclassifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Late-arriving documents merged into a partition flow downstream (#62)."""

    def _doc(accession: str) -> dict[str, object]:
        return {
            "accession_number": accession,
            "cik": "320193",
            "company_name": "Example Inc.",
            "url": "https://sec.example/full.txt",
            "text": (
                "\nITEM INFORMATION: Other Events\n<DOCUMENT>\n<TYPE>8-K\n<TEXT>\n"
                "Item 8.01 Other Events.\nThis is the extracted event text.\n"
                "</TEXT>\n</DOCUMENT>\n"
            ),
            "date": "2024-01-02",
            "resource_uri": None,
        }

    def _write_documents(accessions: list[str]) -> None:
        write_partition_table(
            str(tmp_path / "documents"),
            partition={"date": "2024-01-02", "shard": "0001"},
            table=pd.DataFrame([_doc(a) for a in accessions], columns=DOCUMENT_COLUMNS),
        )

    class SizedFakeModel:
        def decision_function(self: SizedFakeModel, texts: list[str]) -> list[float]:
            return [2.0] * len(texts)

    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (SizedFakeModel(), 0.5, {"threshold": 0.5}),
    )

    _write_documents(["000114036126006577"])
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    classify_pending_items(artifact_root=tmp_path, batch_size=5)
    assert len(read_dataset(classifications_root(tmp_path))) == 1

    # Ingest-style merge: the same partition object grows a second filing.
    _write_documents(["000114036126006577", "000114036126009999"])
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

    classified = read_dataset(classifications_root(tmp_path))
    assert sorted(classified["accession_number"].astype(str).unique()) == [
        "000114036126006577",
        "000114036126009999",
    ]


def test_match_pending_mentions_renews_lease_per_shard(tmp_path: Path) -> None:
    """The matcher extends the writer lease before rewriting each shard (#89)."""
    for index, (cik, date_value, shard) in enumerate(
        [("320193", "2024-01-02", "0001"), ("789019", "2024-01-03", "0002")], start=1
    ):
        write_partition_table(
            tmp_path / "mentions",
            partition={"date": date_value, "shard": shard},
            table=pd.DataFrame(
                [
                    build_mention_row(
                        mention_id=f"m-{index}",
                        item_id=f"item-{index}",
                        accession_number=f"000{index}",
                        cik=cik,
                        date=date_value,
                        name="Term Loan",
                        start_date="2024-01-01",
                        amount="$100 million",
                    )
                ]
            ),
        )
    renewals: list[int] = []

    match_pending_mentions(
        artifact_root=tmp_path, batch_size=5, renew=lambda: renewals.append(1)
    )

    assert len(renewals) == 2


def test_read_table_projects_columns_and_tolerates_missing_ones(
    tmp_path: Path,
) -> None:
    """Column projection is pushed down; absent columns reindex instead of raising (#69)."""
    from cdt.storage import write_table

    path = tmp_path / "table.parquet"
    write_table(path, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    projected = read_table(path, ["a"])
    assert list(projected.columns) == ["a"]

    tolerant = read_table(path, ["a", "missing"])
    assert list(tolerant.columns) == ["a", "missing"]
    assert tolerant["missing"].isna().all()


def test_classifier_loads_model_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pickled model is deserialized once, not once per partition (#76)."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    loads = 0

    def counting_load(path: object) -> tuple[FakeModel, float, dict[str, float]]:
        nonlocal loads
        del path
        loads += 1
        return (FakeModel(), 0.5, {"threshold": 0.5})

    monkeypatch.setattr(classifier_core, "load_training_artifacts", counting_load)

    classified = classify_pending_items(artifact_root=tmp_path, batch_size=1)

    assert len(classified) == 2
    assert loads == 1
