"""The publish contract: what the website reads, checked against what CDT writes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet
import pytest
from support import amended_row

from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.extractor import mentions_root
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.matcher import match_pending_mentions
from cdt.matcher.schema import (
    DEBT_INSTRUMENT_COLUMNS,
    DEBT_INSTRUMENT_REDIRECT_COLUMNS,
    MENTION_CLUSTER_EDGE_COLUMNS,
)
from cdt.publish import (
    FINAL_OUTPUT_TABLES,
    PublishGuardError,
    _guard_against_shrinkage,
    contract_schema,
    final_pointer_path,
    payload_problem,
    publish_contract,
    publish_final_tables,
    publish_source_digest,
    published_table_path,
)
from cdt.segmenter.core import ITEM_COLUMNS, items_root
from cdt.storage.columns import declared_column_type
from cdt.storage.objects import read_json_artifact
from cdt.storage.tables import read_table, write_partition_table

CODE_COLUMNS = {
    "items": [*ITEM_COLUMNS, "form_type"],
    "debt-instruments": DEBT_INSTRUMENT_COLUMNS,
    "debt-instrument-mentions": DEBT_INSTRUMENT_MENTION_COLUMNS,
    "mention-cluster-edges": MENTION_CLUSTER_EDGE_COLUMNS,
    "debt-instrument-redirects": DEBT_INSTRUMENT_REDIRECT_COLUMNS,
}


def test_the_contract_names_every_published_table_and_column() -> None:
    """A column added to or dropped from the code must be added to the contract too."""
    tables = publish_contract()["tables"]

    assert set(tables) == set(FINAL_OUTPUT_TABLES) == set(CODE_COLUMNS)
    for table_name, columns in CODE_COLUMNS.items():
        assert [column["name"] for column in tables[table_name]["columns"]] == columns


def test_contract_types_are_the_declared_column_types() -> None:
    """The contract and the writer agree on every column's physical type."""
    for table_name in FINAL_OUTPUT_TABLES:
        for field in contract_schema(table_name):
            assert field.type == declared_column_type(field.name, field.type), (
                table_name,
                field.name,
            )


def _span(tag_id: str, text: str) -> dict[str, object]:
    return {"tag_id": tag_id, "char_start": 0, "char_end": len(text), "text": text}


def _date_payload(normalized_date: str | None) -> str:
    return json.dumps(
        {
            "normalized_date": normalized_date,
            "derived_from": "stated" if normalized_date else None,
            "spans": [_span("tag-d-1", "June 1, 2024")] if normalized_date else [],
        }
    )


def _production_payloads() -> dict[str, str]:
    """Every ``*_json`` column of one mention, in the shapes the extractor writes."""
    commitment = _span("tag-a-1", "$250,000,000")
    return {
        "name_json": json.dumps({"spans": [_span("tag-i-1", "Credit Agreement")]}),
        "start_date_json": _date_payload("2020-02-03"),
        "maturity_date_json": _date_payload("2029-02-03"),
        "commitment_termination_date_json": _date_payload(None),
        "status_json": json.dumps(
            {
                "status": "amended",
                "status_date": json.loads(_date_payload("2024-06-01")),
                "derived_from_kind": "amendment",
            }
        ),
        "interest_rate_json": json.dumps(
            {
                "kind": "fixed",
                "rate_pct": "5.25",
                "derived_from": "stated",
                "spans": [_span("tag-r-1", "5.25%")],
            }
        ),
        "amounts_json": json.dumps(
            [
                {
                    "kind": "commitment",
                    "normalized_amount": "250000000",
                    "currency": "USD",
                    "as_of_date": None,
                    "prior": False,
                    "derived_from": "stated",
                    "spans": [commitment],
                }
            ]
        ),
        "dates_json": json.dumps(
            [
                {
                    "kind": "amendment",
                    "normalized_date": "2024-06-01",
                    "precision": "day",
                    "prior": False,
                    "expected": False,
                    "derived_from": "stated",
                    "spans": [_span("tag-d-1", "June 1, 2024")],
                }
            ]
        ),
        "parties_json": json.dumps(
            [
                {
                    "role": "borrower",
                    "canonical_name": "Example Inc.",
                    "kind": "named",
                    "spans": [_span("tag-o-1", "Example Inc.")],
                },
                {
                    "role": "lender",
                    "canonical_name": "Bank of America, N.A.",
                    "kind": "named",
                    "spans": [_span("tag-o-2", "Bank of America, N.A.")],
                },
            ]
        ),
        "retired_by_json": "[]",
    }


def _golden_root(tmp_path: Path) -> tuple[Path, Path]:
    """Match and publish one amended credit agreement with cited evidence.

    The payloads are shaped as the extractor writes them
    (:func:`_production_payloads`), so the contract is checked against
    production shapes, not a test shorthand.
    """
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    mention = amended_row(**_production_payloads())
    write_partition_table(
        mentions_root(artifact_root),
        partition={"month": "2024-06"},
        table=pd.DataFrame([mention], columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
    )
    item = dict.fromkeys(CLASSIFIED_ITEM_COLUMNS)
    item.update(
        {
            "item_id": mention["item_id"],
            "accession_number": mention["accession_number"],
            "cik": mention["cik"],
            "date": mention["date"],
            "text": "The Company amended its Credit Agreement.",
            # A column the items table does not publish.
            "label": "relevant",
        }
    )
    unreferenced = {**item, "item_id": "item-never-mentioned"}
    write_partition_table(
        items_root(artifact_root),
        partition={"month": "2024-06"},
        table=pd.DataFrame([item, unreferenced]),
    )
    match_pending_mentions(artifact_root=artifact_root)
    publish_final_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )
    return artifact_root, final_root


def test_every_published_table_has_exactly_the_contract_schema(tmp_path: Path) -> None:
    """Columns, order and Arrow types; a stale column on disk does not pass through."""
    _, final_root = _golden_root(tmp_path)

    for table_name in FINAL_OUTPUT_TABLES:
        schema = pyarrow.parquet.read_schema(
            published_table_path(final_root, table_name)
        )
        assert schema.remove_metadata() == contract_schema(table_name), table_name


def test_only_items_some_mention_references_are_published(tmp_path: Path) -> None:
    """An item no mention cites is not published."""
    _, final_root = _golden_root(tmp_path)

    items = read_table(published_table_path(final_root, "items"))

    assert items["item_id"].to_list() == ["item-1"]
    assert items["form_type"].to_list() == ["8-K"]


def test_the_pointer_records_the_contract_version(tmp_path: Path) -> None:
    """A reader can tell which contract a generation was written against."""
    _, final_root = _golden_root(tmp_path)

    pointer = read_json_artifact(final_pointer_path(final_root))

    assert pointer["contract_version"] == publish_contract()["contract_version"]
    assert set(pointer["tables"]) == set(FINAL_OUTPUT_TABLES)


@pytest.mark.parametrize("table_name", ["debt-instrument-mentions", "debt-instruments"])
def test_published_payloads_have_the_contract_shapes(
    tmp_path: Path, table_name: str
) -> None:
    """Parties carry a role and spans; facts carry the documented keys."""
    _, final_root = _golden_root(tmp_path)
    payloads = publish_contract()["payloads"]
    span_keys = set(payloads["spans"]["keys"])
    table = read_table(published_table_path(final_root, table_name))
    assert not table.empty

    checked = 0
    for column, shape in payloads["columns"][table_name].items():
        for text in table[column].dropna():
            assert payload_problem(json.loads(text), shape, span_keys) is None, column
            checked += 1

    assert checked
    parties = json.loads(table["parties_json"].iloc[0])
    assert {party["role"] for party in parties} == {"borrower", "lender"}


def test_the_pointer_resolves_every_table_relative_to_the_final_root(
    tmp_path: Path,
) -> None:
    """A copied or mirrored final root still resolves its generation."""
    _, final_root = _golden_root(tmp_path)
    moved = tmp_path / "moved"
    shutil.copytree(final_root, moved)

    pointer = read_json_artifact(final_pointer_path(moved))

    for table_name, entry in pointer["tables"].items():
        assert not Path(entry["path"]).is_absolute()
        assert len(read_table(moved / entry["path"])) == entry["rows"], table_name


def test_a_payload_that_breaks_its_shape_is_named() -> None:
    """Missing keys, a wrong container and a broken nested object are each reported."""
    shapes = publish_contract()["payloads"]
    span_keys = set(shapes["spans"]["keys"])
    status = shapes["columns"]["debt-instrument-mentions"]["status_json"]
    parties = shapes["columns"]["debt-instrument-mentions"]["parties_json"]

    assert payload_problem({"status": "active"}, status, span_keys) == (
        "missing keys ['status_date']"
    )
    assert (
        payload_problem({"status": "active", "status_date": None}, status, span_keys)
        is None
    )
    assert (
        payload_problem(
            {"status": "active", "status_date": {"spans": []}}, status, span_keys
        )
        == "status_date: missing keys ['derived_from', 'normalized_date']"
    )
    assert payload_problem({"role": "lender"}, parties, span_keys) == "not an array"


def test_publish_warns_about_absent_columns_and_broken_payloads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A column no source has publishes null, and a bad payload publishes, but neither silently (#295)."""
    artifact_root = tmp_path / "artifacts"
    mention = amended_row(**_production_payloads())
    mention["parties_json"] = json.dumps([{"canonical_name": "Example Inc."}])
    columns = [
        column for column in DEBT_INSTRUMENT_MENTION_COLUMNS if column != "status_date"
    ]
    write_partition_table(
        mentions_root(artifact_root),
        partition={"month": "2024-06"},
        table=pd.DataFrame([mention], columns=columns),
    )

    with caplog.at_level("WARNING"):
        publish_final_tables(
            artifact_root=str(artifact_root),
            final_database_root=str(tmp_path / "final"),
        )

    assert "debt-instrument-mentions with contract column(s) no source" in caplog.text
    assert "status_date" in caplog.text
    assert "debt-instrument-mentions.parties_json has payloads" in caplog.text
    assert "missing keys ['kind', 'role', 'spans'] x1" in caplog.text


def test_the_redirects_table_may_shrink_without_refusing_the_publish() -> None:
    """Redirects shrink when retired ids come back; every other table is guarded (#296)."""
    _guard_against_shrinkage(
        {"debt-instrument-redirects": 0, "debt-instruments": 10},
        {"debt-instrument-redirects": 2, "debt-instruments": 10},
        force_publish=False,
    )

    with pytest.raises(PublishGuardError):
        _guard_against_shrinkage(
            {"debt-instruments": 4}, {"debt-instruments": 10}, force_publish=False
        )


def test_an_edit_to_the_contract_moves_the_publish_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate hashes the contract's bytes, so an edit without a version bump republishes."""
    from cdt import publish

    before = publish_source_digest(tmp_path)
    edited = tmp_path / "publish_contract.json"
    edited.write_text(publish.PUBLISH_CONTRACT_PATH.read_text() + "\n")
    monkeypatch.setattr(publish, "PUBLISH_CONTRACT_PATH", edited)
    publish.publish_contract_digest.cache_clear()
    try:
        assert publish_source_digest(tmp_path) != before
    finally:
        publish.publish_contract_digest.cache_clear()
