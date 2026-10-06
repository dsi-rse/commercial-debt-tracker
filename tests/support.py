"""Fakes, fixtures and row builders shared by several test modules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cdt.classifier import classifications_root
from cdt.extractor.normalize.amounts import normalized_amount_from_text
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.state import ExtractionRowState
from cdt.ingest.core import DOCUMENT_COLUMNS
from cdt.storage.tables import write_partition_table


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
    parties_json: str = "[]",
    lender_disclosure: str = "complete",
    company_name: str | None = "Example Inc.",
    **overrides: object,
) -> dict[str, object]:
    """Return one canonical mention row for matcher tests.

    Built from `DEBT_INSTRUMENT_MENTION_COLUMNS` so every published column is
    present. Listing only the columns a test happened to need let a renamed or
    dropped column pass unnoticed: `prepare_mention` reads them all with
    `row.get`, so an absent one silently became None.
    """
    row: dict[str, object] = dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS)
    row.update(
        {
            "debt_instrument_mention_id": mention_id,
            "item_id": item_id,
            "accession_number": accession_number,
            "cik": cik,
            "company_name": company_name,
            "date": date,
            "raw_id": "i-1",
            "name": name,
            "start_date": start_date,
            # Production writes the parsed, canonical figure here, never the
            # display text a filing used, so fixtures normalize the same way:
            # `principal_amount` publishes as an exact decimal (#185).
            "principal_amount": normalized_amount_from_text(amount) or amount,
            "retired_by_json": "[]",
            "parties_json": parties_json,
            "lender_disclosure": lender_disclosure,
            "name_json": "{}",
            "start_date_json": "{}",
            "maturity_date_json": "{}",
            "amounts_json": "[]",
            "dates_json": "[]",
        }
    )
    unknown = set(overrides) - set(DEBT_INSTRUMENT_MENTION_COLUMNS)
    assert not unknown, f"not published mention columns: {sorted(unknown)}"
    row.update(overrides)
    return row


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


def _fact(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {"spans": [], "derived_from": "stated"}
    base.update(fields)
    return base


def amended_row(
    mention_id: str = "m-amend",
    *,
    item_id: str = "item-1",
    amounts: list[dict[str, object]] | None = None,
    dates: list[dict[str, object]] | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Return one amended instrument the way the IE stage publishes it.

    Default: `reduced commitments from $300,000,000 to $250,000,000` under a
    `Credit Agreement dated as of 2020-02-03`, amended 2024-06-01, maturing
    2029-02-03, 5.25% fixed, borrower and lender named.
    """
    amounts = (
        amounts
        if amounts is not None
        else [
            _fact(
                kind="commitment",
                normalized_amount="300000000",
                currency="USD",
                prior=True,
            ),
            _fact(
                kind="commitment",
                normalized_amount="250000000",
                currency="USD",
                prior=False,
            ),
            _fact(
                kind="outstanding_balance", normalized_amount="100000000", prior=False
            ),
        ]
    )
    dates = (
        dates
        if dates is not None
        else [
            _fact(
                kind="agreement",
                normalized_date="2020-02-03",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2029-02-03",
                prior=False,
                expected=False,
            ),
        ]
    )
    principal = next(
        (
            a
            for a in amounts
            if not a.get("prior") and a.get("kind") in ("commitment", "principal")
        ),
        {},
    )
    row = build_mention_row(
        mention_id=mention_id,
        item_id=item_id,
        accession_number="0002",
        cik="0000320193",
        date="2024-06-01",
        name="Credit Agreement",
        start_date="2020-02-03",
        amount=str(principal.get("normalized_amount") or ""),
        parties_json=json.dumps(
            [
                {"role": "borrower", "canonical_name": "Example Inc.", "spans": []},
                {
                    "role": "lender",
                    "canonical_name": "Bank of America, N.A.",
                    "spans": [],
                },
            ]
        ),
        raw_id="i-1",
        instrument_type="revolving_credit",
        maturity_date="2029-02-03",
        interest_rate_kind="fixed",
        interest_rate_pct="5.25",
        interest_rate_json=json.dumps(_fact(kind="fixed", rate_pct="5.25")),
        amounts_json=json.dumps(amounts, sort_keys=True),
        dates_json=json.dumps(dates, sort_keys=True),
        name_json=json.dumps({"spans": [{"text": "Credit Agreement"}]}),
    )
    if not principal:
        row["principal_amount"] = None
    row.update(overrides)
    return row


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


def seed_document_partitions_across_months(
    tmp_path: Path, partitions: list[tuple[str, str]]
) -> list[str]:
    """Write one document partition per ``(date, shard)``, dates free-form.

    seed_document_partitions' fixed pair is same-month; the registry shards by
    year-month, so anything about sharding needs dates that straddle months.
    """
    paths: list[str] = []
    for index, (filing_date, shard) in enumerate(partitions):
        table = pd.DataFrame(
            [
                {
                    "accession_number": f"00011403612600{index:04d}",
                    "cik": "320193",
                    "company_name": "Example Inc.",
                    "url": "https://sec.example/full.txt",
                    "text": (
                        "ITEM INFORMATION: Other Events\n"
                        "<DOCUMENT>\n<TYPE>8-K\n<TEXT>\n"
                        "Item 8.01 Other Events.\n"
                        "This is the extracted event text.\n"
                        "</TEXT>\n</DOCUMENT>\n"
                    ),
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

    monkeypatch.setattr("cdt.extractor.live.run_extraction_workflow", fake_workflow)
    return calls


def _dates_tag_details() -> dict[str, dict[str, object]]:
    return {
        "tag-1": {
            "text": "5.000% Senior Notes due 2031",
            "type": "debt_instrument",
            "char_start": 0,
            "char_end": 28,
        },
        "tag-2": {
            "text": "March 5, 2026",
            "type": "date",
            "char_start": 40,
            "char_end": 53,
        },
        "tag-3": {
            "text": "March 12, 2026",
            "type": "date",
            "char_start": 60,
            "char_end": 74,
        },
        "tag-4": {
            "text": "June 28, 2026",
            "type": "date",
            "char_start": 80,
            "char_end": 93,
        },
        "tag-5": {
            "text": "June 23, 2031",
            "type": "date",
            "char_start": 100,
            "char_end": 113,
        },
        "tag-6": {
            "text": "in March 2056",
            "type": "date",
            "char_start": 120,
            "char_end": 133,
        },
    }


class FakeModel:
    """Classifier stub returning a fixed relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a single strong-positive score."""
        del texts
        return [2.0]
