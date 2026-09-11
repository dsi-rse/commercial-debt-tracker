"""Tests for inferred amendment lineage (#170)."""

from __future__ import annotations

import json

import pandas as pd

from cdt.matcher.core import prepare_mention
from cdt.matcher.lineage_inference import infer_amendment_parents


def mention(mention_id: str, **overrides: object) -> dict[str, object]:
    """Return a minimal mention row the matcher can prepare."""
    row: dict[str, object] = {
        "debt_instrument_mention_id": mention_id,
        "item_id": f"item-{mention_id}",
        "raw_id": "i-1",
        "accession_number": "acc",
        "cik": "0000000001",
        "company_name": "Test Co",
        "date": "2026-01-01",
        "name": "Credit Agreement",
        "start_date": None,
        "maturity_date": None,
        "principal_amount": None,
        "amounts_json": "[]",
        "parties_json": "[]",
        "lenders_known_incomplete": False,
    }
    row.update(overrides)
    return row


def instrument(instrument_id: str, name: str, **overrides: object) -> dict[str, object]:
    """Return a minimal debt-instrument row."""
    row: dict[str, object] = {
        "debt_instrument_id": instrument_id,
        "cik": "0000000001",
        "company_name": "Test Co",
        "name": name,
        "principal_amount": None,
        "maturity_date": None,
        "amendment_of_debt_instrument_id": None,
        "first_seen_filing_date": "2026-01-01",
    }
    row.update(overrides)
    return row


def index(rows: list[dict[str, object]]) -> dict[str, object]:
    """Prepare mentions the way the matcher does."""
    return {str(r["debt_instrument_mention_id"]): prepare_mention(r) for r in rows}


def test_ordinal_chain_links_each_state_to_its_predecessor() -> None:
    """`Third Amended and Restated X` follows `Second`, which follows the bare name."""
    rows = [
        instrument("i1", "Credit Agreement", first_seen_filing_date="2020-01-01"),
        instrument(
            "i2",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-01-01",
        ),
        instrument(
            "i3",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2024-01-01",
        ),
    ]
    result = infer_amendment_parents(rows, member_groups={}, mention_index={})
    assert result["i3"] == ("i2", "ordinal_chain")
    assert result["i2"][0] == "i1"


def test_prior_marked_amount_links_to_the_instrument_stating_it() -> None:
    """A `prior` commitment equal to another cluster's principal is a predecessor."""
    mentions = [
        mention(
            "m2",
            amounts_json=json.dumps(
                [
                    {
                        "kind": "commitment",
                        "normalized_amount": "2000000000",
                        "prior": True,
                    },
                    {"kind": "commitment", "normalized_amount": "3000000000"},
                ]
            ),
        )
    ]
    rows = [
        instrument(
            "i1",
            "2022 Credit Agreement",
            principal_amount="2000000000",
            first_seen_filing_date="2022-07-07",
        ),
        instrument(
            "i2",
            "2026 Credit Agreement",
            principal_amount="3000000000",
            first_seen_filing_date="2026-04-13",
        ),
    ]
    result = infer_amendment_parents(
        rows, member_groups={"i2": ["m2"]}, mention_index=index(mentions)
    )
    assert result == {"i2": ("i1", "prior_fact")}


def test_dated_reference_resolves_a_named_predecessor() -> None:
    """A replacement clause naming a dated-as-of predecessor links to that cluster."""
    mentions = [
        mention("m1", start_date="2022-07-07", name="2022 Credit Agreement"),
        mention("m2", start_date="2026-04-13", name="New Credit Agreement"),
    ]
    rows = [
        instrument("i1", "2022 Credit Agreement", first_seen_filing_date="2022-07-07"),
        instrument("i2", "New Credit Agreement", first_seen_filing_date="2026-04-13"),
    ]
    texts = {
        "item-m2": (
            "The New Credit Agreement replaced the Company's previously existing "
            "$2.0 billion credit agreement, dated as of July 7, 2022."
        )
    }
    result = infer_amendment_parents(
        rows,
        member_groups={"i1": ["m1"], "i2": ["m2"]},
        mention_index=index(mentions),
        item_texts=texts,
    )
    assert result == {"i2": ("i1", "dated_reference")}


def test_an_ambiguous_predecessor_is_left_alone() -> None:
    """Two equally-qualified parents produce no link: a wrong pointer is worse."""
    mentions = [
        mention(
            "m3",
            amounts_json=json.dumps(
                [{"kind": "commitment", "normalized_amount": "500", "prior": True}]
            ),
        )
    ]
    rows = [
        instrument("i1", "Facility A", principal_amount="500"),
        instrument("i2", "Facility B", principal_amount="500"),
        instrument("i3", "Facility C", principal_amount="900"),
    ]
    result = infer_amendment_parents(
        rows, member_groups={"i3": ["m3"]}, mention_index=index(mentions)
    )
    assert "i3" not in result


def test_an_extracted_pointer_is_never_overwritten() -> None:
    """Inference only fills nulls; the relation stage's answer wins."""
    rows = [
        instrument("i1", "Credit Agreement", first_seen_filing_date="2020-01-01"),
        instrument(
            "i2",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-01-01",
            amendment_of_debt_instrument_id="i9",
        ),
    ]
    result = infer_amendment_parents(rows, member_groups={}, mention_index={})
    assert "i2" not in result


def test_a_predecessor_first_seen_later_is_rejected() -> None:
    """An agreement cannot be amended by something that predates its first filing."""
    rows = [
        instrument("i1", "Credit Agreement", first_seen_filing_date="2026-01-01"),
        instrument(
            "i2",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2020-01-01",
        ),
    ]
    result = infer_amendment_parents(rows, member_groups={}, mention_index={})
    assert result == {}


def test_a_link_that_would_close_a_cycle_is_dropped() -> None:
    """lineage_family_id must stay a DAG, so a cycle-closing link is refused."""
    rows = [
        instrument(
            "i1",
            "Second Amended and Restated Credit Agreement",
            amendment_of_debt_instrument_id="i2",
            first_seen_filing_date="2022-01-01",
        ),
        instrument(
            "i2",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-01-01",
        ),
    ]
    result = infer_amendment_parents(rows, member_groups={}, mention_index={})
    assert "i2" not in result or result["i2"][0] != "i1"


def test_instruments_of_different_issuers_are_never_linked() -> None:
    """Lineage is scoped to one CIK."""
    rows = [
        instrument("i1", "Credit Agreement", first_seen_filing_date="2020-01-01"),
        instrument(
            "i2",
            "Second Amended and Restated Credit Agreement",
            cik="0000000002",
            first_seen_filing_date="2022-01-01",
        ),
    ]
    result = infer_amendment_parents(rows, member_groups={}, mention_index={})
    assert result == {}


def test_match_tables_is_unchanged_without_the_flag() -> None:
    """The published contract only moves when a caller opts in."""
    from cdt.matcher.core import match_tables

    mentions = pd.DataFrame(
        [
            mention("m1", name="Credit Agreement", start_date="2020-01-01"),
            mention(
                "m2",
                name="Second Amended and Restated Credit Agreement",
                item_id="item-m2",
                start_date="2022-01-01",
            ),
        ]
    )
    tables = match_tables(mentions)
    instruments = tables["debt_instrument"]
    assert instruments["amendment_of_debt_instrument_id"].isna().all()
    assert instruments["amendment_inferred_by"].isna().all()
