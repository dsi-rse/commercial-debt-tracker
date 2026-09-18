"""Tests for inferred amendment lineage (#170)."""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from cdt.matcher import core, lineage_inference
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
        "start_date": None,
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
    result = infer_amendment_parents(rows)
    assert result["i3"] == ("i2", "ordinal_chain")
    assert result["i2"][0] == "i1"


def test_the_matcher_never_reads_filing_text(monkeypatch: object) -> None:
    """The stage boundary (#184): no lineage rule may derive a fact from item text.

    A text-reading rule attributes a document-level observation to every
    instrument the filing names, and publishes a relation with no evidence span
    (#154). Two tranches of one new agreement named in one filing must therefore
    produce nothing, however suggestive the prose.
    """
    rows = [
        instrument("i1", "term loan A facility", first_seen_filing_date="2026-05-28"),
        instrument("i2", "revolving facility", first_seen_filing_date="2026-05-28"),
    ]
    result = infer_amendment_parents(rows)
    assert result == {}
    assert not hasattr(lineage_inference, "DATED_REFERENCE")
    assert not hasattr(core, "read_item_texts")
    # Instrument rows are the whole input. `member_groups` and `mention_index`
    # were required and immediately `del`-ed once #203 moved `prior_fact` to the
    # extractor, kept "so a future mention-reading rule keeps one call shape" —
    # which is the pattern the next two assertions have forbidden here since the
    # #177 review. A rule that needs mentions takes them when it exists (#211).
    signature = inspect.signature(infer_amendment_parents)
    for parameter in ("item_texts", "member_groups", "mention_index"):
        assert parameter not in signature.parameters, parameter
    for name in ("match_tables", "match_pending_mentions"):
        parameters = inspect.signature(getattr(core, name)).parameters
        assert "item_texts" not in parameters, name
        assert "infer_lineage" not in parameters, name


def test_a_replaced_state_of_one_rank_steps_aside_for_the_state_that_replaced_it() -> (
    None
):
    """Two states of the Second A&R, linked to each other, are not a tie.

    The extractor mints an amended instrument's prior state as its own row
    (#203), so a restatement amended once publishes two rank-2 rows with the
    later pointing at the earlier. The Third A&R follows the *latest* state of
    the Second, not neither of them.
    """
    rows = [
        instrument(
            "second-original",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2017-11-14",
        ),
        instrument(
            "second-amended",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2017-11-14",
            amendment_of_debt_instrument_id="second-original",
        ),
        instrument(
            "third",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-06-28",
        ),
    ]
    result = infer_amendment_parents(rows)

    assert result["third"] == ("second-amended", "ordinal_chain")


def test_a_same_rank_tie_is_refused_rather_than_decided_by_id_order() -> None:
    """Two equally-ranked predecessors are ambiguity, not a sort-order question."""
    rows = [
        instrument(
            "iA",
            "Amended and Restated Credit Agreement",
            first_seen_filing_date="2021-01-01",
        ),
        instrument(
            "iB",
            "Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-01-01",
        ),
        instrument(
            "iC",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2023-01-01",
        ),
    ]
    result = infer_amendment_parents(rows)
    assert result == {}


def test_a_parent_dated_after_its_child_is_rejected() -> None:
    """Own start dates outrank filing dates, which cannot separate one filing."""
    rows = [
        instrument(
            "i1",
            "Credit Agreement",
            start_date="2026-01-01",
            first_seen_filing_date="2026-05-28",
        ),
        instrument(
            "i2",
            "Second Amended and Restated Credit Agreement",
            start_date="2022-01-01",
            first_seen_filing_date="2026-05-28",
        ),
    ]
    result = infer_amendment_parents(rows)
    assert result == {}


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
    result = infer_amendment_parents(rows)
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
    result = infer_amendment_parents(rows)
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
    result = infer_amendment_parents(rows)
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
    result = infer_amendment_parents(rows)
    assert result == {}


def test_match_tables_itself_never_infers_a_pointer() -> None:
    """Inference is the post-pass's job; the per-shard matcher publishes none.

    `match_tables` sees only the clusters its batch touched, so a rule there
    could never see both states of one facility. The pass runs after every
    match (`run_match_and_finalize`, `cdt match`), not behind a flag.
    """
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


def borrower(name: str) -> str:
    """Return a `parties_json` naming one borrower."""
    return json.dumps([{"role": "borrower", "canonical_name": name, "spans": []}])


def test_a_different_borrower_refuses_the_ordinal_link() -> None:
    """Two issuers' agreements can share a filer CIK, a stem and an ordinal.

    EQT's 2024-07-22 8-K names both its own `Third Amended and Restated Credit
    Agreement` and EQM Midstream Partners' — the latter arriving through the
    Equitrans merger and terminated the same week. Both carry EQT's filer CIK,
    so the CIK check cannot separate them, and both were offered as children of
    EQT's `Second Amended and Restated Credit Agreement`: an acquired
    subsidiary's dead facility welded into the parent's chain, and the real
    predecessor left with two children and so no published `superseded_by`.
    """
    rows = [
        instrument(
            "eqt-second",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2017-11-14",
            parties_json=borrower("EQT Corporation"),
        ),
        instrument(
            "eqt-third",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-06-28",
            parties_json=borrower("EQT Corporation"),
        ),
        instrument(
            "eqm-third",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2024-07-22",
            parties_json=borrower("EQM Midstream Partners, LP"),
        ),
    ]
    result = infer_amendment_parents(rows)

    assert result["eqt-third"] == ("eqt-second", "ordinal_chain")
    assert "eqm-third" not in result


def test_a_legal_form_suffix_is_not_a_different_borrower() -> None:
    """`EQT Corporation` and `EQT Company` are one party, so the chain links.

    Two suffixes that differ from each other, not a bare stem against a
    suffixed one: `EQT` versus `EQT Corporation` also passed by prefix, so the
    suffix list could be deleted outright with this test still green (#205).
    Only `BORROWER_SUFFIXES` can make these two names equal.
    """
    rows = [
        instrument(
            "i1",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-01-01",
            parties_json=borrower("EQT Corporation"),
        ),
        instrument(
            "i2",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2024-01-01",
            parties_json=borrower("EQT Company"),
        ),
    ]
    result = infer_amendment_parents(rows)

    assert result["i2"] == ("i1", "ordinal_chain")


def ordinal_pair(
    parent_parties: str | None, child_parties: str | None
) -> list[dict[str, object]]:
    """Return a Second/Third A&R pair carrying the given `parties_json` values."""
    parent = instrument(
        "i1",
        "Second Amended and Restated Credit Agreement",
        first_seen_filing_date="2022-01-01",
    )
    child = instrument(
        "i2",
        "Third Amended and Restated Credit Agreement",
        first_seen_filing_date="2024-01-01",
    )
    if parent_parties is not None:
        parent["parties_json"] = parent_parties
    if child_parties is not None:
        child["parties_json"] = child_parties
    return [parent, child]


@pytest.mark.parametrize(
    "placeholder",
    ["Issuer", "the Borrowers", "Buyer Parent", "other Borrowers party thereto"],
)
def test_a_placeholder_borrower_is_silence_not_a_different_company(
    placeholder: str,
) -> None:
    """A borrower recorded only by its role or defined term constrains nothing.

    The extractor keeps the longest span, so a filing that never names the
    company records `Issuer`; read as a name, `HSBC Holdings plc` against
    `Issuer` refused the link on 17 rows of one corpus (#205).
    """
    rows = ordinal_pair(borrower("HSBC Holdings plc"), borrower(placeholder))
    result = infer_amendment_parents(rows)

    assert result["i2"] == ("i1", "ordinal_chain")


@pytest.mark.parametrize(
    ("parent", "child"),
    [
        ("EQT Corporation", "EQT Midstream Partners, LP"),
        ("Ford Motor Company", "Ford Motor Credit Company LLC"),
    ],
)
def test_a_subsidiary_named_after_its_parent_is_a_different_borrower(
    parent: str, child: str
) -> None:
    """A finance subsidiary is not its parent, however the name begins (#205)."""
    rows = ordinal_pair(borrower(parent), borrower(child))
    result = infer_amendment_parents(rows)

    assert "i2" not in result


def test_one_shared_borrower_among_several_is_agreement() -> None:
    """A cluster's borrowers are a union; one match among many is enough."""
    both = json.dumps(
        [
            {"role": "borrower", "canonical_name": "MPLX LP", "spans": []},
            {"role": "borrower", "canonical_name": "Andeavor Logistics", "spans": []},
        ]
    )
    rows = ordinal_pair(both, borrower("Andeavor Logistics LP"))
    result = infer_amendment_parents(rows)

    assert result["i2"] == ("i1", "ordinal_chain")


def test_a_borrower_that_is_all_noise_does_not_switch_the_guard_off() -> None:
    """`The` reduces to an empty key, which must not match every other key."""
    noise_and_eqm = json.dumps(
        [
            {"role": "borrower", "canonical_name": "The", "spans": []},
            {
                "role": "borrower",
                "canonical_name": "EQM Midstream Partners",
                "spans": [],
            },
        ]
    )
    rows = ordinal_pair(borrower("EQT Corporation"), noise_and_eqm)
    result = infer_amendment_parents(rows)

    assert "i2" not in result


def test_a_parties_payload_that_is_not_a_list_reads_as_no_borrower() -> None:
    """Valid JSON that is not a list is silence, not an aborted pass."""
    rows = ordinal_pair(borrower("EQT Corporation"), "null")
    result = infer_amendment_parents(rows)

    assert result["i2"] == ("i1", "ordinal_chain")


def test_a_missing_borrower_does_not_refuse_the_link() -> None:
    """Silence is not disagreement: most mentions never name a borrower."""
    rows = [
        instrument(
            "i1",
            "Second Amended and Restated Credit Agreement",
            first_seen_filing_date="2022-01-01",
            parties_json=borrower("EQT Corporation"),
        ),
        instrument(
            "i2",
            "Third Amended and Restated Credit Agreement",
            first_seen_filing_date="2024-01-01",
        ),
    ]
    result = infer_amendment_parents(rows)

    assert result["i2"] == ("i1", "ordinal_chain")
