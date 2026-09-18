"""Matcher scoring tests."""

from __future__ import annotations

import json

import pandas as pd

from cdt.extractor.core import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.matcher.core import (
    NAME_CLASS_GATE,
    PreparedMention,
    borrowed_lender_signature,
    build_debt_instrument_rows,
    build_empty_profile,
    derive_parent_links,
    end_dates_are_compatible,
    lender_signature,
    match_tables,
    mention_sort_key,
    name_rate_tokens,
    name_rates_are_compatible,
    normalize_name_fingerprint,
    prepare_mention,
    score_candidates_for_mention,
)


def mention_row(**overrides: object) -> dict[str, object]:
    """Return one mention row with sensible defaults.

    Seeded from `DEBT_INSTRUMENT_MENTION_COLUMNS` so every published column is
    present: `prepare_mention` reads them all with `row.get`, so a column this
    fixture forgot silently arrived as None and a rename went unnoticed.
    """
    row: dict[str, object] = dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS)
    row.update(
        {
            "debt_instrument_mention_id": "mention-1",
            "item_id": "item-1",
            "raw_id": "raw-1",
            "accession_number": "0000000000-24-000001",
            "cik": "0000320193",
            "company_name": "Example Co",
            "date": "2024-06-01",
            "name": "5.25% senior notes due 2028",
            "start_date": "2024-06-01",
            "maturity_date": "2028-06-01",
            "principal_amount": "500000000",
            "retired_by_json": "[]",
            "parties_json": "[]",
            "lender_disclosure": "complete",
            "maturity_date_json": "{}",
            "amounts_json": "[]",
            "dates_json": "[]",
        }
    )
    unknown = set(overrides) - set(DEBT_INSTRUMENT_MENTION_COLUMNS)
    assert not unknown, f"not published mention columns: {sorted(unknown)}"
    row.update(overrides)
    return row


def profile_from(mention: PreparedMention) -> dict[str, object]:
    """Return a one-cluster profile map seeded from one mention."""
    profile = build_empty_profile(mention.debt_instrument_mention_id, mention)
    profile.add_member(mention)
    return {profile.debt_instrument_id: profile}


def score(
    mention: PreparedMention,
    profiles: dict[str, object],
    name_class_size: int = 1,
) -> list[object]:
    """Score one mention with default thresholds."""
    return score_candidates_for_mention(
        mention,
        profiles,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
        name_class_size=name_class_size,
    )


LENDERS = json.dumps(
    [
        {
            "role": "lender",
            "canonical_name": "Bank of America, N.A.",
            "spans": [{"text": "Bank of America, N.A."}],
        }
    ]
)
BORROWER_ONLY = json.dumps(
    [{"role": "borrower", "canonical_name": "Example Co", "spans": []}]
)


def predecessor_pair() -> tuple[PreparedMention, PreparedMention, PreparedMention]:
    """Return (R, M, P): the original filing's mention, the amendment, its mint.

    R: filing A, `Credit Agreement`, $300M dated 2020-02-03, lenders named.
    M: filing B, `Amendment No. 2 to Credit Agreement`, $250M, same agreement date.
    P: minted from M — the $300M prior state, same item as M, borrower only,
    no name of its own here so the lender path can be exercised alone.
    """
    original = prepare_mention(
        mention_row(
            debt_instrument_mention_id="m-original",
            item_id="item-a",
            accession_number="0001",
            date="2020-02-05",
            name="Credit Agreement",
            start_date="2020-02-03",
            maturity_date=None,
            principal_amount="300000000",
            parties_json=LENDERS,
        )
    )
    amendment = prepare_mention(
        mention_row(
            debt_instrument_mention_id="m-amendment",
            item_id="item-b",
            accession_number="0002",
            date="2024-06-01",
            name="Amendment No. 2 to Credit Agreement",
            start_date="2020-02-03",
            maturity_date=None,
            principal_amount="250000000",
            parties_json=LENDERS,
        )
    )
    minted = prepare_mention(
        mention_row(
            debt_instrument_mention_id="m-prior",
            item_id="item-b",
            accession_number="0002",
            date="2024-06-01",
            name=None,
            start_date="2020-02-03",
            maturity_date=None,
            principal_amount="300000000",
            parties_json=BORROWER_ONLY,
            lender_disclosure="none_named",
            synthesized_by="prior_state",
            synthesized_from_mention_id="m-amendment",
        )
    )
    return original, amendment, minted


def test_a_borrower_only_mint_reaches_membership_only_with_borrowed_lenders() -> None:
    """Keys alone score 0.75 — a `related` edge, never a member (#203).

    A synthesized prior state names no lender, so its own signature is empty
    and lender support is zero. Borrowing its successor's signature, scoring
    only, is what lets it join the original filing's cluster.
    """
    original, amendment, minted = predecessor_pair()
    profiles = profile_from(original)

    on_its_own = score(minted, profiles)
    assert [candidate.match_score for candidate in on_its_own] == [0.75]

    borrowed = borrowed_lender_signature(
        minted, {"m-amendment": amendment, "m-prior": minted}
    )
    assert borrowed == lender_signature(LENDERS)
    with_lenders = score_candidates_for_mention(
        minted,
        profiles,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
        lender_signature=borrowed,
    )
    assert [candidate.match_score for candidate in with_lenders] == [1.0]
    assert with_lenders[0].support_family == "lenders"

    # Nothing is borrowed for a model-emitted mention, or for a mint that
    # names lenders itself, or when the successor is not in the index.
    assert borrowed_lender_signature(original, {"m-amendment": amendment}) is None
    assert borrowed_lender_signature(minted, {}) is None


def test_a_mint_never_joins_its_own_successors_cluster() -> None:
    """Same item, so two instruments by construction (#161): P scores nothing."""
    _, amendment, minted = predecessor_pair()

    assert score(minted, profile_from(amendment)) == []


def test_a_synthesized_members_name_does_not_widen_the_cluster_profile() -> None:
    """P scores in on its name but must not add that name to the profile."""
    original, _, _ = predecessor_pair()
    named_mint = prepare_mention(
        mention_row(
            debt_instrument_mention_id="m-prior",
            item_id="item-b",
            accession_number="0002",
            date="2024-06-01",
            name="Fifth Amended and Restated Credit Agreement",
            start_date="2020-02-03",
            maturity_date=None,
            principal_amount="300000000",
            parties_json=BORROWER_ONLY,
            synthesized_by="prior_state",
            synthesized_from_mention_id="m-amendment",
        )
    )
    profiles = profile_from(original)
    profile = profiles["m-original"]
    profile.add_member(named_mint)

    assert normalize_name_fingerprint("Credit Agreement") in (
        profile.normalized_name_fingerprints
    )
    assert named_mint.normalized_name_fingerprint not in (
        profile.normalized_name_fingerprints
    )
    assert "m-prior" in profile.member_ids


def test_synthesized_mentions_do_not_widen_the_name_class() -> None:
    """A mint carries its successor's name verbatim; it is not another instrument.

    Counting mints pushed one issuer's `Second Amended and Restated Credit
    Agreement` class past `NAME_CLASS_GATE`, and a later mention that had
    always joined that cluster through the name path split off on its own
    (measured on the EQT chain, #203).
    """
    from cdt.matcher.core import name_class_sizes

    original, amendment, _ = predecessor_pair()
    mints = [
        prepare_mention(
            mention_row(
                debt_instrument_mention_id=f"m-mint-{index}",
                item_id=f"item-mint-{index}",
                name="Credit Agreement",
                synthesized_by="prior_state",
                synthesized_from_mention_id="m-amendment",
            )
        )
        for index in range(3)
    ]
    index = {m.debt_instrument_mention_id: m for m in (original, amendment, *mints)}

    sizes = name_class_sizes(index)
    # `Credit Agreement` and `Amendment No. 2 to Credit Agreement` are
    # compatible, so the class is those two — and not the three mints.
    assert sizes["m-original"] == 2
    assert sizes["m-amendment"] == 2

    published_mint = pd.DataFrame(
        [{"cik": "0000320193", "name": "Credit Agreement", "synthesized_only": True}]
    )
    assert name_class_sizes(index, published_mint)["m-original"] == 2


def test_a_prior_state_is_placed_before_the_object_it_was_minted_from() -> None:
    """Two amendments of one restatement chain through their minted prior states.

    EQT's Second Amended and Restated Credit Agreement: increased $1.5B -> $2.5B
    in November 2017, extended in April 2021 with no amount stated. Each
    amendment mints its prior state. The 2021 prior state *is* the November
    2017 state, so it must join that cluster — which only happens if it is
    scored before its own successor, or the same-item guard refuses it. Placed
    first, the history publishes as three states: 1.5B <- 2.5B <- extended.
    """
    nov_2017 = mention_row(
        debt_instrument_mention_id="m-nov-2017",
        item_id="item-nov-2017",
        accession_number="0001",
        date="2017-11-14",
        name="Company's Second Amended and Restated Credit Agreement",
        start_date="2017-07-31",
        maturity_date=None,
        principal_amount="2500000000",
        amendment_of="m-nov-2017-prior",
    )
    nov_2017_prior = mention_row(
        debt_instrument_mention_id="m-nov-2017-prior",
        item_id="item-nov-2017",
        accession_number="0001",
        date="2017-11-14",
        name="Company's Second Amended and Restated Credit Agreement",
        start_date="2017-07-31",
        maturity_date=None,
        principal_amount="1500000000",
        synthesized_by="prior_state",
        synthesized_from_mention_id="m-nov-2017",
    )
    apr_2021 = mention_row(
        debt_instrument_mention_id="m-apr-2021",
        item_id="item-apr-2021",
        accession_number="0002",
        date="2021-04-26",
        name="Second Amended and Restated Credit Agreement",
        start_date="2017-07-31",
        maturity_date=None,
        principal_amount=None,
        amendment_of="m-apr-2021-prior",
    )
    apr_2021_prior = mention_row(
        debt_instrument_mention_id="m-apr-2021-prior",
        item_id="item-apr-2021",
        accession_number="0002",
        date="2021-04-26",
        name="Second Amended and Restated Credit Agreement",
        start_date="2017-07-31",
        maturity_date=None,
        principal_amount=None,
        synthesized_by="prior_state",
        synthesized_from_mention_id="m-apr-2021",
    )
    assert mention_sort_key(prepare_mention(apr_2021_prior)) < mention_sort_key(
        prepare_mention(apr_2021)
    )

    tables = match_tables(
        pd.DataFrame([nov_2017, nov_2017_prior, apr_2021, apr_2021_prior])
    )
    members = (
        tables["debt_instrument_mentions"]
        .query("edge_type == 'member'")
        .set_index("debt_instrument_mention_id")["debt_instrument_id"]
    )
    # The 2021 prior state joins the November-2017 state's cluster.
    assert members["m-apr-2021-prior"] == members["m-nov-2017"]
    assert members["m-nov-2017-prior"] != members["m-nov-2017"]
    assert members["m-apr-2021"] not in {
        members["m-nov-2017"],
        members["m-nov-2017-prior"],
    }

    rows = tables["debt_instrument"].set_index("debt_instrument_id")
    assert (
        rows.loc[members["m-nov-2017"], "amendment_of_debt_instrument_id"]
        == (members["m-nov-2017-prior"])
    )
    assert (
        rows.loc[members["m-apr-2021"], "amendment_of_debt_instrument_id"]
        == (members["m-nov-2017"])
    )
    assert rows["is_lineage_head"].sum() == 1
    assert rows.loc[members["m-apr-2021"], "is_lineage_head"]
    assert not bool(rows.loc[members["m-nov-2017"], "synthesized_only"])
    assert bool(rows.loc[members["m-nov-2017-prior"], "synthesized_only"])


def test_a_synthesized_member_never_supplies_canonical_fields() -> None:
    """The newest member is the mint, but the instrument keeps its own name.

    By recency the mint — stamped with the amendment's filing date — would win
    every canonical field and rename the predecessor cluster after the
    amendment that replaced it, which also ties `ordinal_chain`'s ranks. A
    model-emitted member decides whenever there is one; the mint decides only
    when it is all the cluster has, and the row then says so (#203).
    """
    original, _, _ = predecessor_pair()
    named_mint = prepare_mention(
        mention_row(
            debt_instrument_mention_id="m-prior",
            item_id="item-b",
            accession_number="0002",
            date="2024-06-01",
            company_name="Example Co (as amended)",
            name="Fifth Amended and Restated Credit Agreement",
            start_date="2020-02-03",
            maturity_date=None,
            principal_amount="300000000",
            parties_json=BORROWER_ONLY,
            synthesized_by="prior_state",
            synthesized_from_mention_id="m-amendment",
        )
    )
    index = {"m-original": original, "m-prior": named_mint}

    merged = build_debt_instrument_rows(
        {"inst": ["m-original", "m-prior"]},
        index,
        {},
        existing_instruments=pd.DataFrame(),
        company_names={},
    )[0]
    assert merged["name"] == "Credit Agreement"
    assert merged["name_source_mention_id"] == "m-original"
    assert merged["company_name"] == "Example Co"
    assert merged["principal_source_mention_id"] == "m-original"
    assert merged["synthesized_only"] is False

    alone = build_debt_instrument_rows(
        {"inst": ["m-prior"]},
        index,
        {},
        existing_instruments=pd.DataFrame(),
        company_names={},
    )[0]
    assert alone["name"] == "Fifth Amended and Restated Credit Agreement"
    assert alone["synthesized_only"] is True


def test_conflicting_end_dates_block_membership() -> None:
    """Same-day tranches with different maturities stay separate clusters."""
    seed = prepare_mention(mention_row())
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="6.75% senior notes due 2031",
            maturity_date="2031-06-01",
        )
    )
    assert score(other, profile_from(seed)) == []


def test_matching_end_dates_and_name_score_full_support() -> None:
    """A repeat mention of the same instrument scores with name support."""
    seed = prepare_mention(mention_row())
    repeat = prepare_mention(
        mention_row(debt_instrument_mention_id="mention-2", item_id="item-2")
    )
    candidates = score(repeat, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].match_score == 1.0
    assert candidates[0].support_family == "name"


def test_missing_end_date_still_matches() -> None:
    """A mention without a maturity is compatible with any cluster end date."""
    seed = prepare_mention(mention_row())
    partial = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2", item_id="item-2", maturity_date=None
        )
    )
    candidates = score(partial, profile_from(seed))
    assert len(candidates) == 1


def test_conflicting_name_rates_block_membership_without_end_dates() -> None:
    """Distinct coupon rates keep tranches apart even when maturities are absent."""
    seed = prepare_mention(mention_row(maturity_date=None))
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="6.75% senior notes due 2031",
            maturity_date=None,
        )
    )
    assert score(other, profile_from(seed)) == []


def test_end_dates_are_compatible_handles_missing_values() -> None:
    """Only two present-and-conflicting end dates are incompatible."""
    assert end_dates_are_compatible(None, None)
    assert end_dates_are_compatible("2028-06-01", None)
    assert end_dates_are_compatible(None, "2028-06-01")
    assert end_dates_are_compatible("2028-06-01", "2028-06-01")
    assert not end_dates_are_compatible("2028-06-01", "2031-06-01")


def test_end_dates_treat_only_name_derived_values_as_year_resolution() -> None:
    """A bare year from 'due YYYY' matches any date in that year (#128).

    A stated December 31 maturity is a real day and must agree exactly — the
    loose treatment applies only to the year-resolution marker produced by
    `normalized_end_date_for_matching` for name-derived values.
    """
    assert end_dates_are_compatible("2030", "2030-04-15")
    assert end_dates_are_compatible("2030-04-15", "2030")
    assert end_dates_are_compatible("2030", "2030-12-31")
    assert not end_dates_are_compatible("2030", "2031-04-15")
    assert not end_dates_are_compatible("2030-12-31", "2030-04-15")
    assert not end_dates_are_compatible("2030-04-15", "2030-06-01")


def test_normalized_end_date_for_matching_collapses_name_derived_year_ends() -> None:
    """Only a name-derived YYYY-12-31 collapses to its year (#128)."""
    derived = mention_row(
        maturity_date="2030-12-31",
        maturity_date_json=json.dumps({"derived_from": "name"}),
    )
    stated = mention_row(maturity_date="2030-12-31")
    assert prepare_mention(derived).normalized_end_date == "2030"
    assert prepare_mention(stated).normalized_end_date == "2030-12-31"


def test_year_resolution_end_date_still_matches_exact_maturity() -> None:
    """Pricing 8-K 'due 2030' merges with the closing 8-K's exact maturity."""
    seed = prepare_mention(
        mention_row(
            maturity_date="2030-12-31",
            maturity_date_json=json.dumps({"derived_from": "name"}),
        )
    )
    closing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            maturity_date="2030-04-15",
        )
    )
    assert len(score(closing, profile_from(seed))) == 1


def test_name_rates_are_compatible_requires_rates_on_both_sides() -> None:
    """Rate conflicts only apply when both fingerprints carry a rate."""
    assert name_rates_are_compatible(None, None)
    assert name_rates_are_compatible("term loan", "5.25% senior notes due 2028")
    assert name_rates_are_compatible(
        "5.25% senior notes due 2028", "5.25% senior notes due 2028"
    )
    assert not name_rates_are_compatible(
        "5.25% senior notes due 2028", "6.75% senior notes due 2031"
    )


def test_name_rate_tokens_reads_the_whole_coupon_not_its_fraction() -> None:
    """Read the whole coupon across the fingerprint's token break, canonicalized.

    `normalize_name_fingerprint` turns the decimal point into a space. Reading the raw token compared only the fractional digits: `4.375%` and
    `3.375%` both reduced to `375%`, so the compatibility guard could not tell
    two different coupons apart.
    """
    fp = normalize_name_fingerprint
    assert name_rate_tokens(fp("4.375% Senior Notes")) == frozenset({"4.375"})
    assert name_rate_tokens(fp("0.625% Notes")) == frozenset({"0.625"})
    assert name_rate_tokens(fp("10.25% Notes")) == frozenset({"10.25"})
    assert name_rate_tokens(fp("12.125% Notes")) == frozenset({"12.125"})
    # equivalent spellings collapse to one canonical rate
    assert name_rate_tokens(fp("5% Notes")) == name_rate_tokens(fp("5.00% Notes"))
    assert name_rate_tokens(fp("4.375% Notes")) == name_rate_tokens(fp("4.3750% Notes"))
    # a maturity year is not a coupon, and neither is a principal figure
    assert name_rate_tokens(fp("Notes due 2028 5% coupon")) == frozenset({"5"})
    assert name_rate_tokens(fp("$500 million 5% notes")) == frozenset({"5"})
    assert name_rate_tokens(fp("term loan")) == frozenset()


def test_name_rates_refuse_two_coupons_sharing_a_fractional_part() -> None:
    """The guard exists to refuse; a shared fraction must not read as a match."""
    fp = normalize_name_fingerprint
    for left, right in (
        ("4.375% Senior Notes", "3.375% Senior Notes"),
        ("10.25% Notes", "5.25% Notes"),
        ("0.625% Notes", "4.625% Notes"),
    ):
        assert not name_rates_are_compatible(fp(left), fp(right)), (left, right)
    # and equivalent spellings of one coupon still match
    assert name_rates_are_compatible(fp("5.00% Notes"), fp("5% Notes"))


def test_keyless_mention_matches_on_identifying_fingerprint() -> None:
    """A redemption mention without amount/start joins its instrument by name."""
    seed = prepare_mention(mention_row())
    redemption = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            principal_amount=None,
            start_date=None,
            maturity_date=None,
        )
    )
    candidates = score(redemption, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].match_score == 0.90
    assert candidates[0].basis == "name_fingerprint"
    assert candidates[0].base_match_via == "name_fingerprint"


def test_keyless_mention_with_generic_name_stays_unmatched() -> None:
    """Names without a rate or maturity year cannot identify an instrument."""
    seed = prepare_mention(mention_row(name="senior secured notes"))
    redemption = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="senior secured notes",
            principal_amount=None,
            start_date=None,
            maturity_date=None,
        )
    )
    assert score(redemption, profile_from(seed)) == []


def test_keyless_mention_requires_exact_fingerprint_in_cluster() -> None:
    """An identifying fingerprint only matches clusters that contain it."""
    seed = prepare_mention(mention_row())
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="6.75% senior notes due 2031",
            principal_amount=None,
            start_date=None,
            maturity_date=None,
        )
    )
    assert score(other, profile_from(seed)) == []


def test_keyless_mention_still_blocked_by_end_date_conflict() -> None:
    """A rate-only name match is rejected when maturities conflict."""
    seed = prepare_mention(
        mention_row(name="5.25% senior secured notes", maturity_date="2028-06-01")
    )
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="5.25% senior secured notes",
            principal_amount=None,
            start_date=None,
            maturity_date="2031-06-01",
        )
    )
    assert score(other, profile_from(seed)) == []


def test_partial_key_mention_matches_despite_amount_conflict() -> None:
    """A partial redemption amount does not block a fingerprint match."""
    seed = prepare_mention(mention_row())
    partial = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            principal_amount="400000000",
            start_date=None,
        )
    )
    candidates = score(partial, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_year_only_fingerprint_identifies() -> None:
    """A maturity year individuates within one CIK, so a keyless mention attaches.

    This reverses #78's original call that a coupon was required. Measured on the
    held-out window, requiring the coupon was the main reason an announcement
    8-K could never reach its own closing: recall on hand-labelled merges was
    0.38. The accepted cost is the case this test used to protect — one issuer
    announcing two distinct generic `Senior Secured Notes due YYYY` through
    different subsidiaries now merges them. No such case appears in the labelled
    window, so the risk is real but unmeasured (#123).
    """
    seed = prepare_mention(
        mention_row(name="Senior Secured Notes due 2027", maturity_date="2027-12-31")
    )
    later = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            name="Senior Secured Notes due 2027",
            principal_amount=None,
            start_date=None,
            maturity_date="2027-12-31",
        )
    )
    candidates = score(later, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_announcement_name_without_the_coupon_attaches_to_its_closing() -> None:
    """`senior notes due 2034` and `7.500% senior notes due 2034` are one debt (R3)."""
    announcement = prepare_mention(
        mention_row(
            name="senior secured first lien notes due 2034",
            principal_amount="750000000",
            start_date=None,
            maturity_date="2034-12-31",
            maturity_date_json=json.dumps({"derived_from": "name"}),
        )
    )
    closing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            name="7.500% senior secured first lien notes due 2034",
            principal_amount="750000000",
            start_date="2026-08-21",
            maturity_date="2034-09-15",
        )
    )
    candidates = score(closing, profile_from(announcement))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_a_class_designator_is_not_a_shortened_name() -> None:
    """`Tranche A Loan` must not subsume `Tranche B Loan` (R3 guard)."""
    tranche_a = prepare_mention(
        mention_row(
            name="Tranche A Loan",
            principal_amount="75000000",
            start_date=None,
            maturity_date="2031-07-10",
        )
    )
    tranche_b = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="Tranche B Loan",
            principal_amount="25000000",
            start_date=None,
            maturity_date="2031-07-10",
        )
    )
    assert score(tranche_b, profile_from(tranche_a)) == []


def test_a_generic_issuer_name_turns_off_the_relaxed_key_rule() -> None:
    """A name shared across the issuer is a template, not an identifier (R12 gate).

    FHLB Dallas files dozens of `Consolidated Obligation Bonds` with no dates and
    repeated round amounts, so one amount collision would otherwise merge
    distinct bonds.
    """
    first = prepare_mention(
        mention_row(
            name="Consolidated Obligation Bonds",
            principal_amount="10000000",
            start_date=None,
            maturity_date=None,
        )
    )
    second = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            name="Consolidated Obligation Bonds",
            principal_amount="10000000",
            start_date=None,
            maturity_date=None,
        )
    )
    # Two mentions: inside the gate, the shared amount attaches the second one.
    assert len(score(second, profile_from(first), name_class_size=2)) == 1
    # The same pair inside a CIK that files many identically-named bonds: off.
    assert score(second, profile_from(first), name_class_size=9) == []


def test_upsized_pricing_mention_attaches_by_fingerprint() -> None:
    """A pricing 8-K that upsizes the launch amount still joins the offering.

    Cleveland-Cliffs launched $800M and priced $900M of the same notes on one
    day, so the two filings can share a start date. They are separate items,
    which is what keeps this apart from two siblings in one document (#131).
    """
    launch = prepare_mention(
        mention_row(
            name="9.875% Senior Secured Notes due 2025", principal_amount="400000000"
        )
    )
    pricing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            accession_number="0000000000-24-000002",
            name="9.875% Senior Secured Notes due 2025",
            principal_amount="555159000",
        )
    )
    candidates = score(pricing, profile_from(launch))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_same_item_sibling_with_a_conflicting_amount_does_not_attach() -> None:
    """Two objects from one item that differ only in principal stay apart (#131)."""
    initial = prepare_mention(
        mention_row(
            name="10% Senior Secured Convertible Note",
            principal_amount="1250000",
            start_date="2026-08-13",
            maturity_date=None,
        )
    )
    additional = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="10% Senior Secured Convertible Note",
            principal_amount="1100000",
            start_date="2026-08-13",
            maturity_date=None,
        )
    )
    assert score(additional, profile_from(initial)) == []


def test_same_item_mentions_never_merge() -> None:
    """Same-item pairs are distinct debts however identifying the name is (#161).

    One item returns one object per instrument by extractor construction.

    Gray Media's $70M add-on tap merged into its $775M parent series and
    published the series at the add-on's size. Cross-filing merges still work:
    the same pair from different items attaches by fingerprint.
    """
    series = prepare_mention(
        mention_row(
            name="5.875% Senior Notes due 2034",
            principal_amount="500000000",
            start_date="2026-05-29",
            maturity_date="2034-12-31",
        )
    )
    same_item_add_on = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="5.875% Senior Notes due 2034",
            principal_amount="100000000",
            start_date="2026-08-13",
            maturity_date="2034-12-31",
        )
    )
    assert score(same_item_add_on, profile_from(series)) == []

    cross_item_closing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-3",
            item_id="item-2",
            name="5.875% Senior Notes due 2034",
            principal_amount="500000000",
            start_date="2026-06-02",
            maturity_date="2034-12-31",
        )
    )
    candidates = score(cross_item_closing, profile_from(series))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_closing_mention_with_drifted_start_date_attaches_by_fingerprint() -> None:
    """A closing 8-K dated at settlement still joins the priced offering."""
    pricing = prepare_mention(mention_row(start_date="2024-03-05"))
    closing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            start_date="2024-03-19",
        )
    )
    candidates = score(closing, profile_from(pricing))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_generic_name_upsize_stays_split() -> None:
    """Launch names without a coupon cannot bridge conflicting amounts."""
    launch = prepare_mention(
        mention_row(
            name="Senior Guaranteed Notes due 2029", principal_amount="800000000"
        )
    )
    pricing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="Senior Guaranteed Notes due 2029",
            principal_amount="900000000",
        )
    )
    assert score(pricing, profile_from(launch)) == []


def test_exact_key_match_keeps_amount_start_basis() -> None:
    """Full-evidence matches still report the amount_start basis."""
    seed = prepare_mention(mention_row())
    repeat = prepare_mention(
        mention_row(debt_instrument_mention_id="mention-2", item_id="item-2")
    )
    candidates = score(repeat, profile_from(seed))
    assert candidates[0].basis == "amount_start"
    assert candidates[0].match_score == 1.0


def test_relation_target_cannot_join_declaring_cluster() -> None:
    """The new exchange notes stay outside the retired old notes' cluster."""
    old_notes = prepare_mention(
        mention_row(
            name="6.375% Senior Notes due 2025",
            principal_amount="38400000",
            start_date="2010-05-11",
            retired_by_json='["mention-new"]',
        )
    )
    new_notes = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-new",
            name="6.375% Senior Notes due 2025",
            principal_amount="231800000",
        )
    )
    assert score(new_notes, profile_from(old_notes)) == []


def test_declaring_mention_cannot_join_target_cluster() -> None:
    """The retired old notes stay outside the new notes' cluster too."""
    new_notes = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-new",
            name="6.375% Senior Notes due 2025",
            principal_amount="231800000",
        )
    )
    old_notes = prepare_mention(
        mention_row(
            name="6.375% Senior Notes due 2025",
            principal_amount="38400000",
            start_date="2010-05-11",
            retired_by_json='["mention-new"]',
        )
    )
    assert score(old_notes, profile_from(new_notes)) == []


LENDER_JSON = (
    '[{"mentions": [{"text": "JPMorgan Chase Bank, N.A.", "tag_id": "tag-1"}],'
    ' "tag_ids": ["tag-1"]}]'
)


def test_name_conflict_suppresses_lender_support() -> None:
    """Facility components sharing lenders and totals stay separate."""
    revolver = prepare_mention(
        mention_row(name="Revolving Loans", parties_json=LENDER_JSON)
    )
    swing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            name="Swing Line Loans",
            parties_json=LENDER_JSON,
        )
    )
    candidates = score(swing, profile_from(revolver))
    assert len(candidates) == 1
    assert candidates[0].match_score == 0.75
    assert candidates[0].support_family is None


def test_lender_support_still_applies_without_name_conflict() -> None:
    """Shared lenders vouch for membership when names do not disagree."""
    seed = prepare_mention(
        mention_row(name="Revolving Loans", parties_json=LENDER_JSON)
    )
    unnamed = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            name=None,
            parties_json=LENDER_JSON,
        )
    )
    candidates = score(unnamed, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].support_family == "lenders"
    assert candidates[0].match_score == 1.0


def test_published_retirers_are_sorted_not_set_order() -> None:
    """The published retirer array is sorted rather than set-iteration order (#180).

    `derive_parent_links` collects retirers in a set, so dropping its `sorted()`
    leaves the published order down to how the ids happen to hash. A realistic
    two-retirer fixture therefore catches that only about half the time, and
    which half depends on `PYTHONHASHSEED`. Eight ids, fed in reverse order,
    make an accidentally sorted set vanishingly unlikely.
    """
    retirers = [f"m-retirer-{index}" for index in range(8)]
    retired = prepare_mention(
        mention_row(
            debt_instrument_mention_id="m-old",
            retired_by_json=json.dumps(list(reversed(retirers))),
        )
    )

    links = derive_parent_links(
        {"m-old": ["m-old"]},
        {"m-old": retired},
        {"m-old": "m-old", **{retirer: retirer for retirer in retirers}},
    )

    assert links["m-old"]["retired_by_debt_instrument_ids"] == json.dumps(retirers)


def test_name_derived_month_end_collapses_to_month_resolution() -> None:
    """`due April 2033` in a name matches any stated date in that month (#164)."""
    derived = mention_row(
        maturity_date="2033-04-30",
        maturity_date_json=json.dumps({"derived_from": "name"}),
    )
    stated = mention_row(maturity_date="2033-04-30")
    assert prepare_mention(derived).normalized_end_date == "2033-04"
    assert prepare_mention(stated).normalized_end_date == "2033-04-30"

    assert end_dates_are_compatible("2033-04", "2033-04-15")
    assert end_dates_are_compatible("2033-04-15", "2033-04")
    assert end_dates_are_compatible("2033-04", "2033")
    assert not end_dates_are_compatible("2033-04", "2033-06-15")
    assert not end_dates_are_compatible("2033-04-30", "2033-04-15")


def test_computed_maturity_collapses_to_month_resolution() -> None:
    """Start-plus-tenor maturities are month-trustworthy, not day-exact (#166)."""
    computed = mention_row(
        maturity_date="2031-06-24",
        maturity_date_json=json.dumps({"derived_from": "computed"}),
    )
    assert prepare_mention(computed).normalized_end_date == "2031-06"
    assert end_dates_are_compatible("2031-06", "2031-06-30")
    assert not end_dates_are_compatible("2031-06", "2031-09-30")


def test_a_generic_cluster_cannot_claim_an_individuating_name() -> None:
    """The guard behind GEO's shattered note histories had no test.

    A cluster whose every name is the generic `senior notes` must not absorb a
    mention whose name individuates a series; letting it seeded the tie cascade
    that produced 263 ambiguous edges on the 2026-09 window.
    """
    generic = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-generic",
            name="senior notes",
            start_date="2020-01-01",
            principal_amount="100000000",
            maturity_date=None,
        )
    )
    individuating = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-series",
            item_id="item-2",
            name="5.25% senior notes due 2028",
            start_date="2024-06-01",
            principal_amount="500000000",
        )
    )

    assert score(individuating, profile_from(generic)) == []

    # Control: the same mention against a cluster that carries an identifying
    # name attaches on the fingerprint path, so it is the cluster's genericness
    # that blocked it above, not its keys.
    identifying = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-identifying",
            name="5.25% senior notes due 2028",
            start_date="2020-01-01",
            principal_amount="100000000",
            maturity_date=None,
        )
    )
    attached = score(individuating, profile_from(identifying))
    assert [candidate.basis for candidate in attached] == ["name_fingerprint"]


def test_name_only_tie_prefers_a_live_cluster_over_a_retired_one() -> None:
    """The second of four sort components, which no fixture varied."""
    from cdt.matcher.core import CandidateScore, resolve_candidates

    mention = prepare_mention(mention_row(debt_instrument_mention_id="mention-new"))

    def candidate(instrument_id: str, *, retired: bool) -> CandidateScore:
        return CandidateScore(
            debt_instrument_id=instrument_id,
            match_score=0.90,
            support_family="name",
            basis="name_fingerprint",
            exact_name=True,
            cluster_size=2,
            cluster_retired=retired,
        )

    chosen, _ = resolve_candidates(
        mention,
        [
            candidate("inst-retired", retired=True),
            candidate("inst-live", retired=False),
        ],
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
        ambiguity_margin=0.05,
        evaluated_run_id="run-1",
    )
    assert chosen == "inst-live"


def test_name_only_tie_prefers_the_largest_cluster_then_the_lowest_id() -> None:
    """The third and fourth sort components; the fourth is what makes it stable."""
    from cdt.matcher.core import CandidateScore, resolve_candidates

    mention = prepare_mention(mention_row(debt_instrument_mention_id="mention-new"))

    def candidate(instrument_id: str, *, size: int) -> CandidateScore:
        return CandidateScore(
            debt_instrument_id=instrument_id,
            match_score=0.90,
            support_family="name",
            basis="name_fingerprint",
            exact_name=True,
            cluster_size=size,
            cluster_retired=False,
        )

    kwargs = {
        "strong_match_threshold": 0.90,
        "loose_match_threshold": 0.75,
        "ambiguity_margin": 0.05,
        "evaluated_run_id": "run-1",
    }
    bigger, _ = resolve_candidates(
        mention,
        [candidate("inst-small", size=1), candidate("inst-big", size=5)],
        **kwargs,
    )
    assert bigger == "inst-big"
    # Equal on every earlier component, the id breaks the tie deterministically,
    # which is what keeps resolution reproducible across runs (#171).
    stable, _ = resolve_candidates(
        mention, [candidate("zzz", size=3), candidate("aaa", size=3)], **kwargs
    )
    assert stable == "aaa"


def test_relaxed_key_rule_is_gated_at_exactly_the_name_class_gate() -> None:
    """Pin the gate's value, not just a wide straddle around it.

    `test_a_generic_issuer_name_turns_off_the_relaxed_key_rule` compares 2
    against 9, so raising `NAME_CLASS_GATE` from 2 to 8 left it passing. Testing
    the boundary itself is what fixes the constant in place.
    """
    first = prepare_mention(
        mention_row(
            name="Consolidated Obligation Bonds",
            principal_amount="10000000",
            start_date=None,
            maturity_date=None,
        )
    )
    second = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            item_id="item-2",
            name="Consolidated Obligation Bonds",
            principal_amount="10000000",
            start_date=None,
            maturity_date=None,
        )
    )
    profiles = profile_from(first)

    assert len(score(second, profiles, name_class_size=NAME_CLASS_GATE)) == 1
    assert score(second, profiles, name_class_size=NAME_CLASS_GATE + 1) == []


def test_lender_keys_reads_only_lender_clusters() -> None:
    """Without the role filter a borrower's name joins the lender signature."""
    from cdt.matcher.core import lender_keys

    payload = json.dumps(
        [
            {"role": "lender", "spans": [{"text": "Acme Bank, N.A."}]},
            {"role": "borrower", "spans": [{"text": "Example Co"}]},
            {"role": "agent", "spans": [{"text": "Agent Trust Company"}]},
        ]
    )
    assert lender_keys(payload) == ["acme bank"]
    # A cluster with no role at all predates the unified parties list (#150)
    # and is still read as a lender.
    legacy = json.dumps([{"spans": [{"text": "Acme Bank, N.A."}]}])
    assert lender_keys(legacy) == ["acme bank"]


def test_an_extracted_amendment_pointer_beats_a_carried_inferred_one() -> None:
    """A guess and a fact must not cancel each other out (#203, #204).

    `derive_parent_links` used to seed `amendment_parents` with the existing
    row's pointer whatever its provenance, so a row carrying an inferred
    pointer to one instrument, whose mention now states an extracted pointer to
    another, held two candidates and the ambiguity guard threw both away. The
    pass then re-inferred its guess on the next run and the extracted link
    never came back. What the mentions state wins; the carried pointer is only
    a fallback, and it is what keeps an inferred link alive across an ordinary
    rematch since no mention names it.
    """
    child = prepare_mention(
        mention_row(debt_instrument_mention_id="m-child", amendment_of="m-real-parent")
    )
    existing = pd.DataFrame(
        [
            {
                "debt_instrument_id": "m-child",
                "amendment_of_debt_instrument_id": "m-guessed-parent",
                "amendment_inferred_by": "ordinal_chain",
                "retired_by_debt_instrument_ids": None,
                "split_of_debt_instrument_id": None,
            }
        ]
    )

    links = derive_parent_links(
        {"m-child": ["m-child"]},
        {"m-child": child},
        {"m-child": "m-child", "m-real-parent": "m-real-parent"},
        existing_instruments=existing,
    )

    assert links["m-child"]["amendment_of_debt_instrument_id"] == "m-real-parent"
    # provenance travels with the pointer: this one is extracted, not inferred
    assert links["m-child"]["amendment_inferred_by"] is None


def test_a_carried_pointer_survives_when_no_mention_names_a_parent() -> None:
    """The fallback is the whole reason an inferred pointer outlives a rematch (#184)."""
    child = prepare_mention(mention_row(debt_instrument_mention_id="m-child"))
    existing = pd.DataFrame(
        [
            {
                "debt_instrument_id": "m-child",
                "amendment_of_debt_instrument_id": "m-guessed-parent",
                "amendment_inferred_by": "ordinal_chain",
                "retired_by_debt_instrument_ids": None,
                "split_of_debt_instrument_id": None,
            }
        ]
    )

    links = derive_parent_links(
        {"m-child": ["m-child"]},
        {"m-child": child},
        {"m-child": "m-child"},
        existing_instruments=existing,
    )

    assert links["m-child"]["amendment_of_debt_instrument_id"] == "m-guessed-parent"
    assert links["m-child"]["amendment_inferred_by"] == "ordinal_chain"


def test_two_extracted_parents_refuse_rather_than_fall_back_to_a_guess() -> None:
    """An ambiguous extracted set is a refusal; falling back would publish a guess."""
    first = prepare_mention(
        mention_row(debt_instrument_mention_id="m-a", amendment_of="m-parent-1")
    )
    second = prepare_mention(
        mention_row(debt_instrument_mention_id="m-b", amendment_of="m-parent-2")
    )
    existing = pd.DataFrame(
        [
            {
                "debt_instrument_id": "m-a",
                "amendment_of_debt_instrument_id": "m-guessed-parent",
                "amendment_inferred_by": "ordinal_chain",
                "retired_by_debt_instrument_ids": None,
                "split_of_debt_instrument_id": None,
            }
        ]
    )

    links = derive_parent_links(
        {"m-a": ["m-a", "m-b"]},
        {"m-a": first, "m-b": second},
        {
            "m-a": "m-a",
            "m-b": "m-a",
            "m-parent-1": "m-parent-1",
            "m-parent-2": "m-parent-2",
        },
        existing_instruments=existing,
    )

    assert links["m-a"]["amendment_of_debt_instrument_id"] is None
    assert links["m-a"]["amendment_inferred_by"] is None
