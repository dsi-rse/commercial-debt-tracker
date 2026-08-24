"""Matcher scoring tests."""

from __future__ import annotations

from cdt.matcher.core import (
    PreparedMention,
    build_empty_profile,
    end_dates_are_compatible,
    name_rates_are_compatible,
    prepare_mention,
    score_candidates_for_mention,
)


def mention_row(**overrides: object) -> dict[str, object]:
    """Return one mention row with sensible defaults."""
    row: dict[str, object] = {
        "debt_instrument_mention_id": "mention-1",
        "item_id": "item-1",
        "raw_id": "raw-1",
        "accession_number": "0000000000-24-000001",
        "cik": "0000320193",
        "company_name": "Example Co",
        "date": "2024-06-01",
        "name": "5.25% senior notes due 2028",
        "start_date": "2024-06-01",
        "end_date": "2028-06-01",
        "amount": "500000000",
        "amendment_of": None,
        "retired_of": None,
        "split_of": None,
        "lenders_json": "[]",
        "lenders_known_incomplete": False,
        "other_interested_parties_json": "[]",
    }
    row.update(overrides)
    return row


def profile_from(mention: PreparedMention) -> dict[str, object]:
    """Return a one-cluster profile map seeded from one mention."""
    profile = build_empty_profile(mention.debt_instrument_mention_id, mention)
    profile.add_member(mention)
    return {profile.debt_instrument_id: profile}


def score(mention: PreparedMention, profiles: dict[str, object]) -> list[object]:
    """Score one mention with default thresholds."""
    return score_candidates_for_mention(
        mention,
        profiles,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )


def test_conflicting_end_dates_block_membership() -> None:
    """Same-day tranches with different maturities stay separate clusters."""
    seed = prepare_mention(mention_row())
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="6.75% senior notes due 2031",
            end_date="2031-06-01",
        )
    )
    assert score(other, profile_from(seed)) == []


def test_matching_end_dates_and_name_score_full_support() -> None:
    """A repeat mention of the same instrument scores with name support."""
    seed = prepare_mention(mention_row())
    repeat = prepare_mention(mention_row(debt_instrument_mention_id="mention-2"))
    candidates = score(repeat, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].match_score == 1.0
    assert candidates[0].support_family == "name"


def test_missing_end_date_still_matches() -> None:
    """A mention without a maturity is compatible with any cluster end date."""
    seed = prepare_mention(mention_row())
    partial = prepare_mention(
        mention_row(debt_instrument_mention_id="mention-2", end_date=None)
    )
    candidates = score(partial, profile_from(seed))
    assert len(candidates) == 1


def test_conflicting_name_rates_block_membership_without_end_dates() -> None:
    """Distinct coupon rates keep tranches apart even when maturities are absent."""
    seed = prepare_mention(mention_row(end_date=None))
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="6.75% senior notes due 2031",
            end_date=None,
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


def test_end_dates_treat_december_31_as_year_resolution() -> None:
    """A YYYY-12-31 sentinel from 'due YYYY' matches any date in that year."""
    assert end_dates_are_compatible("2030-12-31", "2030-04-15")
    assert end_dates_are_compatible("2030-04-15", "2030-12-31")
    assert not end_dates_are_compatible("2030-12-31", "2031-04-15")
    assert not end_dates_are_compatible("2030-04-15", "2030-06-01")


def test_year_resolution_end_date_still_matches_exact_maturity() -> None:
    """Pricing 8-K 'due 2030' merges with the closing 8-K's exact maturity."""
    seed = prepare_mention(mention_row(end_date="2030-12-31"))
    closing = prepare_mention(
        mention_row(debt_instrument_mention_id="mention-2", end_date="2030-04-15")
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


def test_keyless_mention_matches_on_identifying_fingerprint() -> None:
    """A redemption mention without amount/start joins its instrument by name."""
    seed = prepare_mention(mention_row())
    redemption = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            amount=None,
            start_date=None,
            end_date=None,
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
            amount=None,
            start_date=None,
            end_date=None,
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
            amount=None,
            start_date=None,
            end_date=None,
        )
    )
    assert score(other, profile_from(seed)) == []


def test_keyless_mention_still_blocked_by_end_date_conflict() -> None:
    """A rate-only name match is rejected when maturities conflict."""
    seed = prepare_mention(
        mention_row(name="5.25% senior secured notes", end_date="2028-06-01")
    )
    other = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="5.25% senior secured notes",
            amount=None,
            start_date=None,
            end_date="2031-06-01",
        )
    )
    assert score(other, profile_from(seed)) == []


def test_partial_key_mention_matches_despite_amount_conflict() -> None:
    """A partial redemption amount does not block a fingerprint match."""
    seed = prepare_mention(mention_row())
    partial = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            amount="400000000",
            start_date=None,
        )
    )
    candidates = score(partial, profile_from(seed))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_year_only_fingerprint_is_not_identifying() -> None:
    """Class-plus-year names collide across subsidiary issuers of one CIK."""
    seed = prepare_mention(
        mention_row(name="Senior Secured Notes due 2027", end_date="2027-12-31")
    )
    other_subsidiary = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="Senior Secured Notes due 2027",
            amount=None,
            start_date=None,
            end_date="2027-12-31",
        )
    )
    assert score(other_subsidiary, profile_from(seed)) == []


def test_upsized_pricing_mention_attaches_by_fingerprint() -> None:
    """A pricing 8-K that upsizes the launch amount still joins the offering."""
    launch = prepare_mention(
        mention_row(name="9.875% Senior Secured Notes due 2025", amount="400000000")
    )
    pricing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="9.875% Senior Secured Notes due 2025",
            amount="555159000",
        )
    )
    candidates = score(pricing, profile_from(launch))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_closing_mention_with_drifted_start_date_attaches_by_fingerprint() -> None:
    """A closing 8-K dated at settlement still joins the priced offering."""
    pricing = prepare_mention(mention_row(start_date="2024-03-05"))
    closing = prepare_mention(
        mention_row(debt_instrument_mention_id="mention-2", start_date="2024-03-19")
    )
    candidates = score(closing, profile_from(pricing))
    assert len(candidates) == 1
    assert candidates[0].basis == "name_fingerprint"


def test_generic_name_upsize_stays_split() -> None:
    """Launch names without a coupon cannot bridge conflicting amounts."""
    launch = prepare_mention(
        mention_row(name="Senior Guaranteed Notes due 2029", amount="800000000")
    )
    pricing = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-2",
            name="Senior Guaranteed Notes due 2029",
            amount="900000000",
        )
    )
    assert score(pricing, profile_from(launch)) == []


def test_exact_key_match_keeps_amount_start_basis() -> None:
    """Full-evidence matches still report the amount_start basis."""
    seed = prepare_mention(mention_row())
    repeat = prepare_mention(mention_row(debt_instrument_mention_id="mention-2"))
    candidates = score(repeat, profile_from(seed))
    assert candidates[0].basis == "amount_start"
    assert candidates[0].match_score == 1.0


def test_relation_target_cannot_join_declaring_cluster() -> None:
    """Old notes retired by an exchange stay outside the new notes' cluster."""
    new_notes = prepare_mention(
        mention_row(
            name="6.375% Senior Notes due 2025",
            amount="231800000",
            retired_of="mention-old",
        )
    )
    old_notes = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-old",
            name="6.375% Senior Notes due 2025",
            amount="38400000",
            start_date="2010-05-11",
        )
    )
    assert score(old_notes, profile_from(new_notes)) == []


def test_declaring_mention_cannot_join_target_cluster() -> None:
    """The new exchange notes stay outside the old notes' cluster too."""
    old_notes = prepare_mention(
        mention_row(
            debt_instrument_mention_id="mention-old",
            name="6.375% Senior Notes due 2025",
            amount="38400000",
            start_date="2010-05-11",
        )
    )
    new_notes = prepare_mention(
        mention_row(
            name="6.375% Senior Notes due 2025",
            amount="231800000",
            retired_of="mention-old",
        )
    )
    assert score(new_notes, profile_from(old_notes)) == []
