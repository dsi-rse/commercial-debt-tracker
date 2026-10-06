"""Tests for turning cited text into dates, amounts, rates, parties and names."""

from __future__ import annotations

import json

from support import _dates_tag_details, maturity_row_state, party_row_state

from cdt.extractor.normalize.amounts import (
    canonical_amount_value,
    currency_candidates_from_text,
    currency_from_name,
    is_rate_like_amount_text,
    name_derived_principal_payload,
    normalized_amount_from_name,
    normalized_amount_from_text,
)
from cdt.extractor.normalize.dates import (
    date_plus_tenor,
    dates_agree,
    normalized_date_from_text,
    normalized_maturity_from_text,
    normalized_month_year_from_text,
)
from cdt.extractor.normalize.parties import canonical_instrument_name
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.stages import InstrumentIEStage
from cdt.extractor.validate import validate_amount_is_not_rate


def instrument_ie_mention(response: str) -> dict[str, object]:
    """Run instrument_ie postprocessing on one response and return its mention."""
    row_state = party_row_state()
    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert len(row_state.debt_instrument_mentions) == 1
    return row_state.debt_instrument_mentions[0]


def test_instrument_ie_postprocess_persists_every_party_with_role_and_kind() -> None:
    """Lenders, collective phrases, and other parties all persist labelled (#150)."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "parties": [
                        {"tag_ids": ["tag-o-named"], "role": "lender", "kind": "named"},
                        {
                            "tag_ids": ["tag-o-collective"],
                            "role": "lender",
                            "kind": "collective",
                        },
                        {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    ],
                }
            ]
        )
    )

    parties = json.loads(str(mention["parties_json"]))
    assert [
        (party["role"], party["kind"], [s["tag_id"] for s in party["spans"]])
        for party in parties
    ] == [
        ("lender", "named", ["tag-o-named"]),
        ("lender", "collective", ["tag-o-collective"]),
        ("agent", "named", ["tag-o-agent"]),
    ]
    assert all(
        set(party) == {"canonical_name", "role", "kind", "spans"} for party in parties
    )
    assert all(party["canonical_name"] for party in parties)
    assert mention["lender_disclosure"] == "collective_present"


def test_instrument_ie_postprocess_leaves_named_only_lenders_unflagged() -> None:
    """A lender list with only named clusters is not known to be incomplete."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "parties": [
                        {"tag_ids": ["tag-o-named"], "role": "lender", "kind": "named"}
                    ],
                }
            ]
        )
    )

    assert mention["lender_disclosure"] == "complete"
    assert len(json.loads(str(mention["parties_json"]))) == 1


def test_instrument_ie_postprocess_keeps_collective_lenders_and_flags() -> None:
    """A collective-only lender list persists with its surface text and flags (#150)."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "parties": [
                        {
                            "tag_ids": ["tag-o-collective"],
                            "role": "lender",
                            "kind": "collective",
                        }
                    ],
                }
            ]
        )
    )

    parties = json.loads(str(mention["parties_json"]))
    assert [(party["role"], party["kind"]) for party in parties] == [
        ("lender", "collective")
    ]
    assert mention["lender_disclosure"] == "collective_present"


def test_instrument_ie_postprocess_keeps_the_borrower_with_its_role() -> None:
    """The borrower persists with role "borrower" (#150).

    A subsidiary obligor under the parent filer's 8-K is exactly the identity
    worth recording.
    """
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "parties": [
                        {"tag_ids": ["tag-o-borrower"], "role": "borrower"},
                        {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    ],
                }
            ]
        )
    )

    parties = json.loads(str(mention["parties_json"]))
    assert [
        (party["role"], [s["tag_id"] for s in party["spans"]]) for party in parties
    ] == [
        ("borrower", ["tag-o-borrower"]),
        ("agent", ["tag-o-agent"]),
    ]


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


def test_normalized_maturity_from_text_rejects_coordinated_maturities() -> None:
    """One `due` listing two maturities identifies no single instrument (#104)."""
    assert normalized_maturity_from_text("notes due 2028 and 2030") is None
    assert normalized_maturity_from_text("notes due 2028, 2030") is None
    assert normalized_maturity_from_text("notes due 2028/2030") is None
    assert normalized_maturity_from_text("notes due October 1, 2028 and 2030") is None
    assert (
        normalized_maturity_from_text("notes due October 1, 2028 and October 1, 2030")
        is None
    )
    # A later year that is not coordinated onto the maturity is still not one.
    assert (
        normalized_maturity_from_text("notes due 2028, and 2030 obligations remain")
        == "2028-12-31"
    )


def test_instrument_ie_postprocess_keeps_name_derived_end_date() -> None:
    """A maturity cited from the name span should survive normalization."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "dates": [
                        {
                            "kind": "maturity",
                            "evidence": ["tag-i-1"],
                            "normalized_date": "2028-12-31",
                        },
                    ],
                }
            ]
        )
    )

    assert mention["maturity_date"] == "2028-12-31"
    payload = json.loads(str(mention["maturity_date_json"]))
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-i-1"]


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

    assert mention["maturity_date"] == "2028-12-31"
    payload = json.loads(str(mention["maturity_date_json"]))
    # A maturity read from the name has no citable date tag of its own.
    assert payload["spans"] == []


def test_instrument_ie_postprocess_drops_end_date_that_contradicts_evidence() -> None:
    """A normalized date that does not match its evidence text is still dropped."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "dates": [
                        {
                            "kind": "maturity",
                            "evidence": ["tag-d-1"],
                            "normalized_date": "2028-12-31",
                        },
                    ],
                }
            ]
        )
    )

    payload = json.loads(str(mention["maturity_date_json"]))
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-d-1"]
    # The cited date tag says March 17, 2025, so the model value is rejected and the
    # name maturity fills the gap instead.
    assert mention["maturity_date"] == "2028-12-31"


def test_is_rate_like_amount_text_separates_rates_from_principal() -> None:
    """Rate detection should key on percentages without currency or scale words."""
    assert is_rate_like_amount_text("0.875% per annum") is True
    assert is_rate_like_amount_text("5.75%") is True
    assert is_rate_like_amount_text("175 basis points") is True
    assert is_rate_like_amount_text("$500.0 million") is False
    assert is_rate_like_amount_text("$500,000,000 (100% of principal)") is False
    assert is_rate_like_amount_text("100% of the outstanding 30.0 million") is False
    assert is_rate_like_amount_text(None) is False


def test_is_rate_like_amount_text_requires_every_number_to_carry_a_rate() -> None:
    """A percentage of a stated principal is not itself a rate (#103).

    The currency symbol and the scale word are not what makes these principals;
    `normalized_amount_from_text` reads the first number, so a span whose first
    number carries no rate marker is stating an amount.
    """
    assert is_rate_like_amount_text("500,000,000 (100% of principal)") is False
    assert (
        is_rate_like_amount_text("500 million U.S. dollars, or 5% of assets") is False
    )
    assert is_rate_like_amount_text("1,500,000") is False
    # A margin range is still every-number-rated.
    assert is_rate_like_amount_text("0.875% to 1.875%") is True
    assert is_rate_like_amount_text("SOFR plus 100 basis points") is True


def test_a_basis_point_margin_is_rate_like_however_the_filing_abbreviates_it() -> None:
    """`bps` is the spelling filings use, and it was not in the marker set (#228).

    `RATE_SUFFIX_PATTERN` carried the spelled-out `basis points` but not the
    abbreviation, so `50 bps` read as a money amount and
    `validate_amount_is_not_rate` passed a basis-point margin through as an
    `amount`. `amounts_agree` does not catch it downstream either: it only
    rejects a model figure that disagrees with the number in the cited span,
    and the model reports the basis-point figure itself, so the two agree and
    the margin publishes as a principal of 50. A wrong published value, not the
    silent null of #182.

    Distinct from #75/#102, which was the whitespace that hid the multi-word
    marker from the predicate. That one is fixed, and the spelled-out assertions
    above still pin it; this is the vocabulary rather than the normalization, so
    it would have been present even with #102 perfect.
    """
    for spelling in ("50 bps", "50 bp", "50 BPS", "50 basis points"):
        assert is_rate_like_amount_text(spelling) is True, spelling
    # The shapes a margin is actually written in.
    assert is_rate_like_amount_text("L+250 bps") is True
    assert is_rate_like_amount_text("a margin of 275 bps") is True

    # ... and the validator that depends on it now refuses the amount.
    failures = validate_amount_is_not_rate(
        index=0,
        value={"evidence": ["tag-1"]},
        tag_details={"tag-1": {"text": "50 bps"}},
    )
    assert len(failures) == 1
    assert "'50 bps'" in failures[0]

    # `bps?\b` must not swallow a real figure: the word boundary keeps it off
    # longer words, and a principal still reads as a principal.
    assert is_rate_like_amount_text("500,000 bpd of crude") is False
    assert is_rate_like_amount_text("$500.0 million") is False
    assert is_rate_like_amount_text("1,500,000") is False


def test_normalized_date_from_text_reads_every_filing_spelling() -> None:
    """A format the parser cannot read discards a date the model got right (#133).

    `M/D/YYYY` alone cost 67 start dates and 67 end dates on one held-out
    window, all from tabular schedules whose rows the model had parsed
    correctly.
    """
    assert normalized_date_from_text("7/28/2026") == "2026-07-28"
    assert normalized_date_from_text("07/28/2026") == "2026-07-28"
    assert normalized_date_from_text("12/31/2030") == "2030-12-31"
    # The comma is optional, and non-US issuers write the day first.
    assert normalized_date_from_text("July 28 2026") == "2026-07-28"
    assert normalized_date_from_text("28 July 2026") == "2026-07-28"
    assert normalized_date_from_text("July 28, 2026") == "2026-07-28"
    # Zero-padding is optional on the way in.
    assert normalized_date_from_text("2026-7-8") == "2026-07-08"
    # A two-digit year needs a century guessed, so it stays unparsed.
    assert normalized_date_from_text("7/28/26") is None
    # Impossible dates and non-dates stay unparsed.
    assert normalized_date_from_text("13/28/2026") is None
    assert normalized_date_from_text("2/30/2026") is None
    assert normalized_date_from_text("Closing Date") is None
    assert normalized_date_from_text("August 2056") is None
    assert normalized_date_from_text("Section 8 2026") is None
    assert normalized_date_from_text("6.875% Senior Notes due March 2027") is None


def test_dates_agree_ignores_shape_but_not_value() -> None:
    """A model writing the same day differently keeps its date (#133)."""
    assert dates_agree("2026-7-28", "2026-07-28") is True
    assert dates_agree("2026-07-28", "2026-07-28") is True
    assert dates_agree("2026-07-29", "2026-07-28") is False
    assert dates_agree("soon", "2026-07-28") is False
    assert dates_agree(None, "2026-07-28") is False


def test_normalized_amount_from_name_reads_an_embedded_principal() -> None:
    """A principal inside the name parses; a coupon or maturity does not (#129).

    NER tags `$183.36 million term loan` as one `debt_instrument`, so there is no
    `amount` span to cite. Requiring a currency marker is what keeps the coupon
    rate and the maturity year from being read as the principal.
    """
    assert normalized_amount_from_name("$183.36 million term loan") == "183360000"
    assert normalized_amount_from_name("C$300 million notes due 2033") == "300000000"
    assert normalized_amount_from_name("$1,299,870.00 Promissory Note") == "1299870"
    assert currency_from_name("C$300 million notes due 2033") == "CAD"
    assert currency_from_name("€600.0 million 3.625% Notes due 2032") == "EUR"
    # No currency marker means no principal is stated in the name.
    assert normalized_amount_from_name("3.875% senior notes due 2028") is None
    assert normalized_amount_from_name("revolving credit facility") is None
    # A name stating two figures names no single principal, as #104 requires for
    # maturities.
    assert (
        normalized_amount_from_name("$500 million and $750 million facilities") is None
    )


def test_normalized_amount_from_text_keeps_cents_exact() -> None:
    """A cents value parses to itself, not to a float artifact (#119).

    `float("372246148.11")` is not that number, and the old `f"{value:.12f}"`
    rendering exposed the difference, so the string never matched what the model
    reported and the amount published as null.
    """
    assert normalized_amount_from_text("$372,246,148.11") == "372246148.11"
    assert normalized_amount_from_text("$55,637.41") == "55637.41"
    assert normalized_amount_from_text("$5,529,722.96") == "5529722.96"
    # Trailing zeros and scale words still collapse to one canonical form.
    assert normalized_amount_from_text("$500,000.00") == "500000"
    assert normalized_amount_from_text("$70.0 million") == "70000000"
    assert normalized_amount_from_text("1.5 billion") == "1500000000"


def test_canonical_amount_value_prefers_the_span_that_parses() -> None:
    """A longer label must not beat the figure it names (#120)."""
    tag_details = {
        "tag-a-1": {"type": "amount", "text": "$2,000,000"},
        "tag-a-2": {"type": "amount", "text": "Principal Amount"},
    }

    assert canonical_amount_value(["tag-a-1", "tag-a-2"], tag_details) == "$2,000,000"
    # With nothing parseable, the longest span is still the canonical text.
    assert canonical_amount_value(["tag-a-2"], tag_details) == "Principal Amount"
    # Among parseable spans the longest still wins, as it did before.
    tag_details["tag-a-3"] = {"type": "amount", "text": "$2,000,000 in principal"}
    assert (
        canonical_amount_value(["tag-a-1", "tag-a-3"], tag_details)
        == "$2,000,000 in principal"
    )


def test_currency_candidates_read_a_qualified_dollar_sign() -> None:
    """`C$` is Canadian, not US, dollars (#121)."""
    assert currency_candidates_from_text("C$300 million") == {"CAD"}
    assert currency_candidates_from_text("A$50,000,000") == {"AUD"}
    assert currency_candidates_from_text("NZ$10 million") == {"NZD"}
    # An unqualified dollar sign keeps its USD reading.
    assert currency_candidates_from_text("$500.0 million") == {"USD"}
    assert currency_candidates_from_text("500 million U.S. dollars") == {"USD"}
    # A span quoting both currencies offers both.
    assert currency_candidates_from_text("C$300 million (US$220 million)") == {
        "CAD",
        "USD",
    }


def test_standardized_payloads_record_where_their_values_came_from() -> None:
    """Payloads carry derived_from so consumers know a value's provenance (#128)."""
    from cdt.extractor.normalize.amounts import standardized_amount_payload
    from cdt.extractor.normalize.dates import (
        standardized_date_payload,
        standardized_end_date_payload,
    )

    tag_details = {
        "tag-d-1": {
            "type": "date",
            "text": "June 1, 2028",
            "char_start": 0,
            "char_end": 12,
        },
        "tag-i-1": {
            "type": "debt_instrument",
            "text": "3.875% senior notes due 2028",
            "char_start": 20,
            "char_end": 48,
        },
        "tag-a-1": {
            "type": "amount",
            "text": "$500,000,000",
            "char_start": 60,
            "char_end": 72,
        },
    }
    stated_date = standardized_date_payload(
        {"evidence": ["tag-d-1"], "normalized_date": "2028-06-01"},
        tag_details,
    )
    assert stated_date["normalized_date"] == "2028-06-01"
    assert stated_date["derived_from"] == "stated"

    name_maturity = standardized_end_date_payload(
        {"evidence": ["tag-i-1"], "normalized_date": "2028-12-31"},
        tag_details,
        name_text="3.875% senior notes due 2028",
    )
    assert name_maturity["normalized_date"] == "2028-12-31"
    assert name_maturity["derived_from"] == "name"

    fallback_maturity = standardized_end_date_payload(
        None,
        tag_details,
        name_text="3.875% senior notes due 2028",
    )
    assert fallback_maturity["normalized_date"] == "2028-12-31"
    assert fallback_maturity["derived_from"] == "name"
    assert fallback_maturity["spans"] == []

    stated_amount = standardized_amount_payload(
        {"evidence": ["tag-a-1"], "normalized_amount": "500000000"},
        tag_details,
    )
    assert stated_amount["normalized_amount"] == "500000000"
    assert stated_amount["derived_from"] == "stated"

    absent = standardized_date_payload(None, tag_details)
    assert absent["normalized_date"] is None
    assert absent["derived_from"] is None


def test_normalized_maturity_from_text_parses_month_year_phrases() -> None:
    """`due April 2033` normalizes to the month's last day (#164)."""
    assert normalized_maturity_from_text("notes due April 2033") == "2033-04-30"
    assert normalized_maturity_from_text("notes due in February 2028") == "2028-02-29"
    assert normalized_maturity_from_text("due September 2031") == "2031-09-30"
    # A full date still wins its own precision, and coordinated month-years
    # are two maturities.
    assert normalized_maturity_from_text("due April 7, 2033") == "2033-04-07"
    assert normalized_maturity_from_text("due April 2033 and June 2035") is None
    assert normalized_maturity_from_text("due April 2033 and 2035") is None
    assert normalized_maturity_from_text("due October 1, 2028 and April 2030") is None
    # A month-year restating the full date's own month is not a second maturity.
    assert (
        normalized_maturity_from_text("due April 7, 2033, i.e. due April 2033")
        == "2033-04-07"
    )


def test_computed_sum_amount_accepts_only_the_exact_sum_of_cited_spans() -> None:
    """An increase-by amendment's unstated total publishes as computed (#165)."""
    from cdt.extractor.normalize.amounts import standardized_amount_payload

    tag_details = {
        "tag-a-before": {
            "type": "amount",
            "text": "$200 million",
            "char_start": 10,
            "char_end": 22,
        },
        "tag-a-increment": {
            "type": "amount",
            "text": "$50 million",
            "char_start": 40,
            "char_end": 51,
        },
        "tag-a-rate": {
            "type": "amount",
            "text": "0.50%",
            "char_start": 60,
            "char_end": 65,
        },
    }
    computed = standardized_amount_payload(
        {
            "evidence": ["tag-a-before", "tag-a-increment"],
            "normalized_amount": "250000000",
            "currency": "USD",
        },
        tag_details,
    )
    assert computed["normalized_amount"] == "250000000"
    assert computed["derived_from"] == "computed"
    assert computed["currency"] == "USD"

    # A value that is not the exact sum stays null.
    wrong = standardized_amount_payload(
        {
            "evidence": ["tag-a-before", "tag-a-increment"],
            "normalized_amount": "300000000",
        },
        tag_details,
    )
    assert wrong["normalized_amount"] is None
    assert wrong["derived_from"] is None

    # A single-span citation is agreement, never computation.
    single = standardized_amount_payload(
        {
            "evidence": ["tag-a-before"],
            "normalized_amount": "200000000",
            "currency": "USD",
        },
        tag_details,
    )
    assert single["normalized_amount"] == "200000000"
    assert single["derived_from"] == "stated"

    # A rate-like span in the citation disables the computed path.
    with_rate = standardized_amount_payload(
        {
            "evidence": ["tag-a-before", "tag-a-rate"],
            "normalized_amount": "200000000.5",
        },
        tag_details,
    )
    assert with_rate["normalized_amount"] is None


def test_tenor_parsing_and_date_arithmetic() -> None:
    """Tenor spans parse conservatively; month-end days clamp (#166)."""
    from cdt.extractor.normalize.dates import date_plus_tenor, tenor_from_text

    assert tenor_from_text("five-year") == (5, "year")
    assert tenor_from_text("364-day") == (364, "day")
    assert tenor_from_text("18-month") == (18, "month")
    assert tenor_from_text("three year") == (3, "year")
    # Two distinct tenors anchor nothing.
    assert tenor_from_text("three-year term plus two one-year extensions") is None
    assert tenor_from_text("no tenor here") is None

    assert date_plus_tenor("2026-06-24", (5, "year")) == "2031-06-24"
    assert date_plus_tenor("2026-01-02", (364, "day")) == "2027-01-01"
    assert date_plus_tenor("2026-08-31", (18, "month")) == "2028-02-29"


def test_date_payload_verifies_against_every_cited_span() -> None:
    """A date co-cited with a defined term keeps its value (2026-09 window)."""
    from cdt.extractor.normalize.dates import standardized_date_payload

    tags = {
        "tag-7": {
            "text": "March 2, 2026",
            "type": "date",
            "char_start": 0,
            "char_end": 13,
        },
        "tag-8": {
            "text": "Redemption Date",
            "type": "date",
            "char_start": 20,
            "char_end": 35,
        },
    }
    payload = standardized_date_payload(
        {"evidence": ["tag-7", "tag-8"], "normalized_date": "2026-03-02"}, tags
    )
    assert payload["normalized_date"] == "2026-03-02"
    assert payload["derived_from"] == "stated"


def test_month_year_maturity_is_read_outside_due_phrases() -> None:
    """`in March 2056` is a stated month-resolution maturity, not a start date."""
    from cdt.extractor.normalize.dates import (
        normalized_date_from_text,
        normalized_month_year_from_text,
        standardized_date_payload,
    )

    assert normalized_month_year_from_text("in March 2056") == "2056-03-31"
    assert normalized_month_year_from_text("June 2016") == "2016-06-30"
    assert normalized_month_year_from_text("Series 2026A") is None
    assert normalized_month_year_from_text("April 2033 and June 2035") is None
    assert normalized_date_from_text("in March 2056") is None
    tags = {
        "tag-3": {
            "text": "in March 2056",
            "type": "date",
            "char_start": 0,
            "char_end": 13,
        }
    }
    value = {"evidence": ["tag-3"], "normalized_date": "2056-03-31"}
    maturity = standardized_date_payload(value, tags, allow_maturity_phrase=True)
    assert maturity["normalized_date"] == "2056-03-31"
    assert maturity["derived_from"] == "stated"
    start = standardized_date_payload(value, tags)
    assert start["normalized_date"] is None


def test_rate_spelled_percent_parses() -> None:
    """`6.5 percent` is a rate, for the rate payload and the amount guard alike."""
    from cdt.extractor.normalize.amounts import is_rate_like_amount_text
    from cdt.extractor.schema import RATE_PCT_PATTERN

    assert RATE_PCT_PATTERN.findall("6.5 percent senior notes due 2028") == ["6.5"]
    assert RATE_PCT_PATTERN.findall("4.950% notes") == ["4.950"]
    assert is_rate_like_amount_text("6.5 percent") is True


def test_computed_maturity_accepts_a_cited_date_minus_a_tenor() -> None:
    """`extended six months to September 3, 2027` anchors the prior maturity (#166)."""
    from cdt.extractor.normalize.dates import computed_maturity_date, date_plus_tenor

    assert date_plus_tenor("2027-09-03", (6, "month"), sign=-1) == "2027-03-03"
    tags = {
        "tag-1": {
            "text": "September 3, 2027",
            "type": "date",
            "char_start": 0,
            "char_end": 17,
        },
        "tag-2": {
            "text": "six months",
            "type": "duration",
            "char_start": 20,
            "char_end": 30,
        },
    }
    assert (
        computed_maturity_date(["tag-1", "tag-2"], tags, "2027-03-03") == "2027-03-03"
    )
    assert (
        computed_maturity_date(["tag-1", "tag-2"], tags, "2028-03-03") == "2028-03-03"
    )
    assert computed_maturity_date(["tag-1", "tag-2"], tags, "2027-04-03") is None


def test_dates_facts_publish_columns_from_current_closing_and_maturity() -> None:
    """dates[] replaces the single-value slots; prior and projected dates stay out of the columns."""
    from cdt.extractor.normalize.dates import (
        select_date_payload,
        standardized_dates_payloads,
    )

    obj = {
        "name": ["tag-1"],
        "dates": [
            {
                "kind": "announcement",
                "evidence": ["tag-2"],
                "normalized_date": "2026-03-05",
            },
            {
                "kind": "closing",
                "expected": True,
                "evidence": ["tag-3"],
                "normalized_date": "2026-03-12",
            },
            {
                "kind": "maturity",
                "evidence": ["tag-4"],
                "normalized_date": "2026-06-28",
                "prior": True,
            },
            {
                "kind": "maturity",
                "evidence": ["tag-5"],
                "normalized_date": "2031-06-23",
            },
        ],
    }
    payloads = standardized_dates_payloads(
        obj, _dates_tag_details(), name_text="5.000% Senior Notes due 2031"
    )
    by_kind = {(p["kind"], p["prior"]): p for p in payloads}
    assert by_kind[("announcement", False)]["normalized_date"] == "2026-03-05"
    expected = [p for p in payloads if p["kind"] == "closing" and p["expected"]]
    assert expected and expected[0]["normalized_date"] == "2026-03-12"
    assert by_kind[("maturity", True)]["normalized_date"] == "2026-06-28"
    assert by_kind[("maturity", False)]["normalized_date"] == "2031-06-23"
    assert by_kind[("maturity", False)]["precision"] == "day"
    # No closing fact: the announced instrument publishes no start date.
    assert select_date_payload(payloads, "closing")["normalized_date"] is None
    assert select_date_payload(payloads, "maturity")["normalized_date"] == "2031-06-23"


def test_dates_facts_precision() -> None:
    """Month and year precision are read off the text."""
    from cdt.extractor.normalize.dates import standardized_dates_payloads

    tags = _dates_tag_details()
    month = standardized_dates_payloads(
        {
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-6"],
                    "normalized_date": "2056-03-31",
                }
            ]
        },
        tags,
        name_text="Class A-2 Notes",
    )
    assert (
        month[0]["normalized_date"] == "2056-03-31" and month[0]["precision"] == "month"
    )
    year = standardized_dates_payloads(
        {"name": ["tag-1"]}, tags, name_text="5.000% Senior Notes due 2031"
    )
    assert year[0]["kind"] == "maturity" and year[0]["normalized_date"] == "2031-12-31"
    assert year[0]["precision"] == "year" and year[0]["derived_from"] == "name"


def test_prior_amounts_never_supply_the_principal() -> None:
    """A `prior: true` commitment is history; the current figure supplies the principal."""
    from cdt.extractor.normalize.amounts import select_principal_amount

    payloads = [
        {"kind": "commitment", "normalized_amount": "25000000", "prior": True},
        {"kind": "commitment", "normalized_amount": "50000000", "prior": False},
    ]
    assert select_principal_amount(payloads)["normalized_amount"] == "50000000"
    assert select_principal_amount(payloads[:1]) == {}


def test_status_is_derived_from_event_date_facts() -> None:
    """Stage 2: the newest completed event decides status; expected events decide nothing."""
    from cdt.extractor.normalize.dates import (
        derived_status_payload,
        expected_retirement_in_payloads,
    )

    facts = [
        {
            "kind": "closing",
            "normalized_date": "2023-10-11",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
        {
            "kind": "termination",
            "normalized_date": "2026-06-02",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
    ]
    status = derived_status_payload(facts)
    assert (
        status["status"] == "terminated"
        and status["status_date"]["normalized_date"] == "2026-06-02"
    )
    announced = derived_status_payload(
        [
            {
                "kind": "closing",
                "normalized_date": "2026-07-06",
                "spans": [],
                "derived_from": "stated",
                "prior": False,
                "expected": True,
            }
        ]
    )
    assert announced["status"] == "announced"
    pending = [
        {
            "kind": "retirement",
            "normalized_date": None,
            "spans": [],
            "derived_from": None,
            "prior": False,
            "expected": True,
        }
    ]
    assert derived_status_payload(pending)["status"] is None
    assert expected_retirement_in_payloads(pending) is True
    undated = [
        {
            "kind": "closing",
            "normalized_date": None,
            "spans": [],
            "derived_from": None,
            "prior": False,
            "expected": False,
        },
        {
            "kind": "retirement",
            "normalized_date": None,
            "spans": [],
            "derived_from": None,
            "prior": False,
            "expected": False,
        },
    ]
    assert derived_status_payload(undated)["status"] == "repaid"
    assert (
        derived_status_payload(
            [
                {
                    "kind": "maturity",
                    "normalized_date": "2031-12-31",
                    "spans": [],
                    "derived_from": "name",
                    "prior": False,
                    "expected": False,
                }
            ]
        )["status"]
        is None
    )


def test_parties_list_derives_lender_disclosure() -> None:
    """Stage 2: one parties list, and disclosure distinguishes its three states."""
    from cdt.extractor.normalize.parties import party_payloads_and_disclosure

    tags = {
        "tag-1": {
            "text": "JPMorgan Chase Bank, N.A.",
            "type": "organization",
            "char_start": 0,
            "char_end": 25,
        },
        "tag-2": {
            "text": "the other lenders party thereto",
            "type": "organization",
            "char_start": 30,
            "char_end": 61,
        },
        "tag-3": {
            "text": "Wells Fargo Bank",
            "type": "organization",
            "char_start": 70,
            "char_end": 86,
        },
        "tag-4": {
            "text": "The Bank of New York Mellon",
            "type": "organization",
            "char_start": 90,
            "char_end": 117,
        },
    }
    parties, disclosure = party_payloads_and_disclosure(
        {
            "parties": [
                {"tag_ids": ["tag-1"], "role": "lender"},
                {"tag_ids": ["tag-2"], "role": "lender", "kind": "collective"},
            ]
        },
        tags,
    )
    assert [(p["role"], p["kind"]) for p in parties] == [
        ("lender", "named"),
        ("lender", "collective"),
    ]
    assert disclosure == "collective_present"
    _, every_lender_named = party_payloads_and_disclosure(
        {
            "parties": [
                {"tag_ids": ["tag-1"], "role": "lender"},
                {"tag_ids": ["tag-3"], "role": "lender", "kind": "named"},
            ]
        },
        tags,
    )
    assert every_lender_named == "complete"
    trustee_only, no_lender = party_payloads_and_disclosure(
        {"parties": [{"tag_ids": ["tag-4"], "role": "trustee"}]}, tags
    )
    assert trustee_only[0]["role"] == "trustee"
    # A trustee-only indenture names nobody who holds the debt. Under the
    # boolean this replaced, that read the same as a collective phrase.
    assert no_lender == "none_named"


def test_entry_without_parties_names_no_lender() -> None:
    """An entry omitting `parties` discloses no lender, rather than every one.

    The prompt tells the model to omit a property the document says nothing
    about, so `{name, instrument_type, amounts}` is an ordinary response.
    """
    from cdt.extractor.normalize.parties import party_payloads_and_disclosure

    parties, disclosure = party_payloads_and_disclosure(
        {"name": ["tag-i-1"], "instrument_type": "revolving_credit", "amounts": []},
        {},
    )
    assert parties == []
    assert disclosure == "none_named"


def test_post_filing_closing_is_expected_and_agreement_supplies_start() -> None:
    """Pilot fixes: a closing dated after the filing is planned; a lone agreement date is the start."""
    from cdt.extractor.normalize.dates import (
        derived_status_payload,
        mark_post_filing_events_expected,
        normalized_date_from_text,
        select_date_payload,
    )

    facts = [
        {
            "kind": "closing",
            "normalized_date": "2022-04-06",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
        {
            "kind": "announcement",
            "normalized_date": "2022-03-25",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
    ]
    mark_post_filing_events_expected(facts, "2022-03-25")
    assert facts[0]["expected"] is True and facts[1]["expected"] is False
    assert select_date_payload(facts, "closing")["normalized_date"] is None
    assert derived_status_payload(facts)["status"] == "announced"
    agreement_only = [
        {
            "kind": "agreement",
            "normalized_date": "2015-02-27",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        }
    ]
    assert (
        select_date_payload(agreement_only, "agreement")["normalized_date"]
        == "2015-02-27"
    )
    status = derived_status_payload(agreement_only)
    assert (
        status["status"] == "entered_into"
        and status["status_date"]["normalized_date"] == "2015-02-27"
    )
    assert normalized_date_from_text("March 5 , 2026") == "2026-03-05"


def test_table_cells_publish_coupon_and_document_currency() -> None:
    """FHLB schedules: a bare `4.125` under COUPON PCT is the rate; `($)` in the header is the currency."""
    from cdt.extractor.normalize.amounts import (
        currency_candidates_from_text,
        standardized_amount_payload,
        standardized_interest_rate_payload,
    )

    tags = {
        "tag-51": {
            "text": "4.125",
            "type": "interest_rate",
            "char_start": 0,
            "char_end": 5,
        },
        "tag-52": {
            "text": "35,000,000",
            "type": "amount",
            "char_start": 10,
            "char_end": 20,
        },
        "tag-53": {
            "text": "4.125% per annum",
            "type": "interest_rate",
            "char_start": 30,
            "char_end": 46,
        },
    }
    rate = standardized_interest_rate_payload(
        {"kind": "fixed", "rate_pct": "4.125", "evidence": ["tag-51"]},
        tags,
        name_text=None,
    )
    assert rate["rate_pct"] == "4.125" and rate["derived_from"] == "stated"
    # The published rate is canonical, not the model's spelling: verification is
    # numeric, so `5`, `5.00` and `5.000` all verified and all persisted
    # verbatim, splitting one rate across three distinct published strings.
    for spelling in ("4.1250", "4.12500"):
        assert (
            standardized_interest_rate_payload(
                {"kind": "fixed", "rate_pct": spelling, "evidence": ["tag-51"]},
                tags,
                name_text=None,
            )["rate_pct"]
            == "4.125"
        )
    assert (
        standardized_interest_rate_payload(
            {"kind": "fixed", "rate_pct": "4.125", "evidence": ["tag-53"]},
            tags,
            name_text=None,
        )["rate_pct"]
        == "4.125"
    )
    doc = frozenset(currency_candidates_from_text("BANK PAR ($)\n35,000,000"))
    assert doc == {"USD"}
    usd = standardized_amount_payload(
        {
            "kind": "principal",
            "evidence": ["tag-52"],
            "normalized_amount": "35000000",
            "currency": "USD",
        },
        tags,
        document_currencies=doc,
    )
    assert usd["currency"] == "USD"
    # No document-level evidence, or two currencies in the document: the model's code is still rejected.
    assert (
        standardized_amount_payload(
            {
                "kind": "principal",
                "evidence": ["tag-52"],
                "normalized_amount": "35000000",
                "currency": "USD",
            },
            tags,
            document_currencies=frozenset(),
        )["currency"]
        is None
    )
    assert (
        standardized_amount_payload(
            {
                "kind": "principal",
                "evidence": ["tag-52"],
                "normalized_amount": "35000000",
                "currency": "USD",
            },
            tags,
            document_currencies=frozenset({"USD", "CAD"}),
        )["currency"]
        is None
    )


def test_fractional_coupons_publish_as_decimal_rates() -> None:
    """`6 1/2%` and `5 7/8% Senior Notes due 2026` are 6.5 and 5.875, not missing rates."""
    from cdt.extractor.normalize.amounts import (
        rate_tokens,
        standardized_interest_rate_payload,
    )

    assert rate_tokens("6 1/2%") == ["6.5"]
    assert rate_tokens("5 7/8 % senior unsecured notes due 2030") == ["5.875"]
    assert rate_tokens("4.125% per annum") == ["4.125"]
    tags = {
        "tag-1": {
            "text": "5 7/8% Senior Notes due 2026",
            "type": "debt_instrument",
            "char_start": 0,
            "char_end": 28,
        }
    }
    payload = standardized_interest_rate_payload(
        {"kind": "fixed", "rate_pct": "5.875", "evidence": ["tag-1"]},
        tags,
        name_text="5 7/8% Senior Notes due 2026",
    )
    assert payload["rate_pct"] == "5.875" and payload["derived_from"] == "name"


def test_published_mention_columns_are_pinned() -> None:
    """The mention schema is a contract; a rename must fail here first."""
    assert DEBT_INSTRUMENT_MENTION_COLUMNS == [
        "debt_instrument_mention_id",
        "item_id",
        "accession_number",
        "cik",
        "company_name",
        "date",
        "raw_id",
        "name",
        "instrument_type",
        "start_date",
        "maturity_date",
        "commitment_termination_date",
        "principal_amount",
        "principal_currency",
        "principal_amount_kind",
        "status",
        "status_date",
        "interest_rate_kind",
        "interest_rate_pct",
        "amendment_of",
        "retired_by_json",
        "split_of",
        "parties_json",
        "name_json",
        "start_date_json",
        "maturity_date_json",
        "commitment_termination_date_json",
        "amounts_json",
        "status_json",
        "interest_rate_json",
        "dates_json",
        "lender_disclosure",
        "synthesized_by",
        "synthesized_from_mention_id",
    ]


def test_name_derived_principal_is_synthesized_when_no_amount_supplies_one() -> None:
    """#129's fallback was unreachable: its payload had no model value to agree with."""
    assert name_derived_principal_payload("$183.36 million term loan") == {
        "spans": [],
        "normalized_amount": "183360000",
        "currency": "USD",
        "derived_from": "name",
        "kind": "principal",
        "as_of_date": None,
        "prior": False,
    }
    assert (
        name_derived_principal_payload("C$300 million notes due 2033")["currency"]
        == "CAD"
    )
    # A name with no embedded principal synthesizes nothing.
    assert name_derived_principal_payload("Revolving Credit Facility") is None


def test_a_repayment_figure_does_not_suppress_the_name_derived_principal() -> None:
    """The old gate asked "any amount at all", so an unrelated figure hid the name."""
    from cdt.extractor.normalize.amounts import standardized_amounts_payloads

    tags = {
        "tag-i-1": {
            "type": "debt_instrument",
            "text": "$183.36 million term loan",
            "char_start": 0,
            "char_end": 25,
        }
    }
    payloads = standardized_amounts_payloads(
        {
            "name": ["tag-i-1"],
            "amounts": [
                {"kind": "repayment", "evidence": [], "normalized_amount": None}
            ],
        },
        tags,
        name_text="$183.36 million term loan",
    )
    principal = [p for p in payloads if p["kind"] == "principal"]
    assert principal and principal[0]["normalized_amount"] == "183360000"


def test_out_of_range_date_arithmetic_returns_none_instead_of_raising() -> None:
    """These raised out of postprocess, past the driver, killing the whole run.

    `extract_pending_items` catches only `InfrastructureError`, so the exception
    unwound past the failure registry, the mentions write and the audit write.
    """
    assert date_plus_tenor("9999-01-01", (999, "year")) is None
    assert date_plus_tenor("9999-12-31", (999, "day")) is None
    assert normalized_month_year_from_text("notes due January 0000") is None
    # The ordinary cases still work.
    assert date_plus_tenor("2026-05-15", (364, "day")) == "2027-05-14"
    assert normalized_month_year_from_text("matures in March 2056") == "2056-03-31"


def test_canonical_instrument_name_keeps_the_obligation_over_the_agreement() -> None:
    """`ner.md` rule 11: the descriptive phrase must survive the agreement title.

    Longest-span selection did the opposite on 9 of the 476 multi-span names in
    the 2026-09 window, and the published name feeds the matcher's fingerprint.
    """

    def tags(*texts: str) -> dict[str, dict[str, object]]:
        return {
            f"tag-{index}": {
                "type": "debt_instrument",
                "text": text,
                "char_start": 0,
                "char_end": len(text),
            }
            for index, text in enumerate(texts)
        }

    pair = tags("Second Amended and Restated Credit Agreement", "term loan B facility")
    assert canonical_instrument_name(list(pair), pair) == "term loan B facility"

    dip = tags("Super-Priority Senior Secured Priming Credit Agreement", "DIP Facility")
    assert canonical_instrument_name(list(dip), dip) == "DIP Facility"

    # A non-agreement span that names no obligation is not an improvement:
    # preferring it published `Local Currency Addendums` over `Credit Agreement
    # (2025 364-Day Facility)` and `RFA` over `receivables financing agreement`.
    vacuous = tags(
        "Credit Agreement (2025 364-Day Facility)", "Local Currency Addendums"
    )
    assert (
        canonical_instrument_name(list(vacuous), vacuous)
        == "Credit Agreement (2025 364-Day Facility)"
    )
    abbreviation = tags("receivables financing agreement", "RFA")
    assert (
        canonical_instrument_name(list(abbreviation), abbreviation)
        == "receivables financing agreement"
    )

    # An instrument the filing only ever names by its agreement keeps that name.
    only_agreement = tags("Credit Agreement", "Amended Credit Agreement")
    assert (
        canonical_instrument_name(list(only_agreement), only_agreement)
        == "Amended Credit Agreement"
    )
    # With no agreement title in play, the longest span still wins.
    notes = tags("the Notes", "5.875% Senior Notes due 2034")
    assert (
        canonical_instrument_name(list(notes), notes) == "5.875% Senior Notes due 2034"
    )


def test_computed_sum_needs_two_addends_even_when_no_span_matches() -> None:
    """The two guards masked each other: either alone rejected the only case.

    The existing test cites one span whose value equals the model's, so both
    "at least two addends" and "no single span equals the value" reject it. This
    case isolates the first: two spans, neither equal to the model's figure, but
    only one of them parseable.
    """
    from cdt.extractor.normalize.amounts import computed_sum_amount

    tags = {
        "tag-a-1": {
            "type": "amount",
            "text": "$200 million",
            "char_start": 0,
            "char_end": 12,
        },
        "tag-a-2": {
            "type": "amount",
            "text": "an undisclosed amount",
            "char_start": 13,
            "char_end": 34,
        },
        "tag-a-3": {
            "type": "amount",
            "text": "$50 million",
            "char_start": 35,
            "char_end": 46,
        },
    }
    # One parseable addend is not a sum, however many spans are cited.
    assert computed_sum_amount(["tag-a-1", "tag-a-2"], tags, "250000000") is None
    # Two parseable addends that do sum to the model's figure are accepted.
    assert computed_sum_amount(["tag-a-1", "tag-a-3"], tags, "250000000") == "250000000"
    # And the second guard, isolated: two addends, but one already equals the
    # model's value, so this is agreement rather than arithmetic.
    equal_tags = dict(tags)
    equal_tags["tag-a-4"] = {
        "type": "amount",
        "text": "$250 million",
        "char_start": 50,
        "char_end": 62,
    }
    assert computed_sum_amount(["tag-a-1", "tag-a-4"], equal_tags, "250000000") is None


def test_abbreviated_magnitudes_parse_to_full_amounts() -> None:
    """#182: `mil.`, `mm`, `bn` and `trillion` were unknown to the multiplier table.

    On a cited span the wrong parse merely published null, because `amounts_agree`
    rejected the mismatch. On the name-derived path (#129) there is no model value
    to disagree with, so `Citibank $382.5 mil. Revolving Credit Facility` published
    a principal of 382.5 — six orders of magnitude out.
    """
    assert (
        normalized_amount_from_name("Citibank $382.5 mil. Revolving Credit Facility")
        == "382500000"
    )
    assert normalized_amount_from_name("Syndicated $850.0 mil. Facility") == "850000000"
    assert normalized_amount_from_name("$500mm notes") == "500000000"
    assert normalized_amount_from_name("$1.2 bn facility") == "1200000000"
    # `trillion` matched the name pattern's scale group but multiplied nowhere.
    assert normalized_amount_from_name("$1.5 trillion facility") == "1500000000000"
    # The spelled-out forms and the no-magnitude case are unchanged.
    assert normalized_amount_from_name("$183.36 million term loan") == "183360000"
    assert normalized_amount_from_name("C$300 million notes due 2033") == "300000000"
    assert normalized_amount_from_text("$472,934,000") == "472934000"
    # A magnitude abbreviation cannot match inside a longer word.
    assert normalized_amount_from_text("$5 millions") == "5000000"


def test_magnitude_in_amount_text_is_the_magnitude_the_parser_applies() -> None:
    """The helper and the parser must not drift about what a magnitude is (#213).

    `scaled_amount_from_sibling` asks two questions the parser answers only
    implicitly, so the loop was factored out rather than copied. This pins the
    invariant that makes that safe: for every magnitude word in the table, the
    helper reports exactly the factor the parser multiplied by.
    """
    from cdt.extractor.normalize.amounts import (
        magnitude_in_amount_text,
        normalized_amount_from_text,
    )
    from cdt.extractor.schema import AMOUNT_MULTIPLIERS

    for word, factor in AMOUNT_MULTIPLIERS.items():
        assert magnitude_in_amount_text(f"$1 {word}") == factor
        assert normalized_amount_from_text(f"$1 {word}") == str(factor)
    # A bare figure carries no magnitude of its own, which is the condition the
    # rescue's fourth guard tests.
    assert magnitude_in_amount_text("$400.0") is None
    assert magnitude_in_amount_text("1,050,000") is None
    assert magnitude_in_amount_text(None) is None
    # `(?<![a-z])` on the left, so a magnitude cannot match inside a word.
    assert magnitude_in_amount_text("$500mm") == 1_000_000
    assert magnitude_in_amount_text("$7 per bnillion") is None


def _shared_magnitude_tags() -> dict[str, dict[str, object]]:
    """Crescent Capital BDC's two spans, at their real offsets (#213).

    `000119312526241887-1-01`: "increased the facility size from $400.0 to
    $500.0 million".
    """
    return {
        "tag-15": {
            "type": "amount",
            "text": "$400.0",
            "char_start": 654,
            "char_end": 660,
        },
        "tag-16": {
            "type": "amount",
            "text": "$500.0 million",
            "char_start": 664,
            "char_end": 678,
        },
    }


def test_scaled_amount_from_sibling_refuses_everything_but_the_exact_product() -> None:
    """Each of the rescue's seven refusals, isolated (#213).

    Mirrors `computed_sum_amount`'s guards one for one, and the two are
    mutually exclusive: a sum needs `MINIMUM_COMPUTED_SUM_SPANS` parsed spans
    and this fires on one.
    """
    from cdt.extractor.normalize.amounts import scaled_amount_from_sibling

    tags = _shared_magnitude_tags()
    own, sibling = ["tag-15"], ["tag-16"]

    # The case the issue was filed for: $400.0 scaled by the sibling's shared
    # `million` is exactly the 400000000 the model returned.
    assert scaled_amount_from_sibling(own, sibling, tags, "400000000") == "400000000"

    # 1. A non-string model amount is not a value to confirm.
    assert scaled_amount_from_sibling(own, sibling, tags, 400000000) is None
    assert scaled_amount_from_sibling(own, sibling, tags, None) is None

    # 2. A model amount that is not a number cannot be compared to a product.
    assert (
        scaled_amount_from_sibling(own, sibling, tags, "four hundred million") is None
    )

    # 3. A rate is not a principal, whether it is this fact's span or a
    #    sibling's. Both halves have to bite: 0.50 x 1,000,000 is 500000, and
    #    a rate span would serve as a borrowed magnitude just as well.
    rate_tags = dict(tags) | {
        "tag-rate": {
            "type": "amount",
            "text": "0.50%",
            "char_start": 700,
            "char_end": 705,
        }
    }
    assert (
        scaled_amount_from_sibling(["tag-rate"], sibling, rate_tags, "500000") is None
    )
    assert (
        scaled_amount_from_sibling(own, ["tag-16", "tag-rate"], rate_tags, "400000000")
        is None
    )

    # 4. A span already carrying a magnitude is never rescaled. `$500.0
    #    million` means what it says; multiplying it again by the sibling's
    #    `million` would publish a figure six orders of magnitude out.
    assert scaled_amount_from_sibling(sibling, own, tags, "500000000") is None
    both_scaled = dict(tags) | {
        "tag-17": {
            "type": "amount",
            "text": "$400.0 million",
            "char_start": 654,
            "char_end": 668,
        }
    }
    assert (
        scaled_amount_from_sibling(["tag-17"], sibling, both_scaled, "400000000000000")
        is None
    )

    # 5. Exactly one distinct magnitude among the siblings, or refuse. None at
    #    all is #214's untagged `($ in thousands)` header: there is nothing
    #    cited to borrow.
    bare_tags = dict(tags) | {
        "tag-bare": {
            "type": "amount",
            "text": "$500.0",
            "char_start": 664,
            "char_end": 670,
        }
    }
    assert scaled_amount_from_sibling(own, ["tag-bare"], bare_tags, "400000000") is None
    #    And two disagreeing magnitudes: guessing between `million` and
    #    `billion` is a three-orders-of-magnitude error.
    two_tags = dict(tags) | {
        "tag-18": {
            "type": "amount",
            "text": "$2.0 billion",
            "char_start": 700,
            "char_end": 712,
        }
    }
    assert (
        scaled_amount_from_sibling(own, ["tag-16", "tag-18"], two_tags, "400000000")
        is None
    )

    # 6. The product must equal the model's value exactly -- no rounding, no
    #    tolerance, in either direction.
    assert scaled_amount_from_sibling(own, sibling, tags, "400000001") is None
    assert scaled_amount_from_sibling(own, sibling, tags, "399999999") is None
    assert scaled_amount_from_sibling(own, sibling, tags, "400000000.5") is None

    # 7. The return is the model's own value re-normalized, so the rescue only
    #    ever confirms a figure and never originates one.
    assert scaled_amount_from_sibling(own, sibling, tags, "400,000,000") == "400000000"


def test_a_shared_magnitude_word_publishes_the_prior_commitment() -> None:
    """A magnitude written once for two figures no longer drops one (#213).

    Crescent Capital BDC's amendment says the facility size went "from $400.0
    to $500.0 million". NER tags the two figures separately, the model reads
    the shared `million` correctly and returns 400000000 for the `prior`
    commitment -- and `amounts_agree` compared that against the parser's
    reading of `$400.0` alone, which is 400, so the correct value published as
    null with `validation_errors: []`.

    Re-derived from the stored responses on `data/genwindow-run-branch`, this
    is the one fact of 587 the rescue reaches, and it moves
    `skipped_unparsed_prior` 1 -> 0 and `minted` 15 -> 16 because
    `mint_prior_state_rows` builds a predecessor only out of `prior` facts that
    carry a value.
    """
    from cdt.extractor.normalize.amounts import standardized_amounts_payloads

    payloads = standardized_amounts_payloads(
        {
            "amounts": [
                {
                    "kind": "commitment",
                    "evidence": ["tag-15"],
                    "normalized_amount": "400000000",
                    "currency": "USD",
                    "prior": True,
                },
                {
                    "kind": "commitment",
                    "evidence": ["tag-16"],
                    "normalized_amount": "500000000",
                    "currency": "USD",
                },
            ]
        },
        _shared_magnitude_tags(),
        name_text=None,
    )
    prior, current = payloads
    assert prior["normalized_amount"] == "400000000"
    # A new marker rather than `"computed"`, which means arithmetic over
    # addends; nothing in `src/` consumes `derived_from` on the amount side.
    assert prior["derived_from"] == "scaled"
    # The currency came from this fact's own cited span and is kept as it was.
    assert prior["currency"] == "USD"
    assert prior["prior"] is True
    # The sibling that carries the magnitude is read as stated, not rescaled.
    assert current["normalized_amount"] == "500000000"
    assert current["derived_from"] == "stated"


def test_the_scale_rescue_leaves_an_untagged_unit_header_alone() -> None:
    """#214's table stays null: no fact cites the scale, so none can borrow it.

    Blue Owl Technology Income Corp (`000186945326000042-8-01`) reports its
    debt-capacity table under a bare `($ in thousands)` header. NER tags 83
    `amount` spans on that item and nothing covering the unit -- the tag
    vocabulary has no category it falls under -- so there is no cited sibling
    magnitude and this rescue cannot reach it. That is #214, deferred to
    post-Beta because it needs a NER category plus `AMOUNT_EVIDENCE_TAG_TYPES`
    plus an `instrument_ie` rule.

    Pinned as a test rather than left to the guard: re-deriving
    `data/genwindow-sol-retried` from its stored responses leaves all 18 of
    this filing's amounts null, and a change that silently started rescaling
    table cells would be out of scope and unreviewed.
    """
    from cdt.extractor.normalize.amounts import standardized_amounts_payloads

    payloads = standardized_amounts_payloads(
        {
            "amounts": [
                {
                    "kind": "commitment",
                    "evidence": ["tag-cell-1"],
                    "normalized_amount": "1050000000",
                    "currency": "USD",
                },
                {
                    "kind": "outstanding_balance",
                    "evidence": ["tag-cell-2"],
                    "normalized_amount": "435230000",
                    "currency": "USD",
                },
            ]
        },
        {
            "tag-cell-1": {
                "type": "amount",
                "text": "1,050,000",
                "char_start": 200,
                "char_end": 209,
            },
            "tag-cell-2": {
                "type": "amount",
                "text": "435,230",
                "char_start": 213,
                "char_end": 220,
            },
        },
        name_text=None,
    )
    assert [payload["normalized_amount"] for payload in payloads] == [None, None]
    assert [payload["derived_from"] for payload in payloads] == [None, None]


def test_the_scale_rescue_refuses_a_magnitude_on_any_own_span_not_just_canonical() -> (
    None
):
    """A fact citing its own magnitude is never rescaled, whichever span holds it.

    The bare-figure guard reads every span the fact cites. It used to ask only
    `canonical_amount_value`, which is the *longest parseable* span, so a
    magnitude sitting on any other cited span was invisible to it and the
    rescue scaled the fact anyway -- while the fact held `$500.0 million` in
    its own evidence. Multiplying a span that already carries `million` by a
    sibling's `million` is the six-orders-out figure the guard exists to
    refuse.

    Reachable from model output rather than hypothetical: unlike the single
    `amount` property, `validate_amounts_property` never calls
    `validate_standardized_single_value_cardinality`, so one `amounts[*]` entry
    may cite several spans with distinct parsed values and still validate. Of
    762 amount facts in the stored corpus, 4 cite two or more spans and 1
    carries a magnitude on a non-canonical span.
    """
    from cdt.extractor.normalize.amounts import (
        canonical_amount_value,
        scaled_amount_from_sibling,
    )

    tags = {
        # The longer span is the bare figure, so it wins canonical selection
        # and the magnitude-bearing span is the one the old guard could not see.
        "own-bare": {
            "type": "amount",
            "text": "aggregate principal amount of $400.0",
            "char_start": 100,
            "char_end": 136,
        },
        "own-scaled": {
            "type": "amount",
            "text": "$500.0 million",
            "char_start": 140,
            "char_end": 154,
        },
        "sibling": {
            "type": "amount",
            "text": "$2.0 million",
            "char_start": 200,
            "char_end": 212,
        },
    }
    own = ["own-bare", "own-scaled"]
    # The premise: the canonical span really is the bare one, so this case
    # reaches the guard rather than being refused earlier for some other reason.
    assert canonical_amount_value(own, tags) == "aggregate principal amount of $400.0"
    # 400 x 1,000,000 is exactly the model's value, so every other refusal --
    # the rate guard, the one-distinct-magnitude guard, the exact product --
    # passes. Only reading the magnitude off `$500.0 million` refuses it.
    assert scaled_amount_from_sibling(own, ["sibling"], tags, "400000000") is None

    # The same guard covers what filtering the siblings against the fact's own
    # span texts used to catch: a sibling citing the very span that carries
    # this fact's magnitude. The filter was redundant once the guard reads
    # every own span, so it is gone and this pins the behaviour it provided.
    assert scaled_amount_from_sibling(own, ["own-scaled"], tags, "400000000") is None

    # A fact whose every cited span is a bare figure is still rescued, so the
    # widened guard did not simply switch the rescue off: the label carries no
    # magnitude, and the reading comes from the span that parses.
    labelled = {
        "own-figure": {
            "type": "amount",
            "text": "$400.0",
            "char_start": 654,
            "char_end": 660,
        },
        "own-label": {
            "type": "amount",
            "text": "Aggregate Commitment",
            "char_start": 600,
            "char_end": 620,
        },
        "sibling": {
            "type": "amount",
            "text": "$500.0 million",
            "char_start": 664,
            "char_end": 678,
        },
    }
    assert (
        scaled_amount_from_sibling(
            ["own-figure", "own-label"], ["sibling"], labelled, "400000000"
        )
        == "400000000"
    )
