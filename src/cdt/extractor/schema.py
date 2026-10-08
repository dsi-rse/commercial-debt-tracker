"""The extraction vocabulary: fact kinds, shared patterns, mention columns and mention ids."""

from __future__ import annotations

import hashlib
import json
import re

INSTRUMENT_ENTITY_TAG_TYPES = {"debt_instrument"}
LENDER_TAG_TYPES = {"person", "organization"}
DEFAULT_LENDER_CLUSTER_KIND = "named"
BORROWER_PARTY_ROLE = "borrower"
COLLECTIVE_LENDER_KIND = "collective"
INSTRUMENT_SINGLE_VALUE_PROPERTIES = {"name": {"debt_instrument"}}
# One mention can state several money facts about one instrument — a $2.5B
# commitment and a $270.5M outstanding balance — so amounts are a kind-typed
# list. The matcher keys only on commitment/principal; the other kinds are
# observations.
AMOUNT_KINDS = {
    "commitment",
    "principal",
    "outstanding_balance",
    "draw",
    "repayment",
    "proceeds",
}
PRINCIPAL_AMOUNT_KINDS = ("commitment", "principal")
# Kind-typed date facts: each stated date is recorded once and the
# post-processor chooses the published columns (`DATE_COLUMN_KINDS`).
DATE_KINDS = {
    "agreement",  # the instrument's own `dated as of` date
    "announcement",  # pricing, launch, or commitment-letter date
    "closing",  # closing, issuance, funding, or effective date: the start
    "maturity",  # when the borrowed money must be repaid
    "commitment_termination",  # when the lender's obligation to lend ends
    # Events are dated facts too, and the mention's status is derived.
    "amendment",  # terms modified; the amendment's effective or signing date
    "repayment",  # a payment that leaves the obligation outstanding
    "retirement",  # repaid in full, redeemed in whole, defeased, discharged
    "termination",  # the agreement or facility ended before its scheduled end
    "exchange",  # satisfied by delivering other securities or equity
    "default",  # default, event of default, or acceleration
}
EVENT_DATE_KINDS = {
    "announcement",
    "closing",
    "amendment",
    "repayment",
    "retirement",
    "termination",
    "exchange",
    "default",
}
TERMINAL_DATE_KINDS = {"retirement", "termination", "exchange", "default"}
# The kinds an instrument has at most one current value of, which
# `instrument_ie.md` advertises as "(validated)". `closing` is an event kind but
# still singular: two current closings describe two instruments, exactly as two
# maturities do, so this set is not derivable from EVENT_DATE_KINDS.
SINGLE_CURRENT_DATE_KINDS = {
    "agreement",
    "closing",
    "maturity",
    "commitment_termination",
}
# Kinds that are only meaningful with a value: a stated maturity without a
# date is nothing, whereas `the notes were redeemed` with no date is still an
# event the filing states.
DATE_KINDS_REQUIRING_EVIDENCE = {"maturity", "commitment_termination", "agreement"}
# The mention-level `status` column derived from the newest completed event.
STATUS_FOR_DATE_KIND = {
    "announcement": "announced",
    "closing": "entered_into",
    "amendment": "amended",
    "retirement": "repaid",
    "termination": "terminated",
    "exchange": "exchanged",
    "default": "defaulted",
}
# Ties among undated events resolve by how much the event says about the
# obligation's life: its end beats a change beats its start.
EVENT_KIND_PRECEDENCE = {
    "default": 6,
    "exchange": 5,
    "retirement": 4,
    "termination": 4,
    "amendment": 3,
    "closing": 2,
    "announcement": 1,
}
DATE_KIND_EVIDENCE_TAG_TYPES = {
    "maturity": {"date", "debt_instrument", "duration"},
}
DEFAULT_DATE_EVIDENCE_TAG_TYPES = {"date"}
# One `parties` list with a role per cluster; `lender_disclosure` is derived
# from it.
PARTY_ROLES = {
    "lender",
    "agent",
    "trustee",
    "underwriter",
    "guarantor",
    "borrower",
    "other",
}
PARTY_KINDS = {"named", "collective"}
# How completely the document identifies who holds the debt: a named-only
# syndicate, a collective phrase present, or no named lender at all. See
# docs/decisions/extraction.md for why this is three-valued.
LENDER_DISCLOSURE_COMPLETE = "complete"
LENDER_DISCLOSURE_COLLECTIVE_PRESENT = "collective_present"
LENDER_DISCLOSURE_NONE_NAMED = "none_named"
LENDER_DISCLOSURE_VALUES = {
    LENDER_DISCLOSURE_COMPLETE,
    LENDER_DISCLOSURE_COLLECTIVE_PRESENT,
    LENDER_DISCLOSURE_NONE_NAMED,
}
# Precedence for rolling several mentions of one instrument into one answer.
# `collective_present` wins outright: one filing showing `the other lenders
# party thereto` means holders are hidden however many other filings name some.
# `complete` beats `none_named`, so a passing reference cannot erase a full
# syndicate list.
LENDER_DISCLOSURE_PRECEDENCE = {
    LENDER_DISCLOSURE_NONE_NAMED: 0,
    LENDER_DISCLOSURE_COMPLETE: 1,
    LENDER_DISCLOSURE_COLLECTIVE_PRESENT: 2,
}
# Published flat columns and the fact kind each one reads.
DATE_COLUMN_KINDS = {
    "start_date": "closing",
    "maturity_date": "maturity",
    "commitment_termination_date": "commitment_termination",
}
DATE_PRECISIONS = ("day", "month", "year")
AMOUNT_EVIDENCE_TAG_TYPES = {"amount", "debt_instrument"}
INTEREST_RATE_KINDS = {"fixed", "floating"}
INTEREST_RATE_EVIDENCE_TAG_TYPES = {"interest_rate", "debt_instrument"}
# `6.5 percent senior notes` spells the marker out; it is still a rate, not an amount.
RATE_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
# The four instrument categories the site facets on; anything else stays null
# rather than stretching a bucket.
INSTRUMENT_TYPES = {
    "term_loan",
    "revolving_credit",
    "credit_line",
    "note_bond",
}
MATURITY_EVIDENCE_TAG_TYPES = {"debt_instrument"}
NAME_EMBEDDED_AMOUNT_TAG_TYPES = {"debt_instrument"}
INSTRUMENT_RELATION_TYPES = {"amendment_of", "retired_by", "split_of"}
NUMERIC_STRING_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
# One `due` can carry a list of maturities: `due 2028 and 2030`,
# `due October 1, 2028 and 2030`, `due October 1, 2028 and October 1, 2030`. Each
# is two maturities, not one.
MATURITY_COORDINATED_YEARS = (
    r"(?:\s*(?:,|/|&|and(?:/or)?|or)\s*(?:[A-Za-z]+\s+\d{1,2},?\s+)?\d{4})*"
)
# Like MATURITY_COORDINATED_YEARS, but each further year may carry a bare month
# with no day, so `due October 1, 2028 and April 2030` and `due April 2033 and
# June 2035` both read as two maturities.
MATURITY_MONTH_YEAR_COORDINATION = (
    r"(?:\s*(?:,|/|&|and(?:/or)?|or)\s*(?:[A-Za-z]+\s+(?:\d{1,2},?\s+)?)?\d{4})*"
)
MATURITY_FULL_DATE_PATTERN = re.compile(
    r"\bdue\s+(?:on\s+)?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    rf"(?P<more>{MATURITY_MONTH_YEAR_COORDINATION})",
    re.IGNORECASE,
)
MATURITY_YEAR_PATTERN = re.compile(
    rf"\bdue\s+(?:in\s+)?(?P<years>\d{{4}}{MATURITY_COORDINATED_YEARS})\b",
    re.IGNORECASE,
)
# A facility tenor such as `five-year` or `364-day`. Only a duration
# span stating exactly one tenor anchors computed-maturity arithmetic.
TENOR_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "eighteen": 18,
}
TENOR_PATTERN = re.compile(
    r"\b(?P<num>\d{1,3}|"
    + "|".join(TENOR_WORD_NUMBERS)
    + r")[-\s](?P<unit>year|month|day)s?\b",
    re.IGNORECASE,
)
# `due April 2033` states a month-resolution maturity; it normalizes to the
# month's last day.
MATURITY_MONTH_YEAR_PATTERN = re.compile(
    r"\bdue\s+(?:in\s+)?(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})"
    rf"(?P<more>{MATURITY_MONTH_YEAR_COORDINATION})",
    re.IGNORECASE,
)
FOUR_DIGIT_YEAR_PATTERN = re.compile(r"\d{4}")
YEAR_ONLY_MATURITY_SUFFIX = "-12-31"
# A rate marker counts only where it sits on a number, so the value the parser
# would read is the rate itself rather than a percentage of something else.
AMOUNT_VALUE_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
# `bps` as well as the spelled-out marker: the abbreviation is what filings
# actually write, and `amounts_agree` cannot catch a basis-point margin the
# model reports as an amount, since the two figures agree.
RATE_SUFFIX_PATTERN = re.compile(
    r"\s*(?:%|percent\b|basis\s+points?\b|bps?\b)", re.IGNORECASE
)
# A principal stated inside an instrument name: `$183.36 million term loan`,
# `C$300 million notes due 2033`. The currency marker is required, so a coupon
# rate or a maturity year in the same name cannot be read as the principal.
# Magnitude words, spelled out and abbreviated: instrument names use the
# abbreviations (`Citibank $382.5 mil. Revolving Credit Facility`), and the
# name-derived principal has no model value to cross-check against.
AMOUNT_MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "mil": 1_000_000,
    "mils": 1_000_000,
    "mm": 1_000_000,
    "mn": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "bil": 1_000_000_000,
    "bln": 1_000_000_000,
    "bn": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}
# Single-letter magnitudes attached to the figure: `$250M`, `$1.5B`, `$500K`,
# `£250m`. Upper case after any figure; lower case only after a currency-marked
# one, since a bare `250m` is as likely metres or months.
LETTER_AMOUNT_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,
    "b": 1_000_000_000,
}
LETTER_AMOUNT_MAGNITUDE_PATTERN = re.compile(
    r"(?:(?<=\d)(?P<upper>MM|[KMB])"
    r"|(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?)(?P<lower>mm|[kmb]))"
    r"(?![A-Za-z])"
)
# Built from the table above so a magnitude this pattern recognizes is always one
# the parser can apply.
AMOUNT_SCALE_ALTERNATION = "|".join(sorted(AMOUNT_MULTIPLIERS, key=len, reverse=True))
NAME_EMBEDDED_AMOUNT_PATTERN = re.compile(
    r"(?P<currency>[A-Z]{0,2}\$|€|£|¥)\s?"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)"
    rf"(?:\s*(?P<scale>{AMOUNT_SCALE_ALTERNATION})\b\.?"
    r"|(?-i:(?P<letter>MM|mm|[KMBkmb]))(?![A-Za-z]))?",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# Zero-padding is optional on the way in, so a model writing `2026-7-28` is read
# as the day it means rather than dropped for its shape.
LENIENT_ISO_DATE_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$"
)
# Filing dates come in three further spellings. The four-digit year is required:
# `7/28/26` needs a century guessed, and a null beats a wrong decade.
NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})(?!\d)"
)
# The comma is optional, so `July 28 2026` reads the same as `July 28, 2026`.
MONTH_FIRST_DATE_PATTERN = re.compile(
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})(?!\d)"
)
# `28 July 2026`, as non-US issuers write it.
DAY_FIRST_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})(?!\d)"
)
# `matures in June 2016`, `is in March 2056`: a month-resolution date outside
# a `due` phrase. Read for maturities only (`normalized_month_year_from_text`),
# to the month's last day.
MONTH_YEAR_DATE_PATTERN = re.compile(
    r"(?<![A-Za-z\d])(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})(?!\d)"
)
MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}
QUALIFIED_DOLLAR_CODES = {
    "A": "AUD",
    "C": "CAD",
    "CA": "CAD",
    "HK": "HKD",
    "NZ": "NZD",
    "R": "BRL",
    "S": "SGD",
}
QUALIFIED_DOLLAR_PATTERN = re.compile(
    rf"\b({'|'.join(sorted(QUALIFIED_DOLLAR_CODES, key=len, reverse=True))})\$",
    re.IGNORECASE,
)
COMMON_CURRENCY_CODES = {
    "AED",
    "AUD",
    "BRL",
    "CAD",
    "CHF",
    "CNY",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "INR",
    "JPY",
    "KRW",
    "MXN",
    "NOK",
    "NZD",
    "SAR",
    "SEK",
    "SGD",
    "TRY",
    "USD",
    "ZAR",
}
CURRENCY_CODE_LENGTH = 3


DEBT_INSTRUMENT_MENTION_COLUMNS = [
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
    # Set only on a row the extractor synthesized rather than the model
    # returned: the rule that minted it, and the mention it was minted from.
    # Null on every model-emitted row. Neither is hashed into the mention id.
    "synthesized_by",
    "synthesized_from_mention_id",
]


def raw_id_for(index: int) -> str:
    """Return the stage-local raw instrument-mention ID."""
    return f"i-{index}"


def debt_instrument_mention_id_for(
    item_id: str,
    mention_row: dict[str, object],
) -> str:
    """Return a stable persisted debt-instrument-mention ID."""
    payload = {
        "amounts_json": normalize_json_text(mention_row.get("amounts_json")),
        "dates_json": normalize_json_text(mention_row.get("dates_json")),
        "commitment_termination_date": mention_row.get("commitment_termination_date"),
        "commitment_termination_date_json": normalize_json_text(
            mention_row.get("commitment_termination_date_json")
        ),
        "maturity_date": mention_row.get("maturity_date"),
        "maturity_date_json": normalize_json_text(
            mention_row.get("maturity_date_json")
        ),
        "instrument_type": mention_row.get("instrument_type"),
        "interest_rate_json": normalize_json_text(
            mention_row.get("interest_rate_json")
        ),
        "item_id": item_id,
        "lender_disclosure": mention_row.get("lender_disclosure"),
        "name_json": normalize_json_text(mention_row.get("name_json")),
        "name": mention_row.get("name"),
        "parties_json": normalize_json_text(mention_row.get("parties_json")),
        "principal_amount": mention_row.get("principal_amount"),
        "status_json": normalize_json_text(mention_row.get("status_json")),
        "start_date": mention_row.get("start_date"),
        "start_date_json": normalize_json_text(mention_row.get("start_date_json")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"dim::{digest}"


def normalize_json_text(value: object) -> str:
    """Normalize one JSON-encoded payload for deterministic hashing."""
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return str(value)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


# Where a normalized value came from: cited evidence spans, the instrument's own
# name, or nowhere (no value). Downstream consumers key on this — the matcher
# treats a name-synthesized YYYY-12-31 maturity as year-resolution only, and the
# site can explain a value whose evidence list is empty.
DERIVED_FROM_STATED = "stated"
DERIVED_FROM_NAME = "name"
DERIVED_FROM_COMPUTED = "computed"
# An amount read off its own cited span and scaled by a magnitude word carried
# by a *sibling* amount fact's cited span: `from $400.0 to $500.0 million`.
# Distinct from `"computed"`, which means arithmetic over addends.
DERIVED_FROM_SCALED = "scaled"
# A term carried onto a synthesized predecessor row from the amended object it
# was minted from, because the filing marked no `prior` value for that kind and
# so states it unchanged. The spans are the successor's; the marker is
# what lets a reader tell an inherited term from one the filing stated for this
# state of the instrument.
DERIVED_FROM_INHERITED = "inherited"
# A sum needs at least two addends; one parsed span is agreement, not arithmetic.
MINIMUM_COMPUTED_SUM_SPANS = 2


LENDER_PARTY_ROLE = "lender"
