"""Rules that refuse a match: incompatible names, coupons and end dates."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

import pandas as pd

from cdt.extractor.normalize.amounts import normalize_numeric_string
from cdt.matcher.normalize import (
    MONTH_TEXT_LENGTH,
    YEAR_TEXT_LENGTH,
    coerce_optional_bool,
    coerce_optional_text,
    normalize_name_fingerprint,
)
from cdt.matcher.schema import PreparedMention


def end_dates_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two normalized end dates can still describe one instrument.

    A missing side is compatible. A bare ``YYYY`` matches any date in that
    year and ``YYYY-MM`` any date in that month; full dates must agree exactly.
    """
    if not left or not right:
        return True
    if left == right:
        return True
    if left[:YEAR_TEXT_LENGTH] != right[:YEAR_TEXT_LENGTH]:
        return False
    if len(left) == YEAR_TEXT_LENGTH or len(right) == YEAR_TEXT_LENGTH:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) == MONTH_TEXT_LENGTH and longer[:MONTH_TEXT_LENGTH] == shorter


# `normalize_name_fingerprint` turns the decimal point into a token break, so a
# coupon arrives here as `4 375%` rather than `4.375%`; the separator is
# therefore optional. `(?<!\d)` keeps a maturity year out of the whole-number
# part, so `notes due 2028 5%` yields the rate and not `2028 5%`.
NAME_RATE_PATTERN = re.compile(r"(?<!\d)(\d{1,3})(?:[ .](\d{1,4}))?%")


def name_rate_tokens(fingerprint: str | None) -> frozenset[str]:
    """Return the coupon rates in one name fingerprint, as canonical numbers.

    ``4 375%`` yields the normalized numeric string for 4.375; an empty
    fingerprint yields the empty set.
    """
    if not fingerprint:
        return frozenset()
    rates: set[str] = set()
    for whole, fraction in NAME_RATE_PATTERN.findall(fingerprint):
        try:
            rates.add(normalize_numeric_string(Decimal(f"{whole}.{fraction or 0}")))
        except InvalidOperation:
            continue
    return frozenset(rates)


NAME_STOPWORDS = frozenset({"the", "of", "and", "its", "new", "existing", "certain"})


NAME_CLASS_TOKEN = re.compile(r"^(?:[a-z]|[a-z]?-?\d+[a-z]?|\d+)$")


NAME_MATURITY_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


# Above this many mentions sharing one compatible name, the name is generic for
# that issuer and the relaxed key rule is off.
NAME_CLASS_GATE = 2


# The shorter of two compatible names needs this many informative tokens, so a
# bare `note` cannot subsume every note one issuer has.
NAME_MIN_SHARED_TOKENS = 2


def name_fingerprint_tokens(fingerprint: str | None) -> frozenset[str]:
    """Return the informative tokens of one name fingerprint."""
    if not fingerprint:
        return frozenset()
    return frozenset(
        token for token in fingerprint.split() if token not in NAME_STOPWORDS
    )


def name_fingerprints_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two name fingerprints can name one instrument.

    Equal fingerprints are compatible. Otherwise one fingerprint's informative
    tokens must be a subset of the other's, and:

    - the shorter has at least ``NAME_MIN_SHARED_TOKENS`` informative tokens
    - coupons present on both sides intersect
    - the differing tokens are not only class or tranche designators
      (`Tranche A Loan` is not a shortened `Tranche B Loan`)

    Unequal fingerprints with identical informative tokens (they differ only in
    stopwords, e.g. `the senior notes due 2034` and `senior notes due 2034`)
    are not compatible. False when either side is missing.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = name_fingerprint_tokens(left)
    right_tokens = name_fingerprint_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if not (left_tokens <= right_tokens or right_tokens <= left_tokens):
        return False
    if min(len(left_tokens), len(right_tokens)) < NAME_MIN_SHARED_TOKENS:
        return False
    left_rates = name_rate_tokens(left)
    right_rates = name_rate_tokens(right)
    if left_rates and right_rates and not (left_rates & right_rates):
        return False
    return any(
        not NAME_CLASS_TOKEN.match(token) for token in left_tokens ^ right_tokens
    )


def name_fingerprint_is_identifying(fingerprint: str | None) -> bool:
    """Return whether a name fingerprint alone can identify one instrument.

    True when it contains a coupon rate or a maturity year.
    """
    if not fingerprint:
        return False
    if NAME_RATE_PATTERN.search(fingerprint):
        return True
    return bool(NAME_MATURITY_YEAR_PATTERN.search(fingerprint))


def name_class_sizes(
    mention_index: dict[str, PreparedMention],
    existing_instruments: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Count, per mention, how many mentions of its CIK share a compatible name.

    A name shared by many of one issuer's mentions is a template rather than an
    identifier, so the relaxed key rule stands down for it.

    Counted over the CIK's mentions and existing instrument rows whose name is
    equal or compatible, excluding synthesized mentions and `synthesized_only`
    rows, which carry another instrument's name. A mention with no CIK gets 1.
    """
    by_cik: dict[str, list[str | None]] = {}
    for mention in mention_index.values():
        if mention.cik is None or mention.synthesized_by is not None:
            continue
        by_cik.setdefault(mention.cik, []).append(mention.normalized_name_fingerprint)
    if existing_instruments is not None and not existing_instruments.empty:
        for row in existing_instruments.to_dict("records"):
            cik = coerce_optional_text(row.get("cik"))
            if cik is None or cik not in by_cik:
                continue
            if coerce_optional_bool(row.get("synthesized_only")):
                continue
            by_cik[cik].append(
                normalize_name_fingerprint(coerce_optional_text(row.get("name")))
            )
    sizes: dict[str, int] = {}
    for mention_id, mention in mention_index.items():
        if mention.cik is None:
            sizes[mention_id] = 1
            continue
        fingerprint = mention.normalized_name_fingerprint
        sizes[mention_id] = sum(
            1
            for other in by_cik.get(mention.cik, [])
            if other == fingerprint
            or name_fingerprints_are_compatible(fingerprint, other)
        )
    return sizes


def name_rates_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two name fingerprints can still describe one instrument."""
    left_rates = name_rate_tokens(left)
    right_rates = name_rate_tokens(right)
    if left_rates and right_rates:
        return bool(left_rates & right_rates)
    return True
