"""Rules that refuse a match: incompatible names and end dates."""

from __future__ import annotations

import re
from functools import lru_cache

from cdt.matcher.normalize import (
    MONTH_TEXT_LENGTH,
    NAME_RATE_PATTERN,
    YEAR_TEXT_LENGTH,
    name_rate_tokens,
)
from cdt.matcher.schema import PreparedMention

# The name rules below are pure functions of fingerprint strings, and matching
# one CIK calls them for the same few hundred fingerprints millions of times.
_FINGERPRINT_CACHE = 1 << 14
_FINGERPRINT_PAIR_CACHE = 1 << 18


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


NAME_STOPWORDS = frozenset({"the", "of", "and", "its", "new", "existing", "certain"})
NAME_CLASS_TOKEN = re.compile(r"^(?:[a-z]|[a-z]?-?\d+[a-z]?|\d+)$")
NAME_MATURITY_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
# Above this many mentions sharing one compatible name, the name is generic for
# that issuer and the relaxed key rule is off.
NAME_CLASS_GATE = 2
# The shorter of two compatible names needs this many informative tokens, so a
# bare `note` cannot subsume every note one issuer has.
NAME_MIN_SHARED_TOKENS = 2


@lru_cache(maxsize=_FINGERPRINT_CACHE)
def name_fingerprint_tokens(fingerprint: str | None) -> frozenset[str]:
    """Return the informative tokens of one name fingerprint."""
    if not fingerprint:
        return frozenset()
    return frozenset(
        token for token in fingerprint.split() if token not in NAME_STOPWORDS
    )


@lru_cache(maxsize=_FINGERPRINT_PAIR_CACHE)
def name_fingerprints_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two name fingerprints can name one instrument.

    Fingerprints that are equal, or have the same informative tokens (they
    differ only in stopwords: `the senior notes due 2034` and `senior notes
    due 2034`), are compatible. Otherwise one fingerprint's informative tokens
    must be a subset of the other's, and:

    - the shorter has at least ``NAME_MIN_SHARED_TOKENS`` informative tokens
    - coupons present on both sides intersect
    - the differing tokens are not only class or tranche designators
      (`Tranche A Loan` is not a shortened `Tranche B Loan`)

    False when either side is missing.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = name_fingerprint_tokens(left)
    right_tokens = name_fingerprint_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
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


@lru_cache(maxsize=_FINGERPRINT_CACHE)
def name_fingerprint_is_identifying(fingerprint: str | None) -> bool:
    """Return whether a name fingerprint alone can identify one instrument.

    True when it contains a coupon rate or a maturity year.
    """
    if not fingerprint:
        return False
    if NAME_RATE_PATTERN.search(fingerprint):
        return True
    return bool(NAME_MATURITY_YEAR_PATTERN.search(fingerprint))


def name_class_sizes(mention_index: dict[str, PreparedMention]) -> dict[str, int]:
    """Count, per mention, how many mentions of its CIK share a compatible name.

    A name shared by many of one issuer's mentions is a template rather than an
    identifier, so the relaxed key rule stands down for it.

    Counted over the CIK's mentions whose name is equal or compatible,
    excluding synthesized mentions, which carry another instrument's name. A
    mention with no CIK gets 1. Compatibility is decided once per pair of
    distinct fingerprints, not per pair of mentions.
    """
    counts_by_cik: dict[str, dict[str | None, int]] = {}
    for mention in mention_index.values():
        if mention.cik is None or mention.synthesized_by is not None:
            continue
        counts = counts_by_cik.setdefault(mention.cik, {})
        fingerprint = mention.normalized_name_fingerprint
        counts[fingerprint] = counts.get(fingerprint, 0) + 1
    class_size_by_cik: dict[str, dict[str | None, int]] = {}
    sizes: dict[str, int] = {}
    for mention_id, mention in mention_index.items():
        if mention.cik is None:
            sizes[mention_id] = 1
            continue
        counts = counts_by_cik.get(mention.cik, {})
        class_sizes = class_size_by_cik.setdefault(mention.cik, {})
        fingerprint = mention.normalized_name_fingerprint
        if fingerprint not in class_sizes:
            class_sizes[fingerprint] = sum(
                count
                for other, count in counts.items()
                if other == fingerprint
                or name_fingerprints_are_compatible(fingerprint, other)
            )
        sizes[mention_id] = class_sizes[fingerprint]
    return sizes
