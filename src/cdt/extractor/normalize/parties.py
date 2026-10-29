"""Build the party list, lender disclosure and canonical instrument name."""

from __future__ import annotations

import re
from typing import Any

from cdt.extractor.schema import (
    COLLECTIVE_LENDER_KIND,
    DEFAULT_LENDER_CLUSTER_KIND,
    LENDER_DISCLOSURE_COLLECTIVE_PRESENT,
    LENDER_DISCLOSURE_COMPLETE,
    LENDER_DISCLOSURE_NONE_NAMED,
    LENDER_PARTY_ROLE,
    PARTY_KINDS,
    PARTY_ROLES,
)
from cdt.extractor.tags import (
    cluster_payload,
    cluster_span_texts,
    payload_tag_ids,
)

# The title of the contract, as opposed to a description of the obligation it
# creates: `Amended and Restated Credit Agreement`, `Indenture`, `Note Purchase
# Agreement`. NER tags both for one facility (ner.md rule 11), and the
# agreement title is usually the longer string.
AGREEMENT_NAME_PATTERN = re.compile(
    r"\b(?:agreement|indenture|supplemental\s+indenture)\b", re.IGNORECASE
)
# What an obligation is called, as opposed to a defined term that merely happens
# not to be an agreement title (`Local Currency Addendums`, `RFA`).
INSTRUMENT_NOUN_PATTERN = re.compile(
    r"\b(?:facility|facilities|loan|loans|note|notes|bond|bonds|debenture|"
    r"debentures|line\s+of\s+credit|revolver|commitment|commitments|"
    r"financing|borrowing|borrowings|credit)\b",
    re.IGNORECASE,
)
# The words a filing uses to refer to a party by its role rather than its name.
# Text made only of these (`Trustee`, `Notes Collateral Agent`, `we`, `the other
# lenders party thereto`) names no one; a real name keeps at least one word
# outside it (`Citibank`, `BHP`).
ROLE_ONLY_WORDS = frozenset(
    {
        # roles
        "administrator",
        "affiliate",
        "agent",
        "bank",
        "borrower",
        "buyer",
        "company",
        "dealer",
        "debtor",
        "entity",
        "guarantor",
        "holder",
        "investor",
        "issuer",
        "lender",
        "noteholder",
        "obligor",
        "partnership",
        "party",
        "person",
        "purchaser",
        "representative",
        "trustee",
        "underwriter",
        # modifiers
        "administrative",
        "co",
        "collateral",
        "credit",
        "facility",
        "increasing",
        "initial",
        "issuing",
        "lead",
        "loan",
        "notes",
        "original",
        "other",
        "parent",
        "required",
        "revolving",
        "securities",
        "specified",
        "subsidiary",
        "successor",
        # pronouns and fillers
        "and",
        "as",
        "its",
        "of",
        "our",
        "the",
        "thereto",
        "us",
        "we",
    }
)


def is_role_only_text(text: str) -> bool:
    """Return whether ``text`` refers to a party only by role, modifier or pronoun.

    Curly quotes fold to plain ones, a possessive `'s` and a `(s)` plural marker
    are dropped, and only the letter words are kept; the text is role-only when
    every word, or its singular, is in `ROLE_ONLY_WORDS`. Text with no letter
    words is not role-only.
    """
    folded = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    folded = re.sub(r"'s\b|\(s\)", "", folded)
    words = re.findall(r"[a-z]+", folded)
    return bool(words) and all(_is_role_word(word) for word in words)


def _is_role_word(word: str) -> bool:
    if word in ROLE_ONLY_WORDS:
        return True
    if word.endswith("ies"):
        return f"{word[:-3]}y" in ROLE_ONLY_WORDS
    return word.endswith("s") and word[:-1] in ROLE_ONLY_WORDS


def canonical_party_name(values: list[str]) -> str | None:
    """Return the longest span that names the party, else the longest span.

    A role-only span (`Securities Administrator`) loses to a name
    (`Citibank, N.A.`) however long it is; None for an empty cluster.
    """
    if not values:
        return None
    named = [value for value in values if not is_role_only_text(value)]
    return max(named or values, key=len)


def canonical_instrument_name(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> str | None:
    """Return the name that describes the obligation, not the contract.

    The longest span that is not an agreement title and contains an
    obligation noun (`ner.md` rule 11); the longest span of any kind when none
    qualifies; None for an empty cluster. The published name feeds the
    matcher's name fingerprint, so the individuating description is preferred
    over a generic agreement title.
    """
    values = cluster_span_texts(tag_ids, tag_details)
    if not values:
        return None
    described = [
        value
        for value in values
        if not AGREEMENT_NAME_PATTERN.search(value)
        # The alternative has to actually name an obligation, or a defined
        # term such as `RFA` would beat `receivables financing agreement`.
        and INSTRUMENT_NOUN_PATTERN.search(value)
    ]
    return max(described or values, key=len)


def party_payloads_and_disclosure(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    """Return every party cluster with its role and kind, plus lender disclosure.

    Each cluster with at least one known span becomes ``canonical_name``
    (`canonical_party_name`), ``role`` (the model's, or ``other`` when
    unknown), ``kind`` and ``spans``. ``kind`` is ``collective`` when every
    span is role-only (`is_role_only_text`), since such a cluster names no
    one; otherwise it is the model's, or ``named`` when absent or unknown. The
    borrower is kept: its identity matters when a subsidiary is the obligor
    under the parent filer's 8-K. The disclosure is computed from the lender
    clusters' kinds (`lender_disclosure_for`).
    """
    parties: list[dict[str, object]] = []
    raw_parties = obj.get("parties")
    for cluster in raw_parties if isinstance(raw_parties, list) else []:
        if not isinstance(cluster, dict):
            continue
        payload = cluster_payload(cluster.get("tag_ids"), tag_details)
        if not payload["spans"]:
            continue
        values = cluster_span_texts(payload_tag_ids(payload), tag_details)
        role = cluster.get("role")
        kind = cluster.get("kind")
        if values and all(is_role_only_text(value) for value in values):
            kind = COLLECTIVE_LENDER_KIND
        parties.append(
            {
                "canonical_name": canonical_party_name(values),
                "role": role if role in PARTY_ROLES else "other",
                "kind": kind if kind in PARTY_KINDS else DEFAULT_LENDER_CLUSTER_KIND,
                "spans": payload["spans"],
            }
        )
    return parties, lender_disclosure_for(
        [p["kind"] for p in parties if p["role"] == LENDER_PARTY_ROLE]
    )


def lender_disclosure_for(lender_kinds: list[object]) -> str:
    """Return how completely the lender clusters identify who holds the debt.

    No lender cluster at all is `none_named` — a public-market series, a
    redemption notice, a syndicate where only the agent is named. A collective
    cluster (`the other lenders party thereto`) is `collective_present`. Only
    when every lender cluster is named is the list `complete`.
    """
    if not lender_kinds:
        return LENDER_DISCLOSURE_NONE_NAMED
    if COLLECTIVE_LENDER_KIND in lender_kinds:
        return LENDER_DISCLOSURE_COLLECTIVE_PRESENT
    return LENDER_DISCLOSURE_COMPLETE
