"""Turn stored mention rows into matcher inputs: coercion, fingerprints and lender keys."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import pandas as pd

from cdt.extractor.schema import (
    LENDER_DISCLOSURE_NONE_NAMED,
    LENDER_DISCLOSURE_PRECEDENCE,
    LENDER_DISCLOSURE_VALUES,
)
from cdt.matcher.schema import PreparedMention
from cdt.storage.columns import coerce_dataset_text, json_column

GENERIC_LENDER_TERMS = frozenset(
    {
        "lender",
        "lenders",
        "purchaser",
        "purchasers",
        "holder",
        "holders",
        "investor",
        "investors",
        "buyer",
        "buyers",
        "noteholder",
        "noteholders",
        "trustee",
        "trustees",
    }
)


def _json_text(row: dict[str, object], column: str) -> str | None:
    """Return one JSON column's text, or None when it holds no parseable JSON.

    An absent column, a missing value and invalid JSON all return None, so
    "absent" stays distinguishable from an empty list; callers spell their own
    default.
    """
    text = coerce_dataset_text(row.get(column))
    if text is None:
        return None
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return None
    return text


def _json_list(row: dict[str, object], column: str) -> list[object]:
    """Return one JSON-array column's entries; absent, missing or junk reads empty."""
    payload = json_column(row, column)
    return list(payload) if isinstance(payload, list) else []


def first_non_null(
    ordered_mention_ids: list[str],
    mention_index: dict[str, PreparedMention],
    field_name: str,
) -> str | None:
    """Return the newest non-null field value across member mentions."""
    for mention_id in ordered_mention_ids:
        value = getattr(mention_index[mention_id], field_name)
        if value is not None:
            return value
    return None


def dedupe_party_clusters(payloads: list[str]) -> list[dict[str, object]]:
    """Return deduped party cluster payloads, keyed by role plus canonical name.

    The role is part of the key so one entity in two roles (an agent that is
    also a lender) keeps both rows.
    """
    deduped: dict[str, dict[str, object]] = {}
    for payload in payloads:
        for cluster in parse_cluster_list(payload):
            canonical = party_dedupe_key(cluster)
            if not canonical:
                continue
            key = f"{cluster.get('role', 'lender')}::{canonical}"
            if key not in deduped:
                deduped[key] = cluster
    return [deduped[key] for key in sorted(deduped)]


def party_dedupe_key(cluster: dict[str, object]) -> str:
    """Return the key a party cluster dedupes on: its normalized `canonical_name`."""
    return normalize_party_text(str(cluster.get("canonical_name") or ""))


def parse_cluster_list(value: str) -> list[dict[str, object]]:
    """Parse one JSON cluster list."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [cluster for cluster in payload if isinstance(cluster, dict)]


def cluster_canonical_key(cluster: dict[str, object]) -> str:
    """Return the normalized canonical key for one cluster.

    A cluster can hold a defined-term alias alongside the party it names, as in
    `Oaktree` and `Purchasers`. The specific name is the useful key, so generic
    party words lose to it even when the alias is the longer string.
    """
    spans = cluster.get("spans", [])
    if not isinstance(spans, list):
        return ""
    texts = [
        normalize_party_text(str(span.get("text", "")))
        for span in spans
        if isinstance(span, dict) and span.get("text")
    ]
    texts = [text for text in texts if text]
    if not texts:
        return ""
    specific = [text for text in texts if text not in GENERIC_LENDER_TERMS]
    return max(specific or texts, key=len)


def prepare_mention(row: dict[str, object]) -> PreparedMention:
    """Normalize one mention row for matching."""
    parties_json = _json_text(row, "parties_json") or "[]"
    return PreparedMention(
        debt_instrument_mention_id=str(row["debt_instrument_mention_id"]),
        item_id=str(row["item_id"]),
        raw_id=str(row["raw_id"]),
        accession_number=coerce_optional_text(row.get("accession_number")),
        cik=coerce_optional_text(row.get("cik")),
        company_name=coerce_optional_text(row.get("company_name")),
        date=coerce_optional_text(row.get("date")),
        name=coerce_optional_text(row.get("name")),
        instrument_type=coerce_optional_text(row.get("instrument_type")),
        start_date=coerce_optional_text(row.get("start_date")),
        maturity_date=coerce_optional_text(row.get("maturity_date")),
        maturity_is_derived=maturity_derivation(row.get("maturity_date_json"))
        in DERIVED_MATURITY_KINDS,
        commitment_termination_date=coerce_optional_text(
            row.get("commitment_termination_date")
        ),
        principal_amount=coerce_optional_text(row.get("principal_amount")),
        principal_currency=coerce_optional_text(row.get("principal_currency")),
        principal_amount_kind=coerce_optional_text(row.get("principal_amount_kind")),
        amounts_json=_json_text(row, "amounts_json") or "[]",
        interest_rate_kind=coerce_optional_text(row.get("interest_rate_kind")),
        interest_rate_pct=coerce_optional_text(row.get("interest_rate_pct")),
        status=coerce_optional_text(row.get("status")),
        amendment_of=coerce_optional_text(row.get("amendment_of")),
        retired_by=tuple(str(entry) for entry in _json_list(row, "retired_by_json")),
        split_of=coerce_optional_text(row.get("split_of")),
        parties_json=parties_json,
        lender_disclosure=coerce_lender_disclosure(row.get("lender_disclosure")),
        normalized_amount=normalize_amount(
            coerce_optional_text(row.get("principal_amount"))
        ),
        normalized_start_date=normalize_date(
            coerce_optional_text(row.get("start_date"))
        ),
        normalized_end_date=normalized_end_date_for_matching(row),
        normalized_name_fingerprint=normalize_name_fingerprint(
            coerce_optional_text(row.get("name"))
        ),
        lender_signature=lender_signature(parties_json),
        synthesized_by=coerce_optional_text(row.get("synthesized_by")),
        synthesized_from_mention_id=coerce_optional_text(
            row.get("synthesized_from_mention_id")
        ),
    )


def mention_sort_key(mention: PreparedMention) -> tuple[str, str, str, int, str]:
    """Return deterministic processing order for cluster assignment.

    Within one item a synthesized prior state sorts before the amended object
    it was minted from, so the earlier state is the one that joins the
    instrument's existing cluster.
    """
    return (
        mention.date or "",
        mention.accession_number or "",
        mention.item_id,
        0 if mention.synthesized_by is not None else 1,
        mention.debt_instrument_mention_id,
    )


def mention_recency_key(mention: PreparedMention) -> tuple[str, str, str, str]:
    """Return recency ordering for field resolution."""
    return (
        mention.date or "",
        mention.accession_number or "",
        mention.item_id,
        mention.debt_instrument_mention_id,
    )


def coerce_optional_text(value: object) -> str | None:
    """Return one trimmed string or None, treating placeholder text as missing."""
    return coerce_dataset_text(value)


def coerce_lender_disclosure(value: object) -> str:
    """Return one known lender-disclosure value, defaulting to `none_named`.

    A mention that records nothing about who holds the debt has named no
    lender, which is exactly `none_named` — the conservative reading, and the
    one that cannot invent a complete syndicate list out of a missing value.
    """
    text = coerce_dataset_text(value)
    return text if text in LENDER_DISCLOSURE_VALUES else LENDER_DISCLOSURE_NONE_NAMED


def aggregate_lender_disclosure(values: list[str | None]) -> str:
    """Roll several mentions' disclosure answers into one for the instrument.

    Worst-of by `LENDER_DISCLOSURE_PRECEDENCE`: a single filing showing a
    collective lender phrase means holders are hidden however many other
    filings name some, while a filing that named every lender supersedes one
    that named none.
    """
    known = [value for value in values if value in LENDER_DISCLOSURE_VALUES]
    if not known:
        return LENDER_DISCLOSURE_NONE_NAMED
    return max(known, key=lambda value: LENDER_DISCLOSURE_PRECEDENCE[value])


def coerce_optional_bool(value: object) -> bool | None:
    """Return one nullable flag read back from a published row.

    A declared `bool` column round-trips as Python or numpy bools with nulls
    read as None or NaN.
    Text spellings are accepted so a hand-built frame reads the same way.
    """
    if value is None or isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def normalize_amount(value: str | None) -> str | None:
    """Normalize amount strings for matcher comparisons."""
    if value is None:
        return None
    lowered = value.lower()
    multiplier = 1
    if "billion" in lowered:
        multiplier = 1_000_000_000
    elif "million" in lowered:
        multiplier = 1_000_000
    elif "thousand" in lowered:
        multiplier = 1_000
    digits = re.findall(r"\d+(?:\.\d+)?", lowered.replace(",", ""))
    if digits:
        amount = float(digits[0]) * multiplier
        if amount.is_integer():
            return str(int(amount))
        return f"{amount:.2f}"
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_date(value: str | None) -> str | None:
    """Normalize date strings for matcher comparisons."""
    if value is None:
        return None
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    month_map = {
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
    match = re.search(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})",
        text,
    )
    if match:
        month = month_map.get(match.group("month").lower())
        if month:
            return f"{match.group('year')}-{month}-{int(match.group('day')):02d}"
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_name_fingerprint(value: str | None) -> str | None:
    """Normalize debt-instrument names for comparison."""
    if value is None:
        return None
    text = value.lower()
    # `4.375 %` -> `4.375%` before the trailing-zero rules look for `%`.
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"(\d+)\.(\d*?[1-9])0+(?=%)", r"\1.\2", text)
    text = re.sub(r"(\d+)\.0+(?=%)", r"\1", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def lender_keys(value: object) -> list[str]:
    """Return normalized lender cluster keys in deterministic order."""
    keys: list[str] = []
    for cluster in parse_cluster_list(str(value or "[]")):
        if cluster.get("role") != "lender":
            continue
        key = cluster_canonical_key(cluster)
        if key:
            keys.append(key)
    return sorted(set(keys))


def lender_signature(value: object) -> str:
    """Return one normalized lender signature from extractor JSON payload."""
    return " | ".join(
        key for key in lender_keys(value) if key not in GENERIC_LENDER_TERMS
    )


def normalize_party_text(value: str) -> str:
    """Normalize party strings before similarity comparison and dedupe."""
    text = value.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(
        r"\b(national association|n a|na|inc|llc|ltd|plc|corp|corporation|company|co)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def borrowed_lender_signature(
    mention: PreparedMention, mention_index: dict[str, PreparedMention]
) -> str | None:
    """Return the successor's lender signature for a synthesized prior state.

    Only when the mention is synthesized, names no lender of its own, and its
    successor is in ``mention_index`` with a signature; None otherwise. Used
    for scoring only, never added to a profile or published row.
    """
    if mention.synthesized_by is None or mention.lender_signature:
        return None
    if mention.synthesized_from_mention_id is None:
        return None
    successor = mention_index.get(mention.synthesized_from_mention_id)
    if successor is None or not successor.lender_signature:
        return None
    return successor.lender_signature


def lender_similarity_score(left: str, right: str) -> float:
    """Return one deterministic similarity score for lender strings."""
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(a=left, b=right).ratio(), 4)


YEAR_TEXT_LENGTH = 4
MONTH_TEXT_LENGTH = 7


def normalized_end_date_for_matching(row: dict[str, object]) -> str | None:
    """Return the end date the matcher compares, at its true resolution.

    Stated maturities keep their day. A name-derived year-end (``due 2030``
    synthesized to ``2030-12-31``) collapses to the year; any other derived
    maturity collapses to ``YYYY-MM``. None when there is no maturity.
    """
    value = normalize_date(coerce_optional_text(row.get("maturity_date")))
    if not value:
        return None
    derivation = maturity_derivation(row.get("maturity_date_json"))
    if derivation not in DERIVED_MATURITY_KINDS:
        return value
    if derivation == "name" and value.endswith("-12-31"):
        return value[:YEAR_TEXT_LENGTH]
    # Other derived dates are month-trustworthy but not day-exact.
    return value[:MONTH_TEXT_LENGTH]


# Maturities the extractor derived rather than read off a stated date: from
# the instrument's name, or computed as start plus tenor.
DERIVED_MATURITY_KINDS = frozenset({"name", "computed"})


def maturity_derivation(payload_text: object) -> str | None:
    """Return one maturity payload's derived_from marker."""
    try:
        payload = json.loads(str(payload_text or "{}"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    derivation = payload.get("derived_from")
    return str(derivation) if isinstance(derivation, str) else None
