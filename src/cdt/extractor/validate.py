"""Validation rules an instrument IE response must pass; failures become retry messages."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from cdt.extractor.normalize.amounts import (
    is_rate_like_amount_text,
    normalized_amount_from_text,
)
from cdt.extractor.normalize.dates import is_valid_iso_date, normalized_date_from_text
from cdt.extractor.schema import (
    AMOUNT_EVIDENCE_TAG_TYPES,
    AMOUNT_KINDS,
    CURRENCY_CODE_LENGTH,
    DATE_KIND_EVIDENCE_TAG_TYPES,
    DATE_KINDS,
    DATE_KINDS_REQUIRING_EVIDENCE,
    DEFAULT_DATE_EVIDENCE_TAG_TYPES,
    EVENT_DATE_KINDS,
    INSTRUMENT_SINGLE_VALUE_PROPERTIES,
    INSTRUMENT_TYPES,
    INTEREST_RATE_EVIDENCE_TAG_TYPES,
    INTEREST_RATE_KINDS,
    ISO_DATE_PATTERN,
    LENDER_TAG_TYPES,
    MATURITY_EVIDENCE_TAG_TYPES,
    NUMERIC_STRING_PATTERN,
    PARTY_KINDS,
    PARTY_ROLES,
    PRINCIPAL_AMOUNT_KINDS,
    SINGLE_CURRENT_DATE_KINDS,
    TERMINAL_DATE_KINDS,
)
from cdt.extractor.tags import normalize_span_whitespace, single_value_evidence_tag_ids


def is_one_of(value: object, allowed: Collection[str]) -> bool:
    """Return whether ``value`` is a string in ``allowed``.

    Model JSON can put a list or object where a string belongs, and testing an
    unhashable value against a set raises; this makes it a validation failure.
    """
    return isinstance(value, str) and value in allowed


def validate_instrument_entry(
    index: int,
    obj: object,
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Return the failures for one instrument entry on its own; empty means valid.

    Per-entry so terminal salvage can keep the individually-valid entries of a
    response whose other entries failed.
    """
    if not isinstance(obj, dict):
        return [f"Entry {index} is not a JSON object."]
    failures: list[str] = []
    for (
        property_name,
        expected_types,
    ) in INSTRUMENT_SINGLE_VALUE_PROPERTIES.items():
        if property_name not in obj:
            continue
        tag_ids = single_value_evidence_tag_ids(obj[property_name])
        if not isinstance(tag_ids, list):
            failures.append(
                f"Entry {index}: '{property_name}' evidence must be a list of tag IDs."
            )
            continue
        if not all(isinstance(tag_id, str) for tag_id in tag_ids):
            failures.append(
                f"Entry {index}: '{property_name}' evidence must contain string tag IDs only."
            )
            continue
        for tag_id in tag_ids:
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(
                    f"Entry {index}: '{property_name}' contains unknown tag ID {tag_id}."
                )
                continue
            if tag_info["type"] not in expected_types:
                expected = ", ".join(sorted(expected_types))
                failures.append(
                    f"Entry {index}: '{property_name}' tag {tag_id} is type '{tag_info['type']}', expected {expected}."
                )
    if "instrument_type" in obj and not is_one_of(
        obj["instrument_type"], INSTRUMENT_TYPES
    ):
        allowed = ", ".join(sorted(INSTRUMENT_TYPES))
        failures.append(
            f"Entry {index}: 'instrument_type' must be one of {allowed}, or omitted."
        )
    failures.extend(
        validate_amounts_property(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
    )
    failures.extend(
        validate_dates_property(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
    )
    failures.extend(
        validate_interest_rate(
            index=index,
            obj=obj,
            tag_details=tag_details,
        )
    )
    failures.extend(
        validate_parties_property(index=index, obj=obj, tag_details=tag_details)
    )
    failures.extend(validate_no_legacy_properties(index, obj))
    failures.extend(validate_cross_field_semantics(index=index, obj=obj))
    return failures


# Which amount kinds fit which instrument types: a facility has a commitment, a
# security has a principal. Balances, draws, repayments and proceeds fit any.
FACILITY_INSTRUMENT_TYPES = {"revolving_credit", "credit_line"}
AMOUNT_KIND_TYPE_CONFLICTS = {
    ("commitment", "note_bond"),
    ("principal", "revolving_credit"),
    ("principal", "credit_line"),
}


# Properties of the pre-facts schema, which the model can still revert to; each
# maps to the instruction naming its replacement.
LEGACY_INSTRUMENT_PROPERTIES = {
    "status_event": "events are `dates` entries (kinds closing, amendment, retirement, ...)",
    "lenders": "parties are one `parties` list with role lender",
    "other_interested_parties": "parties are one `parties` list with a role per cluster",
    "lenders_known_incomplete": "derived from the lender clusters; do not return it",
    "start_date": "a `dates` entry of kind closing (or agreement)",
    "maturity_date": "a `dates` entry of kind maturity",
    "end_date": "a `dates` entry of kind maturity",
    "commitment_termination_date": "a `dates` entry of kind commitment_termination",
    "amount": "an `amounts` entry with a kind",
}


def validate_no_legacy_properties(index: int, obj: object) -> list[str]:
    """Reject pre-facts-schema properties, naming what replaced each."""
    if not isinstance(obj, dict):
        return []
    return [
        f"Entry {index}: '{name}' is not a property of this schema; use {replacement}."
        for name, replacement in LEGACY_INSTRUMENT_PROPERTIES.items()
        if name in obj
    ]


def validate_cross_field_semantics(*, index: int, obj: dict[str, Any]) -> list[str]:
    """Reject shapes the prompt forbids but no single-field check can see."""
    failures: list[str] = []
    dates = obj.get("dates") if isinstance(obj.get("dates"), list) else []
    amounts = obj.get("amounts") if isinstance(obj.get("amounts"), list) else []
    date_kinds = _entry_kinds(dates)
    amount_kinds = _entry_kinds(amounts)
    for entry in amounts:
        if (
            isinstance(entry, dict)
            and entry.get("prior") is True
            and not is_one_of(entry.get("kind"), PRINCIPAL_AMOUNT_KINDS)
        ):
            failures.append(
                f"Entry {index}: 'amounts' entry of kind '{entry.get('kind')}' cannot "
                "be `prior`. Only a commitment or principal has a before-the-change "
                "figure; a balance, draw, repayment or proceeds is dated, not prior."
            )
    if "repayment" in date_kinds and "repayment" not in amount_kinds:
        failures.append(
            f"Entry {index}: a `repayment` date entry needs the repaid figure as a "
            "`repayment` entry in 'amounts' when the document states one; if it states "
            "no figure, the payment is a `retirement` (in full) or nothing."
        )
    if (
        "repayment" in amount_kinds
        and dates
        and not date_kinds & {"repayment", *TERMINAL_DATE_KINDS}
    ):
        # A repayment figure is dated by the payment event that produced it: a
        # partial paydown (`repayment`) or the retirement/exchange that paid the
        # rest, so a terminal event needs no separate `repayment` date.
        failures.append(
            f"Entry {index}: a `repayment` amount is an event; add a `repayment` entry to "
            "'dates' (with `evidence` [] and `normalized_date` null when no date is stated), "
            "unless a retirement, termination or exchange entry already dates the payment."
        )
    instrument_type = obj.get("instrument_type")
    if not isinstance(instrument_type, str):
        return failures
    for kind in sorted(amount_kinds):
        if (kind, instrument_type) in AMOUNT_KIND_TYPE_CONFLICTS:
            failures.append(
                f"Entry {index}: 'amounts' kind '{kind}' does not fit instrument_type "
                f"'{instrument_type}'. A facility's size is a `commitment`; a security's "
                "face amount is a `principal`. Fix the kind or the type."
            )
    return failures


def _entry_kinds(entries: list[Any]) -> set[str]:
    """Return the string ``kind`` values of the object entries in a list."""
    return {
        entry["kind"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("kind"), str)
    }


def validate_parties_property(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the unified `parties` list: role required, kind optional."""
    if "parties" not in obj:
        return []
    value = obj["parties"]
    if not isinstance(value, list):
        return [
            f"Entry {index}: 'parties' must be a list of cluster objects shaped like "
            '{"tag_ids": ["tag-..."], "role": "...", "kind": "named" | "collective"}.'
        ]
    failures: list[str] = []
    roles = ", ".join(sorted(PARTY_ROLES))
    for cluster_index, cluster in enumerate(value):
        location = f"Entry {index}: 'parties'[{cluster_index}]"
        if not isinstance(cluster, dict):
            failures.append(f"{location} must be an object with 'tag_ids' and 'role'.")
            continue
        if not is_one_of(cluster.get("role"), PARTY_ROLES):
            failures.append(f"{location} 'role' must be one of {roles}.")
        if "kind" in cluster and not is_one_of(cluster["kind"], PARTY_KINDS):
            failures.append(f"{location} 'kind' must be named or collective.")
        tag_ids = cluster.get("tag_ids")
        if not isinstance(tag_ids, list):
            failures.append(f"{location} 'tag_ids' must be a list of tag IDs.")
            continue
        for tag_id in tag_ids:
            if not isinstance(tag_id, str):
                failures.append(
                    f"{location} 'tag_ids' must contain string tag IDs only."
                )
                continue
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(f"{location} contains unknown tag ID {tag_id}.")
            elif tag_info["type"] not in LENDER_TAG_TYPES:
                failures.append(
                    f"{location} tag {tag_id} must be person or organization."
                )
    return failures


def validate_standardized_single_value_cardinality(
    *,
    index: int,
    property_name: str,
    value: object,
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate that one standardized single-value payload does not encode conflicts."""
    if not isinstance(value, dict):
        return []
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return []
    evidence_texts = [
        str(tag_details[tag_id]["text"])
        for tag_id in evidence
        if isinstance(tag_id, str)
        and tag_id in tag_details
        # Name spans carrying an embedded maturity are not comparable date mentions.
        and tag_details[tag_id]["type"] not in MATURITY_EVIDENCE_TAG_TYPES
    ]
    if property_name == "amount":
        normalized_values = {
            parsed
            for parsed in (normalized_amount_from_text(text) for text in evidence_texts)
            if parsed is not None
        }
    else:
        normalized_values = {
            parsed
            for parsed in (normalized_date_from_text(text) for text in evidence_texts)
            if parsed is not None
        }
    if len(normalized_values) <= 1:
        return []
    return [
        (
            f"Entry {index}: '{property_name}' contains multiple distinct normalized "
            "values. Split this into separate debt instrument objects instead of "
            "combining them."
        )
    ]


def validate_amount_is_not_rate(
    *,
    index: int,
    value: object,
    tag_details: dict[str, dict[str, object]],
    property_label: str = "amount",
) -> list[str]:
    """Reject an amount whose evidence only describes a rate, margin, or fee."""
    if not isinstance(value, dict):
        return []
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return []
    # The same normalization `canonical_value` applies, so the validator and
    # `standardized_amount_payload` judge one text; collapsing rather than
    # stripping whitespace keeps multi-word markers such as `basis point`.
    evidence_texts = [
        normalize_span_whitespace(str(tag_details[tag_id]["text"]))
        for tag_id in evidence
        if isinstance(tag_id, str) and tag_id in tag_details
    ]
    if not evidence_texts or not all(
        is_rate_like_amount_text(text) for text in evidence_texts
    ):
        return []
    quoted = ", ".join(f"'{text}'" for text in evidence_texts)
    return [
        (
            f"Entry {index}: '{property_label}' evidence {quoted} describes an interest rate, "
            "margin, or fee rather than a money amount. Cite the money amount "
            f"instead, or omit '{property_label}'."
        )
    ]


def validate_dates_property(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the kind-typed dates list on one instrument entry.

    Every entry needs a known kind and date-typed evidence (a maturity may also
    cite the instrument's name span or a duration span). At most one
    non-prior entry per kind: two current maturities describe two instruments,
    exactly as two principals do.
    """
    if "dates" not in obj:
        return []
    entries = obj["dates"]
    if not isinstance(entries, list):
        return [f"Entry {index}: 'dates' must be a list of objects."]
    failures: list[str] = []
    current_kinds: dict[str, int] = {}
    for position, entry in enumerate(entries):
        label = f"dates[{position}]"
        if not isinstance(entry, dict):
            failures.append(f"Entry {index}: '{label}' must be an object.")
            continue
        kind = entry.get("kind")
        if not is_one_of(kind, DATE_KINDS):
            allowed = ", ".join(sorted(DATE_KINDS))
            failures.append(f"Entry {index}: '{label}.kind' must be one of {allowed}.")
            kind = None
        prior = entry.get("prior", False)
        if not isinstance(prior, bool):
            failures.append(f"Entry {index}: '{label}.prior' must be true or false.")
        elif (
            kind is not None
            and not prior
            # `expected` is not current either: `select_date_payload` publishes
            # the entry that is neither prior nor expected, so a real closing
            # beside a planned one is one current closing, not two.
            and entry.get("expected") is not True
            and kind in SINGLE_CURRENT_DATE_KINDS
        ):
            current_kinds[kind] = current_kinds.get(kind, 0) + 1
        if not isinstance(entry.get("expected", False), bool):
            failures.append(f"Entry {index}: '{label}.expected' must be true or false.")
        elif (
            entry.get("expected") is True
            and kind is not None
            and kind not in EVENT_DATE_KINDS
        ):
            failures.append(
                f"Entry {index}: '{label}' of kind '{kind}' cannot be `expected`. Only "
                "events (announcement, closing, amendment, repayment, retirement, "
                "termination, exchange, default) are planned or completed; a term such "
                "as a maturity is simply stated."
            )
        if prior is True and kind is not None and kind in EVENT_DATE_KINDS:
            failures.append(
                f"Entry {index}: '{label}' of kind '{kind}' cannot be `prior`. `prior` marks "
                "a term (maturity, commitment_termination, agreement) as it stood before "
                "a change; an event either happened or is `expected`."
            )
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            failures.append(
                f"Entry {index}: '{label}.evidence' must be a list of tag IDs."
            )
            continue
        if not evidence:
            if kind in DATE_KINDS_REQUIRING_EVIDENCE:
                failures.append(
                    f"Entry {index}: '{label}' of kind '{kind}' must cite a date span."
                )
            if entry.get("normalized_date") is not None:
                failures.append(
                    f"Entry {index}: '{label}.normalized_date' must be null when no "
                    "span is cited."
                )
        expected_types = DATE_KIND_EVIDENCE_TAG_TYPES.get(
            kind or "", DEFAULT_DATE_EVIDENCE_TAG_TYPES
        )
        for tag_id in evidence:
            if not isinstance(tag_id, str):
                failures.append(
                    f"Entry {index}: '{label}.evidence' must contain string tag IDs only."
                )
                continue
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(
                    f"Entry {index}: '{label}' contains unknown tag ID {tag_id}."
                )
            elif tag_info["type"] not in expected_types:
                expected = ", ".join(sorted(expected_types))
                failures.append(
                    f"Entry {index}: '{label}' tag {tag_id} is type "
                    f"'{tag_info['type']}', expected {expected}."
                )
        normalized_date = entry.get("normalized_date")
        if normalized_date is not None and (
            not isinstance(normalized_date, str)
            or not ISO_DATE_PATTERN.fullmatch(normalized_date)
            or not is_valid_iso_date(normalized_date)
        ):
            failures.append(
                f"Entry {index}: '{label}.normalized_date' must be YYYY-MM-DD or null."
            )
        failures.extend(
            validate_standardized_single_value_cardinality(
                index=index,
                property_name=label,
                value=entry,
                tag_details=tag_details,
            )
        )
    for kind, count in sorted(current_kinds.items()):
        if count > 1:
            failures.append(
                f"Entry {index}: 'dates' has {count} current entries of kind "
                f"'{kind}'. One instrument has one {kind} date; a term stated "
                "before and after a change marks the earlier one `prior: true`, "
                "and two unrelated values are two instruments."
            )
    return failures


def validate_amounts_property(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the kind-typed amounts list on one instrument entry."""
    if "amounts" not in obj:
        return []
    entries = obj["amounts"]
    if not isinstance(entries, list):
        return [f"Entry {index}: 'amounts' must be a list of objects."]
    failures: list[str] = []
    for position, entry in enumerate(entries):
        label = f"amounts[{position}]"
        if not isinstance(entry, dict):
            failures.append(f"Entry {index}: '{label}' must be an object.")
            continue
        kind = entry.get("kind")
        if not is_one_of(kind, AMOUNT_KINDS):
            allowed = ", ".join(sorted(AMOUNT_KINDS))
            failures.append(f"Entry {index}: '{label}.kind' must be one of {allowed}.")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            failures.append(
                f"Entry {index}: '{label}.evidence' must be a list of tag IDs."
            )
            continue
        for tag_id in evidence:
            if not isinstance(tag_id, str):
                failures.append(
                    f"Entry {index}: '{label}.evidence' must contain string tag IDs only."
                )
                continue
            tag_info = tag_details.get(tag_id)
            if tag_info is None:
                failures.append(
                    f"Entry {index}: '{label}' contains unknown tag ID {tag_id}."
                )
            elif tag_info["type"] not in AMOUNT_EVIDENCE_TAG_TYPES:
                expected = ", ".join(sorted(AMOUNT_EVIDENCE_TAG_TYPES))
                failures.append(
                    f"Entry {index}: '{label}' tag {tag_id} is type "
                    f"'{tag_info['type']}', expected {expected}."
                )
        normalized_amount = entry.get("normalized_amount")
        if normalized_amount is not None and (
            not isinstance(normalized_amount, str)
            or not NUMERIC_STRING_PATTERN.fullmatch(normalized_amount)
        ):
            failures.append(
                f"Entry {index}: '{label}.normalized_amount' must be a numeric "
                "string or null."
            )
        currency = entry.get("currency")
        if currency is not None and (
            not isinstance(currency, str)
            or len(currency) != CURRENCY_CODE_LENGTH
            or currency != currency.upper()
        ):
            failures.append(
                f"Entry {index}: '{label}.currency' must be an uppercase "
                "3-letter code or null."
            )
        as_of = entry.get("as_of_date")
        if as_of is not None and (
            not isinstance(as_of, str)
            or not ISO_DATE_PATTERN.fullmatch(as_of)
            or not is_valid_iso_date(as_of)
        ):
            failures.append(
                f"Entry {index}: '{label}.as_of_date' must be YYYY-MM-DD or null."
            )
        if not isinstance(entry.get("prior", False), bool):
            failures.append(f"Entry {index}: '{label}.prior' must be true or false.")
        failures.extend(
            validate_amount_is_not_rate(
                index=index,
                value=entry,
                tag_details=tag_details,
                property_label=label,
            )
        )
    return failures


def validate_interest_rate(
    *,
    index: int,
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
) -> list[str]:
    """Validate the optional interest_rate on one instrument entry."""
    if "interest_rate" not in obj:
        return []
    value = obj["interest_rate"]
    if not isinstance(value, dict):
        return [f"Entry {index}: 'interest_rate' must be an object."]
    failures: list[str] = []
    kind = value.get("kind")
    if not is_one_of(kind, INTEREST_RATE_KINDS):
        allowed = ", ".join(sorted(INTEREST_RATE_KINDS))
        failures.append(
            f"Entry {index}: 'interest_rate.kind' must be one of {allowed}."
        )
    rate_pct = value.get("rate_pct")
    if rate_pct is not None and (
        not isinstance(rate_pct, str) or not NUMERIC_STRING_PATTERN.fullmatch(rate_pct)
    ):
        failures.append(
            f"Entry {index}: 'interest_rate.rate_pct' must be a numeric string or null."
        )
    # An absent `evidence` key is an empty citation list, not a malformed
    # response: the prompt's own `3.875% senior notes due 2028` example omits it
    # and postprocess verifies the rate off the instrument name instead.
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        failures.append(
            f"Entry {index}: 'interest_rate.evidence' must be a list of tag IDs."
        )
        return failures
    for tag_id in evidence:
        if not isinstance(tag_id, str):
            failures.append(
                f"Entry {index}: 'interest_rate.evidence' must contain string tag IDs only."
            )
            continue
        tag_info = tag_details.get(tag_id)
        if tag_info is None:
            failures.append(
                f"Entry {index}: 'interest_rate' contains unknown tag ID {tag_id}."
            )
        elif tag_info["type"] not in INTEREST_RATE_EVIDENCE_TAG_TYPES:
            expected = ", ".join(sorted(INTEREST_RATE_EVIDENCE_TAG_TYPES))
            failures.append(
                f"Entry {index}: 'interest_rate' tag {tag_id} is type "
                f"'{tag_info['type']}', expected {expected}."
            )
    return failures
