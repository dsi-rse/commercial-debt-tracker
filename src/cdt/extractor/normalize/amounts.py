"""Turn cited amount and rate text into checked money and interest-rate facts."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from cdt.extractor.normalize.dates import is_valid_iso_date
from cdt.extractor.schema import (
    AMOUNT_KINDS,
    AMOUNT_MULTIPLIERS,
    AMOUNT_VALUE_PATTERN,
    COMMON_CURRENCY_CODES,
    DERIVED_FROM_COMPUTED,
    DERIVED_FROM_NAME,
    DERIVED_FROM_SCALED,
    DERIVED_FROM_STATED,
    INTEREST_RATE_KINDS,
    ISO_DATE_PATTERN,
    MINIMUM_COMPUTED_SUM_SPANS,
    NAME_EMBEDDED_AMOUNT_PATTERN,
    NUMERIC_STRING_PATTERN,
    PRINCIPAL_AMOUNT_KINDS,
    QUALIFIED_DOLLAR_CODES,
    QUALIFIED_DOLLAR_PATTERN,
    RATE_PCT_PATTERN,
    RATE_SUFFIX_PATTERN,
)
from cdt.extractor.tags import (
    cluster_payload,
    cluster_span_texts,
    payload_tag_ids,
    single_value_evidence_tag_ids,
)
from cdt.shared import get_logger
from cdt.storage import canonical_numeric_text

LOGGER = get_logger(__name__)

_SUPPORTED_CURRENCY_CODES: set[str] | None = None


# `6 1/2%`, `5 7/8 %`: coupons written as vulgar fractions, common in older
# indentures and in Targa's note names. Read as a decimal rate.
FRACTION_RATE_PATTERN = re.compile(
    r"(\d+)\s+(\d+)\s*/\s*(\d+)\s*(?:%|percent\b)", re.IGNORECASE
)


def rate_tokens(text: str) -> list[str]:
    """Return every rate figure in a span as a decimal string, fractions included."""
    tokens: list[str] = []
    for whole, numerator, denominator in FRACTION_RATE_PATTERN.findall(text):
        if int(denominator):
            value = Decimal(whole) + Decimal(numerator) / Decimal(denominator)
            tokens.append(format(value.normalize(), "f"))
    stripped = FRACTION_RATE_PATTERN.sub(" ", text)
    tokens.extend(RATE_PCT_PATTERN.findall(stripped))
    return tokens


def rate_tokens_in_rate_span(text: str) -> list[str]:
    """Return the rate figures in a span the tagger already typed `interest_rate`.

    A prose span carries its own marker (`4.125%`, `6.5 percent`, `6 1/2%`). A
    table cell under a `COUPON PCT` header is the bare number `4.125`: the
    tagger's type is the marker, so a span that is nothing but a number is that
    rate. Returns an empty list when the span has neither.
    """
    marked = rate_tokens(text)
    if marked:
        return marked
    bare = text.strip()
    return [bare] if NUMERIC_STRING_PATTERN.fullmatch(bare) else []


def standardized_interest_rate_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
) -> dict[str, object]:
    """Return the persisted interest-rate payload for one entry.

    The model's ``rate_pct`` publishes, in canonical numeric form, only when a
    rate token in the cited evidence — or, failing that, in the instrument's
    name — parses to the same number, mirroring how amounts and dates are
    parser-verified. A non-dict ``value`` yields an all-null payload.
    """
    if not isinstance(value, dict):
        return {"kind": None, "rate_pct": None, "spans": [], "derived_from": None}
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    kind = value.get("kind")
    payload["kind"] = kind if kind in INTEREST_RATE_KINDS else None
    tag_ids = evidence_tag_ids if isinstance(evidence_tag_ids, list) else []
    stated_rates = {
        rate
        for tag_id in tag_ids
        if isinstance(tag_id, str)
        and tag_details.get(tag_id, {}).get("type") == "interest_rate"
        for rate in rate_tokens_in_rate_span(str(tag_details[tag_id]["text"]))
    }
    # A rate read off the instrument's own name span is name-derived, exactly
    # like a maturity embedded in `notes due 2028`.
    name_rates = {
        rate
        for tag_id in tag_ids
        if isinstance(tag_id, str)
        and tag_details.get(tag_id, {}).get("type") == "debt_instrument"
        for rate in rate_tokens(str(tag_details[tag_id]["text"]))
    }
    if not stated_rates and not name_rates and name_text:
        name_rates = set(rate_tokens(name_text))
    evidence_rates = stated_rates or name_rates
    derived_from = (
        DERIVED_FROM_STATED
        if stated_rates
        else DERIVED_FROM_NAME
        if name_rates
        else None
    )
    model_rate = value.get("rate_pct")
    verified_rate = None
    if isinstance(model_rate, str) and NUMERIC_STRING_PATTERN.fullmatch(model_rate):
        for candidate in evidence_rates:
            try:
                if Decimal(candidate) == Decimal(model_rate):
                    # Publish the canonical form, not the model's spelling, so
                    # one rate is one string (`5`, not `5.00` and `5.000`).
                    verified_rate = normalize_numeric_string(Decimal(model_rate))
                    break
            except InvalidOperation:
                continue
    payload["rate_pct"] = verified_rate
    payload["derived_from"] = derived_from if verified_rate is not None else None
    return payload


def canonical_amount_value(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
) -> str | None:
    """Return the amount cluster's canonical text, preferring a parseable span.

    The longest span that parses as an amount, else the longest span; None for
    an empty cluster. An amount cluster often pairs the figure with a longer
    label (`['$2,000,000', 'Principal Amount']`), and the rate guard reads this
    same text.
    """
    values = cluster_span_texts(tag_ids, tag_details)
    if not values:
        return None
    parseable = [
        value for value in values if normalized_amount_from_text(value) is not None
    ]
    return max(parseable or values, key=len)


def supported_currency_codes() -> set[str]:
    """Return supported ISO 4217 codes."""
    global _SUPPORTED_CURRENCY_CODES
    if _SUPPORTED_CURRENCY_CODES is not None:
        return _SUPPORTED_CURRENCY_CODES

    codes: set[str] = set(COMMON_CURRENCY_CODES)
    try:
        import pycountry

        codes.update(
            currency.alpha_3
            for currency in pycountry.currencies
            if getattr(currency, "alpha_3", None)
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug("pycountry unavailable; falling back to bundled currency set.")
    _SUPPORTED_CURRENCY_CODES = codes
    return _SUPPORTED_CURRENCY_CODES


def normalize_numeric_string(value: Decimal) -> str:
    """Return one deterministic numeric string.

    Takes a Decimal rather than a float so amounts with cents render exactly.
    """
    return canonical_numeric_text(value)


def decimal_from_amount_string(value: str | None) -> Decimal | None:
    """Return one amount string as a Decimal, or None when it is not numeric."""
    if not isinstance(value, str):
        return None
    stripped = value.strip().replace(",", "")
    if not stripped:
        return None
    try:
        parsed = Decimal(stripped)
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return parsed


def computed_sum_amount(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
    model_amount: object,
) -> str | None:
    """Return the model's amount when it equals the sum of the cited spans.

    An increase-by amendment states a prior total and an increment but often
    never the result; the sum is deterministic arithmetic anchored to the cited
    evidence, and it is the only arithmetic accepted. Requires at least two
    parseable spans, and refuses when any single span already equals the
    model's value — that is agreement, not computation — or when any cited
    span reads as a rate.
    """
    if not isinstance(model_amount, str):
        return None
    model_value = decimal_from_amount_string(model_amount)
    if model_value is None:
        return None
    texts = cluster_span_texts(tag_ids, tag_details)
    if any(is_rate_like_amount_text(text) for text in texts):
        return None
    parsed = [normalized_amount_from_text(text) for text in texts]
    values = [
        decimal_from_amount_string(value) for value in parsed if value is not None
    ]
    if len(values) < MINIMUM_COMPUTED_SUM_SPANS or None in values:
        return None
    if any(value == model_value for value in values):
        return None
    if sum(values) != model_value:
        return None
    return normalized_amount_from_text(model_amount) or model_amount


def scaled_amount_from_sibling(
    own_tag_ids: object,
    sibling_tag_ids: object,
    tag_details: dict[str, dict[str, object]],
    model_amount: object,
) -> str | None:
    """Return the model's amount when a sibling fact's span carries its magnitude.

    Filings write a magnitude word once, after the second of two figures
    (`from $400.0 to $500.0 million`), so a fact citing only `$400.0` parses to
    400 while the model correctly reports 400000000. Returns the model's value,
    re-normalized, only when no cited span (own or sibling) is rate-like, no
    own span carries a magnitude, the sibling spans (cited by the object's
    other amount facts) carry exactly one distinct magnitude, and the own
    canonical reading times that magnitude equals the model's value exactly.
    Otherwise None, so it only ever confirms the model's figure. See
    docs/decisions/extraction.md.
    """
    # Type-contract refusals mirroring `computed_sum_amount`; the exact-product
    # comparison below would also reject these inputs.
    if not isinstance(model_amount, str):
        return None
    model_value = decimal_from_amount_string(model_amount)
    if model_value is None:
        return None
    own_texts = cluster_span_texts(own_tag_ids, tag_details)
    sibling_texts = cluster_span_texts(sibling_tag_ids, tag_details)
    if any(is_rate_like_amount_text(text) for text in (*own_texts, *sibling_texts)):
        return None
    # `canonical_amount_value` supplies the reading, because that is the span
    # `amounts_agree` just disagreed with.
    own_text = canonical_amount_value(own_tag_ids, tag_details)
    base = normalized_amount_from_text(own_text)
    # Asked of *every* own span, not just the canonical one: a fact whose own
    # evidence already carries a magnitude is never rescaled, even when the
    # canonical span is a bare figure. This also means a sibling citing one of
    # this fact's own spans contributes nothing to `magnitudes`.
    if base is None or any(
        magnitude_in_amount_text(text) is not None for text in own_texts
    ):
        return None
    base_value = decimal_from_amount_string(base)
    if base_value is None:
        return None
    magnitudes = {
        magnitude
        for magnitude in (magnitude_in_amount_text(text) for text in sibling_texts)
        if magnitude is not None
    }
    if len(magnitudes) != 1:
        # None means there is no shared magnitude word to borrow; more than one
        # means the siblings disagree about which, and guessing between
        # `million` and `billion` is a three-orders-of-magnitude error.
        return None
    if base_value * magnitudes.pop() != model_value:
        return None
    return normalized_amount_from_text(model_amount) or model_amount


def amounts_agree(model_amount: object, parsed_amount: str | None) -> bool:
    """Return whether the model's amount is the same value the parser read.

    Compared numerically, so `500000.00` agrees with a parsed `500000`. False
    when either side is missing or not numeric.
    """
    if not isinstance(model_amount, str) or parsed_amount is None:
        return False
    model_value = decimal_from_amount_string(model_amount)
    parsed_value = decimal_from_amount_string(parsed_amount)
    if model_value is None or parsed_value is None:
        return False
    return model_value == parsed_value


def magnitude_in_amount_text(text: str | None) -> int | None:
    """Return the magnitude `normalized_amount_from_text` would apply, or None.

    The one definition of a magnitude word, shared with
    `scaled_amount_from_sibling` so the two cannot disagree.
    """
    if not text:
        return None
    lowered = text.lower().replace(",", "")
    for word in sorted(AMOUNT_MULTIPLIERS, key=len, reverse=True):
        # `(?<![a-z])` rather than `\b` on the left so `$500mm` reads as well as
        # `$500 mm`, while `million` still cannot match inside a longer word.
        if re.search(rf"(?<![a-z]){word}\b", lowered):
            return AMOUNT_MULTIPLIERS[word]
    return None


def normalized_amount_from_text(text: str | None) -> str | None:
    """Parse one amount mention into a normalized numeric string."""
    if not text:
        return None
    lowered = text.lower().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", lowered)
    if not match:
        return None
    amount = decimal_from_amount_string(match.group(0))
    if amount is None:
        return None
    magnitude = magnitude_in_amount_text(text)
    if magnitude is not None:
        amount *= magnitude
    return normalize_numeric_string(amount)


def normalized_amount_from_name(text: str | None) -> str | None:
    """Parse a principal stated inside an instrument name into a numeric string.

    `$183.36 million term loan` carries its own principal, and NER tags the whole
    phrase as one `debt_instrument`, so there is no `amount` span to cite. Only
    currency-marked figures count; a name stating more than one distinct figure
    names no single principal and parses to None, as maturities do.
    """
    if not text:
        return None
    values = {
        normalized_amount_from_text(match.group(0))
        for match in NAME_EMBEDDED_AMOUNT_PATTERN.finditer(text)
    }
    values.discard(None)
    if len(values) != 1:
        return None
    return values.pop()


def currency_from_name(text: str | None) -> str | None:
    """Return the currency of a principal stated inside an instrument name."""
    if not text:
        return None
    matches = list(NAME_EMBEDDED_AMOUNT_PATTERN.finditer(text))
    if len(matches) != 1:
        return None
    candidates = currency_candidates_from_text(matches[0].group(0))
    if len(candidates) != 1:
        return None
    return candidates.pop()


def currency_candidates_from_text(text: str | None) -> set[str]:
    """Infer plausible ISO currency codes from one amount mention."""
    if not text:
        return set()
    lowered = text.lower()
    candidates: set[str] = set()
    # A qualified dollar sign is a different currency: `C$300 million` is CAD,
    # not USD.
    qualified = QUALIFIED_DOLLAR_PATTERN.findall(text)
    candidates.update(QUALIFIED_DOLLAR_CODES[prefix.upper()] for prefix in qualified)
    if (
        text.count("$") > len(qualified)
        or "u.s. dollar" in lowered
        or "us dollar" in lowered
    ):
        candidates.add("USD")
    if "€" in text or " euro" in lowered:
        candidates.add("EUR")
    if "£" in text or " pound sterling" in lowered or " british pound" in lowered:
        candidates.add("GBP")
    if "¥" in text or " yen" in lowered:
        candidates.add("JPY")
    for match in re.findall(r"\b[A-Z]{3}\b", text):
        if match in supported_currency_codes():
            candidates.add(match)
    return candidates


def is_rate_like_amount_text(text: str | None) -> bool:
    """Return whether one amount evidence string reads as a rate, margin, or fee.

    Every number in the span has to carry a rate marker. A marker appearing
    somewhere is not enough: `500,000,000 (100% of principal)` states a
    principal and then a percentage of it, and `normalized_amount_from_text`
    reads the first number. A currency marker or magnitude word means not a
    rate.
    """
    if not text:
        return False
    lowered = text.lower()
    if currency_candidates_from_text(text):
        return False
    if any(re.search(rf"\b{word}\b", lowered) for word in AMOUNT_MULTIPLIERS):
        return False
    values = list(AMOUNT_VALUE_PATTERN.finditer(text))
    if not values:
        return False
    return all(RATE_SUFFIX_PATTERN.match(text, value.end()) for value in values)


def standardized_amount_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None = None,
    document_currencies: frozenset[str] | None = None,
) -> dict[str, object]:
    """Return evidence payload plus validated normalized amount fields.

    The model's amount publishes only when it agrees with the parsed cited
    span, the name-embedded principal, or the sum of the cited spans; its
    currency only when supported and evidenced. ``document_currencies`` are the
    currencies the whole item text evidences: when the cited span shows none
    (a bare table cell) and the document shows exactly one, that one counts.
    """
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    evidence_text = canonical_amount_value(evidence_tag_ids, tag_details)
    parsed_amount = normalized_amount_from_text(evidence_text)
    parsed_currency_candidates = currency_candidates_from_text(evidence_text)
    derived_from = DERIVED_FROM_STATED if parsed_amount is not None else None
    if parsed_amount is None:
        # A principal stated inside the name has no `amount` span to cite, so the
        # name is the only evidence there is.
        name_amount = normalized_amount_from_name(name_text)
        if name_amount is not None:
            parsed_amount = name_amount
            derived_from = DERIVED_FROM_NAME
            name_currency = currency_from_name(name_text)
            parsed_currency_candidates = (
                {name_currency} if name_currency else parsed_currency_candidates
            )
    model_amount = value.get("normalized_amount") if isinstance(value, dict) else None
    model_currency = value.get("currency") if isinstance(value, dict) else None
    if is_rate_like_amount_text(evidence_text):
        # Rates, margins, and fees are not principal amounts.
        parsed_amount = None
        parsed_currency_candidates = set()

    # The parser's own string is published, so a model reporting the same value
    # with different formatting keeps its amount.
    payload["normalized_amount"] = (
        parsed_amount if amounts_agree(model_amount, parsed_amount) else None
    )
    if payload["normalized_amount"] is None:
        computed = computed_sum_amount(evidence_tag_ids, tag_details, model_amount)
        if computed is not None:
            payload["normalized_amount"] = computed
            derived_from = DERIVED_FROM_COMPUTED
            parsed_currency_candidates = {
                currency
                for text in cluster_span_texts(evidence_tag_ids, tag_details)
                for currency in currency_candidates_from_text(text)
            }
    if (
        not parsed_currency_candidates
        and document_currencies is not None
        and len(document_currencies) == 1
        and not is_rate_like_amount_text(evidence_text)
    ):
        parsed_currency_candidates = set(document_currencies)
    payload["currency"] = (
        model_currency
        if isinstance(model_currency, str)
        and model_currency in supported_currency_codes()
        and model_currency in parsed_currency_candidates
        else None
    )
    payload["derived_from"] = (
        derived_from if payload["normalized_amount"] is not None else None
    )
    return payload


def standardized_amounts_payloads(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
    document_currencies: frozenset[str] | None = None,
) -> list[dict[str, object]]:
    """Return the kind-typed amount payloads for one instrument entry.

    Reads the ``amounts`` list, then rescues each unverified amount whose
    magnitude sits on a sibling fact's span (`scaled_amount_from_sibling`; a
    post-pass because it needs this object's other facts). When no principal
    results but the instrument's name embeds one, a name-derived principal
    entry is synthesized.
    """
    payloads: list[dict[str, object]] = []
    entries = obj.get("amounts")
    if isinstance(entries, list):
        model_amounts: list[object] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            payload = standardized_amount_payload(
                entry,
                tag_details,
                name_text=name_text,
                document_currencies=document_currencies,
            )
            kind = entry.get("kind")
            payload["kind"] = kind if kind in AMOUNT_KINDS else None
            as_of = entry.get("as_of_date")
            payload["as_of_date"] = (
                as_of
                if isinstance(as_of, str)
                and ISO_DATE_PATTERN.fullmatch(as_of)
                and is_valid_iso_date(as_of)
                else None
            )
            # A term stated as it stood before a change is history, not the
            # instrument's current figure.
            payload["prior"] = entry.get("prior") is True
            payloads.append(payload)
            model_amounts.append(entry.get("normalized_amount"))
        for index, payload in enumerate(payloads):
            if payload["normalized_amount"] is not None:
                continue
            siblings = [
                tag_id
                for position, sibling in enumerate(payloads)
                if position != index
                for tag_id in payload_tag_ids(sibling)
            ]
            scaled = scaled_amount_from_sibling(
                payload_tag_ids(payload),
                siblings,
                tag_details,
                model_amounts[index],
            )
            if scaled is not None:
                payload["normalized_amount"] = scaled
                # `currency` is left as it stands: it was read from this fact's
                # own cited span, which is the right evidence for a currency
                # even when the magnitude had to be borrowed.
                payload["derived_from"] = DERIVED_FROM_SCALED
    if not select_principal_amount(payloads):
        synthesized = name_derived_principal_payload(name_text)
        # When every stated commitment or principal is `prior`, the figure in
        # the name is the prior one, not the current one: publish null and let
        # the minted prior state carry it.
        prior_values = {
            str(payload["normalized_amount"])
            for payload in payloads
            if payload.get("prior") is True
            and payload.get("normalized_amount") is not None
        }
        if (
            synthesized is not None
            and str(synthesized.get("normalized_amount")) not in prior_values
        ):
            payloads.append(synthesized)
    return payloads


def name_derived_principal_payload(name_text: str | None) -> dict[str, object] | None:
    """Return a principal payload read off the instrument's own name, or None.

    The parser's reading of the name is the value (there is no model value to
    agree with), marked ``derived_from: "name"``.
    """
    parsed_amount = normalized_amount_from_name(name_text)
    if parsed_amount is None:
        return None
    return {
        "spans": [],
        "normalized_amount": parsed_amount,
        # `currency_from_name` already resolves through `currency_candidates_from_text`,
        # so it is either a supported code or None.
        "currency": currency_from_name(name_text),
        "derived_from": DERIVED_FROM_NAME,
        "kind": "principal",
        "as_of_date": None,
        "prior": False,
    }


def select_principal_amount(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Return the payload that supplies the flat principal columns.

    The first current commitment or principal with a value; ``{}`` when there
    is none. Balances, draws, repayments, and proceeds never become the
    headline amount.
    """
    current = [payload for payload in payloads if not payload.get("prior")]
    for payload in current:
        if (
            payload.get("normalized_amount") is not None
            and payload.get("kind") in PRINCIPAL_AMOUNT_KINDS
        ):
            return payload
    return {}
