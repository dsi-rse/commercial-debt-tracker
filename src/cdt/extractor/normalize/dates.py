"""Turn cited date text into checked date facts, published dates and a derived status."""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, cast

from cdt.extractor.schema import (
    DATE_KINDS,
    DAY_FIRST_DATE_PATTERN,
    DERIVED_FROM_COMPUTED,
    DERIVED_FROM_NAME,
    DERIVED_FROM_STATED,
    EVENT_DATE_KINDS,
    EVENT_KIND_PRECEDENCE,
    FOUR_DIGIT_YEAR_PATTERN,
    ISO_DATE_PATTERN,
    LENIENT_ISO_DATE_PATTERN,
    MATURITY_FULL_DATE_PATTERN,
    MATURITY_MONTH_YEAR_PATTERN,
    MATURITY_YEAR_PATTERN,
    MONTH_FIRST_DATE_PATTERN,
    MONTH_MAP,
    MONTH_YEAR_DATE_PATTERN,
    NUMERIC_DATE_PATTERN,
    STATUS_FOR_DATE_KIND,
    TENOR_PATTERN,
    TENOR_WORD_NUMBERS,
    TERMINAL_DATE_KINDS,
    YEAR_ONLY_MATURITY_SUFFIX,
)
from cdt.extractor.tags import (
    cluster_payload,
    cluster_span_texts,
    single_value_evidence_tag_ids,
)


def is_valid_iso_date(value: str) -> bool:
    """Return whether one ISO date string is valid."""
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def normalized_date_from_text(text: str | None) -> str | None:
    """Parse one date mention into ISO format, or None.

    Reads lenient ISO (`2026-7-28`), `M/D/YYYY`, `July 28, 2026` and
    `28 July 2026`; `standardized_date_payload` keeps the model's date only when
    it matches this reading. Two-digit years stay unparsed: a null is better
    than a wrong decade.
    """
    if not text:
        return None
    # `March 5 , 2026` — a stray space before the comma is a formatting artefact.
    stripped = re.sub(r"\s+,", ",", text.strip())
    iso = LENIENT_ISO_DATE_PATTERN.fullmatch(stripped)
    if iso is not None:
        candidate = iso_date_from_numeric_parts(
            iso.group("year"), iso.group("month"), iso.group("day")
        )
        if candidate is not None:
            return candidate
    numeric = NUMERIC_DATE_PATTERN.search(stripped)
    if numeric is not None:
        candidate = iso_date_from_numeric_parts(
            numeric.group("year"), numeric.group("month"), numeric.group("day")
        )
        if candidate is not None:
            return candidate
    for pattern in (MONTH_FIRST_DATE_PATTERN, DAY_FIRST_DATE_PATTERN):
        match = pattern.search(stripped)
        if match is None:
            continue
        candidate = iso_date_from_parts(
            match.group("year"), match.group("month"), match.group("day")
        )
        if candidate is not None:
            return candidate
    return None


def normalized_month_year_from_text(text: str | None) -> str | None:
    """Parse a month-resolution date such as `in March 2056` to the month's last day.

    Used for maturities only: `matures in June 2016` states a maturity as
    precisely as the filing ever will, while a start or status date at month
    resolution would be a guess. A span naming two months returns None.
    """
    if not text:
        return None
    found: set[str] = set()
    for match in MONTH_YEAR_DATE_PATTERN.finditer(text):
        normalized = iso_month_end_from_parts(match.group("year"), match.group("month"))
        if normalized is not None:
            found.add(normalized)
    return next(iter(found)) if len(found) == 1 else None


def dates_agree(model_date: object, parsed_date: str | None) -> bool:
    """Return whether the model's date is the same day the parser read.

    Compared as dates rather than as strings, so a model writing `2026-7-28`
    agrees with a parsed `2026-07-28`. False when either side is missing.
    """
    if not isinstance(model_date, str) or parsed_date is None:
        return False
    normalized = normalized_date_from_text(model_date)
    return normalized is not None and normalized == parsed_date


def normalized_maturity_from_text(text: str | None) -> str | None:
    """Parse one maturity phrase such as 'notes due 2028' into ISO format.

    A phrase stating more than one maturity names no single instrument, so it
    parses to None rather than to whichever maturity comes first.
    """
    if not text:
        return None
    full_dates: set[str] = set()
    month_dates: set[str] = set()
    years: set[str] = set()
    for match in MATURITY_FULL_DATE_PATTERN.finditer(text):
        normalized = iso_date_from_parts(
            match.group("year"), match.group("month"), match.group("day")
        )
        if normalized is not None:
            full_dates.add(normalized)
        years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("more")))
    for match in MATURITY_MONTH_YEAR_PATTERN.finditer(text):
        normalized = iso_month_end_from_parts(match.group("year"), match.group("month"))
        if normalized is not None:
            month_dates.add(normalized)
            years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("more")))
    for match in MATURITY_YEAR_PATTERN.finditer(text):
        years.update(FOUR_DIGIT_YEAR_PATTERN.findall(match.group("years")))
    if full_dates:
        # A bare alternate year alongside a full date states a second maturity
        # too, and so does a month-year phrase for a different month.
        if len(full_dates) != 1 or years - {value[:4] for value in full_dates}:
            return None
        full_date = full_dates.pop()
        if any(value[:7] != full_date[:7] for value in month_dates):
            return None
        return full_date
    if month_dates:
        if len(month_dates) != 1:
            return None
        month_date = month_dates.pop()
        if years - {month_date[:4]}:
            return None
        return month_date
    if len(years) != 1:
        return None
    return f"{years.pop()}{YEAR_ONLY_MATURITY_SUFFIX}"


def iso_date_from_parts(year: str, month_name: str, day: str) -> str | None:
    """Return one ISO date built from year, month name, and day parts."""
    month = MONTH_MAP.get(month_name.lower())
    if month is None:
        return None
    normalized = f"{year}-{month}-{int(day):02d}"
    return normalized if is_valid_iso_date(normalized) else None


def iso_month_end_from_parts(year: str, month_name: str) -> str | None:
    """Return the last day of one month-year maturity such as `due April 2033`.

    None for an unknown month name.
    """
    month = MONTH_MAP.get(month_name.lower())
    if month is None:
        return None
    last_day = calendar.monthrange(int(year), int(month))[1]
    normalized = f"{year}-{month}-{last_day:02d}"
    return normalized if is_valid_iso_date(normalized) else None


def tenor_from_text(text: str | None) -> tuple[int, str] | None:
    """Parse one duration span such as `five-year` into `(number, unit)`.

    None when the span states no tenor or more than one distinct tenor.
    """
    if not text:
        return None
    tenors: set[tuple[int, str]] = set()
    for match in TENOR_PATTERN.finditer(text):
        num_text = match.group("num").lower()
        number = (
            TENOR_WORD_NUMBERS[num_text]
            if num_text in TENOR_WORD_NUMBERS
            else int(num_text)
        )
        tenors.add((number, match.group("unit").lower()))
    if len(tenors) != 1:
        return None
    return tenors.pop()


def date_plus_tenor(start: str, tenor: tuple[int, str], *, sign: int = 1) -> str | None:
    """Return start moved by one tenor (back when ``sign`` is -1), clamping to month ends."""
    try:
        anchor = date.fromisoformat(start)
    except ValueError:
        return None
    number, unit = tenor
    number *= sign
    # A tenor that lands outside the representable calendar is not an answer;
    # raising would unwind out of postprocess and kill the whole run.
    try:
        if unit == "day":
            return (anchor + timedelta(days=number)).isoformat()
        months = number * 12 if unit == "year" else number
        total = anchor.month - 1 + months
        year = anchor.year + total // 12
        month = total % 12 + 1
        day = min(anchor.day, calendar.monthrange(year, month)[1])
        return date(year, month, day).isoformat()
    except (ValueError, OverflowError):
        return None


def computed_maturity_date(
    tag_ids: object,
    tag_details: dict[str, dict[str, object]],
    model_date: object,
) -> str | None:
    """Return the model's maturity when it equals a cited date plus or minus a tenor.

    A filing that states a facility's closing date and its tenor but never the
    maturity supports exactly one arithmetic answer. The model must cite both
    the `date` span and the `duration` span; its normalized date publishes only
    when some cited date plus some cited tenor lands on it, and it carries
    ``derived_from: "computed"``.
    """
    if (
        not isinstance(model_date, str)
        or not ISO_DATE_PATTERN.fullmatch(model_date)
        or not is_valid_iso_date(model_date)
    ):
        return None
    if not isinstance(tag_ids, list):
        return None
    starts: set[str] = set()
    tenors: set[tuple[int, str]] = set()
    for tag_id in tag_ids:
        detail = tag_details.get(tag_id) if isinstance(tag_id, str) else None
        if detail is None:
            continue
        text = str(detail["text"])
        if detail["type"] == "date":
            parsed = normalized_date_from_text(text)
            if parsed is not None:
                starts.add(parsed)
        elif detail["type"] == "duration":
            tenor = tenor_from_text(text)
            if tenor is not None:
                tenors.add(tenor)
    # `extended six months to September 3, 2027` states the prior maturity as
    # the new one minus the tenor, so the arithmetic runs both ways; each
    # direction still lands only on a date the cited spans determine exactly.
    for start in starts:
        for tenor in tenors:
            if model_date in (
                date_plus_tenor(start, tenor),
                date_plus_tenor(start, tenor, sign=-1),
            ):
                return model_date
    return None


def iso_date_from_numeric_parts(year: str, month: str, day: str) -> str | None:
    """Return one ISO date built from numeric month-first parts.

    Month-first, as SEC filings write it. An out-of-range month is rejected
    rather than swapped with the day: a span that means `28/07/2026` is more
    likely a format this parser should not be guessing at than a transposition.
    """
    normalized = f"{year}-{int(month):02d}-{int(day):02d}"
    return normalized if is_valid_iso_date(normalized) else None


def standardized_date_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    allow_maturity_phrase: bool = False,
) -> dict[str, object]:
    """Return evidence payload plus validated normalized date field."""
    evidence_tag_ids = single_value_evidence_tag_ids(value)
    payload = cluster_payload(evidence_tag_ids, tag_details)
    # Every cited span is a candidate reading, so a longer defined term such as
    # `Redemption Date` cited beside `March 2, 2026` does not hide the date.
    span_texts = cluster_span_texts(evidence_tag_ids, tag_details)
    parsed_date = parsed_date_from_spans(span_texts, normalized_date_from_text)
    derived_from = DERIVED_FROM_STATED if parsed_date is not None else None
    if parsed_date is None and allow_maturity_phrase:
        # A stated month-resolution maturity (`in March 2056`) is still stated.
        parsed_date = parsed_date_from_spans(
            span_texts, normalized_month_year_from_text
        )
        derived_from = DERIVED_FROM_STATED if parsed_date is not None else None
    if parsed_date is None and allow_maturity_phrase:
        # A maturity phrase lives inside the instrument's own name span, so a
        # value parsed this way is name-derived even though evidence is cited.
        parsed_date = parsed_date_from_spans(span_texts, normalized_maturity_from_text)
        derived_from = DERIVED_FROM_NAME if parsed_date is not None else None
    model_date = value.get("normalized_date") if isinstance(value, dict) else None
    # The parser's own string is published, so a model writing the same day in a
    # different shape keeps its value.
    payload["normalized_date"] = (
        parsed_date if dates_agree(model_date, parsed_date) else None
    )
    payload["derived_from"] = (
        derived_from if payload["normalized_date"] is not None else None
    )
    return payload


def parsed_date_from_spans(
    span_texts: list[str],
    parser: Callable[[str | None], str | None],
) -> str | None:
    """Return the one date the cited spans state, or None when they disagree."""
    parsed = {value for value in (parser(text) for text in span_texts) if value}
    return next(iter(parsed)) if len(parsed) == 1 else None


def instrument_date_entries(obj: dict[str, Any]) -> list[tuple[str, object, bool]]:
    """Return (kind, value, prior) date entries for one instrument object."""
    dates = obj.get("dates")
    return [
        (str(entry["kind"]), entry, entry.get("prior") is True)
        for entry in (dates if isinstance(dates, list) else [])
        if isinstance(entry, dict) and entry.get("kind") in DATE_KINDS
    ]


def date_precision(
    normalized_date: str | None,
    span_texts: list[str],
    derived_from: str | None,
) -> str | None:
    """Return day / month / year for how precisely the cited text states the date."""
    if normalized_date is None:
        return None
    if derived_from == DERIVED_FROM_COMPUTED:
        return "day"
    if any(normalized_date_from_text(text) == normalized_date for text in span_texts):
        return "day"
    if any(
        normalized_month_year_from_text(text) == normalized_date for text in span_texts
    ) or any(
        MATURITY_MONTH_YEAR_PATTERN.search(text) is not None for text in span_texts
    ):
        return "month"
    if normalized_date.endswith(YEAR_ONLY_MATURITY_SUFFIX):
        return "year"
    return "day"


def standardized_dates_payloads(
    obj: dict[str, Any],
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
) -> list[dict[str, object]]:
    """Return the kind-typed date payloads for one instrument entry.

    Each payload carries ``kind``, ``prior``, ``precision`` and the evidence
    fields of a single-value date payload. A maturity falls back to computed
    and name-derived values (`standardized_end_date_payload`); when the object
    states no maturity at all but its name embeds one, that name-derived
    maturity is synthesized.
    """
    payloads: list[dict[str, object]] = []
    saw_maturity = False
    for kind, value, prior in instrument_date_entries(obj):
        if kind == "maturity":
            saw_maturity = True
            payload = standardized_end_date_payload(
                value, tag_details, name_text=name_text
            )
        else:
            payload = standardized_date_payload(value, tag_details)
        payload["expected"] = isinstance(value, dict) and value.get("expected") is True
        payload["kind"] = kind
        payload["prior"] = prior
        payload["precision"] = date_precision(
            cast(str | None, payload.get("normalized_date")),
            cluster_span_texts(single_value_evidence_tag_ids(value), tag_details)
            + ([name_text] if name_text else []),
            cast(str | None, payload.get("derived_from")),
        )
        payloads.append(payload)
    if not saw_maturity:
        synthesized = standardized_end_date_payload(
            None, tag_details, name_text=name_text
        )
        if synthesized.get("normalized_date") is not None:
            synthesized["kind"] = "maturity"
            synthesized["prior"] = False
            synthesized["expected"] = False
            synthesized["precision"] = date_precision(
                cast(str | None, synthesized.get("normalized_date")),
                [name_text] if name_text else [],
                cast(str | None, synthesized.get("derived_from")),
            )
            payloads.append(synthesized)
    return payloads


def select_date_payload(
    payloads: list[dict[str, object]], kind: str
) -> dict[str, object]:
    """Return the current (non-prior, not merely expected) payload of one kind."""
    for payload in payloads:
        if (
            payload.get("kind") == kind
            and not payload.get("prior")
            and not payload.get("expected")
        ):
            return payload
    return {"spans": [], "normalized_date": None, "derived_from": None}


def mark_post_filing_events_expected(
    date_payloads: list[dict[str, object]], filing_date: str
) -> None:
    """Mark, in place, every non-announcement event dated after the filing as expected.

    `Interest on the Notes will accrue from April 6, 2022` in a March 25 pricing
    8-K reads as a completed closing to the model; the calendar says otherwise.
    No-op when ``filing_date`` is empty.
    """
    if not filing_date:
        return
    for payload in date_payloads:
        value = payload.get("normalized_date")
        if (
            payload.get("kind") in EVENT_DATE_KINDS
            and payload.get("kind") != "announcement"
            and isinstance(value, str)
            and value > filing_date
        ):
            payload["expected"] = True


def derived_status_payload(
    date_payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Derive what this mention says happened from its dated event facts.

    The newest completed (not `expected`, not `prior`) event wins; undated
    events sort with the newest dated one and ties resolve by how much the
    event says about the obligation's life. A planned retirement or a planned
    closing decides nothing here — an instrument whose only closing is expected
    is `announced`, and a pending retirement is left to the matcher, which reads
    the expected facts off `dates_json`.
    """
    completed = [
        payload
        for payload in date_payloads
        if payload.get("kind") in EVENT_DATE_KINDS
        and not payload.get("prior")
        and not payload.get("expected")
        and payload.get("kind") != "repayment"
    ]
    if not completed:
        expected_closing = [
            payload
            for payload in date_payloads
            if payload.get("kind") == "closing" and payload.get("expected")
        ]
        if expected_closing:
            return {
                "status": "announced",
                "status_date": {
                    "spans": [],
                    "normalized_date": None,
                    "derived_from": None,
                },
                "derived_from_kind": "closing:expected",
            }
        agreement = [
            payload
            for payload in date_payloads
            if payload.get("kind") == "agreement" and not payload.get("prior")
        ]
        if agreement:
            # A dated agreement with no other event is an instrument that was
            # entered into on that date.
            return {
                "status": "entered_into",
                "status_date": {
                    key: agreement[0].get(key)
                    for key in ("spans", "normalized_date", "derived_from")
                },
                "derived_from_kind": "agreement",
            }
        return {"status": None, "status_date": None, "derived_from_kind": None}
    newest_dated = max(
        (
            str(payload["normalized_date"])
            for payload in completed
            if payload.get("normalized_date")
        ),
        default=None,
    )

    def rank(payload: dict[str, object]) -> tuple[str, int]:
        value = payload.get("normalized_date")
        return (
            str(value) if value else (newest_dated or ""),
            EVENT_KIND_PRECEDENCE.get(str(payload.get("kind")), 0),
        )

    winner = max(completed, key=rank)
    kind = str(winner["kind"])
    return {
        "status": STATUS_FOR_DATE_KIND[kind],
        "status_date": {
            key: winner.get(key) for key in ("spans", "normalized_date", "derived_from")
        },
        "derived_from_kind": kind,
    }


def expected_retirement_in_payloads(date_payloads: list[dict[str, object]]) -> bool:
    """Return whether the mention states a planned retirement of the obligation."""
    return any(
        payload.get("kind") in TERMINAL_DATE_KINDS and payload.get("expected")
        for payload in date_payloads
    )


def standardized_end_date_payload(
    value: object,
    tag_details: dict[str, dict[str, object]],
    *,
    name_text: str | None,
) -> dict[str, object]:
    """Return the end-date payload, falling back to the maturity in the name."""
    payload = standardized_date_payload(
        value,
        tag_details,
        allow_maturity_phrase=True,
    )
    if payload["normalized_date"] is None:
        model_date = value.get("normalized_date") if isinstance(value, dict) else None
        computed = computed_maturity_date(
            single_value_evidence_tag_ids(value),
            tag_details,
            model_date,
        )
        if computed is not None:
            payload["normalized_date"] = computed
            payload["derived_from"] = DERIVED_FROM_COMPUTED
    if payload["normalized_date"] is None:
        derived_date = normalized_maturity_from_text(name_text)
        if derived_date is not None:
            payload["normalized_date"] = derived_date
            payload["derived_from"] = DERIVED_FROM_NAME
    return payload
