"""Mint the predecessor of each amended instrument, and list the rows an item publishes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cdt.extractor.normalize.amounts import select_principal_amount
from cdt.extractor.normalize.dates import derived_status_payload, select_date_payload
from cdt.extractor.schema import (
    BORROWER_PARTY_ROLE,
    DERIVED_FROM_INHERITED,
    LENDER_DISCLOSURE_NONE_NAMED,
    PRINCIPAL_AMOUNT_KINDS,
    debt_instrument_mention_id_for,
)
from cdt.storage.columns import coerce_dataset_text, json_column

if TYPE_CHECKING:
    from cdt.extractor.state import ExtractionRowState


# The rule name a synthesized prior state carries in `synthesized_by`.
SYNTHESIZED_PRIOR_STATE = "prior_state"
# The term kinds a filing can mark `prior` — the list the dates validator names.
PRIOR_TERM_DATE_KINDS = frozenset({"agreement", "maturity", "commitment_termination"})
# Current terms a prior state inherits when the filing marks no `prior` value
# for the kind. Balances, draws, repayments and proceeds are dated observations
# of the filing's own moment, not terms of the instrument, so they stay with the
# state the filing describes.
INHERITED_DATE_KINDS = frozenset({"maturity", "commitment_termination"})


def _json_list(row: dict[str, object], column: str) -> list[dict[str, object]]:
    """Return one JSON-array column as fresh dicts; junk and NaN read as empty."""
    payload = json_column(row, column)
    if not isinstance(payload, list):
        return []
    return [dict(entry) for entry in payload if isinstance(entry, dict)]


def _json_dict(row: dict[str, object], column: str) -> dict[str, object]:
    """Return one JSON-object column as a fresh dict; junk and NaN read as empty."""
    payload = json_column(row, column)
    return dict(payload) if isinstance(payload, dict) else {}


def _inherited(payload: dict[str, object]) -> dict[str, object]:
    """Return a copy of a term carried onto a prior state unchanged, marked so."""
    copy = dict(payload)
    if (
        copy.get("normalized_amount") is not None
        or copy.get("normalized_date") is not None
    ):
        copy["derived_from"] = DERIVED_FROM_INHERITED
    return copy


def mint_prior_state_rows(
    rows: list[dict[str, object]], counters: dict[str, int] | None = None
) -> list[dict[str, object]]:
    """Return a copy of one item's rows plus a predecessor for each amended object.

    For each model-emitted row M carrying a `prior` commitment/principal or
    agreement/maturity/commitment-termination claim, mint a predecessor P
    (`synthesized_by="prior_state"`) and set M's `amendment_of` to P's id. P is
    M with each `prior` term replacing its kind; M's current terms of kinds with
    no `prior` claim are carried over marked `derived_from: "inherited"`;
    parties are M's borrowers only. P's origin date is a `prior` agreement, else
    M's current closing or agreement, dropped if it equals an `amendment` date.

    No P is minted, and a `skipped_*` counter is bumped, when M has two `prior`
    claims of one kind, no `prior` value that parsed, an existing
    `amendment_of`, no origin candidate, or a sibling row that already is P.
    ``counters`` (optional, mutated in place) partition the rows with a `prior`
    claim into `minted`, `minted_shared` (P identical to one already present)
    and `skipped_*`; `minted_no_origin` tags the subset of `minted` with no
    start date and must not be summed with the others.

    Pure: the input rows are not mutated, and there is no clock or model call.
    Fields are coerced so rows read back from parquet (NaN for None) mint the
    same ids. See docs/decisions/extraction.md.
    """
    counts = counters if counters is not None else {}

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    text = coerce_dataset_text
    # Copy first: `amendment_of` is written onto the successor, and the
    # caller's rows (possibly persisted row state) must not change.
    published = [dict(row) for row in rows]
    real_rows = [row for row in published if text(row.get("synthesized_by")) is None]
    known_ids = {text(row.get("debt_instrument_mention_id")) for row in published}
    for row in real_rows:
        amounts = _json_list(row, "amounts_json")
        dates = _json_list(row, "dates_json")
        # Claims (what the filing marks `prior`) decide which kinds changed;
        # only the values come from what parsed. An unparsed before-value still
        # means the current value must not be inherited.
        prior_amount_claims = [
            entry
            for entry in amounts
            if entry.get("prior") is True
            and entry.get("kind") in PRINCIPAL_AMOUNT_KINDS
        ]
        prior_date_claims = [
            entry
            for entry in dates
            if entry.get("prior") is True and entry.get("kind") in PRIOR_TERM_DATE_KINDS
        ]
        prior_amounts = [
            entry
            for entry in prior_amount_claims
            if entry.get("normalized_amount") is not None
        ]
        prior_dates = [
            entry
            for entry in prior_date_claims
            if entry.get("normalized_date") is not None
        ]
        if not prior_amount_claims and not prior_date_claims:
            continue
        prior_kinds = [
            entry.get("kind") for entry in [*prior_amount_claims, *prior_date_claims]
        ]
        if len(prior_kinds) != len(set(prior_kinds)):
            # Two before-values of one kind is two prior states, or a model
            # error; either way the evidence does not describe one predecessor.
            # Judged on the claims: two stated before-figures are ambiguous
            # whether or not both of them parsed.
            bump("skipped_ambiguous_prior")
            continue
        if not prior_amounts and not prior_dates:
            # The filing does state a before-value, but none of them parsed, so
            # there is nothing to build a predecessor's terms out of.
            bump("skipped_unparsed_prior")
            continue
        if text(row.get("amendment_of")) is not None:
            # The relation stage already paired this object with a predecessor
            # the model returned; minting a second one would duplicate it.
            bump("skipped_model_paired")
            continue

        amendment_dates = {
            entry.get("normalized_date")
            for entry in dates
            # `isinstance`, not truthiness: a list here from a hand-edited
            # partition would raise in the set and kill the whole run.
            if entry.get("kind") == "amendment"
            and isinstance(entry.get("normalized_date"), str)
            and entry.get("normalized_date")
        }
        prior_agreement = next(
            (entry for entry in prior_dates if entry.get("kind") == "agreement"), None
        )
        origin_payloads: list[dict[str, object]]
        minted_without_origin = False
        if prior_agreement is not None:
            # "amends and restates the Credit Agreement dated as of X": X is the
            # predecessor's own date. The current closing and agreement are the
            # restated instrument's and would put the wrong start date on P.
            origin_payloads = []
            origin_date = text(prior_agreement.get("normalized_date"))
        else:
            candidates = [
                payload
                for payload in (
                    select_date_payload(dates, "closing"),
                    select_date_payload(dates, "agreement"),
                )
                if payload.get("normalized_date") is not None
            ]
            if not candidates:
                bump("skipped_no_origin")
                continue
            origin_payloads = [
                payload
                for payload in candidates
                if payload.get("normalized_date") not in amendment_dates
            ]
            minted_without_origin = not origin_payloads
            origin_date = (
                text(origin_payloads[0].get("normalized_date"))
                if origin_payloads
                else None
            )

        prior_values = {str(entry["normalized_amount"]) for entry in prior_amounts}
        if prior_values and any(
            sibling is not row
            and text(sibling.get("principal_amount")) in prior_values
            and text(sibling.get("start_date")) in (None, origin_date)
            for sibling in real_rows
        ):
            # The model returned the predecessor as its own object but the
            # relation stage did not link them; the sibling *is* P.
            bump("skipped_sibling_is_predecessor")
            continue

        # From the claims, not the parsed subset: an unresolvable before-figure
        # still says this term changed, so the current one must not be
        # inherited onto the predecessor as though it had not.
        prior_amount_kinds = {entry.get("kind") for entry in prior_amount_claims}
        prior_date_kinds = {entry.get("kind") for entry in prior_date_claims}
        minted_amounts: list[dict[str, object]] = []
        for entry in amounts:
            kind = entry.get("kind")
            if kind not in PRINCIPAL_AMOUNT_KINDS:
                continue
            if entry.get("prior") is True:
                if entry.get("normalized_amount") is None:
                    continue
                flipped = dict(entry)
                flipped["prior"] = False
                minted_amounts.append(flipped)
            elif kind not in prior_amount_kinds:
                minted_amounts.append(_inherited(entry))
        minted_dates: list[dict[str, object]] = []
        for entry in dates:
            kind = entry.get("kind")
            if entry.get("prior") is True:
                if kind in PRIOR_TERM_DATE_KINDS and entry.get("normalized_date"):
                    flipped = dict(entry)
                    flipped["prior"] = False
                    minted_dates.append(flipped)
                continue
            if (
                kind in INHERITED_DATE_KINDS
                and kind not in prior_date_kinds
                and not entry.get("expected")
            ):
                minted_dates.append(_inherited(entry))
        minted_dates.extend(dict(payload) for payload in origin_payloads)

        start_payload = select_date_payload(minted_dates, "closing")
        if start_payload.get("normalized_date") is None:
            start_payload = select_date_payload(minted_dates, "agreement")
        maturity_payload = select_date_payload(minted_dates, "maturity")
        termination_payload = select_date_payload(
            minted_dates, "commitment_termination"
        )
        principal = select_principal_amount(minted_amounts)
        status_payload = derived_status_payload(minted_dates)
        rate_payload = _json_dict(row, "interest_rate_json") or {
            "kind": None,
            "rate_pct": None,
            "spans": [],
            "derived_from": None,
        }
        if rate_payload.get("rate_pct") is not None:
            rate_payload["derived_from"] = DERIVED_FROM_INHERITED
        borrowers = [
            cluster
            for cluster in _json_list(row, "parties_json")
            if cluster.get("role") == BORROWER_PARTY_ROLE
        ]
        item_id = text(row.get("item_id")) or ""
        successor_id = text(row.get("debt_instrument_mention_id"))
        status_date = status_payload.get("status_date")
        minted: dict[str, object] = {
            "item_id": item_id,
            "accession_number": text(row.get("accession_number")),
            "cik": text(row.get("cik")),
            "company_name": text(row.get("company_name")),
            "date": text(row.get("date")),
            "raw_id": f"{text(row.get('raw_id')) or 'i'}-prior",
            "name": text(row.get("name")),
            "instrument_type": text(row.get("instrument_type")),
            "start_date": start_payload.get("normalized_date"),
            "maturity_date": maturity_payload.get("normalized_date"),
            "commitment_termination_date": termination_payload.get("normalized_date"),
            "principal_amount": principal.get("normalized_amount"),
            "principal_currency": principal.get("currency"),
            "principal_amount_kind": principal.get("kind"),
            "interest_rate_kind": rate_payload.get("kind"),
            "interest_rate_pct": rate_payload.get("rate_pct"),
            "status": status_payload.get("status"),
            "status_date": (
                status_date.get("normalized_date")
                if isinstance(status_date, dict)
                else None
            ),
            "amendment_of": None,
            "retired_by_json": "[]",
            "split_of": None,
            "parties_json": json.dumps(borrowers, sort_keys=True),
            "lender_disclosure": LENDER_DISCLOSURE_NONE_NAMED,
            "name_json": text(row.get("name_json")) or "{}",
            "start_date_json": json.dumps(start_payload, sort_keys=True),
            "maturity_date_json": json.dumps(maturity_payload, sort_keys=True),
            "commitment_termination_date_json": json.dumps(
                termination_payload, sort_keys=True
            ),
            "amounts_json": json.dumps(minted_amounts, sort_keys=True),
            "status_json": json.dumps(status_payload, sort_keys=True),
            "interest_rate_json": json.dumps(rate_payload, sort_keys=True),
            "dates_json": json.dumps(minted_dates, sort_keys=True),
            "synthesized_by": SYNTHESIZED_PRIOR_STATE,
            "synthesized_from_mention_id": successor_id,
        }
        minted_id = debt_instrument_mention_id_for(item_id, minted)
        minted["debt_instrument_mention_id"] = minted_id
        # The pointer is assigned before, and independently of, appending P: two
        # sibling successors that differ only in a term P does not carry mint
        # byte-identical rows, and the second must still point at the one row.
        row["amendment_of"] = minted_id
        if minted_id in known_ids:
            bump("minted_shared")
            continue
        known_ids.add(minted_id)
        published.append(minted)
        # Bumped only after the append, so the tag is a true subset of `minted`.
        if minted_without_origin:
            bump("minted_no_origin")
        bump("minted")
    return published


def published_mention_rows(row_state: ExtractionRowState) -> list[dict[str, object]]:
    """Return the mention rows one row state publishes, prior states included.

    Every publish path (live loop, batch finalize, `extract_tables`, the
    `full.jsonl` audit record) goes through here, so the prior-state mint is
    applied identically on every backend. The row state itself is not mutated.
    """
    return mint_prior_state_rows(row_state.debt_instrument_mentions)
