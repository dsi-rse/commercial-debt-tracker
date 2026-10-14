"""Roll each mention cluster up into one published debt-instrument row."""

from __future__ import annotations

import json

import pandas as pd

from cdt.matcher.normalize import (
    _json_text,
    aggregate_lender_disclosure,
    coerce_optional_bool,
    coerce_optional_text,
    dedupe_party_clusters,
    first_non_null,
    mention_recency_key,
    mention_sort_key,
    parse_cluster_list,
)
from cdt.matcher.schema import PreparedMention
from cdt.shared import get_logger

LOGGER = get_logger(__name__)


def apply_lifecycle_rollup(
    rows: list[dict[str, object]],
    *,
    member_groups: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
) -> None:
    """Fill the lineage rollup and observation columns of ``rows`` in place.

    `superseded_by_debt_instrument_id` is the one row that amends this one
    (None when zero or several do), `is_lineage_head` is True when none does,
    and `lineage_family_id` is the smallest id in the row's connected component
    over amendment, split and retirement pointers. No lifecycle `status` is
    derived; the publisher does that against an explicit `asOf`.
    """
    rows_by_id = {str(row["debt_instrument_id"]): row for row in rows}
    superseded_by: dict[str, set[str]] = {}
    for row in rows:
        parent = coerce_optional_text(row.get("amendment_of_debt_instrument_id"))
        if parent and parent in rows_by_id:
            superseded_by.setdefault(parent, set()).add(str(row["debt_instrument_id"]))

    # Lineage families: connected components over every lineage pointer kind.
    neighbors: dict[str, set[str]] = {
        str(row["debt_instrument_id"]): set() for row in rows
    }
    for row in rows:
        row_id = str(row["debt_instrument_id"])
        targets = [
            coerce_optional_text(row.get("amendment_of_debt_instrument_id")),
            coerce_optional_text(row.get("split_of_debt_instrument_id")),
        ]
        retired = coerce_optional_text(row.get("retired_by_debt_instrument_ids"))
        if retired:
            targets.extend(json.loads(retired))
        for target in targets:
            if target and target in neighbors:
                neighbors[row_id].add(target)
                neighbors[target].add(row_id)
    family_by_id: dict[str, str] = {}
    for start in sorted(neighbors):
        if start in family_by_id:
            continue
        component = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor in neighbors[node]:
                if neighbor not in component:
                    component.add(neighbor)
                    frontier.append(neighbor)
        family_id = min(component)
        for member in component:
            family_by_id[member] = family_id

    for row in rows:
        row_id = str(row["debt_instrument_id"])
        children = superseded_by.get(row_id, set())
        # Like the parent pointers, an ambiguous inverse publishes nothing.
        row["superseded_by_debt_instrument_id"] = (
            next(iter(children)) if len(children) == 1 else None
        )
        row["lineage_family_id"] = family_by_id.get(row_id, row_id)
        row["is_lineage_head"] = not children

    apply_observation_columns(
        rows, member_groups=member_groups, mention_index=mention_index
    )


def apply_observation_columns(
    rows: list[dict[str, object]],
    *,
    member_groups: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
) -> None:
    """Recompute each row's observation columns in place from its members.

    Sets `first_seen_filing_date`, `last_seen_filing_date`, `mention_count` and
    `document_count` from the member mentions present in ``mention_index``; a
    row with none gets None dates and zero counts.
    """
    for row in rows:
        row_id = str(row["debt_instrument_id"])
        member_ids = [
            member_id
            for member_id in member_groups.get(row_id, [])
            if member_id in mention_index
        ]
        dates = sorted(
            mention_index[member_id].date
            for member_id in member_ids
            if mention_index[member_id].date
        )
        row["first_seen_filing_date"] = dates[0] if dates else None
        row["last_seen_filing_date"] = dates[-1] if dates else None
        row["mention_count"] = len(member_ids)
        row["document_count"] = len(
            {
                mention_index[member_id].accession_number
                for member_id in member_ids
                if mention_index[member_id].accession_number
            }
        )


def derive_parent_links(
    member_groups: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
    mention_to_instrument: dict[str, str],
    *,
    existing_instruments: pd.DataFrame | None = None,
) -> dict[str, dict[str, str | None]]:
    """Return each instrument's amendment, retirement and split pointers.

    Pointers come from member mentions, mapped through
    ``mention_to_instrument``. Two or more amendment (or split) parents publish
    None for that kind; retirers are a JSON list. The existing row's amendment
    pointer is used only when the mentions state none and are not ambiguous,
    and `amendment_inferred_by` is kept only while the pointer is unchanged.
    Amendment cycles are broken (see :func:`break_amendment_cycles`).
    """
    existing_rows = (
        {
            str(row["debt_instrument_id"]): row
            for row in existing_instruments.to_dict("records")
        }
        if existing_instruments is not None and not existing_instruments.empty
        else {}
    )
    parent_links: dict[str, dict[str, str | None]] = {}
    for debt_instrument_id, member_ids in member_groups.items():
        existing_row = existing_rows.get(debt_instrument_id, {})
        amendment_parents = set()
        retired_parents = set()
        split_parents = set()
        existing_amendment = coerce_optional_text(
            existing_row.get("amendment_of_debt_instrument_id")
        )
        existing_retired = coerce_optional_text(
            existing_row.get("retired_by_debt_instrument_ids")
        )
        if existing_retired:
            retired_parents.update(json.loads(existing_retired))
        existing_split = coerce_optional_text(
            existing_row.get("split_of_debt_instrument_id")
        )
        if existing_split:
            split_parents.add(existing_split)
        for member_id in member_ids:
            mention = mention_index.get(member_id)
            if mention is None:
                continue
            if (
                mention.amendment_of in mention_to_instrument
                and mention_to_instrument[mention.amendment_of] != debt_instrument_id
            ):
                amendment_parents.add(mention_to_instrument[mention.amendment_of])
            for retirer in mention.retired_by:
                if (
                    retirer in mention_to_instrument
                    and mention_to_instrument[retirer] != debt_instrument_id
                ):
                    retired_parents.add(mention_to_instrument[retirer])
            if (
                mention.split_of in mention_to_instrument
                and mention_to_instrument[mention.split_of] != debt_instrument_id
            ):
                split_parents.add(mention_to_instrument[mention.split_of])
        # Each pointer kind is judged on its own; ambiguity within a kind
        # publishes nothing for it. Retirers are a list, so they keep them all.
        amendment_is_ambiguous = len(amendment_parents) > 1
        if amendment_is_ambiguous:
            amendment_parents.clear()
        if len(split_parents) > 1:
            split_parents.clear()
        # The carried pointer is a fallback, not a candidate: what the mentions
        # state wins, and an ambiguous extracted set does not fall back. It is
        # what keeps an inferred link alive across a rematch.
        if not amendment_parents and not amendment_is_ambiguous and existing_amendment:
            amendment_parents.add(existing_amendment)
        amendment_parent = next(iter(amendment_parents), None)
        # Provenance travels with the carried pointer and is cleared when the
        # pointer changes.
        inferred_by = coerce_optional_text(existing_row.get("amendment_inferred_by"))
        if amendment_parent != coerce_optional_text(
            existing_row.get("amendment_of_debt_instrument_id")
        ):
            inferred_by = None
        parent_links[debt_instrument_id] = {
            "amendment_of_debt_instrument_id": amendment_parent,
            "amendment_inferred_by": inferred_by,
            "retired_by_debt_instrument_ids": (
                json.dumps(sorted(retired_parents)) if retired_parents else None
            ),
            "split_of_debt_instrument_id": next(iter(split_parents), None),
        }
    first_seen = {
        debt_instrument_id: min(
            (
                mention_sort_key(mention_index[member_id])
                for member_id in member_ids
                if member_id in mention_index
            ),
            default=(),
        )
        for debt_instrument_id, member_ids in member_groups.items()
    }
    break_amendment_cycles(parent_links, first_seen)
    return parent_links


def break_amendment_cycles(
    parent_links: dict[str, dict[str, str | None]],
    first_seen: dict[str, tuple[object, ...]],
) -> None:
    """Drop one amendment pointer per cycle, in place, so every family has a head.

    In each cycle the pointer to the instrument first seen latest
    (``first_seen``, ties broken by id) is dropped, since nothing amends an
    instrument that appeared after it; that instrument becomes the head.
    """
    state: dict[str, str] = {}
    for start in sorted(parent_links):
        path: list[str] = []
        node: str | None = start
        while node is not None and node in parent_links and node not in state:
            state[node] = "on_path"
            path.append(node)
            node = parent_links[node]["amendment_of_debt_instrument_id"]
        if node is not None and state.get(node) == "on_path":
            cycle = path[path.index(node) :]
            newest = max(cycle, key=lambda member: (first_seen.get(member, ()), member))
            child = next(
                member
                for member in cycle
                if parent_links[member]["amendment_of_debt_instrument_id"] == newest
            )
            parent_links[child]["amendment_of_debt_instrument_id"] = None
            parent_links[child]["amendment_inferred_by"] = None
            LOGGER.warning(
                "Amendment cycle %s: dropped %s -> %s, the instrument first seen last",
                " -> ".join([*cycle, node]),
                child,
                newest,
            )
        for member in path:
            state[member] = "done"


def build_debt_instrument_rows(
    member_groups: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
    parent_links: dict[str, dict[str, str | None]],
    *,
    existing_instruments: pd.DataFrame | None = None,
    company_names: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Build persisted debt instrument rows from member groups and lineage."""
    existing_rows = (
        {
            str(row["debt_instrument_id"]): row
            for row in existing_instruments.to_dict("records")
        }
        if existing_instruments is not None and not existing_instruments.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for debt_instrument_id, member_ids in sorted(member_groups.items()):
        existing_row = existing_rows.get(debt_instrument_id, {})
        present_member_ids = [
            member_id for member_id in member_ids if member_id in mention_index
        ]
        ordered_member_ids = sorted(
            present_member_ids,
            key=lambda mention_id: mention_recency_key(mention_index[mention_id]),
            reverse=True,
        )
        # A synthesized member carries its successor's filing date and would win
        # on recency, so model-emitted members decide the canonical values
        # whenever there is one.
        canonical_member_ids = [
            mention_id
            for mention_id in ordered_member_ids
            if mention_index[mention_id].synthesized_by is None
        ] or ordered_member_ids
        if present_member_ids:
            seed_mention = mention_index[
                min(
                    present_member_ids,
                    key=lambda mention_id: mention_sort_key(mention_index[mention_id]),
                )
            ]
            seed_mention_id = (
                coerce_optional_text(
                    existing_row.get("seed_debt_instrument_mention_id")
                )
                or seed_mention.debt_instrument_mention_id
            )
            cik = coerce_optional_text(existing_row.get("cik")) or seed_mention.cik
        else:
            seed_mention = None
            seed_mention_id = coerce_optional_text(
                existing_row.get("seed_debt_instrument_mention_id")
            )
            cik = coerce_optional_text(existing_row.get("cik"))
        if seed_mention_id is None or cik is None:
            continue
        parties_json = json.dumps(
            dedupe_party_clusters(
                [
                    _json_text(existing_row, "parties_json") or "[]",
                    *[
                        mention_index[mention_id].parties_json
                        for mention_id in present_member_ids
                    ],
                ]
            ),
            sort_keys=True,
        )
        lender_disclosure = aggregate_lender_disclosure(
            [
                coerce_optional_text(existing_row.get("lender_disclosure")),
                *[
                    mention_index[mention_id].lender_disclosure
                    for mention_id in present_member_ids
                ],
            ]
        )
        rows.append(
            {
                "debt_instrument_id": debt_instrument_id,
                "cik": cik,
                # Fall back to the filer name any mention for this CIK carries, so
                # one member mention without display metadata cannot blank the page.
                "company_name": first_non_null(
                    canonical_member_ids, mention_index, "company_name"
                )
                or coerce_optional_text(existing_row.get("company_name"))
                or (company_names or {}).get(cik),
                "seed_debt_instrument_mention_id": seed_mention_id,
                "amendment_of_debt_instrument_id": parent_links.get(
                    debt_instrument_id, {}
                ).get("amendment_of_debt_instrument_id"),
                "amendment_inferred_by": parent_links.get(debt_instrument_id, {}).get(
                    "amendment_inferred_by"
                ),
                "retired_by_debt_instrument_ids": parent_links.get(
                    debt_instrument_id, {}
                ).get("retired_by_debt_instrument_ids"),
                "split_of_debt_instrument_id": parent_links.get(
                    debt_instrument_id, {}
                ).get("split_of_debt_instrument_id"),
                **canonical_scalar_fields(
                    canonical_member_ids,
                    mention_index,
                    existing_row,
                    field_name="name",
                    source_column="name_source_mention_id",
                ),
                **canonical_scalar_fields(
                    canonical_member_ids,
                    mention_index,
                    existing_row,
                    field_name="instrument_type",
                    source_column="instrument_type_source_mention_id",
                ),
                **canonical_scalar_fields(
                    canonical_member_ids,
                    mention_index,
                    existing_row,
                    field_name="start_date",
                    source_column="start_date_source_mention_id",
                ),
                **canonical_maturity_fields(
                    canonical_member_ids, mention_index, existing_row
                ),
                **canonical_scalar_fields(
                    canonical_member_ids,
                    mention_index,
                    existing_row,
                    field_name="commitment_termination_date",
                    source_column="commitment_termination_source_mention_id",
                ),
                **principal_amount_fields(
                    canonical_member_ids, mention_index, existing_row
                ),
                **outstanding_balance_fields(
                    canonical_member_ids, mention_index, existing_row
                ),
                **interest_rate_fields(
                    canonical_member_ids, mention_index, existing_row
                ),
                "parties_json": parties_json,
                "lender_disclosure": lender_disclosure,
                # Every member synthesized: a minted prior state no filing
                # describes on its own. Carried forward when this run loaded no
                # member for the row, like every other canonical field.
                "synthesized_only": (
                    all(
                        mention_index[mention_id].synthesized_by is not None
                        for mention_id in present_member_ids
                    )
                    if present_member_ids
                    else coerce_optional_bool(existing_row.get("synthesized_only"))
                ),
            }
        )
    return rows


def company_names_by_cik(mention_rows: pd.DataFrame) -> dict[str, str]:
    """Return the newest known filer display name for each CIK."""
    if mention_rows.empty or "company_name" not in mention_rows.columns:
        return {}
    newest: dict[str, tuple[tuple[str, str], str]] = {}
    for row in mention_rows.to_dict("records"):
        cik = coerce_optional_text(row.get("cik"))
        company_name = coerce_optional_text(row.get("company_name"))
        if cik is None or company_name is None:
            continue
        recency = (
            str(row.get("date") or ""),
            str(row.get("accession_number") or ""),
        )
        current = newest.get(cik)
        if current is None or recency > current[0]:
            newest[cik] = (recency, company_name)
    return {cik: company_name for cik, (_recency, company_name) in newest.items()}


def canonical_scalar_fields(
    ordered_member_ids: list[str],
    mention_index: dict[str, PreparedMention],
    existing_row: dict[str, object],
    *,
    field_name: str,
    source_column: str,
    existing_keys: tuple[str, ...] | None = None,
) -> dict[str, str | None]:
    """Return one canonical field and the mention it came from.

    The first non-null value across ``ordered_member_ids`` wins. Otherwise the
    first non-null of ``existing_keys`` (default: the field) on the existing
    row is carried with that row's recorded source; both None when neither has
    one.
    """
    for mention_id in ordered_member_ids:
        value = getattr(mention_index[mention_id], field_name)
        if value is not None:
            return {field_name: value, source_column: mention_id}
    for key in existing_keys or (field_name,):
        value = coerce_optional_text(existing_row.get(key))
        if value is not None:
            return {
                field_name: value,
                source_column: coerce_optional_text(existing_row.get(source_column)),
            }
    return {field_name: None, source_column: None}


def canonical_maturity_fields(
    ordered_member_ids: list[str],
    mention_index: dict[str, PreparedMention],
    existing_row: dict[str, object],
) -> dict[str, str | None]:
    """Return the canonical maturity and its source mention.

    The newest stated maturity wins. A derived one (`DERIVED_MATURITY_KINDS`)
    is used only when no member states one, and the existing row's value only
    when no member carries any.
    """
    fallback: dict[str, str | None] | None = None
    for mention_id in ordered_member_ids:
        mention = mention_index[mention_id]
        if mention.maturity_date is None:
            continue
        fields = {
            "maturity_date": mention.maturity_date,
            "maturity_source_mention_id": mention_id,
        }
        if not mention.maturity_is_derived:
            return fields
        if fallback is None:
            fallback = fields
    if fallback is not None:
        return fallback
    value = coerce_optional_text(existing_row.get("maturity_date"))
    return {
        "maturity_date": value,
        "maturity_source_mention_id": (
            coerce_optional_text(existing_row.get("maturity_source_mention_id"))
            if value is not None
            else None
        ),
    }


def principal_amount_fields(
    ordered_member_ids: list[str],
    mention_index: dict[str, PreparedMention],
    existing_row: dict[str, object],
) -> dict[str, str | None]:
    """Return the canonical principal columns from the newest carrying mention.

    Currency and kind come from the same mention as the amount, so an older
    mention's currency never relabels a newer amount.
    """
    for mention_id in ordered_member_ids:
        mention = mention_index[mention_id]
        if mention.principal_amount is not None:
            return {
                "principal_amount": mention.principal_amount,
                "principal_currency": mention.principal_currency,
                "principal_amount_kind": mention.principal_amount_kind,
                "principal_source_mention_id": mention_id,
            }
    return {
        "principal_amount": coerce_optional_text(existing_row.get("principal_amount")),
        "principal_currency": coerce_optional_text(
            existing_row.get("principal_currency")
        ),
        "principal_amount_kind": coerce_optional_text(
            existing_row.get("principal_amount_kind")
        ),
        "principal_source_mention_id": coerce_optional_text(
            existing_row.get("principal_source_mention_id")
        ),
    }


def outstanding_balance_fields(
    ordered_member_ids: list[str],
    mention_index: dict[str, PreparedMention],
    existing_row: dict[str, object],
) -> dict[str, str | bool | None]:
    """Return the newest outstanding-balance observation.

    Kept apart from principal so a balance never doubles as the headline
    amount. An undated balance takes its mention's filing date as
    `outstanding_balance_as_of`, and `outstanding_balance_as_of_is_filing_date`
    records that substitution.
    """
    for mention_id in ordered_member_ids:
        mention = mention_index[mention_id]
        for entry in parse_cluster_list(mention.amounts_json):
            if (
                entry.get("kind") == "outstanding_balance"
                and entry.get("normalized_amount") is not None
            ):
                as_of = entry.get("as_of_date")
                return {
                    "outstanding_balance": str(entry["normalized_amount"]),
                    "outstanding_balance_currency": (
                        str(entry["currency"])
                        if entry.get("currency") is not None
                        else None
                    ),
                    # The mention's filing date bounds an undated balance.
                    "outstanding_balance_as_of": (
                        str(as_of) if as_of is not None else mention.date
                    ),
                    "outstanding_balance_as_of_is_filing_date": as_of is None,
                    "outstanding_balance_source_mention_id": mention_id,
                }
    return {
        "outstanding_balance": coerce_optional_text(
            existing_row.get("outstanding_balance")
        ),
        "outstanding_balance_currency": coerce_optional_text(
            existing_row.get("outstanding_balance_currency")
        ),
        "outstanding_balance_as_of": coerce_optional_text(
            existing_row.get("outstanding_balance_as_of")
        ),
        "outstanding_balance_as_of_is_filing_date": coerce_optional_bool(
            existing_row.get("outstanding_balance_as_of_is_filing_date")
        ),
        "outstanding_balance_source_mention_id": coerce_optional_text(
            existing_row.get("outstanding_balance_source_mention_id")
        ),
    }


def interest_rate_fields(
    ordered_member_ids: list[str],
    mention_index: dict[str, PreparedMention],
    existing_row: dict[str, object],
) -> dict[str, str | None]:
    """Return the canonical interest rate from the newest carrying mention."""
    for mention_id in ordered_member_ids:
        mention = mention_index[mention_id]
        if mention.interest_rate_kind is not None or (
            mention.interest_rate_pct is not None
        ):
            return {
                "interest_rate_kind": mention.interest_rate_kind,
                "interest_rate_pct": mention.interest_rate_pct,
                "interest_rate_source_mention_id": mention_id,
            }
    return {
        "interest_rate_kind": coerce_optional_text(
            existing_row.get("interest_rate_kind")
        ),
        "interest_rate_pct": coerce_optional_text(
            existing_row.get("interest_rate_pct")
        ),
        "interest_rate_source_mention_id": coerce_optional_text(
            existing_row.get("interest_rate_source_mention_id")
        ),
    }
