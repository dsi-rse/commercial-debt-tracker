"""Infer amendment lineage the item-scoped relation stage cannot express.

`instrument_relation` resolves `amendment_of` against `raw_id`s from **one item**
(`by_raw_id` in `extractor/core.py`), so a pointer can only ever name another
object in the same filing. A replacement agreement is named in a *later* filing
than the agreement it replaces, so its pointer never exists — and the #155
lifecycle rollup, which derives `superseded_by` and `lineage_family_id` from that
pointer, is starved of input. Measured on a 364-item window: 537 of 542
instruments came out as lineage heads, with EQT publishing three simultaneously
active revolvers (#170).

The matcher, unlike the relation stage, already works across filings within a
CIK, so the missing links can be inferred here. Two rules, both reasoning only
over facts the extractor bound to an object and cited:

* **prior_fact** — an instrument carrying a `prior`-marked amount whose value
  equals an earlier instrument's canonical principal. The `prior` mark *is* the
  predecessor's term, so an exact match on it is strong evidence.
* **ordinal_chain** — "Fifth Amended and Restated X" follows "Fourth Amended and
  Restated X" follows "X". The ordinal in the name literally encodes chain
  position within one issuer and name stem.

Both only ever fill an `amendment_of_debt_instrument_id` that is already null,
never overwrite an extracted pointer, and refuse a candidate whenever the
evidence does not single out one parent — an ambiguous guess is worse than the
status quo, because a wrong pointer silently rewrites a published history.

Stage boundary (#184). Every input here is extractor output: the `prior` marks in
`amounts_json`, the canonical `name`, `principal_amount`, and `start_date`. This
module does **not** read filing text, and must not: a value derived from text in
the matcher carries no evidence span (#154) and cannot acquire one, and item text
is scoped to a document rather than to an object, so a text rule attributes a
document-level observation to every instrument the filing names. A third rule
(`dated_reference`) did read item text and was removed for exactly that reason;
it resolved a replacement clause's "dated as of" date against other clusters and,
on real filings, linked sibling tranches of one agreement and whole enumerated
lists to a single parent. The evidence it needed — which object a clause is about
— exists at extraction time and is discarded before the matcher runs. Recording
it belongs in the extractor as a `governing_agreement` property (#167).
"""

from __future__ import annotations

import json
import logging
import re

LOGGER = logging.getLogger(__name__)

ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
ORDINAL_AR = re.compile(
    r"\b(" + "|".join(ORDINALS) + r")\s+amended\s+and\s+restated\b", re.IGNORECASE
)
BARE_AR = re.compile(r"\bamended\s+and\s+restated\b", re.IGNORECASE)
NAME_NOISE = frozenset({"the", "a", "an", "that", "certain"})
# A stem needs at least two states before an ordinal can order them into a chain.
MIN_CHAIN_MEMBERS = 2


def _name_rank_and_stem(name: object) -> tuple[int, str]:
    """Return the amend-and-restate ordinal and the name with it stripped."""
    text = " ".join(str(name or "").lower().split())
    match = ORDINAL_AR.search(text)
    if match:
        rank, stripped = ORDINALS[match.group(1).lower()], ORDINAL_AR.sub("", text)
    elif BARE_AR.search(text):
        rank, stripped = 1, BARE_AR.sub("", text)
    else:
        rank, stripped = 0, text
    return rank, " ".join(w for w in stripped.split() if w not in NAME_NOISE)


def _prior_amounts(member_ids: list[str], mention_index: dict) -> set[str]:
    """Return every `prior`-marked amount across a cluster's mentions.

    Reads `amounts_json` by direct attribute access rather than `getattr` with a
    default: if `PreparedMention` ever stops carrying the field, that should be a
    loud `AttributeError` and not a silently empty set. `PreparedMention` carries
    no `dates_json`, so prior *dates* are not available here — adding the field
    would let this rule match a predecessor's maturity as well (#170).
    """
    values: set[str] = set()
    for member_id in member_ids:
        mention = mention_index.get(member_id)
        if mention is None:
            continue
        try:
            payloads = json.loads(mention.amounts_json or "[]")
        except (TypeError, ValueError):
            continue
        for payload in payloads:
            if payload.get("prior") and payload.get("normalized_amount"):
                values.add(str(payload["normalized_amount"]))
    return values


def _canonical_date(row: dict[str, object]) -> str | None:
    """Return the date a row's own evidence says it started, when it has one."""
    value = row.get("start_date")
    if value in (None, "", "None") or str(value) == "nan":
        return None
    return str(value)


def infer_amendment_parents(
    rows: list[dict[str, object]],
    *,
    member_groups: dict[str, list[str]],
    mention_index: dict,
) -> dict[str, tuple[str, str]]:
    """Return {child_id: (parent_id, rule)} for links the two rules support.

    Only instruments whose `amendment_of_debt_instrument_id` is null are
    considered, and a child is left alone whenever the evidence does not single
    out one parent.
    """
    by_id = {str(row["debt_instrument_id"]): row for row in rows}
    by_cik: dict[str, list[str]] = {}
    for row_id, row in by_id.items():
        by_cik.setdefault(str(row.get("cik") or ""), []).append(row_id)

    open_children = [
        row_id
        for row_id, row in by_id.items()
        if not row.get("amendment_of_debt_instrument_id")
    ]
    candidates: dict[str, dict[str, str]] = {row_id: {} for row_id in open_children}

    def offer(child_id: str, parent_id: str, rule: str) -> None:
        if child_id == parent_id or child_id not in candidates:
            return
        child, parent = by_id[child_id], by_id[parent_id]
        if str(child.get("cik")) != str(parent.get("cik")):
            return
        # The predecessor must not first appear after the state that replaces it.
        # Equality is allowed and must stay allowed: the predecessor objects the
        # extractor mints to carry a pre-amendment figure are always first seen in
        # the same filing as the successor describing them (#155), the corpus has
        # a left edge, and 8-K Item 1.01 postdates 2004 — so a real predecessor is
        # routinely first heard about no earlier than its successor.
        if (parent.get("first_seen_filing_date") or "") > (
            child.get("first_seen_filing_date") or ""
        ):
            return
        # Where both rows carry their own start date, that is direct evidence of
        # order and outranks the filing-date check above, which cannot separate
        # two instruments named in one filing.
        child_date, parent_date = _canonical_date(child), _canonical_date(parent)
        if child_date and parent_date and parent_date > child_date:
            return
        # never point at something that already points here
        if str(parent.get("amendment_of_debt_instrument_id") or "") == child_id:
            return
        candidates[child_id].setdefault(parent_id, rule)

    # Rule 1: a prior-marked amount equals an earlier instrument's principal.
    for child_id in open_children:
        priors = _prior_amounts(member_groups.get(child_id, []), mention_index)
        if not priors:
            continue
        for other_id in by_cik.get(str(by_id[child_id].get("cik") or ""), []):
            principal = by_id[other_id].get("principal_amount")
            if principal in (None, "", "None") or str(principal) == "nan":
                continue
            if str(principal) in priors:
                offer(child_id, other_id, "prior_fact")

    # Rule 2: the ordinal chain within one issuer and name stem.
    stems: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
    for row_id, row in by_id.items():
        rank, stem = _name_rank_and_stem(row.get("name"))
        stems.setdefault((str(row.get("cik") or ""), stem), []).append(
            (rank, str(row.get("first_seen_filing_date") or ""), row_id)
        )
    for members in stems.values():
        if len(members) < MIN_CHAIN_MEMBERS:
            continue
        members.sort()
        ranks = sorted({rank for rank, _, _ in members})
        for index, rank in enumerate(ranks):
            if index == 0:
                continue
            previous_rank = ranks[index - 1]
            # Offer every member holding the preceding rank, not just the one the
            # sort happens to put first. A tie is genuine ambiguity, and routing
            # it through the per-child guard below refuses the link instead of
            # letting instrument-id order decide a published history.
            parents = [row_id for r, _, row_id in members if r == previous_rank]
            for _, _, child_id in [m for m in members if m[0] == rank]:
                for parent_id in parents:
                    offer(child_id, parent_id, "ordinal_chain")

    resolved: dict[str, tuple[str, str]] = {}
    ambiguous = 0
    for child_id, offers in candidates.items():
        if len(offers) == 1:
            parent_id, rule = next(iter(offers.items()))
            resolved[child_id] = (parent_id, rule)
        elif len(offers) > 1:
            ambiguous += 1

    # A mutual pair means the rules produced both directions, i.e. the evidence
    # does not settle which instrument is the predecessor. Drop BOTH links rather
    # than keeping whichever one iteration order reaches first — an arbitrary
    # survivor is a coin flip written into a published history.
    mutual = {
        child_id
        for child_id, (parent_id, _) in resolved.items()
        if parent_id in resolved and resolved[parent_id][0] == child_id
    }
    for child_id in sorted(mutual):
        parent_id, rule = resolved.pop(child_id)
        LOGGER.info(
            "Lineage inference: dropped %s -> %s (%s), both directions were offered",
            child_id,
            parent_id,
            rule,
        )
        ambiguous += 1

    # Refuse any link that would close a longer cycle, so lineage_family_id stays
    # a DAG and `superseded_by` cannot chase its own tail.
    def parent_of(node: str) -> str | None:
        if node in resolved:
            return resolved[node][0]
        existing = by_id.get(node, {}).get("amendment_of_debt_instrument_id")
        return str(existing) if existing else None

    for child_id in sorted(resolved):
        if child_id not in resolved:
            continue
        parent_id, rule = resolved[child_id]
        seen: set[str] = set()
        node: str | None = parent_id
        while node and node not in seen:
            if node == child_id:
                LOGGER.info(
                    "Lineage inference: dropped %s -> %s (%s), would close a cycle",
                    child_id,
                    parent_id,
                    rule,
                )
                del resolved[child_id]
                break
            seen.add(node)
            node = parent_of(node)

    LOGGER.info(
        "Lineage inference: %s links (%s), %s children left alone as ambiguous",
        len(resolved),
        ", ".join(
            f"{rule}={sum(1 for _, r in resolved.values() if r == rule)}"
            for rule in ("prior_fact", "ordinal_chain")
        ),
        ambiguous,
    )
    return resolved
