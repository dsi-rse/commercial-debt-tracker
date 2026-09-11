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
CIK. So the missing links can be inferred here from evidence the extractor
already records. Three rules, each conservative and each measured:

* **prior_fact** — an instrument carrying a `prior`-marked amount or date whose
  value equals an earlier instrument's canonical principal or maturity. The
  `prior` mark *is* the predecessor's term, so an exact match on it is strong.
* **ordinal_chain** — "Fifth Amended and Restated X" follows "Fourth Amended and
  Restated X" follows "X". The ordinal in the name literally encodes chain
  position within one issuer and name stem.
* **dated_reference** — a replacement clause naming a predecessor by its
  dated-as-of date ("replaced the previously existing $2.0 billion credit
  agreement, dated as of July 7, 2022"), resolved against the agreement and
  closing dates of the issuer's other clusters. This is the lineage half of
  #167; it needs the item text, so it is skipped when text is unavailable.

All three only ever fill an `amendment_of_debt_instrument_id` that is already
null, never overwrite an extracted pointer, and refuse a candidate when more than
one parent qualifies — an ambiguous guess is worse than the status quo, because
a wrong pointer silently rewrites a published history.
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
# A replacement clause and the predecessor's dated-as-of date in one clause. The
# clause bound matters: across a sentence boundary the date usually belongs to
# the new agreement, not the one it replaces.
DATED_REFERENCE = re.compile(
    r"(replac\w+|refinanc\w+|amend\w+\s+and\s+restat\w+|supersed\w+|previously\s+existing|prior)"
    # A period followed by whitespace ends the sentence; one followed by a digit
    # is a decimal ("$2.0 billion"), which the predecessor's own amount routinely
    # contains. Excluding all periods silently refused those clauses.
    r"(?:[^.;]|\.(?=\d))"
    r"{0,200}?dated\s+as\s+of\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)
MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


def _iso(text: str) -> str | None:
    """Normalize 'July 7, 2022' to '2022-07-07'."""
    match = re.match(r"([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})", text.strip())
    if not match or match.group(1) not in MONTHS:
        return None
    return f"{match.group(3)}-{MONTHS[match.group(1)]:02d}-{int(match.group(2)):02d}"


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


def _prior_values(member_ids: list[str], mention_index: dict) -> set[str]:
    """Return every `prior`-marked amount and date across a cluster's mentions."""
    values: set[str] = set()
    for member_id in member_ids:
        mention = mention_index.get(member_id)
        if mention is None:
            continue
        for column, key in (
            ("amounts_json", "normalized_amount"),
            ("dates_json", "normalized_date"),
        ):
            try:
                payloads = json.loads(getattr(mention, column, None) or "[]")
            except (TypeError, ValueError):
                continue
            for payload in payloads:
                if payload.get("prior") and payload.get(key):
                    values.add(str(payload[key]))
    return values


def _identity_dates(member_ids: list[str], mention_index: dict) -> set[str]:
    """Return the dates by which a cluster could be referred to elsewhere."""
    dates: set[str] = set()
    for member_id in member_ids:
        mention = mention_index.get(member_id)
        if mention is None:
            continue
        if getattr(mention, "start_date", None):
            dates.add(str(mention.start_date))
        try:
            payloads = json.loads(getattr(mention, "dates_json", None) or "[]")
        except (TypeError, ValueError):
            continue
        for payload in payloads:
            if payload.get("kind") in ("agreement", "closing") and payload.get(
                "normalized_date"
            ):
                dates.add(str(payload["normalized_date"]))
    return dates


def infer_amendment_parents(
    rows: list[dict[str, object]],
    *,
    member_groups: dict[str, list[str]],
    mention_index: dict,
    item_texts: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Return {child_id: (parent_id, rule)} for links the three rules support.

    Only instruments whose `amendment_of_debt_instrument_id` is null are
    considered, and a child with more than one candidate parent is left alone.
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
        # the predecessor must not first appear after the state that replaces it
        if (parent.get("first_seen_filing_date") or "") > (
            child.get("first_seen_filing_date") or ""
        ):
            return
        # never point at something that already points here
        if str(parent.get("amendment_of_debt_instrument_id") or "") == child_id:
            return
        candidates[child_id].setdefault(parent_id, rule)

    # Rule 1: prior-marked value equals an earlier instrument's canonical term.
    for child_id in open_children:
        priors = _prior_values(member_groups.get(child_id, []), mention_index)
        if not priors:
            continue
        for other_id in by_cik.get(str(by_id[child_id].get("cik") or ""), []):
            other = by_id[other_id]
            terms = {
                str(other.get(field))
                for field in ("principal_amount", "maturity_date")
                if other.get(field) not in (None, "", "None")
            }
            if priors & terms:
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
        for index in range(1, len(members)):
            rank, _, row_id = members[index]
            prev_rank, _, prev_id = members[index - 1]
            if rank > prev_rank:
                offer(row_id, prev_id, "ordinal_chain")

    # Rule 3: a replacement clause naming the predecessor by dated-as-of date.
    if item_texts:
        identity = {
            row_id: _identity_dates(member_groups.get(row_id, []), mention_index)
            for row_id in by_id
        }
        for child_id in open_children:
            item_ids = {
                getattr(mention_index[m], "item_id", None)
                for m in member_groups.get(child_id, [])
                if m in mention_index
            }
            referenced: set[str] = set()
            for item_id in item_ids:
                for match in DATED_REFERENCE.finditer(item_texts.get(str(item_id), "")):
                    iso = _iso(match.group(2))
                    if iso:
                        referenced.add(iso)
            if not referenced:
                continue
            for other_id in by_cik.get(str(by_id[child_id].get("cik") or ""), []):
                if referenced & identity.get(other_id, set()):
                    offer(child_id, other_id, "dated_reference")

    resolved: dict[str, tuple[str, str]] = {}
    ambiguous = 0
    for child_id, offers in candidates.items():
        if len(offers) == 1:
            parent_id, rule = next(iter(offers.items()))
            resolved[child_id] = (parent_id, rule)
        elif len(offers) > 1:
            ambiguous += 1

    # Refuse any link that would close a cycle, so lineage_family_id stays a DAG
    # and `superseded_by` cannot chase its own tail.
    def parent_of(node: str) -> str | None:
        if node in resolved:
            return resolved[node][0]
        existing = by_id.get(node, {}).get("amendment_of_debt_instrument_id")
        return str(existing) if existing else None

    for child_id in sorted(resolved):
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
            for rule in ("prior_fact", "ordinal_chain", "dated_reference")
        ),
        ambiguous,
    )
    return resolved
