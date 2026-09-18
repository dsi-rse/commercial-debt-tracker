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
CIK, so a link the filings encode across filings can be inferred here. One rule,
reasoning only over facts the extractor bound to an object and cited:

* **ordinal_chain** — "Fifth Amended and Restated X" follows "Fourth Amended and
  Restated X" follows "X". The ordinal in the name literally encodes chain
  position within one issuer and name stem.

A second rule, `prior_fact`, linked an instrument carrying a `prior`-marked
amount to the earlier instrument whose principal it equalled. It is gone
because the extractor now mints that predecessor itself, as its own mention,
from the same `prior` marks (#203): the successor's `amendment_of` names it
directly, so the link is an extracted pointer, not an inference — and the
extractor sees the prior *dates* this module never could, since
`PreparedMention` carries no `dates_json`. On the corpus it was measured on,
`prior_fact` produced 1 link in 542.

The rule only ever fills an `amendment_of_debt_instrument_id` that is null or
that this module inferred on an earlier run — never an extracted pointer — and
refuses a candidate whenever the evidence does not single out one parent: an
ambiguous guess is worse than the status quo, because a wrong pointer silently
rewrites a published history. Every inferred pointer is re-opened and
re-derived on each run (`apply_lineage_inference_pass`, #204), so a link the
rule would now refuse does not survive because it was written first.

Stage boundary (#184). Every input here is extractor output: the canonical
`name`, `first_seen_filing_date` and `start_date`, and the borrower the
extractor bound to each row in `parties_json`, which refuses a link between two
issuers' agreements filed under one CIK (#197, #205). This
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
# Legal-form suffixes carry no identity: "EQT" and "EQT Corporation" are one
# borrower. Everything else in a name is part of who it is.
BORROWER_SUFFIXES = frozenset(
    {
        "corporation",
        "corp",
        "incorporated",
        "inc",
        "company",
        "co",
        "llc",
        "lp",
        "llp",
        "plc",
        "ltd",
        "limited",
        "na",
    }
)
# A borrower "named" only by its role or defined term. The extractor records a
# party as the longest span it saw, so a filing that never uses the company's
# name records the borrower as literally `Issuer` — and read as a name, that
# made `HSBC Holdings plc` against `Issuer` positive evidence of two different
# companies (#205). It is silence: the filing did not say who the borrower is.
# Keys are compared after `NAME_NOISE` and `BORROWER_SUFFIXES` are stripped, so
# `the Borrowers` and `Co-Borrower` both land on `borrower(s)` here.
GENERIC_BORROWER_PHRASES = frozenset(
    {
        "borrower",
        "borrowers",
        "subsidiary borrower",
        "subsidiary borrowers",
        "parent borrower",
        "other borrowers party thereto",
        "issuer",
        "issuers",
        "obligor",
        "obligors",
        "buyer",
        "buyers",
        "buyer parent",
        "parent",
        "loan party",
        "loan parties",
        "credit party",
        "credit parties",
    }
)


def _borrower_key(name: object) -> tuple[str, ...]:
    """Return one borrower's canonical name as comparable tokens."""
    return tuple(
        token
        for token in re.sub(r"[^0-9a-z]+", " ", str(name or "").lower()).split()
        if token not in BORROWER_SUFFIXES and token not in NAME_NOISE
    )


def _borrowers(row: dict[str, object]) -> set[tuple[str, ...]]:
    """Return the borrower keys the extractor bound to this instrument.

    A placeholder (`Issuer`, `the Borrowers`) is dropped rather than compared:
    it names a role, not a company, so the row reads as naming no borrower and
    takes the silence path below. Valid JSON that is not a list reads the same
    way instead of raising, as `parse_cluster_list` does for every other payload.
    """
    try:
        parties = json.loads(str(row.get("parties_json") or "[]"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(parties, list):
        return set()
    keys = {
        _borrower_key(party.get("canonical_name"))
        for party in parties
        if isinstance(party, dict) and party.get("role") == "borrower"
    }
    return {
        key for key in keys if key and " ".join(key) not in GENERIC_BORROWER_PHRASES
    }


def _borrowers_disagree(child: dict[str, object], parent: dict[str, object]) -> bool:
    """Return whether two rows name borrowers that cannot be the same party.

    Silence is not disagreement. An instrument with no borrower recorded is
    unconstrained, because refusing on a missing party would drop ordinary
    links to the many mentions that never name one — so this only ever fires on
    positive evidence of a different borrower.

    Two rows agree when they share one borrower key exactly. A prefix rule used
    to count `EQT Corporation` and `EQT Midstream Partners, LP` as one party —
    finance subsidiaries are almost always named after their parent, and the
    EQT/EQM case this guard was written for only worked because `eqt` and `eqm`
    differ in the first token (#205). `BORROWER_SUFFIXES` already makes `EQT` and
    `EQT Corporation` equal, so the prefix bought nothing but that hole. A
    cluster's borrowers are the union across its member mentions, so one shared
    key among several is enough — the guard is deliberately looser on a row
    that names many borrowers than on one that names one.
    """
    child_keys, parent_keys = _borrowers(child), _borrowers(parent)
    if not child_keys or not parent_keys:
        return False
    return not (child_keys & parent_keys)


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
    # The ordinal rule reads instrument rows only. The mention-level inputs
    # served `prior_fact`, now the extractor's job (#203); the parameters stay
    # so the pass and a future mention-reading rule keep one call shape.
    del member_groups, mention_index
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
        # One amendment chain has one borrower. The CIK check above is the
        # *filer's* CIK, which cannot separate two agreements named in one 8-K:
        # EQT's own Third Amended and Restated Credit Agreement and EQM
        # Midstream Partners' both appear in EQT's 2024-07-22 filing, share a
        # name stem and an ordinal, and so were both offered as children of
        # EQT's Second Amended and Restated Credit Agreement — welding an
        # acquired subsidiary's terminated facility into the parent's chain and
        # nulling the real predecessor's `superseded_by` under the two-child
        # ambiguity rule. The extractor recorded both borrowers; this reads them.
        if _borrowers_disagree(child, parent):
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

    # The ordinal chain within one issuer and name stem.
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
            # One restatement can have several published states — the extractor
            # mints an amended instrument's prior state as its own row (#203),
            # and an amendment within a restatement keeps the ordinal. Those
            # states already point at each other, so the one another same-rank
            # state names as its `amendment_of` has been replaced: it steps
            # aside, and the chain lands on the state that replaced it. Two
            # unlinked rows of one rank are still a tie.
            replaced = {
                str(by_id[row_id].get("amendment_of_debt_instrument_id"))
                for row_id in parents
                if by_id[row_id].get("amendment_of_debt_instrument_id")
            }
            parents = [row_id for row_id in parents if row_id not in replaced]
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
        "Lineage inference: %s ordinal_chain links, %s children left alone as ambiguous",
        len(resolved),
        ambiguous,
    )
    return resolved
