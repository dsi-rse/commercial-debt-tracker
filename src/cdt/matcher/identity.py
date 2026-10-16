"""Instrument ids: carried forward by member overlap, minted from content when new.

Clusters are a pure function of one CIK's mentions; their ids are not. A
cluster that shares members with one of the CIK's previous clusters keeps that
cluster's id, so ids survive backfills and re-extraction. A cluster that shares
none gets an id minted from its content. Ids that stop being used get a
redirect to the cluster that took most of their members.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import pandas as pd

from cdt.matcher.normalize import (
    coerce_optional_text,
    mention_sort_key,
    normalize_name_fingerprint,
)
from cdt.matcher.schema import PreparedMention

#: Prefix of every minted instrument id, so an instrument id is never
#: mistaken for a mention id (``dim::``).
INSTRUMENT_ID_PREFIX = "di::"
_DIGEST_CHARS = 24


def mint_instrument_id(cik: str, seed_sort_key: tuple[object, ...]) -> str:
    """Return the content id of a cluster: its CIK and its smallest member sort key."""
    payload = json.dumps([cik, [str(part) for part in seed_sort_key]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return f"{INSTRUMENT_ID_PREFIX}{digest}"


#: Previous-instrument fields compared with a new cluster's members when the
#: two share items but no mentions, as after a re-extraction.
CONTENT_FIELDS = (
    "normalized_name_fingerprint",
    "instrument_type",
    "start_date",
    "maturity_date",
    "interest_rate_pct",
)


@dataclass(frozen=True)
class PreviousClusters:
    """One CIK's clusters as the last match wrote them.

    ``mention_ids`` and ``item_ids`` are each instrument's members;
    ``content`` holds its :data:`CONTENT_FIELDS` values as its row stated them.
    """

    mention_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    item_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    content: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_tables(
        cls: type[PreviousClusters],
        edges: pd.DataFrame | None,
        instruments: pd.DataFrame | None,
    ) -> PreviousClusters:
        """Read one CIK's previous member edges and instrument rows."""
        mention_ids: dict[str, set[str]] = {}
        item_ids: dict[str, set[str]] = {}
        if edges is not None and not edges.empty:
            members = edges.loc[edges["edge_type"] == "member"]
            for row in members.to_dict("records"):
                instrument_id = str(row["debt_instrument_id"])
                mention_ids.setdefault(instrument_id, set()).add(
                    str(row["debt_instrument_mention_id"])
                )
                item_id = row.get("item_id")
                if isinstance(item_id, str) and item_id:
                    item_ids.setdefault(instrument_id, set()).add(item_id)
        content: dict[str, dict[str, str]] = {}
        if instruments is not None and not instruments.empty:
            for row in instruments.to_dict("records"):
                content[str(row["debt_instrument_id"])] = _row_content(row)
        return cls(
            mention_ids={key: frozenset(value) for key, value in mention_ids.items()},
            item_ids={key: frozenset(value) for key, value in item_ids.items()},
            content=content,
        )


def _row_content(row: dict[str, object]) -> dict[str, str]:
    """Return an instrument row's non-null :data:`CONTENT_FIELDS` values."""
    values = {
        "normalized_name_fingerprint": normalize_name_fingerprint(
            coerce_optional_text(row.get("name"))
        ),
        **{
            name: coerce_optional_text(row.get(name))
            for name in CONTENT_FIELDS
            if name != "normalized_name_fingerprint"
        },
    }
    return {name: _comparable(name, value) for name, value in values.items() if value}


def _comparable(name: str, value: str | None) -> str | None:
    """Return a content value in the form both sides compare in."""
    if value is None or name != "interest_rate_pct":
        return value
    try:
        return format(Decimal(value).normalize(), "f")
    except InvalidOperation:
        return value


def _content_agreement(
    members: list[str],
    mention_index: dict[str, PreparedMention],
    content: dict[str, str],
) -> int:
    """Count the previous instrument's content values some member also states."""
    return sum(
        any(
            _comparable(name, getattr(mention_index[member], name)) == value
            for member in members
        )
        for name, value in content.items()
    )


@dataclass(frozen=True)
class _Pair:
    """A new cluster and a previous id that share members, with their evidence."""

    mentions_shared: int
    content_agreed: int
    items_shared: int
    seed_key: tuple[object, ...]
    cluster_id: str
    previous_id: str

    @property
    def evidence(self: _Pair) -> tuple[int, int, int]:
        return (self.mentions_shared, self.content_agreed, self.items_shared)

    @property
    def sort_key(self: _Pair) -> tuple[object, ...]:
        return (
            -self.mentions_shared,
            -self.content_agreed,
            -self.items_shared,
            self.seed_key,
            self.cluster_id,
            self.previous_id,
        )


def _pairs(
    clusters: dict[str, list[str]],
    seed_keys: dict[str, tuple[object, ...]],
    mention_index: dict[str, PreparedMention],
    previous: PreviousClusters,
) -> list[_Pair]:
    """Return every admissible pairing of a cluster with a previous id, best first.

    A pair that shares mentions is always admissible. A pair that shares only
    items is admissible when the cluster states at least one of the previous
    instrument's content values.
    """
    by_mention = {
        mention_id: previous_id
        for previous_id, mention_ids in previous.mention_ids.items()
        for mention_id in mention_ids
    }
    by_item: dict[str, set[str]] = {}
    for previous_id, item_ids in previous.item_ids.items():
        for item_id in item_ids:
            by_item.setdefault(item_id, set()).add(previous_id)
    pairs: list[_Pair] = []
    for cluster_id, members in clusters.items():
        mention_counts: Counter[str] = Counter(
            by_mention[member] for member in members if member in by_mention
        )
        item_counts: Counter[str] = Counter()
        for item_id in {mention_index[member].item_id for member in members}:
            for previous_id in by_item.get(item_id, ()):
                item_counts[previous_id] += 1
        for previous_id in set(mention_counts) | set(item_counts):
            mentions_shared = mention_counts[previous_id]
            content_agreed = (
                0
                if mentions_shared
                else _content_agreement(
                    members, mention_index, previous.content.get(previous_id, {})
                )
            )
            if not mentions_shared and not content_agreed:
                continue
            pairs.append(
                _Pair(
                    mentions_shared=mentions_shared,
                    content_agreed=content_agreed,
                    items_shared=item_counts[previous_id],
                    seed_key=seed_keys[cluster_id],
                    cluster_id=cluster_id,
                    previous_id=previous_id,
                )
            )
    return sorted(pairs, key=lambda pair: pair.sort_key)


def _carry(pairs: list[_Pair]) -> tuple[dict[str, str], set[str]]:
    """Pair clusters with previous ids greedily, best evidence first.

    Returns ``(cluster -> carried id, ambiguous previous ids)``. Pairs that
    share mentions are taken in sort order, each cluster and previous id used
    once. Pairs that share only items are taken a whole evidence level at a
    time: within one level, a cluster or previous id with two or more
    available partners is ambiguous, so neither it nor its partners carry.
    """
    final: dict[str, str] = {}
    carried: set[str] = set()
    ambiguous: set[str] = set()
    blocked: set[str] = set()
    index = 0
    while index < len(pairs):
        level = pairs[index].evidence
        end = index
        while end < len(pairs) and pairs[end].evidence == level:
            end += 1
        available = [
            pair
            for pair in pairs[index:end]
            if pair.cluster_id not in final
            and pair.cluster_id not in blocked
            and pair.previous_id not in carried
            and pair.previous_id not in ambiguous
        ]
        if level[0]:
            for pair in available:
                if pair.cluster_id in final or pair.previous_id in carried:
                    continue
                final[pair.cluster_id] = pair.previous_id
                carried.add(pair.previous_id)
        else:
            by_cluster = Counter(pair.cluster_id for pair in available)
            by_previous = Counter(pair.previous_id for pair in available)
            for pair in available:
                if (
                    by_cluster[pair.cluster_id] == 1
                    and by_previous[pair.previous_id] == 1
                ):
                    final[pair.cluster_id] = pair.previous_id
                    carried.add(pair.previous_id)
                    continue
                ambiguous.add(pair.previous_id)
                blocked.add(pair.cluster_id)
        index = end
    return final, ambiguous


def assign_instrument_ids(
    clusters: dict[str, list[str]],
    *,
    cik: str,
    mention_index: dict[str, PreparedMention],
    previous: PreviousClusters,
    remint: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return final ids for one CIK's clusters, and redirects for ids no longer used.

    ``clusters`` maps a provisional cluster id to its member mention ids.
    Clusters are paired with previous ids by evidence: shared mention ids
    first, then, for a cluster that shares only items with a previous
    instrument (an 8-K item re-extracted, so its mention ids changed), how
    many of that instrument's content values the cluster states, then shared
    item ids, then the earliest seed. An item-only pairing that is tied, two
    clusters for one previous id or two previous ids for one cluster, is
    ambiguous and carries nothing (:func:`_carry`). An unpaired cluster gets
    :func:`mint_instrument_id`, re-hashed in the rare case that collides with
    an id already assigned. ``remint`` pairs nothing, so every cluster gets
    its content id.

    Returns ``(provisional -> final, retired previous id -> successor)``. A
    retired id's successor is its best admissible pair's cluster; an
    ambiguous id, or one with no admissible pair, has no redirect.
    """
    seed_keys = {
        cluster_id: min(mention_sort_key(mention_index[member]) for member in members)
        for cluster_id, members in clusters.items()
    }
    pairs = _pairs(clusters, seed_keys, mention_index, previous)
    final, ambiguous = ({}, set()) if remint else _carry(pairs)
    assigned = set(final.values())
    for cluster_id in sorted(clusters, key=lambda key: seed_keys[key]):
        if cluster_id in final:
            continue
        minted = mint_instrument_id(cik, seed_keys[cluster_id])
        attempt = 0
        while minted in assigned:
            attempt += 1
            minted = mint_instrument_id(cik, (*seed_keys[cluster_id], attempt))
        final[cluster_id] = minted
        assigned.add(minted)

    redirects: dict[str, str] = {}
    for pair in pairs:
        previous_id = pair.previous_id
        if previous_id in assigned or previous_id in redirects:
            continue
        if previous_id in ambiguous:
            continue
        redirects[previous_id] = final[pair.cluster_id]
    return final, redirects


def updated_redirects(
    stored: dict[str, str], retired: dict[str, str], live_ids: set[str]
) -> dict[str, str]:
    """Return one CIK's redirects after a re-cluster: every target a live id.

    ``stored`` are the CIK's redirects before, ``retired`` this run's
    (:func:`assign_instrument_ids`), ``live_ids`` the CIK's ids now. A stored
    redirect whose target retired follows that target's redirect; one whose
    old id is live again, or whose target has no successor, is dropped.
    """
    merged = {**stored, **retired}
    resolved: dict[str, str] = {}
    for old_id, target in merged.items():
        if old_id in live_ids:
            continue
        seen = {old_id}
        while target not in live_ids and target in merged and target not in seen:
            seen.add(target)
            target = merged[target]
        if target in live_ids:
            resolved[old_id] = target
    return resolved
