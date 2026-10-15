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

import pandas as pd

from cdt.matcher.normalize import mention_sort_key
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


@dataclass(frozen=True)
class PreviousClusters:
    """One CIK's clusters as the last match wrote them: members by instrument id."""

    mention_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    item_ids: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls: type[PreviousClusters], edges: pd.DataFrame
    ) -> PreviousClusters:
        """Read the member edges of one CIK's previous clusters."""
        if edges.empty:
            return cls()
        members = edges.loc[edges["edge_type"] == "member"]
        mention_ids: dict[str, set[str]] = {}
        item_ids: dict[str, set[str]] = {}
        for row in members.to_dict("records"):
            instrument_id = str(row["debt_instrument_id"])
            mention_ids.setdefault(instrument_id, set()).add(
                str(row["debt_instrument_mention_id"])
            )
            item_id = row.get("item_id")
            if isinstance(item_id, str) and item_id:
                item_ids.setdefault(instrument_id, set()).add(item_id)
        return cls(
            mention_ids={key: frozenset(value) for key, value in mention_ids.items()},
            item_ids={key: frozenset(value) for key, value in item_ids.items()},
        )


def _overlaps(
    members: list[str],
    mention_index: dict[str, PreparedMention],
    by_mention: dict[str, str],
    by_item: dict[str, set[str]],
) -> list[tuple[int, int, str]]:
    """Return ``(shared mentions, shared items, previous id)`` for each overlapping id."""
    mention_counts: Counter[str] = Counter(
        by_mention[member] for member in members if member in by_mention
    )
    item_counts: Counter[str] = Counter()
    for item_id in {mention_index[member].item_id for member in members}:
        for previous_id in by_item.get(item_id, ()):
            item_counts[previous_id] += 1
    return [
        (mention_counts[previous_id], item_counts[previous_id], previous_id)
        for previous_id in sorted(set(mention_counts) | set(item_counts))
    ]


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
    Clusters are paired with previous ids greedily, largest overlap first:
    shared mention ids, then shared item ids (which survive re-extraction for
    8-K), then the earliest seed, each previous id used at most once. An
    unpaired cluster gets :func:`mint_instrument_id`, re-hashed in the rare
    case that collides with an id already assigned. ``remint`` pairs nothing,
    so every cluster gets its content id.

    Returns ``(provisional -> final, retired previous id -> successor)``. A
    retired id's successor is the cluster that shares most of its members; an
    id that shares none with any cluster has no redirect.
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
    seed_keys = {
        cluster_id: min(mention_sort_key(mention_index[member]) for member in members)
        for cluster_id, members in clusters.items()
    }
    pairs: list[tuple[int, int, tuple[object, ...], str, str]] = []
    for cluster_id, members in clusters.items():
        for mentions_shared, items_shared, previous_id in _overlaps(
            members, mention_index, by_mention, by_item
        ):
            pairs.append(
                (
                    mentions_shared,
                    items_shared,
                    seed_keys[cluster_id],
                    cluster_id,
                    previous_id,
                )
            )
    pairs.sort(key=lambda pair: (-pair[0], -pair[1], pair[2], pair[3], pair[4]))

    final: dict[str, str] = {}
    if not remint:
        carried: set[str] = set()
        for _, _, _, cluster_id, previous_id in pairs:
            if cluster_id in final or previous_id in carried:
                continue
            final[cluster_id] = previous_id
            carried.add(previous_id)
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
    for _, _, _, cluster_id, previous_id in pairs:
        if previous_id in assigned or previous_id in redirects:
            continue
        redirects[previous_id] = final[cluster_id]
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
