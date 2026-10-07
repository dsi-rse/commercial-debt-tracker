"""Infer amendment lineage the item-scoped relation stage cannot express.

The extractor resolves `amendment_of` only within one item, so a replacement
agreement named in a later filing than the one it replaces carries no pointer.
This module infers such links across filings within one CIK, with one rule:

* **ordinal_chain** — "Fifth Amended and Restated X" follows "Fourth Amended and
  Restated X" follows "X", within one issuer and name stem.

Only a null `amendment_of_debt_instrument_id` is filled, and a child is left
unlinked whenever the evidence does not single out one parent. Every input is
extractor output (canonical `name`, `first_seen_filing_date`, `start_date`, and
the borrowers in `parties_json`); filing text is never read.

`infer_amendment_parents` is the rule over instrument rows.
`apply_lineage_inference_pass` runs it across the whole corpus after every shard
has matched, rewriting the debt-instruments shards and writing an
`infer-lineage` run manifest. Rationale and measurements:
docs/decisions/matching-and-lineage.md.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from cdt.datasets import (
    dataset_root,
    resolve_artifact_root,
    run_manifest_path,
    shard_for_cik,
)
from cdt.extractor.outputs import MENTIONS_DATASET_NAME
from cdt.extractor.schema import (
    DEBT_INSTRUMENT_MENTION_COLUMNS as EXTRACTED_MENTION_COLUMNS,
)
from cdt.matcher.instruments import apply_lifecycle_rollup, apply_observation_columns
from cdt.matcher.normalize import _json_list, coerce_optional_text, prepare_mention
from cdt.matcher.schema import (
    DEBT_INSTRUMENT_COLUMNS,
    MATCHER_SCHEMA_VERSION,
    debt_instruments_root,
    mention_cluster_edges_root,
)
from cdt.storage.objects import write_json_artifact
from cdt.storage.tables import read_dataset, write_partition_table

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
# A borrower "named" only by its role or defined term, which reads as naming no
# borrower. Keys are compared after `NAME_NOISE` and `BORROWER_SUFFIXES` are
# stripped, so `the Borrowers` and `Co-Borrower` both land on `borrower(s)`.
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

    Placeholders in `GENERIC_BORROWER_PHRASES` are dropped. Absent, unparseable
    or non-list `parties_json` returns the empty set, meaning no borrower named.
    """
    keys = {
        _borrower_key(party.get("canonical_name"))
        for party in _json_list(row, "parties_json")
        if isinstance(party, dict) and party.get("role") == "borrower"
    }
    return {
        key for key in keys if key and " ".join(key) not in GENERIC_BORROWER_PHRASES
    }


def _borrowers_disagree(child: dict[str, object], parent: dict[str, object]) -> bool:
    """Return whether two rows name borrowers that cannot be the same party.

    False when either row names no borrower. Otherwise the rows agree when
    they share at least one borrower key exactly; a row naming several
    borrowers needs only one in common.
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
) -> dict[str, tuple[str, str]]:
    """Return {child_id: (parent_id, rule)} for links the ordinal rule supports.

    `rows` are debt-instrument rows. Only rows whose
    `amendment_of_debt_instrument_id` is null are children. A child with more
    than one candidate parent, either side of a mutual pair, and any link that
    would close a cycle are left out; an empty dict means no link was supported.
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
        # One amendment chain has one borrower; the filer's CIK alone cannot
        # separate two issuers' agreements named in one filing.
        if _borrowers_disagree(child, parent):
            return
        # The predecessor must not first appear after the state that replaces it.
        # Equality must stay allowed: a minted predecessor is first seen in the
        # same filing as its successor.
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
            # Offer every member of the preceding rank so a tie is refused by the
            # ambiguity check below rather than broken by id order.
            parents = [row_id for r, _, row_id in members if r == previous_rank]
            # A same-rank state that another same-rank state names as its
            # `amendment_of` has been replaced; it steps aside so the chain lands
            # on its replacement. Two unlinked rows of one rank are still a tie.
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

    # A mutual pair means the evidence does not settle which is the predecessor:
    # drop both links.
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


def apply_lineage_inference_pass(
    artifact_root: str | Path,
    *,
    data_dir: Path | None = None,
    renew: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Infer amendment lineage across the whole corpus, after all shards match.

    Every pointer with an `amendment_inferred_by` is re-opened and re-derived
    with ``infer_amendment_parents``, so published lineage depends only on the
    current rules and rows; extracted pointers are never re-opened. The
    observation and rollup columns are recomputed, every debt-instruments shard
    is rewritten, and an `infer-lineage` run manifest is written. ``renew`` is
    called before the work starts and before each shard is rewritten, and must
    raise if the writer lease has been lost.

    Returns counts: ``links``, ``reopened``, ``heads_before``, ``heads_after``;
    all zero, with nothing written, when instruments, edges or mentions are
    empty.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    instruments = read_dataset(debt_instruments_root(resolved_root, data_dir=data_dir))
    edges = read_dataset(mention_cluster_edges_root(resolved_root, data_dir=data_dir))
    mentions = read_dataset(
        dataset_root(
            MENTIONS_DATASET_NAME, artifact_root=resolved_root, data_dir=data_dir
        ),
        columns=EXTRACTED_MENTION_COLUMNS,
    )
    if instruments.empty or edges.empty or mentions.empty:
        return {"links": 0, "reopened": 0, "heads_before": 0, "heads_after": 0}
    if renew is not None:
        renew()

    member_edges = edges[edges["edge_type"] == "member"]
    member_groups: dict[str, list[str]] = {}
    for row in member_edges.to_dict("records"):
        member_groups.setdefault(str(row["debt_instrument_id"]), []).append(
            str(row["debt_instrument_mention_id"])
        )
    mention_index = {
        str(row["debt_instrument_mention_id"]): prepare_mention(row)
        for row in mentions.to_dict("records")
    }
    rows = instruments.to_dict("records")
    heads_before = sum(1 for row in rows if row.get("is_lineage_head"))

    # Before inferring as well as after: the rules read `first_seen_filing_date`,
    # which the rollup rewrites, so the pass must not depend on the on-disk value.
    apply_observation_columns(
        rows, member_groups=member_groups, mention_index=mention_index
    )

    reopened = 0
    for row in rows:
        if coerce_optional_text(row.get("amendment_inferred_by")) is None:
            continue
        row["amendment_of_debt_instrument_id"] = None
        row["amendment_inferred_by"] = None
        reopened += 1

    inferred = infer_amendment_parents(rows)
    by_id = {str(row["debt_instrument_id"]): row for row in rows}
    for child_id, (parent_id, rule) in inferred.items():
        by_id[child_id]["amendment_of_debt_instrument_id"] = parent_id
        by_id[child_id]["amendment_inferred_by"] = rule

    apply_lifecycle_rollup(
        rows,
        member_groups=member_groups,
        mention_index=mention_index,
    )
    heads_after = sum(1 for row in rows if row.get("is_lineage_head"))
    frame = pd.DataFrame(rows, columns=DEBT_INSTRUMENT_COLUMNS)
    # Same shard assignment as `match_pending_mentions`, null cik included, or a
    # rewritten row lands in a second shard beside its stale copy.
    frame["_shard"] = (
        frame["cik"].fillna("").map(lambda value: shard_for_cik(str(value)))
    )
    partitions_written: list[str] = []
    for cik_shard, shard_rows in frame.groupby("_shard"):
        if renew is not None:
            renew()
        partitions_written.append(
            write_partition_table(
                debt_instruments_root(resolved_root, data_dir=data_dir),
                partition={"cik_shard": str(cik_shard)},
                table=shard_rows.drop(columns=["_shard"]).reindex(
                    columns=DEBT_INSTRUMENT_COLUMNS
                ),
            )
        )
    # This pass rewrites partitions the match manifest lists, so it records its
    # own manifest.
    write_json_artifact(
        run_manifest_path(
            "infer-lineage",
            "latest",
            artifact_root=resolved_root,
            data_dir=data_dir,
        ),
        {
            "artifact_root": resolved_root,
            "stage": "infer-lineage",
            "partitions_written": partitions_written,
            "links": len(inferred),
            "reopened": reopened,
            "heads_before": heads_before,
            "heads_after": heads_after,
            "schema_version": MATCHER_SCHEMA_VERSION,
        },
    )
    LOGGER.info(
        "Lineage inference pass: %s links (%s inferred pointers re-opened), heads %s -> %s",
        len(inferred),
        reopened,
        heads_before,
        heads_after,
    )
    return {
        "links": len(inferred),
        "reopened": reopened,
        "heads_before": heads_before,
        "heads_after": heads_after,
    }
