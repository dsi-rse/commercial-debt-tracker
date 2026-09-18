"""Matcher stage for consolidating debt instrument mentions into stable clusters."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

import pandas as pd

from cdt.datasets import (
    cik_shard_partition_path,
    dataset_root,
    normalize_cik,
    resolve_artifact_root,
    run_manifest_path,
    shard_for_cik,
)
from cdt.extractor.core import (
    DEBT_INSTRUMENT_MENTION_COLUMNS as EXTRACTED_MENTION_COLUMNS,
)
from cdt.extractor.core import (
    LENDER_DISCLOSURE_NONE_NAMED,
    LENDER_DISCLOSURE_PRECEDENCE,
    LENDER_DISCLOSURE_VALUES,
    MENTIONS_DATASET_NAME,
    normalize_numeric_string,
)
from cdt.matcher.lineage_inference import infer_amendment_parents
from cdt.storage import (
    artifact_exists,
    coerce_dataset_text,
    read_dataset,
    read_json_artifact,
    write_json_artifact,
    write_partition_table,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RELATED_THRESHOLD = 0.75
DEFAULT_MEMBERSHIP_THRESHOLD = 0.90
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_LENDER_SUPPORT_THRESHOLD = 0.5
# Bumped 5 -> 6 for the four status columns this stage no longer publishes
# (#196), and 6 -> 7 for the two it gained: `synthesized_only` and
# `outstanding_balance_as_of_is_filing_date` (#203). A reader holding rows
# written at an older version is missing columns or holding removed ones, so it
# needs to know a rebuild happened.
MATCHER_SCHEMA_VERSION = 7
EDGE_TYPES = ("member", "related", "ambiguous_candidate")
GENERIC_LENDER_TERMS = frozenset(
    {
        "lender",
        "lenders",
        "purchaser",
        "purchasers",
        "holder",
        "holders",
        "investor",
        "investors",
        "buyer",
        "buyers",
        "noteholder",
        "noteholders",
        "trustee",
        "trustees",
    }
)
MENTION_CLUSTER_EDGE_COLUMNS = [
    "debt_instrument_mention_id",
    "debt_instrument_id",
    "edge_type",
    "match_score",
    "candidate_rank",
    "match_via",
    "evaluated_run_id",
]
DEBT_INSTRUMENT_COLUMNS = [
    "debt_instrument_id",
    "cik",
    "company_name",
    "seed_debt_instrument_mention_id",
    "amendment_of_debt_instrument_id",
    "retired_by_debt_instrument_ids",
    "split_of_debt_instrument_id",
    "superseded_by_debt_instrument_id",
    "lineage_family_id",
    "is_lineage_head",
    "first_seen_filing_date",
    "last_seen_filing_date",
    "mention_count",
    "document_count",
    "name",
    "name_source_mention_id",
    "instrument_type",
    "instrument_type_source_mention_id",
    "start_date",
    "start_date_source_mention_id",
    "maturity_date",
    "maturity_source_mention_id",
    "commitment_termination_date",
    "commitment_termination_source_mention_id",
    "principal_amount",
    "principal_currency",
    "principal_amount_kind",
    "principal_source_mention_id",
    "outstanding_balance",
    "outstanding_balance_currency",
    "outstanding_balance_as_of",
    # True when `outstanding_balance_as_of` is the filing date substituted for
    # a balance the filing dated no other way, so a consumer can tell a stated
    # as-of from a derived one (#203).
    "outstanding_balance_as_of_is_filing_date",
    "outstanding_balance_source_mention_id",
    "interest_rate_kind",
    "interest_rate_pct",
    "interest_rate_source_mention_id",
    "parties_json",
    "lender_disclosure",
    "amendment_inferred_by",
    # True when every member mention was synthesized by the extractor rather
    # than returned by the model — a minted prior state that never merged with
    # a mention of the instrument it describes (#203). The row is a real prior
    # state, cited from its successor's filing, but no filing describes it on
    # its own, and a reader summing capacity or counting live obligations needs
    # to know that.
    "synthesized_only",
]
MENTION_CLUSTER_EDGE_DATASET_NAME = "mention-cluster-edges"
DEBT_INSTRUMENT_DATASET_NAME = "debt-instruments"


def mention_cluster_edges_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical mention-cluster-edges dataset root."""
    return dataset_root(
        MENTION_CLUSTER_EDGE_DATASET_NAME,
        artifact_root=artifact_root,
        data_dir=data_dir,
    )


def mention_matches_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Backward-compatible alias for the canonical mention-cluster-edges root."""
    return mention_cluster_edges_root(artifact_root=artifact_root, data_dir=data_dir)


def debt_instruments_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical debt-instruments dataset root."""
    return dataset_root(
        DEBT_INSTRUMENT_DATASET_NAME,
        artifact_root=artifact_root,
        data_dir=data_dir,
    )


@dataclass(frozen=True)
class PreparedMention:
    """Normalized mention record used during incremental cluster assignment."""

    debt_instrument_mention_id: str
    item_id: str
    raw_id: str
    accession_number: str | None
    cik: str | None
    company_name: str | None
    date: str | None
    name: str | None
    instrument_type: str | None
    start_date: str | None
    maturity_date: str | None
    maturity_is_derived: bool
    commitment_termination_date: str | None
    principal_amount: str | None
    principal_currency: str | None
    principal_amount_kind: str | None
    amounts_json: str
    interest_rate_kind: str | None
    interest_rate_pct: str | None
    status: str | None
    amendment_of: str | None
    retired_by: tuple[str, ...]
    split_of: str | None
    parties_json: str
    lender_disclosure: str
    normalized_amount: str | None
    normalized_start_date: str | None
    normalized_end_date: str | None
    normalized_name_fingerprint: str | None
    lender_signature: str
    # Set only on a row the extractor synthesized (#203): the rule that minted
    # it and the model-emitted mention it was minted from. Read by the
    # canonical-field, profile and scoring rules; never a match key itself.
    synthesized_by: str | None = None
    synthesized_from_mention_id: str | None = None


@dataclass
class ClusterProfile:
    """Cached cluster state used while matching mentions incrementally."""

    debt_instrument_id: str
    cik: str
    seed_mention_id: str
    member_ids: list[str]
    normalized_amounts: set[str]
    normalized_start_dates: set[str]
    normalized_end_dates: set[str]
    normalized_name_fingerprints: set[str]
    lender_signatures: set[str]
    relation_target_ids: set[str] = field(default_factory=set)
    member_item_ids: set[str] = field(default_factory=set)
    retired: bool = False

    def add_member(self: ClusterProfile, mention: PreparedMention) -> None:
        """Update the cluster cache with one newly accepted member."""
        if mention.debt_instrument_mention_id not in self.member_ids:
            self.member_ids.append(mention.debt_instrument_mention_id)
        if mention.retired_by or mention.status in TERMINAL_STATUS_EVENTS:
            self.retired = True
        if mention.item_id:
            self.member_item_ids.add(mention.item_id)
        if mention.normalized_amount:
            self.normalized_amounts.add(mention.normalized_amount)
        if mention.normalized_start_date:
            self.normalized_start_dates.add(mention.normalized_start_date)
        if mention.normalized_end_date:
            self.normalized_end_dates.add(mention.normalized_end_date)
        # A synthesized prior state carries its successor's name. It scores its
        # own way in on that name, but must not widen the cluster's name class
        # afterward: the next mention would then be judged against the
        # amendment's name as well as the instrument's own (#203).
        if mention.normalized_name_fingerprint and mention.synthesized_by is None:
            self.normalized_name_fingerprints.add(mention.normalized_name_fingerprint)
        if mention.lender_signature:
            self.lender_signatures.add(mention.lender_signature)
        for target in (mention.amendment_of, *mention.retired_by, mention.split_of):
            if target:
                self.relation_target_ids.add(target)


@dataclass(frozen=True)
class CandidateScore:
    """One scored mention-to-cluster candidate relationship."""

    debt_instrument_id: str
    match_score: float
    support_family: str | None
    basis: str = "amount_start"
    exact_name: bool = False
    cluster_size: int = 0
    cluster_retired: bool = False

    @property
    def base_match_via(self: CandidateScore) -> str:
        """Return the explanation family without the outcome prefix."""
        if self.basis != "amount_start" or self.support_family is None:
            return self.basis
        return f"amount_start+{self.support_family}"


def _stale_schema_forces_rematch(
    resolved_root: str,
    *,
    data_dir: Path | None = None,
) -> bool:
    """Return True when the root was matched under an older matcher schema.

    ``MATCHER_SCHEMA_VERSION`` was written into the match manifest and read by
    nothing, which made an incremental match over an older root publish rows
    that are wrong rather than merely stale. Mention ids are content hashes, so
    a schema change that alters the hashed payload changes every id: the
    surviving clusters are then keyed on ids the mentions dataset no longer
    contains, and the rollup recomputes lifecycle columns from a member list
    that filtered to empty.

    It also silently degraded #203. Minting an amended instrument's prior state
    adds mentions, so on a root at an older version the pre-existing clusters
    hold the slots the mints would take on a clean build, and a cluster can end
    up with two members naming two different amendment parents — which
    ``derive_parent_links`` correctly refuses. Measured on ``lineage-verify``
    (recorded at version 4, code at 7): backfill plus one plain match published
    19 amendment pointers and 539 heads against a forced match's 22 and 536,
    and further plain matches never recovered it. EQT's Second Amended and
    Restated agreement was one of the rows that lost its pointer.

    A rematch is deterministic local compute, so promoting the run is cheaper
    than refusing it and safer than proceeding: refusing would wedge the
    scheduled daily run behind an operator, and proceeding publishes the wrong
    answer with a successful-looking log line (#208).
    """
    manifest_path = run_manifest_path(
        "match",
        "latest",
        artifact_root=resolved_root,
        data_dir=data_dir,
    )
    if not artifact_exists(manifest_path):
        return False
    recorded = read_json_artifact(manifest_path).get("schema_version")
    if not isinstance(recorded, int) or recorded >= MATCHER_SCHEMA_VERSION:
        return False
    LOGGER.warning(
        "Matcher schema is %s but %s was matched at %s; forcing a full rematch "
        "so clusters are not keyed on mention ids that have since changed",
        MATCHER_SCHEMA_VERSION,
        resolved_root,
        recorded,
    )
    return True


def match_pending_mentions(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    renew: Callable[[], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Match canonical debt instrument mentions into canonical matcher outputs.

    ``renew`` is called before each shard is rewritten; a full match pass can
    outlast the pipeline-writer lease TTL, and it must raise rather than let
    this run keep rewriting shards a lease thief now owns (#89).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    if not force:
        force = _stale_schema_forces_rematch(resolved_root, data_dir=data_dir)
    mention_rows = read_dataset(
        dataset_root(
            MENTIONS_DATASET_NAME, artifact_root=resolved_root, data_dir=data_dir
        ),
        columns=EXTRACTED_MENTION_COLUMNS,
    )
    if mention_rows.empty:
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=MENTION_CLUSTER_EDGE_COLUMNS
            ),
            "debt_instrument": pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
        }
    mention_rows = mention_rows.copy()
    company_names = company_names_by_cik(mention_rows)
    mention_rows["cik_shard"] = (
        mention_rows["cik"].fillna("").map(lambda value: shard_for_cik(str(value)))
    )
    edge_frames: list[pd.DataFrame] = []
    instrument_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    shard_groups = list(mention_rows.groupby("cik_shard"))
    total_partitions = len(shard_groups)
    for chunk_start in range(0, total_partitions, batch_size):
        chunk_groups = shard_groups[chunk_start : chunk_start + batch_size]
        for partition_index, (cik_shard, shard_mentions) in enumerate(
            chunk_groups, start=chunk_start + 1
        ):
            if renew is not None:
                renew()
            partition_start = perf_counter()
            if force:
                existing_edges = pd.DataFrame(columns=MENTION_CLUSTER_EDGE_COLUMNS)
                existing_instruments = pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS)
            else:
                existing_edges = read_dataset(
                    mention_cluster_edges_root(resolved_root, data_dir=data_dir),
                    partition_filter={"cik_shard": str(cik_shard)},
                )
                existing_instruments = read_dataset(
                    debt_instruments_root(resolved_root, data_dir=data_dir),
                    partition_filter={"cik_shard": str(cik_shard)},
                )
            tables = match_tables(
                shard_mentions.drop(columns=["cik_shard"]),
                existing_edges=existing_edges,
                existing_instruments=existing_instruments,
                strong_match_threshold=strong_match_threshold,
                loose_match_threshold=loose_match_threshold,
                ambiguity_margin=ambiguity_margin,
                company_names=company_names,
            )
            mention_cluster_edges = tables["debt_instrument_mentions"].reindex(
                columns=MENTION_CLUSTER_EDGE_COLUMNS
            )
            debt_instruments = tables["debt_instrument"].reindex(
                columns=DEBT_INSTRUMENT_COLUMNS
            )
            write_partition_table(
                mention_cluster_edges_root(resolved_root, data_dir=data_dir),
                partition={"cik_shard": str(cik_shard)},
                table=mention_cluster_edges,
            )
            write_partition_table(
                debt_instruments_root(resolved_root, data_dir=data_dir),
                partition={"cik_shard": str(cik_shard)},
                table=debt_instruments,
            )
            edge_frames.append(mention_cluster_edges)
            instrument_frames.append(debt_instruments)
            partitions_written.append(
                cik_shard_partition_path(
                    DEBT_INSTRUMENT_DATASET_NAME,
                    cik_shard=str(cik_shard),
                    artifact_root=resolved_root,
                    data_dir=data_dir,
                )
            )
            LOGGER.info(
                "Matcher partition complete: cik_shard=%s progress=%s/%s mentions=%s edge_rows=%s debt_instruments=%s elapsed=%.1fs",
                cik_shard,
                partition_index,
                total_partitions,
                len(shard_mentions),
                len(mention_cluster_edges),
                len(debt_instruments),
                perf_counter() - partition_start,
            )
    write_json_artifact(
        run_manifest_path(
            "match",
            "latest",
            artifact_root=resolved_root,
            data_dir=data_dir,
        ),
        {
            "artifact_root": resolved_root,
            "stage": "match",
            "batch_size": batch_size,
            "partitions_written": partitions_written,
            "membership_threshold": strong_match_threshold,
            "related_threshold": loose_match_threshold,
            "ambiguity_margin": ambiguity_margin,
            "schema_version": MATCHER_SCHEMA_VERSION,
        },
    )
    LOGGER.info(
        "Matcher complete: edge_rows=%s debt_instruments=%s",
        sum(len(frame) for frame in edge_frames),
        sum(len(frame) for frame in instrument_frames),
    )
    return {
        "debt_instrument_mentions": pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame(columns=MENTION_CLUSTER_EDGE_COLUMNS),
        "debt_instrument": pd.concat(instrument_frames, ignore_index=True)
        if instrument_frames
        else pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
    }


def match_tables(
    debt_instrument_mentions: pd.DataFrame,
    *,
    existing_edges: pd.DataFrame | None = None,
    existing_instruments: pd.DataFrame | None = None,
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    company_names: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Match in-memory debt instrument mentions into stable debt instrument clusters.

    Lineage inference is not performed here. It cannot be: this is called once
    per shard batch and sees only the clusters that batch touched, so no rule can
    ever see both states of one facility. ``apply_lineage_inference_pass`` runs it
    as a post-pass over the complete dataset instead.
    """
    if strong_match_threshold < loose_match_threshold:
        raise ValueError("strong_match_threshold must be >= loose_match_threshold")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative")

    rows = sorted(
        debt_instrument_mentions.to_dict("records"),
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("accession_number") or ""),
            str(row.get("item_id") or ""),
            str(row.get("debt_instrument_mention_id") or ""),
        ),
    )
    if not rows and (existing_edges is None or existing_edges.empty):
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=MENTION_CLUSTER_EDGE_COLUMNS
            ),
            "debt_instrument": pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
        }

    mention_index = {
        str(row["debt_instrument_mention_id"]): prepare_mention(row) for row in rows
    }
    edge_rows = (
        existing_edges.copy()
        if existing_edges is not None and not existing_edges.empty
        else pd.DataFrame(columns=MENTION_CLUSTER_EDGE_COLUMNS)
    )
    instrument_rows = (
        existing_instruments.copy()
        if existing_instruments is not None and not existing_instruments.empty
        else pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS)
    )
    processed_mentions = {
        str(row["debt_instrument_mention_id"])
        for row in edge_rows.to_dict("records")
        if str(row.get("edge_type")) == "member"
    }
    profiles = build_cluster_profiles(
        mention_index=mention_index,
        existing_edges=edge_rows,
        existing_instruments=instrument_rows,
    )
    class_sizes = name_class_sizes(mention_index, instrument_rows)
    new_edge_rows: list[dict[str, object]] = []

    for mention_id in sorted(
        mention_index, key=lambda key: mention_sort_key(mention_index[key])
    ):
        if mention_id in processed_mentions:
            continue
        mention = mention_index[mention_id]
        if mention.cik is None:
            continue
        candidates = score_candidates_for_mention(
            mention,
            profiles,
            strong_match_threshold=strong_match_threshold,
            loose_match_threshold=loose_match_threshold,
            name_class_size=class_sizes.get(mention_id, 1),
            lender_signature=borrowed_lender_signature(mention, mention_index),
        )
        chosen_cluster_id, chosen_edge_rows = resolve_candidates(
            mention,
            candidates,
            strong_match_threshold=strong_match_threshold,
            loose_match_threshold=loose_match_threshold,
            ambiguity_margin=ambiguity_margin,
            evaluated_run_id="latest",
        )
        new_edge_rows.extend(chosen_edge_rows)
        if chosen_cluster_id not in profiles:
            profiles[chosen_cluster_id] = build_empty_profile(
                chosen_cluster_id, mention
            )
        profiles[chosen_cluster_id].add_member(mention)

    edge_frames = []
    if not edge_rows.empty:
        edge_frames.append(edge_rows.reindex(columns=MENTION_CLUSTER_EDGE_COLUMNS))
    if new_edge_rows:
        edge_frames.append(
            pd.DataFrame(new_edge_rows, columns=MENTION_CLUSTER_EDGE_COLUMNS)
        )
    combined_edges = (
        pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame(columns=MENTION_CLUSTER_EDGE_COLUMNS)
    )
    member_edges = combined_edges[combined_edges["edge_type"] == "member"].copy()
    member_map = {
        str(row["debt_instrument_mention_id"]): str(row["debt_instrument_id"])
        for row in member_edges.to_dict("records")
    }
    normalized_members = build_member_groups(member_map)
    parent_links = derive_parent_links(
        normalized_members,
        mention_index,
        member_map,
        existing_instruments=instrument_rows,
    )
    debt_instrument_rows = build_debt_instrument_rows(
        normalized_members,
        mention_index,
        parent_links,
        existing_instruments=instrument_rows,
        company_names=company_names or company_names_by_cik(debt_instrument_mentions),
    )
    apply_lifecycle_rollup(
        debt_instrument_rows,
        member_groups=normalized_members,
        mention_index=mention_index,
    )
    return {
        "debt_instrument_mentions": combined_edges.reindex(
            columns=MENTION_CLUSTER_EDGE_COLUMNS
        ),
        "debt_instrument": pd.DataFrame(
            debt_instrument_rows, columns=DEBT_INSTRUMENT_COLUMNS
        ),
    }


# An extracted terminal event. The matcher reads these off the *mentions* to
# decide whether a cluster is retired, which breaks a name-only tie towards the
# live obligation (`ClusterProfile.retired`). It derives no status of its own:
# "is this borrowing still alive" needs a notion of now, and now is not an
# input this repository has (#196).
TERMINAL_STATUS_EVENTS = {"terminated", "repaid", "exchanged", "defaulted"}


def apply_lifecycle_rollup(
    rows: list[dict[str, object]],
    *,
    member_groups: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
) -> None:
    """Fill lineage-head and observation columns in place (#155).

    `superseded_by` marks a state that a later amendment replaced,
    `lineage_family_id` groups every state of one obligation history, and the
    observation columns count what the corpus has seen of each cluster.

    Deliberately no lifecycle `status`. Deriving one requires a notion of "now",
    which this repository cannot obtain without making its output a function of
    the run's scope rather than of the filings; the publisher derives it at
    publish time against an explicit `asOf` instead (#196).
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
    """Recompute what the corpus has seen of each cluster, in place.

    Split out of `apply_lifecycle_rollup` because the lineage rules *read*
    `first_seen_filing_date` — `infer_amendment_parents` uses it as both the
    predecessor-ordering guard and the chain sort key — while the rollup
    rewrites it from the member edges. Running the rollup only afterwards meant
    a pass could infer against a value it then overwrote, so pass N+1 saw a
    different corpus than pass N: on a row whose members are gone, pass 1
    yields one link and nulls the column, and pass 2 then adds a link pass 1
    refused. That row shape arises on its own, because mention ids are content
    hashes — re-extracting an item mints a new id, the old member edge is never
    deleted, and the old instrument survives with `mention_count` 0. The pass
    now recomputes these columns before inferring as well as after (#211).
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


def build_cluster_profiles(
    *,
    mention_index: dict[str, PreparedMention],
    existing_edges: pd.DataFrame,
    existing_instruments: pd.DataFrame,
) -> dict[str, ClusterProfile]:
    """Construct incremental cluster profiles from existing member edges."""
    member_rows = [
        row
        for row in existing_edges.to_dict("records")
        if str(row.get("edge_type")) == "member"
    ]
    members_by_instrument: dict[str, list[str]] = {}
    for row in member_rows:
        members_by_instrument.setdefault(str(row["debt_instrument_id"]), []).append(
            str(row["debt_instrument_mention_id"])
        )
    instrument_rows = {
        str(row["debt_instrument_id"]): row
        for row in existing_instruments.to_dict("records")
    }
    profiles: dict[str, ClusterProfile] = {}
    for debt_instrument_id, instrument_row in instrument_rows.items():
        member_ids = sorted(members_by_instrument.get(debt_instrument_id, []))
        seed_mention_id = str(
            instrument_row.get(
                "seed_debt_instrument_mention_id",
                member_ids[0] if member_ids else debt_instrument_id,
            )
        )
        profile = ClusterProfile(
            debt_instrument_id=debt_instrument_id,
            cik=coerce_optional_cik(instrument_row.get("cik")) or "",
            seed_mention_id=seed_mention_id,
            member_ids=list(member_ids),
            normalized_amounts=set(),
            normalized_start_dates=set(),
            normalized_end_dates=set(),
            normalized_name_fingerprints=set(),
            lender_signatures=set(),
        )
        normalized_amount = normalize_amount(
            coerce_optional_text(instrument_row.get("principal_amount"))
            or coerce_optional_text(instrument_row.get("amount"))
        )
        if normalized_amount:
            profile.normalized_amounts.add(normalized_amount)
        normalized_start_date = normalize_date(
            coerce_optional_text(instrument_row.get("start_date"))
        )
        if normalized_start_date:
            profile.normalized_start_dates.add(normalized_start_date)
        normalized_end_date = normalize_date(
            coerce_optional_text(instrument_row.get("maturity_date"))
            or coerce_optional_text(instrument_row.get("end_date"))
        )
        if normalized_end_date:
            profile.normalized_end_dates.add(normalized_end_date)
        normalized_name = normalize_name_fingerprint(
            coerce_optional_text(instrument_row.get("name"))
        )
        if normalized_name:
            profile.normalized_name_fingerprints.add(normalized_name)
        lenders = lender_signature(
            instrument_row.get("parties_json") or instrument_row.get("lenders_json")
        )
        if lenders:
            profile.lender_signatures.add(lenders)
        for member_id in member_ids:
            mention = mention_index.get(member_id)
            if mention is not None:
                profile.add_member(mention)
        profiles[debt_instrument_id] = profile

    for debt_instrument_id, member_ids in members_by_instrument.items():
        if debt_instrument_id in profiles:
            continue
        ordered_member_ids = sorted(
            [member_id for member_id in member_ids if member_id in mention_index],
            key=lambda mention_id: mention_sort_key(mention_index[mention_id]),
        )
        if not ordered_member_ids:
            continue
        seed_mention_id = str(
            instrument_rows.get(debt_instrument_id, {}).get(
                "seed_debt_instrument_mention_id", ordered_member_ids[0]
            )
        )
        seed_mention = mention_index[ordered_member_ids[0]]
        profile = ClusterProfile(
            debt_instrument_id=debt_instrument_id,
            cik=seed_mention.cik or "",
            seed_mention_id=seed_mention_id,
            member_ids=[],
            normalized_amounts=set(),
            normalized_start_dates=set(),
            normalized_end_dates=set(),
            normalized_name_fingerprints=set(),
            lender_signatures=set(),
        )
        for member_id in ordered_member_ids:
            profile.add_member(mention_index[member_id])
        profiles[debt_instrument_id] = profile
    return profiles


def build_empty_profile(
    debt_instrument_id: str,
    mention: PreparedMention,
) -> ClusterProfile:
    """Create an empty profile for a newly created singleton cluster."""
    return ClusterProfile(
        debt_instrument_id=debt_instrument_id,
        cik=mention.cik or "",
        seed_mention_id=debt_instrument_id,
        member_ids=[],
        normalized_amounts=set(),
        normalized_start_dates=set(),
        normalized_end_dates=set(),
        normalized_name_fingerprints=set(),
        lender_signatures=set(),
    )


def relaxed_keys_support_membership(
    mention: PreparedMention,
    profile: ClusterProfile,
) -> bool:
    """Return whether one agreeing key and no conflicting key joins the cluster.

    Requiring amount *and* start date together was a cheap identity proxy that
    left roughly half of all mentions unable to join anything: an announcement
    carries an amount but no closing date, an amendment carries dates but no
    principal. With the name already compatible, one key agreeing and none
    disagreeing is enough evidence.
    """
    agrees = (
        (
            mention.normalized_amount is not None
            and mention.normalized_amount in profile.normalized_amounts
        )
        or (
            mention.normalized_start_date is not None
            and mention.normalized_start_date in profile.normalized_start_dates
        )
        or (
            mention.normalized_end_date is not None
            and mention.normalized_end_date in profile.normalized_end_dates
        )
    )
    if not agrees:
        return False
    conflicts = (
        mention.normalized_amount is not None
        and profile.normalized_amounts
        and mention.normalized_amount not in profile.normalized_amounts
    ) or (
        mention.normalized_start_date is not None
        and profile.normalized_start_dates
        and mention.normalized_start_date not in profile.normalized_start_dates
    )
    return not conflicts


def score_candidates_for_mention(
    mention: PreparedMention,
    profiles: dict[str, ClusterProfile],
    *,
    strong_match_threshold: float,
    loose_match_threshold: float,
    name_class_size: int = 1,
    lender_signature: str | None = None,
) -> list[CandidateScore]:
    """Return scored candidate clusters for one mention.

    ``lender_signature`` overrides the mention's own for scoring only. A
    synthesized prior state publishes no lenders — a joinder adds and removes
    them, so the filing never states who lent under the earlier terms — but
    may borrow its successor's signature to vouch for a membership (#203).
    Nothing borrowed reaches the cluster profile or a published row.
    """
    del loose_match_threshold
    scoring_lender_signature = (
        lender_signature if lender_signature is not None else mention.lender_signature
    )
    if mention.cik is None:
        return []
    has_match_keys = (
        mention.normalized_amount is not None
        and mention.normalized_start_date is not None
    )
    name_is_identifying = name_fingerprint_is_identifying(
        mention.normalized_name_fingerprint
    )
    if (
        not has_match_keys
        and not name_is_identifying
        and name_class_size > NAME_CLASS_GATE
    ):
        return []
    candidates: list[CandidateScore] = []
    for profile in profiles.values():
        if profile.cik != mention.cik:
            continue
        if mention.item_id in profile.member_item_ids:
            # One item returns one object per instrument — the extractor's
            # invariant — so a same-item pair is two instruments by
            # construction, whatever their names and keys say. The partial
            # sibling test this replaces needed a shared start date and both
            # amounts present, so Gray Media's $70M add-on tap, whose parent
            # series stated no start date, slid through the identifying-name
            # path and published the series at the add-on's size (#161). It
            # also covers Longevity Health's same-day twin notes (#131) and
            # Kestra's four tranches. Cross-filing launch/pricing/closing
            # merges are different items and unaffected.
            continue
        if mention.amendment_of and mention.amendment_of in profile.member_ids:
            continue
        if any(target in profile.member_ids for target in mention.retired_by):
            continue
        if mention.split_of and mention.split_of in profile.member_ids:
            continue
        if mention.debt_instrument_mention_id in profile.relation_target_ids:
            continue
        if profile.normalized_end_dates and not any(
            end_dates_are_compatible(mention.normalized_end_date, candidate_end_date)
            for candidate_end_date in profile.normalized_end_dates
        ):
            continue
        if profile.normalized_name_fingerprints and not any(
            name_rates_are_compatible(
                mention.normalized_name_fingerprint, candidate_name
            )
            for candidate_name in profile.normalized_name_fingerprints
        ):
            continue
        keys_match = (
            has_match_keys
            and mention.normalized_amount in profile.normalized_amounts
            and mention.normalized_start_date in profile.normalized_start_dates
        )
        name_compatible = any(
            name_fingerprints_are_compatible(
                mention.normalized_name_fingerprint, candidate_name
            )
            for candidate_name in profile.normalized_name_fingerprints
        )
        if not keys_match:
            if name_compatible:
                # A cluster whose every name is generic (`senior notes`) cannot
                # claim a mention whose name individuates a series; letting it
                # seeded the tie cascade that shattered GEO's note histories.
                if (
                    name_is_identifying
                    and profile.normalized_name_fingerprints
                    and not any(
                        name_fingerprint_is_identifying(candidate_name)
                        for candidate_name in profile.normalized_name_fingerprints
                    )
                ):
                    continue
                # Launch, pricing, and closing 8-Ks for one offering drift on
                # amount (upsizes) and start date (pricing vs settlement), so an
                # identifying name may attach a mention whose keys conflict.
                # Otherwise one agreeing key with none conflicting will do, but
                # only while the name still individuates within the issuer.
                if name_is_identifying or (
                    name_class_size <= NAME_CLASS_GATE
                    and relaxed_keys_support_membership(mention, profile)
                ):
                    candidates.append(
                        CandidateScore(
                            debt_instrument_id=profile.debt_instrument_id,
                            match_score=round(strong_match_threshold, 4),
                            support_family="name",
                            basis="name_fingerprint",
                            exact_name=(
                                mention.normalized_name_fingerprint
                                in profile.normalized_name_fingerprints
                            ),
                            cluster_size=len(profile.member_ids),
                            cluster_retired=profile.retired,
                        )
                    )
            continue
        lender_similarity = max(
            (
                lender_similarity_score(scoring_lender_signature, candidate_signature)
                for candidate_signature in profile.lender_signatures
                if scoring_lender_signature and candidate_signature
            ),
            default=0.0,
        )
        name_support = 1.0 if name_compatible else 0.0
        name_conflict = bool(
            mention.normalized_name_fingerprint is not None
            and profile.normalized_name_fingerprints
            and not name_compatible
        )
        support_family: str | None = None
        support_strength = 0.0
        # Distinct facilities under one credit agreement share amount, start
        # date, and lenders, so shared lenders cannot vouch for a membership
        # when the two sides actively disagree on the instrument name.
        if lender_similarity >= DEFAULT_LENDER_SUPPORT_THRESHOLD and not name_conflict:
            support_family = "lenders"
            support_strength = lender_similarity
        if name_support > support_strength:
            support_family = "name"
            support_strength = name_support
        match_score = round(min(1.0, 0.75 + 0.25 * support_strength), 4)
        candidates.append(
            CandidateScore(
                debt_instrument_id=profile.debt_instrument_id,
                match_score=match_score,
                support_family=support_family,
            )
        )
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.match_score, candidate.debt_instrument_id),
    )


def resolve_candidates(
    mention: PreparedMention,
    candidates: list[CandidateScore],
    *,
    strong_match_threshold: float,
    loose_match_threshold: float,
    ambiguity_margin: float,
    evaluated_run_id: str,
) -> tuple[str, list[dict[str, object]]]:
    """Resolve one mention into one member edge plus optional related edges."""
    qualifying_members = [
        candidate
        for candidate in candidates
        if candidate.match_score >= strong_match_threshold
    ]
    if qualifying_members:
        top_candidate = qualifying_members[0]
        close_competitors = [
            candidate
            for candidate in qualifying_members[1:]
            if top_candidate.match_score - candidate.match_score <= ambiguity_margin
        ]
        tied = [top_candidate, *close_competitors]
        name_only_tie = bool(close_competitors) and all(
            candidate.basis == "name_fingerprint" for candidate in tied
        )
        if name_only_tie:
            # A mention that ties several existing clusters on its name belongs
            # to at most one of them; seeding a third can never be right, and
            # the third guarantees every later mention of the series ties too
            # (the cascade behind 263 ambiguous edges on the 2026-09 window).
            # Prefer the cluster already carrying this exact name, then a live
            # obligation over a retired one, then the largest cluster.
            tied.sort(
                key=lambda candidate: (
                    not candidate.exact_name,
                    candidate.cluster_retired,
                    -candidate.cluster_size,
                    candidate.debt_instrument_id,
                )
            )
            top_candidate = tied[0]
            edge_rows = [
                build_edge_row(
                    mention_id=mention.debt_instrument_mention_id,
                    debt_instrument_id=top_candidate.debt_instrument_id,
                    edge_type="member",
                    match_score=top_candidate.match_score,
                    candidate_rank=1,
                    match_via=render_match_via(
                        "member", top_candidate.support_family, top_candidate.basis
                    ),
                    evaluated_run_id=evaluated_run_id,
                )
            ]
            for rank, candidate in enumerate(tied[1:], start=2):
                edge_rows.append(
                    build_edge_row(
                        mention_id=mention.debt_instrument_mention_id,
                        debt_instrument_id=candidate.debt_instrument_id,
                        edge_type="ambiguous_candidate",
                        match_score=candidate.match_score,
                        candidate_rank=rank,
                        match_via=render_match_via(
                            "ambiguous", candidate.support_family, candidate.basis
                        ),
                        evaluated_run_id=evaluated_run_id,
                    )
                )
            return top_candidate.debt_instrument_id, edge_rows
        if not close_competitors:
            edge_rows = [
                build_edge_row(
                    mention_id=mention.debt_instrument_mention_id,
                    debt_instrument_id=top_candidate.debt_instrument_id,
                    edge_type="member",
                    match_score=top_candidate.match_score,
                    candidate_rank=1,
                    match_via=render_match_via(
                        "member", top_candidate.support_family, top_candidate.basis
                    ),
                    evaluated_run_id=evaluated_run_id,
                )
            ]
            for rank, candidate in enumerate(candidates[1:], start=2):
                if candidate.match_score < loose_match_threshold:
                    continue
                edge_rows.append(
                    build_edge_row(
                        mention_id=mention.debt_instrument_mention_id,
                        debt_instrument_id=candidate.debt_instrument_id,
                        edge_type="related",
                        match_score=candidate.match_score,
                        candidate_rank=rank,
                        match_via=render_match_via(
                            "related", candidate.support_family, candidate.basis
                        ),
                        evaluated_run_id=evaluated_run_id,
                    )
                )
            return top_candidate.debt_instrument_id, edge_rows

        new_cluster_id = mention.debt_instrument_mention_id
        edge_rows = [
            build_edge_row(
                mention_id=mention.debt_instrument_mention_id,
                debt_instrument_id=new_cluster_id,
                edge_type="member",
                match_score=1.0,
                candidate_rank=1,
                match_via="member:seed",
                evaluated_run_id=evaluated_run_id,
            )
        ]
        for rank, candidate in enumerate(qualifying_members, start=1):
            edge_rows.append(
                build_edge_row(
                    mention_id=mention.debt_instrument_mention_id,
                    debt_instrument_id=candidate.debt_instrument_id,
                    edge_type="ambiguous_candidate",
                    match_score=candidate.match_score,
                    candidate_rank=rank,
                    match_via=render_match_via(
                        "ambiguous", candidate.support_family, candidate.basis
                    ),
                    evaluated_run_id=evaluated_run_id,
                )
            )
        return new_cluster_id, edge_rows

    related_candidates = [
        candidate
        for candidate in candidates
        if candidate.match_score >= loose_match_threshold
    ]
    new_cluster_id = mention.debt_instrument_mention_id
    edge_rows = [
        build_edge_row(
            mention_id=mention.debt_instrument_mention_id,
            debt_instrument_id=new_cluster_id,
            edge_type="member",
            match_score=1.0,
            candidate_rank=1,
            match_via="member:seed",
            evaluated_run_id=evaluated_run_id,
        )
    ]
    for rank, candidate in enumerate(related_candidates, start=1):
        edge_rows.append(
            build_edge_row(
                mention_id=mention.debt_instrument_mention_id,
                debt_instrument_id=candidate.debt_instrument_id,
                edge_type="related",
                match_score=candidate.match_score,
                candidate_rank=rank,
                match_via=render_match_via(
                    "related", candidate.support_family, candidate.basis
                ),
                evaluated_run_id=evaluated_run_id,
            )
        )
    return new_cluster_id, edge_rows


def build_edge_row(
    *,
    mention_id: str,
    debt_instrument_id: str,
    edge_type: str,
    match_score: float,
    candidate_rank: int,
    match_via: str,
    evaluated_run_id: str,
) -> dict[str, object]:
    """Return one persisted edge row."""
    return {
        "debt_instrument_mention_id": mention_id,
        "debt_instrument_id": debt_instrument_id,
        "edge_type": edge_type,
        "match_score": match_score,
        "candidate_rank": candidate_rank,
        "match_via": match_via,
        "evaluated_run_id": evaluated_run_id,
    }


def render_match_via(
    outcome: str, support_family: str | None, basis: str = "amount_start"
) -> str:
    """Render one stable explanation-family label for an edge."""
    base = f"{outcome}:{basis}"
    if support_family is None or basis != "amount_start":
        return base
    return f"{base}+{support_family}"


def build_member_groups(member_map: dict[str, str]) -> dict[str, list[str]]:
    """Group mention IDs by assigned debt instrument ID."""
    members: dict[str, list[str]] = {}
    for mention_id, debt_instrument_id in member_map.items():
        members.setdefault(debt_instrument_id, []).append(mention_id)
    return {
        debt_instrument_id: sorted(member_ids)
        for debt_instrument_id, member_ids in members.items()
    }


def derive_parent_links(
    member_groups: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
    mention_to_instrument: dict[str, str],
    *,
    existing_instruments: pd.DataFrame | None = None,
) -> dict[str, dict[str, str | None]]:
    """Map mention-level lineage onto debt instrument parent links."""
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
        # Ambiguity within one parent-pointer kind is unresolvable — there is no
        # way to choose between two amendment parents — so that kind publishes
        # nothing. Each kind is judged on its own: the columns are independent,
        # and an instrument that splits from one predecessor and is later
        # retired has a place to record both. Nulling every column whenever a
        # second kind appeared discarded lineage that was individually
        # unambiguous (#130). Retirers are exempt: several instruments jointly
        # retiring one obligation is a legitimate state of the world, so the
        # column is a list and keeps them all.
        amendment_is_ambiguous = len(amendment_parents) > 1
        if amendment_is_ambiguous:
            amendment_parents.clear()
        if len(split_parents) > 1:
            split_parents.clear()
        # The existing row's amendment pointer is a *fallback*, not a candidate.
        # Seeding it alongside the extracted ones put a guess and a fact in the
        # same set, and the guard above then threw both away: a row carrying a
        # stale inferred pointer lost the #203 pointer its own mention now
        # states, the pass re-inferred its guess on the next run, and the
        # extracted link never came back. Measured on `data/lineage-verify`,
        # backfill plus one plain match published 19 pointers and 539 heads
        # against a clean rebuild's 22 and 536, and three further matches did
        # not recover it. What the mentions state wins; the carried pointer is
        # what keeps an inferred link alive across an ordinary rematch, since
        # no mention names it (#184, #204). An ambiguous extracted set is a
        # refusal, so it does not fall back — a guess is worse than no pointer.
        if not amendment_parents and not amendment_is_ambiguous and existing_amendment:
            amendment_parents.add(existing_amendment)
        amendment_parent = next(iter(amendment_parents), None)
        # Provenance travels with the pointer it describes. The amendment pointer
        # is carried forward from the existing row above, so without this an
        # ordinary rematch kept an inferred pointer and dropped the column saying
        # it was inferred — publishing a guess as indistinguishable from an
        # extracted relation, which is the one thing the column exists to prevent
        # (#184). Cleared when the pointer changes, because the old rule no
        # longer describes the new target.
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
    return parent_links


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
        # A synthesized prior state carries its successor's filing date, so by
        # recency it is the newest member and would supply every canonical
        # field — renaming a predecessor cluster after the amendment that
        # replaced it, which in turn hands `ordinal_chain` two rows of one rank
        # and makes it refuse the link (#203). Model-emitted members decide the
        # canonical values whenever there is one; a synthesized member does only
        # when it is all the cluster has.
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
                    str(existing_row.get("parties_json") or "[]"),
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
        cik = coerce_optional_cik(row.get("cik"))
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


def first_non_null(
    ordered_mention_ids: list[str],
    mention_index: dict[str, PreparedMention],
    field_name: str,
) -> str | None:
    """Return the newest non-null field value across member mentions."""
    for mention_id in ordered_mention_ids:
        value = getattr(mention_index[mention_id], field_name)
        if value is not None:
            return value
    return None


def canonical_scalar_fields(
    ordered_member_ids: list[str],
    mention_index: dict[str, PreparedMention],
    existing_row: dict[str, object],
    *,
    field_name: str,
    source_column: str,
    existing_keys: tuple[str, ...] | None = None,
) -> dict[str, str | None]:
    """Return one canonical field plus the mention it actually came from (#151).

    The site attributes each canonical value to a source document; without the
    pointer it guessed, and could stamp the value with the wrong filing. A
    value carried forward from the existing row keeps that row's recorded
    source.
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
    """Return the canonical maturity, preferring stated dates over name-derived.

    Every post-closing `due 2030` mention re-introduces the synthesized
    year-end, so recency-only selection let a name-derived `2030-12-31`
    outrank the closing 8-K's stated `2030-07-01` (#162). The newest stated
    maturity wins; a derived value — name-derived or computed (#166) —
    publishes only when no mention in the cluster states one.
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
    value = coerce_optional_text(
        existing_row.get("maturity_date") or existing_row.get("end_date")
    )
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

    Currency and kind travel with the amount they describe (#140): mixing the
    newest amount with an older mention's currency could relabel an AUD
    facility as USD.
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
        "principal_amount": coerce_optional_text(
            existing_row.get("principal_amount") or existing_row.get("amount")
        ),
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

    Kept apart from principal so a balance can never double-count as the
    headline amount (#140). An undated balance is bounded by the filing that
    observed it, and `outstanding_balance_as_of_is_filing_date` records that
    the date was substituted rather than stated: a view may derive a value, but
    it may not publish a derived value as if the filing had said it (#203).
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
    """Return the canonical interest rate from the newest carrying mention (#157)."""
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


def dedupe_party_clusters(payloads: list[str]) -> list[dict[str, object]]:
    """Return deduped party cluster payloads, keyed by role plus canonical name.

    The role is part of the key so one entity appearing in two roles — an agent
    that is also a lender — keeps both rows (#150).
    """
    deduped: dict[str, dict[str, object]] = {}
    for payload in payloads:
        for cluster in parse_cluster_list(payload):
            canonical = party_dedupe_key(cluster)
            if not canonical:
                continue
            key = f"{cluster.get('role', 'lender')}::{canonical}"
            if key not in deduped:
                deduped[key] = cluster
    return [deduped[key] for key in sorted(deduped)]


def party_dedupe_key(cluster: dict[str, object]) -> str:
    """Return the key one party cluster dedupes on: its extractor-chosen name.

    The extractor already picked the cluster's `canonical_name` (#150), and
    re-deriving it here from the spans was worse: `normalize_party_text` strips
    legal-form words before the longest span is chosen, so `NCL Corporation
    Ltd.` shrank to `ncl` and lost to its own `NCLC` alias, and `EQT
    Corporation` lost to `Buyer Parent`. Measured over the 1,632 party clusters
    of one eval window, the two agreed on 1,595 and the matcher's choice was the
    worse one in the differences (#203). Payloads written before #150 carry no
    `canonical_name`, so those still take the span-derived key. `lender_keys`
    deliberately keeps the span-derived key: it is a match-scoring surface, and
    changing it re-scores clusters, which a dedupe fix must not do.
    """
    canonical_name = cluster.get("canonical_name")
    if isinstance(canonical_name, str) and canonical_name.strip():
        return normalize_party_text(canonical_name)
    return cluster_canonical_key(cluster)


def parse_cluster_list(value: str) -> list[dict[str, object]]:
    """Parse one JSON cluster list."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [cluster for cluster in payload if isinstance(cluster, dict)]


def cluster_canonical_key(cluster: dict[str, object]) -> str:
    """Return the normalized canonical key for one cluster.

    A cluster can hold a defined-term alias alongside the party it names, as in
    `Oaktree` and `Purchasers`. The specific name is the useful key, so generic
    party words lose to it even when the alias is the longer string.
    """
    # Current payloads carry `spans`; partitions written before the evidence
    # shape change (#128) carry `mentions`. Both list {text, offsets} dicts.
    spans = cluster.get("spans", cluster.get("mentions", []))
    if not isinstance(spans, list):
        return ""
    texts = [
        normalize_party_text(str(span.get("text", "")))
        for span in spans
        if isinstance(span, dict) and span.get("text")
    ]
    texts = [text for text in texts if text]
    if not texts:
        return ""
    specific = [text for text in texts if text not in GENERIC_LENDER_TERMS]
    return max(specific or texts, key=len)


def prepare_mention(row: dict[str, object]) -> PreparedMention:
    """Normalize one mention row for matching."""
    return PreparedMention(
        debt_instrument_mention_id=str(row["debt_instrument_mention_id"]),
        item_id=str(row["item_id"]),
        raw_id=str(row["raw_id"]),
        accession_number=coerce_optional_text(row.get("accession_number")),
        # Normalized so mentions written before CIKs were zero-padded (#153)
        # still group with rows written after.
        cik=coerce_optional_cik(row.get("cik")),
        company_name=coerce_optional_text(row.get("company_name")),
        date=coerce_optional_text(row.get("date")),
        name=coerce_optional_text(row.get("name")),
        instrument_type=coerce_optional_text(row.get("instrument_type")),
        start_date=coerce_optional_text(row.get("start_date")),
        maturity_date=coerce_optional_text(row.get("maturity_date")),
        maturity_is_derived=maturity_derivation(row.get("maturity_date_json"))
        in DERIVED_MATURITY_KINDS,
        commitment_termination_date=coerce_optional_text(
            row.get("commitment_termination_date")
        ),
        principal_amount=coerce_optional_text(row.get("principal_amount")),
        principal_currency=coerce_optional_text(row.get("principal_currency")),
        principal_amount_kind=coerce_optional_text(row.get("principal_amount_kind")),
        amounts_json=str(row.get("amounts_json") or "[]"),
        interest_rate_kind=coerce_optional_text(row.get("interest_rate_kind")),
        interest_rate_pct=coerce_optional_text(row.get("interest_rate_pct")),
        status=coerce_optional_text(row.get("status")),
        amendment_of=coerce_optional_text(row.get("amendment_of")),
        retired_by=tuple(json.loads(str(row.get("retired_by_json") or "[]"))),
        split_of=coerce_optional_text(row.get("split_of")),
        parties_json=str(row.get("parties_json") or "[]"),
        lender_disclosure=coerce_lender_disclosure(row.get("lender_disclosure")),
        normalized_amount=normalize_amount(
            coerce_optional_text(row.get("principal_amount"))
        ),
        normalized_start_date=normalize_date(
            coerce_optional_text(row.get("start_date"))
        ),
        normalized_end_date=normalized_end_date_for_matching(row),
        normalized_name_fingerprint=normalize_name_fingerprint(
            coerce_optional_text(row.get("name"))
        ),
        lender_signature=lender_signature(row.get("parties_json")),
        synthesized_by=coerce_optional_text(row.get("synthesized_by")),
        synthesized_from_mention_id=coerce_optional_text(
            row.get("synthesized_from_mention_id")
        ),
    )


def mention_sort_key(mention: PreparedMention) -> tuple[str, str, str, int, str]:
    """Return deterministic processing order for cluster assignment.

    Within one item a synthesized prior state is placed before the amended
    object it was minted from: it is the earlier state, and it must be the one
    that joins the instrument's existing cluster. Left to id order, the amended
    object joined first and the same-item guard then refused its own prior
    state, which stranded that state as a head and — where the cluster held two
    amended objects — gave the cluster two amendment parents and so none (#203).
    """
    return (
        mention.date or "",
        mention.accession_number or "",
        mention.item_id,
        0 if mention.synthesized_by is not None else 1,
        mention.debt_instrument_mention_id,
    )


def mention_recency_key(mention: PreparedMention) -> tuple[str, str, str, str]:
    """Return recency ordering for field resolution."""
    return (
        mention.date or "",
        mention.accession_number or "",
        mention.item_id,
        mention.debt_instrument_mention_id,
    )


def coerce_optional_text(value: object) -> str | None:
    """Return one trimmed string or None, treating placeholder text as missing."""
    return coerce_dataset_text(value)


def coerce_lender_disclosure(value: object) -> str:
    """Return one known lender-disclosure value, defaulting to `none_named`.

    A mention that records nothing about who holds the debt has named no
    lender, which is exactly `none_named` — the conservative reading, and the
    one that cannot invent a complete syndicate list out of a missing value.
    """
    text = coerce_dataset_text(value)
    return text if text in LENDER_DISCLOSURE_VALUES else LENDER_DISCLOSURE_NONE_NAMED


def aggregate_lender_disclosure(values: list[str | None]) -> str:
    """Roll several mentions' disclosure answers into one for the instrument.

    Worst-of by `LENDER_DISCLOSURE_PRECEDENCE`: a single filing showing a
    collective lender phrase means holders are hidden however many other
    filings name some, while a filing that named every lender supersedes one
    that named none.
    """
    known = [value for value in values if value in LENDER_DISCLOSURE_VALUES]
    if not known:
        return LENDER_DISCLOSURE_NONE_NAMED
    return max(known, key=lambda value: LENDER_DISCLOSURE_PRECEDENCE[value])


def coerce_optional_bool(value: object) -> bool | None:
    """Return one nullable flag read back from a published row.

    A declared `bool` column round-trips as Python or numpy bools with nulls
    read as None or NaN; a row that predates the column has nothing at all.
    Text spellings are accepted so a hand-built frame reads the same way.
    """
    if value is None or isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def coerce_optional_cik(value: object) -> str | None:
    """Return one canonical zero-padded CIK or None (#153)."""
    text = coerce_dataset_text(value)
    return normalize_cik(text) if text is not None else None


def normalize_amount(value: str | None) -> str | None:
    """Normalize amount strings for matcher comparisons."""
    if value is None:
        return None
    lowered = value.lower()
    multiplier = 1
    if "billion" in lowered:
        multiplier = 1_000_000_000
    elif "million" in lowered:
        multiplier = 1_000_000
    elif "thousand" in lowered:
        multiplier = 1_000
    digits = re.findall(r"\d+(?:\.\d+)?", lowered.replace(",", ""))
    if digits:
        amount = float(digits[0]) * multiplier
        if amount.is_integer():
            return str(int(amount))
        return f"{amount:.2f}"
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_date(value: str | None) -> str | None:
    """Normalize date strings for matcher comparisons."""
    if value is None:
        return None
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    month_map = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    match = re.search(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})",
        text,
    )
    if match:
        month = month_map.get(match.group("month").lower())
        if month:
            return f"{match.group('year')}-{month}-{int(match.group('day')):02d}"
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_name_fingerprint(value: str | None) -> str | None:
    """Normalize debt-instrument names for comparison."""
    if value is None:
        return None
    text = value.lower()
    # Close a gap between the coupon digits and the percent sign before the
    # trailing-zero rules below look for `%`. Filings write both `4.375%` and
    # `4.375 %`, and the punctuation pass turns the space into a token break, so
    # the two spellings fingerprinted differently and never matched.
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"(\d+)\.(\d*?[1-9])0+(?=%)", r"\1.\2", text)
    text = re.sub(r"(\d+)\.0+(?=%)", r"\1", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def lender_keys(value: object) -> list[str]:
    """Return normalized lender cluster keys in deterministic order.

    Clusters without a ``role`` key are treated as lenders: they come from
    payloads written before parties were unified (#150), when the lender list
    was its own column.
    """
    keys: list[str] = []
    for cluster in parse_cluster_list(str(value or "[]")):
        if str(cluster.get("role", "lender")) != "lender":
            continue
        key = cluster_canonical_key(cluster)
        if key:
            keys.append(key)
    return sorted(set(keys))


def lender_signature(value: object) -> str:
    """Return one normalized lender signature from extractor JSON payload."""
    return " | ".join(
        key for key in lender_keys(value) if key not in GENERIC_LENDER_TERMS
    )


def normalize_party_text(value: str) -> str:
    """Normalize party strings before similarity comparison and dedupe."""
    text = value.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(
        r"\b(national association|n a|na|inc|llc|ltd|plc|corp|corporation|company|co)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def borrowed_lender_signature(
    mention: PreparedMention, mention_index: dict[str, PreparedMention]
) -> str | None:
    """Return the successor's lender signature for a synthesized prior state.

    Only when the mention is synthesized, names no lender of its own, and its
    successor is in this run's index; None otherwise, which leaves the scorer
    on the mention's own signature. Scoring-only by construction: the caller
    still adds the *original* mention to the profile it joins (#203).
    """
    if mention.synthesized_by is None or mention.lender_signature:
        return None
    if mention.synthesized_from_mention_id is None:
        return None
    successor = mention_index.get(mention.synthesized_from_mention_id)
    if successor is None or not successor.lender_signature:
        return None
    return successor.lender_signature


def lender_similarity_score(left: str, right: str) -> float:
    """Return one deterministic similarity score for lender strings."""
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(a=left, b=right).ratio(), 4)


YEAR_TEXT_LENGTH = 4
MONTH_TEXT_LENGTH = 7


def normalized_end_date_for_matching(row: dict[str, object]) -> str | None:
    """Return the end date the matcher compares, at its true resolution.

    A year-only maturity such as "due 2030" is synthesized to ``2030-12-31`` on
    the way into the dataset, and its payload says ``derived_from: "name"``.
    Comparing that synthesized day would either invent precision or force every
    genuine December 31 maturity to be treated loosely — which is what happened
    while the provenance flag was missing (#128). Name-derived year-end values
    collapse to the bare year here; other name-derived values — the month-end
    synthesized from "due April 2033" (#164), or a full date embedded in the
    name — collapse to their month, so a stated mid-month maturity does not
    falsely conflict with the name's synthetic day. Stated dates keep their
    day.
    """
    value = normalize_date(coerce_optional_text(row.get("maturity_date")))
    if not value:
        return None
    derivation = maturity_derivation(row.get("maturity_date_json"))
    if derivation not in DERIVED_MATURITY_KINDS:
        return value
    if derivation == "name" and value.endswith("-12-31"):
        return value[:YEAR_TEXT_LENGTH]
    # Name-embedded full dates, month-end synthetics (#164), and start-plus-
    # tenor arithmetic (#166) are all month-trustworthy but not day-exact.
    return value[:MONTH_TEXT_LENGTH]


# Maturities the extractor derived rather than read off a stated date: from
# the instrument's name, or computed as start plus tenor (#166).
DERIVED_MATURITY_KINDS = frozenset({"name", "computed"})


def maturity_derivation(payload_text: object) -> str | None:
    """Return one maturity payload's derived_from marker."""
    try:
        payload = json.loads(str(payload_text or "{}"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    derivation = payload.get("derived_from")
    return str(derivation) if isinstance(derivation, str) else None


def end_dates_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two normalized end dates can still describe one instrument.

    A bare four-digit year is a year-resolution value from a name-derived
    maturity (#128) and matches any date in that year; a seven-character
    ``YYYY-MM`` is month-resolution (#164) and matches any date in that month.
    Full stated dates — including a genuine December 31 — must agree exactly.
    """
    if not left or not right:
        return True
    if left == right:
        return True
    if left[:YEAR_TEXT_LENGTH] != right[:YEAR_TEXT_LENGTH]:
        return False
    if len(left) == YEAR_TEXT_LENGTH or len(right) == YEAR_TEXT_LENGTH:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) == MONTH_TEXT_LENGTH and longer[:MONTH_TEXT_LENGTH] == shorter


# `normalize_name_fingerprint` turns the decimal point into a token break, so a
# coupon arrives here as `4 375%` rather than `4.375%`; the separator is
# therefore optional. `(?<!\d)` keeps a maturity year out of the whole-number
# part, so `notes due 2028 5%` yields the rate and not `2028 5%`.
NAME_RATE_PATTERN = re.compile(r"(?<!\d)(\d{1,3})(?:[ .](\d{1,4}))?%")


def name_rate_tokens(fingerprint: str | None) -> frozenset[str]:
    """Return the coupon rates in one name fingerprint, as canonical numbers.

    Returns the rate's canonical numeric string rather than the matched text.
    Comparing the raw token compared only the fractional digits, because the
    pattern could not see past the token break: `4.375%` and `3.375%` both
    reduced to `375%`, so `name_rates_are_compatible` called two different
    coupons compatible and declined to refuse the merge it exists to refuse.
    """
    if not fingerprint:
        return frozenset()
    rates: set[str] = set()
    for whole, fraction in NAME_RATE_PATTERN.findall(fingerprint):
        try:
            rates.add(normalize_numeric_string(Decimal(f"{whole}.{fraction or 0}")))
        except InvalidOperation:
            continue
    return frozenset(rates)


NAME_STOPWORDS = frozenset({"the", "of", "and", "its", "new", "existing", "certain"})
NAME_CLASS_TOKEN = re.compile(r"^(?:[a-z]|[a-z]?-?\d+[a-z]?|\d+)$")
NAME_MATURITY_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
# Above this many mentions sharing one compatible name, the name is generic for
# that issuer and the relaxed key rule is off. FHLB Dallas files 67
# `Consolidated Obligation Bonds` with no dates and repeated round amounts, so
# without the gate a single amount collision merges dozens of distinct bonds.
NAME_CLASS_GATE = 2
# The shorter of two compatible names needs this many informative tokens, so a
# bare `note` cannot subsume every note one issuer has.
NAME_MIN_SHARED_TOKENS = 2


def name_fingerprint_tokens(fingerprint: str | None) -> frozenset[str]:
    """Return the informative tokens of one name fingerprint."""
    if not fingerprint:
        return frozenset()
    return frozenset(
        token for token in fingerprint.split() if token not in NAME_STOPWORDS
    )


def name_fingerprints_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two name fingerprints can name one instrument.

    Equality is too strict for the filing sequence: an announcement 8-K names
    `senior notes due 2034` and the closing names the same debt `7.500% senior
    notes due 2034`, so one name is the other plus the details settled since.
    One fingerprint being a token-subset of the other captures that.

    Guards, each of which cost real precision without it:

    - the shorter name needs two informative tokens, so a bare `note` cannot
      subsume every note the issuer has
    - coupon tokens present on both sides must intersect
    - the tokens that differ must not be *only* a class or tranche designator.
      `Tranche A Loan` is not a shortened `Tranche B Loan`, and Kestra Medical's
      four tranches collapse into one instrument without this.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = name_fingerprint_tokens(left)
    right_tokens = name_fingerprint_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if not (left_tokens <= right_tokens or right_tokens <= left_tokens):
        return False
    if min(len(left_tokens), len(right_tokens)) < NAME_MIN_SHARED_TOKENS:
        return False
    left_rates = name_rate_tokens(left)
    right_rates = name_rate_tokens(right)
    if left_rates and right_rates and not (left_rates & right_rates):
        return False
    return any(
        not NAME_CLASS_TOKEN.match(token) for token in left_tokens ^ right_tokens
    )


def name_fingerprint_is_identifying(fingerprint: str | None) -> bool:
    """Return whether a name fingerprint alone can identify one instrument.

    A coupon rate does it. So does a maturity year: within one CIK,
    `Convertible Senior Notes due 2031` picks out one debt, and requiring the
    coupon meant an announcement 8-K that had not priced yet could never attach
    to its own closing.
    """
    if not fingerprint:
        return False
    if NAME_RATE_PATTERN.search(fingerprint):
        return True
    return bool(NAME_MATURITY_YEAR_PATTERN.search(fingerprint))


def name_class_sizes(
    mention_index: dict[str, PreparedMention],
    existing_instruments: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Count, per mention, how many mentions of its CIK share a compatible name.

    A name shared by many of one issuer's mentions is a template rather than an
    identifier, so the relaxed key rule stands down for it.

    A synthesized prior state carries its successor's name verbatim, so it is
    not another instrument bearing that name: counting it widened the class past
    the gate and split a mention out of the cluster it had always joined (#203).
    Neither synthesized mentions nor a row whose members are all synthesized
    count here — the same exclusion `ClusterProfile.add_member` applies.
    """
    by_cik: dict[str, list[str | None]] = {}
    for mention in mention_index.values():
        if mention.cik is None or mention.synthesized_by is not None:
            continue
        by_cik.setdefault(mention.cik, []).append(mention.normalized_name_fingerprint)
    if existing_instruments is not None and not existing_instruments.empty:
        for row in existing_instruments.to_dict("records"):
            cik = coerce_optional_text(row.get("cik"))
            if cik is None or cik not in by_cik:
                continue
            if coerce_optional_bool(row.get("synthesized_only")):
                continue
            by_cik[cik].append(
                normalize_name_fingerprint(coerce_optional_text(row.get("name")))
            )
    sizes: dict[str, int] = {}
    for mention_id, mention in mention_index.items():
        if mention.cik is None:
            sizes[mention_id] = 1
            continue
        fingerprint = mention.normalized_name_fingerprint
        sizes[mention_id] = sum(
            1
            for other in by_cik.get(mention.cik, [])
            if other == fingerprint
            or name_fingerprints_are_compatible(fingerprint, other)
        )
    return sizes


def name_rates_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two name fingerprints can still describe one instrument."""
    left_rates = name_rate_tokens(left)
    right_rates = name_rate_tokens(right)
    if left_rates and right_rates:
        return bool(left_rates & right_rates)
    return True


def apply_lineage_inference_pass(
    artifact_root: str | Path,
    *,
    data_dir: Path | None = None,
    renew: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Infer amendment lineage across the whole corpus, after all shards match.

    This cannot run inside `match_tables`: that is called once per shard batch
    and sees only the clusters a batch touched (measured at 1-7 rows per call on
    a 364-item window), so no rule can ever see both states of one facility. The
    rules need every cluster for a CIK at once, which only exists after the shard
    loop has written them all.

    Every pointer this pass wrote before is re-opened and re-derived, so the
    published lineage is a function of the current rules and the current rows,
    not of which run happened to write first. `infer_amendment_parents` only
    considers rows whose pointer is null, and an ordinary rematch carries an
    existing pointer forward from disk — so without this, a link a tightened
    rule now refuses (the EQT/EQM cross-borrower link #197's guard was written
    to remove) survived on every already-matched root and was republished by the
    next plain `cdt match`; 14 of 542 pointers differed from a clean rebuild
    (#204). `amendment_inferred_by` is what distinguishes those rows: it is set
    only by this pass and cleared whenever an extracted pointer takes over, so
    an extracted relation is never re-opened.

    Rewrites `amendment_of_debt_instrument_id` and re-derives the rollup columns
    from the updated pointers, so `superseded_by`, `lineage_family_id` and
    `is_lineage_head` stay consistent. ``renew`` extends the caller's writer
    lease: this pass reads three whole datasets and rewrites every shard, which
    can outlast a lease TTL between two phases that renew it (#89).
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

    # Before inferring, not only after: the rules read `first_seen_filing_date`
    # and the rollup below rewrites it, so inferring against the on-disk value
    # made this pass a function of how many times it had already run (#211).
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
    # Same shard assignment as `match_pending_mentions`: a null cik must map to
    # the shard the matcher put it in, or the rewrite lands the row in a second
    # shard and the original copy is never removed (#204).
    frame["_shard"] = (
        frame["cik"].fillna("").map(lambda value: shard_for_cik(str(value)))
    )
    for cik_shard, shard_rows in frame.groupby("_shard"):
        if renew is not None:
            renew()
        write_partition_table(
            debt_instruments_root(resolved_root, data_dir=data_dir),
            partition={"cik_shard": str(cik_shard)},
            table=shard_rows.drop(columns=["_shard"]).reindex(
                columns=DEBT_INSTRUMENT_COLUMNS
            ),
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
