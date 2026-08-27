"""Matcher stage for consolidating debt instrument mentions into stable clusters."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

import pandas as pd

from cdt.datasets import (
    cik_shard_partition_path,
    dataset_root,
    resolve_artifact_root,
    run_manifest_path,
    shard_for_cik,
)
from cdt.extractor.core import (
    DEBT_INSTRUMENT_MENTION_COLUMNS as EXTRACTED_MENTION_COLUMNS,
)
from cdt.extractor.core import MENTIONS_DATASET_NAME
from cdt.storage import (
    coerce_dataset_text,
    read_dataset,
    write_json_artifact,
    write_partition_table,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RELATED_THRESHOLD = 0.75
DEFAULT_MEMBERSHIP_THRESHOLD = 0.90
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_LENDER_SUPPORT_THRESHOLD = 0.5
MATCHER_SCHEMA_VERSION = 3
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
    "retired_of_debt_instrument_id",
    "split_of_debt_instrument_id",
    "name",
    "start_date",
    "end_date",
    "amount",
    "lenders_json",
    "other_interested_parties_json",
    "lenders_known_incomplete",
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
    start_date: str | None
    end_date: str | None
    amount: str | None
    amendment_of: str | None
    retired_of: str | None
    split_of: str | None
    lenders_json: str
    lenders_known_incomplete: bool
    other_interested_parties_json: str
    normalized_amount: str | None
    normalized_start_date: str | None
    normalized_end_date: str | None
    normalized_name_fingerprint: str | None
    lender_signature: str


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

    def add_member(self: ClusterProfile, mention: PreparedMention) -> None:
        """Update the cluster cache with one newly accepted member."""
        if mention.debt_instrument_mention_id not in self.member_ids:
            self.member_ids.append(mention.debt_instrument_mention_id)
        if mention.normalized_amount:
            self.normalized_amounts.add(mention.normalized_amount)
        if mention.normalized_start_date:
            self.normalized_start_dates.add(mention.normalized_start_date)
        if mention.normalized_end_date:
            self.normalized_end_dates.add(mention.normalized_end_date)
        if mention.normalized_name_fingerprint:
            self.normalized_name_fingerprints.add(mention.normalized_name_fingerprint)
        if mention.lender_signature:
            self.lender_signatures.add(mention.lender_signature)
        for target in (mention.amendment_of, mention.retired_of, mention.split_of):
            if target:
                self.relation_target_ids.add(target)


@dataclass(frozen=True)
class CandidateScore:
    """One scored mention-to-cluster candidate relationship."""

    debt_instrument_id: str
    match_score: float
    support_family: str | None
    basis: str = "amount_start"

    @property
    def base_match_via(self: CandidateScore) -> str:
        """Return the explanation family without the outcome prefix."""
        if self.basis != "amount_start" or self.support_family is None:
            return self.basis
        return f"amount_start+{self.support_family}"


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
    """Match in-memory debt instrument mentions into stable debt instrument clusters."""
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
    return {
        "debt_instrument_mentions": combined_edges.reindex(
            columns=MENTION_CLUSTER_EDGE_COLUMNS
        ),
        "debt_instrument": pd.DataFrame(
            debt_instrument_rows, columns=DEBT_INSTRUMENT_COLUMNS
        ),
    }


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
            cik=coerce_optional_text(instrument_row.get("cik")) or "",
            seed_mention_id=seed_mention_id,
            member_ids=list(member_ids),
            normalized_amounts=set(),
            normalized_start_dates=set(),
            normalized_end_dates=set(),
            normalized_name_fingerprints=set(),
            lender_signatures=set(),
        )
        normalized_amount = normalize_amount(
            coerce_optional_text(instrument_row.get("amount"))
        )
        if normalized_amount:
            profile.normalized_amounts.add(normalized_amount)
        normalized_start_date = normalize_date(
            coerce_optional_text(instrument_row.get("start_date"))
        )
        if normalized_start_date:
            profile.normalized_start_dates.add(normalized_start_date)
        normalized_end_date = normalize_date(
            coerce_optional_text(instrument_row.get("end_date"))
        )
        if normalized_end_date:
            profile.normalized_end_dates.add(normalized_end_date)
        normalized_name = normalize_name_fingerprint(
            coerce_optional_text(instrument_row.get("name"))
        )
        if normalized_name:
            profile.normalized_name_fingerprints.add(normalized_name)
        lenders = lender_signature(instrument_row.get("lenders_json"))
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


def score_candidates_for_mention(
    mention: PreparedMention,
    profiles: dict[str, ClusterProfile],
    *,
    strong_match_threshold: float,
    loose_match_threshold: float,
) -> list[CandidateScore]:
    """Return scored candidate clusters for one mention."""
    del loose_match_threshold
    if mention.cik is None:
        return []
    has_match_keys = (
        mention.normalized_amount is not None
        and mention.normalized_start_date is not None
    )
    if not has_match_keys and not name_fingerprint_is_identifying(
        mention.normalized_name_fingerprint
    ):
        return []
    candidates: list[CandidateScore] = []
    for profile in profiles.values():
        if profile.cik != mention.cik:
            continue
        if mention.amendment_of and mention.amendment_of in profile.member_ids:
            continue
        if mention.retired_of and mention.retired_of in profile.member_ids:
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
        if not keys_match:
            # Launch, pricing, and closing 8-Ks for one offering drift on
            # amount (upsizes) and start date (pricing vs settlement), so an
            # identifying fingerprint may attach a mention whose keys conflict.
            if name_fingerprint_is_identifying(
                mention.normalized_name_fingerprint
            ) and (
                mention.normalized_name_fingerprint
                in profile.normalized_name_fingerprints
            ):
                candidates.append(
                    CandidateScore(
                        debt_instrument_id=profile.debt_instrument_id,
                        match_score=round(strong_match_threshold, 4),
                        support_family="name",
                        basis="name_fingerprint",
                    )
                )
            continue
        lender_similarity = max(
            (
                lender_similarity_score(mention.lender_signature, candidate_signature)
                for candidate_signature in profile.lender_signatures
                if mention.lender_signature and candidate_signature
            ),
            default=0.0,
        )
        name_support = (
            1.0
            if mention.normalized_name_fingerprint
            and mention.normalized_name_fingerprint
            in profile.normalized_name_fingerprints
            else 0.0
        )
        name_conflict = bool(
            mention.normalized_name_fingerprint is not None
            and profile.normalized_name_fingerprints
            and mention.normalized_name_fingerprint
            not in profile.normalized_name_fingerprints
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
        if existing_amendment:
            amendment_parents.add(existing_amendment)
        existing_retired = coerce_optional_text(
            existing_row.get("retired_of_debt_instrument_id")
        )
        if existing_retired:
            retired_parents.add(existing_retired)
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
            if (
                mention.retired_of in mention_to_instrument
                and mention_to_instrument[mention.retired_of] != debt_instrument_id
            ):
                retired_parents.add(mention_to_instrument[mention.retired_of])
            if (
                mention.split_of in mention_to_instrument
                and mention_to_instrument[mention.split_of] != debt_instrument_id
            ):
                split_parents.add(mention_to_instrument[mention.split_of])
        if (
            len(amendment_parents) > 1
            or len(retired_parents) > 1
            or len(split_parents) > 1
        ):
            parent_links[debt_instrument_id] = {
                "amendment_of_debt_instrument_id": None,
                "retired_of_debt_instrument_id": None,
                "split_of_debt_instrument_id": None,
            }
            continue
        parent_types_present = sum(
            bool(parent_set)
            for parent_set in (amendment_parents, retired_parents, split_parents)
        )
        if parent_types_present > 1:
            parent_links[debt_instrument_id] = {
                "amendment_of_debt_instrument_id": None,
                "retired_of_debt_instrument_id": None,
                "split_of_debt_instrument_id": None,
            }
            continue
        parent_links[debt_instrument_id] = {
            "amendment_of_debt_instrument_id": next(iter(amendment_parents), None),
            "retired_of_debt_instrument_id": next(iter(retired_parents), None),
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
        lenders_json = json.dumps(
            dedupe_party_clusters(
                [
                    str(existing_row.get("lenders_json") or "[]"),
                    *[
                        mention_index[mention_id].lenders_json
                        for mention_id in present_member_ids
                    ],
                ]
            ),
            sort_keys=True,
        )
        lenders_known_incomplete = coerce_flag(
            existing_row.get("lenders_known_incomplete")
        ) or any(
            mention_index[mention_id].lenders_known_incomplete
            for mention_id in present_member_ids
        )
        other_interested_parties_json = json.dumps(
            dedupe_party_clusters(
                [
                    str(existing_row.get("other_interested_parties_json") or "[]"),
                    *[
                        mention_index[mention_id].other_interested_parties_json
                        for mention_id in present_member_ids
                    ],
                ]
            ),
            sort_keys=True,
        )
        rows.append(
            {
                "debt_instrument_id": debt_instrument_id,
                "cik": cik,
                # Fall back to the filer name any mention for this CIK carries, so
                # one member mention without display metadata cannot blank the page.
                "company_name": first_non_null(
                    ordered_member_ids, mention_index, "company_name"
                )
                or coerce_optional_text(existing_row.get("company_name"))
                or (company_names or {}).get(cik),
                "seed_debt_instrument_mention_id": seed_mention_id,
                "amendment_of_debt_instrument_id": parent_links.get(
                    debt_instrument_id, {}
                ).get("amendment_of_debt_instrument_id"),
                "retired_of_debt_instrument_id": parent_links.get(
                    debt_instrument_id, {}
                ).get("retired_of_debt_instrument_id"),
                "split_of_debt_instrument_id": parent_links.get(
                    debt_instrument_id, {}
                ).get("split_of_debt_instrument_id"),
                "name": first_non_null(ordered_member_ids, mention_index, "name")
                or coerce_optional_text(existing_row.get("name")),
                "start_date": first_non_null(
                    ordered_member_ids, mention_index, "start_date"
                )
                or coerce_optional_text(existing_row.get("start_date")),
                "end_date": first_non_null(
                    ordered_member_ids, mention_index, "end_date"
                )
                or coerce_optional_text(existing_row.get("end_date")),
                "amount": first_non_null(ordered_member_ids, mention_index, "amount")
                or coerce_optional_text(existing_row.get("amount")),
                "lenders_json": lenders_json,
                "lenders_known_incomplete": lenders_known_incomplete,
                "other_interested_parties_json": other_interested_parties_json,
            }
        )
    rows_by_id = {str(row["debt_instrument_id"]): row for row in rows}
    for _child_id, row in rows_by_id.items():
        parent_id = coerce_optional_text(row.get("retired_of_debt_instrument_id"))
        if not parent_id or parent_id not in rows_by_id:
            continue
        child_end_date = coerce_optional_text(row.get("end_date"))
        if child_end_date and not coerce_optional_text(
            rows_by_id[parent_id].get("end_date")
        ):
            rows_by_id[parent_id]["end_date"] = child_end_date
    return [rows_by_id[str(row["debt_instrument_id"])] for row in rows]


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


def dedupe_party_clusters(payloads: list[str]) -> list[dict[str, object]]:
    """Return deduped party cluster payloads."""
    deduped: dict[str, dict[str, object]] = {}
    for payload in payloads:
        for cluster in parse_cluster_list(payload):
            key = cluster_canonical_key(cluster)
            if key and key not in deduped:
                deduped[key] = cluster
    return [deduped[key] for key in sorted(deduped)]


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
    mentions = cluster.get("mentions", [])
    if not isinstance(mentions, list):
        return ""
    texts = [
        normalize_party_text(str(mention.get("text", "")))
        for mention in mentions
        if isinstance(mention, dict) and mention.get("text")
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
        cik=coerce_optional_text(row.get("cik")),
        company_name=coerce_optional_text(row.get("company_name")),
        date=coerce_optional_text(row.get("date")),
        name=coerce_optional_text(row.get("name")),
        start_date=coerce_optional_text(row.get("start_date")),
        end_date=coerce_optional_text(row.get("end_date")),
        amount=coerce_optional_text(row.get("amount")),
        amendment_of=coerce_optional_text(row.get("amendment_of")),
        retired_of=coerce_optional_text(row.get("retired_of")),
        split_of=coerce_optional_text(row.get("split_of")),
        lenders_json=str(row.get("lenders_json") or "[]"),
        lenders_known_incomplete=coerce_flag(row.get("lenders_known_incomplete")),
        other_interested_parties_json=str(
            row.get("other_interested_parties_json") or "[]"
        ),
        normalized_amount=normalize_amount(coerce_optional_text(row.get("amount"))),
        normalized_start_date=normalize_date(
            coerce_optional_text(row.get("start_date"))
        ),
        normalized_end_date=normalize_date(coerce_optional_text(row.get("end_date"))),
        normalized_name_fingerprint=normalize_name_fingerprint(
            coerce_optional_text(row.get("name"))
        ),
        lender_signature=lender_signature(row.get("lenders_json")),
    )


def mention_sort_key(mention: PreparedMention) -> tuple[str, str, str, str]:
    """Return deterministic processing order for cluster assignment."""
    return (
        mention.date or "",
        mention.accession_number or "",
        mention.item_id,
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


def coerce_flag(value: object) -> bool:
    """Return one boolean flag, treating missing parquet values as False."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def coerce_optional_text(value: object) -> str | None:
    """Return one trimmed string or None, treating placeholder text as missing."""
    return coerce_dataset_text(value)


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
    """Normalize debt-instrument names for exact comparisons."""
    if value is None:
        return None
    text = value.lower()
    text = re.sub(r"(\d+)\.(\d*?[1-9])0+(?=%)", r"\1.\2", text)
    text = re.sub(r"(\d+)\.0+(?=%)", r"\1", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def lender_keys(value: object) -> list[str]:
    """Return normalized lender cluster keys in deterministic order."""
    keys: list[str] = []
    for cluster in parse_cluster_list(str(value or "[]")):
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


def lender_similarity_score(left: str, right: str) -> float:
    """Return one deterministic similarity score for lender strings."""
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(a=left, b=right).ratio(), 4)


def end_dates_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two normalized end dates can still describe one instrument."""
    if not left or not right:
        return True
    if left == right:
        return True
    if left[:4] != right[:4]:
        return False
    # A YYYY-12-31 value may come from a year-only maturity such as "due 2030",
    # so it is only year-resolution evidence and matches any date in that year.
    return left.endswith("-12-31") or right.endswith("-12-31")


NAME_RATE_PATTERN = re.compile(r"\d+(?:\.\d+)?%")


def name_rate_tokens(fingerprint: str | None) -> frozenset[str]:
    """Return the coupon-rate tokens embedded in one name fingerprint."""
    if not fingerprint:
        return frozenset()
    return frozenset(NAME_RATE_PATTERN.findall(fingerprint))


def name_fingerprint_is_identifying(fingerprint: str | None) -> bool:
    """Return whether a name fingerprint alone can identify one instrument.

    A coupon rate is required: one issuer group can announce several
    generic "Senior Secured Notes due YYYY" through different subsidiaries,
    so a class-plus-year name is not identifying on its own.
    """
    if not fingerprint:
        return False
    return bool(NAME_RATE_PATTERN.search(fingerprint))


def name_rates_are_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two name fingerprints can still describe one instrument."""
    left_rates = name_rate_tokens(left)
    right_rates = name_rate_tokens(right)
    if left_rates and right_rates:
        return bool(left_rates & right_rates)
    return True
