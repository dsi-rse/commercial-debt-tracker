"""The matcher stage: select shards to match and write their edges and instruments."""

from __future__ import annotations

import logging
from collections.abc import Callable
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
from cdt.extractor.outputs import MENTIONS_DATASET_NAME
from cdt.extractor.schema import (
    DEBT_INSTRUMENT_MENTION_COLUMNS as EXTRACTED_MENTION_COLUMNS,
)
from cdt.matcher.compat import name_class_sizes
from cdt.matcher.instruments import (
    apply_lifecycle_rollup,
    build_debt_instrument_rows,
    company_names_by_cik,
    derive_parent_links,
)
from cdt.matcher.normalize import (
    borrowed_lender_signature,
    mention_sort_key,
    prepare_mention,
)
from cdt.matcher.schema import (
    DEBT_INSTRUMENT_COLUMNS,
    DEBT_INSTRUMENT_DATASET_NAME,
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    MATCHER_SCHEMA_VERSION,
    MENTION_CLUSTER_EDGE_COLUMNS,
    debt_instruments_root,
    mention_cluster_edges_root,
)
from cdt.matcher.scoring import (
    build_cluster_profiles,
    build_empty_profile,
    build_member_groups,
    resolve_candidates,
    score_candidates_for_mention,
)
from cdt.storage.objects import artifact_exists, read_json_artifact, write_json_artifact
from cdt.storage.tables import read_dataset, write_partition_table

LOGGER = logging.getLogger(__name__)


def _stale_schema_forces_rematch(
    resolved_root: str,
    *,
    data_dir: Path | None = None,
) -> bool:
    """Return True when the root's latest match manifest records an older schema.

    False when there is no match manifest or its `schema_version` is not an
    int. Mention ids are content hashes, so clusters from an older schema can
    be keyed on ids the mentions dataset no longer contains; the caller forces
    a full rematch instead of refusing or proceeding.
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
    """Match the root's mentions shard by shard and write edges, instruments and manifest.

    Existing edges and instruments are extended unless ``force`` is set or the
    root was matched at an older ``MATCHER_SCHEMA_VERSION``, in which case each
    shard is rebuilt from scratch. ``renew`` is called before each shard is
    rewritten and must raise if the writer lease has been lost. Returns
    ``{"debt_instrument_mentions": edges, "debt_instrument": instruments}``
    for every shard written (empty frames when there are no mentions).

    Raises:
        ValueError: ``batch_size`` is not positive.
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
    """Match in-memory mentions into clusters, extending any existing edges and rows.

    Returns ``{"debt_instrument_mentions": edges, "debt_instrument": instruments}``.
    Mentions with no CIK are skipped; mentions already holding a member edge
    keep it.
    Amendment lineage is not inferred here, because one shard batch never sees
    every cluster of a CIK; ``apply_lineage_inference_pass`` does that.

    Raises:
        ValueError: ``strong_match_threshold < loose_match_threshold`` or
            ``ambiguity_margin`` is negative.
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
