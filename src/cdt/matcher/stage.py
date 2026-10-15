"""The matcher stage: re-cluster every CIK whose mentions changed, and write its rows.

Matching is a pure function of one CIK's mentions. A run finds the CIKs whose
set of mention ids differs from what the last run matched (a digest per CIK,
stored per ``cik_shard``), clusters each from scratch, assigns ids
(:mod:`cdt.matcher.identity`), infers lineage, and rewrites only the shards
those CIKs live in.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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
from cdt.extractor.outputs import MENTIONS_DATASET_NAME
from cdt.extractor.schema import (
    DEBT_INSTRUMENT_MENTION_COLUMNS as EXTRACTED_MENTION_COLUMNS,
)
from cdt.matcher.compat import name_class_sizes
from cdt.matcher.identity import (
    PreviousClusters,
    assign_instrument_ids,
    updated_redirects,
)
from cdt.matcher.instruments import (
    apply_lifecycle_rollup,
    apply_observation_columns,
    build_debt_instrument_rows,
    company_names_by_cik,
    derive_parent_links,
)
from cdt.matcher.lineage_inference import apply_inferred_lineage
from cdt.matcher.normalize import (
    borrowed_lender_signature,
    mention_sort_key,
    prepare_mention,
)
from cdt.matcher.schema import (
    DEBT_INSTRUMENT_COLUMNS,
    DEBT_INSTRUMENT_DATASET_NAME,
    DEBT_INSTRUMENT_REDIRECT_COLUMNS,
    DEBT_INSTRUMENT_REDIRECT_DATASET_NAME,
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_RELATED_THRESHOLD,
    MATCHER_SCHEMA_VERSION,
    MENTION_CLUSTER_EDGE_COLUMNS,
    MENTION_CLUSTER_EDGE_DATASET_NAME,
    ClusterProfile,
    PreparedMention,
    debt_instrument_redirects_root,
    debt_instruments_root,
    mention_cluster_edges_root,
)
from cdt.matcher.scoring import (
    build_empty_profile,
    build_member_groups,
    resolve_candidates,
    score_candidates_for_mention,
)
from cdt.storage.objects import (
    artifact_exists,
    join_artifact_path,
    list_artifacts,
    read_json_artifact,
    write_json_artifact,
)
from cdt.storage.tables import read_dataset, read_table, write_partition_table

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchThresholds:
    """The scoring thresholds one match run uses."""

    strong: float = DEFAULT_MEMBERSHIP_THRESHOLD
    loose: float = DEFAULT_RELATED_THRESHOLD
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN

    def __post_init__(self: MatchThresholds) -> None:
        """Reject thresholds the scoring rules cannot use.

        Raises:
            ValueError: ``strong < loose`` or ``ambiguity_margin`` is negative.
        """
        if self.strong < self.loose:
            raise ValueError("strong_match_threshold must be >= loose_match_threshold")
        if self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be non-negative")


@dataclass
class CikMatch:
    """One CIK's matcher output: edges, instruments, redirects and lineage links."""

    edges: pd.DataFrame
    instruments: pd.DataFrame
    redirects: pd.DataFrame
    inferred_links: int = 0


@dataclass
class MatchResult:
    """What one match run rewrote: the rows of every CIK it matched."""

    edges: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=MENTION_CLUSTER_EDGE_COLUMNS)
    )
    instruments: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS)
    )
    ciks_matched: int = 0
    #: CIKs whose last mention went away, so their rows were deleted.
    ciks_removed: int = 0
    partitions_written: list[str] = field(default_factory=list)


def cik_digests_root(artifact_root: str | Path | None = None) -> str:
    """Return the prefix holding the match stage's per-shard CIK digests."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root), "runs", "match", "cik-digests"
    )


def cik_digests_path(cik_shard: str, artifact_root: str | Path | None = None) -> str:
    """Return the digest file of one ``cik_shard``."""
    return join_artifact_path(
        cik_digests_root(artifact_root), f"cik_shard={cik_shard}.json"
    )


def mention_set_digest(mention_ids: list[str]) -> str:
    """Return the digest of one CIK's set of mention ids."""
    payload = "\n".join(sorted(set(mention_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stale_schema_forces_rematch(
    resolved_root: str,
) -> bool:
    """Return True when the root's latest match manifest records an older schema.

    False when there is no match manifest or its `schema_version` is not an
    int. The caller re-clusters every CIK instead of refusing or proceeding.
    """
    manifest_path = run_manifest_path(
        "match",
        "latest",
        artifact_root=resolved_root,
    )
    if not artifact_exists(manifest_path):
        return False
    recorded = read_json_artifact(manifest_path).get("schema_version")
    if not isinstance(recorded, int) or recorded >= MATCHER_SCHEMA_VERSION:
        return False
    LOGGER.warning(
        "Matcher schema is %s but %s was matched at %s; re-clustering every CIK",
        MATCHER_SCHEMA_VERSION,
        resolved_root,
        recorded,
    )
    return True


@dataclass(frozen=True)
class CikMentions:
    """What two columns of every mentions file say about each normalized CIK."""

    digests: dict[str, str]
    #: The CIK's spellings as stored, which a filtered read must match.
    spellings: dict[str, set[str]]
    counts: dict[str, int]


def _current_cik_mentions(resolved_root: str) -> CikMentions:
    """Return each normalized CIK's mention-set digest, stored spellings and count.

    Reads two columns of every mentions month file. Mentions with no CIK are
    never matched, so they are left out.
    """
    table = read_dataset(
        dataset_root(MENTIONS_DATASET_NAME, artifact_root=resolved_root),
        columns=["cik", "debt_instrument_mention_id"],
    )
    ids: dict[str, list[str]] = {}
    spellings: dict[str, set[str]] = {}
    for raw_cik, mention_id in zip(
        table["cik"], table["debt_instrument_mention_id"], strict=True
    ):
        if raw_cik is None or pd.isna(raw_cik) or not str(raw_cik).strip():
            continue
        cik = normalize_cik(raw_cik)
        ids.setdefault(cik, []).append(str(mention_id))
        spellings.setdefault(cik, set()).add(str(raw_cik))
    return CikMentions(
        digests={cik: mention_set_digest(values) for cik, values in ids.items()},
        spellings=spellings,
        counts={cik: len(values) for cik, values in ids.items()},
    )


def _stored_cik_digests(resolved_root: str) -> dict[str, dict[str, str]]:
    """Return the digests the last runs stored, by ``cik_shard`` then CIK."""
    stored: dict[str, dict[str, str]] = {}
    for path in list_artifacts(cik_digests_root(resolved_root), suffix=".json"):
        name = path.rsplit("/", 1)[-1]
        if not name.startswith("cik_shard="):
            continue
        payload = read_json_artifact(path)
        ciks = payload.get("ciks") if isinstance(payload, dict) else None
        if isinstance(ciks, dict):
            stored[name.removeprefix("cik_shard=").removesuffix(".json")] = {
                str(cik): str(digest) for cik, digest in ciks.items()
            }
    return stored


def _existing_shards(resolved_root: str) -> set[str]:
    """Return every ``cik_shard`` the matcher's datasets hold a file for."""
    shards: set[str] = set()
    for root in (
        mention_cluster_edges_root(resolved_root),
        debt_instruments_root(resolved_root),
        debt_instrument_redirects_root(resolved_root),
    ):
        for path in list_artifacts(root, suffix=".parquet"):
            for segment in path.split("/"):
                if segment.startswith("cik_shard="):
                    shards.add(segment.removeprefix("cik_shard="))
    return shards


def _read_shard(
    dataset_name: str, cik_shard: str, columns: list[str], resolved_root: str
) -> pd.DataFrame:
    """Read one ``cik_shard`` file of a matcher dataset; empty when absent."""
    path = cik_shard_partition_path(
        dataset_name, cik_shard=cik_shard, artifact_root=resolved_root
    )
    return read_table(path, columns).reindex(columns=columns)


#: Mention rows one filtered read of the mentions dataset may return: the
#: affected CIKs of consecutive shards are read together up to this many, so a
#: run reads the dataset once per batch rather than once per shard.
MENTION_READ_BATCH_ROWS = 50_000


def _shard_batches(
    shards: list[str], by_shard: dict[str, set[str]], counts: dict[str, int]
) -> list[list[str]]:
    """Group sorted shards into batches of at most ``MENTION_READ_BATCH_ROWS`` mentions."""
    batches: list[list[str]] = []
    batch: list[str] = []
    rows = 0
    for cik_shard in shards:
        shard_rows = sum(counts.get(cik, 0) for cik in by_shard.get(cik_shard, ()))
        if batch and rows + shard_rows > MENTION_READ_BATCH_ROWS:
            batches.append(batch)
            batch, rows = [], 0
        batch.append(cik_shard)
        rows += shard_rows
    if batch:
        batches.append(batch)
    return batches


def match_pending_mentions(
    *,
    artifact_root: str | Path | None = None,
    force: bool = False,
    remint: bool = False,
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    renew: Callable[[], None] | None = None,
) -> MatchResult:
    """Re-cluster every CIK whose mentions changed and rewrite its shards.

    A CIK is matched when its mention-set digest is new, changed or gone since
    the last run stored it; ``force`` (or a root matched at an older
    ``MATCHER_SCHEMA_VERSION``) matches every CIK and rewrites every shard.
    ``remint`` does the same and also replaces every instrument id with its
    content id (:func:`cdt.matcher.identity.assign_instrument_ids`). Each
    affected CIK is clustered from all of its current mentions only, so the
    result never depends on the order mentions arrived in. Shards holding no
    affected CIK are not touched. A shard's digests are written after its
    tables, so a run that stops part-way is finished by the next one.
    ``renew`` is called before each shard is rewritten and must raise if the
    writer lease has been lost.
    """
    thresholds = MatchThresholds(
        strong=strong_match_threshold,
        loose=loose_match_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    resolved_root = resolve_artifact_root(artifact_root)
    rebuild_all = force or remint or _stale_schema_forces_rematch(resolved_root)
    current_mentions = _current_cik_mentions(resolved_root)
    current = current_mentions.digests
    stored = _stored_cik_digests(resolved_root)
    stored_by_cik = {
        cik: digest
        for shard_digests in stored.values()
        for cik, digest in shard_digests.items()
    }
    affected = {
        cik
        for cik in set(current) | set(stored_by_cik)
        if rebuild_all or current.get(cik) != stored_by_cik.get(cik)
    }
    by_shard: dict[str, set[str]] = {}
    for cik in affected:
        by_shard.setdefault(shard_for_cik(cik), set()).add(cik)
    shards = sorted(
        set(by_shard) | (_existing_shards(resolved_root) if rebuild_all else set())
    )

    result = MatchResult()
    edge_frames: list[pd.DataFrame] = []
    instrument_frames: list[pd.DataFrame] = []
    inferred_links = 0
    progress = 0
    for batch in _shard_batches(shards, by_shard, current_mentions.counts):
        batch_mentions = read_dataset(
            dataset_root(MENTIONS_DATASET_NAME, artifact_root=resolved_root),
            columns=EXTRACTED_MENTION_COLUMNS,
            where={
                "cik": sorted(
                    spelling
                    for cik_shard in batch
                    for cik in by_shard.get(cik_shard, ())
                    for spelling in current_mentions.spellings.get(cik, ())
                )
            },
        )
        batch_mentions = batch_mentions.assign(
            cik=batch_mentions["cik"].map(normalize_cik)
        )
        mentions_by_shard = dict(
            tuple(batch_mentions.groupby(batch_mentions["cik"].map(shard_for_cik)))
        )
        for cik_shard in batch:
            progress += 1
            started = perf_counter()
            shard_ciks = by_shard.get(cik_shard, set())
            mentions = mentions_by_shard.get(cik_shard, batch_mentions.iloc[0:0])
            edges = _read_shard(
                MENTION_CLUSTER_EDGE_DATASET_NAME,
                cik_shard,
                MENTION_CLUSTER_EDGE_COLUMNS,
                resolved_root,
            )
            instruments = _read_shard(
                DEBT_INSTRUMENT_DATASET_NAME,
                cik_shard,
                DEBT_INSTRUMENT_COLUMNS,
                resolved_root,
            )
            redirects = _read_shard(
                DEBT_INSTRUMENT_REDIRECT_DATASET_NAME,
                cik_shard,
                DEBT_INSTRUMENT_REDIRECT_COLUMNS,
                resolved_root,
            )
            edges_by_cik = dict(tuple(edges.groupby("cik")))
            redirects_by_cik = dict(tuple(redirects.groupby("cik")))
            new_edges: list[pd.DataFrame] = []
            new_instruments: list[pd.DataFrame] = []
            new_redirects: list[pd.DataFrame] = []
            for cik, cik_mentions in mentions.groupby("cik", sort=True):
                matched = match_cik(
                    cik_mentions,
                    cik=str(cik),
                    previous_edges=edges_by_cik.get(cik),
                    previous_redirects=redirects_by_cik.get(cik),
                    thresholds=thresholds,
                    remint=remint,
                )
                new_edges.append(matched.edges)
                new_instruments.append(matched.instruments)
                new_redirects.append(matched.redirects)
                inferred_links += matched.inferred_links
                result.ciks_matched += 1
            result.ciks_removed += len(shard_ciks - set(mentions["cik"]))

            if renew is not None:
                renew()
            replaced = set(mentions["cik"]) | shard_ciks
            for root, kept, frames, columns in (
                (
                    mention_cluster_edges_root(resolved_root),
                    edges,
                    new_edges,
                    MENTION_CLUSTER_EDGE_COLUMNS,
                ),
                (
                    debt_instruments_root(resolved_root),
                    instruments,
                    new_instruments,
                    DEBT_INSTRUMENT_COLUMNS,
                ),
                (
                    debt_instrument_redirects_root(resolved_root),
                    redirects,
                    new_redirects,
                    DEBT_INSTRUMENT_REDIRECT_COLUMNS,
                ),
            ):
                survivors = (
                    pd.DataFrame(columns=columns)
                    if rebuild_all
                    else kept.loc[~kept["cik"].isin(replaced)]
                )
                parts = [frame for frame in (survivors, *frames) if not frame.empty]
                table = (
                    pd.concat(parts, ignore_index=True)
                    if parts
                    else pd.DataFrame(columns=columns)
                )
                write_partition_table(
                    root,
                    partition={"cik_shard": cik_shard},
                    table=table.reindex(columns=columns),
                )
            shard_digests = {} if rebuild_all else dict(stored.get(cik_shard, {}))
            for cik in replaced:
                shard_digests.pop(cik, None)
                if cik in current:
                    shard_digests[cik] = current[cik]
            write_json_artifact(
                cik_digests_path(cik_shard, resolved_root),
                {
                    "schema_version": MATCHER_SCHEMA_VERSION,
                    "ciks": dict(sorted(shard_digests.items())),
                },
            )
            edge_frames.extend(new_edges)
            instrument_frames.extend(new_instruments)
            result.partitions_written.append(
                cik_shard_partition_path(
                    DEBT_INSTRUMENT_DATASET_NAME,
                    cik_shard=cik_shard,
                    artifact_root=resolved_root,
                )
            )
            LOGGER.info(
                "Matcher shard complete: cik_shard=%s progress=%s/%s ciks=%s "
                "mentions=%s elapsed=%.1fs",
                cik_shard,
                progress,
                len(shards),
                len(replaced),
                len(mentions),
                perf_counter() - started,
            )

    populated_edges = [frame for frame in edge_frames if not frame.empty]
    if populated_edges:
        result.edges = pd.concat(populated_edges, ignore_index=True)
    populated_instruments = [frame for frame in instrument_frames if not frame.empty]
    if populated_instruments:
        result.instruments = pd.concat(populated_instruments, ignore_index=True)
    write_json_artifact(
        run_manifest_path("match", "latest", artifact_root=resolved_root),
        {
            "artifact_root": resolved_root,
            "stage": "match",
            "rebuild_all": rebuild_all,
            "remint": remint,
            "ciks_matched": result.ciks_matched,
            "ciks_removed": result.ciks_removed,
            "inferred_lineage_links": inferred_links,
            "partitions_written": result.partitions_written,
            "membership_threshold": thresholds.strong,
            "related_threshold": thresholds.loose,
            "ambiguity_margin": thresholds.ambiguity_margin,
            "schema_version": MATCHER_SCHEMA_VERSION,
        },
    )
    LOGGER.info(
        "Matcher complete: ciks_matched=%s ciks_removed=%s shards=%s "
        "debt_instruments=%s lineage_links=%s",
        result.ciks_matched,
        result.ciks_removed,
        len(shards),
        len(result.instruments),
        inferred_links,
    )
    return result


def cluster_mentions(
    mention_index: dict[str, PreparedMention],
    *,
    thresholds: MatchThresholds,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Cluster prepared mentions from scratch, in ``mention_sort_key`` order.

    Returns ``(mention id -> provisional cluster id, edge rows)``. Mentions
    with no CIK are skipped. A provisional id is the id of the mention that
    seeded the cluster.
    """
    profiles: dict[str, ClusterProfile] = {}
    class_sizes = name_class_sizes(mention_index)
    edge_rows: list[dict[str, object]] = []
    member_map: dict[str, str] = {}
    for mention_id in sorted(
        mention_index, key=lambda key: mention_sort_key(mention_index[key])
    ):
        mention = mention_index[mention_id]
        if mention.cik is None:
            continue
        candidates = score_candidates_for_mention(
            mention,
            profiles,
            strong_match_threshold=thresholds.strong,
            name_class_size=class_sizes.get(mention_id, 1),
            lender_signature=borrowed_lender_signature(mention, mention_index),
        )
        chosen_cluster_id, chosen_edge_rows = resolve_candidates(
            mention,
            candidates,
            strong_match_threshold=thresholds.strong,
            loose_match_threshold=thresholds.loose,
            ambiguity_margin=thresholds.ambiguity_margin,
            evaluated_run_id="latest",
        )
        edge_rows.extend(chosen_edge_rows)
        member_map[mention_id] = chosen_cluster_id
        if chosen_cluster_id not in profiles:
            profiles[chosen_cluster_id] = build_empty_profile(
                chosen_cluster_id, mention
            )
        profiles[chosen_cluster_id].add_member(mention)
    return member_map, edge_rows


def match_cik(
    mentions: pd.DataFrame,
    *,
    cik: str,
    previous_edges: pd.DataFrame | None = None,
    previous_redirects: pd.DataFrame | None = None,
    thresholds: MatchThresholds | None = None,
    remint: bool = False,
) -> CikMatch:
    """Match one CIK's mentions from scratch: clusters, ids, rows and lineage.

    ``mentions`` are every current mention of ``cik`` (a normalized CIK).
    ``previous_edges`` and ``previous_redirects`` are what the last match
    wrote for it: they decide which ids carry over and which redirect, never
    which mentions cluster together.
    """
    resolved_thresholds = thresholds or MatchThresholds()
    rows = mentions.to_dict("records")
    for row in rows:
        row["cik"] = cik
    mention_index = {
        str(row["debt_instrument_mention_id"]): prepare_mention(row) for row in rows
    }
    provisional_map, provisional_edges = cluster_mentions(
        mention_index, thresholds=resolved_thresholds
    )
    final_ids, retired = assign_instrument_ids(
        build_member_groups(provisional_map),
        cik=cik,
        mention_index=mention_index,
        previous=PreviousClusters.from_edges(
            previous_edges
            if previous_edges is not None
            else pd.DataFrame(columns=MENTION_CLUSTER_EDGE_COLUMNS)
        ),
        remint=remint,
    )
    member_map = {
        mention_id: final_ids[cluster_id]
        for mention_id, cluster_id in provisional_map.items()
    }
    member_groups = build_member_groups(member_map)
    edge_rows = [
        {
            **edge,
            "debt_instrument_id": final_ids[str(edge["debt_instrument_id"])],
            "cik": cik,
            "item_id": mention_index[str(edge["debt_instrument_mention_id"])].item_id,
        }
        for edge in provisional_edges
    ]
    instrument_rows = build_debt_instrument_rows(
        member_groups,
        mention_index,
        derive_parent_links(member_groups, mention_index, member_map),
        company_names=company_names_by_cik(mentions.assign(cik=cik)),
    )
    # The ordinal rule reads first-seen dates, so they are set before it runs.
    apply_observation_columns(
        instrument_rows, member_groups=member_groups, mention_index=mention_index
    )
    inferred_links = apply_inferred_lineage(instrument_rows)
    apply_lifecycle_rollup(
        instrument_rows, member_groups=member_groups, mention_index=mention_index
    )
    stored_redirects = (
        {
            str(row["old_debt_instrument_id"]): str(row["debt_instrument_id"])
            for row in previous_redirects.to_dict("records")
        }
        if previous_redirects is not None and not previous_redirects.empty
        else {}
    )
    redirects = updated_redirects(stored_redirects, retired, set(member_groups))
    return CikMatch(
        edges=pd.DataFrame(edge_rows, columns=MENTION_CLUSTER_EDGE_COLUMNS),
        instruments=pd.DataFrame(instrument_rows, columns=DEBT_INSTRUMENT_COLUMNS),
        redirects=pd.DataFrame(
            [
                {"old_debt_instrument_id": old, "debt_instrument_id": new, "cik": cik}
                for old, new in sorted(redirects.items())
            ],
            columns=DEBT_INSTRUMENT_REDIRECT_COLUMNS,
        ),
        inferred_links=inferred_links,
    )


def match_tables(
    debt_instrument_mentions: pd.DataFrame,
    *,
    strong_match_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    loose_match_threshold: float = DEFAULT_RELATED_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> dict[str, pd.DataFrame]:
    """Match in-memory mentions of any CIKs from scratch, with nothing carried over.

    Returns ``{"debt_instrument_mentions": edges, "debt_instrument":
    instruments}``. Mentions with no CIK are skipped.

    Raises:
        ValueError: ``strong_match_threshold < loose_match_threshold`` or
            ``ambiguity_margin`` is negative.
    """
    thresholds = MatchThresholds(
        strong=strong_match_threshold,
        loose=loose_match_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    with_cik = debt_instrument_mentions.loc[
        debt_instrument_mentions["cik"].notna()
        & debt_instrument_mentions["cik"].astype(str).str.strip().ne("")
    ]
    normalized = with_cik.assign(cik=with_cik["cik"].map(normalize_cik))
    matches = [
        match_cik(cik_mentions, cik=str(cik), thresholds=thresholds)
        for cik, cik_mentions in normalized.groupby("cik", sort=True)
    ]

    def combine(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        populated = [frame for frame in frames if not frame.empty]
        if not populated:
            return pd.DataFrame(columns=columns)
        return pd.concat(populated, ignore_index=True).reindex(columns=columns)

    return {
        "debt_instrument_mentions": combine(
            [match.edges for match in matches], MENTION_CLUSTER_EDGE_COLUMNS
        ),
        "debt_instrument": combine(
            [match.instruments for match in matches], DEBT_INSTRUMENT_COLUMNS
        ),
    }
