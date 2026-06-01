"""Matcher stage for consolidating debt instrument mentions into debt instruments."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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
from cdt.extractor.core import (
    MENTIONS_DATASET_NAME,
)
from cdt.storage import read_dataset, write_json_artifact, write_partition_table

LOGGER = logging.getLogger(__name__)
DEFAULT_LOOSE_MATCH_THRESHOLD = 0.75
DEFAULT_STRONG_MATCH_THRESHOLD = 0.90
MATCHER_STATUSES = ("singleton", "matched", "ambiguous")
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
DEBT_INSTRUMENT_MENTION_COLUMNS = [
    "debt_instrument_mention_id",
    "debt_instrument_id",
    "matcher_status",
]
DEBT_INSTRUMENT_COLUMNS = [
    "debt_instrument_id",
    "cik",
    "seed_debt_instrument_mention_id",
    "amendment_of_debt_instrument_id",
    "split_of_debt_instrument_id",
    "name",
    "start_date",
    "end_date",
    "amount",
    "direct_mentions_json",
    "lenders_json",
    "other_interested_parties_json",
    "possibly_related_json",
]
MENTION_MATCH_DATASET_NAME = "mention-matches"
DEBT_INSTRUMENT_DATASET_NAME = "debt-instruments"


def mention_matches_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical mention-matches dataset root."""
    return dataset_root(
        MENTION_MATCH_DATASET_NAME,
        artifact_root=artifact_root,
        data_dir=data_dir,
    )


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
class MentionSurface:
    """Normalized matching surface for one mention or one-hop lineage neighbor."""

    source_mention_id: str
    match_via: str
    normalized_amount: str | None
    normalized_start_date: str | None
    normalized_end_date: str | None
    normalized_name_fingerprint: str | None
    lender_signature: str


@dataclass(frozen=True)
class PreparedMention:
    """Denormalized mention record used during matching."""

    debt_instrument_mention_id: str
    item_id: str
    raw_id: str
    accession_number: str | None
    cik: str | None
    date: str | None
    name: str | None
    start_date: str | None
    end_date: str | None
    amount: str | None
    amendment_of: str | None
    split_of: str | None
    lenders_json: str
    other_interested_parties_json: str
    normalized_amount: str | None
    normalized_start_date: str | None
    normalized_end_date: str | None
    normalized_name_fingerprint: str | None
    lender_signature: str
    lender_evidence_state: str


def match_pending_mentions(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    strong_match_threshold: float = DEFAULT_STRONG_MATCH_THRESHOLD,
    loose_match_threshold: float = DEFAULT_LOOSE_MATCH_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    """Match canonical debt instrument mentions into canonical matcher outputs."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    del force
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
                columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            ),
            "debt_instrument": pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
        }
    mention_rows = mention_rows.copy()
    mention_rows["cik_shard"] = (
        mention_rows["cik"].fillna("").map(lambda value: shard_for_cik(str(value)))
    )
    mention_frames: list[pd.DataFrame] = []
    instrument_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    shard_groups = list(mention_rows.groupby("cik_shard"))[:batch_size]
    total_partitions = len(shard_groups)
    for partition_index, (cik_shard, shard_mentions) in enumerate(
        shard_groups, start=1
    ):
        partition_start = perf_counter()
        tables = match_tables(
            shard_mentions.drop(columns=["cik_shard"]),
            strong_match_threshold=strong_match_threshold,
            loose_match_threshold=loose_match_threshold,
        )
        mention_matches = tables["debt_instrument_mentions"].reindex(
            columns=DEBT_INSTRUMENT_MENTION_COLUMNS
        )
        debt_instruments = tables["debt_instrument"].reindex(
            columns=DEBT_INSTRUMENT_COLUMNS
        )
        write_partition_table(
            mention_matches_root(resolved_root, data_dir=data_dir),
            partition={"cik_shard": str(cik_shard)},
            table=mention_matches,
        )
        write_partition_table(
            debt_instruments_root(resolved_root, data_dir=data_dir),
            partition={"cik_shard": str(cik_shard)},
            table=debt_instruments,
        )
        mention_frames.append(mention_matches)
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
            "Matcher partition complete: cik_shard=%s progress=%s/%s mentions=%s matched_rows=%s debt_instruments=%s elapsed=%.1fs",
            cik_shard,
            partition_index,
            total_partitions,
            len(shard_mentions),
            len(mention_matches),
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
            "strong_match_threshold": strong_match_threshold,
            "loose_match_threshold": loose_match_threshold,
        },
    )
    LOGGER.info(
        "Matcher complete: mention_rows=%s debt_instruments=%s",
        sum(len(frame) for frame in mention_frames),
        sum(len(frame) for frame in instrument_frames),
    )
    return {
        "debt_instrument_mentions": pd.concat(mention_frames, ignore_index=True)
        if mention_frames
        else pd.DataFrame(columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
        "debt_instrument": pd.concat(instrument_frames, ignore_index=True)
        if instrument_frames
        else pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
    }


def match_tables(
    debt_instrument_mentions: pd.DataFrame,
    *,
    strong_match_threshold: float = DEFAULT_STRONG_MATCH_THRESHOLD,
    loose_match_threshold: float = DEFAULT_LOOSE_MATCH_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    """Match in-memory debt instrument mentions into debt instruments."""
    if debt_instrument_mentions.empty:
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            ),
            "debt_instrument": pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
        }
    if strong_match_threshold < loose_match_threshold:
        raise ValueError("strong_match_threshold must be >= loose_match_threshold")

    rows = sorted(
        debt_instrument_mentions.to_dict("records"),
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("accession_number") or ""),
            str(row.get("item_id") or ""),
            str(row.get("debt_instrument_mention_id") or ""),
        ),
    )
    mention_index = {
        str(row["debt_instrument_mention_id"]): prepare_mention(row) for row in rows
    }
    direct_members: dict[str, list[str]] = {}
    mention_results: list[dict[str, object]] = []

    for mention_id in sorted(
        mention_index, key=lambda key: mention_sort_key(mention_index[key])
    ):
        mention = mention_index[mention_id]
        if mention.cik is None:
            continue
        new_instrument_id = mention.debt_instrument_mention_id
        direct_members.setdefault(
            new_instrument_id, [mention.debt_instrument_mention_id]
        )
        candidates = []
        for debt_instrument_id, member_ids in direct_members.items():
            if debt_instrument_id == new_instrument_id:
                continue
            candidate_members = [mention_index[member_id] for member_id in member_ids]
            if not candidate_members or candidate_members[0].cik != mention.cik:
                continue
            candidate = score_candidate(mention, candidate_members, mention_index)
            if candidate is not None:
                candidates.append(
                    candidate | {"debt_instrument_id": debt_instrument_id}
                )
        candidates.sort(
            key=lambda candidate: (
                1 if bool(candidate["auto_match"]) else 0,
                float(candidate["lender_similarity"]),
                str(candidate["debt_instrument_id"]),
            ),
            reverse=True,
        )
        strong_candidates = [
            candidate
            for candidate in candidates
            if bool(candidate["auto_match"])
            or float(candidate["lender_similarity"]) >= strong_match_threshold
        ]
        loose_candidates = [
            candidate
            for candidate in candidates
            if loose_match_threshold
            <= float(candidate["lender_similarity"])
            < strong_match_threshold
        ]
        if len(strong_candidates) == 1:
            chosen_instrument_id = str(strong_candidates[0]["debt_instrument_id"])
            direct_members[chosen_instrument_id].append(
                mention.debt_instrument_mention_id
            )
            del direct_members[new_instrument_id]
            mention_results.append(
                {
                    "debt_instrument_mention_id": mention.debt_instrument_mention_id,
                    "debt_instrument_id": chosen_instrument_id,
                    "matcher_status": "matched",
                }
            )
            continue
        potential_matches = [
            {
                "amount_match": True,
                "debt_instrument_id": candidate["debt_instrument_id"],
                "lender_similarity": candidate["lender_similarity"],
                "match_via": candidate["match_via"],
                "start_date_match": True,
            }
            for candidate in (
                strong_candidates if len(strong_candidates) > 1 else loose_candidates
            )
        ]
        mention_results.append(
            {
                "debt_instrument_mention_id": mention.debt_instrument_mention_id,
                "debt_instrument_id": new_instrument_id,
                "matcher_status": "ambiguous" if potential_matches else "singleton",
            }
        )

    mention_to_instrument = {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in mention_results
    }
    normalized_members = normalize_instrument_members(direct_members, mention_index)
    parent_links = derive_parent_links(
        normalized_members, mention_index, mention_to_instrument
    )
    debt_instrument_rows = build_debt_instrument_rows(
        normalized_members,
        mention_index,
        parent_links,
        loose_match_threshold=loose_match_threshold,
    )
    return {
        "debt_instrument_mentions": pd.DataFrame(
            mention_results, columns=DEBT_INSTRUMENT_MENTION_COLUMNS
        ),
        "debt_instrument": pd.DataFrame(
            debt_instrument_rows, columns=DEBT_INSTRUMENT_COLUMNS
        ),
    }


def normalize_instrument_members(
    direct_members: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
) -> dict[str, list[str]]:
    """Rekey direct-member groups to the earliest mention ID in each group."""
    normalized: dict[str, list[str]] = {}
    for member_ids in direct_members.values():
        ordered_member_ids = sorted(
            member_ids, key=lambda item: mention_sort_key(mention_index[item])
        )
        normalized[ordered_member_ids[0]] = ordered_member_ids
    return normalized


def derive_parent_links(
    normalized_members: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
    mention_to_instrument: dict[str, str],
) -> dict[str, dict[str, str | None]]:
    """Map mention-level lineage onto debt instrument parent links."""
    parent_links: dict[str, dict[str, str | None]] = {}
    for debt_instrument_id, member_ids in normalized_members.items():
        amendment_parents = {
            mention_to_instrument[parent_id]
            for member_id in member_ids
            for parent_id in [mention_index[member_id].amendment_of]
            if parent_id in mention_to_instrument
            and mention_to_instrument[parent_id] != debt_instrument_id
        }
        split_parents = {
            mention_to_instrument[parent_id]
            for member_id in member_ids
            for parent_id in [mention_index[member_id].split_of]
            if parent_id in mention_to_instrument
            and mention_to_instrument[parent_id] != debt_instrument_id
        }
        if len(amendment_parents) > 1 or len(split_parents) > 1:
            parent_links[debt_instrument_id] = {
                "amendment_of_debt_instrument_id": None,
                "split_of_debt_instrument_id": None,
            }
            continue
        if amendment_parents and split_parents:
            parent_links[debt_instrument_id] = {
                "amendment_of_debt_instrument_id": None,
                "split_of_debt_instrument_id": None,
            }
            continue
        parent_links[debt_instrument_id] = {
            "amendment_of_debt_instrument_id": next(iter(amendment_parents), None),
            "split_of_debt_instrument_id": next(iter(split_parents), None),
        }
    return parent_links


def build_debt_instrument_rows(
    normalized_members: dict[str, list[str]],
    mention_index: dict[str, PreparedMention],
    parent_links: dict[str, dict[str, str | None]],
    *,
    loose_match_threshold: float,
) -> list[dict[str, object]]:
    """Build persisted debt instrument rows from direct groups and parent links."""
    rows: list[dict[str, object]] = []
    for debt_instrument_id, direct_member_ids in sorted(normalized_members.items()):
        cumulative_mentions = cumulative_member_ids(
            debt_instrument_id,
            normalized_members,
            parent_links,
        )
        ordered_cumulative_mentions = sorted(
            cumulative_mentions,
            key=lambda mention_id: mention_recency_key(mention_index[mention_id]),
            reverse=True,
        )
        direct_mentions_json = json.dumps(direct_member_ids, sort_keys=True)
        lenders_json = json.dumps(
            dedupe_party_clusters(
                [
                    mention_index[mention_id].lenders_json
                    for mention_id in ordered_cumulative_mentions
                ]
            ),
            sort_keys=True,
        )
        other_interested_parties_json = json.dumps(
            dedupe_party_clusters(
                [
                    mention_index[mention_id].other_interested_parties_json
                    for mention_id in ordered_cumulative_mentions
                ]
            ),
            sort_keys=True,
        )
        possibly_related_json = json.dumps(
            find_possibly_related_mentions(
                debt_instrument_id,
                cumulative_mentions,
                mention_index,
                loose_match_threshold=loose_match_threshold,
            ),
            sort_keys=True,
        )
        rows.append(
            {
                "debt_instrument_id": debt_instrument_id,
                "cik": mention_index[direct_member_ids[0]].cik,
                "seed_debt_instrument_mention_id": direct_member_ids[0],
                "amendment_of_debt_instrument_id": parent_links[debt_instrument_id][
                    "amendment_of_debt_instrument_id"
                ],
                "split_of_debt_instrument_id": parent_links[debt_instrument_id][
                    "split_of_debt_instrument_id"
                ],
                "name": first_non_null(
                    ordered_cumulative_mentions, mention_index, "name"
                ),
                "start_date": first_non_null(
                    ordered_cumulative_mentions, mention_index, "start_date"
                ),
                "end_date": first_non_null(
                    ordered_cumulative_mentions, mention_index, "end_date"
                ),
                "amount": first_non_null(
                    ordered_cumulative_mentions, mention_index, "amount"
                ),
                "direct_mentions_json": direct_mentions_json,
                "lenders_json": lenders_json,
                "other_interested_parties_json": other_interested_parties_json,
                "possibly_related_json": possibly_related_json,
            }
        )
    return rows


def cumulative_member_ids(
    debt_instrument_id: str,
    normalized_members: dict[str, list[str]],
    parent_links: dict[str, dict[str, str | None]],
) -> list[str]:
    """Return cumulative direct member mentions across the ancestor chain."""
    cumulative: list[str] = []
    seen_states: set[str] = set()
    current_id: str | None = debt_instrument_id
    while current_id and current_id not in seen_states:
        seen_states.add(current_id)
        cumulative.extend(normalized_members.get(current_id, []))
        parent = parent_links.get(current_id, {})
        current_id = parent.get("amendment_of_debt_instrument_id") or parent.get(
            "split_of_debt_instrument_id"
        )
    return sorted(set(cumulative))


def find_possibly_related_mentions(
    debt_instrument_id: str,
    cumulative_mentions: list[str],
    mention_index: dict[str, PreparedMention],
    *,
    loose_match_threshold: float,
) -> list[str]:
    """Return advisory related mention IDs for one debt instrument."""
    del debt_instrument_id
    owned = set(cumulative_mentions)
    owned_mentions = [mention_index[mention_id] for mention_id in cumulative_mentions]
    owned_cik = owned_mentions[0].cik if owned_mentions else None
    related_ids: list[str] = []
    for candidate in mention_index.values():
        if candidate.debt_instrument_mention_id in owned:
            continue
        if candidate.cik != owned_cik:
            continue
        if not candidate.lender_signature:
            continue
        if any(
            lender_similarity_score(
                candidate.lender_signature, owned_mention.lender_signature
            )
            >= loose_match_threshold
            for owned_mention in owned_mentions
            if owned_mention.lender_signature
        ):
            related_ids.append(candidate.debt_instrument_mention_id)
    return sorted(set(related_ids))


def first_non_null(
    ordered_mention_ids: list[str],
    mention_index: dict[str, PreparedMention],
    field_name: str,
) -> str | None:
    """Return the newest non-null field value across cumulative mentions."""
    for mention_id in ordered_mention_ids:
        value = getattr(mention_index[mention_id], field_name)
        if value is not None:
            return value
    return None


def dedupe_party_clusters(payloads: list[str]) -> list[dict[str, object]]:
    """Return newest-first deduped party cluster payloads."""
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
    """Return the normalized canonical key for one cluster."""
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
    return max(texts, key=len)


def prepare_mention(row: dict[str, object]) -> PreparedMention:
    """Normalize one mention row for matching."""
    return PreparedMention(
        debt_instrument_mention_id=str(row["debt_instrument_mention_id"]),
        item_id=str(row["item_id"]),
        raw_id=str(row["raw_id"]),
        accession_number=coerce_optional_text(row.get("accession_number")),
        cik=coerce_optional_text(row.get("cik")),
        date=coerce_optional_text(row.get("date")),
        name=coerce_optional_text(row.get("name")),
        start_date=coerce_optional_text(row.get("start_date")),
        end_date=coerce_optional_text(row.get("end_date")),
        amount=coerce_optional_text(row.get("amount")),
        amendment_of=coerce_optional_text(row.get("amendment_of")),
        split_of=coerce_optional_text(row.get("split_of")),
        lenders_json=str(row.get("lenders_json") or "[]"),
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
        lender_evidence_state=lender_evidence_state(row.get("lenders_json")),
    )


def mention_sort_key(mention: PreparedMention) -> tuple[str, str, str, str]:
    """Return deterministic processing order for direct matching."""
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


def score_candidate(
    mention: PreparedMention,
    candidate_members: list[PreparedMention],
    mention_index: dict[str, PreparedMention],
) -> dict[str, object] | None:
    """Return the best candidate score when normalized terms match exactly."""
    best: dict[str, object] | None = None
    for mention_surface in build_surfaces(mention, mention_index):
        if (
            not mention_surface.normalized_amount
            or not mention_surface.normalized_start_date
        ):
            continue
        for candidate_member in candidate_members:
            for candidate_surface in build_surfaces(candidate_member, mention_index):
                if (
                    mention_surface.normalized_amount
                    != candidate_surface.normalized_amount
                ):
                    continue
                if (
                    mention_surface.normalized_start_date
                    != candidate_surface.normalized_start_date
                ):
                    continue
                match_via = (
                    f"{mention_surface.match_via}->{candidate_surface.match_via}"
                )
                lender_similarity = lender_similarity_score(
                    mention_surface.lender_signature,
                    candidate_surface.lender_signature,
                )
                lender_usable = (
                    mention.lender_evidence_state == "usable"
                    and candidate_member.lender_evidence_state == "usable"
                )
                if lender_usable:
                    candidate = {
                        "auto_match": False,
                        "lender_similarity": lender_similarity,
                        "match_via": match_via,
                    }
                    if best is None or lender_similarity > float(
                        best["lender_similarity"]
                    ):
                        best = candidate
                    continue
                if (
                    mention_surface.normalized_name_fingerprint
                    and mention_surface.normalized_name_fingerprint
                    == candidate_surface.normalized_name_fingerprint
                    and end_dates_are_compatible(
                        mention_surface.normalized_end_date,
                        candidate_surface.normalized_end_date,
                    )
                ):
                    candidate = {
                        "auto_match": True,
                        "lender_similarity": lender_similarity,
                        "match_via": f"{match_via}:name_fingerprint_fallback",
                    }
                    if best is None or not bool(best["auto_match"]):
                        best = candidate
    return best


def build_surfaces(
    mention: PreparedMention,
    mention_index: dict[str, PreparedMention],
) -> list[MentionSurface]:
    """Return self-only matching surfaces for one mention state."""
    del mention_index
    return [surface_for(mention, match_via="self")]


def surface_for(mention: PreparedMention, *, match_via: str) -> MentionSurface:
    """Build one normalized matching surface."""
    return MentionSurface(
        source_mention_id=mention.debt_instrument_mention_id,
        match_via=match_via,
        normalized_amount=mention.normalized_amount,
        normalized_start_date=mention.normalized_start_date,
        normalized_end_date=mention.normalized_end_date,
        normalized_name_fingerprint=mention.normalized_name_fingerprint,
        lender_signature=mention.lender_signature,
    )


def coerce_optional_text(value: object) -> str | None:
    """Return one trimmed string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_amount(value: str | None) -> str | None:
    """Normalize amount strings for exact matcher comparisons."""
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
    """Normalize date strings for exact matcher comparisons."""
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
    """Normalize debt-instrument names for exact fallback matching."""
    if value is None:
        return None
    text = value.lower()
    text = re.sub(r"(\d+)\.(\d*?[1-9])0+(?=%)", r"\1.\2", text)
    text = re.sub(r"(\d+)\.0+(?=%)", r"\1", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def lender_evidence_state(value: object) -> str:
    """Return whether lender evidence is usable, generic-only, or missing."""
    clusters = parse_cluster_list(str(value or "[]"))
    if not clusters:
        return "missing"
    usable_keys = [key for key in lender_keys(value) if key not in GENERIC_LENDER_TERMS]
    if usable_keys:
        return "usable"
    return "generic_only"


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
    if left and right:
        return left == right
    return True
