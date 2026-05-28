"""Matcher stage for consolidating instrument mentions into debt instruments."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from cdt.database import (
    cdt_db_path,
    clear_matcher_assignments,
    connect_cdt_db,
    read_matcher_mentions,
    replace_debt_instruments,
    update_matcher_results,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_LOOSE_MATCH_THRESHOLD = 0.75
DEFAULT_STRONG_MATCH_THRESHOLD = 0.90
MATCHER_STATUSES = ("singleton", "matched", "ambiguous")
MATCHER_COLUMNS = [
    "instrument_mention_id",
    "debt_instrument_id",
    "matcher_status",
    "potential_matches_json",
]
DEBT_INSTRUMENT_COLUMNS = [
    "debt_instrument_id",
    "cik",
    "created_from_mention_id",
]


@dataclass(frozen=True)
class MentionSurface:
    """Normalized matching surface for one mention or one-hop lineage neighbor."""

    source_mention_id: str
    match_via: str
    normalized_amount: str | None
    normalized_start_date: str | None
    lender_signature: str


def match_pending_mentions(
    *,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    strong_match_threshold: float = DEFAULT_STRONG_MATCH_THRESHOLD,
    loose_match_threshold: float = DEFAULT_LOOSE_MATCH_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    """Match pending instrument mentions into debt instruments in SQLite."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    mention_rows = _load_matcher_rows(
        data_dir=data_dir, batch_size=batch_size, force=force
    )
    if not mention_rows and not force:
        return {
            "instrument_mentions": pd.DataFrame(columns=MATCHER_COLUMNS),
            "debt_instruments": pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
        }
    tables = match_tables(
        pd.DataFrame(mention_rows),
        strong_match_threshold=strong_match_threshold,
        loose_match_threshold=loose_match_threshold,
    )
    conn = connect_cdt_db(cdt_db_path(data_dir))
    try:
        clear_matcher_assignments(conn)
        if not tables["instrument_mentions"].empty:
            update_matcher_results(
                conn,
                tables["instrument_mentions"].to_dict("records"),
            )
        replace_debt_instruments(
            conn,
            tables["debt_instruments"].to_dict("records"),
        )
    finally:
        conn.close()
    LOGGER.info(
        "Matcher complete: mentions=%s debt_instruments=%s force=%s",
        len(tables["instrument_mentions"]),
        len(tables["debt_instruments"]),
        force,
    )
    return tables


def _load_matcher_rows(
    *,
    data_dir: Path | None,
    batch_size: int,
    force: bool,
) -> list[dict[str, object]]:
    conn = connect_cdt_db(cdt_db_path(data_dir))
    try:
        if force:
            return read_matcher_mentions(conn, pending_only=False)
        pending_rows = read_matcher_mentions(conn, pending_only=True, limit=batch_size)
        if not pending_rows:
            return []
        return read_matcher_mentions(conn, pending_only=False)
    finally:
        conn.close()


def match_tables(
    instrument_mentions: pd.DataFrame,
    *,
    strong_match_threshold: float = DEFAULT_STRONG_MATCH_THRESHOLD,
    loose_match_threshold: float = DEFAULT_LOOSE_MATCH_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    """Match in-memory instrument mentions into debt instruments."""
    if instrument_mentions.empty:
        return {
            "instrument_mentions": pd.DataFrame(columns=MATCHER_COLUMNS),
            "debt_instruments": pd.DataFrame(columns=DEBT_INSTRUMENT_COLUMNS),
        }
    if strong_match_threshold < loose_match_threshold:
        raise ValueError("strong_match_threshold must be >= loose_match_threshold")

    rows = sorted(
        instrument_mentions.to_dict("records"),
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("accession_number") or ""),
            str(row.get("item_id") or ""),
            str(row.get("raw_id") or ""),
        ),
    )
    mention_index = {
        str(row["instrument_mention_id"]): prepare_mention(row) for row in rows
    }
    instrument_members: dict[str, list[str]] = {}
    mention_results: list[dict[str, object]] = []

    for mention_id in sorted(
        mention_index, key=lambda key: mention_sort_key(mention_index[key])
    ):
        mention = mention_index[mention_id]
        if mention.cik is None:
            continue
        new_instrument_id = debt_instrument_id_for(mention.instrument_mention_id)
        instrument_members.setdefault(
            new_instrument_id, [mention.instrument_mention_id]
        )
        candidates = []
        for debt_instrument_id, member_ids in instrument_members.items():
            if debt_instrument_id == new_instrument_id:
                continue
            candidate_members = [mention_index[member_id] for member_id in member_ids]
            if not candidate_members or candidate_members[0].cik != mention.cik:
                continue
            candidate = score_candidate(
                mention,
                candidate_members,
                mention_index,
            )
            if candidate is not None:
                candidates.append(
                    candidate | {"debt_instrument_id": debt_instrument_id}
                )
        candidates.sort(
            key=lambda candidate: (
                float(candidate["lender_similarity"]),
                str(candidate["debt_instrument_id"]),
            ),
            reverse=True,
        )

        strong_candidates = [
            candidate
            for candidate in candidates
            if float(candidate["lender_similarity"]) >= strong_match_threshold
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
            instrument_members[chosen_instrument_id].append(
                mention.instrument_mention_id
            )
            del instrument_members[new_instrument_id]
            mention_results.append(
                {
                    "instrument_mention_id": mention.instrument_mention_id,
                    "debt_instrument_id": chosen_instrument_id,
                    "matcher_status": "matched",
                    "potential_matches_json": json.dumps([], sort_keys=True),
                }
            )
            continue

        potential_matches = [
            {
                "debt_instrument_id": candidate["debt_instrument_id"],
                "match_via": candidate["match_via"],
                "lender_similarity": candidate["lender_similarity"],
                "amount_match": True,
                "start_date_match": True,
            }
            for candidate in (
                strong_candidates if len(strong_candidates) > 1 else loose_candidates
            )
        ]
        mention_results.append(
            {
                "instrument_mention_id": mention.instrument_mention_id,
                "debt_instrument_id": new_instrument_id,
                "matcher_status": "ambiguous" if potential_matches else "singleton",
                "potential_matches_json": json.dumps(potential_matches, sort_keys=True),
            }
        )

    debt_instruments = [
        {
            "debt_instrument_id": debt_instrument_id,
            "cik": mention_index[member_ids[0]].cik,
            "created_from_mention_id": member_ids[0],
        }
        for debt_instrument_id, member_ids in sorted(instrument_members.items())
        if member_ids
    ]
    return {
        "instrument_mentions": pd.DataFrame(mention_results, columns=MATCHER_COLUMNS),
        "debt_instruments": pd.DataFrame(
            debt_instruments, columns=DEBT_INSTRUMENT_COLUMNS
        ),
    }


@dataclass(frozen=True)
class PreparedMention:
    """Denormalized mention record used during matching."""

    instrument_mention_id: str
    item_id: str
    raw_id: str
    accession_number: str | None
    cik: str | None
    date: str | None
    amendment_of: str | None
    split_of: str | None
    normalized_amount: str | None
    normalized_start_date: str | None
    lender_signature: str


def prepare_mention(row: dict[str, object]) -> PreparedMention:
    """Normalize one mention row for matching."""
    return PreparedMention(
        instrument_mention_id=str(row["instrument_mention_id"]),
        item_id=str(row["item_id"]),
        raw_id=str(row["raw_id"]),
        accession_number=coerce_optional_text(row.get("accession_number")),
        cik=coerce_optional_text(row.get("cik")),
        date=coerce_optional_text(row.get("date")),
        amendment_of=coerce_optional_text(row.get("amendment_of")),
        split_of=coerce_optional_text(row.get("split_of")),
        normalized_amount=normalize_amount(coerce_optional_text(row.get("amount"))),
        normalized_start_date=normalize_date(
            coerce_optional_text(row.get("start_date"))
        ),
        lender_signature=lender_signature(row.get("lenders_json")),
    )


def mention_sort_key(mention: PreparedMention) -> tuple[str, str, str, str]:
    """Return deterministic ordering for mention processing."""
    return (
        mention.date or "",
        mention.accession_number or "",
        mention.item_id,
        mention.raw_id,
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
                lender_similarity = lender_similarity_score(
                    mention_surface.lender_signature,
                    candidate_surface.lender_signature,
                )
                match_via = (
                    f"{mention_surface.match_via}->{candidate_surface.match_via}"
                )
                candidate = {
                    "lender_similarity": lender_similarity,
                    "match_via": match_via,
                }
                if best is None or lender_similarity > float(best["lender_similarity"]):
                    best = candidate
    return best


def build_surfaces(
    mention: PreparedMention,
    mention_index: dict[str, PreparedMention],
) -> list[MentionSurface]:
    """Return own plus one-hop lineage match surfaces for one mention."""
    surfaces = [surface_for(mention, match_via="self")]
    for neighbor_id, relation_type in (
        (mention.amendment_of, "amendment_parent"),
        (mention.split_of, "split_parent"),
    ):
        if neighbor_id and neighbor_id in mention_index:
            surfaces.append(
                surface_for(mention_index[neighbor_id], match_via=relation_type)
            )
    for candidate in mention_index.values():
        if candidate.amendment_of == mention.instrument_mention_id:
            surfaces.append(surface_for(candidate, match_via="amendment_child"))
        if candidate.split_of == mention.instrument_mention_id:
            surfaces.append(surface_for(candidate, match_via="split_child"))
    unique: dict[tuple[str | None, str | None, str, str], MentionSurface] = {}
    for surface in surfaces:
        key = (
            surface.normalized_amount,
            surface.normalized_start_date,
            surface.lender_signature,
            surface.match_via,
        )
        unique[key] = surface
    return list(unique.values())


def surface_for(mention: PreparedMention, *, match_via: str) -> MentionSurface:
    """Build one normalized matching surface."""
    return MentionSurface(
        source_mention_id=mention.instrument_mention_id,
        match_via=match_via,
        normalized_amount=mention.normalized_amount,
        normalized_start_date=mention.normalized_start_date,
        lender_signature=mention.lender_signature,
    )


def debt_instrument_id_for(instrument_mention_id: str) -> str:
    """Return the deterministic debt instrument identifier for one seed mention."""
    return f"di::{instrument_mention_id}"


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


def lender_signature(value: object) -> str:
    """Return one normalized lender signature from extractor JSON payload."""
    try:
        clusters = json.loads(str(value)) if value is not None else []
    except json.JSONDecodeError:
        clusters = []
    parts: list[str] = []
    if not isinstance(clusters, list):
        return ""
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        mentions = cluster.get("mentions", [])
        if not isinstance(mentions, list):
            continue
        texts = sorted(
            normalize_lender_text(str(mention.get("text", "")))
            for mention in mentions
            if isinstance(mention, dict) and mention.get("text")
        )
        if texts:
            parts.append(max(texts, key=len))
    return " | ".join(sorted({part for part in parts if part}))


def normalize_lender_text(value: str) -> str:
    """Normalize lender strings before similarity comparison."""
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
