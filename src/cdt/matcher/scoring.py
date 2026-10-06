"""Decide mention identity: cluster profiles, candidate scores, resolution and edge rows."""

from __future__ import annotations

import pandas as pd

from cdt.matcher.compat import (
    NAME_CLASS_GATE,
    end_dates_are_compatible,
    name_fingerprint_is_identifying,
    name_fingerprints_are_compatible,
    name_rates_are_compatible,
)
from cdt.matcher.normalize import (
    _json_text,
    coerce_optional_text,
    lender_signature,
    lender_similarity_score,
    mention_sort_key,
    normalize_amount,
    normalize_date,
    normalize_name_fingerprint,
)
from cdt.matcher.schema import (
    DEFAULT_LENDER_SUPPORT_THRESHOLD,
    CandidateScore,
    ClusterProfile,
    PreparedMention,
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
            coerce_optional_text(instrument_row.get("principal_amount"))
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
        )
        if normalized_end_date:
            profile.normalized_end_dates.add(normalized_end_date)
        normalized_name = normalize_name_fingerprint(
            coerce_optional_text(instrument_row.get("name"))
        )
        if normalized_name:
            profile.normalized_name_fingerprints.add(normalized_name)
        lenders = lender_signature(_json_text(instrument_row, "parties_json") or "[]")
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
    """Return whether one key agrees with the cluster and none conflicts.

    Amount, start date or end date may agree; amount and start date may
    conflict. Used only when the names are already compatible.
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
    """Return scored candidate clusters for one mention, best first.

    Sorted by descending score, then id. Empty when the mention has no CIK, or
    has neither both match keys nor an identifying name and its name class
    exceeds ``NAME_CLASS_GATE``. ``lender_signature`` overrides the mention's
    own for scoring only (see ``borrowed_lender_signature``).
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
            # One item returns one object per instrument, so a same-item pair
            # is two instruments whatever their names and keys say.
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
                # claim a mention whose name individuates a series.
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
    """Return the mention's chosen cluster id and its edge rows.

    One member edge, plus related or ambiguous-candidate edges. A tie that is
    not name-only seeds a new cluster keyed on the mention's own id, as does a
    mention with no candidate at the strong threshold.
    """
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
            # A name-only tie joins one of the tied clusters rather than seeding
            # another: the exact name first, then a live cluster over a retired
            # one, then the largest.
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
