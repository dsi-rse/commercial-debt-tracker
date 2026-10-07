"""Matcher constants, published column lists, dataset roots and working records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cdt.datasets import dataset_root

DEFAULT_RELATED_THRESHOLD = 0.75
DEFAULT_MEMBERSHIP_THRESHOLD = 0.90
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_LENDER_SUPPORT_THRESHOLD = 0.5
# Bump when published columns or hashed mention payloads change; a root matched
# at an older version is fully rematched (`_stale_schema_forces_rematch`).
MATCHER_SCHEMA_VERSION = 7
EDGE_TYPES = ("member", "related", "ambiguous_candidate")


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
    # an undated balance rather than a stated as-of date.
    "outstanding_balance_as_of_is_filing_date",
    "outstanding_balance_source_mention_id",
    "interest_rate_kind",
    "interest_rate_pct",
    "interest_rate_source_mention_id",
    "parties_json",
    "lender_disclosure",
    "amendment_inferred_by",
    # True when every member mention was synthesized by the extractor: a prior
    # state cited from its successor's filing that no filing describes on its
    # own, which a reader summing capacity or counting live obligations needs
    # to know.
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
    # Set only on a row the extractor synthesized: the rule that minted it and
    # the model-emitted mention it was minted from. Never a match key itself.
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
        # A synthesized prior state carries its successor's name, which must not
        # widen the cluster's name class for later mentions.
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


# Extracted terminal events. A cluster with a member carrying one is retired,
# which breaks a name-only tie toward the live obligation; the matcher derives
# no lifecycle status of its own.
TERMINAL_STATUS_EVENTS = {"terminated", "repaid", "exchanged", "defaulted"}
