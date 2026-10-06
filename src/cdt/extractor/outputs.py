"""Select pending partitions and write extraction outputs; shared by the live and batch backends."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from cdt.classifier.core import CLASSIFICATION_DATASET_NAME, CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import (
    PARTITION_PATTERN,
    SIXK_SNIPPET_DATASET_NAME,
    CompletedPartition,
    CompletionRegistry,
    completion_registry_path,
    dataset_root,
    date_shard_partition_path,
    extractor_run_path,
    iter_date_shard_partitions,
    load_completion_registry,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
    save_completion_registry,
)
from cdt.extractor.prior_state import mint_prior_state_rows, published_mention_rows
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.state import PUBLISHABLE_ROW_STATES, ExtractionRowState
from cdt.extractor.workflow import _failure_record, _merge_row_failures
from cdt.shared import get_logger
from cdt.storage import (
    artifact_exists,
    coerce_dataset_text,
    list_artifacts_with_versions,
    read_table,
    write_json_artifact,
    write_partition_table,
    write_text_artifact,
)

LOGGER = get_logger(__name__)

#: Datasets the extractor takes work from, in claim order. Both hold rows in
#: CLASSIFIED_ITEM_COLUMNS with a `relevance` flag — 8-K items scored by the
#: item classifier, 6-K windows scored by the two-stage triage — so every stage
#: below reads them identically and none of them knows which genre it has.
CLASSIFICATION_SOURCES: tuple[str, ...] = (
    CLASSIFICATION_DATASET_NAME,
    SIXK_SNIPPET_DATASET_NAME,
)


MENTIONS_DATASET_NAME = "mentions"


def mentions_root(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the canonical mentions dataset root."""
    return dataset_root(
        MENTIONS_DATASET_NAME, artifact_root=artifact_root, data_dir=data_dir
    )


def extracted_tables_path(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
) -> str:
    """Return the root that stores extractor audit artifacts."""
    return dataset_root(
        "extractor-runs", artifact_root=artifact_root, data_dir=data_dir
    )


@dataclass
class PendingExtractPartition:
    """One source partition with extraction work outstanding.

    ``classification_path`` points into any of CLASSIFICATION_SOURCES; the rows
    it holds are classification rows either way.
    """

    classification_path: str
    date: str
    shard: str
    fingerprint: str | None
    done_item_ids: frozenset[str]


def pending_extract_partitions(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    exclude_paths: set[str] | None = None,
) -> tuple[list[PendingExtractPartition], dict[str, CompletedPartition]]:
    """Select partitions with unextracted rows, keyed on outcomes and versions.

    A partition is pending when it has no completion entry, its entry is marked
    incomplete (an aborted pass), or its source fingerprint changed (ingest
    merged late-arriving rows into it). ``done_item_ids`` are rows that
    already reached a terminal state and must not be re-paid; ``force``
    makes every partition pending with none done.

    Also returns the loaded registry (empty under ``force``) so the caller
    can update and persist it.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    registry = (
        # Not a plain ``{}``: save_completion_registry treats every key of one
        # as changed, which re-sends the whole run at every batch boundary.
        CompletionRegistry()
        if force
        else load_completion_registry(
            "extract", artifact_root=resolved_root, data_dir=data_dir
        )
    )
    fingerprints: dict[str, str | None] = {}
    for source in CLASSIFICATION_SOURCES:
        fingerprints.update(
            {
                path: version
                for path, version in list_artifacts_with_versions(
                    dataset_root(
                        source,
                        artifact_root=resolved_root,
                        data_dir=data_dir,
                    ),
                    suffix=".parquet",
                ).items()
                if PARTITION_PATTERN.search(path)
            }
        )
    pending: list[PendingExtractPartition] = []
    for classification_path in sorted(fingerprints):
        if exclude_paths and classification_path in exclude_paths:
            continue
        partition = parse_date_shard_partition(classification_path)
        fingerprint = fingerprints[classification_path]
        entry = registry.get(classification_path)
        if force or entry is None:
            pending.append(
                PendingExtractPartition(
                    classification_path=classification_path,
                    date=partition["date"],
                    shard=partition["shard"],
                    fingerprint=fingerprint,
                    done_item_ids=frozenset(),
                )
            )
            continue
        if entry.complete and entry.fingerprint == fingerprint:
            continue
        pending.append(
            PendingExtractPartition(
                classification_path=classification_path,
                date=partition["date"],
                shard=partition["shard"],
                fingerprint=fingerprint,
                done_item_ids=entry.item_ids,
            )
        )
    return pending, registry


def backfill_mentions(
    artifact_root: str | Path | None = None,
    *,
    data_dir: Path | None = None,
    dry_run: bool = False,
    renew: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Re-derive the synthesized rows over every existing `mentions` partition.

    Drops the rows an earlier run synthesized, clears the pointers that named
    them, and mints again from the model-emitted rows, with no model call;
    running it twice is a no-op. Returns the mint counters plus `partitions`
    and `partitions_rewritten`; `dry_run` counts without rewriting.

    ``renew`` is called before each partition rewrite to extend the caller's
    writer lease, since rewriting the whole dataset can outlast the lease TTL.
    A dry run writes nothing and never calls it.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    counts: dict[str, int] = {"partitions": 0, "partitions_rewritten": 0}
    for path in iter_date_shard_partitions(
        MENTIONS_DATASET_NAME, artifact_root=resolved_root, data_dir=data_dir
    ):
        table = read_table(path)
        if table.empty:
            continue
        counts["partitions"] += 1
        records = table.to_dict("records")
        synthesized_ids = {
            coerce_dataset_text(record.get("debt_instrument_mention_id"))
            for record in records
            if coerce_dataset_text(record.get("synthesized_by")) is not None
        }
        real: dict[str, list[dict[str, object]]] = {}
        for record in records:
            if coerce_dataset_text(record.get("synthesized_by")) is not None:
                continue
            if coerce_dataset_text(record.get("amendment_of")) in synthesized_ids:
                record["amendment_of"] = None
            real.setdefault(
                coerce_dataset_text(record.get("item_id")) or "", []
            ).append(record)
        published: list[dict[str, object]] = []
        for item_rows in real.values():
            published.extend(mint_prior_state_rows(item_rows, counts))
        if dry_run:
            continue
        if renew is not None:
            renew()
        partition = parse_date_shard_partition(path)
        write_partition_table(
            mentions_root(resolved_root, data_dir=data_dir),
            partition={"date": partition["date"], "shard": partition["shard"]},
            table=pd.DataFrame(published, columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
        )
        counts["partitions_rewritten"] += 1
    LOGGER.info("Mentions backfill%s: %s", " (dry run)" if dry_run else "", counts)
    return counts


def _mentions_partition_needs_write(
    resolved_root: str,
    *,
    data_dir: Path | None,
    partition: dict[str, str],
    new_mentions: pd.DataFrame,
    replaced_item_ids: set[str],
    retired_item_ids: set[str],
) -> bool:
    """Whether this mentions partition has rows to add or rows to take away.

    True when there are new mentions, or when there are replaced or retired
    item ids and the partition already exists (a re-extracted item that now
    yields no mentions must withdraw its old rows). Never true for a partition
    that does not exist and would only be written empty. Shared by both
    backends; see docs/decisions/extraction.md.
    """
    if not new_mentions.empty:
        return True
    if not (replaced_item_ids or retired_item_ids):
        return False
    return artifact_exists(
        date_shard_partition_path(
            MENTIONS_DATASET_NAME,
            partition_date=partition["date"],
            shard=partition["shard"],
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
    )


def _merge_mentions_partition(
    resolved_root: str,
    *,
    data_dir: Path | None,
    partition: dict[str, str],
    new_mentions: pd.DataFrame,
    replaced_item_ids: set[str],
    retired_item_ids: set[str] | None = None,
) -> str:
    """Merge newly extracted mentions into a partition, replacing per item.

    Row-level re-processing means a target partition can already hold mentions
    from earlier passes; overwriting it wholesale would drop them.

    ``retired_item_ids`` are ids a claimed source partition used to hold and
    no longer does -- a row that stopped being relevant, or, on the 6-K path,
    windows that merged into one snippet so their own ids ceased to exist.
    Rows of those items are dropped; rows of items named in neither set are
    left alone, since one mentions partition holds both genres and several
    accessions. Nothing else prunes retired items' mentions. Returns the
    partition path written.
    """
    target_path = date_shard_partition_path(
        MENTIONS_DATASET_NAME,
        partition_date=partition["date"],
        shard=partition["shard"],
        artifact_root=resolved_root,
        data_dir=data_dir,
    )
    table = new_mentions
    if artifact_exists(target_path):
        existing = read_table(target_path, DEBT_INSTRUMENT_MENTION_COLUMNS)
        dropped = replaced_item_ids | (retired_item_ids or set())
        kept = existing.loc[~existing["item_id"].astype(str).isin(dropped)]
        table = pd.concat([kept, new_mentions], ignore_index=True)
    write_partition_table(
        mentions_root(resolved_root, data_dir=data_dir),
        partition=partition,
        table=table.reindex(columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
    )
    return target_path


def finalize_extract_outputs(
    row_entries: list[tuple[ExtractionRowState, str, str]],
    *,
    claimed: dict[str, dict[str, object]],
    run_id: str,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Write mention partitions, audit log, and manifests for a completed job.

    This is the batch backend's analogue of the tail of ``extract_pending_items``.
    Every row in ``row_entries`` must already be terminal. ``claimed`` carries each
    claimed classification partition's fingerprint and prior terminal item_ids
    (from ``collect_pending_extract_items``), so completion is recorded per row
    outcome rather than per visit. Mentions are regrouped by their originating
    ``(date, shard)`` partition and merged into existing targets per item.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    mentions_by_partition: dict[tuple[str, str], list[dict[str, object]]] = {}
    audit_records: list[str] = []
    failed_rows: dict[str, dict[str, object]] = {}
    succeeded_item_ids: set[str] = set()
    for row_state, partition_date, shard in row_entries:
        audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
        if row_state.state in PUBLISHABLE_ROW_STATES:
            mentions_by_partition.setdefault((partition_date, shard), []).extend(
                published_mention_rows(row_state)
            )
        if row_state.state == "SUCCESS":
            succeeded_item_ids.add(row_state.item_id)
        else:
            failed_rows[row_state.item_id] = _failure_record(
                row_state,
                partition_date=partition_date,
                shard=shard,
                run_id=run_id,
                backend="batch",
            )
        # Ensure a visited-but-empty partition still exists as a key so we do not
        # lose track of which partitions the job covered.
        mentions_by_partition.setdefault((partition_date, shard), [])

    terminal_by_partition: dict[tuple[str, str], set[str]] = {}
    for row_state, partition_date, shard in row_entries:
        if row_state.state is not None:
            terminal_by_partition.setdefault((partition_date, shard), set()).add(
                row_state.item_id
            )

    # Read each claimed source once, before the merge loop needs it and before
    # the registry loop below records it. A mentions partition is keyed by
    # (date, shard) while a claim is keyed by path, and both genres can claim
    # the same (date, shard) — so retired ids are accumulated per partition
    # across every claim that lands there, never inferred from one source.
    relevant_by_path: dict[str, set[str]] = {}
    retired_by_partition: dict[tuple[str, str], set[str]] = {}
    for classification_path, claim in claimed.items():
        claim_partition = parse_date_shard_partition(classification_path)
        claim_relevant = read_table(
            classification_path, CLASSIFIED_ITEM_COLUMNS
        ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
        claim_relevant = claim_relevant.loc[claim_relevant["relevance"].fillna(False)]
        relevant_ids = {str(value) for value in claim_relevant["item_id"].astype(str)}
        relevant_by_path[classification_path] = relevant_ids
        prior_ids = {
            str(item) for item in cast(list[object], claim.get("prior_item_ids") or [])
        }
        retired_by_partition.setdefault(
            (claim_partition["date"], claim_partition["shard"]), set()
        ).update(prior_ids - relevant_ids)
    # A partition whose rows were all done already contributes no mention rows,
    # but may still have ids to prune.
    for partition_key, retired_ids in retired_by_partition.items():
        if retired_ids:
            mentions_by_partition.setdefault(partition_key, [])

    processed_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    empty_partitions = 0
    for (partition_date, shard), mention_rows in sorted(mentions_by_partition.items()):
        mentions = pd.DataFrame(mention_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
        replaced = terminal_by_partition.get((partition_date, shard), set())
        retired = retired_by_partition.get((partition_date, shard), set())
        # `replaced` as well as `retired`: a still-relevant item re-extracted to
        # zero mentions must still have its old rows purged.
        if not _mentions_partition_needs_write(
            resolved_root,
            data_dir=data_dir,
            partition={"date": partition_date, "shard": shard},
            new_mentions=mentions,
            replaced_item_ids=replaced,
            retired_item_ids=retired,
        ):
            empty_partitions += 1
            continue
        partitions_written.append(
            _merge_mentions_partition(
                resolved_root,
                data_dir=data_dir,
                partition={"date": partition_date, "shard": shard},
                new_mentions=mentions,
                replaced_item_ids=replaced,
                retired_item_ids=retired,
            )
        )
        if mentions.empty:
            empty_partitions += 1
        else:
            processed_frames.append(mentions)

    registry = load_completion_registry(
        "extract", artifact_root=resolved_root, data_dir=data_dir
    )
    for classification_path, claim in claimed.items():
        partition = parse_date_shard_partition(classification_path)
        prior = {
            str(item) for item in cast(list[object], claim.get("prior_item_ids") or [])
        }
        terminal = prior | terminal_by_partition.get(
            (partition["date"], partition["shard"]), set()
        )
        relevant_ids = relevant_by_path[classification_path]
        fingerprint = claim.get("fingerprint")
        registry[classification_path] = CompletedPartition(
            fingerprint=str(fingerprint) if fingerprint else None,
            # As in the synchronous path: only ids the source still holds, and
            # only this source's — one (date, shard) can be claimed by both
            # genres, so `terminal_by_partition` mixes them.
            item_ids=frozenset(terminal & relevant_ids),
            complete=relevant_ids <= terminal,
        )
    save_completion_registry(
        "extract", registry, artifact_root=resolved_root, data_dir=data_dir
    )
    # Claiming the partitions above marks these rows done for good, so record the
    # ones that produced nothing before that fact is only visible in the audit log.
    failure_registry, total_known_failures = _merge_row_failures(
        failed_rows,
        succeeded_item_ids,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )

    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    write_text_artifact(
        full_jsonl_path,
        ("\n".join(audit_records) + "\n") if audit_records else "",
    )
    write_json_artifact(
        run_manifest_path(
            "extract", run_id, artifact_root=resolved_root, data_dir=data_dir
        ),
        {
            "artifact_root": resolved_root,
            "stage": "extract",
            "backend": "batch",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_attempts": max_attempts,
            "partitions_completed": sorted(claimed),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
            "failure_count": len(failed_rows),
            "audit_path": full_jsonl_path,
            "completion_registry": completion_registry_path(
                "extract", artifact_root=resolved_root, data_dir=data_dir
            ),
            "failure_registry": failure_registry,
        },
    )
    LOGGER.info(
        "Batch extractor finalize complete: rows=%s successes=%s failures=%s "
        "mentions=%s synthesized=%s audit=%s failure_registry=%s (%s total)",
        len(row_entries),
        len(row_entries) - len(failed_rows),
        len(failed_rows),
        sum(len(frame) for frame in processed_frames),
        sum(
            int(frame["synthesized_by"].notna().sum())
            for frame in processed_frames
            if not frame.empty
        ),
        full_jsonl_path,
        failure_registry,
        total_known_failures,
    )
    if not processed_frames:
        return pd.DataFrame(columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=DEBT_INSTRUMENT_MENTION_COLUMNS
    )
