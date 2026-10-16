"""Select pending work and write extraction outputs: mentions and the completion and failure registries; shared by both backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, cast

import pandas as pd

from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.completion import (
    CompletedPartition,
    CompletionRegistry,
    completion_registry_root,
    load_completion_registry,
    save_completion_registry,
)
from cdt.datasets import (
    GENRES,
    MONTH_PARTITION_PATTERN,
    dataset_root,
    extractor_run_path,
    load_row_failures,
    month_partition_path,
    parse_source_partition,
    resolve_artifact_root,
    run_manifest_path,
    save_row_failures,
    source_partition_month,
)
from cdt.extractor.prior_state import published_mention_rows
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.stages import EXTRACTOR_STAGES
from cdt.extractor.state import (
    PUBLISHABLE_ROW_STATES,
    STATE_ITEM_ROW_FIELDS,
    ExtractionRowState,
    coerce_native,
)
from cdt.partition_stage import sort_partition_rows
from cdt.shared import get_logger
from cdt.storage.objects import (
    artifact_exists,
    join_artifact_path,
    list_artifacts_with_versions,
    write_json_artifact,
    write_text_artifact,
)
from cdt.storage.tables import read_table, write_partition_table

LOGGER = get_logger(__name__)

#: Datasets the extractor takes work from, one per genre, in genre order. Each
#: holds rows in CLASSIFIED_ITEM_COLUMNS with a `relevance` flag — 8-K items
#: scored by the item classifier, 6-K windows scored by the two-stage triage —
#: so every stage below reads them identically and none of them knows which
#: genre it has.
CLASSIFICATION_SOURCES: tuple[str, ...] = tuple(
    genre.classified_dataset for genre in GENRES.values()
)
MENTIONS_DATASET_NAME = "mentions"


def mentions_root(
    artifact_root: str | Path | None = None,
) -> str:
    """Return the canonical mentions dataset root."""
    return dataset_root(MENTIONS_DATASET_NAME, artifact_root=artifact_root)


def extracted_tables_path(
    artifact_root: str | Path | None = None,
) -> str:
    """Return the root that stores extractor audit artifacts."""
    return dataset_root("extractor-runs", artifact_root=artifact_root)


@dataclass
class PendingExtractPartition:
    """One classification month file with extraction work outstanding.

    ``classification_path`` points into any of CLASSIFICATION_SOURCES; the rows
    it holds are classification rows either way. ``month`` is its month, and
    the month of the mentions file its rows' mentions go to.
    """

    classification_path: str
    month: str
    fingerprint: str | None
    done_item_ids: frozenset[str]


def pending_extract_partitions(
    *,
    artifact_root: str | Path | None = None,
    force: bool = False,
    force_paths: frozenset[str] = frozenset(),
    exclude_paths: set[str] | None = None,
) -> tuple[list[PendingExtractPartition], dict[str, CompletedPartition]]:
    """Select partitions with unextracted rows, keyed on outcomes and versions.

    A partition is pending when it has no completion entry, its entry is marked
    incomplete (an aborted pass), or its source fingerprint changed (ingest
    merged late-arriving rows into it). ``done_item_ids`` are rows that
    already reached a terminal state and must not be re-paid; ``force``
    makes every partition pending with none done, and ``force_paths`` does
    that for the partitions it names.

    Also returns the loaded registry (empty under ``force``) so the caller
    can update and persist it.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    registry = (
        # Not a plain ``{}``: save_completion_registry treats every key of one
        # as changed, which re-sends the whole run at every batch boundary.
        CompletionRegistry()
        if force
        else load_completion_registry("extract", artifact_root=resolved_root)
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
                    ),
                    suffix=".parquet",
                ).items()
                if MONTH_PARTITION_PATTERN.search(path)
            }
        )
    pending: list[PendingExtractPartition] = []
    for classification_path in sorted(fingerprints):
        if exclude_paths and classification_path in exclude_paths:
            continue
        month = source_partition_month(parse_source_partition(classification_path))
        fingerprint = fingerprints[classification_path]
        entry = registry.get(classification_path)
        if force or entry is None or classification_path in force_paths:
            pending.append(
                PendingExtractPartition(
                    classification_path=classification_path,
                    month=month,
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
                month=month,
                fingerprint=fingerprint,
                done_item_ids=entry.item_ids,
            )
        )
    return pending, registry


def collect_pending_extract_items(
    *,
    artifact_root: str | Path | None = None,
    force: bool = False,
    force_paths: frozenset[str] = frozenset(),
    max_rows: int | None = None,
) -> tuple[list[tuple[dict[str, str | None], str]], dict[str, dict[str, object]]]:
    """Collect relevant items awaiting extraction across pending partitions.

    Returns ``(entries, claimed)`` where each entry is a native-typed item row
    plus the month of the file it came from, and ``claimed`` maps each claimed
    classification partition to its source fingerprint and the item_ids that
    were already terminal before this job — the state finalize needs to record
    row-outcome-keyed completion and to detect source growth. Uses the same
    selection as ``extract_pending_items`` so both backends claim the same
    work, row by row.

    ``max_rows`` stops claiming partitions once the collected row count reaches
    it (whole partitions stay the atomic claim unit, so the last claimed
    partition may overshoot); None claims everything. Unclaimed partitions stay
    pending for the next job, which bounds the full text one poll tick holds.

    ``force_paths`` names partitions to claim as forced (no row counted done),
    as ``pending_extract_partitions`` does.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    pending, _registry = pending_extract_partitions(
        artifact_root=resolved_root,
        force=force,
        force_paths=force_paths,
    )
    entries: list[tuple[dict[str, str | None], str]] = []
    claimed: dict[str, dict[str, object]] = {}
    deferred_partitions = 0
    for pending_partition in pending:
        if max_rows is not None and len(entries) >= max_rows:
            deferred_partitions += 1
            continue
        claimed[pending_partition.classification_path] = {
            "fingerprint": pending_partition.fingerprint,
            "prior_item_ids": sorted(pending_partition.done_item_ids),
        }
        batch_items = read_table(
            pending_partition.classification_path, CLASSIFIED_ITEM_COLUMNS
        ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
        relevant_items = batch_items.loc[batch_items["relevance"].fillna(False)]
        for item_row in relevant_items.to_dict("records"):
            if str(item_row["item_id"]) in pending_partition.done_item_ids:
                continue
            coerced = {
                key: coerce_native(item_row.get(key)) for key in STATE_ITEM_ROW_FIELDS
            }
            entries.append((coerced, pending_partition.month))
    if deferred_partitions:
        LOGGER.info(
            "Deferred %s pending partition(s) beyond the %s-row job cap; the "
            "next job claims them once this one completes.",
            deferred_partitions,
            max_rows,
        )
    return entries, claimed


def summarize_failure(row_state: ExtractionRowState) -> str:
    """Summarize what this row lost, for its failure-registry entry.

    A salvaged row is terminal-but-publishable: its last attempt often
    succeeded, so the attempt carries no validation errors and the generic
    "unexpected response" summary below would describe a stage that worked.
    The salvage notes are the only record of what was actually dropped, so they
    are what the registry reports.
    """
    if row_state.salvage_notes:
        return "; ".join(row_state.salvage_notes)
    failures = row_state.current_attempt.validation_errors
    if failures:
        return "; ".join(failures)
    if row_state.current_attempt.response:
        return f"Unexpected response at stage {row_state.current_attempt.stage_name}"
    return f"Extractor failed at stage {row_state.current_attempt.stage_name}"


def failed_stage_name(row_state: ExtractionRowState) -> str:
    """Return the stage whose failure this row is registered for.

    For a salvaged row that is the stage salvage fired in, not the last stage
    the row ran — an operator retrying the row needs the former.
    """
    for note in row_state.salvage_notes:
        stage_name, _, _ = note.partition(" ")
        if stage_name in {stage.name for stage in EXTRACTOR_STAGES}:
            return stage_name
    return row_state.current_attempt.stage_name


def failure_record(
    row_state: ExtractionRowState,
    *,
    month: str,
    run_id: str,
    backend: str,
) -> dict[str, object]:
    """Build one failure-registry entry for a terminal non-SUCCESS row."""
    return {
        "item_id": row_state.item_id,
        "accession_number": row_state.item_row.get("accession_number"),
        "cik": row_state.item_row.get("cik"),
        "date": row_state.item_row.get("date"),
        "month": month,
        "state": row_state.state,
        "stage": failed_stage_name(row_state),
        "run_id": run_id,
        "backend": backend,
        "error": summarize_failure(row_state),
    }


def merge_row_failures(
    failures: dict[str, dict[str, object]],
    succeeded_item_ids: set[str],
    *,
    artifact_root: str,
) -> tuple[str, int]:
    """Merge this run's row outcomes into the extract failure registry.

    Failures are added or refreshed; rows that succeeded this run clear any
    earlier entry, so a re-extract that fixes a row does not leave a stale
    failure behind. Returns the registry path and its total entry count.
    """
    registry = load_row_failures("extract", artifact_root=artifact_root)
    for item_id in succeeded_item_ids:
        registry.pop(item_id, None)
    registry.update(failures)
    path = save_row_failures("extract", registry, artifact_root=artifact_root)
    return path, len(registry)


@dataclass
class RowOutcomes:
    """One extract run's per-row records: audit lines, failures and successes."""

    run_id: str
    backend: str
    audit_records: list[str] = field(default_factory=list)
    failed_rows: dict[str, dict[str, object]] = field(default_factory=dict)
    succeeded_item_ids: set[str] = field(default_factory=set)

    def add(
        self: Self, row_state: ExtractionRowState, *, month: str
    ) -> list[dict[str, object]]:
        """Record one terminal row; return the mention rows it publishes."""
        self.audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
        mention_rows = (
            published_mention_rows(row_state)
            if row_state.state in PUBLISHABLE_ROW_STATES
            else []
        )
        if row_state.state == "SUCCESS":
            self.succeeded_item_ids.add(row_state.item_id)
        else:
            self.failed_rows[row_state.item_id] = failure_record(
                row_state,
                month=month,
                run_id=self.run_id,
                backend=self.backend,
            )
        return mention_rows


def write_mentions_partition(
    resolved_root: str,
    *,
    month: str,
    new_mentions: pd.DataFrame,
    replaced_item_ids: set[str],
    retired_item_ids: set[str],
) -> str | None:
    """Merge new mentions into one mentions month file, replacing per item.

    The file can already hold mentions from earlier passes, both genres and
    many accessions, so only rows of ``replaced_item_ids`` (re-extracted
    this run) and ``retired_item_ids`` (gone from a claimed source) are
    dropped. Replaced ids purge too, so an item re-extracted to zero mentions
    withdraws what it published before. Rows are written sorted
    (:func:`cdt.partition_stage.sort_partition_rows`). Returns the path
    written, or None when there is nothing to add or take away; a month file
    that does not exist is never written empty. See docs/decisions/extraction.md.
    """
    target_path = month_partition_path(
        MENTIONS_DATASET_NAME, month=month, artifact_root=resolved_root
    )
    dropped = replaced_item_ids | retired_item_ids
    if new_mentions.empty and not dropped:
        return None
    exists = artifact_exists(target_path)
    if new_mentions.empty and not exists:
        return None
    table = new_mentions
    if exists:
        existing = read_table(target_path, DEBT_INSTRUMENT_MENTION_COLUMNS)
        kept = existing.loc[~existing["item_id"].astype(str).isin(dropped)]
        table = pd.concat([kept, new_mentions], ignore_index=True)
    write_partition_table(
        mentions_root(resolved_root),
        partition={"month": month},
        table=sort_partition_rows(table).reindex(
            columns=DEBT_INSTRUMENT_MENTION_COLUMNS
        ),
    )
    return target_path


def completion_entry(
    fingerprint: str | None, terminal_ids: set[str], relevant_ids: set[str]
) -> CompletedPartition:
    """Return a source partition's registry entry after this run's verdicts.

    Only ids the source still holds are recorded: an id whose row is gone just
    had its mentions pruned, and keeping it would retire it again on every
    later pass. The partition is complete once every relevant row is terminal.
    """
    return CompletedPartition(
        fingerprint=fingerprint,
        item_ids=frozenset(terminal_ids & relevant_ids),
        complete=relevant_ids <= terminal_ids,
    )


def write_run_records(
    outcomes: RowOutcomes,
    registry: CompletionRegistry,
    *,
    artifact_root: str,
    manifest: dict[str, object],
) -> tuple[str, str, int]:
    """Save one extract run's registry, failures, audit log and run manifest.

    ``manifest`` holds the backend's own manifest fields; the shared ones are
    added here. Returns the audit path, the failure-registry path and the
    failure registry's total entry count.
    """
    # Failures first: once the registry marks a row done it is never retried,
    # so a row that produced nothing must already be on record when it does.
    failure_registry, total_known_failures = merge_row_failures(
        outcomes.failed_rows,
        outcomes.succeeded_item_ids,
        artifact_root=artifact_root,
    )
    save_completion_registry("extract", registry, artifact_root=artifact_root)
    audit_path = extractor_run_path(outcomes.run_id, artifact_root=artifact_root)
    write_text_artifact(
        audit_path,
        ("\n".join(outcomes.audit_records) + "\n") if outcomes.audit_records else "",
    )
    write_json_artifact(
        run_manifest_path("extract", outcomes.run_id, artifact_root=artifact_root),
        {
            "artifact_root": artifact_root,
            "stage": "extract",
            **manifest,
            "failure_count": len(outcomes.failed_rows),
            "audit_path": audit_path,
            "completion_registry": completion_registry_root(
                "extract", artifact_root=artifact_root
            ),
            "failure_registry": failure_registry,
        },
    )
    return audit_path, failure_registry, total_known_failures


def checkpoint_audit_path(run_id: str, index: int, *, artifact_root: str) -> str:
    """Return the path of one checkpoint's audit records, beside ``full.jsonl``."""
    return join_artifact_path(
        resolve_artifact_root(artifact_root),
        "extractor-runs",
        f"run_id={run_id}",
        f"checkpoint-{index:04d}.jsonl",
    )


def write_run_checkpoint(
    outcomes: RowOutcomes,
    registry: CompletionRegistry,
    *,
    artifact_root: str,
    index: int,
    audit_offset: int,
) -> int:
    """Persist what a running extract has finished so far; return the new audit offset.

    Writes the audit records from ``audit_offset`` on to checkpoint ``index``'s
    file, merges the failures so far, then saves the registry: a row the
    registry marks done already has its audit record and, if it failed, its
    failure record. The run's final :func:`write_run_records` writes
    ``full.jsonl``, after which its caller deletes the checkpoint files; a run
    that never got that far leaves them as its audit.
    """
    records = outcomes.audit_records[audit_offset:]
    if records:
        write_text_artifact(
            checkpoint_audit_path(outcomes.run_id, index, artifact_root=artifact_root),
            "\n".join(records) + "\n",
        )
    merge_row_failures(
        outcomes.failed_rows,
        outcomes.succeeded_item_ids,
        artifact_root=artifact_root,
    )
    save_completion_registry("extract", registry, artifact_root=artifact_root)
    return len(outcomes.audit_records)


def finalize_extract_outputs(
    row_entries: list[tuple[ExtractionRowState, str]],
    *,
    claimed: dict[str, dict[str, object]],
    run_id: str,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    artifact_root: str | Path | None = None,
) -> pd.DataFrame:
    """Write mention month files, audit log, and manifests for a completed job.

    This is the batch backend's analogue of the tail of ``extract_pending_items``.
    Every row in ``row_entries`` (a row and its month) must already be
    terminal. ``claimed`` carries each claimed classification file's
    fingerprint and prior terminal item_ids (from
    ``collect_pending_extract_items``), so completion is recorded per row
    outcome rather than per visit. Mentions are grouped by month and each
    month file is written once, merged per item. A row whose item no claimed
    classification file of its month still lists as relevant (the month was
    rewritten while the job ran) is retired instead of written.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    outcomes = RowOutcomes(run_id=run_id, backend="batch")
    # Keyed even when a month publishes nothing, so every month the job
    # covered is visited below.
    mentions_by_month: dict[str, list[dict[str, object]]] = {}
    terminal_by_month: dict[str, set[str]] = {}
    for row_state, month in row_entries:
        mentions_by_month.setdefault(month, []).extend(
            outcomes.add(row_state, month=month)
        )
        if row_state.state is not None:
            terminal_by_month.setdefault(month, set()).add(row_state.item_id)

    # Read each claimed source once, before the merge loop needs it and before
    # the registry loop below records it. Both genres' classification files
    # for one month write the same mentions file, so retired ids are
    # accumulated per month across every claim that lands there.
    relevant_by_path: dict[str, set[str]] = {}
    retired_by_month: dict[str, set[str]] = {}
    for classification_path, claim in claimed.items():
        month = source_partition_month(parse_source_partition(classification_path))
        claim_relevant = read_table(
            classification_path, CLASSIFIED_ITEM_COLUMNS
        ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
        claim_relevant = claim_relevant.loc[claim_relevant["relevance"].fillna(False)]
        relevant_ids = {str(value) for value in claim_relevant["item_id"].astype(str)}
        relevant_by_path[classification_path] = relevant_ids
        prior_ids = {
            str(item) for item in cast(list[object], claim.get("prior_item_ids") or [])
        }
        retired_by_month.setdefault(month, set()).update(prior_ids - relevant_ids)
    # A row whose item left its month while the job was in flight (the month
    # was re-segmented or reclassified) is retired, not written: no later
    # claim lists it as done, so nothing would prune its mentions.
    relevant_by_month: dict[str, set[str]] = {}
    for classification_path, relevant_ids in relevant_by_path.items():
        month = source_partition_month(parse_source_partition(classification_path))
        relevant_by_month.setdefault(month, set()).update(relevant_ids)
    for month, relevant_ids in relevant_by_month.items():
        vanished = terminal_by_month.get(month, set()) - relevant_ids
        if vanished:
            retired_by_month.setdefault(month, set()).update(vanished)
            mentions_by_month[month] = [
                row
                for row in mentions_by_month[month]
                if str(row["item_id"]) not in vanished
            ]
    # A month whose rows were all done already contributes no mention rows,
    # but may still have ids to prune.
    for month, retired_ids in retired_by_month.items():
        if retired_ids:
            mentions_by_month.setdefault(month, [])

    processed_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    empty_partitions = 0
    for month, mention_rows in sorted(mentions_by_month.items()):
        mentions = pd.DataFrame(mention_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
        written = write_mentions_partition(
            resolved_root,
            month=month,
            new_mentions=mentions,
            replaced_item_ids=terminal_by_month.get(month, set()),
            retired_item_ids=retired_by_month.get(month, set()),
        )
        if written is None:
            empty_partitions += 1
            continue
        partitions_written.append(written)
        if mentions.empty:
            empty_partitions += 1
        else:
            processed_frames.append(mentions)

    registry = load_completion_registry("extract", artifact_root=resolved_root)
    for classification_path, claim in claimed.items():
        month = source_partition_month(parse_source_partition(classification_path))
        prior = {
            str(item) for item in cast(list[object], claim.get("prior_item_ids") or [])
        }
        terminal = prior | terminal_by_month.get(month, set())
        fingerprint = claim.get("fingerprint")
        # Only this source's relevant ids: one month is claimed by both
        # genres, so `terminal_by_month` mixes them.
        registry[classification_path] = completion_entry(
            str(fingerprint) if fingerprint else None,
            terminal,
            relevant_by_path[classification_path],
        )
    full_jsonl_path, failure_registry, total_known_failures = write_run_records(
        outcomes,
        registry,
        artifact_root=resolved_root,
        manifest={
            "backend": "batch",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_attempts": max_attempts,
            "partitions_completed": sorted(claimed),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
        },
    )
    LOGGER.info(
        "Batch extractor finalize complete: rows=%s successes=%s failures=%s "
        "mentions=%s synthesized=%s audit=%s failure_registry=%s (%s total)",
        len(row_entries),
        len(row_entries) - len(outcomes.failed_rows),
        len(outcomes.failed_rows),
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
