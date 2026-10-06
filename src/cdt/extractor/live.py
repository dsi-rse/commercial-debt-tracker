"""The live extraction backend: run pending items through the workflow one call at a time."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from cdt import settings
from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.completion import (
    CompletedPartition,
    completion_registry_path,
    save_completion_registry,
)
from cdt.datasets import extractor_run_path, resolve_artifact_root, run_manifest_path
from cdt.extractor.batch import active_job_claimed_partition_paths
from cdt.extractor.llm import normalize_reasoning_effort
from cdt.extractor.outputs import (
    _mentions_partition_needs_write,
    _merge_mentions_partition,
    failure_record,
    merge_row_failures,
    pending_extract_partitions,
    summarize_failure,
)
from cdt.extractor.prior_state import published_mention_rows
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.state import (
    DEFAULT_MAX_ATTEMPTS,
    PUBLISHABLE_ROW_STATES,
    InfrastructureError,
    SupportsChatCompletion,
)
from cdt.extractor.workflow import (
    run_extraction_workflow,
)
from cdt.shared import get_logger
from cdt.storage import (
    read_table,
    write_json_artifact,
    write_text_artifact,
)

LOGGER = get_logger(__name__)

EXTRACTOR_PROGRESS_LOG_INTERVAL = 10


def extract_pending_items(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    client: SupportsChatCompletion | None = None,
) -> pd.DataFrame:
    """Extract instrument mentions for classified item partitions."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    resolved_model = model or settings.EXTRACTOR_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    processed_frames: list[pd.DataFrame] = []
    failed_rows: dict[str, dict[str, object]] = {}
    succeeded_item_ids: set[str] = set()
    audit_records: list[str] = []
    partitions_written: list[str] = []
    visited_classification_paths: set[str] = set()
    empty_partitions = 0
    # Partitions the active batch job claimed are its to finish: extracting them
    # live too would pay for every row twice and let the job's later finalize
    # overwrite the newer live mentions with stale results.
    claimed_by_batch_job = (
        set()
        if force
        else active_job_claimed_partition_paths(resolved_root, data_dir=data_dir)
    )
    if claimed_by_batch_job:
        LOGGER.info(
            "Skipping %s classification partition(s) claimed by the active batch "
            "extract job; a poll tick will finish them.",
            len(claimed_by_batch_job),
        )

    pending_partitions, registry = pending_extract_partitions(
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
        exclude_paths=claimed_by_batch_job,
    )
    aborted: str | None = None
    total_partitions = len(pending_partitions)
    for partition_index, pending in enumerate(pending_partitions, start=1):
        partition = {"date": pending.date, "shard": pending.shard}
        partition_label = f"date={pending.date} shard={pending.shard}"
        partition_start = perf_counter()
        visited_classification_paths.add(pending.classification_path)
        batch_items = read_table(
            pending.classification_path,
            CLASSIFIED_ITEM_COLUMNS,
        ).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
        relevant_items = batch_items.loc[batch_items["relevance"].fillna(False)]
        # Row-level work list: rows that already reached a terminal state in an
        # earlier pass are never re-paid; rows ingest merged in later are
        # exactly the ones missing from done_item_ids.
        relevant_records = [
            record
            for record in relevant_items.to_dict("records")
            if str(record["item_id"]) not in pending.done_item_ids
        ]
        relevant_item_ids = {
            str(value) for value in relevant_items["item_id"].astype(str)
        }
        terminal_ids = set(pending.done_item_ids)
        mention_rows: list[dict[str, object]] = []
        replaced_item_ids: set[str] = set()
        partition_failures = 0
        total_relevant_items = len(relevant_records)
        for item_index, item_row in enumerate(relevant_records, start=1):
            try:
                row_state = asyncio.run(
                    run_extraction_workflow(
                        item_row=item_row,
                        model=resolved_model,
                        reasoning_effort=resolved_reasoning,
                        max_attempts=max_attempts,
                        client=client,
                    )
                )
            except InfrastructureError as exc:
                # A provider failure predicts thousands more: stop the run now.
                # Everything terminal so far in this partition is persisted, so
                # the retry pays only for what never got a verdict.
                aborted = str(exc)
                break
            audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
            terminal_ids.add(row_state.item_id)
            replaced_item_ids.add(row_state.item_id)
            if row_state.state in PUBLISHABLE_ROW_STATES:
                mention_rows.extend(published_mention_rows(row_state))
            if row_state.state == "SUCCESS":
                succeeded_item_ids.add(row_state.item_id)
            else:
                failed_rows[row_state.item_id] = failure_record(
                    row_state,
                    partition_date=pending.date,
                    shard=pending.shard,
                    run_id=run_id,
                    backend="live",
                )
                partition_failures += 1
            if (
                item_index == total_relevant_items
                or item_index % EXTRACTOR_PROGRESS_LOG_INTERVAL == 0
            ):
                LOGGER.info(
                    "Extractor item progress: %s partition=%s/%s items=%s/%s mentions=%s failures=%s elapsed=%.1fs",
                    partition_label,
                    partition_index,
                    total_partitions,
                    item_index,
                    total_relevant_items,
                    len(mention_rows),
                    partition_failures,
                    perf_counter() - partition_start,
                )

        mentions = pd.DataFrame(mention_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
        # Rows this partition was extracted for last time and no longer has.
        retired_item_ids = set(pending.done_item_ids) - relevant_item_ids
        # One condition, shared with the batch path.
        if _mentions_partition_needs_write(
            resolved_root,
            data_dir=data_dir,
            partition=partition,
            new_mentions=mentions,
            replaced_item_ids=replaced_item_ids,
            retired_item_ids=retired_item_ids,
        ):
            partitions_written.append(
                _merge_mentions_partition(
                    resolved_root,
                    data_dir=data_dir,
                    partition=partition,
                    new_mentions=mentions,
                    replaced_item_ids=replaced_item_ids,
                    retired_item_ids=retired_item_ids,
                )
            )
            if not mentions.empty:
                processed_frames.append(mentions)
        else:
            empty_partitions += 1
        registry[pending.classification_path] = CompletedPartition(
            fingerprint=pending.fingerprint,
            # Scoped to rows the source still holds: an id whose row is gone
            # just had its mentions pruned, and keeping it here would retire
            # it again on every later pass.
            item_ids=frozenset(terminal_ids & relevant_item_ids),
            complete=relevant_item_ids <= terminal_ids,
        )
        LOGGER.info(
            "Extraction partition complete: %s progress=%s/%s classified_items=%s relevant_items=%s mentions=%s wrote_output=%s elapsed=%.1fs",
            partition_label,
            partition_index,
            total_partitions,
            len(batch_items),
            total_relevant_items,
            len(mentions),
            not mentions.empty,
            perf_counter() - partition_start,
        )
        if aborted is not None:
            break

    save_completion_registry(
        "extract",
        registry,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )
    failure_registry, total_known_failures = merge_row_failures(
        failed_rows,
        succeeded_item_ids,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )

    if audit_records:
        write_text_artifact(full_jsonl_path, "\n".join(audit_records) + "\n")
    else:
        write_text_artifact(full_jsonl_path, "")
    write_json_artifact(
        run_manifest_path(
            "extract",
            run_id,
            artifact_root=resolved_root,
            data_dir=data_dir,
        ),
        {
            "artifact_root": resolved_root,
            "stage": "extract",
            "batch_size": batch_size,
            "force": force,
            "model": resolved_model,
            "reasoning_effort": resolved_reasoning,
            "max_attempts": max_attempts,
            "partitions_visited": sorted(visited_classification_paths),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
            "failure_count": len(failed_rows),
            "audit_path": full_jsonl_path,
            "completion_registry": completion_registry_path(
                "extract", artifact_root=resolved_root, data_dir=data_dir
            ),
            "failure_registry": failure_registry,
            "aborted_on_infrastructure_error": aborted,
        },
    )
    if aborted is not None:
        # Persisted everything first (registry, mentions, audit, failures), so
        # the retry resumes from exactly the rows that never got a verdict.
        raise InfrastructureError(aborted)

    LOGGER.info(
        "Extractor complete: successes=%s failures=%s mentions=%s synthesized=%s "
        "run_dir=%s failure_registry=%s (%s total)",
        sum(
            len(frame["item_id"].unique())
            for frame in processed_frames
            if not frame.empty
        ),
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


def extract_tables(
    classified_items: pd.DataFrame,
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    client: SupportsChatCompletion | None = None,
) -> dict[str, pd.DataFrame]:
    """Run in-memory extraction and return instrument mention tables."""
    del force
    if classified_items.empty:
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            )
        }
    relevant_items = (
        classified_items.loc[classified_items["relevance"].fillna(False)]
        if "relevance" in classified_items
        else classified_items
    )
    if relevant_items.empty:
        return {
            "debt_instrument_mentions": pd.DataFrame(
                columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            )
        }
    resolved_model = model or settings.EXTRACTOR_MODEL
    resolved_reasoning = normalize_reasoning_effort(reasoning_effort)
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    full_jsonl_path = extractor_run_path(
        run_id, artifact_root=resolved_root, data_dir=data_dir
    )
    rows: list[dict[str, object]] = []
    audit_records: list[str] = []
    for item_row in relevant_items.to_dict("records"):
        row_state = asyncio.run(
            run_extraction_workflow(
                item_row=item_row,
                model=resolved_model,
                reasoning_effort=resolved_reasoning,
                max_attempts=max_attempts,
                client=client,
            )
        )
        audit_records.append(json.dumps(row_state.to_audit_dict(), sort_keys=True))
        if row_state.state in PUBLISHABLE_ROW_STATES:
            rows.extend(published_mention_rows(row_state))
        else:
            LOGGER.warning(
                "In-memory extractor failed for item %s: %s",
                row_state.item_id,
                summarize_failure(row_state),
            )
    write_text_artifact(
        full_jsonl_path,
        ("\n".join(audit_records) + "\n") if audit_records else "",
    )
    return {
        "debt_instrument_mentions": pd.DataFrame(
            rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS
        )
    }
