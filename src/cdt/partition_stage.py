"""The loop every whole-partition stage shares: pending source → month output.

A stage reads one source partition at a time (a ``documents`` date/shard file,
or a month file of a derived dataset) and turns it into output rows, which
belong in the output dataset's month file for the source's month
(:func:`cdt.datasets.source_partition_month`). Which source partitions are
pending comes from the stage's completion registry, keyed by source
fingerprint (``cdt.completion.pending_source_partitions``). Filings a stage
could not process are recorded in its failure list
(:func:`cdt.datasets.failure_registry_path`), keyed by accession number, so they
can be listed and retried.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Self

import pandas as pd

from cdt.completion import (
    CHECKPOINT_INTERVAL_SECONDS,
    CompletedPartition,
    completion_registry_root,
    pending_source_partitions,
    save_completion_registry,
)
from cdt.datasets import (
    dataset_root,
    document_shard,
    load_row_failures,
    month_partition_path,
    parse_source_partition,
    resolve_artifact_root,
    run_manifest_path,
    save_row_failures,
    source_partition_month,
)
from cdt.lease import throttled
from cdt.shared import get_logger
from cdt.storage.objects import artifact_exists, write_json_artifact
from cdt.storage.tables import read_table, write_partition_table

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PartitionOutput:
    """What one source partition produced.

    ``rows`` are every output row the source yields; they replace the rows the
    source produced before. ``complete`` False leaves the partition pending.
    ``done_keys`` is the stage's done set for the source
    (:attr:`cdt.completion.CompletedPartition.item_ids`): rows the next
    processing of this source may carry over instead of recomputing. A held
    output (``complete`` False) with no ``done_keys`` writes nothing; one with
    ``done_keys`` writes its rows and records the keys, so the retry redoes
    only the rest. ``failures`` maps the accession number of each filing the
    stage could not process to why; the stage still writes rows for it,
    marked failed.
    """

    rows: pd.DataFrame
    source_rows: int
    complete: bool = True
    failures: dict[str, str] = field(default_factory=dict)
    done_keys: frozenset[str] | None = None

    @property
    def writes(self: Self) -> bool:
        """Whether this output's rows replace the source's rows in the target."""
        return self.complete or self.done_keys is not None


#: Process one source partition: ``(source_path, partition, done_keys)``.
#: ``partition`` is the parsed source path (``{date, shard}`` or ``{month}``);
#: ``done_keys`` the done set its completion entry recorded (empty when forced
#: or never processed).
ProcessPartition = Callable[[str, dict[str, str], frozenset[str]], PartitionOutput]


def sort_partition_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by ``(cik, date, accession_number)``, stably, for a month file.

    Sorted files let a reader filtering on ``cik`` skip row groups.
    """
    keys = [column for column in ("cik", "date", "accession_number") if column in table]
    table = table.reset_index(drop=True)
    if table.empty or not keys:
        return table
    order = pd.DataFrame({key: table[key].astype(str) for key in keys})
    return table.iloc[
        order.sort_values(by=keys, kind="stable").index.to_numpy()
    ].reset_index(drop=True)


def rows_from_sources(rows: pd.DataFrame, sources: list[dict[str, str]]) -> pd.Series:
    """Return a mask of the rows of a month file that came from ``sources``.

    A month source owns the whole file. A date/shard source owns the rows whose
    ``date`` and ``documents`` shard of ``accession_number`` are its own.
    """
    if any("month" in source for source in sources):
        return pd.Series(True, index=rows.index)
    if rows.empty:
        return pd.Series(False, index=rows.index)
    owned = {(source["date"], source["shard"]) for source in sources}
    keys = zip(
        rows["date"].astype(str),
        rows["accession_number"].astype(str).map(document_shard),
        strict=True,
    )
    return pd.Series([key in owned for key in keys], index=rows.index)


def replace_partition_failures(
    registry: dict[str, dict[str, object]],
    source_path: str,
    failures: dict[str, str],
) -> bool:
    """Replace one source partition's entries in a failure list; return if it changed.

    Every earlier entry for ``source_path`` goes, so a filing that now
    succeeded leaves the list.
    """
    entries = {
        accession: {"source_partition": source_path, "error": error}
        for accession, error in failures.items()
    }
    previous = {
        accession: entry
        for accession, entry in registry.items()
        if entry.get("source_partition") == source_path
    }
    if previous == entries:
        return False
    for accession in previous:
        del registry[accession]
    registry.update(entries)
    return True


@dataclass
class PartitionStageResult:
    """What a stage run produced across every pending partition."""

    rows: pd.DataFrame
    source_rows: int = 0
    #: Month files written, once per flush.
    partitions_written: list[str] = field(default_factory=list)
    #: Completed source partitions that produced no rows.
    empty_partitions: int = 0
    #: Source partitions whose output reported ``complete=False``.
    held_partitions: list[str] = field(default_factory=list)


def run_partition_stage(
    stage_name: str,
    *,
    source_dataset: str,
    output_dataset: str,
    output_columns: list[str],
    process: ProcessPartition,
    artifact_root: str | Path | None = None,
    force: bool = False,
    renew: Callable[[], None] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> PartitionStageResult:
    """Run ``process`` over every pending source partition of ``source_dataset``.

    Outputs are buffered per target month. A month is written once, when the
    run moves on to a source of another month and at each checkpoint: its file
    is read, the rows of the buffered sources are replaced by their new
    output (:func:`rows_from_sources`), and the result is written back sorted
    (:func:`sort_partition_rows`). A source whose output is empty deletes its
    rows; a month file left empty is still written, so the next stage sees its
    rows go, and no file is created for a month that never had rows. A held
    output with no ``done_keys`` writes nothing and stays pending. A completed
    output's ``failures`` replace its entries in the stage's failure list.

    Completion entries are recorded only after their month is written, and
    saved at most every :data:`cdt.completion.CHECKPOINT_INTERVAL_SECONDS` and
    once at the end, so an interruption loses at most that much finished work
    and never records work that was not written. ``renew`` is called before
    each month write, throttled (:func:`cdt.lease.throttled`). A run manifest
    is written at ``runs/<stage_name>/run_id=latest.json``.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    pending, registry = pending_source_partitions(
        stage_name,
        source_dataset,
        artifact_root=resolved_root,
        force=force,
    )
    output_root = dataset_root(output_dataset, artifact_root=resolved_root)

    # Loaded on the first completed partition, so an idle run reads nothing.
    failures: dict[str, dict[str, object]] | None = None
    failures_changed = False
    keep_lease = throttled(renew) if renew is not None else None
    result = PartitionStageResult(rows=pd.DataFrame(columns=output_columns))
    frames: list[pd.DataFrame] = []

    buffered_month: str | None = None
    buffered_sources: list[tuple[dict[str, str], pd.DataFrame]] = []
    buffered_entries: dict[str, CompletedPartition] = {}

    def flush() -> None:
        nonlocal buffered_month
        if buffered_month is None:
            return
        if buffered_sources:
            if keep_lease is not None:
                keep_lease()
            output_path = month_partition_path(
                output_dataset, month=buffered_month, artifact_root=resolved_root
            )
            exists = artifact_exists(output_path)
            existing = (
                read_table(output_path, output_columns).reindex(columns=output_columns)
                if exists
                else pd.DataFrame(columns=output_columns)
            )
            kept = existing.loc[
                ~rows_from_sources(existing, [source for source, _ in buffered_sources])
            ]
            new_frames = [
                rows.reindex(columns=output_columns)
                for _, rows in buffered_sources
                if not rows.empty
            ]
            parts = [frame for frame in [kept, *new_frames] if not frame.empty]
            table = (
                pd.concat(parts, ignore_index=True)
                if parts
                else pd.DataFrame(columns=output_columns)
            )
            if exists or not table.empty:
                write_partition_table(
                    output_root,
                    partition={"month": buffered_month},
                    table=sort_partition_rows(table).reindex(columns=output_columns),
                )
                result.partitions_written.append(output_path)
        for key, entry in buffered_entries.items():
            registry[key] = entry
        buffered_month = None
        buffered_sources.clear()
        buffered_entries.clear()

    def save() -> None:
        nonlocal failures_changed
        flush()
        save_completion_registry(stage_name, registry, artifact_root=resolved_root)
        if failures is not None and failures_changed:
            save_row_failures(stage_name, failures, artifact_root=resolved_root)
            failures_changed = False

    checkpoint = throttled(save, interval_seconds=CHECKPOINT_INTERVAL_SECONDS)
    total = len(pending)
    for index, (source_path, fingerprint) in enumerate(pending, start=1):
        partition = parse_source_partition(source_path)
        month = source_partition_month(partition)
        if buffered_month is not None and month != buffered_month:
            flush()
        buffered_month = month
        previous = registry.get(source_path)
        done = previous.item_ids if previous is not None and not force else frozenset()
        started = perf_counter()
        output = process(source_path, partition, done)
        result.source_rows += output.source_rows
        if output.writes:
            buffered_sources.append((partition, output.rows))
            if not output.rows.empty:
                frames.append(output.rows)
            elif output.complete:
                result.empty_partitions += 1
        if not output.complete:
            result.held_partitions.append(source_path)
        else:
            if failures is None:
                failures = load_row_failures(stage_name, artifact_root=resolved_root)
            failures_changed |= replace_partition_failures(
                failures, source_path, output.failures
            )
        # A held partition's entry is incomplete or matches no fingerprint, so
        # it stays pending even when an earlier run's entry is still stored.
        buffered_entries[source_path] = (
            CompletedPartition(
                fingerprint=fingerprint,
                item_ids=output.done_keys or frozenset(),
                complete=output.complete,
            )
            if output.writes
            else CompletedPartition(fingerprint=None)
        )
        LOGGER.info(
            "%s partition complete: source=%s month=%s progress=%s/%s "
            "source_rows=%s rows=%s held=%s elapsed=%.1fs",
            stage_name,
            source_path,
            month,
            index,
            total,
            output.source_rows,
            len(output.rows),
            not output.complete,
            perf_counter() - started,
        )
        checkpoint()
    if pending:
        save()

    write_json_artifact(
        run_manifest_path(stage_name, "latest", artifact_root=resolved_root),
        {
            "artifact_root": resolved_root,
            "stage": stage_name,
            "force": force,
            **(manifest_extra or {}),
            "source_rows_processed": result.source_rows,
            "partitions_visited": [path for path, _ in pending],
            "partitions_written": result.partitions_written,
            "empty_partitions_skipped_from_write": result.empty_partitions,
            "partitions_held": result.held_partitions,
            "completion_registry": completion_registry_root(
                stage_name, artifact_root=resolved_root
            ),
        },
    )
    LOGGER.info(
        "%s complete: partitions=%s months_written=%s held=%s source_rows=%s",
        stage_name,
        total,
        len(result.partitions_written),
        len(result.held_partitions),
        result.source_rows,
    )
    if frames:
        result.rows = pd.concat(frames, ignore_index=True).reindex(
            columns=output_columns
        )
    return result
