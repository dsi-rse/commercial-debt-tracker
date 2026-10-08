"""The loop every whole-partition stage shares: pending source → output partition.

A stage reads one date/shard source partition at a time, turns it into output
rows, and writes them to the same date/shard of its output dataset. Which
source partitions are pending comes from the stage's completion registry,
keyed by source fingerprint (``cdt.completion.pending_source_partitions``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

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
    date_shard_partition_path,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
)
from cdt.lease import throttled
from cdt.shared import get_logger
from cdt.storage.objects import artifact_exists, write_json_artifact
from cdt.storage.tables import write_partition_table

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PartitionOutput:
    """What one source partition produced.

    ``complete`` False leaves the partition pending: nothing is written and no
    completion entry is recorded, so the next run processes it again.
    """

    rows: pd.DataFrame
    source_rows: int
    complete: bool = True


@dataclass
class PartitionStageResult:
    """What a stage run produced across every pending partition."""

    rows: pd.DataFrame
    source_rows: int = 0
    partitions_written: list[str] = field(default_factory=list)
    #: Completed source partitions that produced no rows, so wrote nothing.
    empty_partitions: int = 0
    #: Source partitions whose output reported ``complete=False``.
    held_partitions: list[str] = field(default_factory=list)


def run_partition_stage(
    stage_name: str,
    *,
    source_dataset: str,
    output_dataset: str,
    output_columns: list[str],
    process: Callable[[str, dict[str, str]], PartitionOutput],
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    force: bool = False,
    renew: Callable[[], None] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> PartitionStageResult:
    """Run ``process`` over every pending source partition of ``source_dataset``.

    ``process`` receives the source partition path and its ``{date, shard}``
    and returns that partition's output, which is written to the same
    date/shard of ``output_dataset``. Empty output completes the partition and
    writes nothing, unless an earlier run's output is there, which is then
    overwritten empty so later stages see the rows go. A held partition (output
    ``complete`` False) writes nothing and stays pending.

    Completion is saved at most every
    :data:`cdt.completion.CHECKPOINT_INTERVAL_SECONDS` and once at the end, so
    an interruption loses at most that much finished work. ``renew`` is called
    before each partition's write, throttled (:func:`cdt.lease.throttled`). A
    run manifest is written at ``runs/<stage_name>/run_id=latest.json``.
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    pending, registry = pending_source_partitions(
        stage_name,
        source_dataset,
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
    )
    output_root = dataset_root(
        output_dataset, artifact_root=resolved_root, data_dir=data_dir
    )

    def save() -> None:
        save_completion_registry(
            stage_name, registry, artifact_root=resolved_root, data_dir=data_dir
        )

    checkpoint = throttled(save, interval_seconds=CHECKPOINT_INTERVAL_SECONDS)
    keep_lease = throttled(renew) if renew is not None else None
    result = PartitionStageResult(rows=pd.DataFrame(columns=output_columns))
    frames: list[pd.DataFrame] = []
    total = len(pending)
    for index, (source_path, fingerprint) in enumerate(pending, start=1):
        partition = parse_date_shard_partition(source_path)
        started = perf_counter()
        output = process(source_path, partition)
        # Before writing: a writer that lost its lease must not write again.
        if keep_lease is not None:
            keep_lease()
        result.source_rows += output.source_rows
        output_path = date_shard_partition_path(
            output_dataset,
            partition_date=partition["date"],
            shard=partition["shard"],
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        if not output.complete:
            result.held_partitions.append(source_path)
        elif not output.rows.empty or artifact_exists(output_path):
            # An emptied partition is overwritten, not left or deleted: its
            # new fingerprint is what tells the next stage to drop the rows it
            # derived from the old ones. With no earlier output, an empty
            # partition writes no file at all.
            write_partition_table(
                output_root,
                partition={"date": partition["date"], "shard": partition["shard"]},
                table=output.rows.reindex(columns=output_columns),
            )
            if not output.rows.empty:
                frames.append(output.rows)
            result.partitions_written.append(output_path)
        else:
            result.empty_partitions += 1
        # A held partition gets an entry that matches no fingerprint, so it stays
        # pending even when an earlier run's entry for it is still stored.
        registry[source_path] = CompletedPartition(
            fingerprint=fingerprint if output.complete else None
        )
        LOGGER.info(
            "%s partition complete: date=%s shard=%s progress=%s/%s "
            "source_rows=%s rows=%s wrote_output=%s held=%s elapsed=%.1fs",
            stage_name,
            partition["date"],
            partition["shard"],
            index,
            total,
            output.source_rows,
            len(output.rows),
            output_path in result.partitions_written,
            not output.complete,
            perf_counter() - started,
        )
        checkpoint()
    if pending:
        save()

    write_json_artifact(
        run_manifest_path(
            stage_name, "latest", artifact_root=resolved_root, data_dir=data_dir
        ),
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
                stage_name, artifact_root=resolved_root, data_dir=data_dir
            ),
        },
    )
    LOGGER.info(
        "%s complete: partitions=%s written=%s held=%s source_rows=%s",
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
