"""Run the post-match lineage pass, then publish the final snapshot tables unless their sources are unchanged."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cdt.classifier.sixk import sixk_snippets_root
from cdt.datasets import (
    GENRES,
)
from cdt.extractor import mentions_root
from cdt.matcher import (
    debt_instruments_root,
    mention_cluster_edges_root,
)
from cdt.matcher.lineage_inference import apply_lineage_inference_pass
from cdt.matcher.schema import MATCHER_SCHEMA_VERSION
from cdt.segmenter.core import ITEM_COLUMNS, items_root
from cdt.shared import get_logger, log_stage_complete, log_stage_start
from cdt.storage.columns import coerce_dataset_text
from cdt.storage.objects import (
    ArtifactPath,
    artifact_exists,
    artifact_tree_digest,
    delete_artifact,
    join_artifact_path,
    list_artifacts,
    read_json_artifact,
    write_json_artifact,
)
from cdt.storage.tables import count_table_rows, read_dataset, write_table

LOGGER = get_logger(__name__)

FINAL_OUTPUT_TABLES: dict[str, tuple[Callable[..., str], ...]] = {
    "items": (items_root, sixk_snippets_root),
    "debt-instruments": (debt_instruments_root,),
    "debt-instrument-mentions": (mentions_root,),
    "mention-cluster-edges": (mention_cluster_edges_root,),
}

#: Columns a published table is projected to when its datasets differ in width.
FINAL_OUTPUT_TABLE_COLUMNS: dict[str, list[str]] = {"items": ITEM_COLUMNS}

#: Column stamped on a unioned table's rows naming the genre they came from.
FORM_TYPE_COLUMN = "form_type"


#: Published table -> the ``form_type`` stamped on each dataset it unions,
#: positionally matching FINAL_OUTPUT_TABLES. The ``items`` union takes one
#: dataset per genre, in genre order (each genre's ``item_dataset``). Its root
#: functions stay named rather than derived, because their names are part of
#: the publish digest.
FINAL_OUTPUT_TABLE_FORM_TYPES: dict[str, tuple[str, ...]] = {
    "items": tuple(GENRES),
}


# A table shrinking below this fraction of its published row count blocks the
# publish unless ``force_publish``.
FINAL_SNAPSHOT_GUARD_RATIO = 0.5


#: Pointer key holding the source digest a generation was built from. Absent
#: means unknown, which publishes.
PUBLISH_SOURCE_DIGEST_KEY = "source_digest"

#: Bump whenever the publish writes something different for the same source
#: bytes (a projection, the ``form_type`` stamp, ``normalize_snapshot_text``):
#: the source digest cannot see code, so without a bump the gate keeps skipping.
PUBLISH_FORMAT_VERSION = 1


def publish_source_digest(
    artifact_root: ArtifactPath, *, data_dir: Path | None = None
) -> str:
    """Digest every partition a publish would read, and how it would read them.

    Covers the content versions (not mtimes) of every ``.parquet`` under the
    source roots, plus the table layout, ``MATCHER_SCHEMA_VERSION`` and
    ``PUBLISH_FORMAT_VERSION``. One LIST per root on S3.
    """
    layout = {
        table_name: [
            dataset_root_fn.__name__
            for dataset_root_fn in (entry if isinstance(entry, tuple) else (entry,))
        ]
        for table_name, entry in FINAL_OUTPUT_TABLES.items()
    }
    return artifact_tree_digest(
        _publish_source_roots(artifact_root, data_dir=data_dir),
        suffix=".parquet",
        context={
            "publish_format_version": PUBLISH_FORMAT_VERSION,
            "matcher_schema_version": MATCHER_SCHEMA_VERSION,
            "layout": layout,
        },
    )


def _publish_source_roots(
    artifact_root: ArtifactPath, *, data_dir: Path | None
) -> list[str]:
    """Every dataset root a publish reads, deduplicated, in table order."""
    roots: list[str] = []
    for entry in FINAL_OUTPUT_TABLES.values():
        dataset_root_fns = entry if isinstance(entry, tuple) else (entry,)
        for dataset_root_fn in dataset_root_fns:
            root = dataset_root_fn(artifact_root, data_dir=data_dir)
            if root not in roots:
                roots.append(root)
    return roots


def publish_would_republish_nothing(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force_publish: bool = False,
    source_digest: str | None = None,
) -> bool:
    """Return whether the publish can be skipped because its sources are unchanged.

    True when there is no final database root, or when the pointer's recorded
    source digest equals the current one and every published table's
    ``latest.parquet`` exists. False when ``force_publish`` is set, the pointer or its
    digest is missing, the digest differs, or a published table is missing.
    ``source_digest`` is the caller's ``publish_source_digest`` if already
    computed; None computes it here.
    """
    if force_publish:
        return False
    if final_database_root is None:
        return True
    pointer_path = final_pointer_path(artifact_root)
    if not artifact_exists(pointer_path):
        return False
    pointer = read_json_artifact(pointer_path)
    recorded = (
        pointer.get(PUBLISH_SOURCE_DIGEST_KEY) if isinstance(pointer, dict) else None
    )
    if not recorded:
        LOGGER.info(
            "Publishing: %s records no source digest, so whether the sources "
            "have moved since it was written is unknown.",
            pointer_path,
        )
        return False
    if source_digest is None:
        source_digest = publish_source_digest(artifact_root, data_dir=data_dir)
    if recorded != source_digest:
        return False
    if not all(
        artifact_exists(
            join_artifact_path(str(final_database_root), table_name, "latest.parquet")
        )
        for table_name in FINAL_OUTPUT_TABLES
    ):
        LOGGER.info(
            "Publishing despite unchanged sources: %s has no complete published "
            "generation yet.",
            final_database_root,
        )
        return False
    LOGGER.info(
        "Skipping final publish: no partition under the published datasets has "
        "changed since generation %s, so the published snapshot is already "
        "current. Use --force-publish to publish anyway.",
        pointer.get("run_id", "unknown"),
    )
    return True


def finalize_after_match(
    matched_instruments: pd.DataFrame,
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force_publish: bool = False,
    renew: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Run the lineage post-pass, then publish unless the gate says skip.

    Every whole run that matches and publishes finishes through here; ``cdt
    match`` runs the lineage pass itself and ``cdt publish`` only publishes. The lineage
    pass is skipped when ``matched_instruments`` is empty. ``renew`` extends
    the caller's writer lease before each long step, so a stolen lease cannot
    keep publishing. Each step logs its ``Starting stage``/``Completed stage``
    lines, whichever entry point called it.

    Returns:
        Published table name -> snapshot path; empty when nothing was published.
    """
    if not matched_instruments.empty:
        if renew is not None:
            renew()
        log_stage_start(LOGGER, "infer-lineage")
        lineage_stats = apply_lineage_inference_pass(
            artifact_root, data_dir=data_dir, renew=renew
        )
        log_stage_complete(LOGGER, "infer-lineage", **lineage_stats)
    return publish_final_tables(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        data_dir=data_dir,
        force_publish=force_publish,
        renew=renew,
    )


def publish_final_tables(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force_publish: bool = False,
    renew: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Publish the four final tables from the canonical datasets, unless the gate says skip.

    Reads whatever ``match`` (and its lineage pass) last wrote; ``cdt publish``
    calls this directly, every whole run through :func:`finalize_after_match`.
    ``renew`` extends the caller's writer lease before the write.

    Returns:
        Published table name -> snapshot path; empty when nothing was published.
    """
    log_stage_start(LOGGER, "finalize", output_root=final_database_root)
    # After lineage (which writes debt-instruments), before the publish reads.
    source_digest = (
        None
        if final_database_root is None
        else publish_source_digest(artifact_root, data_dir=data_dir)
    )
    if publish_would_republish_nothing(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        data_dir=data_dir,
        force_publish=force_publish,
        source_digest=source_digest,
    ):
        log_stage_complete(
            LOGGER, "finalize", tables=0, output_root=final_database_root
        )
        return {}
    if renew is not None:
        renew()
    final_outputs = write_final_output_tables(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        data_dir=data_dir,
        force_publish=force_publish,
        source_digest=source_digest,
    )
    log_stage_complete(
        LOGGER, "finalize", tables=len(final_outputs), output_root=final_database_root
    )
    return final_outputs


def final_snapshots_root(artifact_root: ArtifactPath) -> str:
    """Return the root, under the artifact root, of snapshot generations and pointer."""
    return join_artifact_path(str(artifact_root), "final-snapshots")


def final_pointer_path(artifact_root: ArtifactPath) -> str:
    """Return the path of the atomic latest.json snapshot pointer."""
    return join_artifact_path(final_snapshots_root(artifact_root), "latest.json")


def write_final_output_tables(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    data_dir: Path | None = None,
    force_publish: bool = False,
    source_digest: str | None = None,
) -> dict[str, str]:
    """Publish the final tables as one generation behind an atomic pointer.

    Writes every table under ``final-snapshots/snapshot=<run_id>/``, replaces
    ``latest.json`` (run id, schema version, per-table paths and row counts),
    refreshes each ``<table>/latest.parquet`` under the final database root,
    then records ``source_digest`` in the pointer. Prunes all generations but
    the current and prior one. No-op when ``final_database_root`` is None.

    Returns:
        Published table name -> snapshot path.

    Raises:
        ValueError: Unless ``force_publish``, if a published table would shrink below
            FINAL_SNAPSHOT_GUARD_RATIO of its current row count.
    """
    if final_database_root is None:
        return {}
    # Taken before the reads, so a source written while they run reads as moved
    # (one redundant publish next time) rather than as already published.
    if source_digest is None:
        source_digest = publish_source_digest(artifact_root, data_dir=data_dir)

    pointer_path = final_pointer_path(artifact_root)
    previous: dict[str, object] = {}
    if artifact_exists(pointer_path):
        payload = read_json_artifact(pointer_path)
        if isinstance(payload, dict):
            previous = payload

    # Placeholder text is nulled once, up front, so the immutable snapshot and
    # the parquet-only database root publish the same normalized values.
    tables = {
        table_name: normalize_snapshot_text(
            _read_published_table(
                table_name,
                dataset_root_fns,
                artifact_root=artifact_root,
                data_dir=data_dir,
            )
        )
        for table_name, dataset_root_fns in FINAL_OUTPUT_TABLES.items()
    }
    # Guard against the published tables, not the pointer: a half-built
    # artifact root has no pointer and must still not clobber a good database.
    previous_counts = {
        table_name: rows
        for table_name in FINAL_OUTPUT_TABLES
        if (
            rows := count_table_rows(
                join_artifact_path(
                    str(final_database_root), table_name, "latest.parquet"
                )
            )
        )
        is not None
    }
    _guard_against_shrinkage(tables, previous_counts, force_publish=force_publish)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots_root = final_snapshots_root(artifact_root)
    snapshot_prefix = join_artifact_path(snapshots_root, f"snapshot={run_id}")
    written_paths: dict[str, str] = {}
    pointer_tables: dict[str, object] = {}
    for table_name, table in tables.items():
        snapshot_path = join_artifact_path(snapshot_prefix, f"{table_name}.parquet")
        written_paths[table_name] = write_table(snapshot_path, table)
        pointer_tables[table_name] = {
            "path": written_paths[table_name],
            "rows": len(table),
        }
    pointer = {
        "run_id": run_id,
        "written_at": datetime.now(UTC).isoformat(),
        "schema_version": MATCHER_SCHEMA_VERSION,
        "tables": pointer_tables,
    }
    write_json_artifact(pointer_path, pointer)

    # The parquet-only contract surface, refreshed after the pointer so the
    # consistent generation is always resolvable first.
    for table_name, table in tables.items():
        write_table(
            join_artifact_path(str(final_database_root), table_name, "latest.parquet"),
            table,
        )

    # The digest goes in last: a crash before here must leave no digest, so
    # the next run publishes rather than skipping over stale tables.
    write_json_artifact(
        pointer_path, {**pointer, PUBLISH_SOURCE_DIGEST_KEY: source_digest}
    )

    _prune_old_snapshots(
        snapshots_root,
        keep_run_ids={run_id, str(previous.get("run_id", ""))},
    )
    return written_paths


def _read_published_table(
    table_name: str,
    dataset_root_fns: Sequence[Callable[..., str]],
    *,
    artifact_root: ArtifactPath,
    data_dir: Path | None,
) -> pd.DataFrame:
    """Read one published table, concatenating the datasets it unions.

    Each frame is projected to FINAL_OUTPUT_TABLE_COLUMNS and stamped with its
    FINAL_OUTPUT_TABLE_FORM_TYPES entry, where the table has them. Empty frames
    are left out of the concat so they cannot widen integer columns to float.
    """
    columns = FINAL_OUTPUT_TABLE_COLUMNS.get(table_name)
    form_types = FINAL_OUTPUT_TABLE_FORM_TYPES.get(table_name)
    frames: list[pd.DataFrame] = []
    for index, dataset_root_fn in enumerate(dataset_root_fns):
        frame = read_dataset(
            dataset_root_fn(artifact_root, data_dir=data_dir), columns=columns
        )
        if form_types is not None:
            frame[FORM_TYPE_COLUMN] = form_types[index]
        frames.append(frame)
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        return frames[0]
    if len(populated) == 1:
        return populated[0]
    return pd.concat(populated, ignore_index=True)


def normalize_snapshot_text(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a table with placeholder text (such as ``nan``) nulled."""
    if table.empty:
        return table
    normalized = table.copy()
    for column in normalized.columns:
        if normalized[column].dtype != object:
            continue
        normalized[column] = normalized[column].map(normalize_snapshot_cell)
    return normalized


def normalize_snapshot_cell(value: object) -> object:
    """Null one placeholder string; return non-string values unchanged."""
    if isinstance(value, str):
        return coerce_dataset_text(value)
    return value


def _guard_against_shrinkage(
    tables: dict[str, pd.DataFrame],
    previous_counts: dict[str, int],
    *,
    force_publish: bool,
) -> None:
    """Refuse to publish a table shrinking below the guard ratio, unless ``force_publish``.

    Raises:
        ValueError: If any table regressed and ``force_publish`` is False.
    """
    regressions: list[str] = []
    for table_name, table in tables.items():
        prior_rows = previous_counts.get(table_name, 0)
        if prior_rows <= 0:
            continue
        if len(table) < prior_rows * FINAL_SNAPSHOT_GUARD_RATIO:
            regressions.append(f"{table_name}: {prior_rows} -> {len(table)} rows")
    if not regressions:
        return
    if force_publish:
        LOGGER.warning(
            "Publishing snapshot despite row-count regressions (forced): %s",
            "; ".join(regressions),
        )
        return
    msg = (
        "Refusing to publish a final snapshot with large row-count regressions "
        f"({'; '.join(regressions)}). This usually means a bug or a half-built "
        "artifact root; re-run with --force-publish to publish anyway."
    )
    raise ValueError(msg)


def _prune_old_snapshots(snapshots_root: str, *, keep_run_ids: set[str]) -> None:
    """Delete snapshot generations whose run id is not in ``keep_run_ids``.

    The prior generation is kept so a reader of the previous pointer can finish.
    """
    keep_prefixes = tuple(f"snapshot={run_id}/" for run_id in keep_run_ids if run_id)
    for path in list_artifacts(snapshots_root, suffix=".parquet"):
        if not any(prefix in path for prefix in keep_prefixes):
            delete_artifact(path)
