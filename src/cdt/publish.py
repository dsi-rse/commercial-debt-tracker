"""Publish the final snapshot tables, cut to the publish contract, unless their sources are unchanged.

The contract (``publish_contract.json``, beside this module) is what the
website reads: every published table's columns and Arrow types, and the shape
of every JSON payload column. A generation is written under the final
database root as ``snapshots/snapshot=<run_id>/<table>.parquet`` behind one
``latest.json`` pointer, and each table is also copied to
``<table>/latest.parquet``. Every table is streamed one source partition at a
time, so memory does not grow with the corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import pyarrow as pa
import pyarrow.compute

from cdt.classifier.sixk import sixk_snippets_root
from cdt.datasets import GENRES
from cdt.extractor import mentions_root
from cdt.matcher import (
    debt_instruments_root,
    mention_cluster_edges_root,
)
from cdt.matcher.schema import MATCHER_SCHEMA_VERSION, debt_instrument_redirects_root
from cdt.segmenter.core import items_root
from cdt.shared import get_logger, log_stage_complete, log_stage_start
from cdt.storage.columns import MISSING_TEXT_VALUES
from cdt.storage.objects import (
    ArtifactPath,
    artifact_exists,
    artifact_tree_digest,
    copy_artifact,
    delete_artifact,
    join_artifact_path,
    list_artifacts,
    read_json_artifact,
    write_json_artifact,
)
from cdt.storage.tables import (
    count_table_rows,
    parquet_table_writer,
    read_arrow_table,
    read_dataset,
)

LOGGER = get_logger(__name__)

PUBLISH_CONTRACT_PATH = Path(__file__).with_name("publish_contract.json")

FINAL_OUTPUT_TABLES: dict[str, tuple[Callable[..., str], ...]] = {
    "items": (items_root, sixk_snippets_root),
    "debt-instruments": (debt_instruments_root,),
    "debt-instrument-mentions": (mentions_root,),
    "mention-cluster-edges": (mention_cluster_edges_root,),
    "debt-instrument-redirects": (debt_instrument_redirects_root,),
}

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

#: The published table whose rows are kept only when a published mention
#: references them, by this column.
REFERENCED_ITEMS_TABLE = "items"
REFERENCED_ITEMS_KEY = "item_id"


# A table shrinking below this fraction of its published row count blocks the
# publish unless ``force_publish``.
FINAL_SNAPSHOT_GUARD_RATIO = 0.5
#: Tables the shrink guard does not check: redirects shrink legitimately (a
#: retired id comes back into use) and are small enough to swing by half.
SHRINK_GUARD_EXEMPT_TABLES = frozenset({"debt-instrument-redirects"})

#: Non-null payloads checked against the contract per column and partition: a
#: change in the writer's shapes affects every row, so a sample finds it.
PAYLOAD_CHECK_SAMPLE = 200


#: Pointer key holding the source digest a generation was built from. Absent
#: means unknown, which publishes.
PUBLISH_SOURCE_DIGEST_KEY = "source_digest"

#: Bump whenever the publish writes something different for the same source
#: bytes (a projection, the ``form_type`` stamp, ``null_placeholder_text``,
#: a contract change): the source digest cannot see code, so without a bump
#: the gate keeps skipping.
PUBLISH_FORMAT_VERSION = 2

_DECIMAL_TYPE = re.compile(r"decimal128\((\d+), (\d+)\)")
_NAMED_TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "bool": pa.bool_(),
    "int64": pa.int64(),
    "double": pa.float64(),
}


@cache
def publish_contract() -> dict[str, object]:
    """Return the publish contract, parsed from ``publish_contract.json``."""
    return json.loads(PUBLISH_CONTRACT_PATH.read_text(encoding="utf-8"))


@cache
def publish_contract_digest() -> str:
    """Return the sha256 of ``publish_contract.json``'s bytes."""
    return hashlib.sha256(PUBLISH_CONTRACT_PATH.read_bytes()).hexdigest()


def payload_problem(
    value: object, shape: dict[str, object], span_keys: set[str]
) -> str | None:
    """Return how one parsed payload breaks its contract shape, or None.

    ``shape`` is one entry of the contract's ``payloads.columns``: a
    ``container`` (``object`` or ``array``) whose elements carry the listed
    ``keys`` (or are ``items`` strings), whose ``spans`` carry
    ``span_keys``, and whose ``nested`` objects, when not null, have their
    own shape.
    """
    if shape["container"] == "object":
        if not isinstance(value, dict):
            return "not an object"
        elements: list[object] = [value]
    else:
        if not isinstance(value, list):
            return "not an array"
        elements = value
    for element in elements:
        if "items" in shape:
            if not isinstance(element, str):
                return "an element is not a string"
            continue
        if not isinstance(element, dict):
            return "an element is not an object"
        missing = set(shape["keys"]) - set(element)  # type: ignore[arg-type]
        if missing:
            return f"missing keys {sorted(missing)}"
        for span in element.get("spans") or []:
            if not isinstance(span, dict) or not span_keys <= set(span):
                return "a span is missing keys"
        nested_shapes: dict[str, dict[str, object]] = shape.get("nested", {})  # type: ignore[assignment]
        for name, nested_shape in nested_shapes.items():
            nested = element.get(name)
            if nested is None:
                continue
            problem = payload_problem(nested, nested_shape, span_keys)
            if problem is not None:
                return f"{name}: {problem}"
    return None


def contract_type(name: str) -> pa.DataType:
    """Return the Arrow type a contract type name stands for."""
    if name in _NAMED_TYPES:
        return _NAMED_TYPES[name]
    match = _DECIMAL_TYPE.fullmatch(name)
    if match is None:
        msg = f"Unknown publish contract type: {name!r}"
        raise ValueError(msg)
    return pa.decimal128(int(match.group(1)), int(match.group(2)))


def contract_schema(table_name: str) -> pa.Schema:
    """Return one published table's Arrow schema, from the contract."""
    tables = publish_contract()["tables"]
    columns = tables[table_name]["columns"]  # type: ignore[index]
    return pa.schema(
        [pa.field(column["name"], contract_type(column["type"])) for column in columns]
    )


def publish_source_digest(
    artifact_root: ArtifactPath,
) -> str:
    """Digest every partition a publish would read, and how it would read them.

    Covers the content versions (not mtimes) of every ``.parquet`` under the
    source roots, plus the table layout, ``MATCHER_SCHEMA_VERSION``,
    ``PUBLISH_FORMAT_VERSION`` and the contract file's bytes. One LIST per
    root on S3.
    """
    layout = {
        table_name: [dataset_root_fn.__name__ for dataset_root_fn in entry]
        for table_name, entry in FINAL_OUTPUT_TABLES.items()
    }
    return artifact_tree_digest(
        _publish_source_roots(artifact_root),
        suffix=".parquet",
        context={
            "publish_format_version": PUBLISH_FORMAT_VERSION,
            "matcher_schema_version": MATCHER_SCHEMA_VERSION,
            "contract_sha256": publish_contract_digest(),
            "layout": layout,
        },
    )


def _publish_source_roots(
    artifact_root: ArtifactPath,
) -> list[str]:
    """Every dataset root a publish reads, deduplicated, in table order."""
    roots: list[str] = []
    for entry in FINAL_OUTPUT_TABLES.values():
        for dataset_root_fn in entry:
            root = dataset_root_fn(artifact_root)
            if root not in roots:
                roots.append(root)
    return roots


def published_table_path(final_database_root: ArtifactPath, table_name: str) -> str:
    """Return one table's ``latest.parquet`` under the final database root."""
    return join_artifact_path(str(final_database_root), table_name, "latest.parquet")


def final_snapshots_root(final_database_root: ArtifactPath) -> str:
    """Return the root, under the final database root, of snapshot generations."""
    return join_artifact_path(str(final_database_root), "snapshots")


def final_pointer_path(final_database_root: ArtifactPath) -> str:
    """Return the path of the ``latest.json`` generation pointer.

    It sits beside the published tables, so a reader resolves every table of
    one generation from it.
    """
    return join_artifact_path(str(final_database_root), "latest.json")


def publish_would_republish_nothing(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
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
    pointer_path = final_pointer_path(final_database_root)
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
        source_digest = publish_source_digest(artifact_root)
    if recorded != source_digest:
        return False
    if not all(
        artifact_exists(published_table_path(final_database_root, table_name))
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


def publish_final_tables(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    force_publish: bool = False,
    renew: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Publish the final tables from the canonical datasets, unless the gate says skip.

    Reads whatever ``match`` last wrote; ``cdt publish`` and every whole run
    call this. ``renew`` extends the caller's writer lease before each table
    is written.

    Returns:
        Published table name -> snapshot path; empty when nothing was published.
    """
    log_stage_start(LOGGER, "finalize", output_root=final_database_root)
    # Before the publish reads, so a source written meanwhile reads as moved.
    source_digest = (
        None if final_database_root is None else publish_source_digest(artifact_root)
    )
    if publish_would_republish_nothing(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        force_publish=force_publish,
        source_digest=source_digest,
    ):
        log_stage_complete(
            LOGGER, "finalize", tables=0, output_root=final_database_root
        )
        return {}
    final_outputs = write_final_output_tables(
        artifact_root=artifact_root,
        final_database_root=final_database_root,
        force_publish=force_publish,
        source_digest=source_digest,
        renew=renew,
    )
    log_stage_complete(
        LOGGER, "finalize", tables=len(final_outputs), output_root=final_database_root
    )
    return final_outputs


def write_final_output_tables(
    *,
    artifact_root: ArtifactPath,
    final_database_root: ArtifactPath | None,
    force_publish: bool = False,
    source_digest: str | None = None,
    renew: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Publish the final tables as one generation behind an atomic pointer.

    Streams every table to ``snapshots/snapshot=<run_id>/`` under the final
    database root, then replaces ``latest.json`` (run id, schema, contract and
    format versions, per-table row counts and paths relative to the final
    database root, so a copied root still resolves), copies each table to
    ``<table>/latest.parquet``, and last records ``source_digest`` in the
    pointer. Prunes all generations but the current and prior one. No-op when
    ``final_database_root`` is None.

    Returns:
        Published table name -> snapshot path.

    Raises:
        PublishGuardError: Unless ``force_publish``, if a published table would
            shrink below FINAL_SNAPSHOT_GUARD_RATIO of its current row count.
            The generation just written is deleted first.
    """
    if final_database_root is None:
        return {}
    # Taken before the reads, so a source written while they run reads as moved
    # (one redundant publish next time) rather than as already published.
    if source_digest is None:
        source_digest = publish_source_digest(artifact_root)

    pointer_path = final_pointer_path(final_database_root)
    previous: dict[str, object] = {}
    if artifact_exists(pointer_path):
        payload = read_json_artifact(pointer_path)
        if isinstance(payload, dict):
            previous = payload

    referenced_items = _referenced_item_ids(artifact_root)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots_root = final_snapshots_root(final_database_root)
    snapshot_prefix = join_artifact_path(snapshots_root, f"snapshot={run_id}")
    written_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for table_name, dataset_root_fns in FINAL_OUTPUT_TABLES.items():
        if renew is not None:
            renew()
        snapshot_path = join_artifact_path(snapshot_prefix, f"{table_name}.parquet")
        row_counts[table_name] = _stream_published_table(
            table_name,
            dataset_root_fns,
            snapshot_path,
            artifact_root=artifact_root,
            referenced_items=referenced_items,
        )
        written_paths[table_name] = snapshot_path

    # Guard against the published tables, not the pointer: a database whose
    # pointer is missing must still not be clobbered by a half-built root.
    previous_counts = {
        table_name: rows
        for table_name in FINAL_OUTPUT_TABLES
        if (
            rows := count_table_rows(
                published_table_path(final_database_root, table_name)
            )
        )
        is not None
    }
    try:
        _guard_against_shrinkage(
            row_counts, previous_counts, force_publish=force_publish
        )
    except PublishGuardError:
        for path in written_paths.values():
            delete_artifact(path)
        raise

    pointer = {
        "run_id": run_id,
        "written_at": datetime.now(UTC).isoformat(),
        "schema_version": MATCHER_SCHEMA_VERSION,
        "contract_version": publish_contract()["contract_version"],
        "publish_format_version": PUBLISH_FORMAT_VERSION,
        "tables": {
            table_name: {
                "path": f"snapshots/snapshot={run_id}/{table_name}.parquet",
                "rows": row_counts[table_name],
            }
            for table_name in FINAL_OUTPUT_TABLES
        },
    }
    write_json_artifact(pointer_path, pointer)

    # The per-table surface, refreshed after the pointer so the consistent
    # generation is always resolvable first.
    for table_name, snapshot_path in written_paths.items():
        if renew is not None:
            renew()
        copy_artifact(
            snapshot_path, published_table_path(final_database_root, table_name)
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


def _referenced_item_ids(artifact_root: ArtifactPath) -> set[str]:
    """Return the item ids some mention references, which are the items published."""
    mentions = read_dataset(mentions_root(artifact_root), columns=["item_id"])
    return set(mentions["item_id"].dropna().astype(str))


def null_placeholder_text(table: pa.Table) -> pa.Table:
    """Return ``table`` with every text value trimmed and placeholder text nulled.

    The Arrow form of :func:`cdt.storage.columns.coerce_dataset_text`: empty
    text and the ``MISSING_TEXT_VALUES`` spellings (``nan``, ``None``, ...) of
    any case become null. Non-text columns are unchanged.
    """
    placeholders = pa.array(sorted(MISSING_TEXT_VALUES | {""}), pa.string())
    for index, field in enumerate(table.schema):
        if not pa.types.is_string(field.type):
            continue
        trimmed = pyarrow.compute.utf8_trim_whitespace(table.column(index))
        placeholder = pyarrow.compute.is_in(
            pyarrow.compute.utf8_lower(trimmed), value_set=placeholders
        )
        table = table.set_column(
            index,
            field,
            pyarrow.compute.if_else(placeholder, pa.scalar(None, field.type), trimmed),
        )
    return table


def conform_to_contract(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Return ``table`` with exactly the contract's columns, in order and types.

    Columns the contract does not name are dropped and missing ones are added
    null, so a stale column on disk never reaches a reader.
    """
    columns = [
        table.column(field.name).cast(field.type)
        if field.name in table.column_names
        else pa.nulls(table.num_rows, field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(columns, schema=schema)


def _stream_published_table(
    table_name: str,
    dataset_root_fns: tuple[Callable[..., str], ...],
    output_path: str,
    *,
    artifact_root: ArtifactPath,
    referenced_items: set[str],
) -> int:
    """Write one published table a source partition at a time; return its row count.

    Each partition is read as Arrow, never as pandas, cut to referenced items
    for ``items``, stamped with its FINAL_OUTPUT_TABLE_FORM_TYPES entry where
    the table has one, has placeholder text nulled
    (:func:`null_placeholder_text`), and is conformed to the contract. Memory
    the partition used is handed back to the system before the next one, so
    the peak does not grow with the number of partitions.

    Logs a warning when rows were written but some contract column was in no
    source partition (it publishes all null), and when sampled payloads
    (:data:`PAYLOAD_CHECK_SAMPLE` per column and partition) break their
    contract shape (:func:`payload_problem`).
    """
    schema = contract_schema(table_name)
    form_types = FINAL_OUTPUT_TABLE_FORM_TYPES.get(table_name)
    referenced = pa.array(sorted(referenced_items), pa.string())
    payloads = publish_contract()["payloads"]
    payload_shapes: dict[str, dict[str, object]] = payloads["columns"].get(  # type: ignore[index]
        table_name, {}
    )
    span_keys = set(payloads["spans"]["keys"])  # type: ignore[index]
    present: set[str] = set()
    problems: dict[str, dict[str, int]] = {}
    rows = 0
    with parquet_table_writer(output_path, schema) as writer:
        for index, dataset_root_fn in enumerate(dataset_root_fns):
            for path in list_artifacts(
                dataset_root_fn(artifact_root), suffix=".parquet"
            ):
                table = read_arrow_table(path, schema.names)
                if table_name == REFERENCED_ITEMS_TABLE:
                    table = table.filter(
                        pyarrow.compute.is_in(
                            table.column(REFERENCED_ITEMS_KEY).cast(pa.string()),
                            value_set=referenced,
                        )
                    )
                if form_types is not None:
                    table = table.append_column(
                        FORM_TYPE_COLUMN,
                        pa.array([form_types[index]] * table.num_rows, pa.string()),
                    )
                if table.num_rows:
                    present.update(table.column_names)
                    _count_payload_problems(table, payload_shapes, span_keys, problems)
                    writer.write_table(
                        conform_to_contract(null_placeholder_text(table), schema)
                    )
                    rows += table.num_rows
                del table
                pa.default_memory_pool().release_unused()
    absent = [name for name in schema.names if name not in present]
    if rows and absent:
        LOGGER.warning(
            "Published %s with contract column(s) no source partition has, so "
            "they are all null: %s",
            table_name,
            ", ".join(absent),
        )
    for column, counts in sorted(problems.items()):
        LOGGER.warning(
            "Published %s.%s has payloads that break the contract shape "
            "(sampled): %s",
            table_name,
            column,
            "; ".join(
                f"{problem} x{count}" for problem, count in sorted(counts.items())
            ),
        )
    return rows


def _count_payload_problems(
    table: pa.Table,
    shapes: dict[str, dict[str, object]],
    span_keys: set[str],
    problems: dict[str, dict[str, int]],
) -> None:
    """Add the contract-shape problems of a sample of ``table``'s payloads to ``problems``."""
    for column, shape in shapes.items():
        if column not in table.column_names:
            continue
        values = table.column(column).drop_null()
        for text in values.slice(0, PAYLOAD_CHECK_SAMPLE).to_pylist():
            if isinstance(text, str) and text.strip().lower() in MISSING_TEXT_VALUES | {
                ""
            }:
                continue
            try:
                problem = payload_problem(json.loads(text), shape, span_keys)
            except (TypeError, json.JSONDecodeError):
                problem = "not JSON"
            if problem is not None:
                counts = problems.setdefault(column, {})
                counts[problem] = counts.get(problem, 0) + 1


def _guard_against_shrinkage(
    row_counts: dict[str, int],
    previous_counts: dict[str, int],
    *,
    force_publish: bool,
) -> None:
    """Refuse to publish a table shrinking below the guard ratio, unless ``force_publish``.

    Tables in SHRINK_GUARD_EXEMPT_TABLES are not checked.

    Raises:
        PublishGuardError: If any table regressed and ``force_publish`` is False.
    """
    regressions: list[str] = []
    for table_name, rows in row_counts.items():
        if table_name in SHRINK_GUARD_EXEMPT_TABLES:
            continue
        prior_rows = previous_counts.get(table_name, 0)
        if prior_rows <= 0:
            continue
        if rows < prior_rows * FINAL_SNAPSHOT_GUARD_RATIO:
            regressions.append(f"{table_name}: {prior_rows} -> {rows} rows")
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
    raise PublishGuardError(msg)


class PublishGuardError(RuntimeError):
    """The publish was refused because a table would shrink past the guard ratio."""


def _prune_old_snapshots(snapshots_root: str, *, keep_run_ids: set[str]) -> None:
    """Delete snapshot generations whose run id is not in ``keep_run_ids``.

    The prior generation is kept so a reader of the previous pointer can finish.
    """
    keep_prefixes = tuple(f"snapshot={run_id}/" for run_id in keep_run_ids if run_id)
    for path in list_artifacts(snapshots_root, suffix=".parquet"):
        if not any(prefix in path for prefix in keep_prefixes):
            delete_artifact(path)
