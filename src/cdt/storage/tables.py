"""Parquet tables and datasets: projected reads, Arrow scans with a per-file fallback, writes."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet

from cdt.shared import get_logger
from cdt.storage.columns import apply_declared_column_types
from cdt.storage.objects import (
    S3_CLIENT_CONFIG,
    S3_MAX_ATTEMPTS,
    ArtifactPath,
    boto3_session,
    configured_s3_profile,
    is_s3_uri,
    join_artifact_path,
    list_artifacts,
    normalize_artifact_path,
    parse_s3_uri,
    s3_client,
)

LOGGER = get_logger(__name__)

# Empty ``tmp*.parquet`` files left in a partition directory by a crash under
# write_table's former temp naming; ``**/*.parquet`` readers must skip them.
# Current writes use ``.parquet.tmp``, which this does not match.
_ORPHANED_TEMP_RE = re.compile(r"(?:^|/)tmp[^/]*\.parquet$")


def is_orphaned_temp_artifact(path: ArtifactPath) -> bool:
    """Return whether a path is a ``tmp*.parquet`` file of write_table's old naming."""
    return _ORPHANED_TEMP_RE.search(normalize_artifact_path(path)) is not None


def iter_partition_paths(
    base: ArtifactPath,
    *,
    partition_filter: dict[str, str] | None = None,
) -> Iterator[str]:
    """Yield partition file paths under a dataset, optionally filtered by key/value."""
    for path in list_artifacts(base, suffix=".parquet"):
        if is_orphaned_temp_artifact(path):
            continue
        if partition_filter is None:
            yield path
            continue
        normalized = normalize_artifact_path(path)
        if all(
            f"{key}={value}" in normalized for key, value in partition_filter.items()
        ):
            yield path


# The errors that mean "Arrow cannot read these files as one table" and send
# ``read_partitions`` to its per-file fallback: any ArrowException (partitions
# whose physical types will not unify) or a partition deleted between listing
# and scan. ArrowIOError, how auth and transport failures surface, is not caught:
# those must stay fatal rather than become a silent fallback.
_ARROW_READ_ERRORS = (pyarrow.ArrowException, FileNotFoundError)

# Threads for per-partition footer reads, which happen before the dataset
# scanner (and its threads) exist. Arrow releases the GIL, so they overlap.
_SCHEMA_WORKERS = 32


def _unified_schema(dataset: pyarrow.dataset.Dataset) -> pa.Schema:
    """Merge every fragment's physical schema, reading the footers in parallel.

    Required for correctness, not speed: ``ds.dataset`` infers its schema from
    the first fragment alone, so columns added after that file was written
    (``form_type`` and ``source`` on ``documents``) would be silently dropped.
    Raises the ArrowException from ``unify_schemas`` when fragments are
    incompatible.
    """
    fragments = list(dataset.get_fragments())
    if len(fragments) <= 1:
        # Nothing to merge (the matcher's per-shard reads): skip the pool.
        return dataset.schema if not fragments else fragments[0].physical_schema
    with ThreadPoolExecutor(max_workers=min(_SCHEMA_WORKERS, len(fragments))) as pool:
        schemas = list(pool.map(lambda fragment: fragment.physical_schema, fragments))
    return pyarrow.unify_schemas(schemas)


#: Warn before a scan when less credential lifetime than this remains:
#: ``S3FileSystem`` holds a static key triple and cannot refresh mid-scan, and
#: 900s (botocore's advisory refresh window) is all a freshly frozen credential
#: is guaranteed.
_CREDENTIAL_LIFETIME_WARN_SECONDS = 900


def strip_s3_scheme(path: ArtifactPath) -> str:
    """Return the ``bucket/key`` form Arrow wants for an S3 URI; local paths unchanged."""
    normalized = normalize_artifact_path(path)
    if not is_s3_uri(normalized):
        return normalized
    bucket, key = parse_s3_uri(normalized)
    return f"{bucket}/{key.lstrip('/')}"


def arrow_filesystem(path: ArtifactPath) -> tuple[object | None, str]:
    """Return the pyarrow filesystem and stripped path Arrow should read through.

    Local paths return ``(None, path)``; pyarrow resolves those itself. S3
    paths return a new ``pyarrow.fs.S3FileSystem`` built from the configured
    profile's boto3 Session, with timeouts and retries matching
    ``S3_CLIENT_CONFIG``, and the ``bucket/key`` path. Built per call, not
    memoized, because the Session's frozen credentials may be temporary.
    Construction issues no request.

    Raises RuntimeError when the Session resolves no credentials, rather than
    letting pyarrow fall back to its own credential chain, which can resolve a
    different identity than boto3.
    """
    normalized = normalize_artifact_path(path)
    resolved = strip_s3_scheme(normalized)
    if not is_s3_uri(normalized):
        return None, resolved
    session = boto3_session()
    credentials = session.get_credentials()
    if credentials is None:
        msg = (
            f"No AWS credentials for profile {configured_s3_profile()!r}; refusing "
            f"to read {normalized} through pyarrow's own credential chain, which "
            f"resolves differently from boto3's and would read as another identity."
        )
        raise RuntimeError(msg)
    frozen = credentials.get_frozen_credentials()
    _warn_on_short_credential_lifetime(credentials, normalized)
    kwargs: dict[str, object] = {
        "access_key": frozen.access_key,
        "secret_key": frozen.secret_key,
        "session_token": frozen.token,
        "connect_timeout": S3_CLIENT_CONFIG.connect_timeout,
        "request_timeout": S3_CLIENT_CONFIG.read_timeout,
        "retry_strategy": pyarrow.fs.AwsStandardS3RetryStrategy(
            max_attempts=S3_MAX_ATTEMPTS
        ),
    }
    # Usually unset in production: botocore reads only AWS_DEFAULT_REGION, ECS
    # injects AWS_REGION, and pyarrow resolves that itself. Passed when known
    # only to save the lookup.
    if session.region_name:
        kwargs["region"] = session.region_name
    return pyarrow.fs.S3FileSystem(**kwargs), resolved


def _warn_on_short_credential_lifetime(credentials: object, path: str) -> None:
    """Warn when ``credentials`` expire within ``_CREDENTIAL_LIFETIME_WARN_SECONDS``.

    pyarrow holds the key triple statically for the whole scan, so an expiry
    mid-scan surfaces only as an opaque ``AWS Error UNKNOWN (HTTP status 400)``;
    this names the cause up front. Credentials without an expiry are ignored.
    """
    expiry = getattr(credentials, "_expiry_time", None)
    if expiry is None:
        return
    remaining = (expiry - datetime.now(UTC)).total_seconds()
    if remaining < _CREDENTIAL_LIFETIME_WARN_SECONDS:
        LOGGER.warning(
            "AWS credentials for profile %r expire in %.0fs; pyarrow holds a "
            "static copy for the whole of %s, so a longer scan will fail with an "
            "opaque AWS 400 rather than refreshing.",
            configured_s3_profile(),
            remaining,
            path,
        )


def _open_parquet_file(path: ArtifactPath) -> pyarrow.parquet.ParquetFile:
    """Open a parquet footer without downloading the object's column data."""
    filesystem, resolved = arrow_filesystem(path)
    if filesystem is None:
        return pyarrow.parquet.ParquetFile(Path(resolved))
    return pyarrow.parquet.ParquetFile(resolved, filesystem=filesystem)


def read_table(
    path: ArtifactPath, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a Parquet table projected to ``columns``; an empty table if absent.

    ``columns=None`` reads every column. Projection reads only the wanted
    column chunks (ranged GETs on S3). A requested column the file lacks comes
    back all-null; this is decided from the footer schema, because
    ``ParquetFile.read`` silently omits absent columns rather than raising.
    """
    normalized = normalize_artifact_path(path)
    try:
        parquet_file = _open_parquet_file(normalized)
    except FileNotFoundError:
        # Not checked first: the open already HEADs the object, so a separate
        # existence check would double the requests of every read.
        return pd.DataFrame(columns=columns)
    if columns is None:
        return parquet_file.read().to_pandas()
    requested = list(columns)
    available = set(parquet_file.schema_arrow.names)
    present = [name for name in requested if name in available]
    table = parquet_file.read(columns=present).to_pandas()
    if len(present) == len(requested):
        return table
    return table.reindex(columns=requested)


def count_table_rows(path: ArtifactPath) -> int | None:
    """Return a parquet table's row count from its footer alone; None if absent."""
    normalized = normalize_artifact_path(path)
    try:
        return int(_open_parquet_file(normalized).metadata.num_rows)
    except FileNotFoundError:
        return None


def count_partition_rows(paths: Sequence[ArtifactPath]) -> int:
    """Total the footer row counts of many partitions, reading footers in parallel.

    No paths returns 0; a partition missing since the listing counts as 0.
    """
    if not paths:
        return 0
    with ThreadPoolExecutor(max_workers=min(_SCHEMA_WORKERS, len(paths))) as pool:
        counts = list(pool.map(count_table_rows, paths))
    return sum(count or 0 for count in counts)


def _read_dataset_with_arrow(
    paths: list[str], columns: Sequence[str] | None
) -> tuple[pd.DataFrame | None, Exception | None]:
    """Read many partitions as one parallel Arrow dataset scan.

    The schema is merged across every file by ``_unified_schema``. Returns
    ``(table, None)`` on success, or ``(None, error)`` when the files cannot be
    read as one table (an error in ``_ARROW_READ_ERRORS``: incompatible
    per-partition types, or a partition deleted since the listing), so the
    caller can report which. Requested ``columns`` absent from every file come
    back all-null.
    """
    filesystem, _ = arrow_filesystem(paths[0])
    # One filesystem (and connection pool) for the whole scan; the other paths
    # need only the scheme stripped.
    resolved = [strip_s3_scheme(path) for path in paths]
    try:
        dataset = pyarrow.dataset.dataset(
            resolved, format="parquet", filesystem=filesystem
        )
        schema = _unified_schema(dataset)
        unified = pyarrow.dataset.dataset(
            resolved, format="parquet", filesystem=filesystem, schema=schema
        )
        if columns is None:
            return unified.to_table().to_pandas(), None
        # Project only the columns this root actually has; a name absent
        # everywhere is reindexed in below, matching the pandas path.
        present = [name for name in columns if name in schema.names]
        table = unified.to_table(columns=present).to_pandas()
        return table.reindex(columns=list(columns)), None
    except _ARROW_READ_ERRORS as error:
        return None, error


def read_dataset(
    base: ArtifactPath,
    *,
    columns: Sequence[str] | None = None,
    partition_filter: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read and concatenate all Parquet files under a dataset prefix.

    ``partition_filter`` keeps only paths containing every ``key=value``. Reads
    through ``read_partitions``: one parallel Arrow scan, falling back to
    per-file reads when the partitions cannot be read as one table.
    """
    paths = list(iter_partition_paths(base, partition_filter=partition_filter))
    return read_partitions(paths, columns=columns, label=normalize_artifact_path(base))


def read_partitions(
    paths: Sequence[ArtifactPath],
    *,
    columns: Sequence[str] | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """Read an explicit list of partitions as one table, Arrow scan first.

    Tries ``_read_dataset_with_arrow``; when the partitions cannot be read as
    one table, logs the reason and concatenates per-file ``read_table`` reads.
    ``label`` names the group in that log (default: the first path). No paths
    returns an empty frame with ``columns``.
    """
    if not paths:
        return pd.DataFrame(columns=columns)
    resolved_paths = [normalize_artifact_path(path) for path in paths]
    table, error = _read_dataset_with_arrow(resolved_paths, columns)
    if table is not None:
        return table
    LOGGER.info(
        "Arrow could not read %s partitions under %s as one table (%s: %s); "
        "falling back to a per-file read.",
        len(resolved_paths),
        label or resolved_paths[0],
        type(error).__name__,
        error,
    )
    frames = [read_table(path, columns) for path in resolved_paths]
    return pd.concat(frames, ignore_index=True)


def write_table(path: ArtifactPath, table: pd.DataFrame) -> str:
    """Atomically write a Parquet table, with declared column types; return the path."""
    normalized = normalize_artifact_path(path)
    arrow_table = apply_declared_column_types(table)
    if is_s3_uri(normalized):
        buffer = io.BytesIO()
        pyarrow.parquet.write_table(arrow_table, buffer)
        bucket, key = parse_s3_uri(normalized)
        s3_client().put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        return normalized

    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # The temp suffix must not end in .parquet, or a crash before the rename
    # leaves a file every ``**/*.parquet`` reader picks up and dies on.
    with NamedTemporaryFile(
        dir=local_path.parent, suffix=".parquet.tmp", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        pyarrow.parquet.write_table(arrow_table, temp_path)
        temp_path.replace(local_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(local_path)


def write_partition_table(
    dataset_root: ArtifactPath,
    *,
    partition: dict[str, str],
    table: pd.DataFrame,
    filename: str | None = None,
) -> str:
    """Write one deterministic partition file beneath a dataset root."""
    partition_path = normalize_artifact_path(dataset_root).rstrip("/")
    for key, value in partition.items():
        partition_path = join_artifact_path(partition_path, f"{key}={value}")
    final_path = join_artifact_path(partition_path, filename or "part-0000.parquet")
    return write_table(final_path, table)
