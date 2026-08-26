"""Helpers for reading and updating pipeline artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast
from urllib.parse import urlparse

import boto3
import pandas as pd

ArtifactPath = str | Path

# Basenames NamedTemporaryFile produced before write_table stopped giving temp
# files a .parquet suffix. A crash between create and rename orphaned them
# inside partition directories, where every ``**/*.parquet`` reader choked on
# the empty file (#68).
_ORPHANED_TEMP_RE = re.compile(r"(?:^|/)tmp[^/]*\.parquet$")

# One client for every S3 call in this module: construction is expensive
# (credential resolution, endpoint discovery), and the partition scans issue
# thousands of calls per run (#83).
_S3_CLIENT = None


def _s3_client():  # noqa: ANN202
    global _S3_CLIENT  # noqa: PLW0603
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def is_s3_uri(path: ArtifactPath) -> bool:
    """Return whether the provided path points to S3."""
    return str(path).startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3 URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        msg = f"Expected an s3:// URI, got {uri!r}"
        raise ValueError(msg)
    return parsed.netloc, parsed.path.lstrip("/")


def normalize_artifact_path(path: ArtifactPath) -> str:
    """Return a normalized string path for local or S3 storage."""
    if isinstance(path, Path):
        return str(path)
    return path


def join_artifact_path(base: ArtifactPath, *parts: str) -> str:
    """Join local path segments or S3 URI segments using the right separator."""
    normalized = normalize_artifact_path(base).rstrip("/")
    if is_s3_uri(normalized):
        suffix = "/".join(part.strip("/") for part in parts if part)
        return f"{normalized}/{suffix}" if suffix else normalized
    return str(Path(normalized, *parts))


def artifact_exists(path: ArtifactPath) -> bool:
    """Return whether a local or S3-backed artifact exists."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        client = _s3_client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except client.exceptions.ClientError as error:  # type: ignore[attr-defined]
            code = error.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
    return Path(normalized).exists()


def list_artifacts(base: ArtifactPath, *, suffix: str = "") -> list[str]:
    """List artifact paths recursively beneath a local directory or S3 prefix."""
    normalized = normalize_artifact_path(base).rstrip("/")
    if is_s3_uri(normalized):
        bucket, prefix = parse_s3_uri(normalized)
        paginator = _s3_client().get_paginator("list_objects_v2")
        results: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            contents = cast(list[dict[str, str]], page.get("Contents", []))
            for obj in contents:
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                results.append(f"s3://{bucket}/{key}")
        return sorted(results)
    root = Path(normalized)
    if not root.exists():
        return []
    pattern = f"**/*{suffix}" if suffix else "**/*"
    return sorted(str(path) for path in root.glob(pattern) if path.is_file())


def is_orphaned_temp_artifact(path: ArtifactPath) -> bool:
    """Return whether a path is a tempfile orphaned by a crash mid-write_table."""
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


def read_json_artifact(path: ArtifactPath) -> dict[str, object] | list[object]:
    """Read a JSON artifact from local storage or S3."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return cast(dict[str, object] | list[object], json.loads(body.decode("utf-8")))
    return cast(
        dict[str, object] | list[object],
        json.loads(Path(normalized).read_text(encoding="utf-8")),
    )


def read_text_artifact(path: ArtifactPath) -> str:
    """Read a text artifact from local storage or S3."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8")
    return Path(normalized).read_text(encoding="utf-8")


# S3 error codes raised when a conditional write loses: 412 on a failed
# precondition, 409 when racing another in-flight conditional write.
_CONDITIONAL_WRITE_LOST_CODES = frozenset(
    {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}
)


def _json_body(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def write_json_artifact_if_absent(
    path: ArtifactPath, payload: dict[str, object]
) -> bool:
    """Create a JSON artifact only if it does not exist; return whether we won.

    Uses S3 conditional ``PutObject`` (``If-None-Match: *``) or an exclusive
    local create, so exactly one of several concurrent callers succeeds.
    """
    body = _json_body(payload)
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        client = _s3_client()
        try:
            client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        except client.exceptions.ClientError as error:  # type: ignore[attr-defined]
            code = error.response.get("Error", {}).get("Code")
            if code in _CONDITIONAL_WRITE_LOST_CODES:
                return False
            raise
        return True
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with local_path.open("xb") as handle:
            handle.write(body)
    except FileExistsError:
        return False
    return True


def read_json_artifact_versioned(
    path: ArtifactPath,
) -> tuple[dict[str, object] | list[object] | None, str]:
    """Read a JSON artifact plus an opaque version token for conditional replace.

    A body that does not parse returns ``(None, version)`` rather than raising:
    the callers are lease/lock readers, where a truncated file (a local writer
    killed mid-write) must read as "corrupt, stealable via compare-and-swap"
    instead of wedging every subsequent run before its self-heal logic runs.
    """
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        response = _s3_client().get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        version = str(response["ETag"])
    else:
        body = Path(normalized).read_bytes()
        version = hashlib.sha256(body).hexdigest()
    try:
        payload = cast(
            dict[str, object] | list[object], json.loads(body.decode("utf-8"))
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, version
    return payload, version


def replace_json_artifact_if_match(
    path: ArtifactPath, payload: dict[str, object], *, version: str
) -> bool:
    """Replace a JSON artifact only if it still has ``version``; return success.

    ``version`` is the token from ``read_json_artifact_versioned`` (S3 ETag or a
    local content hash). On S3 this is an atomic compare-and-swap; locally it is
    a best-effort check acceptable for single-user development.
    """
    body = _json_body(payload)
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        client = _s3_client()
        try:
            client.put_object(Bucket=bucket, Key=key, Body=body, IfMatch=version)
        except client.exceptions.ClientError as error:  # type: ignore[attr-defined]
            code = error.response.get("Error", {}).get("Code")
            if code in _CONDITIONAL_WRITE_LOST_CODES:
                return False
            raise
        return True
    local_path = Path(normalized)
    if not local_path.exists():
        return False
    if hashlib.sha256(local_path.read_bytes()).hexdigest() != version:
        return False
    local_path.write_bytes(body)
    return True


def write_json_artifact(path: ArtifactPath, payload: dict[str, object]) -> str:
    """Persist JSON to local storage or S3."""
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=body)
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(body)
    return str(local_path)


def write_text_artifact(path: ArtifactPath, body: str) -> str:
    """Persist text content to local storage or S3."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(body, encoding="utf-8")
    return str(local_path)


def write_gzip_text_artifact(path: ArtifactPath, body: str) -> str:
    """Write text gzip-compressed.

    The extract job state embeds full item text and message histories, so
    compression cuts the repeatedly rewritten object by roughly an order of
    magnitude (#86).
    """
    compressed = gzip.compress(body.encode("utf-8"))
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=compressed)
        return normalized
    target = Path(normalized)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(compressed)
    return normalized


def read_gzip_text_artifact(path: ArtifactPath) -> str:
    """Read a gzip-compressed text artifact."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    else:
        body = Path(normalized).read_bytes()
    return gzip.decompress(body).decode("utf-8")


def read_table(
    path: ArtifactPath, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a Parquet table or return an empty table when it does not exist."""
    normalized = normalize_artifact_path(path)
    if not artifact_exists(normalized):
        return pd.DataFrame(columns=columns)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    return pd.read_parquet(Path(normalized))


def read_dataset(
    base: ArtifactPath,
    *,
    columns: Sequence[str] | None = None,
    partition_filter: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read and concatenate all Parquet files under a dataset prefix."""
    frames = [
        read_table(path, columns)
        for path in iter_partition_paths(base, partition_filter=partition_filter)
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def write_table(path: ArtifactPath, table: pd.DataFrame) -> str:
    """Atomically write a Parquet table to local storage or S3."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        buffer = io.BytesIO()
        table.to_parquet(buffer, index=False)
        bucket, key = parse_s3_uri(normalized)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        return normalized

    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # The temp suffix must not be .parquet: a crash between create and rename
    # would leave a file every ``**/*.parquet`` reader picks up and dies on.
    with NamedTemporaryFile(
        dir=local_path.parent, suffix=".parquet.tmp", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        table.to_parquet(temp_path, index=False)
        temp_path.replace(local_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(local_path)


def delete_artifact(path: ArtifactPath) -> None:
    """Delete a local or S3-backed artifact, tolerating a missing target."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        _s3_client().delete_object(Bucket=bucket, Key=key)
        return
    Path(normalized).unlink(missing_ok=True)


def next_batch_path(directory: Path, prefix: str) -> Path:
    """Return the next sequential Parquet batch path in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.parquet$")
    highest = 0
    for path in directory.glob(f"{prefix}-*.parquet"):
        match = pattern.match(path.name)
        if match is None:
            continue
        highest = max(highest, int(match.group(1)))
    return directory / f"{prefix}-{highest + 1:06d}.parquet"


def write_parquet_batch(directory: Path, prefix: str, table: pd.DataFrame) -> Path:
    """Write a sequentially numbered Parquet batch and return its path."""
    path = next_batch_path(directory, prefix)
    write_table(path, table)
    return path


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


def append_new_rows(
    path: Path,
    rows: pd.DataFrame,
    key_columns: Sequence[str],
    columns: Sequence[str],
    *,
    replace_keys: Iterable[object] | None = None,
    replace_key_column: str | None = None,
) -> pd.DataFrame:
    """Append rows to a keyed Parquet table and return the full updated table."""
    existing = read_table(path, columns)
    existing = existing.reindex(columns=columns)
    rows = rows.reindex(columns=columns)
    if existing.empty and rows.empty:
        write_table(path, existing)
        return existing

    if (
        replace_keys is not None
        and replace_key_column is not None
        and not existing.empty
    ):
        existing = existing.loc[~existing[replace_key_column].isin(set(replace_keys))]

    if rows.empty:
        updated = existing
    elif existing.empty:
        updated = rows
    else:
        combined = pd.concat([existing, rows], ignore_index=True)
        updated = combined.drop_duplicates(subset=list(key_columns), keep="last")

    write_table(path, updated)
    return updated


def missing_keys(
    existing: pd.DataFrame, candidates: Iterable[object], key_column: str
) -> set[object]:
    """Return candidate keys that are absent from an existing table."""
    candidate_set = set(candidates)
    if existing.empty or key_column not in existing:
        return candidate_set
    return candidate_set.difference(set(existing[key_column]))
