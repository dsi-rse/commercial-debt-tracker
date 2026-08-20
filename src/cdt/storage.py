"""Helpers for reading and updating pipeline artifacts."""

from __future__ import annotations

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
        client = boto3.client("s3")
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
        paginator = boto3.client("s3").get_paginator("list_objects_v2")
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


def iter_partition_paths(
    base: ArtifactPath,
    *,
    partition_filter: dict[str, str] | None = None,
) -> Iterator[str]:
    """Yield partition file paths under a dataset, optionally filtered by key/value."""
    for path in list_artifacts(base, suffix=".parquet"):
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
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
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
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8")
    return Path(normalized).read_text(encoding="utf-8")


def write_json_artifact(path: ArtifactPath, payload: dict[str, object]) -> str:
    """Persist JSON to local storage or S3."""
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)
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
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(body, encoding="utf-8")
    return str(local_path)


MISSING_TEXT_VALUES = frozenset({"nan", "none", "null", "<na>", "n/a"})


def coerce_dataset_text(value: object) -> str | None:
    """Return one trimmed dataset text value, or None when it carries no name.

    Parquet round-trips a missing value as NaN, and ``str(float("nan"))`` is the
    literal text ``nan``, so placeholder strings are treated as missing too.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in MISSING_TEXT_VALUES:
        return None
    return text or None


def read_table(
    path: ArtifactPath, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a Parquet table or return an empty table when it does not exist."""
    normalized = normalize_artifact_path(path)
    if not artifact_exists(normalized):
        return pd.DataFrame(columns=columns)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
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
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        return normalized

    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=local_path.parent, suffix=".parquet", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        table.to_parquet(temp_path, index=False)
        temp_path.replace(local_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(local_path)


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
