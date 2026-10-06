"""Helpers for reading and updating pipeline artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import cast
from urllib.parse import urlparse

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet
from botocore.config import Config
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from botocore.exceptions import IncompleteReadError, ReadTimeoutError

from cdt.shared import get_logger

LOGGER = get_logger(__name__)

ArtifactPath = str | Path

# Empty ``tmp*.parquet`` files left in a partition directory by a crash under
# write_table's former temp naming; ``**/*.parquet`` readers must skip them.
# Current writes use ``.parquet.tmp``, which this does not match.
_ORPHANED_TEMP_RE = re.compile(r"(?:^|/)tmp[^/]*\.parquet$")

# One client and one Session per AWS profile for the whole process: construction
# resolves credentials and endpoints, and a run issues thousands of S3 calls.
_S3_CLIENTS: dict[str, object] = {}
_BOTO3_SESSIONS: dict[str, boto3.Session] = {}
# The profile ``--aws-profile`` selected; empty means the ambient credential chain.
_CONFIGURED_S3_PROFILE = ""

# Bounds the API call itself, not the streaming read of a returned body;
# ``_get_object_with_body`` retries that half.
S3_CLIENT_CONFIG = Config(
    retries={"mode": "standard", "max_attempts": 5},
    connect_timeout=10,
    read_timeout=60,
)


def configure_s3_profile(profile_name: str | None) -> None:
    """Select the AWS profile every unqualified S3 client in this process uses.

    Call once per entry point with the parsed ``--aws-profile``. ``None`` and
    ``""`` both mean the ambient credential chain (the CLI default).
    """
    global _CONFIGURED_S3_PROFILE  # noqa: PLW0603
    _CONFIGURED_S3_PROFILE = profile_name or ""


def configured_s3_profile() -> str:
    """Return the AWS profile unqualified S3 clients resolve to."""
    return _CONFIGURED_S3_PROFILE


def s3_client(profile_name: str | None = None):  # noqa: ANN201
    """Return the memoized S3 client for a profile.

    ``None`` means the profile ``configure_s3_profile`` selected; an explicit
    name (``""`` for the ambient chain) wins. One client per resolved profile.
    """
    resolved = _CONFIGURED_S3_PROFILE if profile_name is None else profile_name
    client = _S3_CLIENTS.get(resolved)
    if client is None:
        client = boto3_session(resolved).client("s3", config=S3_CLIENT_CONFIG)
        _S3_CLIENTS[resolved] = client
    return client


def boto3_session(profile_name: str | None = None) -> boto3.Session:
    """Return the memoized boto3 Session for a profile.

    ``None`` means the configured profile, as in ``s3_client``. Both
    ``s3_client`` and ``arrow_filesystem`` build from this Session, so one
    profile resolution serves both credentialed objects.
    """
    resolved = _CONFIGURED_S3_PROFILE if profile_name is None else profile_name
    session = _BOTO3_SESSIONS.get(resolved)
    if session is None:
        session = boto3.Session(profile_name=resolved) if resolved else boto3.Session()
        _BOTO3_SESSIONS[resolved] = session
    return session


# Failures a streaming body read raises after get_object returned, outside
# botocore's retries. ClientError (NoSuchKey, AccessDenied) is deliberately
# absent: those are permanent.
_STREAMING_READ_ERRORS = (
    ReadTimeoutError,
    BotocoreConnectionError,
    IncompleteReadError,
)
_GET_OBJECT_ATTEMPTS = 5
_GET_OBJECT_BACKOFF_SECONDS = 1.0


def _get_object_with_body(
    s3_client: object, bucket: str, key: str
) -> tuple[bytes, dict[str, object]]:
    """GET an object and read the whole body, retrying streaming failures.

    Returns ``(body, response)``. A mid-stream failure cannot be resumed, so
    each retry re-issues get_object and reads from the start; the last failure
    is re-raised after ``_GET_OBJECT_ATTEMPTS``.
    """
    for attempt in range(1, _GET_OBJECT_ATTEMPTS + 1):
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read(), response
        except _STREAMING_READ_ERRORS as error:
            if attempt == _GET_OBJECT_ATTEMPTS:
                raise
            LOGGER.warning(
                "S3 object read failed (attempt %s/%s) for s3://%s/%s: %s",
                attempt,
                _GET_OBJECT_ATTEMPTS,
                bucket,
                key,
                error,
            )
            sleep(_GET_OBJECT_BACKOFF_SECONDS * attempt)
    msg = "unreachable: loop returns or raises"
    raise AssertionError(msg)


def get_object_bytes(s3_client: object, bucket: str, key: str) -> bytes:
    """GET an S3 object body with bounded retries around the streaming read."""
    body, _ = _get_object_with_body(s3_client, bucket, key)
    return body


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
        client = s3_client()
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
        paginator = s3_client().get_paginator("list_objects_v2")
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
    """Return whether a path is a ``tmp*.parquet`` file of write_table's old naming."""
    return _ORPHANED_TEMP_RE.search(normalize_artifact_path(path)) is not None


def list_artifacts_with_versions(
    base: ArtifactPath, *, suffix: str = ""
) -> dict[str, str]:
    """Map artifact path -> opaque source version, from one LIST.

    The version is the S3 ETag, or ``<size>-<mtime_ns>`` locally, so it changes
    whenever the object is rewritten; completion registries use it to detect a
    source partition ingest merged rows into. Costs no request beyond the LIST.
    A missing local root returns ``{}``.
    """
    normalized = normalize_artifact_path(base).rstrip("/")
    if is_s3_uri(normalized):
        bucket, prefix = parse_s3_uri(normalized)
        paginator = s3_client().get_paginator("list_objects_v2")
        results: dict[str, str] = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in cast(list[dict[str, str]], page.get("Contents", [])):
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                results[f"s3://{bucket}/{key}"] = str(obj["ETag"])
        return results
    root = Path(normalized)
    if not root.exists():
        return {}
    pattern = f"**/*{suffix}" if suffix else "**/*"
    results = {}
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        stat = path.stat()
        results[str(path)] = f"{stat.st_size}-{stat.st_mtime_ns}"
    return results


def artifact_content_versions(
    base: ArtifactPath, *, suffix: str = ""
) -> dict[str, str]:
    """Map artifact path -> a version that changes only when the bytes change.

    On S3 the version is the ETag (the content MD5 of ``write_table``'s
    single-part put), costing nothing beyond the LIST; locally it is the
    SHA-256 of the bytes. Unlike ``list_artifacts_with_versions``, a
    byte-identical rewrite keeps its version. A missing local root returns
    ``{}``.
    """
    normalized = normalize_artifact_path(base).rstrip("/")
    if is_s3_uri(normalized):
        return list_artifacts_with_versions(normalized, suffix=suffix)
    root = Path(normalized)
    if not root.exists():
        return {}
    pattern = f"**/*{suffix}" if suffix else "**/*"
    versions: dict[str, str] = {}
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        versions[str(path)] = digest.hexdigest()
    return versions


def artifact_tree_digest(
    roots: Iterable[ArtifactPath],
    *,
    suffix: str = "",
    context: Mapping[str, object] | None = None,
) -> str:
    """Digest the content versions of every artifact under a set of roots.

    Returns a SHA-256 hex digest that changes when any artifact's bytes change,
    independent of listing order. ``context`` is any other JSON-serialisable
    input the caller's answer depends on (a code or schema version, say); it is
    folded into the same digest, so a change to it changes the digest too.
    """
    versions: dict[str, str] = {}
    for root in roots:
        versions.update(artifact_content_versions(root, suffix=suffix))
    payload = json.dumps(
        {"context": context or {}, "versions": sorted(versions.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        body = get_object_bytes(s3_client(), bucket, key)
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
        body = get_object_bytes(s3_client(), bucket, key)
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
        client = s3_client()
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

    Returns ``(payload, version)``. A body that does not parse returns
    ``(None, version)`` rather than raising, so a truncated lock file reads as
    corrupt and stealable by compare-and-swap instead of wedging later runs.
    """
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        body, response = _get_object_with_body(s3_client(), bucket, key)
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
        client = s3_client()
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
    """Persist JSON atomically to local storage or S3.

    Readers see the whole old or the whole new body, never a truncated one:
    S3 PUTs are atomic, and locally a temp file is renamed into place.
    Returns the written path.
    """
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        s3_client().put_object(Bucket=bucket, Key=key, Body=body)
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=local_path.parent, suffix=".tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        temp_path.write_bytes(body)
        temp_path.replace(local_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(local_path)


def write_text_artifact(path: ArtifactPath, body: str) -> str:
    """Persist text content to local storage or S3."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        s3_client().put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(body, encoding="utf-8")
    return str(local_path)


def write_bytes_artifact(path: ArtifactPath, body: bytes) -> str:
    """Persist raw bytes, unchanged, to local storage or S3; return the path.

    For byte-for-byte copies such as mirrored SEC submissions, which
    ``ingest.decode_document_bytes`` decodes on read.
    """
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        s3_client().put_object(Bucket=bucket, Key=key, Body=body)
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(body)
    return str(local_path)


MISSING_TEXT_VALUES = frozenset({"nan", "none", "null", "<na>", "n/a"})

# Every published column's physical type, declared rather than inferred from
# values, so a column has the same type in every partition. Keyed by name because
# a column name means one thing across datasets. Money and rates are exact
# decimals, never float. Rationale: docs/decisions/storage-and-completion.md.
DECLARED_COLUMN_TYPES: dict[str, pa.DataType] = {
    "principal_amount": pa.decimal128(38, 2),
    "outstanding_balance": pa.decimal128(38, 2),
    # Four places carries basis points with room to spare; the corpus uses at
    # most three.
    "interest_rate_pct": pa.decimal128(9, 4),
    "mention_count": pa.int64(),
    "document_count": pa.int64(),
    "candidate_rank": pa.int64(),
    "start_line": pa.int64(),
    "end_line": pa.int64(),
    "section_char_count": pa.int64(),
    "is_lineage_head": pa.bool_(),
    "relevance": pa.bool_(),
    "synthesized_only": pa.bool_(),
    "outstanding_balance_as_of_is_filing_date": pa.bool_(),
    # Model scores, not measured quantities, so a float is the honest type.
    "classification_score": pa.float64(),
    "match_score": pa.float64(),
}
# Everything not declared above is nullable text, which is the contract
# `docs/schema.md` states.
DEFAULT_COLUMN_TYPE = pa.string()


def canonical_numeric_text(value: Decimal) -> str:
    """Return one deterministic numeric string for a decimal value.

    Trailing zeros are dropped (``Decimal("5.0000")`` -> ``"5"``) so a value has
    one spelling: decimal columns read back scale-padded, and the pipeline
    compares amounts as text.
    """
    quantized = value.normalize()
    if quantized == quantized.to_integral_value():
        quantized = quantized.to_integral_value()
    return f"{quantized:f}"


def decimal_column_values(
    values: Iterable[object], dtype: pa.Decimal128Type, *, column: str
) -> list[Decimal | None]:
    """Return one column's values as decimals quantized half-up to ``dtype``'s scale.

    Digits beyond the scale are rounded away rather than rejected, and
    placeholder text (``nan``, empty) becomes None. Raises ValueError for text
    that is not a number: these columns are always written from a parsed
    amount, so that is an upstream bug.
    """
    exponent = Decimal(1).scaleb(-dtype.scale)
    coerced: list[Decimal | None] = []
    for value in values:
        if isinstance(value, Decimal):
            coerced.append(value.quantize(exponent, rounding=ROUND_HALF_UP))
            continue
        text = coerce_dataset_text(value)
        if text is None:
            coerced.append(None)
            continue
        try:
            coerced.append(Decimal(text).quantize(exponent, rounding=ROUND_HALF_UP))
        except InvalidOperation as error:
            message = f"{column} is not a number: {text!r}"
            raise ValueError(message) from error
    return coerced


def declared_column_type(name: str, inferred: pa.DataType) -> pa.DataType:
    """Return the physical type one column publishes as.

    A declared type wins. Otherwise an inferred ``null`` (an object column with
    no value in this frame) becomes ``DEFAULT_COLUMN_TYPE``, and any other
    inferred type, which came from a real pandas dtype, is kept.
    """
    declared = DECLARED_COLUMN_TYPES.get(name)
    if declared is not None:
        return declared
    if pa.types.is_null(inferred):
        return DEFAULT_COLUMN_TYPE
    return inferred


def apply_declared_column_types(table: pd.DataFrame) -> pa.Table:
    """Return ``table`` as an Arrow table with the declared physical types applied.

    Declared decimal columns are canonicalised to text first: rewriting an
    existing partition mixes parquet-read ``Decimal`` values with in-memory
    text in one object column, which ``Table.from_pandas`` rejects.
    """
    prepared = table.copy()
    for name, dtype in DECLARED_COLUMN_TYPES.items():
        if name in prepared.columns and pa.types.is_decimal(dtype):
            prepared[name] = pd.Series(
                [coerce_dataset_text(value) for value in prepared[name]],
                index=prepared.index,
                dtype=object,
            )
    arrow = pa.Table.from_pandas(prepared, preserve_index=False)
    for index, field in enumerate(arrow.schema):
        dtype = declared_column_type(field.name, field.type)
        if dtype == field.type:
            continue
        if pa.types.is_decimal(dtype):
            values = decimal_column_values(
                arrow.column(field.name).to_pylist(), dtype, column=field.name
            )
            column = pa.array(values, type=dtype)
        else:
            column = arrow.column(field.name).cast(dtype)
        arrow = arrow.set_column(index, pa.field(field.name, dtype), column)
    return arrow


def coerce_dataset_text(value: object) -> str | None:
    """Return one trimmed dataset text value, or None when it carries no name.

    Parquet round-trips a missing value as NaN, and ``str(float("nan"))`` is the
    literal text ``nan``, so placeholder strings are treated as missing too.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        # Not ``str()``: that gives the scale-padded ``2000000000.00`` where the
        # pipeline wrote ``2000000000``, and textual comparisons would miss.
        return canonical_numeric_text(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in MISSING_TEXT_VALUES:
        return None
    return text or None


def json_column(row: Mapping[str, object], column: str) -> object | None:
    """Parse one JSON text column; None when it is absent, missing or not JSON."""
    text = coerce_dataset_text(row.get(column))
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def write_gzip_text_artifact(path: ArtifactPath, body: str) -> str:
    """Write text gzip-compressed to local storage or S3; return the path."""
    compressed = gzip.compress(body.encode("utf-8"))
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        s3_client().put_object(Bucket=bucket, Key=key, Body=compressed)
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
        body = get_object_bytes(s3_client(), bucket, key)
    else:
        body = Path(normalized).read_bytes()
    return gzip.decompress(body).decode("utf-8")


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
            max_attempts=S3_CLIENT_CONFIG.retries["max_attempts"]
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


def delete_artifact(path: ArtifactPath) -> None:
    """Delete a local or S3-backed artifact, tolerating a missing target."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        s3_client().delete_object(Bucket=bucket, Key=key)
        return
    Path(normalized).unlink(missing_ok=True)


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
