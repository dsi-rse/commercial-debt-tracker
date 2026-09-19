"""Helpers for reading and updating pipeline artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Iterable, Iterator, Sequence
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

# Basenames NamedTemporaryFile produced before write_table stopped giving temp
# files a .parquet suffix. A crash between create and rename orphaned them
# inside partition directories, where every ``**/*.parquet`` reader choked on
# the empty file (#68).
_ORPHANED_TEMP_RE = re.compile(r"(?:^|/)tmp[^/]*\.parquet$")

# One client per AWS profile for every S3 call in the process: construction is
# expensive (credential resolution, endpoint discovery), and the partition scans
# issue thousands of calls per run (#83).
#
# Keyed by profile because there used to be two unreconciled factories. This
# module's was a singleton built with no profile at all, so `--aws-profile`
# reached ingest's own client and nothing else — every read_table, every
# list_objects_v2 behind a partition scan, and every artifact write resolved
# credentials from the ambient environment instead (#71). The other,
# `ingest.default_s3_client`, took a profile but was uncached, so it built a
# fresh Session per call and callers had to memoize it by hand
# (`itemizer.core.ensure_s3_client`) — and its own default was the empty
# profile, which is why the itemize and extract stages dropped the flag too.
# `ingest.default_s3_client` now delegates here, leaving one cache and one
# place that knows how a profile becomes a client.
_S3_CLIENTS: dict[str, object] = {}
_BOTO3_SESSIONS: dict[str, boto3.Session] = {}
# The profile `--aws-profile` selected for this process. Empty means the
# ambient credential chain, which is both the CLI default and what every
# caller got before.
_CONFIGURED_S3_PROFILE = ""

# Explicit API-call retries and timeouts: nothing configured them before, so
# every client ran botocore's legacy retry mode (2 attempts) with no bound on
# a stalled socket. This covers the get_object call itself, NOT the streaming
# read of a returned body -- _get_object_with_body handles that half (#112).
S3_CLIENT_CONFIG = Config(
    retries={"mode": "standard", "max_attempts": 5},
    connect_timeout=10,
    read_timeout=60,
)


def configure_s3_profile(profile_name: str | None) -> None:
    """Select the AWS profile every unqualified S3 client in this process uses.

    Called once per entry point, from the parsed ``--aws-profile``. Before this
    existed the flag was threaded by hand to the one factory ingest happened to
    call, so a run against a non-default account read its manifests through the
    right credentials and then wrote every artifact through the wrong ones —
    or, more usually, failed on the first write with no hint that the flag had
    not been honored (#71).

    ``None`` and ``""`` both mean the ambient credential chain, which is the
    CLI default and the historical behaviour.
    """
    global _CONFIGURED_S3_PROFILE  # noqa: PLW0603
    _CONFIGURED_S3_PROFILE = profile_name or ""


def configured_s3_profile() -> str:
    """Return the AWS profile unqualified S3 clients resolve to."""
    return _CONFIGURED_S3_PROFILE


def s3_client(profile_name: str | None = None):  # noqa: ANN201
    """Return the memoized S3 client for a profile, or the configured one.

    ``profile_name=None`` means "whatever ``--aws-profile`` selected" — the
    case that was broken. An explicit name still wins, so a caller holding a
    config with its own profile (ingest, the 6-K scraper) keeps passing it and
    gets the same object the generic helpers in this module use.
    """
    resolved = _CONFIGURED_S3_PROFILE if profile_name is None else profile_name
    client = _S3_CLIENTS.get(resolved)
    if client is None:
        client = boto3_session(resolved).client("s3", config=S3_CLIENT_CONFIG)
        _S3_CLIENTS[resolved] = client
    return client


def boto3_session(profile_name: str | None = None) -> boto3.Session:
    """Return the memoized boto3 Session for a profile, or the configured one.

    Split out of ``s3_client`` because the Arrow read path needs the *same*
    profile resolved for a second credentialed object, ``pyarrow.fs.S3FileSystem``
    (#190). Keeping one Session per profile means the profile is still decided
    in exactly one place even though two client objects come out of it.
    """
    resolved = _CONFIGURED_S3_PROFILE if profile_name is None else profile_name
    session = _BOTO3_SESSIONS.get(resolved)
    if session is None:
        session = boto3.Session(profile_name=resolved) if resolved else boto3.Session()
        _BOTO3_SESSIONS[resolved] = session
    return session


def _s3_client():  # noqa: ANN202
    return s3_client()


# Failures a streaming body read surfaces after get_object has returned, where
# botocore's retry logic no longer applies (#112). ClientError (NoSuchKey,
# AccessDenied) is deliberately absent: those are permanent.
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

    A mid-stream timeout cannot be resumed, so each retry re-issues get_object
    and reads from the start; one such timeout previously killed a 2.5h itemize
    outright (#112).
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


def list_artifacts_with_versions(
    base: ArtifactPath, *, suffix: str = ""
) -> dict[str, str]:
    """Map artifact path -> opaque source version, from one LIST.

    The version is the S3 ETag (size+mtime locally): it changes whenever the
    object is rewritten, which is how completion registries detect a source
    partition that ingest merged new rows into (#62). Captured during the same
    pagination the plain listing uses, so it costs no extra requests.
    """
    normalized = normalize_artifact_path(base).rstrip("/")
    if is_s3_uri(normalized):
        bucket, prefix = parse_s3_uri(normalized)
        paginator = _s3_client().get_paginator("list_objects_v2")
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
        body = get_object_bytes(_s3_client(), bucket, key)
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
        body = get_object_bytes(_s3_client(), bucket, key)
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
        body, response = _get_object_with_body(_s3_client(), bucket, key)
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
    """Persist JSON atomically to local storage or S3.

    S3 PUTs are atomic; the local branch writes a temp file and renames so a
    crash mid-write can never leave a truncated registry or pointer — readers
    see whole-old or whole-new, matching write_table's contract.
    """
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=body)
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
        _s3_client().put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(body, encoding="utf-8")
    return str(local_path)


def write_bytes_artifact(path: ArtifactPath, body: bytes) -> str:
    """Persist raw bytes to local storage or S3.

    Used where a byte-for-byte copy is the point: a mirrored SEC submission is
    read back through ``ingest.decode_document_bytes``, so re-encoding it here
    would change the text a stage extracts from.
    """
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=body)
        return normalized
    local_path = Path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(body)
    return str(local_path)


MISSING_TEXT_VALUES = frozenset({"nan", "none", "null", "<na>", "n/a"})

# Every published column's physical type, declared rather than inferred (#187).
#
# `pa.Table.from_pandas` infers an object column's type from its values, so a
# column with no value in one partition serialised as parquet `null` and as
# `string` in the next. That made the type a function of the data rather than of
# the schema, and 23 of 42 `debt-instruments` columns did it — enough that
# `pyarrow.dataset`, `pq.read_table`, `ParquetDataset` and `pandas.read_parquet`
# all failed on the directory with "Unsupported cast from string to null". Only
# this module's own `read_dataset` worked, because it concatenates per file in
# pandas instead of asking Arrow to unify the schemas, which is why it went
# unnoticed. An empty frame is worse still: it infers `null` for every column,
# including the counts and flags.
#
# Column names carry one meaning across datasets here (`cik` is the same thing
# everywhere), so keying by name keeps `write_table` generic.
#
# Money and rates are exact decimals (#185). Float is not an option:
# `float("372246148.11")` is not that number, and rendering it at fixed
# precision leaks the difference, which is what made every amount carrying cents
# publish as null (#119).
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

    Trailing zeros are dropped so a value has exactly one spelling: a decimal
    column round-trips scale-padded (`Decimal("5.0000")` for a rate stored at
    scale four), and every internal comparison in the pipeline is textual.
    """
    quantized = value.normalize()
    if quantized == quantized.to_integral_value():
        quantized = quantized.to_integral_value()
    return f"{quantized:f}"


def decimal_column_values(
    values: Iterable[object], dtype: pa.Decimal128Type, *, column: str
) -> list[Decimal | None]:
    """Return one column's values as decimals at the declared scale.

    Quantizes rather than casting straight through Arrow, for two reasons. A
    partition written before #119 carries float error in its text
    (`372246148.110000014305`), and Arrow refuses that as a rescale that would
    lose data — but the extra digits are the error, not the value, so dropping
    them is correct and a replay must not fail on them. And placeholder text
    (`nan`, an empty string) has to become null rather than an Arrow error.

    Text that is not a number at all raises: every stage writes these columns
    from a parsed, verified amount, so anything else is a bug upstream rather
    than drift worth tolerating.
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

    A declared type always wins. Otherwise an inferred `null` — an object column
    with no value in this frame — becomes text, so the column publishes the same
    type whether or not this particular partition happened to carry a value.
    Any other inferred type is left alone: it came from a real pandas dtype and
    is already stable across partitions.
    """
    declared = DECLARED_COLUMN_TYPES.get(name)
    if declared is not None:
        return declared
    if pa.types.is_null(inferred):
        return DEFAULT_COLUMN_TYPE
    return inferred


def apply_declared_column_types(table: pd.DataFrame) -> pa.Table:
    """Return one Arrow table with the declared physical types applied.

    Declared decimal columns are canonicalised to text before Arrow sees them.
    A frame that mixes rows read back from parquet (which carry `Decimal`) with
    rows built in memory (which carry the parser's text) is one object column
    of two Python types, and `Table.from_pandas` refuses it outright — before
    the decimal path below could quantize either. Every rewrite of an existing
    partition is that mix (#203).
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
        # A decimal column round-trips scale-padded, so `str()` would hand the
        # pipeline `2000000000.00` where it wrote `2000000000` and every textual
        # comparison — match keys, prior-amount equality — would miss.
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
        body = get_object_bytes(_s3_client(), bucket, key)
    else:
        body = Path(normalized).read_bytes()
    return gzip.decompress(body).decode("utf-8")


# The Arrow read path (#190). Every read used to be "GET the whole object, wrap
# it in a BytesIO, then hand pandas a `columns=` list" — so `columns` saved
# deserialization and not one byte of transfer. Measured on
# data/genwindow-eval-apr: the `documents` dataset is 12,206.6 MB over 1,640
# partitions at 1.345 MB/row, and its `accession_number` column is 0.2988 MB of
# compressed column chunks. The projection a dedup scan wants is 0.0024% of the
# bytes the old path moved (40,856x). Reading through a pyarrow filesystem makes
# the projection a set of ranged GETs instead.
#
# `pa.ArrowException` is the common base of ArrowInvalid (also a ValueError),
# ArrowTypeError (also a TypeError) and ArrowNotImplementedError, which are the
# three ways a directory of partitions with per-partition physical types breaks
# a dataset scan. Catching the base is deliberate: any of them means "Arrow
# cannot read these files as one table", and the answer is always the same.
_ARROW_READ_ERRORS = (pyarrow.ArrowException,)


def arrow_filesystem(path: ArtifactPath) -> tuple[object | None, str]:
    """Return the pyarrow filesystem and stripped path Arrow should read through.

    Local paths get ``None``: pyarrow resolves those itself, and threading a
    LocalFileSystem through would only add a way to get it wrong. S3 gets a
    ``pyarrow.fs.S3FileSystem`` — a second credentialed object, which is why #71
    had to be settled first. It is built from the boto3 Session for the
    configured profile, so ``--aws-profile`` decides both clients from one
    place, and it is built per call rather than memoized because a Session's
    frozen credentials can be temporary (SSO, assume-role) and a historical
    backfill outlives them. Construction issues no request.
    """
    normalized = normalize_artifact_path(path)
    if not is_s3_uri(normalized):
        return None, normalized
    bucket, key = parse_s3_uri(normalized)
    session = boto3_session()
    credentials = session.get_credentials()
    kwargs: dict[str, object] = {}
    if credentials is not None:
        frozen = credentials.get_frozen_credentials()
        kwargs = {
            "access_key": frozen.access_key,
            "secret_key": frozen.secret_key,
            "session_token": frozen.token,
        }
    if session.region_name:
        # Without a region pyarrow issues its own bucket-location lookup per
        # filesystem; the Session already knows the answer.
        kwargs["region"] = session.region_name
    return pyarrow.fs.S3FileSystem(**kwargs), f"{bucket}/{key.lstrip('/')}"


def _open_parquet_file(path: ArtifactPath) -> pyarrow.parquet.ParquetFile:
    """Open a parquet footer without downloading the object's column data."""
    filesystem, resolved = arrow_filesystem(path)
    if filesystem is None:
        return pyarrow.parquet.ParquetFile(Path(resolved))
    return pyarrow.parquet.ParquetFile(resolved, filesystem=filesystem)


def read_table(
    path: ArtifactPath, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a Parquet table (projected to ``columns``) or an empty table if absent.

    ``columns`` used to shape only the empty fallback while both real branches
    deserialized every column (#69) — and then, once it did project, the S3
    branch still GET the whole object first, so it never saved transfer (#190).
    Reading through ``_open_parquet_file`` makes the projection ranged reads of
    just the wanted column chunks. A partition written before a column existed
    still gets the column back, as null, rather than raising (#69).

    The absent-column case is decided from the footer schema rather than by
    catching a read error, because ``ParquetFile.read`` does not raise on one:
    it silently returns the columns it *does* have. Relying on an exception
    quietly dropped the requested-but-absent column from the result, which is
    the one thing this behaviour exists to prevent.
    """
    normalized = normalize_artifact_path(path)
    if not artifact_exists(normalized):
        return pd.DataFrame(columns=columns)
    parquet_file = _open_parquet_file(normalized)
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
    """Return a parquet table's row count from footer metadata; None if absent.

    The footer alone carries num_rows. The S3 branch used to GET the entire
    object and then read only its last few KB, which on `documents` meant
    moving 1.345 MB per row to learn a single integer (#190); reading the
    footer through a pyarrow filesystem makes it two small ranged GETs. This is
    the count the publish guard compares against, so it runs once per final
    table on every publish.
    """
    normalized = normalize_artifact_path(path)
    if not artifact_exists(normalized):
        return None
    return int(_open_parquet_file(normalized).metadata.num_rows)


def _read_dataset_with_arrow(
    paths: list[str], columns: Sequence[str] | None
) -> pd.DataFrame | None:
    """Read many partitions as one Arrow dataset, or None if they cannot unify.

    Worth a separate path from looping ``read_table`` because the scan is
    parallel: measured on data/genwindow-eval-apr's 1,554-file ``items``
    dataset, a full read is 4.836s looping pandas, 2.348s looping
    ``pq.read_table`` and concatenating, and 0.599s as one dataset — the win is
    the scanner's threads, not the Arrow decode.

    The schema is unified explicitly, and that is not an optimization detail.
    ``ds.dataset`` infers its schema from the *first* fragment alone, so a
    dataset whose first file predates a column reads as though the column never
    existed — the later values are silently dropped rather than reported. This
    pipeline has exactly that shape on purpose (`form_type` and `source` were
    added to `documents` after the fact). Unifying costs 0.166s over those
    1,554 footers.

    Returning None means "these files are not one table". That happens on any
    root written before #187 gave every column a declared physical type: 26 of
    the 41 columns under data/lineage-probe/debt-instruments still have a type
    that varies by partition, and unification fails with
    ``ArrowTypeError: Unable to merge: Field amendment_inferred_by has
    incompatible types: double vs string``. The production dev root has not been
    rebuilt (#107), so this is the live case, not a hypothetical.
    """
    filesystem, _ = arrow_filesystem(paths[0])
    resolved = [arrow_filesystem(path)[1] for path in paths]
    try:
        dataset = pyarrow.dataset.dataset(
            resolved, format="parquet", filesystem=filesystem
        )
        schema = pyarrow.unify_schemas(
            [fragment.physical_schema for fragment in dataset.get_fragments()]
        )
        unified = pyarrow.dataset.dataset(
            resolved, format="parquet", filesystem=filesystem, schema=schema
        )
        if columns is None:
            return unified.to_table().to_pandas()
        # Project only the columns this root actually has; a name absent
        # everywhere is reindexed in below, matching the pandas path.
        present = [name for name in columns if name in schema.names]
        table = unified.to_table(columns=present).to_pandas()
        return table.reindex(columns=list(columns))
    except _ARROW_READ_ERRORS:
        return None


def read_dataset(
    base: ArtifactPath,
    *,
    columns: Sequence[str] | None = None,
    partition_filter: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read and concatenate all Parquet files under a dataset prefix.

    Tries one parallel, projection-pushed-down Arrow scan and falls back to the
    original per-file pandas concatenation when the partitions cannot be read as
    one table. The fallback is kept rather than retired because partitions
    written before #187 keep their old per-partition physical types and the
    production root has not been rebuilt (#107) — making the Arrow path
    mandatory would put an ordering dependency on that rebuild. It is also a
    permanent safety net: #187 pins *declared* columns, so an undeclared object
    column can still infer a different physical type in different partitions.
    """
    paths = list(iter_partition_paths(base, partition_filter=partition_filter))
    if not paths:
        return pd.DataFrame(columns=columns)
    table = _read_dataset_with_arrow(paths, columns)
    if table is not None:
        return table
    LOGGER.info(
        "Arrow could not read %s partitions under %s as one table "
        "(pre-#187 per-partition types); falling back to a per-file read.",
        len(paths),
        normalize_artifact_path(base),
    )
    frames = [read_table(path, columns) for path in paths]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def write_table(path: ArtifactPath, table: pd.DataFrame) -> str:
    """Atomically write a Parquet table to local storage or S3."""
    normalized = normalize_artifact_path(path)
    arrow_table = apply_declared_column_types(table)
    if is_s3_uri(normalized):
        buffer = io.BytesIO()
        pyarrow.parquet.write_table(arrow_table, buffer)
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
