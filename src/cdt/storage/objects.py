"""Artifacts on S3 or local disk: clients per AWS profile, object reads and writes, listings and digests."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import cast
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from botocore.exceptions import IncompleteReadError, ReadTimeoutError

from cdt.shared import get_logger

LOGGER = get_logger(__name__)


ArtifactPath = str | Path


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


def delete_artifact(path: ArtifactPath) -> None:
    """Delete a local or S3-backed artifact, tolerating a missing target."""
    normalized = normalize_artifact_path(path)
    if is_s3_uri(normalized):
        bucket, key = parse_s3_uri(normalized)
        s3_client().delete_object(Bucket=bucket, Key=key)
        return
    Path(normalized).unlink(missing_ok=True)
