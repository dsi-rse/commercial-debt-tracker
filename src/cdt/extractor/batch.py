"""OpenAI Batch API backend and poll-driven state machine for extraction.

The synchronous extractor (``cdt.extractor.core``) drives the same stage objects
through live OpenRouter chat completions. This module instead advances those
stages asynchronously through OpenAI's Batch API, which is ~50% cheaper but can
take up to 24h per round. Because the workflow is several sequential, retryable
stages, a single item can take many rounds, so the state machine is fully
resumable and file-native: an hourly ``poll`` tick loads job state from the
artifact root, folds any completed OpenAI batch results into row states, submits
the next batch for rows that still need a call, and persists everything back.

Layout under ``{artifact_root}/extract-batches/``::

    active.json                 # {"job_id": ...}; absent when idle
    job_id=<id>/manifest.json   # static job config + claimed partitions
    job_id=<id>/state.jsonl     # one line per item: source partition + pending
                                # request marker + row state
    job_id=<id>/batches.json    # in-flight batches, seen batch ids, tick counter
    job_id=<id>/ticks/tick=<n>.json  # per-tick audit counts

Only the ``poll`` mode ever mutates this state, and the orchestrator runs each
poll tick under the single ``pipeline-writer`` lease (``cdt.lease``), so
overlapping ticks and the ``daily`` schedule's match/finalize cannot race it.
"""

# ruff: noqa: ANN101, ANN102, D102, D105, D107

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from cdt import settings
from cdt.datasets import dataset_root, resolve_artifact_root
from cdt.extractor.core import (
    DEBT_INSTRUMENT_MENTION_COLUMNS,
    DEFAULT_MAX_ATTEMPTS,
    ExtractionRowState,
    collect_pending_extract_items,
    extract_batch_response_text,
    finalize_extract_outputs,
    handle_response,
    initial_messages,
    native_model_id,
    record_stage_error,
    sampling_params,
)
from cdt.shared import get_logger
from cdt.storage import (
    ArtifactPath,
    artifact_exists,
    join_artifact_path,
    read_json_artifact,
    read_text_artifact,
    write_json_artifact,
    write_text_artifact,
)

LOGGER = get_logger(__name__)

EXTRACT_BATCHES_DATASET = "extract-batches"
ACTIVE_JOB_FILENAME = "active.json"
BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_COMPLETION_WINDOW = "24h"
MAX_CUSTOM_ID_LENGTH = 64
# Stay comfortably under OpenAI's 50k-requests-per-batch limit.
DEFAULT_MAX_REQUESTS_PER_BATCH = 40_000
# OpenAI also caps batch input files by size (200 MB); chunk at half that so
# a batch never wedges the job on the byte limit.
DEFAULT_MAX_BATCH_BYTES = 100 * 1024 * 1024
# A row whose batch expires without a result re-submits next tick; each round
# costs another 24h window, so cap the rounds instead of looping forever.
DEFAULT_MAX_RESUBMISSIONS = 3
# OpenAI reasoning_effort vocabulary (distinct from OpenRouter's "none"/"xhigh").
OPENAI_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
# Efforts the live backend accepts that OpenAI does not. Translating beats
# dropping: omitting reasoning_effort would leave a batch on the API default
# (medium) while the live backend ran with reasoning off, so the same config
# would silently cost and reason differently across backends.
OPENROUTER_TO_OPENAI_REASONING = {"none": "minimal", "xhigh": "high"}
# OpenAI batch statuses whose results we fold into row states.
RESULT_STATUSES = frozenset({"completed", "expired"})
# OpenAI batch statuses that terminate a batch without usable results.
FATAL_STATUSES = frozenset({"failed", "cancelled"})
TERMINAL_STATUSES = RESULT_STATUSES | FATAL_STATUSES


@dataclass
class BatchStatus:
    """Minimal view of one OpenAI batch."""

    id: str
    status: str
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    # Batch-level failures (bad input file, malformed lines). Often the only
    # diagnostic a `failed` batch has, since it may never produce an error file.
    errors: list[str] = field(default_factory=list)


class SupportsBatchClient(Protocol):
    """Protocol for OpenAI-Batch-compatible clients (fakeable in tests)."""

    def submit(
        self, requests: list[dict[str, object]], *, metadata: dict[str, str]
    ) -> str:
        """Upload requests and create a batch; return its id."""

    def retrieve(self, batch_id: str) -> BatchStatus:
        """Return the current status of one batch."""

    def download_file(self, file_id: str) -> str:
        """Return the raw JSONL text of one batch file."""

    def list_job_batches(
        self, job_id: str, *, created_after: datetime
    ) -> list[BatchStatus]:
        """Return batches tagged with ``job_id`` created after the cutoff."""


class OpenAIBatchClient:
    """OpenAI Batch API client used by the deployed extractor."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the batch extractor.")

    def _client(self) -> object:
        from openai import OpenAI

        return OpenAI(api_key=self.api_key)

    def submit(
        self, requests: list[dict[str, object]], *, metadata: dict[str, str]
    ) -> str:
        client = self._client()
        body = ("\n".join(json.dumps(request) for request in requests)).encode("utf-8")
        upload = client.files.create(file=("batch.jsonl", body), purpose="batch")  # type: ignore[attr-defined]
        batch = client.batches.create(  # type: ignore[attr-defined]
            input_file_id=upload.id,
            endpoint=BATCH_ENDPOINT,
            completion_window=BATCH_COMPLETION_WINDOW,
            metadata=metadata,
        )
        return batch.id

    def retrieve(self, batch_id: str) -> BatchStatus:
        batch = self._client().batches.retrieve(batch_id)  # type: ignore[attr-defined]
        return _batch_status_from_object(batch)

    def download_file(self, file_id: str) -> str:
        return self._client().files.content(file_id).text  # type: ignore[attr-defined]

    def list_job_batches(
        self, job_id: str, *, created_after: datetime
    ) -> list[BatchStatus]:
        # The SDK cursor auto-paginates newest-first over the whole account's
        # batch history; stop at the cutoff so the scan is bounded by this
        # job's lifetime rather than growing with every batch ever created.
        cutoff = created_after.timestamp()
        batches: list[BatchStatus] = []
        for batch in self._client().batches.list(limit=100):  # type: ignore[attr-defined]
            if getattr(batch, "created_at", 0) < cutoff:
                break
            metadata = getattr(batch, "metadata", None) or {}
            if metadata.get("job_id") == job_id:
                batches.append(_batch_status_from_object(batch))
        return batches


def _field(item: object, name: str) -> object:
    """Read one field from an SDK object or the equivalent plain dict."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _batch_error_messages(errors: object) -> list[str]:
    """Flatten an OpenAI ``batch.errors`` payload into readable messages."""
    if errors is None:
        return []
    messages: list[str] = []
    for item in cast(list[object], _field(errors, "data") or []):
        code = _field(item, "code")
        line = _field(item, "line")
        rendered = (
            f"{code}: {_field(item, 'message')}"
            if code
            else str(_field(item, "message"))
        )
        if line is not None:
            rendered = f"{rendered} (input line {line})"
        messages.append(rendered)
    return messages


def _batch_status_from_object(batch: object) -> BatchStatus:
    """Build a BatchStatus from an OpenAI SDK batch object."""
    return BatchStatus(
        id=batch.id,  # type: ignore[attr-defined]
        status=batch.status,  # type: ignore[attr-defined]
        input_file_id=getattr(batch, "input_file_id", None),
        output_file_id=getattr(batch, "output_file_id", None),
        error_file_id=getattr(batch, "error_file_id", None),
        errors=_batch_error_messages(getattr(batch, "errors", None)),
    )


def normalize_batch_model(model: str) -> str:
    """Strip any provider prefix so an OpenRouter slug becomes a native id."""
    return native_model_id(model)


def openai_reasoning_effort(reasoning_effort: str) -> str:
    """Translate a configured reasoning effort into OpenAI's vocabulary.

    Returns ``""`` when no effort is configured, so the caller omits the field
    and takes the model default. Raises for a value neither vocabulary accepts,
    which lets a bad config fail before a job is created.
    """
    effort = (reasoning_effort or "").strip().lower()
    if not effort:
        return ""
    translated = OPENROUTER_TO_OPENAI_REASONING.get(effort, effort)
    if translated not in OPENAI_REASONING_EFFORTS:
        allowed = ", ".join(
            sorted(OPENAI_REASONING_EFFORTS | set(OPENROUTER_TO_OPENAI_REASONING))
        )
        raise ValueError(
            f"Unsupported OpenAI reasoning_effort {effort!r}; expected one of {allowed}."
        )
    return translated


def build_request_body(
    messages: list[dict[str, str]], *, model: str, reasoning_effort: str
) -> dict[str, object]:
    """Build one ``/v1/chat/completions`` request body for the batch endpoint.

    Sampling params come from the shared ``sampling_params`` policy so this body
    matches what the live backend sends for the same model, and the configured
    effort is translated into OpenAI's vocabulary.
    """
    body: dict[str, object] = {
        "model": native_model_id(model),
        "messages": messages,
        **sampling_params(model),
    }
    effort = openai_reasoning_effort(reasoning_effort)
    if effort:
        body["reasoning_effort"] = effort
    return body


# --------------------------------------------------------------------------- #
# File-native job state
# --------------------------------------------------------------------------- #


@dataclass
class RowEntry:
    """One item's resumable state plus its originating partition.

    ``pending`` records the outstanding request the row is waiting on
    (``{"batch_id", "stage", "attempt_index"}``); it is set at submit time and
    cleared when that batch's result is folded. Folding only applies a result
    whose batch matches ``pending``, so re-delivery of a stale batch result
    (crash replay, adopted orphan) is a no-op rather than a mis-stage fold.

    ``resubmissions`` counts consecutive expired-without-result rounds; it
    resets whenever a real result folds and terminates the row at the cap.
    """

    row_state: ExtractionRowState
    date: str
    shard: str
    pending: dict[str, object] | None = None
    resubmissions: int = 0


@dataclass
class BatchRecord:
    """One in-flight OpenAI batch and the item custom_ids it carries."""

    batch_id: str
    tick: int
    custom_ids: list[str]


@dataclass
class JobState:
    """In-memory snapshot of one extract job during a tick."""

    job_id: str
    model: str
    reasoning_effort: str
    max_attempts: int
    claimed_partitions: list[str]
    rows: dict[str, RowEntry]
    batches: list[BatchRecord] = field(default_factory=list)
    all_batch_ids: list[str] = field(default_factory=list)
    tick: int = 0


@dataclass
class ExtractTickResult:
    """Outcome of one ``advance_extract_job`` tick."""

    status: str  # idle | submitted | waiting | completed
    job_id: str | None = None
    submitted_batches: int = 0
    folded_rows: int = 0
    awaiting_rows: int = 0
    in_flight_batches: int = 0
    terminal_rows: int = 0
    mentions: pd.DataFrame | None = None


def batches_root(
    artifact_root: ArtifactPath | None = None, *, data_dir: Path | None = None
) -> str:
    """Return the root that stores extract batch job state."""
    return dataset_root(
        EXTRACT_BATCHES_DATASET, artifact_root=artifact_root, data_dir=data_dir
    )


def active_job_path(root: str) -> str:
    """Return the path of the single-active-job marker for an artifact root."""
    return join_artifact_path(root, EXTRACT_BATCHES_DATASET, ACTIVE_JOB_FILENAME)


def _job_dir(root: str, job_id: str) -> str:
    return join_artifact_path(root, EXTRACT_BATCHES_DATASET, f"job_id={job_id}")


def _manifest_path(root: str, job_id: str) -> str:
    return join_artifact_path(_job_dir(root, job_id), "manifest.json")


def _state_path(root: str, job_id: str) -> str:
    return join_artifact_path(_job_dir(root, job_id), "state.jsonl")


def _batches_path(root: str, job_id: str) -> str:
    return join_artifact_path(_job_dir(root, job_id), "batches.json")


def _tick_path(root: str, job_id: str, tick: int) -> str:
    return join_artifact_path(_job_dir(root, job_id), "ticks", f"tick={tick}.json")


def _read_active_job(root: str) -> str | None:
    path = active_job_path(root)
    if not artifact_exists(path):
        return None
    payload = cast(dict[str, object], read_json_artifact(path))
    job_id = payload.get("job_id")
    return str(job_id) if job_id else None


def _new_job_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _job_created_at(job_id: str) -> datetime:
    """Recover a job's creation time from its timestamp-shaped id."""
    try:
        return datetime.strptime(job_id, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    except ValueError:
        # Unknown id shape: fall back to an unbounded scan rather than miss
        # orphans.
        return datetime.fromtimestamp(0, tz=UTC)


def _serialize_state_jsonl(rows: dict[str, RowEntry]) -> str:
    lines = [
        json.dumps(
            {
                "source": {"date": entry.date, "shard": entry.shard},
                "pending": entry.pending,
                "resubmissions": entry.resubmissions,
                "row": entry.row_state.to_state_dict(),
            },
            sort_keys=True,
        )
        for entry in rows.values()
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def _load_state_jsonl(text: str) -> dict[str, RowEntry]:
    rows: dict[str, RowEntry] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        row_state = ExtractionRowState.from_state_dict(payload["row"])
        source = payload["source"]
        rows[row_state.item_id] = RowEntry(
            row_state=row_state,
            date=source["date"],
            shard=source["shard"],
            pending=cast(dict[str, object] | None, payload.get("pending")),
            resubmissions=int(cast(int, payload.get("resubmissions", 0))),
        )
    return rows


def _load_job_state(root: str, job_id: str) -> JobState:
    manifest = cast(dict[str, object], read_json_artifact(_manifest_path(root, job_id)))
    rows = _load_state_jsonl(read_text_artifact(_state_path(root, job_id)))
    batches_payload = cast(
        dict[str, object], read_json_artifact(_batches_path(root, job_id))
    )
    batches = [
        BatchRecord(
            batch_id=str(record["batch_id"]),
            tick=int(cast(int, record["tick"])),
            custom_ids=[str(cid) for cid in cast(list[object], record["custom_ids"])],
        )
        for record in cast(list[dict[str, object]], batches_payload.get("batches", []))
    ]
    return JobState(
        job_id=job_id,
        model=str(manifest["model"]),
        reasoning_effort=str(manifest["reasoning_effort"]),
        max_attempts=int(cast(int, manifest["max_attempts"])),
        claimed_partitions=[
            str(path) for path in cast(list[object], manifest["claimed_partitions"])
        ],
        rows=rows,
        batches=batches,
        all_batch_ids=[
            str(bid)
            for bid in cast(list[object], batches_payload.get("all_batch_ids", []))
        ],
        tick=int(cast(int, batches_payload.get("tick", 0))),
    )


def _save_state(root: str, job: JobState) -> None:
    write_text_artifact(_state_path(root, job.job_id), _serialize_state_jsonl(job.rows))


def _save_batches(root: str, job: JobState) -> None:
    write_json_artifact(
        _batches_path(root, job.job_id),
        {
            "tick": job.tick,
            "all_batch_ids": job.all_batch_ids,
            "batches": [
                {
                    "batch_id": record.batch_id,
                    "tick": record.tick,
                    "custom_ids": record.custom_ids,
                }
                for record in job.batches
            ],
        },
    )


def _parse_jsonl_by_custom_id(text: str) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = payload.get("custom_id")
        if custom_id is not None:
            results[str(custom_id)] = payload
    return results


# --------------------------------------------------------------------------- #
# The tick
# --------------------------------------------------------------------------- #


def _create_job(
    root: str,
    *,
    data_dir: Path | None,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    force: bool,
) -> JobState | None:
    """Start a new job from pending classification partitions, or None if idle."""
    entries, claimed_paths = collect_pending_extract_items(
        artifact_root=root, data_dir=data_dir, force=force
    )
    if not claimed_paths:
        return None
    job_id = _new_job_id()
    rows: dict[str, RowEntry] = {}
    for item_row, partition_date, shard in entries:
        row_state = ExtractionRowState(
            item_row=cast(dict[str, object], item_row),
            stage_name="ner",
        )
        # Prime the first request; a preprocess failure terminates the row now.
        initial_messages(row_state)
        item_id = row_state.item_id
        if len(item_id) > MAX_CUSTOM_ID_LENGTH:
            raise ValueError(
                f"item_id {item_id!r} exceeds the {MAX_CUSTOM_ID_LENGTH}-char "
                "OpenAI custom_id limit."
            )
        rows[item_id] = RowEntry(row_state=row_state, date=partition_date, shard=shard)
    job = JobState(
        job_id=job_id,
        model=model,
        reasoning_effort=reasoning_effort,
        max_attempts=max_attempts,
        claimed_partitions=sorted(claimed_paths),
        rows=rows,
    )
    write_json_artifact(
        _manifest_path(root, job_id),
        {
            "job_id": job_id,
            "created_at": job_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_attempts": max_attempts,
            "claimed_partitions": job.claimed_partitions,
        },
    )
    _save_state(root, job)
    write_json_artifact(active_job_path(root), {"job_id": job_id})
    LOGGER.info(
        "Started extract job %s: items=%s partitions=%s",
        job_id,
        len(rows),
        len(claimed_paths),
    )
    return job


def _reconcile_orphans(root: str, job: JobState, client: SupportsBatchClient) -> None:
    """Adopt any batch tagged with this job_id that we never recorded (crash gap).

    An orphan can only be the most recent submission (every recorded submit
    persists ``batches.json`` immediately), so its rows' saved states still hold
    exactly the messages it carries; re-attaching ``pending`` lets its results
    fold normally instead of being discarded as stale.
    """
    known = set(job.all_batch_ids)
    # An hour of margin absorbs clock skew between this host and the API.
    cutoff = _job_created_at(job.job_id) - timedelta(hours=1)
    for status in client.list_job_batches(job.job_id, created_after=cutoff):
        if status.id in known:
            continue
        LOGGER.warning("Adopting orphan batch %s for job %s", status.id, job.job_id)
        custom_ids: list[str] = []
        if status.input_file_id:
            custom_ids = list(
                _parse_jsonl_by_custom_id(client.download_file(status.input_file_id))
            )
        for custom_id in custom_ids:
            entry = job.rows.get(custom_id)
            if (
                entry is not None
                and entry.row_state.state is None
                and entry.pending is None
            ):
                entry.pending = _pending_marker(status.id, entry)
        job.batches.append(
            BatchRecord(batch_id=status.id, tick=job.tick, custom_ids=custom_ids)
        )
        job.all_batch_ids.append(status.id)


def _pending_marker(batch_id: str, entry: RowEntry) -> dict[str, object]:
    """Describe the request a row is waiting on in the given batch."""
    return {
        "batch_id": batch_id,
        "stage": entry.row_state.current_attempt.stage_name,
        "attempt_index": entry.row_state.current_attempt.attempt_index,
    }


def _fatal_batch_error(status: BatchStatus) -> str:
    """Describe a failed/cancelled batch for a row that got no result line."""
    detail = (
        "; ".join(status.errors) if status.errors else "no batch-level errors reported"
    )
    return f"OpenAI batch {status.status}: {status.id} ({detail})"


def _fold_completed_batches(
    job: JobState,
    client: SupportsBatchClient,
    *,
    max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS,
) -> int:
    """Poll in-flight batches; fold terminal ones into row states.

    Returns the number of rows advanced. Batches still running are left in place.
    """
    folded = 0
    still_in_flight: list[BatchRecord] = []
    for record in job.batches:
        status = client.retrieve(record.batch_id)
        if status.status not in TERMINAL_STATUSES:
            still_in_flight.append(record)
            continue
        # Fetch whatever the batch produced, whatever its terminal status. A
        # failed/cancelled batch can still have an error file whose per-request
        # lines say why, which beats a generic "batch failed" on every row.
        results: dict[str, dict[str, object]] = {}
        for file_id in (status.output_file_id, status.error_file_id):
            if not file_id:
                continue
            try:
                results.update(_parse_jsonl_by_custom_id(client.download_file(file_id)))
            except Exception as exc:  # noqa: BLE001
                # Diagnostics are best-effort: a download failure must not strand
                # the rows this batch owns.
                LOGGER.warning(
                    "Could not read file %s for batch %s (%s): %s",
                    file_id,
                    record.batch_id,
                    status.status,
                    exc,
                )
        for custom_id in record.custom_ids:
            entry = job.rows.get(custom_id)
            if entry is None or entry.row_state.state is not None:
                continue
            if (
                entry.pending is None
                or entry.pending.get("batch_id") != record.batch_id
            ):
                # The row is not waiting on this batch (stale record replayed
                # after a partial tick): discard rather than mis-stage fold.
                LOGGER.warning(
                    "Ignoring stale result for %s from batch %s (pending=%s)",
                    custom_id,
                    record.batch_id,
                    entry.pending,
                )
                continue
            entry.pending = None
            line = results.get(custom_id)
            if line is not None:
                entry.resubmissions = 0
                try:
                    text = extract_batch_response_text(line)
                except Exception as exc:  # noqa: BLE001
                    record_stage_error(entry.row_state, str(exc))
                else:
                    handle_response(
                        entry.row_state, text, max_attempts=job.max_attempts
                    )
                folded += 1
            elif status.status in FATAL_STATUSES:
                # Whole-batch failure (usually bad input): terminate, don't loop.
                # No per-request line for this row, so carry the batch-level
                # errors into its audit record instead of a bare status.
                record_stage_error(entry.row_state, _fatal_batch_error(status))
                folded += 1
            else:
                # Expired-and-missing: pending is cleared so the row re-submits
                # next tick — but each round costs a 24h window, so terminate at
                # the cap instead of blocking the job forever.
                entry.resubmissions += 1
                if entry.resubmissions >= max_resubmissions:
                    record_stage_error(
                        entry.row_state,
                        f"Batch request expired {entry.resubmissions} times "
                        "without a result.",
                    )
                    folded += 1
        if status.status in FATAL_STATUSES:
            LOGGER.error(
                "OpenAI batch %s ended as %s for job %s: errors=[%s] "
                "per_request_diagnostics=%s/%s error_file=%s",
                record.batch_id,
                status.status,
                job.job_id,
                "; ".join(status.errors) or "none reported",
                len(results),
                len(record.custom_ids),
                status.error_file_id or "none",
            )
    job.batches = still_in_flight
    return folded


def _awaiting_rows(job: JobState) -> list[RowEntry]:
    """Non-terminal rows with no outstanding batch request."""
    return [
        entry
        for entry in job.rows.values()
        if entry.row_state.state is None and entry.pending is None
    ]


def _build_requests(
    job: JobState, awaiting: list[RowEntry], *, max_batch_bytes: int
) -> list[tuple[RowEntry, dict[str, object], int]]:
    """Build one request per awaiting row, with its serialized JSONL line size.

    A single request that alone exceeds the byte budget can never be submitted;
    its row is terminated rather than wedging the job on every tick.
    """
    buildable: list[tuple[RowEntry, dict[str, object], int]] = []
    for entry in awaiting:
        request: dict[str, object] = {
            "custom_id": entry.row_state.item_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": build_request_body(
                entry.row_state.current_attempt.messages,
                model=job.model,
                reasoning_effort=job.reasoning_effort,
            ),
        }
        size = len(json.dumps(request).encode("utf-8")) + 1  # newline
        if size > max_batch_bytes:
            record_stage_error(
                entry.row_state,
                f"Batch request of {size} bytes exceeds the "
                f"{max_batch_bytes}-byte per-batch budget.",
            )
            continue
        buildable.append((entry, request, size))
    return buildable


def _plan_chunks(
    requests: list[tuple[RowEntry, dict[str, object], int]],
    *,
    max_requests_per_batch: int,
    max_batch_bytes: int,
) -> list[list[tuple[RowEntry, dict[str, object]]]]:
    """Pack requests into batches bounded by both request count and bytes."""
    chunks: list[list[tuple[RowEntry, dict[str, object]]]] = []
    current: list[tuple[RowEntry, dict[str, object]]] = []
    current_bytes = 0
    for entry, request, size in requests:
        if current and (
            len(current) >= max_requests_per_batch
            or current_bytes + size > max_batch_bytes
        ):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append((entry, request))
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def _submit_awaiting(
    root: str,
    job: JobState,
    awaiting: list[RowEntry],
    client: SupportsBatchClient,
    *,
    max_requests_per_batch: int,
    max_batch_bytes: int,
) -> int:
    """Submit awaiting rows in one or more batches. Returns batches submitted.

    State and the batch record are persisted after every chunk, so a crash
    mid-submission loses at most the very last batch — which orphan
    reconciliation recovers by its job_id tag next tick.
    """
    requests = _build_requests(job, awaiting, max_batch_bytes=max_batch_bytes)
    if len(requests) < len(awaiting):
        # Oversized rows were terminated above; persist before any submit.
        _save_state(root, job)
    submitted = 0
    for chunk in _plan_chunks(
        requests,
        max_requests_per_batch=max_requests_per_batch,
        max_batch_bytes=max_batch_bytes,
    ):
        batch_id = client.submit(
            [request for _, request in chunk],
            metadata={"job_id": job.job_id, "tick": str(job.tick)},
        )
        for entry, _ in chunk:
            entry.pending = _pending_marker(batch_id, entry)
        job.batches.append(
            BatchRecord(
                batch_id=batch_id,
                tick=job.tick,
                custom_ids=[entry.row_state.item_id for entry, _ in chunk],
            )
        )
        job.all_batch_ids.append(batch_id)
        _save_state(root, job)
        _save_batches(root, job)
        submitted += 1
    return submitted


def advance_extract_job(
    *,
    batch_client: SupportsBatchClient,
    artifact_root: ArtifactPath | None = None,
    data_dir: Path | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_requests_per_batch: int = DEFAULT_MAX_REQUESTS_PER_BATCH,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS,
    force: bool = False,
) -> ExtractTickResult:
    """Advance the extract job by one tick.

    If no job is active, start one from pending classification partitions. Fold any
    completed OpenAI batches into row states, submit the next batch for rows still
    needing a call, then persist state. When every row is terminal, write the
    mention partitions + audit log + completion registry and clear the active-job
    marker (match/finalize is the orchestrator's responsibility).
    """
    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    resolved_model = model or settings.EXTRACTOR_BATCH_MODEL
    resolved_reasoning = (
        reasoning_effort
        if reasoning_effort is not None
        else settings.EXTRACTOR_BATCH_REASONING
    )
    # Fail on the config, not mid-tick: an effort OpenAI rejects would otherwise
    # surface only once the first request body is built, after a job exists.
    translated_reasoning = openai_reasoning_effort(resolved_reasoning)
    if translated_reasoning != resolved_reasoning.strip().lower():
        LOGGER.info(
            "Translated reasoning effort %r to OpenAI's %r for the batch backend.",
            resolved_reasoning,
            translated_reasoning or "<model default>",
        )

    active_job_id = _read_active_job(resolved_root)
    if active_job_id is None:
        job = _create_job(
            resolved_root,
            data_dir=data_dir,
            model=resolved_model,
            reasoning_effort=resolved_reasoning,
            max_attempts=max_attempts,
            force=force,
        )
        if job is None:
            return ExtractTickResult(status="idle")
        folded = 0
    else:
        if force:
            LOGGER.warning(
                "Ignoring force=True: extract job %s is already active and keeps the "
                "partitions it claimed at creation. Let it finish (or clear %s) "
                "before forcing a re-extract.",
                active_job_id,
                active_job_path(resolved_root),
            )
        job = _load_job_state(resolved_root, active_job_id)
        _reconcile_orphans(resolved_root, job, batch_client)
        folded = _fold_completed_batches(
            job, batch_client, max_resubmissions=max_resubmissions
        )

    job.tick += 1
    awaiting = _awaiting_rows(job)

    # Durable-before-side-effect: persist the folded row states AND the pruned
    # batch list before creating any batch. A terminal batch must never survive
    # in batches.json once its results are folded (it would be re-folded on the
    # next tick if a later submit fails), and a crash after submit is
    # recoverable via orphan reconciliation next tick.
    _save_state(resolved_root, job)
    _save_batches(resolved_root, job)
    submitted = _submit_awaiting(
        resolved_root,
        job,
        awaiting,
        batch_client,
        max_requests_per_batch=max_requests_per_batch,
        max_batch_bytes=max_batch_bytes,
    )
    # Recompute after submitting: submitted rows are now pending and oversized
    # rows were terminated, so anything left here could not be dispatched.
    still_awaiting = _awaiting_rows(job)

    terminal_rows = sum(
        1 for entry in job.rows.values() if entry.row_state.state is not None
    )
    write_json_artifact(
        _tick_path(resolved_root, job.job_id, job.tick),
        {
            "tick": job.tick,
            "folded_rows": folded,
            "submitted_batches": submitted,
            "awaiting_rows": len(awaiting),
            "in_flight_batches": len(job.batches),
            "terminal_rows": terminal_rows,
            "total_rows": len(job.rows),
        },
    )

    if not job.batches and not still_awaiting:
        mentions = _finalize_job(resolved_root, job, data_dir=data_dir)
        return ExtractTickResult(
            status="completed",
            job_id=job.job_id,
            folded_rows=folded,
            submitted_batches=submitted,
            terminal_rows=terminal_rows,
            mentions=mentions,
        )

    LOGGER.info(
        "Extract job %s tick=%s folded=%s submitted=%s in_flight=%s terminal=%s/%s",
        job.job_id,
        job.tick,
        folded,
        submitted,
        len(job.batches),
        terminal_rows,
        len(job.rows),
    )
    return ExtractTickResult(
        status="submitted" if submitted else "waiting",
        job_id=job.job_id,
        folded_rows=folded,
        submitted_batches=submitted,
        awaiting_rows=len(awaiting),
        in_flight_batches=len(job.batches),
        terminal_rows=terminal_rows,
    )


def _finalize_job(root: str, job: JobState, *, data_dir: Path | None) -> pd.DataFrame:
    """Write mentions/audit/completion for a finished job and clear the marker."""
    row_entries = [
        (entry.row_state, entry.date, entry.shard) for entry in job.rows.values()
    ]
    mentions = finalize_extract_outputs(
        row_entries,
        claimed_classification_paths=set(job.claimed_partitions),
        run_id=job.job_id,
        model=job.model,
        reasoning_effort=job.reasoning_effort,
        max_attempts=job.max_attempts,
        artifact_root=root,
        data_dir=data_dir,
    )
    # Removing the active marker frees the next tick to start a fresh job. Use an
    # empty-body write since storage has no delete helper; absence is not required.
    write_json_artifact(active_job_path(root), {"job_id": None})
    LOGGER.info(
        "Extract job %s complete: rows=%s mentions=%s",
        job.job_id,
        len(job.rows),
        len(mentions),
    )
    return mentions.reindex(columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
