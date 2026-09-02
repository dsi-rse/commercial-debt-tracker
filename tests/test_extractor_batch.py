"""Tests for the OpenAI Batch extraction backend and resumable state machine."""

# ruff: noqa: ANN101, D102, D107

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import cdt.extractor.batch as batch_module
from cdt.classifier import classifications_root
from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import completion_registry_path, load_row_failures
from cdt.extractor import (
    advance_extract_job,
    describe_active_job,
    mentions_root,
    reset_active_job,
)
from cdt.extractor.batch import (
    BatchRecord,
    BatchStatus,
    JobState,
    OpenAIBatchClient,
    RowEntry,
    _batch_error_messages,
    _fold_completed_batches,
    _load_state_jsonl,
    _reconcile_orphans,
    _serialize_state_jsonl,
    active_job_path,
    build_request_body,
    normalize_batch_model,
    openai_reasoning_effort,
)
from cdt.extractor.core import (
    EXTRACTOR_TEMPERATURE,
    REASONING_EFFORTS,
    CompletionResult,
    ExtractionRowState,
    extract_batch_response_text,
    handle_response,
    initial_messages,
    load_prompt,
    run_extraction_workflow,
    sampling_params,
)
from cdt.storage import (
    artifact_exists,
    read_dataset,
    read_json_artifact,
    write_partition_table,
)

# --------------------------------------------------------------------------- #
# Canned prompts/responses that exercise all three stages deterministically
# --------------------------------------------------------------------------- #

NODEBT_TEXT = "This is the extracted event text."
NODEBT_NER = "<body>This is the extracted event text.</body>"

MULTI_TEXT = "Company entered into a Term Loan on January 1, 2024."
MULTI_NER = (
    "<body>Company entered into a <debt_instrument>Term Loan</debt_instrument> "
    "on <date>January 1, 2024</date>.</body>"
)
MULTI_IE = json.dumps(
    [
        {
            "name": ["tag-1"],
            "start_date": {"evidence": ["tag-2"], "normalized_date": "2024-01-01"},
        }
    ]
)

_NER_PROMPT = load_prompt("ner")
_IE_PROMPT = load_prompt("instrument_ie")
_RELATION_PROMPT = load_prompt("instrument_relation")


def _detect_stage(body: dict[str, object]) -> str:
    system = body["messages"][0]["content"]  # type: ignore[index]
    if system == _NER_PROMPT:
        return "ner"
    if system == _IE_PROMPT:
        return "instrument_ie"
    if system == _RELATION_PROMPT:
        return "instrument_relation"
    raise AssertionError("Unrecognized system prompt in request body.")


def _item_row(item_id: str, text: str) -> dict[str, str | None]:
    return {
        "item_id": item_id,
        "text": text,
        "accession_number": "000000000000000000",
        "cik": "320193",
        "company_name": "Example Inc.",
        "date": "2024-01-02",
        "item": "8.01",
    }


def seed_classification(
    tmp_path: Path,
    items: list[dict[str, object]],
    *,
    date: str = "2024-01-02",
    shard: str = "0001",
) -> str:
    """Write one canonical classification partition for the given items."""
    rows: list[dict[str, object]] = []
    for item in items:
        row: dict[str, object] = {column: None for column in CLASSIFIED_ITEM_COLUMNS}
        row.update(
            {
                "item_id": item["item_id"],
                "text": item.get("text", ""),
                "accession_number": item.get("accession_number", "000000000000000000"),
                "cik": item.get("cik", "320193"),
                "company_name": "Example Inc.",
                "date": date,
                "item": "8.01",
                "label": "relevant" if item.get("relevance", True) else "irrelevant",
                "relevance": item.get("relevance", True),
                "classification_score": 1.0,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=CLASSIFIED_ITEM_COLUMNS)
    return write_partition_table(
        classifications_root(tmp_path),
        partition={"date": date, "shard": shard},
        table=table,
    )


class FakeBatchClient:
    """In-memory SupportsBatchClient that completes batches immediately.

    ``scripts`` maps item_id -> {stage_name: response_spec}. A spec is either a
    string (successful assistant text), ``("error", message)``, or ``("missing",)``
    to omit the result. ``forced_status`` overrides the OpenAI status for a batch
    by its submission index (0-based).

    ``batch_errors`` and ``fatal_request_errors`` model what a real failed batch
    can still expose: batch-level errors on the batch object, and per-request
    lines in its error file. Both are keyed by submission index.
    """

    def __init__(
        self,
        scripts: dict[str, dict[str, object]],
        *,
        forced_status: dict[int, str] | None = None,
        batch_errors: dict[int, list[str]] | None = None,
        fatal_request_errors: dict[int, dict[str, str]] | None = None,
    ) -> None:
        self.scripts = scripts
        self.forced_status = forced_status or {}
        self.batch_errors = batch_errors or {}
        self.fatal_request_errors = fatal_request_errors or {}
        self.batches: dict[str, dict[str, object]] = {}
        self._submit_count = 0
        self.submitted: list[tuple[str, list[str]]] = []
        self.downloaded: list[str] = []

    def submit(
        self, requests: list[dict[str, object]], *, metadata: dict[str, str]
    ) -> str:
        index = self._submit_count
        self._submit_count += 1
        batch_id = f"batch-{index}"
        status = self.forced_status.get(index, "completed")
        out_lines: list[str] = []
        err_lines: list[str] = []
        if status not in {"failed", "cancelled", "expired"}:
            for request in requests:
                custom_id = str(request["custom_id"])
                stage = _detect_stage(request["body"])  # type: ignore[arg-type]
                spec = self.scripts[custom_id][stage]
                if isinstance(spec, tuple) and spec[0] == "error":
                    err_lines.append(
                        json.dumps(
                            {"custom_id": custom_id, "error": {"message": spec[1]}}
                        )
                    )
                elif isinstance(spec, tuple) and spec[0] == "status":
                    # A per-request non-200 (e.g. a provider 500) in the output file.
                    out_lines.append(
                        json.dumps(
                            {
                                "custom_id": custom_id,
                                "response": {
                                    "status_code": spec[1],
                                    "body": {"error": {"type": "server_error"}},
                                },
                            }
                        )
                    )
                elif isinstance(spec, tuple) and spec[0] == "missing":
                    continue
                else:
                    out_lines.append(
                        json.dumps(
                            {
                                "custom_id": custom_id,
                                "response": {
                                    "status_code": 200,
                                    "body": {
                                        "choices": [{"message": {"content": str(spec)}}]
                                    },
                                },
                            }
                        )
                    )
        # A fatal batch produces no results, but can still explain itself
        # per request in its error file.
        for custom_id, message in (self.fatal_request_errors.get(index) or {}).items():
            err_lines.append(
                json.dumps({"custom_id": custom_id, "error": {"message": message}})
            )
        self.batches[batch_id] = {
            "status": status,
            "metadata": metadata,
            "input": requests,
            "out": "\n".join(out_lines),
            "err": "\n".join(err_lines),
            "errors": list(self.batch_errors.get(index, [])),
        }
        self.submitted.append(
            (batch_id, [str(request["custom_id"]) for request in requests])
        )
        return batch_id

    def retrieve(self, batch_id: str) -> BatchStatus:
        record = self.batches[batch_id]
        return BatchStatus(
            id=batch_id,
            status=str(record["status"]),
            input_file_id=f"{batch_id}-in",
            output_file_id=f"{batch_id}-out" if record["out"] else None,
            error_file_id=f"{batch_id}-err" if record["err"] else None,
            errors=list(record.get("errors") or []),  # type: ignore[arg-type]
        )

    def download_file(self, file_id: str) -> str:
        self.downloaded.append(file_id)
        batch_id, kind = file_id.rsplit("-", 1)
        record = self.batches[batch_id]
        if kind == "out":
            return str(record["out"])
        if kind == "err":
            return str(record["err"])
        if kind == "in":
            return "\n".join(
                json.dumps(request)
                for request in list(record["input"])  # type: ignore[arg-type]
            )
        raise KeyError(file_id)

    def list_job_batches(
        self, job_id: str, *, created_after: object = None
    ) -> list[BatchStatus]:
        del created_after  # in-memory fake keeps no timestamps
        return [
            self.retrieve(batch_id)
            for batch_id, record in self.batches.items()
            if dict(record["metadata"]).get("job_id") == job_id  # type: ignore[arg-type]
        ]


class FlakySubmitClient(FakeBatchClient):
    """FakeBatchClient whose submit raises on selected call indexes (0-based)."""

    def __init__(
        self, scripts: dict[str, dict[str, object]], *, fail_on_calls: set[int]
    ) -> None:
        super().__init__(scripts)
        self.fail_on_calls = fail_on_calls
        self.submit_calls = 0

    def submit(
        self, requests: list[dict[str, object]], *, metadata: dict[str, str]
    ) -> str:
        call = self.submit_calls
        self.submit_calls += 1
        if call in self.fail_on_calls:
            raise RuntimeError("simulated submit failure")
        return super().submit(requests, metadata=metadata)


class ScriptedChatClient:
    """Fake SupportsChatCompletion returning a fixed response sequence."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    async def complete(
        self, *, messages: list[dict[str, str]], model: str, reasoning_effort: str
    ) -> CompletionResult:
        del messages, model, reasoning_effort
        response = self.responses[self.index]
        self.index += 1
        return CompletionResult(
            text=response,
            finish_reason="stop",
            usage={"completion_tokens": 7, "prompt_tokens": 11, "total_tokens": 18},
            response_id=f"gen-{self.index}",
            served_model=model_for_test(),
        )


def model_for_test() -> str:
    """Return a stable served-model value for the fake client."""
    return "openai/gpt-5.4"


def _drive_resumable(
    item_row: dict[str, object], responses: list[str], max_attempts: int
) -> ExtractionRowState:
    row_state = ExtractionRowState(item_row=item_row, stage_name="ner")
    messages = initial_messages(row_state)
    index = 0
    while messages is not None:
        response = responses[index]
        index += 1
        # Mirror what `_fold_completed_batches` now passes through, so parity
        # with the live workflow still means parity of the recorded attempt.
        messages = handle_response(
            row_state,
            response,
            max_attempts=max_attempts,
            completion=CompletionResult(
                text=response,
                finish_reason="stop",
                usage={
                    "completion_tokens": 7,
                    "prompt_tokens": 11,
                    "total_tokens": 18,
                },
                response_id=f"gen-{index}",
                served_model=model_for_test(),
            ),
        )
    return row_state


# --------------------------------------------------------------------------- #
# Resumable state-machine parity with the synchronous workflow
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,responses",
    [
        (NODEBT_TEXT, [NODEBT_NER]),
        (MULTI_TEXT, [MULTI_NER, MULTI_IE]),
        ("bad", ["not xml", "still not xml", "nope"]),
    ],
)
def test_resumable_matches_sync_workflow(text: str, responses: list[str]) -> None:
    """The resumable machine must produce the same audit as the sync loop."""
    item_row = _item_row("item-1", text)
    sync_state = asyncio.run(
        run_extraction_workflow(
            item_row=dict(item_row),
            model="m",
            reasoning_effort="none",
            max_attempts=3,
            client=ScriptedChatClient(list(responses)),
        )
    )
    resumable_state = _drive_resumable(dict(item_row), list(responses), max_attempts=3)
    assert resumable_state.to_audit_dict() == sync_state.to_audit_dict()


def test_early_stop_and_multi_stage_states() -> None:
    """Confirm the crafted responses reach the intended terminal states."""
    nodebt = _drive_resumable(_item_row("a", NODEBT_TEXT), [NODEBT_NER], 3)
    assert nodebt.state == "SUCCESS"
    assert nodebt.debt_instrument_mentions == []

    multi = _drive_resumable(_item_row("b", MULTI_TEXT), [MULTI_NER, MULTI_IE], 3)
    assert multi.state == "SUCCESS"
    assert len(multi.debt_instrument_mentions) == 1
    assert multi.debt_instrument_mentions[0]["name"] == "Term Loan"


# --------------------------------------------------------------------------- #
# state.jsonl round-trip
# --------------------------------------------------------------------------- #


def test_state_dict_round_trip_midflight() -> None:
    """Serializing a mid-flight row and resuming yields the same result."""
    row_state = ExtractionRowState(
        item_row=_item_row("b", MULTI_TEXT), stage_name="ner"
    )
    initial_messages(row_state)
    # Advance past NER so the row is mid-flight awaiting the instrument_ie response.
    handle_response(row_state, MULTI_NER, max_attempts=3)
    assert row_state.state is None
    assert row_state.current_attempt.stage_name == "instrument_ie"

    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())
    assert restored.to_state_dict() == row_state.to_state_dict()

    finished = handle_response(restored, MULTI_IE, max_attempts=3)
    assert finished is None
    assert restored.state == "SUCCESS"
    assert len(restored.debt_instrument_mentions) == 1


# --------------------------------------------------------------------------- #
# Full poll-driven job lifecycle
# --------------------------------------------------------------------------- #


def _advance(tmp_path: Path, client: FakeBatchClient) -> object:
    return advance_extract_job(
        batch_client=client,
        artifact_root=tmp_path,
        model="gpt-5.4",
        reasoning_effort="none",
        max_attempts=3,
    )


def test_job_lifecycle_completes_and_writes_mentions(tmp_path: Path) -> None:
    """A two-item job advances across ticks and finalizes mentions."""
    seed_classification(
        tmp_path,
        [
            {"item_id": "item-nodebt", "text": NODEBT_TEXT},
            {"item_id": "item-multi", "text": MULTI_TEXT},
        ],
    )
    client = FakeBatchClient(
        {
            "item-nodebt": {"ner": NODEBT_NER},
            "item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE},
        }
    )

    first = _advance(tmp_path, client)
    assert first.status == "submitted"
    assert artifact_exists(active_job_path(str(tmp_path)))

    second = _advance(tmp_path, client)
    assert second.status == "submitted"

    third = _advance(tmp_path, client)
    assert third.status == "completed"

    written = read_dataset(mentions_root(tmp_path))
    assert written["name"].to_list() == ["Term Loan"]

    completed = read_json_artifact(
        completion_registry_path("extract", artifact_root=tmp_path)
    )
    assert len(completed["partitions"]) == 1

    # Active marker cleared; a subsequent tick is idle (nothing pending).
    idle = _advance(tmp_path, client)
    assert idle.status == "idle"


def test_empty_job_finalizes_immediately(tmp_path: Path) -> None:
    """A partition with no relevant items completes in one tick with no batch."""
    seed_classification(
        tmp_path, [{"item_id": "item-x", "text": NODEBT_TEXT, "relevance": False}]
    )
    client = FakeBatchClient({})

    result = _advance(tmp_path, client)

    assert result.status == "completed"
    assert client.submitted == []
    completed = read_json_artifact(
        completion_registry_path("extract", artifact_root=tmp_path)
    )
    assert len(completed["partitions"]) == 1


def test_request_error_terminates_row(tmp_path: Path) -> None:
    """A per-request error marks the row ERROR without blocking completion."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient({"item-multi": {"ner": ("error", "content filtered")}})

    assert _advance(tmp_path, client).status == "submitted"
    assert _advance(tmp_path, client).status == "completed"

    assert read_dataset(mentions_root(tmp_path)).empty
    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit) == 1
    assert '"state": "ERROR"' in audit[0].read_text(encoding="utf-8")


def test_expired_batch_resubmits_missing_items(tmp_path: Path) -> None:
    """An expired batch salvages nothing and resubmits the item next tick."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}},
        forced_status={0: "expired"},
    )

    statuses = [_advance(tmp_path, client).status for _ in range(4)]

    assert statuses[-1] == "completed"
    # The NER request was submitted twice: the expired batch and its resubmission.
    ner_submissions = [ids for _, ids in client.submitted if "item-multi" in ids]
    assert len(ner_submissions) >= 3  # b0 (expired NER), b1 (NER), b2 (IE)
    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]


def test_repeatedly_expired_row_terminates_at_cap(tmp_path: Path) -> None:
    """A row whose batches keep expiring stops resubmitting at the cap."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}},
        forced_status={0: "expired", 1: "expired", 2: "expired", 3: "expired"},
    )

    def _tick() -> object:
        return advance_extract_job(
            batch_client=client,
            artifact_root=tmp_path,
            model="gpt-5.4",
            reasoning_effort="none",
            max_attempts=3,
            max_resubmissions=3,
        )

    assert _tick().status == "submitted"  # b0 (expires)
    assert _tick().status == "submitted"  # fold expiry 1, resubmit b1 (expires)
    assert _tick().status == "submitted"  # fold expiry 2, resubmit b2 (expires)
    # Third consecutive expiry hits the cap: terminate instead of resubmitting.
    assert _tick().status == "completed"

    assert len(client.submitted) == 3
    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    record = json.loads(audit[0].read_text(encoding="utf-8"))
    assert record["state"] == "ERROR"
    assert "expired 3 times" in json.dumps(record["attempts"])


def test_failed_batch_retries_and_recovers(tmp_path: Path) -> None:
    """A transient whole-batch failure requeues its rows instead of dropping them.

    One failed batch can carry a 40k-row chunk, and the cause can be quota-shaped
    (#84) — so the rows resubmit under the same cap as expiries and succeed when
    the next round goes through.
    """
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}},
        forced_status={0: "failed"},  # the retry (index 1) completes normally
    )

    assert _advance(tmp_path, client).status == "submitted"
    result = _advance(tmp_path, client)  # folds the failure, resubmits
    assert result.status == "submitted"
    for _ in range(4):
        result = _advance(tmp_path, client)
        if result.status == "completed":
            break
    assert result.status == "completed"
    assert not read_dataset(mentions_root(tmp_path)).empty


def test_persistently_failed_batch_terminates_at_cap(tmp_path: Path) -> None:
    """Rows whose batches keep failing terminate at the resubmission cap."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER}},
        forced_status={0: "failed", 1: "failed", 2: "failed"},
    )

    statuses = [_advance(tmp_path, client).status for _ in range(4)]
    assert statuses[-1] == "completed"
    assert read_dataset(mentions_root(tmp_path)).empty
    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    record = json.loads(audit[0].read_text(encoding="utf-8"))
    assert record["state"] == "ERROR"
    assert "OpenAI batch failed" in json.dumps(record["attempts"])


def test_cancelled_batch_requeues_rows(tmp_path: Path) -> None:
    """Cancelling a batch means "stop", not "abandon these rows" (#84)."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}},
        forced_status={0: "cancelled"},
    )

    assert _advance(tmp_path, client).status == "submitted"
    result = _advance(tmp_path, client)
    assert result.status == "submitted"  # requeued and resubmitted, not dropped
    for _ in range(4):
        result = _advance(tmp_path, client)
        if result.status == "completed":
            break
    assert result.status == "completed"
    assert not read_dataset(mentions_root(tmp_path)).empty


def test_failed_batch_records_per_request_diagnostics(tmp_path: Path) -> None:
    """A failed batch's error file is still read, so rows get the real reason."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER}},
        forced_status={0: "failed"},
        fatal_request_errors={0: {"item-multi": "context_length_exceeded"}},
    )

    assert _advance(tmp_path, client).status == "submitted"
    # A row WITH a per-request error line terminates on that concrete reason
    # immediately; only line-less rows get resubmission rounds.
    assert _advance(tmp_path, client).status == "completed"

    assert "batch-0-err" in client.downloaded
    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    record = json.loads(audit[0].read_text(encoding="utf-8"))
    assert record["state"] == "ERROR"
    assert "context_length_exceeded" in json.dumps(record["attempts"])


def test_failed_batch_falls_back_to_batch_level_errors(tmp_path: Path) -> None:
    """With no error file, the batch-level errors land in the row's audit record."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    errors = ["invalid_json_line: could not parse (input line 1)"]
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER}},
        forced_status={0: "failed", 1: "failed", 2: "failed"},
        batch_errors={0: errors, 1: errors, 2: errors},
    )

    statuses = [_advance(tmp_path, client).status for _ in range(4)]
    assert statuses[-1] == "completed"

    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    record = json.loads(audit[0].read_text(encoding="utf-8"))
    attempts = json.dumps(record["attempts"])
    assert record["state"] == "ERROR"
    assert "invalid_json_line" in attempts
    assert "input line 1" in attempts


def test_unreadable_diagnostics_file_does_not_strand_rows(tmp_path: Path) -> None:
    """A download failure degrades to the generic reason instead of raising."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])

    class BrokenDownloadClient(FakeBatchClient):
        def download_file(self, file_id: str) -> str:
            if file_id.endswith("-err"):
                raise RuntimeError("403 while fetching error file")
            return super().download_file(file_id)

    errs = {"item-multi": "context_length_exceeded"}
    client = BrokenDownloadClient(
        {"item-multi": {"ner": MULTI_NER}},
        forced_status={0: "failed", 1: "failed", 2: "failed"},
        fatal_request_errors={0: errs, 1: errs, 2: errs},
    )

    statuses = [_advance(tmp_path, client).status for _ in range(4)]
    assert statuses[-1] == "completed"

    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    record = json.loads(audit[0].read_text(encoding="utf-8"))
    assert record["state"] == "ERROR"
    assert "OpenAI batch failed" in json.dumps(record["attempts"])


def test_batch_error_messages_reads_sdk_and_dict_shapes() -> None:
    """batch.errors is flattened from either an SDK object or a plain dict."""
    assert _batch_error_messages(None) == []
    assert _batch_error_messages(
        {"data": [{"code": "invalid_url", "message": "bad endpoint", "line": 7}]}
    ) == ["invalid_url: bad endpoint (input line 7)"]
    sdk_like = SimpleNamespace(
        data=[SimpleNamespace(code=None, message="no code here", line=None)]
    )
    assert _batch_error_messages(sdk_like) == ["no code here"]


def test_failed_submit_does_not_refold_previous_batch(tmp_path: Path) -> None:
    """A submit failure after folding must not replay the folded batch.

    Regression test for the crash-between-fold-and-submit window: the folded
    batch must be durably removed from batches.json before submitting, and the
    row's advanced stage must survive the failed tick without a burned attempt.
    """
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FlakySubmitClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}},
        # Call 0 = the NER submit; call 1 = the instrument_ie submit right
        # after the NER batch is folded.
        fail_on_calls={1},
    )

    assert _advance(tmp_path, client).status == "submitted"
    with pytest.raises(RuntimeError, match="simulated submit failure"):
        _advance(tmp_path, client)

    # Recovery tick: no re-fold of the terminal NER batch, just the IE submit.
    assert _advance(tmp_path, client).status == "submitted"
    assert _advance(tmp_path, client).status == "completed"

    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]
    audit_paths = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit_paths) == 1
    record = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    assert record["state"] == "SUCCESS"
    # Exactly one clean attempt per stage: a re-fold would have burned an
    # instrument_ie attempt on the stale NER response.
    assert [(a["stage_name"], a["status"]) for a in record["attempts"]] == [
        ("ner", "SUCCESS"),
        ("instrument_ie", "SUCCESS"),
    ]


def test_stale_batch_result_is_ignored() -> None:
    """A terminal batch a row is no longer pending on folds as a no-op."""
    client = FakeBatchClient({"item-1": {"ner": MULTI_NER}})
    row_state = ExtractionRowState(
        item_row=_item_row("item-1", MULTI_TEXT), stage_name="ner"
    )
    initial_messages(row_state)
    batch_id = client.submit(
        [
            {
                "custom_id": "item-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4",
                    "messages": row_state.current_attempt.messages,
                },
            }
        ],
        metadata={"job_id": "J", "tick": "0"},
    )
    # The row already folded this batch and advanced to instrument_ie.
    handle_response(row_state, MULTI_NER, max_attempts=3)
    assert row_state.current_attempt.stage_name == "instrument_ie"
    entry = RowEntry(row_state=row_state, date="2024-01-02", shard="0001")
    job = JobState(
        job_id="J",
        model="gpt-5.4",
        reasoning_effort="none",
        max_attempts=3,
        claimed_partitions=[],
        rows={"item-1": entry},
        batches=[BatchRecord(batch_id=batch_id, tick=0, custom_ids=["item-1"])],
        all_batch_ids=[batch_id],
    )

    folded = _fold_completed_batches(job, client)

    assert folded == 0
    assert job.batches == []  # stale record pruned
    assert row_state.state is None
    assert row_state.current_attempt.stage_name == "instrument_ie"
    assert row_state.current_attempt.attempt_index == 0  # no attempt burned


def test_state_jsonl_round_trip_preserves_pending() -> None:
    """Pending survives serialization; legacy lines without it load as None."""
    row_state = ExtractionRowState(
        item_row=_item_row("item-1", MULTI_TEXT), stage_name="ner"
    )
    initial_messages(row_state)
    pending = {"batch_id": "batch-0", "stage": "ner", "attempt_index": 0}
    entry = RowEntry(
        row_state=row_state, date="2024-01-02", shard="0001", pending=pending
    )

    loaded = _load_state_jsonl(_serialize_state_jsonl({"item-1": entry}))
    assert loaded["item-1"].pending == pending

    legacy_line = json.loads(_serialize_state_jsonl({"item-1": entry}).splitlines()[0])
    del legacy_line["pending"]
    loaded_legacy = _load_state_jsonl(json.dumps(legacy_line) + "\n")
    assert loaded_legacy["item-1"].pending is None


def _ner_request_size(item_id: str, text: str) -> int:
    """Serialized JSONL line size of one NER batch request, as submitted."""
    row_state = ExtractionRowState(item_row=_item_row(item_id, text), stage_name="ner")
    initial_messages(row_state)
    request = {
        "custom_id": item_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": build_request_body(
            row_state.current_attempt.messages,
            model="gpt-5.4",
            reasoning_effort="none",
        ),
    }
    return len(json.dumps(request).encode("utf-8")) + 1


def test_submit_splits_batches_on_byte_budget(tmp_path: Path) -> None:
    """Two requests that don't fit one byte budget go out as two batches."""
    seed_classification(
        tmp_path,
        [
            {"item_id": "item-nodebt", "text": NODEBT_TEXT},
            {"item_id": "item-multi", "text": MULTI_TEXT},
        ],
    )
    client = FakeBatchClient(
        {
            "item-nodebt": {"ner": NODEBT_NER},
            "item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE},
        }
    )
    # Big enough for either single request, too small for both together.
    budget = (
        max(
            _ner_request_size("item-nodebt", NODEBT_TEXT),
            _ner_request_size("item-multi", MULTI_TEXT),
        )
        + 10
    )

    first = advance_extract_job(
        batch_client=client,
        artifact_root=tmp_path,
        model="gpt-5.4",
        reasoning_effort="none",
        max_attempts=3,
        max_batch_bytes=budget,
    )

    assert first.status == "submitted"
    assert first.submitted_batches == 2
    assert [ids for _, ids in client.submitted] == [["item-nodebt"], ["item-multi"]]

    # The job still runs to completion on later ticks (default budget: the
    # instrument_ie request has a different size than the NER one).
    statuses = []
    while (status := _advance(tmp_path, client).status) != "completed":
        statuses.append(status)
        assert len(statuses) < 10
    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]


def test_oversized_request_terminates_row_not_job(tmp_path: Path) -> None:
    """A single request over the budget errors its row; the job still finishes."""
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
    )
    # Roomy enough for every stage's request for item-multi, far too small for
    # item-huge, whose text alone exceeds it.
    budget = _ner_request_size("item-multi", MULTI_TEXT) + 300_000
    seed_classification(
        tmp_path,
        [
            {"item_id": "item-huge", "text": "x" * (2 * budget)},
            {"item_id": "item-multi", "text": MULTI_TEXT},
        ],
    )

    statuses = [
        advance_extract_job(
            batch_client=client,
            artifact_root=tmp_path,
            model="gpt-5.4",
            reasoning_effort="none",
            max_attempts=3,
            max_batch_bytes=budget,
        ).status
        for _ in range(3)
    ]

    assert statuses[-1] == "completed"
    # Only item-multi was ever submitted; item-huge terminated without a batch.
    for _, ids in client.submitted:
        assert "item-huge" not in ids
    audit = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    records = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in audit[0].read_text(encoding="utf-8").splitlines()
    }
    assert records["item-huge"]["state"] == "ERROR"
    assert records["item-multi"]["state"] == "SUCCESS"


def test_reconcile_adopts_orphan_batch(tmp_path: Path) -> None:
    """A batch tagged for the job but never recorded is adopted on the next tick."""
    client = FakeBatchClient({"item-1": {"ner": NODEBT_NER}})
    row_state = ExtractionRowState(
        item_row=_item_row("item-1", NODEBT_TEXT), stage_name="ner"
    )
    initial_messages(row_state)
    orphan_id = client.submit(
        [
            {
                "custom_id": "item-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4",
                    "messages": row_state.current_attempt.messages,
                },
            }
        ],
        metadata={"job_id": "J", "tick": "0"},
    )
    job = JobState(
        job_id="J",
        model="gpt-5.4",
        reasoning_effort="none",
        max_attempts=3,
        claimed_partitions=[],
        rows={"item-1": RowEntry(row_state=row_state, date="2024-01-02", shard="0001")},
    )

    _reconcile_orphans(str(tmp_path), job, client)

    assert [record.batch_id for record in job.batches] == [orphan_id]
    assert job.batches[0].custom_ids == ["item-1"]
    assert orphan_id in job.all_batch_ids
    # pending is re-attached so the adopted batch's results fold normally.
    assert job.rows["item-1"].pending == {
        "batch_id": orphan_id,
        "stage": "ner",
        "attempt_index": 0,
    }


def test_openai_client_bounds_orphan_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listing stops at the created_after cutoff instead of walking all history."""
    cutoff = datetime(2026, 7, 1, tzinfo=UTC)

    def _sdk_batch(batch_id: str, created_at: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=batch_id,
            status="completed",
            created_at=created_at,
            metadata={"job_id": "J"},
            input_file_id=None,
            output_file_id=None,
            error_file_id=None,
        )

    def _batches_newest_first(limit: int) -> object:
        del limit

        def _iterate() -> object:
            yield _sdk_batch("b-new", int(cutoff.timestamp()) + 100)
            yield _sdk_batch("b-old", int(cutoff.timestamp()) - 100)
            pytest.fail("scan iterated past the created_after cutoff")

        return _iterate()

    client = OpenAIBatchClient(api_key="test-key")
    monkeypatch.setattr(
        OpenAIBatchClient,
        "_client",
        lambda self: SimpleNamespace(
            batches=SimpleNamespace(list=_batches_newest_first)
        ),
    )

    found = client.list_job_batches("J", created_after=cutoff)

    assert [status.id for status in found] == ["b-new"]


# --------------------------------------------------------------------------- #
# Small unit helpers
# --------------------------------------------------------------------------- #


def test_extract_batch_response_text_variants() -> None:
    """Success lines return text; error and non-200 lines raise."""
    ok = {
        "custom_id": "x",
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": "hi"}}]},
        },
    }
    assert extract_batch_response_text(ok) == "hi"

    with pytest.raises(RuntimeError):
        extract_batch_response_text({"custom_id": "x", "error": {"message": "boom"}})

    with pytest.raises(RuntimeError):
        extract_batch_response_text(
            {"custom_id": "x", "response": {"status_code": 400, "body": {}}}
        )


def test_configured_default_model_is_usable_by_both_backends() -> None:
    """The default must be an undated id, or every batch request 400s.

    `normalize_batch_model` only strips the provider prefix, so a dated
    OpenRouter alias such as `openai/gpt-5.6-terra-20260709` reaches the OpenAI
    API as `gpt-5.6-terra-20260709` — an id it does not publish, which fails the
    whole batch rather than one row. The live backend needs the prefix present.
    """
    from cdt import settings
    from cdt.extractor.core import is_reasoning_model

    default = settings.DEFAULT_EXTRACTOR_MODEL
    assert default.startswith("openai/"), "live backend needs the OpenRouter prefix"
    native = normalize_batch_model(default)
    assert "/" not in native
    assert not re.search(r"-\d{8}$", native), (
        f"{default!r} carries a date suffix; the OpenAI API publishes this model "
        "undated and rejects the dated id"
    )
    # Reasoning models take reasoning_effort rather than temperature; a default
    # that stopped matching would silently switch sampling behaviour.
    assert is_reasoning_model(default)


def test_build_request_body_reasoning_and_model() -> None:
    """Model prefixes are stripped and reasoning_effort is mapped/validated."""
    assert normalize_batch_model("openai/gpt-5.4") == "gpt-5.4"
    assert normalize_batch_model("gpt-5.4") == "gpt-5.4"

    # "none" is a value gpt-5.4 accepts, so it must reach the API unchanged:
    # sending "minimal" instead earned a 400 on every request (#31), and dropping
    # the field would silently fall back to the model default.
    body = build_request_body(
        [{"role": "user", "content": "x"}],
        model="openai/gpt-5.4",
        reasoning_effort="none",
    )
    assert body == {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "x"}],
        "reasoning_effort": "none",
    }

    body = build_request_body([], model="gpt-5.4", reasoning_effort="medium")
    assert body["reasoning_effort"] == "medium"
    assert "temperature" not in body

    # xhigh is supported natively; mapping it down to high would lose effort.
    body = build_request_body([], model="gpt-5.4", reasoning_effort="xhigh")
    assert body["reasoning_effort"] == "xhigh"

    # minimal is the one OpenRouter level gpt-5.4 rejects, so it maps to low.
    body = build_request_body([], model="gpt-5.4", reasoning_effort="minimal")
    assert body["reasoning_effort"] == "low"

    # A non-reasoning model can honor temperature, so both backends send it.
    body = build_request_body([], model="openai/gpt-4.1-mini", reasoning_effort="")
    assert body["temperature"] == EXTRACTOR_TEMPERATURE
    assert "reasoning_effort" not in body

    with pytest.raises(ValueError, match="reasoning_effort"):
        build_request_body([], model="gpt-5.4", reasoning_effort="ludicrous")


def test_openai_reasoning_vocabulary_matches_the_api() -> None:
    """Pin the accepted efforts to what the API itself reported.

    Verbatim from a real 400 on gpt-5.4: "Supported values are: 'none', 'low',
    'medium', 'high', and 'xhigh'." Every OpenRouter level must resolve into this
    set, or the batch is paid for and then rejected per request (#31).
    """
    assert set(batch_module.OPENAI_REASONING_EFFORTS) == {
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    }
    for effort in REASONING_EFFORTS:
        assert openai_reasoning_effort(effort) in batch_module.OPENAI_REASONING_EFFORTS


def test_sampling_params_match_across_backends() -> None:
    """Both backends resolve temperature from one shared policy."""
    assert sampling_params("openai/gpt-5.4") == {}
    assert sampling_params("gpt-5.4") == {}
    assert sampling_params("o3-mini") == {}
    assert sampling_params("openai/gpt-4.1-mini") == {
        "temperature": EXTRACTOR_TEMPERATURE
    }


def test_unsupported_reasoning_effort_fails_before_job_creation(
    tmp_path: Path,
) -> None:
    """A bad effort raises on the tick, leaving no half-created job behind."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient({"item-multi": {"ner": MULTI_NER}})

    with pytest.raises(ValueError, match="reasoning_effort"):
        advance_extract_job(
            batch_client=client,
            artifact_root=tmp_path,
            model="gpt-5.4",
            reasoning_effort="ludicrous",
            max_attempts=3,
        )

    assert not client.submitted
    assert not Path(active_job_path(str(tmp_path))).exists()


def test_force_warns_when_a_job_is_already_active(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """Force only applies at job creation, so a mid-job force must not go silent."""
    propagate_logger(batch_module.LOGGER)
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
    )

    # First tick creates the job; force is honored here and must stay quiet.
    with caplog.at_level("WARNING"):
        created = advance_extract_job(
            batch_client=client,
            artifact_root=tmp_path,
            model="gpt-5.4",
            reasoning_effort="none",
            max_attempts=3,
            force=True,
        )
    assert created.status == "submitted"
    assert "Ignoring force=True" not in caplog.text

    # Second tick has an active job, so force is inert and must warn.
    caplog.clear()
    with caplog.at_level("WARNING"):
        advance_extract_job(
            batch_client=client,
            artifact_root=tmp_path,
            model="gpt-5.4",
            reasoning_effort="none",
            max_attempts=3,
            force=True,
        )
    assert "Ignoring force=True" in caplog.text
    assert created.job_id in caplog.text


# --------------------------------------------------------------------------- #
# Corrupt/missing job state recovery
# --------------------------------------------------------------------------- #


def _seed_active_job(tmp_path: Path, client: FakeBatchClient) -> str:
    """Create a job with one in-flight batch and return its job id."""
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    result = advance_extract_job(
        batch_client=client,
        artifact_root=tmp_path,
        model="gpt-5.4",
        reasoning_effort="none",
        max_attempts=3,
    )
    assert result.status == "submitted"
    assert result.job_id is not None
    return result.job_id


def test_missing_job_directory_self_heals(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """A deleted job directory clears the marker instead of wedging every tick."""
    propagate_logger(batch_module.LOGGER)
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
    )
    job_id = _seed_active_job(tmp_path, client)

    shutil.rmtree(tmp_path / "extract-batches" / f"job_id={job_id}")

    with caplog.at_level("ERROR"):
        reset_tick = _advance(tmp_path, client)
    assert reset_tick.status == "reset"
    assert reset_tick.job_id == job_id
    assert "Clearing the active extract job marker" in caplog.text

    # The very next tick makes progress again on the same partitions.
    assert _advance(tmp_path, client).status == "submitted"
    while (status := _advance(tmp_path, client).status) != "completed":
        assert status in {"submitted", "waiting"}
    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]


def test_truncated_manifest_self_heals(tmp_path: Path) -> None:
    """A partially written manifest is treated as corrupt, not a crash."""
    client = FakeBatchClient({"item-multi": {"ner": MULTI_NER}})
    job_id = _seed_active_job(tmp_path, client)

    manifest = tmp_path / "extract-batches" / f"job_id={job_id}" / "manifest.json"
    # Valid JSON, but missing every key _load_job_state indexes.
    manifest.write_text('{"job_id": "x"}', encoding="utf-8")

    assert _advance(tmp_path, client).status == "reset"


def test_unparseable_state_jsonl_self_heals(tmp_path: Path) -> None:
    """Garbage in state.jsonl is recoverable rather than permanent."""
    client = FakeBatchClient({"item-multi": {"ner": MULTI_NER}})
    job_id = _seed_active_job(tmp_path, client)

    state = tmp_path / "extract-batches" / f"job_id={job_id}" / "state.jsonl.gz"
    state.write_text("{not json at all\n", encoding="utf-8")

    assert _advance(tmp_path, client).status == "reset"


def test_describe_active_job_reports_idle_active_and_corrupt(tmp_path: Path) -> None:
    """The read-only summary covers all three states without mutating anything."""
    assert describe_active_job(tmp_path).status == "idle"

    client = FakeBatchClient({"item-multi": {"ner": MULTI_NER}})
    job_id = _seed_active_job(tmp_path, client)

    summary = describe_active_job(tmp_path)
    assert summary.status == "active"
    assert summary.job_id == job_id
    assert summary.total_rows == 1
    assert summary.in_flight_batches == 1
    assert summary.claimed_partitions == 1

    shutil.rmtree(tmp_path / "extract-batches" / f"job_id={job_id}")
    corrupt = describe_active_job(tmp_path)
    assert corrupt.status == "corrupt"
    assert corrupt.job_id == job_id
    assert "missing" in corrupt.detail
    # Reporting must not have cleared the marker.
    assert read_json_artifact(active_job_path(str(tmp_path)))["job_id"] == job_id


def test_reset_active_job_clears_marker_and_keeps_state(tmp_path: Path) -> None:
    """The admin reset frees the next tick but leaves the job dir for inspection."""
    client = FakeBatchClient({"item-multi": {"ner": MULTI_NER}})
    job_id = _seed_active_job(tmp_path, client)

    assert reset_active_job(tmp_path) == job_id
    assert (
        tmp_path / "extract-batches" / f"job_id={job_id}" / "state.jsonl.gz"
    ).exists()
    assert describe_active_job(tmp_path).status == "idle"
    assert reset_active_job(tmp_path) is None


def test_batch_failures_are_recorded_in_the_failure_registry(tmp_path: Path) -> None:
    """Rows a fatal batch drops are recorded as a work-list, not just audited."""
    seed_classification(
        tmp_path,
        [
            {"item_id": "item-ok", "text": MULTI_TEXT},
            {"item_id": "item-bad", "text": MULTI_TEXT},
        ],
    )
    client = FakeBatchClient(
        {
            "item-ok": {"ner": MULTI_NER, "instrument_ie": MULTI_IE},
            "item-bad": {"ner": ("error", "content filtered")},
        }
    )

    while (status := _advance(tmp_path, client).status) != "completed":
        assert status in {"submitted", "waiting"}

    failures = load_row_failures("extract", artifact_root=tmp_path)
    assert list(failures) == ["item-bad"]
    entry = failures["item-bad"]
    assert entry["backend"] == "batch"
    assert entry["state"] == "ERROR"
    assert "content filtered" in str(entry["error"])
    assert entry["date"] == "2024-01-02"
    assert entry["shard"] == "0001"
    # The successful row is absent, and its mentions still landed.
    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]


def test_failure_registry_survives_across_jobs(tmp_path: Path) -> None:
    """A second job's failures accumulate rather than replacing the first's."""
    seed_classification(
        tmp_path, [{"item_id": "item-a", "text": MULTI_TEXT}], shard="0001"
    )
    client = FakeBatchClient({"item-a": {"ner": ("error", "first failure")}})
    while _advance(tmp_path, client).status != "completed":
        pass

    seed_classification(
        tmp_path, [{"item_id": "item-b", "text": MULTI_TEXT}], shard="0002"
    )
    client = FakeBatchClient({"item-b": {"ner": ("error", "second failure")}})
    while _advance(tmp_path, client).status != "completed":
        pass

    failures = load_row_failures("extract", artifact_root=tmp_path)
    assert sorted(failures) == ["item-a", "item-b"]
    assert failures["item-a"]["shard"] == "0001"
    assert failures["item-b"]["shard"] == "0002"
    assert failures["item-a"]["run_id"] != failures["item-b"]["run_id"]


def test_reasoning_effort_omitted_for_non_reasoning_models() -> None:
    """A configured effort must not reach a model family that 400s on it."""
    body = build_request_body(
        [{"role": "user", "content": "x"}],
        model="openai/gpt-4.1-mini",
        reasoning_effort="none",
    )
    assert "reasoning_effort" not in body
    assert body["temperature"] == EXTRACTOR_TEMPERATURE


def test_corrupt_active_marker_self_heals(tmp_path: Path) -> None:
    """A truncated active.json is cleared, not a crash-loop on every tick."""
    marker = Path(active_job_path(str(tmp_path)))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"job_id": "2026')  # killed mid-write

    assert describe_active_job(tmp_path).status == "corrupt"

    result = advance_extract_job(
        batch_client=FakeBatchClient({}), artifact_root=tmp_path
    )
    assert result.status == "reset"
    # Marker is cleared; the next tick proceeds normally (idle: nothing pending).
    assert (
        advance_extract_job(
            batch_client=FakeBatchClient({}), artifact_root=tmp_path
        ).status
        == "idle"
    )


def test_reset_active_job_pins_to_the_inspected_job(tmp_path: Path) -> None:
    """A reset targeting a job that is no longer active must not clear the new one."""
    client = FakeBatchClient({"item-multi": {"ner": MULTI_NER}})
    job_id = _seed_active_job(tmp_path, client)

    assert reset_active_job(tmp_path, expected_job_id="some-older-job") is None
    assert read_json_artifact(active_job_path(str(tmp_path)))["job_id"] == job_id

    assert reset_active_job(tmp_path, expected_job_id=job_id) == job_id
    assert describe_active_job(tmp_path).status == "idle"


def test_stranded_pending_rows_requeue_instead_of_finalizing(tmp_path: Path) -> None:
    """Rows pending on an unrecorded, unadopted batch re-queue, not fail-finalize.

    Models the crash between _save_state (pending markers written) and
    _save_batches, combined with an orphan-listing gap: the row is neither
    awaiting nor terminal and no batch is in flight.
    """
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
    )
    job_id = _seed_active_job(tmp_path, client)
    batches_path = tmp_path / "extract-batches" / f"job_id={job_id}" / "batches.json"
    payload = json.loads(batches_path.read_text())
    payload["batches"] = []  # the submit was never recorded...
    batches_path.write_text(json.dumps(payload))
    client.batches.clear()  # ...and the orphan scan cannot see it

    requeued = advance_extract_job(batch_client=client, artifact_root=tmp_path)
    # The stranded row was re-queued and re-submitted on this or the next tick,
    # and the job must NOT have finalized with the row recorded as a failure.
    assert requeued.status != "completed"
    assert describe_active_job(tmp_path).status == "active"

    # Let the job run to completion: results now fold normally.
    for _ in range(4):
        result = advance_extract_job(batch_client=client, artifact_root=tmp_path)
        if result.status == "completed":
            break
    assert result.status == "completed"
    failures = load_row_failures("extract", artifact_root=tmp_path)
    assert failures == {}


def test_orphan_input_download_failure_skips_adoption(tmp_path: Path) -> None:
    """A purged orphan input file must not crash the tick before state is saved."""

    class PurgedInputClient(FakeBatchClient):
        def download_file(self, file_id: str) -> str:
            if file_id.endswith("-in"):
                raise RuntimeError("file purged by retention")
            return super().download_file(file_id)

    client = PurgedInputClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
    )
    job_id = _seed_active_job(tmp_path, client)
    batches_path = tmp_path / "extract-batches" / f"job_id={job_id}" / "batches.json"
    payload = json.loads(batches_path.read_text())
    payload["batches"] = []  # unrecorded submit: the batch is now an orphan
    batches_path.write_text(json.dumps(payload))

    # The orphan is visible to list_job_batches but its input download fails:
    # the tick must survive, skip adoption, and requeue the stranded row.
    result = advance_extract_job(batch_client=client, artifact_root=tmp_path)
    assert result.status in {"waiting", "submitted"}
    assert describe_active_job(tmp_path).status == "active"


def test_legacy_uncompressed_state_still_loads(tmp_path: Path) -> None:
    """A job started before #86 (plain state.jsonl) keeps advancing after deploy."""
    client = FakeBatchClient(
        {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
    )
    job_id = _seed_active_job(tmp_path, client)
    job_dir = tmp_path / "extract-batches" / f"job_id={job_id}"
    gz = job_dir / "state.jsonl.gz"
    import gzip as _gzip

    (job_dir / "state.jsonl").write_bytes(_gzip.decompress(gz.read_bytes()))
    gz.unlink()

    for _ in range(4):
        result = advance_extract_job(batch_client=client, artifact_root=tmp_path)
        if result.status == "completed":
            break
    assert result.status == "completed"
    assert not read_dataset(mentions_root(tmp_path)).empty


def test_batch_job_pays_only_for_new_rows_after_partition_grows(
    tmp_path: Path,
) -> None:
    """A grown classification partition yields a job over only the new rows (#62).

    The first job's mentions survive the second job's merge, and completion is
    recorded per row outcome rather than per partition visit (#49).
    """
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])
    client = FakeBatchClient(
        {
            "item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE},
            "item-nodebt": {"ner": NODEBT_NER},
        }
    )
    for _ in range(4):
        if _advance(tmp_path, client).status == "completed":
            break
    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]
    first_job_rows = {cid for _, cids in client.submitted for cid in cids}
    assert first_job_rows == {"item-multi"}

    # Ingest-style growth: the same partition object gains a second relevant row.
    seed_classification(
        tmp_path,
        [
            {"item_id": "item-multi", "text": MULTI_TEXT},
            {"item_id": "item-nodebt", "text": NODEBT_TEXT},
        ],
    )
    for _ in range(4):
        result = _advance(tmp_path, client)
        if result.status == "completed":
            break
    assert result.status == "completed"

    second_job_rows = {
        cid for _, cids in client.submitted for cid in cids
    } - first_job_rows
    assert second_job_rows == {"item-nodebt"}
    # The first job's mentions survived the second job's finalize.
    assert read_dataset(mentions_root(tmp_path))["name"].to_list() == ["Term Loan"]


def test_per_request_server_error_requeues_instead_of_terminating(
    tmp_path: Path,
) -> None:
    """A 500 on one request is infrastructure, not a verdict (#49, seen live).

    The scripted client returns a 500 line on the first NER round; the row must
    requeue and succeed on the next round rather than land in the failure
    registry.
    """

    class FlakyOnceClient(FakeBatchClient):
        def __init__(self) -> None:
            super().__init__(
                {"item-multi": {"ner": MULTI_NER, "instrument_ie": MULTI_IE}}
            )
            self._first = True

        def submit(
            self, requests: list[dict[str, object]], *, metadata: dict[str, str]
        ) -> str:
            if self._first:
                self._first = False
                self.scripts["item-multi"] = {
                    "ner": ("status", 500),
                    "instrument_ie": MULTI_IE,
                }
                batch_id = super().submit(requests, metadata=metadata)
                self.scripts["item-multi"] = {
                    "ner": MULTI_NER,
                    "instrument_ie": MULTI_IE,
                }
                return batch_id
            return super().submit(requests, metadata=metadata)

    client = FlakyOnceClient()
    seed_classification(tmp_path, [{"item_id": "item-multi", "text": MULTI_TEXT}])

    for _ in range(6):
        result = _advance(tmp_path, client)
        if result.status == "completed":
            break
    assert result.status == "completed"
    assert not read_dataset(mentions_root(tmp_path)).empty
    assert load_row_failures("extract", artifact_root=tmp_path) == {}


def test_stall_warning_fires_only_past_the_tick_threshold(
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """The stall literal appears once a job outlives STALL_WARNING_TICKS (#85)."""
    from cdt.extractor.batch import LOGGER as batch_logger
    from cdt.extractor.batch import (
        STALL_WARNING_TICKS,
        JobState,
        _warn_if_stalled,
    )

    propagate_logger(batch_logger)
    job = JobState(
        job_id="J",
        model="m",
        reasoning_effort="low",
        max_attempts=3,
        claimed_partitions=[],
        rows={},
    )

    job.tick = STALL_WARNING_TICKS - 1
    with caplog.at_level("WARNING"):
        _warn_if_stalled(job, terminal_rows=0)
    assert not caplog.records

    job.tick = STALL_WARNING_TICKS
    with caplog.at_level("WARNING"):
        _warn_if_stalled(job, terminal_rows=0)
    assert any("Extract job stalled" in record.message for record in caplog.records)


def test_collect_pending_extract_items_caps_claimed_rows(tmp_path: Path) -> None:
    """Claiming stops at max_rows; unclaimed partitions stay pending (#92)."""
    from cdt.extractor.core import collect_pending_extract_items

    for index, (date_value, shard) in enumerate(
        [("2024-01-02", "0001"), ("2024-01-03", "0002"), ("2024-01-04", "0003")],
        start=1,
    ):
        seed_classification(
            tmp_path,
            [{"item_id": f"item-{index}", "text": f"text {index}"}],
            date=date_value,
            shard=shard,
        )

    entries, claimed = collect_pending_extract_items(artifact_root=tmp_path, max_rows=1)

    assert len(entries) == 1
    assert len(claimed) == 1
    uncapped_entries, uncapped_claimed = collect_pending_extract_items(
        artifact_root=tmp_path
    )
    assert len(uncapped_entries) == 3
    assert len(uncapped_claimed) == 3


def test_live_client_sends_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every live chat call carries a client-side timeout (#93)."""
    import openrouter as openrouter_module

    from cdt.extractor.core import (
        LIVE_REQUEST_TIMEOUT_SECONDS,
        OpenRouterChatClient,
    )

    captured: dict[str, object] = {}

    class FakeOpenRouter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            raise RuntimeError("constructed")

    monkeypatch.setattr(openrouter_module, "OpenRouter", FakeOpenRouter)
    client = OpenRouterChatClient(api_key="k")

    with pytest.raises(RuntimeError, match="constructed"):
        asyncio.run(client.complete(messages=[], model="m", reasoning_effort=""))

    assert captured["timeout_ms"] == LIVE_REQUEST_TIMEOUT_SECONDS * 1000
