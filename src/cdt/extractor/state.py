"""Per-row extraction state, attempt records, and provider-failure classification."""

# ruff: noqa: ANN101, ANN102, D102, D105, D107

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, cast

import httpx
import pandas as pd

from cdt.extractor.prior_state import published_mention_rows

# One attempt budget for every stage (`--max-attempts`). See
# docs/decisions/extraction.md before raising it.
DEFAULT_MAX_ATTEMPTS = 3
# Resends of a `content_filter` abort per stage call. The abort is classified
# by the callers, before `handle_response`, so it never becomes a scored
# attempt; the cap bounds the cost if the abort turns out to be billed.
# `max_tokens` is deliberately unset. See docs/decisions/extraction.md.
MAX_CONTENT_FILTER_RESENDS = 6
# Status for an attempt the provider aborted: a call was made, but it returned
# no answer to score. Distinct from "FAILED", which means the model answered
# and the answer was rejected -- the difference every cross-attempt check needs.
ABORTED_ATTEMPT_STATUS = "ABORTED"
# PARTIAL rows publish their mentions like SUCCESS but also keep a failure
# registry entry recording what salvage dropped.
PUBLISHABLE_ROW_STATES = frozenset({"SUCCESS", "PARTIAL"})


class InfrastructureError(RuntimeError):
    """A provider/transport failure that says nothing about the row's content.

    Billing (402), throttling (429), timeouts, connection resets, and 5xx are
    properties of the run environment, not of the filing being extracted: one
    occurrence predicts thousands more, so the live driver aborts the run at the
    first one instead of burning retries and terminating rows that never got a
    real verdict.
    """


# HTTP statuses that indicate the provider, not the content (529: OpenRouter's
# provider-overloaded status).
_INFRASTRUCTURE_STATUSES = frozenset({402, 408, 429, 500, 502, 503, 504, 529})


def is_infrastructure_status(status: object) -> bool:
    """Classify an HTTP status from a batch result line as infrastructure."""
    return isinstance(status, int) and status in _INFRASTRUCTURE_STATUSES


def is_infrastructure_error(exc: BaseException) -> bool:
    """Classify an exception from a chat call as infrastructure vs content."""
    if isinstance(exc, InfrastructureError):
        return True
    # The openrouter SDK re-raises httpx transport errors (connect, read,
    # protocol) unwrapped once its own retries run out.
    if isinstance(exc, ConnectionError | TimeoutError | httpx.TransportError):
        return True
    for attribute in ("status_code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and value in _INFRASTRUCTURE_STATUSES:
            return True
    # Provider SDKs name their billing/rate/transport errors without exposing a
    # status (openai's APIConnectionError and APITimeoutError among them).
    name = type(exc).__name__.casefold()
    return any(
        marker in name
        for marker in (
            "paymentrequired",
            "ratelimit",
            "serviceunavailable",
            "connectionerror",
            "timeout",
        )
    )


class SupportsChatCompletion(Protocol):
    """Protocol for chat-capable extractor clients."""

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> CompletionResult:
        """Return one chat completion with its provider metadata."""


@dataclass
class AttemptRecord:
    """Recorded data for one stage attempt."""

    stage_name: str
    attempt_index: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    response: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    status: str = "incomplete"
    # Provider metadata. `finish_reason` separates a provider abort from a
    # response the model chose to end; they need opposite remedies.
    finish_reason: str | None = None
    refusal: str | None = None
    usage: dict[str, object] | None = None
    response_id: str | None = None
    served_model: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert one attempt to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CompletionResult:
    """One model response plus the provider metadata that explains it."""

    text: str
    finish_reason: str | None = None
    refusal: str | None = None
    usage: dict[str, object] | None = None
    response_id: str | None = None
    served_model: str | None = None


@dataclass
class ExtractionRowState:
    """Mutable row-level extractor state."""

    item_row: dict[str, object]
    stage_name: str
    all_attempts: list[AttemptRecord] = field(default_factory=list)
    stage_responses: dict[str, str] = field(default_factory=dict)
    debt_instrument_mentions: list[dict[str, object]] = field(default_factory=list)
    ner_tagged_xml: str | None = None
    state: str | None = None
    salvage_notes: list[str] = field(default_factory=list)
    current_attempt: AttemptRecord = field(init=False)

    def __post_init__(self) -> None:
        self.current_attempt = AttemptRecord(stage_name=self.stage_name)

    @property
    def item_id(self) -> str:
        """Return the item identifier."""
        return str(self.item_row["item_id"])

    @property
    def text(self) -> str:
        """Return the item text."""
        return str(self.item_row.get("text", ""))

    def add_messages(self, messages: list[dict[str, str]]) -> None:
        """Append prompt messages to the current attempt."""
        self.current_attempt.messages.extend(messages)

    def add_response(
        self, response: str, completion: CompletionResult | None = None
    ) -> None:
        """Record the latest model response and the provider metadata for it."""
        self.current_attempt.response = response
        self.current_attempt.attempt_index += 1
        self.stage_responses[self.current_attempt.stage_name] = response
        if completion is not None:
            self.current_attempt.finish_reason = completion.finish_reason
            self.current_attempt.refusal = completion.refusal
            self.current_attempt.usage = completion.usage
            self.current_attempt.response_id = completion.response_id
            self.current_attempt.served_model = completion.served_model

    def add_validation(self, failures: list[str]) -> None:
        """Record validation output for the current attempt."""
        self.current_attempt.validation_errors = failures
        self.current_attempt.status = "FAILED" if failures else "SUCCESS"

    def retry(self, retry_message: str) -> None:
        """Prepare a retry attempt for the current stage."""
        self.all_attempts.append(self.current_attempt)
        new_messages = list(self.current_attempt.messages)
        if self.current_attempt.response is not None:
            new_messages.append(
                {"role": "assistant", "content": self.current_attempt.response}
            )
        new_messages.append({"role": "user", "content": retry_message})
        self.current_attempt = AttemptRecord(
            stage_name=self.current_attempt.stage_name,
            attempt_index=self.current_attempt.attempt_index,
            messages=new_messages,
        )

    def record_unbilled_abort(
        self, response: str, completion: CompletionResult
    ) -> None:
        """Record a call the provider aborted, without scoring it.

        Appends an ``ABORTED`` record (response and provider metadata, empty
        ``messages``) to ``all_attempts`` and leaves ``current_attempt``
        untouched, so the caller resends the identical request and the next
        real answer is scored as the attempt it is. The appended records are
        what the resend cap counts, so the count survives a batch resume.
        """
        self.all_attempts.append(
            AttemptRecord(
                stage_name=self.current_attempt.stage_name,
                attempt_index=self.current_attempt.attempt_index,
                response=response,
                status=ABORTED_ATTEMPT_STATUS,
                finish_reason=completion.finish_reason,
                refusal=completion.refusal,
                usage=completion.usage,
                response_id=completion.response_id,
                served_model=completion.served_model,
            )
        )

    def finish(self, state: str) -> None:
        """Finish processing for this row.

        A row that reached the end only because a terminal failure was salvaged
        finishes PARTIAL rather than SUCCESS: its mentions publish, but the
        failure registry keeps a record of what was lost.
        """
        self.all_attempts.append(self.current_attempt)
        if state == "SUCCESS" and self.salvage_notes:
            state = "PARTIAL"
        self.state = state

    def next_stage(self, stage_name: str) -> None:
        """Advance to the next stage."""
        self.all_attempts.append(self.current_attempt)
        self.current_attempt = AttemptRecord(stage_name=stage_name)

    def to_audit_dict(self) -> dict[str, object]:
        """Return one audit record for full.jsonl."""
        attempts = [attempt.to_dict() for attempt in self.all_attempts]
        return {
            "item_id": self.item_id,
            "accession_number": self.item_row.get("accession_number"),
            "item": self.item_row.get("item"),
            "stage_responses": self.stage_responses,
            # What the item publishes, not only what the model returned, so the
            # audit log shows every row a reader will find in `mentions`.
            "debt_instrument_mentions": published_mention_rows(self),
            "state": self.state,
            "salvage_notes": self.salvage_notes,
            "attempts": attempts,
        }

    def to_state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for resumable batch extraction.

        Unlike ``to_audit_dict`` this preserves everything the resumable state
        machine needs to continue after a process exit: the in-flight
        ``current_attempt`` (which ``__post_init__`` rebuilds blank), the current
        stage, ``ner_tagged_xml``, partial mentions, and a native-typed copy of
        the item fields the stages consume.
        """
        return {
            "item_row": {
                key: coerce_native(self.item_row.get(key))
                for key in STATE_ITEM_ROW_FIELDS
            },
            "all_attempts": [attempt.to_dict() for attempt in self.all_attempts],
            "stage_responses": dict(self.stage_responses),
            "debt_instrument_mentions": self.debt_instrument_mentions,
            "ner_tagged_xml": self.ner_tagged_xml,
            "state": self.state,
            "salvage_notes": self.salvage_notes,
            "current_attempt": self.current_attempt.to_dict(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> ExtractionRowState:
        """Rebuild a row state previously produced by ``to_state_dict``."""
        current_attempt = AttemptRecord(
            **cast(dict[str, Any], payload["current_attempt"])
        )
        row_state = cls(
            item_row=cast(dict[str, object], payload["item_row"]),
            stage_name=current_attempt.stage_name,
        )
        row_state.all_attempts = [
            AttemptRecord(**cast(dict[str, Any], attempt))
            for attempt in cast(list[dict[str, object]], payload["all_attempts"])
        ]
        row_state.stage_responses = cast(dict[str, str], payload["stage_responses"])
        row_state.debt_instrument_mentions = cast(
            list[dict[str, object]], payload["debt_instrument_mentions"]
        )
        row_state.ner_tagged_xml = cast(str | None, payload["ner_tagged_xml"])
        row_state.state = cast(str | None, payload["state"])
        row_state.salvage_notes = cast(list[str], payload["salvage_notes"])
        row_state.current_attempt = current_attempt
        return row_state


class StageSpec(Protocol):
    """Minimal stage interface for the local extractor workflow."""

    name: str

    def preprocess(self, row_state: ExtractionRowState) -> list[dict[str, str]]:
        """Build messages for the LLM."""

    def validate(self, row_state: ExtractionRowState, response: str) -> list[str]:
        """Validate raw stage output."""

    def postprocess(self, row_state: ExtractionRowState) -> None:
        """Mutate row state after validation succeeds."""

    def early_stop(self, row_state: ExtractionRowState) -> bool:
        """Return whether the row should finish early."""

    def build_retry_message(self, failures: list[str]) -> str:
        """Build retry guidance after a validation failure."""


# Only the item fields the stages actually read are persisted in batch job state.
STATE_ITEM_ROW_FIELDS = (
    "item_id",
    "text",
    "accession_number",
    "cik",
    "company_name",
    "date",
    "item",
)


def coerce_native(value: object) -> str | None:
    """Coerce one pandas/numpy scalar to a JSON-safe native string or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
