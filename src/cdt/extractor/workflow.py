"""Advance one row through the stages: apply responses, retry, resend aborts, salvage or fail."""

from __future__ import annotations

from pathlib import Path

from cdt.datasets import (
    load_row_failures,
    save_row_failures,
)
from cdt.extractor.llm import OpenRouterChatClient, is_content_filter_abort
from cdt.extractor.stages import (
    EXTRACTOR_STAGES,
    STAGE_BY_NAME,
    STAGE_INDEX,
    InstrumentIEStage,
    InstrumentRelationStage,
    salvage_instrument_ie_entries,
)
from cdt.extractor.state import (
    ABORTED_ATTEMPT_STATUS,
    MAX_CONTENT_FILTER_RESENDS,
    CompletionResult,
    ExtractionRowState,
    InfrastructureError,
    StageSpec,
    SupportsChatCompletion,
    is_infrastructure_error,
)
from cdt.shared import get_logger

LOGGER = get_logger(__name__)


def record_stage_error(row_state: ExtractionRowState, message: str) -> None:
    """Mark the current attempt as a terminal ERROR for one row."""
    row_state.current_attempt.validation_errors = [message]
    row_state.current_attempt.status = "ERROR"
    row_state.finish("ERROR")


def _begin_stage(row_state: ExtractionRowState, stage: StageSpec) -> bool:
    """Populate the current attempt's messages via ``stage.preprocess``.

    Returns True on success. On a preprocess exception the row is finished as a
    terminal ERROR (matching the synchronous workflow) and False is returned.
    """
    row_state.current_attempt.stage_name = stage.name
    row_state.current_attempt.messages = []
    try:
        row_state.add_messages(stage.preprocess(row_state))
    except Exception as exc:  # noqa: BLE001
        record_stage_error(row_state, f"{type(exc).__name__}: {exc}")
        return False
    return True


def initial_messages(row_state: ExtractionRowState) -> list[dict[str, str]] | None:
    """Prepare the first request for a fresh row.

    Returns the messages for the first LLM call, or None if the row terminates
    before any call (a preprocess failure on the first stage).
    """
    if not _begin_stage(row_state, EXTRACTOR_STAGES[0]):
        return None
    return list(row_state.current_attempt.messages)


def handle_response(
    row_state: ExtractionRowState,
    response: str,
    *,
    max_attempts: int,
    completion: CompletionResult | None = None,
) -> list[dict[str, str]] | None:
    """Advance one row given the response to its outstanding request.

    Applies the current stage's validate/postprocess, then either advances to the
    next stage, schedules a retry, or terminates the row. Returns the messages
    for the next LLM call, or None when the row has reached a terminal state.

    Every call that reaches here is a scored attempt. Callers must divert
    provider aborts first (`is_content_filter_abort`,
    `ExtractionRowState.record_unbilled_abort`), as they do infrastructure
    errors. `max_attempts` is the per-stage budget, the same for every stage.
    """
    stage = STAGE_BY_NAME[row_state.current_attempt.stage_name]
    stage_index = STAGE_INDEX[stage.name]
    row_state.add_response(response, completion)
    failures = stage.validate(row_state, response)
    row_state.add_validation(failures)
    if not failures:
        stage.postprocess(row_state)
        return _advance_after_stage(row_state, stage, stage_index)

    if row_state.current_attempt.attempt_index >= max_attempts:
        return _salvage_or_fail(row_state, stage, stage_index, max_attempts)
    row_state.retry(stage.build_retry_message(failures))
    return list(row_state.current_attempt.messages)


def count_content_filter_aborts(row_state: ExtractionRowState, stage_name: str) -> int:
    """Count this stage's calls on this row that the provider aborted unscored.

    Read off `all_attempts` rather than a counter held by the caller, because
    the batch backend folds one response per tick: the cap has to survive a
    process exit, and `to_state_dict` already round-trips these records.
    """
    return sum(
        1
        for attempt in row_state.all_attempts
        if attempt.stage_name == stage_name and attempt.status == ABORTED_ATTEMPT_STATUS
    )


def terminate_on_provider_aborts(
    row_state: ExtractionRowState, stage: StageSpec, aborts: int
) -> None:
    """End a row at the resend cap, without scoring the abort against the model.

    Salvages like `_salvage_or_fail`, except that an `instrument_ie` salvage
    finishes here instead of advancing: an `instrument_ie` response whose valid
    entries yield at least one mention, or mentions already held at
    `instrument_relation`, finish PARTIAL without lineage; otherwise FAILED.
    The salvage note records ``aborts`` and, when scored attempts of this stage
    also failed, the most recent validation errors, since the aborts need not
    have been consecutive.
    """
    scored_failures = [
        attempt
        for attempt in row_state.all_attempts
        if attempt.stage_name == stage.name and attempt.status == "FAILED"
    ]
    note = (
        f"{stage.name} aborted by the provider on {aborts} of its calls "
        f"(finish_reason=content_filter)"
    )
    if scored_failures:
        errors = "; ".join(scored_failures[-1].validation_errors) or (
            "no validation errors recorded"
        )
        plural = "s" if len(scored_failures) > 1 else ""
        note += (
            f"; {len(scored_failures)} scored attempt{plural} also failed, most "
            f"recently: {errors}"
        )
    else:
        note += "; no attempt was scored"
    if stage.name == InstrumentIEStage.name:
        # A response rejected as a whole can still hold individually valid
        # entries, and they are in `stage_responses` already -- the aborts came
        # after it, not instead of it. Published without lineage, because the
        # relation stage is where the provider stopped.
        dropped = salvage_instrument_ie_entries(row_state)
        if dropped is not None:
            stage.postprocess(row_state)
            if row_state.debt_instrument_mentions:
                row_state.salvage_notes.append(
                    f"{note}; the entries its last scored answer validated are "
                    "published without lineage relations"
                )
                row_state.finish("PARTIAL")
                return
    if (
        stage.name == InstrumentRelationStage.name
        and row_state.debt_instrument_mentions
    ):
        row_state.salvage_notes.append(
            f"{note}; mentions published without lineage relations"
        )
        row_state.finish("PARTIAL")
        return
    row_state.salvage_notes.append(note)
    row_state.finish("FAILED")


def handle_provider_abort(
    row_state: ExtractionRowState, completion: CompletionResult
) -> bool:
    """Record an unbilled provider abort. True if the request should go back out.

    Shared by both backends so the live loop and the batch fold cannot drift on
    a decision neither of them scores. Returns False when the resend cap is
    reached and the row has been terminated.
    """
    stage = STAGE_BY_NAME[row_state.current_attempt.stage_name]
    resend = (
        count_content_filter_aborts(row_state, stage.name) < MAX_CONTENT_FILTER_RESENDS
    )
    row_state.record_unbilled_abort(completion.text, completion)
    aborts = count_content_filter_aborts(row_state, stage.name)
    LOGGER.warning(
        "Provider aborted item=%s stage=%s abort=%s/%s (finish_reason=%s, "
        "usage=%s); %s",
        row_state.item_id,
        stage.name,
        aborts,
        MAX_CONTENT_FILTER_RESENDS + 1,
        completion.finish_reason,
        completion.usage,
        "re-sending unscored" if resend else "resend cap reached, terminating row",
    )
    if resend:
        return True
    terminate_on_provider_aborts(row_state, stage, aborts)
    return False


def _advance_after_stage(
    row_state: ExtractionRowState,
    stage: StageSpec,
    stage_index: int,
) -> list[dict[str, str]] | None:
    """Move one row past a completed stage: finish it or start the next stage."""
    if stage.early_stop(row_state):
        # A zero-tag NER response that passed validation is the honest "this
        # filing disclosed no debt", retried or not; a give-up the row holds
        # evidence for was already rejected by `NERStage.validate`.
        row_state.finish("SUCCESS")
        return None
    if stage_index == len(EXTRACTOR_STAGES) - 1:
        row_state.finish("SUCCESS")
        return None
    next_stage = EXTRACTOR_STAGES[stage_index + 1]
    if (
        next_stage.name == "instrument_relation"
        and len(row_state.debt_instrument_mentions) <= 1
    ):
        row_state.finish("SUCCESS")
        return None
    row_state.next_stage(next_stage.name)
    if not _begin_stage(row_state, next_stage):
        return None
    return list(row_state.current_attempt.messages)


def _salvage_or_fail(
    row_state: ExtractionRowState,
    stage: StageSpec,
    stage_index: int,
    max_attempts: int,
) -> list[dict[str, str]] | None:
    """Keep what the row's final failed attempt still supports.

    At `instrument_ie`, the individually valid entries are kept and the row
    advances; at `instrument_relation`, mentions already held publish without
    lineage. Either way the row finishes PARTIAL and the failure registry
    records the loss. Otherwise (including NER) the row finishes FAILED.
    Returns the next messages, or None when the row is terminal.
    """
    if stage.name == InstrumentIEStage.name:
        dropped = salvage_instrument_ie_entries(row_state)
        if dropped is not None:
            row_state.salvage_notes.append(
                f"instrument_ie kept the valid entries and dropped {dropped} "
                f"invalid ones after {max_attempts} failed attempts"
            )
            stage.postprocess(row_state)
            return _advance_after_stage(row_state, stage, stage_index)
    if (
        stage.name == InstrumentRelationStage.name
        and row_state.debt_instrument_mentions
    ):
        row_state.salvage_notes.append(
            f"instrument_relation failed after {max_attempts} attempts; "
            "mentions published without lineage relations"
        )
        row_state.finish("PARTIAL")
        return None
    row_state.finish("FAILED")
    return None


async def run_extraction_workflow(
    *,
    item_row: dict[str, object],
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    client: SupportsChatCompletion | None = None,
) -> ExtractionRowState:
    """Run the three-stage extraction workflow for one item row (live backend)."""
    resolved_client = client or OpenRouterChatClient()
    row_state = ExtractionRowState(
        item_row=item_row, stage_name=EXTRACTOR_STAGES[0].name
    )
    messages = initial_messages(row_state)
    while messages is not None:
        try:
            completion = await resolved_client.complete(
                messages=messages,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001
            if is_infrastructure_error(exc):
                # Not a verdict on this row: leave it non-terminal and let the
                # driver abort the run. The row stays pending via the registry.
                raise InfrastructureError(f"{type(exc).__name__}: {exc}") from exc
            record_stage_error(row_state, f"{type(exc).__name__}: {exc}")
            return row_state
        if is_content_filter_abort(completion):
            # Classified here, beside the infrastructure branch above, because
            # it is the same kind of event: the provider returned no answer, so
            # there is nothing to score and nothing for the model to correct.
            # `messages` is untouched, so the identical request goes back out.
            if not handle_provider_abort(row_state, completion):
                return row_state
            continue
        messages = handle_response(
            row_state,
            completion.text,
            max_attempts=max_attempts,
            completion=completion,
        )
    return row_state


def summarize_failure(row_state: ExtractionRowState) -> str:
    """Summarize what this row lost, for its failure-registry entry.

    A salvaged row is terminal-but-publishable: its last attempt often
    succeeded, so the attempt carries no validation errors and the generic
    "unexpected response" summary below would describe a stage that worked.
    The salvage notes are the only record of what was actually dropped, so they
    are what the registry reports.
    """
    if row_state.salvage_notes:
        return "; ".join(row_state.salvage_notes)
    failures = row_state.current_attempt.validation_errors
    if failures:
        return "; ".join(failures)
    if row_state.current_attempt.response:
        return f"Unexpected response at stage {row_state.current_attempt.stage_name}"
    return f"Extractor failed at stage {row_state.current_attempt.stage_name}"


def failed_stage_name(row_state: ExtractionRowState) -> str:
    """Return the stage whose failure this row is registered for.

    For a salvaged row that is the stage salvage fired in, not the last stage
    the row ran — an operator retrying the row needs the former.
    """
    for note in row_state.salvage_notes:
        stage_name, _, _ = note.partition(" ")
        if stage_name in {stage.name for stage in EXTRACTOR_STAGES}:
            return stage_name
    return row_state.current_attempt.stage_name


def _failure_record(
    row_state: ExtractionRowState,
    *,
    partition_date: str,
    shard: str,
    run_id: str,
    backend: str,
) -> dict[str, object]:
    """Build one failure-registry entry for a terminal non-SUCCESS row."""
    return {
        "item_id": row_state.item_id,
        "accession_number": row_state.item_row.get("accession_number"),
        "cik": row_state.item_row.get("cik"),
        "date": partition_date,
        "shard": shard,
        "state": row_state.state,
        "stage": failed_stage_name(row_state),
        "run_id": run_id,
        "backend": backend,
        "error": summarize_failure(row_state),
    }


def _merge_row_failures(
    failures: dict[str, dict[str, object]],
    succeeded_item_ids: set[str],
    *,
    artifact_root: str,
    data_dir: Path | None,
) -> tuple[str, int]:
    """Merge this run's row outcomes into the extract failure registry.

    Failures are added or refreshed; rows that succeeded this run clear any
    earlier entry, so a re-extract that fixes a row does not leave a stale
    failure behind. Returns the registry path and its total entry count.
    """
    registry = load_row_failures(
        "extract", artifact_root=artifact_root, data_dir=data_dir
    )
    for item_id in succeeded_item_ids:
        registry.pop(item_id, None)
    registry.update(failures)
    path = save_row_failures(
        "extract", registry, artifact_root=artifact_root, data_dir=data_dir
    )
    return path, len(registry)
