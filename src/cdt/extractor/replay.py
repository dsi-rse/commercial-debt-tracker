"""Re-derive stored mentions from the recorded model responses, without the model.

What an item publishes is a pure function of its recorded stage responses and
its text: the stages' ``validate`` and ``postprocess`` call no model. So when
postprocess changes (a party-naming rule, a date normalization), every stored
mention can be brought up to date by replaying the responses in the audit log
(``extractor-runs/run_id=*/``) through the current code. The model's answers,
the row outcomes and both registries stay as they are; only mention rows change.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import (
    MONTH_PARTITION_PATTERN,
    dataset_root,
    month_partition_path,
    parse_source_partition,
    resolve_artifact_root,
    run_manifest_path,
    source_partition_month,
)
from cdt.extractor.batch import active_job_claimed_partition_paths
from cdt.extractor.outputs import (
    CLASSIFICATION_SOURCES,
    MENTIONS_DATASET_NAME,
    checkpoint_audit_path,
    extracted_tables_path,
    write_mentions_partition,
)
from cdt.extractor.prior_state import published_mention_rows
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.stages import EXTRACTOR_STAGES, InstrumentIEStage
from cdt.extractor.state import (
    ABORTED_ATTEMPT_STATUS,
    PUBLISHABLE_ROW_STATES,
    STATE_ITEM_ROW_FIELDS,
    ExtractionRowState,
    coerce_native,
)
from cdt.lease import throttled
from cdt.shared import get_logger
from cdt.storage.columns import apply_declared_column_types
from cdt.storage.objects import (
    iter_text_lines,
    list_artifacts,
    write_json_artifact,
    write_text_artifact,
)
from cdt.storage.tables import read_table

LOGGER = get_logger(__name__)

#: The ``backend`` a replay's run manifest records.
REPLAY_BACKEND = "replay"


@dataclass(frozen=True)
class RecordedOutcome:
    """The part of one audit record a replay needs.

    ``accepted_stages`` are the stages whose recorded response the run kept:
    the stage's last scored attempt passed validation, or, for
    ``instrument_ie``, salvage kept its valid entries (the stored response is
    then already the kept entries). ``mention_ids`` are the mentions the record
    published, which the mentions dataset must still hold for the replay to
    touch the item.
    """

    run_id: str
    state: str | None
    salvage_notes: tuple[str, ...]
    stage_responses: dict[str, str]
    accepted_stages: frozenset[str]
    mention_ids: frozenset[str]


@dataclass
class ReplayResult:
    """What one replay found and, unless a dry run, wrote."""

    run_id: str
    dry_run: bool
    items_replayed: int = 0
    items_changed: int = 0
    mention_rows_before: int = 0
    mention_rows_after: int = 0
    #: Items the current validators now reject; their mentions are left as they are.
    items_rejected: list[str] = field(default_factory=list)
    #: Items whose stored mentions are not the ones their newest record published.
    items_out_of_step: list[str] = field(default_factory=list)
    partitions_written: list[str] = field(default_factory=list)
    audit_paths: list[str] = field(default_factory=list)


def accepted_stages(record: dict[str, object]) -> frozenset[str]:
    """Return the stages whose recorded response one audit record's run kept."""
    last_status: dict[str, object] = {}
    for attempt in cast(list[dict[str, object]], record.get("attempts") or []):
        if attempt.get("status") == ABORTED_ATTEMPT_STATUS:
            continue
        last_status[str(attempt.get("stage_name"))] = attempt.get("status")
    accepted = {name for name, status in last_status.items() if status == "SUCCESS"}
    salvage_notes = cast(list[str], record.get("salvage_notes") or [])
    if any(note.startswith(f"{InstrumentIEStage.name} ") for note in salvage_notes):
        accepted.add(InstrumentIEStage.name)
    return frozenset(accepted)


def recorded_outcome(record: dict[str, object], run_id: str) -> RecordedOutcome:
    """Return the replay's view of one parsed audit record."""
    mentions = cast(
        list[dict[str, object]], record.get("debt_instrument_mentions") or []
    )
    return RecordedOutcome(
        run_id=run_id,
        state=cast(str | None, record.get("state")),
        salvage_notes=tuple(cast(list[str], record.get("salvage_notes") or [])),
        # Only a publishable record is replayed, so only its responses are kept.
        stage_responses=(
            dict(cast(dict[str, str], record.get("stage_responses") or {}))
            if record.get("state") in PUBLISHABLE_ROW_STATES
            else {}
        ),
        accepted_stages=accepted_stages(record),
        mention_ids=frozenset(
            str(mention["debt_instrument_mention_id"]) for mention in mentions
        ),
    )


def _audit_files(resolved_root: str) -> list[tuple[str, list[str]]]:
    """Return each extract run's id and audit files, oldest run first.

    A run's ``full.jsonl`` holds every record its checkpoint files did, so the
    checkpoints are read only for a run that never wrote it.
    """
    by_run: dict[str, list[str]] = {}
    for path in list_artifacts(extracted_tables_path(resolved_root), suffix=".jsonl"):
        run_dir = path.rpartition("/")[0]
        _, marker, run_id = run_dir.rpartition("/run_id=")
        if marker:
            by_run.setdefault(run_id, []).append(path)
    files: list[tuple[str, list[str]]] = []
    for run_id in sorted(by_run):
        paths = sorted(by_run[run_id])
        full = [path for path in paths if path.endswith("/full.jsonl")]
        files.append((run_id, full or paths))
    return files


def latest_recorded_outcomes(resolved_root: str) -> dict[str, RecordedOutcome]:
    """Return each item's newest audit record, across every extract run.

    Run ids are UTC timestamps, so the newest run's record wins. Files are
    streamed a line at a time, and only the fields a replay needs are kept.
    """
    outcomes: dict[str, RecordedOutcome] = {}
    for run_id, paths in _audit_files(resolved_root):
        for path in paths:
            for line in iter_text_lines(path):
                if not line.strip():
                    continue
                record = json.loads(line)
                outcomes[str(record["item_id"])] = recorded_outcome(record, run_id)
    return outcomes


def replay_item(
    item_row: dict[str, object], outcome: RecordedOutcome
) -> list[dict[str, object]] | None:
    """Return the mention rows the current code publishes from one recorded outcome.

    Each accepted stage's response is validated and postprocessed in stage
    order, stopping at the first stage the run did not accept, as the run
    itself stopped there (an early stop, or lineage dropped by salvage). None
    when the current validators reject a response the run accepted. Retry-history
    checks see no earlier attempts: they judged the run's retries, which are
    not replayed.
    """
    row_state = ExtractionRowState(
        item_row=item_row, stage_name=EXTRACTOR_STAGES[0].name
    )
    row_state.stage_responses = dict(outcome.stage_responses)
    for stage in EXTRACTOR_STAGES:
        response = row_state.stage_responses.get(stage.name)
        if stage.name not in outcome.accepted_stages or response is None:
            break
        if stage.validate(row_state, response):
            return None
        stage.postprocess(row_state)
    return published_mention_rows(row_state)


def _row_keys(frame: pd.DataFrame) -> list[str]:
    """Return one sorted, storage-typed text key per mention row."""
    typed = apply_declared_column_types(
        frame.reindex(columns=DEBT_INSTRUMENT_MENTION_COLUMNS)
    ).to_pandas()
    return sorted(
        json.dumps([None if pd.isna(value) else str(value) for value in row])
        for row in typed.itertuples(index=False, name=None)
    )


def _classified_partitions_by_month(
    resolved_root: str, exclude_paths: set[str]
) -> dict[str, list[str]]:
    """Return every classification month file, by month, except ``exclude_paths``."""
    by_month: dict[str, list[str]] = {}
    for source in CLASSIFICATION_SOURCES:
        for path in list_artifacts(
            dataset_root(source, artifact_root=resolved_root), suffix=".parquet"
        ):
            if not MONTH_PARTITION_PATTERN.search(path) or path in exclude_paths:
                continue
            month = source_partition_month(parse_source_partition(path))
            by_month.setdefault(month, []).append(path)
    return by_month


def _relevant_item_rows(paths: list[str]) -> Iterator[dict[str, object]]:
    """Yield each relevant classification row in ``paths``, native-typed."""
    for path in paths:
        rows = read_table(path, CLASSIFIED_ITEM_COLUMNS).reindex(
            columns=CLASSIFIED_ITEM_COLUMNS
        )
        for row in rows.loc[rows["relevance"].fillna(False)].to_dict("records"):
            yield {key: coerce_native(row.get(key)) for key in STATE_ITEM_ROW_FIELDS}


def _replay_audit_record(
    item_row: dict[str, object],
    outcome: RecordedOutcome,
    mention_rows: list[dict[str, object]],
) -> str:
    """Return the audit line recording what a replayed item now publishes.

    It keeps the replay invariant for the next replay: an item's newest record
    names the mentions the dataset holds. ``attempts`` keeps only what
    :func:`accepted_stages` reads; the full attempts stay in ``replay_of``'s log.
    """
    attempts = [
        {"stage_name": name, "status": "SUCCESS"}
        for name in sorted(outcome.accepted_stages)
    ]
    return json.dumps(
        {
            "item_id": item_row["item_id"],
            "accession_number": item_row.get("accession_number"),
            "item": item_row.get("item"),
            "stage_responses": outcome.stage_responses,
            "debt_instrument_mentions": mention_rows,
            "state": outcome.state,
            "salvage_notes": list(outcome.salvage_notes),
            "attempts": attempts,
            "replay_of": outcome.run_id,
        },
        sort_keys=True,
        default=str,
    )


def replay_extracted_mentions(
    *,
    artifact_root: str | Path | None = None,
    dry_run: bool = False,
    renew: Callable[[], None] | None = None,
) -> ReplayResult:
    """Rewrite stored mentions as the current code derives them from the audit log.

    An item is replayed when its newest audit record is publishable
    (``SUCCESS`` or ``PARTIAL``), its classification row is still relevant, and
    the mentions dataset holds exactly the mentions that record published.
    Otherwise it is left alone; an item whose stored mentions differ from its
    record is reported out of step. Each month file is rewritten only for the
    items whose rows changed, and each changed item gets a new audit record in
    this replay's run. Classification files the active batch job claimed are
    skipped: the job's finalize writes them. ``dry_run`` writes nothing.

    ``renew`` extends the caller's writer lease, throttled to at most every
    :data:`cdt.lease.RENEW_INTERVAL_SECONDS`.
    """
    resolved_root = resolve_artifact_root(artifact_root)
    keep_lease = throttled(renew) if renew is not None else None
    result = ReplayResult(
        run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"), dry_run=dry_run
    )
    outcomes = latest_recorded_outcomes(resolved_root)
    claimed = active_job_claimed_partition_paths(resolved_root)
    if claimed:
        LOGGER.info(
            "Skipping %s classification partition(s) claimed by the active batch "
            "extract job.",
            len(claimed),
        )
    for month, paths in sorted(
        _classified_partitions_by_month(resolved_root, claimed).items()
    ):
        if keep_lease is not None:
            keep_lease()
        stored = read_table(
            month_partition_path(
                MENTIONS_DATASET_NAME, month=month, artifact_root=resolved_root
            ),
            DEBT_INSTRUMENT_MENTION_COLUMNS,
        )
        stored_by_item = {
            str(item_id): rows
            for item_id, rows in stored.groupby(stored["item_id"].astype(str))
        }
        new_rows: list[dict[str, object]] = []
        changed_ids: set[str] = set()
        audit_records: list[str] = []
        for item_row in _relevant_item_rows(paths):
            item_id = str(item_row["item_id"])
            outcome = outcomes.get(item_id)
            current = stored_by_item.get(item_id, stored.iloc[0:0])
            if outcome is None or outcome.state not in PUBLISHABLE_ROW_STATES:
                if not current.empty:
                    result.items_out_of_step.append(item_id)
                continue
            if (
                frozenset(current["debt_instrument_mention_id"].astype(str))
                != outcome.mention_ids
            ):
                result.items_out_of_step.append(item_id)
                continue
            replayed = replay_item(item_row, outcome)
            if replayed is None:
                result.items_rejected.append(item_id)
                continue
            result.items_replayed += 1
            replayed_frame = pd.DataFrame(
                replayed, columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            )
            if _row_keys(replayed_frame) == _row_keys(current):
                continue
            result.items_changed += 1
            result.mention_rows_before += len(current)
            result.mention_rows_after += len(replayed)
            changed_ids.add(item_id)
            new_rows.extend(replayed)
            audit_records.append(_replay_audit_record(item_row, outcome, replayed))
        if dry_run or not changed_ids:
            continue
        written = write_mentions_partition(
            resolved_root,
            month=month,
            new_mentions=pd.DataFrame(
                new_rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            ),
            replaced_item_ids=changed_ids,
            retired_item_ids=set(),
        )
        if written is not None:
            result.partitions_written.append(written)
        # Right after its month file, so a replay that stops part way leaves
        # every month it wrote with records naming what that month now holds.
        audit_path = checkpoint_audit_path(
            result.run_id, len(result.audit_paths), artifact_root=resolved_root
        )
        write_text_artifact(audit_path, "\n".join(audit_records) + "\n")
        result.audit_paths.append(audit_path)
    if result.audit_paths:
        write_json_artifact(
            run_manifest_path("extract", result.run_id, artifact_root=resolved_root),
            {
                "artifact_root": resolved_root,
                "stage": "extract",
                "backend": REPLAY_BACKEND,
                "items_replayed": result.items_replayed,
                "items_changed": result.items_changed,
                "items_rejected": result.items_rejected,
                "items_out_of_step": result.items_out_of_step,
                "partitions_written": result.partitions_written,
                "audit_paths": result.audit_paths,
            },
        )
    LOGGER.info(
        "Extract replay%s: replayed=%s changed=%s rows %s -> %s rejected=%s "
        "out_of_step=%s partitions_written=%s",
        " (dry run)" if dry_run else "",
        result.items_replayed,
        result.items_changed,
        result.mention_rows_before,
        result.mention_rows_after,
        len(result.items_rejected),
        len(result.items_out_of_step),
        len(result.partitions_written),
    )
    return result
