"""Tests for `cdt extract replay`: stored mentions re-derived from the audit log."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Self

import pandas as pd
import pytest
from support import _seed_classifications

from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS, classifications_root
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.llm import load_prompt
from cdt.extractor.outputs import extracted_tables_path
from cdt.extractor.replay import accepted_stages, replay_extracted_mentions
from cdt.extractor.state import CompletionResult
from cdt.storage.tables import read_dataset, write_partition_table

TEXT = "Citibank, N.A., as Securities Administrator, lent the Company a Term Loan."
TAGGED = (
    "<body><organization>Citibank, N.A.</organization>, as <organization>"
    "Securities Administrator</organization>, lent the <organization>Company"
    "</organization> a <debt_instrument>Term Loan</debt_instrument>.</body>"
)
ENTRIES = json.dumps(
    [
        {
            "name": ["tag-4"],
            "parties": [
                {"tag_ids": ["tag-1", "tag-2"], "role": "lender", "kind": "named"},
                {"tag_ids": ["tag-3"], "role": "borrower", "kind": "named"},
            ],
        }
    ]
)


class _StageClient:
    """Answer each stage with its fixed response, told apart by system prompt."""

    def __init__(self: Self) -> None:
        self.responses = {
            load_prompt("ner"): TAGGED,
            load_prompt("instrument_ie"): ENTRIES,
        }

    async def complete(
        self: Self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> CompletionResult:
        del model, reasoning_effort
        return CompletionResult(text=self.responses[messages[0]["content"]])


def _extract(tmp_path: Path) -> None:
    _seed_classifications(tmp_path, ["a-8-01"])
    _set_text(tmp_path, TEXT)
    extract_pending_items(artifact_root=tmp_path, client=_StageClient())


def _set_text(tmp_path: Path, text: str) -> None:
    rows = read_dataset(classifications_root(tmp_path))
    rows["text"] = text
    write_partition_table(
        str(classifications_root(tmp_path)),
        partition={"month": "2024-01"},
        table=rows.reindex(columns=CLASSIFIED_ITEM_COLUMNS),
    )


def _parties(tmp_path: Path) -> list[tuple[object, object, object]]:
    mentions = read_dataset(mentions_root(tmp_path))
    assert len(mentions) == 1
    return [
        (party["canonical_name"], party["role"], party["kind"])
        for party in json.loads(mentions["parties_json"].iloc[0])
    ]


def _extract_with_longest_span_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extract as the code did before #323: longest span, no role-only check."""
    monkeypatch.setattr(
        "cdt.extractor.normalize.parties.canonical_party_name",
        lambda values: max(values, key=len) if values else None,
    )
    monkeypatch.setattr(
        "cdt.extractor.normalize.parties.is_role_only_text", lambda text: False
    )
    _extract(tmp_path)
    monkeypatch.undo()
    assert _parties(tmp_path) == [
        ("Securities Administrator", "lender", "named"),
        ("Company", "borrower", "named"),
    ]


def test_replay_rewrites_mentions_with_the_current_party_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay applies the current postprocess to the recorded responses (#323)."""
    _extract_with_longest_span_names(tmp_path, monkeypatch)

    result = replay_extracted_mentions(artifact_root=tmp_path)

    assert (result.items_replayed, result.items_changed) == (1, 1)
    assert (result.mention_rows_before, result.mention_rows_after) == (1, 1)
    assert _parties(tmp_path) == [
        ("Citibank, N.A.", "lender", "named"),
        ("Company", "borrower", "collective"),
    ]
    # The replay's own audit record names what the dataset now holds, so a
    # second replay finds the item in step and has nothing left to change.
    again = replay_extracted_mentions(artifact_root=tmp_path)
    assert (again.items_replayed, again.items_changed) == (1, 0)
    assert again.items_out_of_step == []
    assert again.partitions_written == []


def test_a_dry_run_reports_the_change_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--dry-run` counts what would change without touching the root."""
    _extract_with_longest_span_names(tmp_path, monkeypatch)
    runs = sorted(Path(extracted_tables_path(tmp_path)).iterdir())

    result = replay_extracted_mentions(artifact_root=tmp_path, dry_run=True)

    assert result.items_changed == 1
    assert result.partitions_written == []
    assert sorted(Path(extracted_tables_path(tmp_path)).iterdir()) == runs
    assert _parties(tmp_path)[0][0] == "Securities Administrator"


def test_a_replay_under_unchanged_code_rewrites_nothing(tmp_path: Path) -> None:
    """Replayed rows round-trip to the stored ones, so nothing is rewritten."""
    _extract(tmp_path)
    stored = read_dataset(mentions_root(tmp_path))

    result = replay_extracted_mentions(artifact_root=tmp_path)

    assert (result.items_replayed, result.items_changed) == (1, 0)
    assert result.partitions_written == []
    pd.testing.assert_frame_equal(read_dataset(mentions_root(tmp_path)), stored)


def test_an_item_the_current_validators_reject_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded NER response no longer matching the item's text is not replayed."""
    _extract_with_longest_span_names(tmp_path, monkeypatch)
    _set_text(tmp_path, TEXT.replace("Term Loan", "Revolver"))

    result = replay_extracted_mentions(artifact_root=tmp_path)

    assert result.items_rejected == ["a-8-01"]
    assert result.items_changed == 0
    assert _parties(tmp_path)[0][0] == "Securities Administrator"


def test_an_item_without_its_audit_record_is_out_of_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored mentions no audit record accounts for are reported, not rewritten."""
    _extract_with_longest_span_names(tmp_path, monkeypatch)
    shutil.rmtree(extracted_tables_path(tmp_path))

    result = replay_extracted_mentions(artifact_root=tmp_path)

    assert result.items_out_of_step == ["a-8-01"]
    assert result.items_replayed == 0
    assert _parties(tmp_path)[0][0] == "Securities Administrator"


def test_an_item_whose_stored_mentions_its_record_did_not_publish_is_out_of_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay touches an item only while the dataset holds what its record wrote."""
    _extract_with_longest_span_names(tmp_path, monkeypatch)
    mentions = read_dataset(mentions_root(tmp_path))
    mentions["debt_instrument_mention_id"] = "dim::written-by-something-else"
    write_partition_table(
        mentions_root(tmp_path),
        partition={"month": "2024-01"},
        table=mentions.drop(columns=["month"], errors="ignore"),
    )

    result = replay_extracted_mentions(artifact_root=tmp_path)

    assert result.items_out_of_step == ["a-8-01"]
    assert result.items_replayed == 0
    assert _parties(tmp_path)[0][0] == "Securities Administrator"


def test_accepted_stages_follow_the_last_scored_attempt_and_salvage() -> None:
    """A stage counts only when the run kept its response."""
    record = {
        "attempts": [
            {"stage_name": "ner", "status": "SUCCESS"},
            {"stage_name": "instrument_ie", "status": "FAILED"},
            {"stage_name": "instrument_relation", "status": "SUCCESS"},
            {"stage_name": "instrument_relation", "status": "FAILED"},
            # An unscored provider abort is not the stage's verdict.
            {"stage_name": "instrument_relation", "status": "ABORTED"},
        ],
        "salvage_notes": [
            "instrument_ie kept the valid entries and dropped 1 invalid ones "
            "after 3 failed attempts"
        ],
    }

    assert accepted_stages(record) == frozenset({"ner", "instrument_ie"})
