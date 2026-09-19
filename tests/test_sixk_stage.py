"""Tests for the 6-K triage stage: window, score, prune, persist."""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Self

import pandas as pd
import pytest

from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS, score_model
from cdt.datasets import load_completion_registry, run_manifest_path
from cdt.ingest import DOCUMENT_COLUMNS, SIXK_DOCUMENT_DATASET_NAME
from cdt.ingest import documents_root as ingest_documents_root
from cdt.sixk import stage as sixk_stage
from cdt.sixk.documents import prose_documents
from cdt.sixk.stage import (
    SIXK_SNIPPET_COLUMNS,
    VERDICT_DROPPED_DUPLICATE,
    VERDICT_DROPPED_NO_DETAILS,
    VERDICT_KEPT,
    VERDICT_KEPT_DEGRADED,
    item_id_for,
    sixk_snippets_root,
    snippet_id_for,
    triage_documents,
    triage_pending_documents,
)
from cdt.sixk.windows import prepare_filing, strip_inline_xbrl_prologue
from cdt.storage import (
    list_artifacts,
    read_dataset,
    read_json_artifact,
    write_partition_table,
)

# Long enough to pass the debt-vocabulary gate and window into one crop.
DEBT_PROSE = (
    "On March 3, 2026 the Company entered into a credit agreement with Example "
    "Bank plc providing for a term loan of $250,000,000 maturing March 3, 2031. "
    "The notes due 2031 bear interest at 5.25% per annum. "
)
OTHER_PROSE = (
    "The board declared a quarterly dividend of $0.10 per ordinary share, "
    "payable on April 15, 2026 to holders of record on March 31, 2026. "
)


# One paragraph per line, because an expansion walks back to a line boundary:
# a single unbroken paragraph has none within reach and cannot expand at all.
_FILLER_LINE = (
    "The board declared a quarterly dividend of $0.10 per ordinary share "
    "payable on April 15, 2026."
)
_LOAN_LINE = (
    "On March 3, 2026 the Company entered into a credit agreement with Example "
    "Bank plc for a term loan of $250,000,000."
)
_NOTES_LINE = (
    "The notes due 2031 bear interest at 5.25% per annum and were issued at par."
)
# Windows 0-3 are dividend prose the stand-in rejects; the debt lands last, so
# the admitted window has context above it that its crop cut off.
LONG_BODY_ONE_ADMITTED = (
    "OPERATING REVIEW\n"
    + "\n".join(f"{_FILLER_LINE} Paragraph {index}." for index in range(60))
    + "\n"
    + _LOAN_LINE
)
# Same, with enough debt prose to fill two adjacent windows, which merge.
LONG_BODY_TWO_ADMITTED = (
    "OPERATING REVIEW\n"
    + "\n".join(f"{_FILLER_LINE} Paragraph {index}." for index in range(60))
    + "\n"
    + "\n".join(f"{_LOAN_LINE} Tranche {index}." for index in range(6))
    + "\n"
    + "\n".join(f"{_NOTES_LINE} Series {index}." for index in range(6))
)


def _submission(*documents: tuple[str, str]) -> str:
    """Build a complete submission text file from (type, body) pairs."""
    return "".join(
        f"<DOCUMENT><TYPE>{document_type}\n<TEXT>{body}</TEXT></DOCUMENT>"
        for document_type, body in documents
    )


#: Margins the stage-1 stand-in answers with. Signed and well clear of the
#: threshold in both directions; see FakeStage1.
ADMITTED_MARGIN = 5.0
REJECTED_MARGIN = -5.0


class FakeStage1:
    """A stage-1 stand-in admitting windows that mention debt, rejecting others.

    Margins, not probabilities, and signed: `stage1_admit` scores through
    `classifier.core.score_model`, which puts a logistic transform over
    whatever `decision_function` returns. A stand-in answering 0.9 and 0.1
    therefore produces 0.71 and 0.52 — both above any threshold the real
    artifact carries (0.332), so *every* window would be admitted and no test
    could observe a stage-1 rejection. ±5 maps to 0.993 and 0.0067, which
    straddle it the way the fitted model's margins do.
    """

    def decision_function(self: Self, texts: list[str]) -> list[float]:
        """Return a positive margin for debt text and a negative one for rest."""
        return [
            ADMITTED_MARGIN
            if "credit agreement" in text or "notes due" in text
            else REJECTED_MARGIN
            for text in texts
        ]


class RecordingStage1(FakeStage1):
    """FakeStage1, remembering every text it was asked to score."""

    def __init__(self: Self) -> None:
        """Start with nothing scored."""
        self.scored: list[str] = []

    def decision_function(self: Self, texts: list[str]) -> list[float]:
        """Record the texts, then score them as FakeStage1 does."""
        self.scored.extend(texts)
        return super().decision_function(texts)


class FakeChatClient:
    """A stage-2 stand-in returning canned verdicts, or raising."""

    def __init__(
        self: Self,
        responses: list[str] | None = None,
        *,
        error: Exception | None = None,
        keep_all: bool = False,
    ) -> None:
        """Initialize with canned responses, an error, or keep-everything."""
        self.responses = list(responses or [])
        self.error = error
        self.keep_all = keep_all
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self: Self,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> str:
        """Return the next canned verdict for one filing."""
        del model, reasoning_effort
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        if self.keep_all:
            # Verdicts are 1-based indices into the snippets as numbered in the
            # prompt. Count line-anchored fences across the whole conversation:
            # the header quotes the boundary format inline, and a retry message
            # carries no fences at all, so neither a substring count nor a look
            # at the last message alone gets this right.
            indices = [
                int(match.group(1))
                for message in messages
                for match in re.finditer(
                    r"^--- snippet (\d+) \[", message["content"], re.MULTILINE
                )
            ]
            return json.dumps(
                {"keep": list(range(1, max(indices, default=0) + 1)), "drop": []}
            )
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _stub_stage1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a keyword-scoring stand-in for the fitted stage-1 artifact.

    The committed artifact lives under the real DATA_DIR, which conftest
    redirects per test; loading it would also make these tests depend on the
    model's actual scores rather than on the stage's behaviour.
    """
    monkeypatch.setattr(
        sixk_stage,
        "load_stage1_model",
        lambda model_dir=None: (FakeStage1(), 0.332),
    )


def _document_row(
    tmp_path: Path,
    *,
    accession_number: str = "000000000026000001",
    submission: str,
    date: str = "2026-03-03",
    mirror: bool = False,
) -> dict[str, object]:
    """Build one 6-K document row, optionally backed by a gzipped mirror."""
    row: dict[str, object] = {
        "accession_number": accession_number,
        "cik": "312069",
        "company_name": "Example PLC",
        "url": f"https://sec.example/{accession_number}.txt",
        "text": "" if mirror else submission,
        "date": date,
        "resource_uri": None,
        "form_type": "6-K",
        "source": "edgar",
    }
    if mirror:
        target = tmp_path / "raw-documents" / "sixk" / f"{accession_number}.txt.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(submission.encode()))
        row["resource_uri"] = str(target)
    return row


def test_stripping_the_xbrl_prologue_is_idempotent() -> None:
    """The claim documents.py relies on: the strip can safely happen twice.

    `prose_documents` deliberately leaves stripping to `prepare_filing`, and the
    research harness did it in both places. That is only harmless if a second
    pass is a no-op.
    """
    prologue = ["lzm-20250630", "false", "0001958217", "iso4217:USD"] * 8
    text = "\n".join([*prologue, DEBT_PROSE])
    once = strip_inline_xbrl_prologue(text)

    assert once == DEBT_PROSE
    assert strip_inline_xbrl_prologue(once) == once


def test_prose_documents_keeps_the_body_and_exhibits_and_drops_artifacts() -> None:
    """Only document types that can carry agreement prose are windowed."""
    submission = _submission(
        ("6-K", "<p>Cover page.</p>"),
        ("EX-99.1", "<p>Press release.</p>"),
        ("EX-10.1", "<p>Credit agreement.</p>"),
        ("GRAPHIC", "begin 644 logo.jpg"),
        ("XML", "<xbrl>facts</xbrl>"),
    )

    assert [document.document_type for document in prose_documents(submission)] == [
        "6-K",
        "EX-99.1",
        "EX-10.1",
    ]


def test_triage_keeps_stage_two_s_choice_and_records_the_rest(
    tmp_path: Path,
) -> None:
    """Admitted windows are all persisted; only the kept ones are relevant."""
    submission = _submission(
        ("6-K", f"<p>{DEBT_PROSE}</p>"),
        ("EX-99.1", f"<p>{DEBT_PROSE}</p>"),
    )
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    kept_id = snippet_id_for("000000000026000001", 0, 0)
    dropped_id = snippet_id_for("000000000026000001", 1, 0)
    client = FakeChatClient(
        [
            json.dumps(
                {
                    "keep": [1],
                    "drop": [{"id": 2, "reason": "duplicate", "covered_by": 1}],
                }
            )
        ]
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    assert list(snippets.columns) == SIXK_SNIPPET_COLUMNS
    assert len(client.calls) == 1
    by_item = snippets.set_index("item")
    assert by_item.loc[kept_id, "sixk_verdict"] == VERDICT_KEPT
    assert bool(by_item.loc[kept_id, "relevance"]) is True
    assert by_item.loc[kept_id, "label"] == "relevant"
    assert by_item.loc[dropped_id, "sixk_verdict"] == VERDICT_DROPPED_DUPLICATE
    assert by_item.loc[dropped_id, "sixk_duplicate_of"] == kept_id
    assert bool(by_item.loc[dropped_id, "relevance"]) is False


def test_triage_rows_satisfy_the_extractor_s_input_contract(tmp_path: Path) -> None:
    """The extractor reads these rows with its own projection and no changes."""
    documents = pd.DataFrame(
        [
            _document_row(
                tmp_path, submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>"))
            )
        ],
        columns=DOCUMENT_COLUMNS,
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=FakeChatClient(keep_all=True)
    )

    projected = snippets.reindex(columns=CLASSIFIED_ITEM_COLUMNS)
    assert list(projected.columns) == CLASSIFIED_ITEM_COLUMNS
    row = projected.iloc[0]
    # The id names the span the row carries, so it moves whenever the text does.
    span = snippets.iloc[0]
    assert row["item_id"] == item_id_for(
        "000000000026000001", 0, span["sixk_window_start"], span["sixk_window_end"]
    )
    # The extractor reads `text` and nothing else about the source.
    assert "credit agreement" in row["text"]
    assert row["accession_number"] == "000000000026000001"
    assert row["cik"] == "312069"
    assert row["date"] == "2026-03-03"
    assert row["relevance"]
    # classification_score is the logistic-transformed margin, the same
    # transformation the 8-K classifier's score goes through.
    assert row["classification_score"] == pytest.approx(
        1 / (1 + math.exp(-ADMITTED_MARGIN))
    )
    # 6-K item ids cannot collide with 8-K ones, which lets both genres merge
    # mentions into one partition.
    assert row["item_id"] != "000000000026000001-8-01"


def test_triage_judges_a_whole_filing_in_one_call(tmp_path: Path) -> None:
    """Stage 2 sees all of a filing's admitted windows at once, by design.

    Whether a window merely repeats a sibling cannot be judged from the window
    alone, and grouping is what makes stage 2 cheap. Per-window calls would be a
    different configuration from the measured one.
    """
    submission = _submission(
        ("6-K", f"<p>{DEBT_PROSE}</p>"),
        ("EX-99.1", f"<p>{DEBT_PROSE}</p>"),
        ("EX-10.1", f"<p>{DEBT_PROSE}</p>"),
    )
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    client = FakeChatClient(keep_all=True)

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    assert len(snippets) == 3
    assert len(client.calls) == 1
    # All three windows are fenced into the one prompt. Stage 2 is shown text
    # under numbered fences and never the snippet ids themselves; the verdict
    # comes back as indices, which triage_filing maps back to ids.
    prompt = client.calls[0][-1]["content"]
    assert [f"--- snippet {number} [" in prompt for number in (1, 2, 3)] == [
        True,
        True,
        True,
    ]


def test_a_window_stage_one_rejects_is_neither_sent_nor_persisted(
    tmp_path: Path,
) -> None:
    """Stage 1 admits 5.8% of windows; the rest leave no row behind.

    Distinct from the vocabulary gate, which rejects a whole *document* for
    mentioning no debt. Here the document passes the gate — it discusses a
    credit agreement — and stage 1 still rejects the windows of it that do not,
    which is the 17x the dataset would grow by if they were persisted to record
    that nothing happened.
    """
    submission = _submission(("6-K", f"<p>{LONG_BODY_ONE_ADMITTED}</p>"))
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    windows = prepare_filing(prose_documents(submission)[0].text)
    admitted = [window for window in windows if "credit agreement" in window.text]
    rejected = [window for window in windows if "credit agreement" not in window.text]
    assert admitted and rejected
    client = FakeChatClient(keep_all=True)

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    # One row for the one admission, and none for the rejections.
    assert snippets["sixk_member_windows"].to_list() == [str(admitted[0].index)]
    assert snippets["item"].to_list() == [
        snippet_id_for("000000000026000001", 0, admitted[0].index)
    ]
    # Stage 2 was asked about the admission only: a rejected window's text is
    # absent from the prompt, so it cost neither a row nor a token.
    prompt = "".join(message["content"] for message in client.calls[0])
    assert len(re.findall(r"^--- snippet \d+ \[", prompt, re.MULTILINE)) == 1
    assert rejected[0].text not in prompt


def test_stage_one_scores_every_gated_window_and_admits_some(
    tmp_path: Path,
) -> None:
    """The stand-in must be able to reject, or nothing above proves anything.

    Pinned because the failure is silent and was real: `score_model` puts a
    logistic transform over the margin, so a stand-in answering 0.1 for
    "reject" produces 0.52 and every window is admitted. Every assertion about
    admission in this file would then hold vacuously.
    """
    texts = ["a credit agreement dated March 3", "a quarterly dividend of $0.10"]
    scores = score_model(FakeStage1(), texts)

    assert scores[0] > 0.332
    assert scores[1] < 0.332


def test_triage_makes_no_call_for_a_filing_the_gate_rejects(tmp_path: Path) -> None:
    """Most filings mention no debt; those must not reach stage 2 at all."""
    documents = pd.DataFrame(
        [
            _document_row(
                tmp_path, submission=_submission(("6-K", f"<p>{OTHER_PROSE}</p>"))
            )
        ],
        columns=DOCUMENT_COLUMNS,
    )
    client = FakeChatClient()

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    assert snippets.empty
    assert client.calls == []


def test_triage_needs_no_api_key_when_nothing_is_admitted(tmp_path: Path) -> None:
    """A batch with nothing to judge must not build a client to find out."""
    documents = pd.DataFrame(
        [
            _document_row(
                tmp_path, submission=_submission(("6-K", f"<p>{OTHER_PROSE}</p>"))
            )
        ],
        columns=DOCUMENT_COLUMNS,
    )

    # client=None, so a client would be constructed if one were needed — and
    # with no credentials configured that construction raises.
    snippets = triage_documents(documents, artifacts=(FakeStage1(), 0.332))

    assert snippets.empty


def test_triage_degrades_to_stage_one_output_when_stage_two_fails(
    tmp_path: Path,
) -> None:
    """A stage-2 outage keeps every admitted window rather than losing data.

    Costs roughly twice as much extraction for that filing and loses no recall,
    which is the right direction to fail in.
    """
    documents = pd.DataFrame(
        [
            _document_row(
                tmp_path,
                submission=_submission(
                    ("6-K", f"<p>{DEBT_PROSE}</p>"), ("EX-99.1", f"<p>{DEBT_PROSE}</p>")
                ),
            )
        ],
        columns=DOCUMENT_COLUMNS,
    )
    client = FakeChatClient(error=RuntimeError("provider down"))

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    assert len(snippets) == 2
    assert snippets["relevance"].all()
    assert set(snippets["sixk_verdict"]) == {VERDICT_KEPT_DEGRADED}


def test_triage_resolves_text_from_a_mirrored_submission(tmp_path: Path) -> None:
    """A 6-K row carries a resource_uri, exactly as an 8-K row does."""
    documents = pd.DataFrame(
        [
            _document_row(
                tmp_path,
                submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>")),
                mirror=True,
            )
        ],
        columns=DOCUMENT_COLUMNS,
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=FakeChatClient(keep_all=True)
    )

    assert len(snippets) == 1
    assert "credit agreement" in snippets.iloc[0]["text"]
    assert snippets.iloc[0]["resource_uri"].endswith(".txt.gz")


def test_triage_records_the_window_span_and_document_type(tmp_path: Path) -> None:
    """Audit columns place a snippet in its document."""
    submission = _submission(("EX-99.1", f"<p>{DEBT_PROSE}</p>"))
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=FakeChatClient(keep_all=True)
    )

    row = snippets.iloc[0]
    assert row["section_heading"] == "EX-99.1"
    assert row["sixk_window_start"] == 0
    assert row["sixk_window_end"] > 0
    assert row["sixk_token_count"] > 0
    assert row["section_char_count"] == len(row["text"])
    # Line columns mean lines of an 8-K body; a window is a character span.
    assert pd.isna(row["start_line"])
    assert pd.isna(row["end_line"])


def test_an_admitted_window_is_sent_with_the_context_its_crop_cut_off(
    tmp_path: Path,
) -> None:
    """The row's text is the expanded window, and stage 2 reads it (#172).

    A 400-token crop can keep an instrument's amounts and dates while cutting
    away the noun naming it, which leaves extraction nothing to anchor on.
    """
    submission = _submission(("6-K", f"<p>{LONG_BODY_ONE_ADMITTED}</p>"))
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    crop = next(
        window
        for window in prepare_filing(prose_documents(submission)[0].text)
        if "credit agreement" in window.text
    )
    client = FakeChatClient(keep_all=True)

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    row = snippets.iloc[0]
    assert row["sixk_window_start"] < crop.start
    assert row["sixk_window_end"] == crop.end
    assert row["text"].endswith(crop.text)
    assert len(row["text"]) > len(crop.text)
    # Span, text and the count the row reports all describe the same window.
    assert row["section_char_count"] == len(row["text"])
    assert row["sixk_token_count"] > crop.token_count
    # What stage 2 actually received, not merely what was persisted.
    prompt = "".join(message["content"] for message in client.calls[0])
    assert row["text"] in prompt


def test_stage_one_scores_the_crop_and_never_the_expansion(tmp_path: Path) -> None:
    """Expansion runs after stage 1, because its threshold assumes the crop.

    `WINDOW_TOKENS` is what the shipped model was fitted on and what the
    threshold in its metadata was calibrated against, so widening the text
    stage 1 scores would invalidate both at once — silently, since a threshold
    cannot report that its input changed shape.
    """
    submission = _submission(("6-K", f"<p>{LONG_BODY_ONE_ADMITTED}</p>"))
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    windows = prepare_filing(prose_documents(submission)[0].text)
    stage1 = RecordingStage1()

    triage_documents(
        documents, artifacts=(stage1, 0.332), client=FakeChatClient(keep_all=True)
    )

    assert stage1.scored == [window.text for window in windows]


def test_adjacent_admitted_windows_become_one_row(tmp_path: Path) -> None:
    """Merged windows are one snippet and one row, naming their members.

    Two admitted windows whose expansions run into each other share text. Sent
    separately they would send it twice and, persisted separately, be extracted
    twice — so they merge, and the row records which admitted windows it
    answers for.
    """
    submission = _submission(("6-K", f"<p>{LONG_BODY_TWO_ADMITTED}</p>"))
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    admitted = [
        window
        for window in prepare_filing(prose_documents(submission)[0].text)
        if "credit agreement" in window.text or "notes due" in window.text
    ]
    assert len(admitted) > 1
    client = FakeChatClient(keep_all=True)

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    assert len(snippets) == 1
    row = snippets.iloc[0]
    assert row["sixk_member_windows"] == ",".join(
        str(window.index) for window in admitted
    )
    # The merged text carries both members, and was sent once.
    assert "Tranche 0." in row["text"]
    assert "Series 5." in row["text"]
    prompt = "".join(message["content"] for message in client.calls[0])
    # Line-anchored, because the system prompt quotes the fence format inline.
    assert len(re.findall(r"^--- snippet \d+ \[", prompt, re.MULTILINE)) == 1
    # Identity comes from the earliest member, so no two groups can claim it.
    assert row["item"] == snippet_id_for("000000000026000001", 0, admitted[0].index)
    # The item id names the merged span, not the first member: a regrouping has
    # to read as a different row, or the extractor skips it as already done.
    assert row["item_id"] == item_id_for(
        "000000000026000001", 0, row["sixk_window_start"], row["sixk_window_end"]
    )
    assert row["item_id"] != item_id_for(
        "000000000026000001", 0, admitted[0].start, admitted[0].end
    )


def test_an_unmerged_snippet_still_names_its_one_window(tmp_path: Path) -> None:
    """The member column is populated for every row, not only merged ones."""
    submission = _submission(("6-K", f"<p>{DEBT_PROSE}</p>"))
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=FakeChatClient(keep_all=True)
    )

    assert snippets.iloc[0]["sixk_member_windows"] == "0"


def test_windows_of_two_documents_are_never_merged_together(tmp_path: Path) -> None:
    """Offsets only mean anything inside the document they index into.

    Merging across documents would splice unrelated text into one snippet, and
    the two documents' windows are numbered from zero independently, so the
    members of a merged window would not even identify themselves.
    """
    submission = _submission(
        ("6-K", f"<p>{DEBT_PROSE}</p>"),
        ("EX-99.1", f"<p>{DEBT_PROSE}</p>"),
    )
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=FakeChatClient(keep_all=True)
    )

    assert len(snippets) == 2
    assert snippets["sixk_member_windows"].to_list() == ["0", "0"]
    assert snippets["section_heading"].to_list() == ["6-K", "EX-99.1"]


def _write_documents(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    write_partition_table(
        ingest_documents_root(str(tmp_path), dataset_name=SIXK_DOCUMENT_DATASET_NAME),
        partition={"date": "2026-03-03", "shard": "0001"},
        table=pd.DataFrame(rows, columns=DOCUMENT_COLUMNS),
    )


def test_stage_writes_snippet_partitions_and_records_completion(
    tmp_path: Path,
) -> None:
    """The stage persists partitions, a registry entry and a run manifest."""
    _write_documents(
        tmp_path,
        [
            _document_row(
                tmp_path, submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>"))
            )
        ],
    )

    snippets = triage_pending_documents(
        artifact_root=str(tmp_path),
        client=FakeChatClient(keep_all=True),
    )

    assert len(snippets) == 1
    persisted = read_dataset(
        sixk_snippets_root(str(tmp_path)), columns=SIXK_SNIPPET_COLUMNS
    )
    assert len(persisted) == 1
    registry = load_completion_registry("sixk", artifact_root=str(tmp_path))
    assert len(registry) == 1
    manifest = read_json_artifact(
        run_manifest_path("sixk", "latest", artifact_root=str(tmp_path))
    )
    assert isinstance(manifest, dict)
    assert manifest["stage"] == "sixk"
    assert manifest["documents_processed"] == 1


def test_stage_skips_a_partition_it_has_already_triaged(tmp_path: Path) -> None:
    """Completion is fingerprint-keyed, so a second run pays for nothing."""
    _write_documents(
        tmp_path,
        [
            _document_row(
                tmp_path, submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>"))
            )
        ],
    )
    first_client = FakeChatClient(keep_all=True)
    triage_pending_documents(artifact_root=str(tmp_path), client=first_client)

    second_client = FakeChatClient(keep_all=True)
    again = triage_pending_documents(artifact_root=str(tmp_path), client=second_client)

    assert first_client.calls
    assert second_client.calls == []
    assert again.empty


def test_stage_retriages_a_partition_ingest_grew(tmp_path: Path) -> None:
    """A documents partition with new rows is pending again."""
    first_row = _document_row(
        tmp_path, submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>"))
    )
    _write_documents(tmp_path, [first_row])
    triage_pending_documents(
        artifact_root=str(tmp_path), client=FakeChatClient(keep_all=True)
    )

    second_row = _document_row(
        tmp_path,
        accession_number="000000000026000002",
        submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>")),
    )
    _write_documents(tmp_path, [first_row, second_row])
    client = FakeChatClient(keep_all=True)
    snippets = triage_pending_documents(artifact_root=str(tmp_path), client=client)

    assert len(client.calls) == 2
    assert set(snippets["accession_number"]) == {
        "000000000026000001",
        "000000000026000002",
    }


def test_snippets_land_in_their_source_documents_partition(tmp_path: Path) -> None:
    """Snippets inherit the source partition, which is what groups a filing.

    Stage 2 must see a whole filing at once. Documents shard by accession
    (`ingest._document_shard`), so every window of one filing is co-partitioned
    and reaches one call; snippets are then written at the same coordinates, so
    a re-run maps output back to input without a lookup.
    """
    _write_documents(
        tmp_path,
        [
            _document_row(
                tmp_path, submission=_submission(("6-K", f"<p>{DEBT_PROSE}</p>"))
            )
        ],
    )

    triage_pending_documents(
        artifact_root=str(tmp_path), client=FakeChatClient(keep_all=True)
    )

    written = list_artifacts(sixk_snippets_root(str(tmp_path)), suffix=".parquet")
    assert [Path(path).parent.parts[-2:] for path in written] == [
        ("date=2026-03-03", "shard=0001")
    ]


def test_a_dropped_window_records_why(tmp_path: Path) -> None:
    """The no-details drop reason is persisted, not collapsed into a boolean."""
    submission = _submission(
        ("6-K", f"<p>{DEBT_PROSE}</p>"), ("EX-99.1", f"<p>{DEBT_PROSE}</p>")
    )
    documents = pd.DataFrame(
        [_document_row(tmp_path, submission=submission)], columns=DOCUMENT_COLUMNS
    )
    client = FakeChatClient(
        [json.dumps({"keep": [1], "drop": [{"id": 2, "reason": "no_details"}]})]
    )

    snippets = triage_documents(
        documents, artifacts=(FakeStage1(), 0.332), client=client
    )

    verdicts = dict(zip(snippets["item"], snippets["sixk_verdict"], strict=True))
    assert verdicts[snippet_id_for("000000000026000001", 1, 0)] == (
        VERDICT_DROPPED_NO_DETAILS
    )
    assert (
        snippets.loc[
            snippets["sixk_verdict"] == VERDICT_DROPPED_NO_DETAILS, "sixk_duplicate_of"
        ]
        .isna()
        .all()
    )


def test_windows_of_one_document_are_numbered_consecutively() -> None:
    """Snippet ids are (document, window) pairs, and must be stable."""
    windows = prepare_filing(DEBT_PROSE * 20)

    assert [window.index for window in windows] == list(range(len(windows)))
    assert len(windows) > 1
