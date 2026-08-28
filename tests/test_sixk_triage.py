"""Tests for the Form 6-K two-stage triage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Self

import pytest

from cdt import settings
from cdt.sixk import (
    DEFAULT_STAGE1_THRESHOLD,
    Snippet,
    build_retry_message,
    default_model_dir,
    load_stage1_model,
    matched_debt_keywords,
    split_into_windows,
    stage1_admit,
    strip_inline_xbrl_prologue,
    triage_filing,
    validate_verdict,
)


class FakeClient:
    """Chat client returning canned responses in order."""

    def __init__(self: Self, responses: list[str]) -> None:
        """Store the responses to hand back in order."""
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self: Self, *, messages: list[dict[str, str]], model: str, reasoning_effort: str
    ) -> str:
        """Return the next canned response."""
        del model, reasoning_effort
        self.calls.append(list(messages))
        return self.responses.pop(0)


class ExplodingClient:
    """Chat client that always raises."""

    async def complete(
        self: Self, *, messages: list[dict[str, str]], model: str, reasoning_effort: str
    ) -> str:
        """Raise, to exercise the degradation path."""
        del messages, model, reasoning_effort
        raise RuntimeError("provider down")


def _snippets(count: int) -> list[Snippet]:
    """Build placeholder snippets."""
    return [
        Snippet(snippet_id=f"s{index}", text=f"text {index}", score=0.5)
        for index in range(1, count + 1)
    ]


def test_windows_respect_the_token_budget() -> None:
    """No window exceeds the target size."""
    text = "\n\n".join(f"Paragraph {index} of the filing body." for index in range(40))
    windows = split_into_windows(text, target_tokens=60)
    assert windows
    assert all(window.token_count <= 60 for window in windows)


def test_windows_cover_the_text_without_loss() -> None:
    """Concatenating window spans reproduces every non-space character."""
    text = "First para.\n\nSecond para is longer.\n\nThird para ends it."
    windows = split_into_windows(text, target_tokens=8)
    joined = "".join(window.text for window in windows)
    assert "".join(joined.split()) == "".join(text.split())


def test_keyword_gate_matches_plurals() -> None:
    """The gate lemmatises trailing plurals so it does not miss filings."""
    assert matched_debt_keywords(
        "amended its credit agreements and two term loans"
    ) == (
        "credit agreement",
        "term loan",
    )
    assert matched_debt_keywords("the debentures were issued") == ("debenture",)


def test_xbrl_prologue_stripped_but_prose_untouched() -> None:
    """A tag-dominated prologue goes; ordinary prose is returned unchanged."""
    prose = "The Company entered into a term loan with the bank on that date."
    assert strip_inline_xbrl_prologue(prose) == prose
    prologue = "\n".join(
        ["ifrs-full:PropertyPlantAndEquipmentMember", "0001234567", "2025-06-30"] * 9
    )
    stripped = strip_inline_xbrl_prologue(f"{prologue}\n{prose}")
    assert stripped.strip() == prose


def test_stage1_admits_only_at_or_above_threshold() -> None:
    """Windows below the cutoff are discarded."""

    class Model:
        def decision_function(self: Self, texts: list[str]) -> list[float]:
            return [5.0 if "keep" in text else -5.0 for text in texts]

    admitted = stage1_admit(Model(), [("a", "keep me"), ("b", "drop me")])
    assert [snippet.snippet_id for snippet in admitted] == ["a"]
    assert admitted[0].score > DEFAULT_STAGE1_THRESHOLD


def test_stage1_handles_no_input() -> None:
    """An empty batch does not reach the model."""
    assert stage1_admit(object(), []) == []


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ({"keep": [1, 2], "drop": []}, []),
        ({"keep": [1], "drop": [{"id": 2, "reason": "no_details"}]}, []),
        ({"keep": [1], "drop": []}, ["no verdict for snippet 2"]),
        (
            {"keep": [1, 2], "drop": [{"id": 2, "reason": "no_details"}]},
            ["snippet 2 appears in both keep and drop"],
        ),
        (
            {"keep": [1, 2, 7], "drop": []},
            ["snippet 7 does not exist; ids run 1 to 2"],
        ),
        (
            {"keep": [1], "drop": [{"id": 2, "reason": "duplicate"}]},
            ["snippet 2 dropped as a duplicate without a covered_by id"],
        ),
        (
            {
                "keep": [1],
                "drop": [{"id": 2, "reason": "duplicate", "covered_by": 7}],
            },
            ["snippet 2 dropped as covered by snippet 7, which is not being kept"],
        ),
        (
            {
                "keep": [1],
                "drop": [{"id": 2, "reason": "duplicate", "covered_by": 0}],
            },
            ["snippet 2 dropped as covered by snippet 0, which is not being kept"],
        ),
        (
            {
                "keep": [],
                "drop": [
                    {"id": 1, "reason": "no_details"},
                    {"id": 2, "reason": "duplicate", "covered_by": 1},
                ],
            },
            ["snippet 2 dropped as covered by snippet 1, which is not being kept"],
        ),
        ("not an object", ["response was not a JSON object"]),
    ],
)
def test_validate_verdict(verdict: object, expected: list[str]) -> None:
    """Every malformed shape is reported, and a good one passes."""
    assert validate_verdict(verdict, 2) == expected


def test_retry_message_names_every_failure() -> None:
    """The corrective turn lists the failures and the id range."""
    message = build_retry_message(["no verdict for snippet 2"], 3)
    assert "no verdict for snippet 2" in message
    assert "1 to 3" in message


def test_triage_resolves_indices_to_snippet_ids() -> None:
    """Keeps, no-detail drops and duplicates all come back as snippet ids."""
    client = FakeClient(
        [
            json.dumps(
                {
                    "keep": [1],
                    "drop": [
                        {"id": 2, "reason": "no_details"},
                        {"id": 3, "reason": "duplicate", "covered_by": 1},
                    ],
                }
            )
        ]
    )
    verdict = asyncio.run(triage_filing(client, "acc-1", _snippets(3)))
    assert verdict.kept == ["s1"]
    assert verdict.dropped_no_details == ["s2"]
    assert verdict.dropped_duplicate == [("s3", "s1")]
    assert verdict.attempts == 1
    assert verdict.error is None


def test_triage_retries_a_malformed_verdict() -> None:
    """A verdict missing an id is fed back and the second answer is used."""
    client = FakeClient(
        [
            json.dumps({"keep": [1], "drop": []}),
            json.dumps({"keep": [1, 2], "drop": []}),
        ]
    )
    verdict = asyncio.run(triage_filing(client, "acc-1", _snippets(2)))
    assert verdict.kept == ["s1", "s2"]
    assert verdict.attempts == 2
    assert "not usable" in client.calls[1][-1]["content"]


def test_triage_retries_an_out_of_range_covered_by() -> None:
    """A covered_by outside the id range is fed back for retry, not an IndexError."""
    bad = {"keep": [1], "drop": [{"id": 2, "reason": "duplicate", "covered_by": 7}]}
    good = {"keep": [1], "drop": [{"id": 2, "reason": "duplicate", "covered_by": 1}]}
    client = FakeClient([json.dumps(bad), json.dumps(good)])
    verdict = asyncio.run(triage_filing(client, "acc-1", _snippets(2)))
    assert verdict.kept == ["s1"]
    assert verdict.dropped_duplicate == [("s2", "s1")]
    assert verdict.attempts == 2
    assert verdict.error is None


def test_triage_keeps_everything_when_attempts_run_out() -> None:
    """Exhausting retries degrades to stage-1 behaviour rather than losing data."""
    client = FakeClient([json.dumps({"keep": [1], "drop": []})] * 3)
    verdict = asyncio.run(triage_filing(client, "acc-1", _snippets(2), max_attempts=3))
    assert verdict.kept == ["s1", "s2"]
    assert verdict.error is not None


def test_triage_keeps_everything_when_the_provider_fails() -> None:
    """A transport error must not silently drop a filing's snippets."""
    verdict = asyncio.run(triage_filing(ExplodingClient(), "acc-1", _snippets(2)))
    assert verdict.kept == ["s1", "s2"]
    assert "provider down" in (verdict.error or "")


def test_default_model_dir_sits_under_the_data_dir(tmp_path: Path) -> None:
    """The stage-1 path derives from DATA_DIR, as the 8-K classifier's does."""
    assert default_model_dir(tmp_path) == (
        tmp_path / "models" / "sixk" / "stage1-tfidf-linear-svc"
    )


def test_load_stage1_model_reports_a_missing_artifact(tmp_path: Path) -> None:
    """A clear error beats an opaque pickle failure."""
    with pytest.raises(FileNotFoundError, match="no stage-1 model"):
        load_stage1_model(tmp_path / "absent")


def test_shipped_artifact_loads_with_its_calibrated_threshold() -> None:
    """The committed stage-1 artifact carries the threshold it was tuned with.

    Reads the repo path directly: conftest points ``settings.DATA_DIR`` at a tmp
    directory for every test, so ``default_model_dir()`` would not find it.
    """
    model, threshold = load_stage1_model(
        default_model_dir(settings.PROJECT_ROOT / "data")
    )
    assert threshold == DEFAULT_STAGE1_THRESHOLD
    assert hasattr(model, "decision_function")
