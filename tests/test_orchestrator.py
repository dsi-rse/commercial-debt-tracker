"""Tests for the cdt-orchestrator daily/poll entrypoints."""

from __future__ import annotations

from pathlib import Path

import pytest

import cdt.orchestrator as orch
from cdt.extractor import ExtractTickResult


def test_poll_finalizes_on_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed poll tick runs match + finalize after advancing the job."""
    calls: list[str] = []
    monkeypatch.setattr(orch, "OpenAIBatchClient", lambda: object())

    def fake_advance(**kwargs: object) -> ExtractTickResult:
        del kwargs
        calls.append("advance")
        return ExtractTickResult(status="completed", job_id="J")

    monkeypatch.setattr(orch, "advance_extract_job", fake_advance)
    monkeypatch.setattr(
        orch, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )

    assert orch.main(["--artifact-root", str(tmp_path), "poll"]) == 0
    assert calls == ["advance", "match"]


def test_poll_skips_finalize_when_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-running poll tick does not run match/finalize."""
    calls: list[str] = []
    monkeypatch.setattr(orch, "OpenAIBatchClient", lambda: object())

    def fake_advance(**kwargs: object) -> ExtractTickResult:
        del kwargs
        calls.append("advance")
        return ExtractTickResult(status="waiting")

    monkeypatch.setattr(orch, "advance_extract_job", fake_advance)
    monkeypatch.setattr(
        orch, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )

    assert orch.main(["--artifact-root", str(tmp_path), "poll"]) == 0
    assert calls == ["advance"]


def test_daily_batch_defers_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily with the batch backend prepares + finalizes but never runs the full pipeline."""
    calls: list[str] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: (calls.append(f"prepare:{config.mode}"), str(tmp_path))[1],
    )
    monkeypatch.setattr(
        orch, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )
    monkeypatch.setattr(
        orch,
        "run_pipeline",
        lambda config: pytest.fail("run_pipeline must not run for daily batch"),
    )

    assert (
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])
        == 0
    )
    assert calls == ["prepare:daily", "match"]


def test_daily_live_runs_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily with the live backend runs the original synchronous pipeline."""
    calls: list[str] = []

    class Result:
        artifact_root = str(tmp_path)

    monkeypatch.setattr(
        orch, "run_pipeline", lambda config: (calls.append("pipeline"), Result())[1]
    )
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: pytest.fail("prepare-only must not run for live backend"),
    )

    assert (
        orch.main(
            [
                "--extractor-backend",
                "live",
                "--artifact-root",
                str(tmp_path),
                "daily",
                "--cik-file",
                "c.txt",
            ]
        )
        == 0
    )
    assert calls == ["pipeline"]
