"""Tests for the cdt-orchestrator daily/historical/poll entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

import cdt.orchestrator as orch
from cdt.extractor import ExtractTickResult
from cdt.lease import acquire_lease


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


def test_poll_skips_tick_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poll tick that cannot acquire the pipeline-writer lease does nothing."""
    monkeypatch.setattr(
        orch,
        "advance_extract_job",
        lambda **kwargs: pytest.fail("tick must not run while the lease is held"),
    )
    held = acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE)
    assert held is not None

    assert orch.main(["--artifact-root", str(tmp_path), "poll"]) == 0


def test_poll_releases_lease_after_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lease is released even on an ordinary tick, freeing the next run."""
    monkeypatch.setattr(orch, "OpenAIBatchClient", lambda: object())
    monkeypatch.setattr(
        orch,
        "advance_extract_job",
        lambda **kwargs: ExtractTickResult(status="waiting"),
    )

    assert orch.main(["--artifact-root", str(tmp_path), "poll"]) == 0

    assert acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE) is not None


def test_daily_batch_skips_match_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily skips match/finalize (but still prepares) when a poll holds the lease."""
    calls: list[str] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: (calls.append("prepare"), tmp_path)[1],
    )
    monkeypatch.setattr(
        orch,
        "run_match_and_finalize",
        lambda **kwargs: pytest.fail("match must not run while the lease is held"),
    )
    held = acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE)
    assert held is not None

    assert (
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])
        == 0
    )
    assert calls == ["prepare"]


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


def test_daily_batch_warns_on_extract_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """--extract-batch-size is inert under the batch backend, so it must warn."""
    propagate_logger(orch.LOGGER)
    # main() calls basicConfig(force=True), which would drop caplog's handler.
    monkeypatch.setattr(orch, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(orch, "run_prepare_stages", lambda config: str(tmp_path))
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    with caplog.at_level("WARNING"):
        assert (
            orch.main(
                [
                    "--artifact-root",
                    str(tmp_path),
                    "daily",
                    "--cik-file",
                    "c.txt",
                    "--extract-batch-size",
                    "25",
                ]
            )
            == 0
        )

    assert "Ignoring --extract-batch-size=25" in caplog.text


def test_daily_batch_quiet_without_extract_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """The unset default must not warn, and still reaches the pipeline config."""
    propagate_logger(orch.LOGGER)
    monkeypatch.setattr(orch, "configure_logging", lambda **kwargs: None)
    seen: list[int] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: (seen.append(config.extract_batch_size), str(tmp_path))[1],
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    with caplog.at_level("WARNING"):
        assert (
            orch.main(
                ["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"]
            )
            == 0
        )

    assert seen == [orch.DEFAULT_STAGE_BATCH_SIZE]
    assert "--extract-batch-size" not in caplog.text


def test_historical_batch_defers_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical with the batch backend prepares + finalizes; poll owns extraction."""
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
        lambda config: pytest.fail("run_pipeline must not run for historical batch"),
    )

    assert (
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "historical",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-31",
            ]
        )
        == 0
    )
    assert calls == ["prepare:historical", "match"]


def test_historical_batch_passes_dates_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deferred-extract path keeps historical's explicit date range."""
    seen: list[tuple[object, object]] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: (
            seen.append((config.start_date, config.end_date)),
            str(tmp_path),
        )[1],
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    assert (
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "historical",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-31",
            ]
        )
        == 0
    )
    assert seen == [(date(2024, 1, 1), date(2024, 1, 31))]


def test_historical_live_runs_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical with the live backend keeps the original synchronous pipeline."""
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
                "historical",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-31",
            ]
        )
        == 0
    )
    assert calls == ["pipeline"]


def test_historical_batch_skips_match_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical skips match/finalize (but still prepares) when poll holds the lease."""
    calls: list[str] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: (calls.append("prepare"), tmp_path)[1],
    )
    monkeypatch.setattr(
        orch,
        "run_match_and_finalize",
        lambda **kwargs: pytest.fail("match must not run while the lease is held"),
    )
    held = acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE)
    assert held is not None

    assert (
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "historical",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-31",
            ]
        )
        == 0
    )
    assert calls == ["prepare"]


def test_placeholder_secret_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-placeholder API key exits immediately, before any stage runs."""
    monkeypatch.setenv("OPENAI_API_KEY", "PLACEHOLDER-set-via-aws-ssm-put-parameter")
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config: pytest.fail("stages must not run with a placeholder key"),
    )

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])


def test_real_secret_values_pass_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary key values do not trip the placeholder guard."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setattr(orch, "run_prepare_stages", lambda config: str(tmp_path))
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    assert (
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])
        == 0
    )


def test_live_backend_skips_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live pipeline must not interleave with a running poll tick."""
    monkeypatch.setattr(
        orch,
        "run_pipeline",
        lambda config: pytest.fail("live run must not start while the lease is held"),
    )
    held = acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE)
    assert held is not None

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
        == 1
    )


def test_live_backend_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live run's lease is released even on an ordinary exit."""

    class Result:
        artifact_root = str(tmp_path)

    monkeypatch.setattr(orch, "run_pipeline", lambda config: Result())

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
    assert acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE) is not None
