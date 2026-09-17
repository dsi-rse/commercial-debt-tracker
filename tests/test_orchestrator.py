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
from cdt.pipeline import DEFAULT_GENRES, GENRE_6K, GENRE_8K


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


def test_daily_batch_does_not_run_while_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily fails loudly, without preparing, when another run holds the lease.

    Prepare rewrites the same completion registries a poll tick's finalize does;
    running it unserialized loses registry updates (#88).
    """
    monkeypatch.setattr(orch, "LEASE_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "prepare must not run while the lease is held"
        ),
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
        == 1
    )


def test_daily_batch_holds_lease_through_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prepare stages run under the writer lease, serialized with poll ticks (#88)."""
    calls: list[str] = []

    def fake_prepare(config: object, **kwargs: object) -> str:
        del config, kwargs
        assert (
            acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE) is None
        ), "prepare must run while the lease is held"
        calls.append("prepare")
        return str(tmp_path)

    monkeypatch.setattr(orch, "run_prepare_stages", fake_prepare)
    monkeypatch.setattr(
        orch, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )

    assert (
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])
        == 0
    )
    assert calls == ["prepare", "match"]
    # Released on the way out, so the next scheduled run proceeds immediately.
    assert acquire_lease(tmp_path, orch.PIPELINE_WRITER_LEASE) is not None


def test_daily_batch_defers_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily with the batch backend prepares + finalizes but never runs the full pipeline."""
    calls: list[str] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: (
            calls.append(f"prepare:{config.mode}"),
            str(tmp_path),
        )[1],
    )
    monkeypatch.setattr(
        orch, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )
    monkeypatch.setattr(
        orch,
        "run_pipeline",
        lambda config, **kwargs: pytest.fail(
            "run_pipeline must not run for daily batch"
        ),
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
        orch,
        "run_pipeline",
        lambda config, **kwargs: (calls.append("pipeline"), Result())[1],
    )
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "prepare-only must not run for live backend"
        ),
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
    monkeypatch.setattr(
        orch, "run_prepare_stages", lambda config, **kwargs: str(tmp_path)
    )
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
        lambda config, **kwargs: (
            seen.append(config.extract_batch_size),
            str(tmp_path),
        )[1],
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
        lambda config, **kwargs: (
            calls.append(f"prepare:{config.mode}"),
            str(tmp_path),
        )[1],
    )
    monkeypatch.setattr(
        orch, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )
    monkeypatch.setattr(
        orch,
        "run_pipeline",
        lambda config, **kwargs: pytest.fail(
            "run_pipeline must not run for historical batch"
        ),
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
        lambda config, **kwargs: (
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
        orch,
        "run_pipeline",
        lambda config, **kwargs: (calls.append("pipeline"), Result())[1],
    )
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "prepare-only must not run for live backend"
        ),
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


def test_historical_batch_does_not_run_while_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical fails loudly, without preparing, when another run holds the lease.

    Prepare rewrites the same completion registries a poll tick's finalize does;
    running it unserialized loses registry updates (#88).
    """
    monkeypatch.setattr(orch, "LEASE_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "prepare must not run while the lease is held"
        ),
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
        == 1
    )


def test_scheduled_runs_prepare_both_genres_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deployed daily run acquires whatever the CIKs filed, not just 8-K."""
    captured: list[object] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: (captured.append(config), str(tmp_path))[1],
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    assert (
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "daily",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2026-09-08",
                "--end-date",
                "2026-09-08",
            ]
        )
        == 0
    )
    assert captured[0].genres == DEFAULT_GENRES
    # Unset means "the run's CIKs", so one list covers both genres.
    assert captured[0].sixk_cik_file is None


def test_genres_can_be_narrowed_on_a_scheduled_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run deliberately about one genre says so, and the config carries it."""
    captured: list[object] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: (captured.append(config), str(tmp_path))[1],
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    assert (
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "--genres",
                "6-K",
                "--sixk-cik-file",
                "fpi.txt",
                "historical",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2026-09-01",
                "--end-date",
                "2026-09-08",
            ]
        )
        == 0
    )
    assert captured[0].genres == (GENRE_6K,)
    assert captured[0].sixk_cik_file == "fpi.txt"


def test_genres_come_from_the_environment_when_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task definitions set env, not argv, so GENRES has to reach the config."""
    monkeypatch.setenv("GENRES", "8-K")
    captured: list[object] = []
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: (captured.append(config), str(tmp_path))[1],
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    assert (
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "daily",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2026-09-08",
                "--end-date",
                "2026-09-08",
            ]
        )
        == 0
    )
    assert captured[0].genres == (GENRE_8K,)


def test_an_unknown_genre_fails_before_any_stage_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Argparse rejects it, so a typo cannot silently narrow a scheduled run."""
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail("prepare must not run"),
    )

    with pytest.raises(SystemExit):
        orch.main(
            [
                "--artifact-root",
                str(tmp_path),
                "--genres",
                "10-K",
                "daily",
                "--cik-file",
                "c.txt",
                "--start-date",
                "2026-09-08",
                "--end-date",
                "2026-09-08",
            ]
        )


def test_placeholder_secret_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-placeholder API key exits immediately, before any stage runs."""
    monkeypatch.setenv("OPENAI_API_KEY", "PLACEHOLDER-set-via-aws-ssm-put-parameter")
    monkeypatch.setattr(
        orch,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "stages must not run with a placeholder key"
        ),
    )

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])


def test_real_secret_values_pass_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary key values do not trip the placeholder guard."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setattr(
        orch, "run_prepare_stages", lambda config, **kwargs: str(tmp_path)
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    assert (
        orch.main(["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"])
        == 0
    )


def test_live_backend_skips_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live pipeline must not interleave with a running poll tick."""
    monkeypatch.setattr(orch, "LEASE_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        orch,
        "run_pipeline",
        lambda config, **kwargs: pytest.fail(
            "live run must not start while the lease is held"
        ),
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

    monkeypatch.setattr(orch, "run_pipeline", lambda config, **kwargs: Result())

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


def test_poll_aborts_when_lease_stolen_mid_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick whose lease was stolen must not run match/finalize (#89)."""
    from cdt.lease import lease_path
    from cdt.storage import read_json_artifact, write_json_artifact

    monkeypatch.setattr(orch, "OpenAIBatchClient", lambda: object())

    def fake_advance(**kwargs: object) -> ExtractTickResult:
        del kwargs
        # Simulate a tick that outlasted its TTL: another run steals the lease
        # mid-flight, so the post-tick renewal must fail.
        path = lease_path(tmp_path, orch.PIPELINE_WRITER_LEASE)
        payload = dict(read_json_artifact(path))
        payload["holder"] = "thief"
        write_json_artifact(path, payload)
        return ExtractTickResult(status="completed", job_id="J")

    monkeypatch.setattr(orch, "advance_extract_job", fake_advance)
    monkeypatch.setattr(
        orch,
        "run_match_and_finalize",
        lambda **kwargs: pytest.fail("must not finalize on a stolen lease"),
    )

    assert orch.main(["--artifact-root", str(tmp_path), "poll"]) == 1


def test_daily_batch_logs_heartbeat_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    propagate_logger: Callable[[logging.Logger], None],
) -> None:
    """A completed daily run logs the exact literal the heartbeat alarm counts (#85)."""
    propagate_logger(orch.LOGGER)
    monkeypatch.setattr(orch, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        orch, "run_prepare_stages", lambda config, **kwargs: str(tmp_path)
    )
    monkeypatch.setattr(orch, "run_match_and_finalize", lambda **kwargs: None)

    with caplog.at_level("INFO"):
        assert (
            orch.main(
                ["--artifact-root", str(tmp_path), "daily", "--cik-file", "c.txt"]
            )
            == 0
        )

    assert any(
        "Orchestrator run complete: mode=daily" in record.message
        for record in caplog.records
    )


def test_runtime_watchdog_reaps_past_deadline() -> None:
    """The watchdog hard-exits a run that outlives its deadline (#93)."""
    exits: list[int] = []

    timer = orch.start_runtime_watchdog("poll", 0.05 / 3600, exit_fn=exits.append)
    timer.join(timeout=5)

    assert exits == [orch.WATCHDOG_EXIT_CODE]


def test_runtime_watchdog_defaults_per_mode() -> None:
    """Without an override, the deadline comes from the mode table (#93)."""
    exits: list[int] = []

    timer = orch.start_runtime_watchdog("historical", exit_fn=exits.append)
    try:
        assert timer.interval == orch.MODE_DEADLINE_HOURS["historical"] * 3600
    finally:
        timer.cancel()
    assert exits == []
