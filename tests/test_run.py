"""Tests for ``cdt run daily|historical|poll``: leases, backends, genres, heartbeats."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

import cdt.run as run_module
from cdt import cli
from cdt.datasets import GENRE_6K, GENRE_8K
from cdt.extractor import ExtractTickResult
from cdt.lease import PIPELINE_WRITER_LEASE, acquire_lease
from cdt.pipeline import (
    DEFAULT_GENRES,
    DEFAULT_STAGE_BATCH_SIZE,
    PipelineConfig,
    PipelineRunResult,
    PrepareResult,
)

ARGPARSE_USAGE_ERROR = 2


@pytest.fixture(autouse=True)
def armed_watchdogs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    """Record the watchdog ``cdt run`` arms instead of starting a real timer."""
    armed: list[tuple[str, object]] = []
    monkeypatch.setattr(
        cli,
        "start_runtime_watchdog",
        lambda mode, hours=None: armed.append((mode, hours)),
    )
    return armed


@pytest.fixture
def keep_caplog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop ``cli.main`` reconfiguring the root logger, which drops caplog's handler."""
    monkeypatch.setattr(cli, "configure_logging", lambda **kwargs: None)


def _poll(tmp_path: Path, *extra: str) -> list[str]:
    return ["run", "poll", "--artifact-root", str(tmp_path), *extra]


def _daily(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "run",
        "daily",
        "--artifact-root",
        str(tmp_path),
        "--cik-file",
        "c.txt",
        *extra,
    ]


def _historical(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "run",
        "historical",
        "--artifact-root",
        str(tmp_path),
        "--cik-file",
        "c.txt",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-31",
        *extra,
    ]


def _result(tmp_path: Path, failed_genres: tuple[str, ...] = ()) -> PipelineRunResult:
    return PipelineRunResult(
        mode="daily",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        extracted_rows=0,
        matched_rows=0,
        debt_instrument_rows=0,
        artifact_root=str(tmp_path),
        extractor_run_path=str(tmp_path / "run.jsonl"),
        failed_genres=failed_genres,
    )


def _patch_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configs: list[PipelineConfig] | None = None,
    calls: list[str] | None = None,
    failed_genres: tuple[str, ...] = (),
) -> None:
    """Make prepare record its config, and match/finalize record that it ran."""
    seen = configs if configs is not None else []
    steps = calls if calls is not None else []

    def prepare(config: PipelineConfig, **kwargs: object) -> PrepareResult:
        del kwargs
        seen.append(config)
        steps.append(f"prepare:{config.mode}")
        return PrepareResult(str(tmp_path), failed_genres)

    monkeypatch.setattr(run_module, "run_prepare_stages", prepare)
    monkeypatch.setattr(
        run_module, "run_match_and_finalize", lambda **kwargs: steps.append("match")
    )


def _patch_tick(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    seen: list[dict[str, object]] | None = None,
) -> None:
    record = seen if seen is not None else []
    monkeypatch.setattr(run_module, "OpenAIBatchClient", lambda: object())
    monkeypatch.setattr(
        run_module,
        "advance_extract_job",
        lambda **kwargs: (
            record.append(kwargs),
            ExtractTickResult(status=status, job_id="J"),
        )[1],
    )


# --- poll ---------------------------------------------------------------------


def test_poll_finalizes_on_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed poll tick runs match + finalize after advancing the job."""
    calls: list[str] = []
    ticks: list[dict[str, object]] = []
    _patch_tick(monkeypatch, "completed", ticks)
    monkeypatch.setattr(
        run_module, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )

    assert cli.main(_poll(tmp_path)) == 0
    assert len(ticks) == 1
    assert calls == ["match"]


def test_poll_skips_finalize_when_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-running poll tick does not run match/finalize."""
    _patch_tick(monkeypatch, "waiting")
    monkeypatch.setattr(
        run_module,
        "run_match_and_finalize",
        lambda **kwargs: pytest.fail("a waiting tick must not finalize"),
    )

    assert cli.main(_poll(tmp_path)) == 0


def test_poll_passes_only_the_limits_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Limits not given are left out, so the tick keeps the backend's defaults."""
    ticks: list[dict[str, object]] = []
    _patch_tick(monkeypatch, "idle", ticks)

    status = cli.main(
        _poll(tmp_path, "--max-rows-per-job", "7", "--max-attempts", "2", "--force")
    )

    assert status == 0
    assert ticks[0]["max_rows_per_job"] == 7
    assert ticks[0]["max_attempts"] == 2
    assert ticks[0]["force"] is True
    assert "max_requests_per_batch" not in ticks[0]
    assert "max_batch_bytes" not in ticks[0]


def test_poll_skips_tick_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A poll tick that cannot acquire the pipeline-writer lease does nothing."""
    monkeypatch.setattr(
        run_module,
        "advance_extract_job",
        lambda **kwargs: pytest.fail("tick must not run while the lease is held"),
    )
    held = acquire_lease(tmp_path, PIPELINE_WRITER_LEASE)
    assert held is not None

    assert cli.main(_poll(tmp_path)) == 0
    assert capsys.readouterr().out.strip() == "locked"


def test_poll_releases_lease_after_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lease is released even on an ordinary tick, freeing the next run."""
    _patch_tick(monkeypatch, "waiting")

    assert cli.main(_poll(tmp_path)) == 0

    assert acquire_lease(tmp_path, PIPELINE_WRITER_LEASE) is not None


def test_poll_aborts_when_lease_stolen_mid_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick whose lease was stolen must not run match/finalize."""
    from cdt.lease import lease_path
    from cdt.storage.objects import read_json_artifact, write_json_artifact

    monkeypatch.setattr(run_module, "OpenAIBatchClient", lambda: object())

    def fake_advance(**kwargs: object) -> ExtractTickResult:
        del kwargs
        # A tick that outlasted its TTL: another run steals the lease mid-flight,
        # so the post-tick renewal must fail.
        path = lease_path(tmp_path, PIPELINE_WRITER_LEASE)
        payload = dict(read_json_artifact(path))
        payload["holder"] = "thief"
        write_json_artifact(path, payload)
        return ExtractTickResult(status="completed", job_id="J")

    monkeypatch.setattr(run_module, "advance_extract_job", fake_advance)
    monkeypatch.setattr(
        run_module,
        "run_match_and_finalize",
        lambda **kwargs: pytest.fail("must not finalize on a stolen lease"),
    )

    assert cli.main(_poll(tmp_path)) == 1


def test_poll_is_unaffected_by_a_bad_genres_environment_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GENRES`` only feeds the prepare modes, so a bad value cannot stop polling."""
    monkeypatch.setenv("GENRES", "10-K")
    _patch_tick(monkeypatch, "idle")

    assert cli.main(_poll(tmp_path)) == 0


def test_poll_logs_its_liveness_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    keep_caplog: None,
) -> None:
    """Every tick logs the literal the poll-liveness alarm counts."""
    _patch_tick(monkeypatch, "waiting")

    with caplog.at_level("INFO"):
        assert cli.main(_poll(tmp_path)) == 0

    assert "Poll tick complete: status=waiting" in caplog.text


# --- daily / historical, batch backend ---------------------------------------


@pytest.mark.parametrize("argv", [_daily, _historical])
def test_batch_run_does_not_start_while_lease_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: Callable[[Path], list[str]],
) -> None:
    """A prepare run fails loudly, without preparing, when another run holds the lease.

    Prepare rewrites the same completion registries a poll tick's finalize does,
    so running it unserialized would lose registry updates.
    """
    monkeypatch.setattr(run_module, "LEASE_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        run_module,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "prepare must not run while the lease is held"
        ),
    )
    monkeypatch.setattr(
        run_module,
        "run_match_and_finalize",
        lambda **kwargs: pytest.fail("match must not run while the lease is held"),
    )
    held = acquire_lease(tmp_path, PIPELINE_WRITER_LEASE)
    assert held is not None

    assert cli.main(argv(tmp_path)) == 1


def test_daily_batch_holds_lease_through_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prepare stages run under the writer lease, serialized with poll ticks."""
    calls: list[str] = []

    def fake_prepare(config: object, **kwargs: object) -> PrepareResult:
        del config, kwargs
        assert (
            acquire_lease(tmp_path, PIPELINE_WRITER_LEASE) is None
        ), "prepare must run while the lease is held"
        calls.append("prepare")
        return PrepareResult(str(tmp_path))

    monkeypatch.setattr(run_module, "run_prepare_stages", fake_prepare)
    monkeypatch.setattr(
        run_module, "run_match_and_finalize", lambda **kwargs: calls.append("match")
    )

    assert cli.main(_daily(tmp_path)) == 0
    assert calls == ["prepare", "match"]
    # Released on the way out, so the next scheduled run proceeds immediately.
    assert acquire_lease(tmp_path, PIPELINE_WRITER_LEASE) is not None


@pytest.mark.parametrize(
    ("argv", "mode"), [(_daily, "daily"), (_historical, "historical")]
)
def test_batch_backend_defers_extract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: Callable[[Path], list[str]],
    mode: str,
) -> None:
    """The batch backend prepares + publishes but never runs the full pipeline."""
    calls: list[str] = []
    _patch_prepare(monkeypatch, tmp_path, calls=calls)
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: pytest.fail("run_pipeline must not run for batch"),
    )

    assert cli.main(argv(tmp_path)) == 0
    assert calls == [f"prepare:{mode}", "match"]


def test_historical_batch_passes_dates_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deferred-extract path keeps historical's explicit date range."""
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    assert cli.main(_historical(tmp_path)) == 0
    assert (configs[0].start_date, configs[0].end_date) == (
        date(2024, 1, 1),
        date(2024, 1, 31),
    )


def test_historical_requires_its_dates() -> None:
    """A backfill is never run over an implied range."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "historical", "--cik-file", "c.txt"])

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR


def test_daily_without_a_cik_file_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --cik-file and no CDT_DEFAULT_CIK_FILE fails before any stage runs."""
    monkeypatch.delenv("CDT_DEFAULT_CIK_FILE", raising=False)
    monkeypatch.setattr(
        run_module,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail("prepare must not run"),
    )

    assert (
        cli.main(["run", "daily", "--artifact-root", str(tmp_path)])
        == ARGPARSE_USAGE_ERROR
    )


def test_the_cik_file_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task definitions set env, not argv, so CDT_DEFAULT_CIK_FILE reaches the config."""
    monkeypatch.setenv("CDT_DEFAULT_CIK_FILE", "s3://bucket/ciks.txt")
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    assert cli.main(["run", "daily", "--artifact-root", str(tmp_path)]) == 0
    assert configs[0].cik_file == "s3://bucket/ciks.txt"


def test_the_artifact_root_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARTIFACT_ROOT supplies the root when --artifact-root is not passed."""
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "from-env"))
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    assert cli.main(["run", "daily", "--cik-file", "c.txt"]) == 0
    assert configs[0].artifact_root == str(tmp_path / "from-env")


def test_daily_batch_warns_on_extract_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    keep_caplog: None,
) -> None:
    """--extract-batch-size is inert under the batch backend, so it must warn."""
    _patch_prepare(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        assert cli.main(_daily(tmp_path, "--extract-batch-size", "25")) == 0

    assert "Ignoring --extract-batch-size=25" in caplog.text


def test_daily_batch_quiet_without_extract_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    keep_caplog: None,
) -> None:
    """The unset default must not warn, and still reaches the pipeline config."""
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    with caplog.at_level("WARNING"):
        assert cli.main(_daily(tmp_path)) == 0

    assert configs[0].extract_batch_size == DEFAULT_STAGE_BATCH_SIZE
    assert "--extract-batch-size" not in caplog.text


def test_stage_batch_sizes_reach_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each stage's batch size flag lands on its own config field."""
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    status = cli.main(
        _daily(
            tmp_path,
            "--ingest-batch-size",
            "3",
            "--segment-batch-size",
            "4",
            "--classify-batch-size",
            "5",
            "--match-batch-size",
            "6",
        )
    )

    assert status == 0
    config = configs[0]
    assert (
        config.ingest_batch_size,
        config.segment_batch_size,
        config.classify_batch_size,
        config.match_batch_size,
    ) == (3, 4, 5, 6)


# --- genres -------------------------------------------------------------------


def test_scheduled_runs_prepare_both_genres_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deployed daily run acquires whatever the CIKs filed, not just 8-K."""
    monkeypatch.delenv("GENRES", raising=False)
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    assert cli.main(_daily(tmp_path)) == 0
    assert configs[0].genres == DEFAULT_GENRES


def test_genres_can_be_narrowed_on_a_scheduled_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run deliberately about one genre says so, and the config carries it."""
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    assert cli.main(_historical(tmp_path, "--genres", "6-K")) == 0
    assert configs[0].genres == (GENRE_6K,)


def test_genres_come_from_the_environment_when_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task definitions set env, not argv, so GENRES has to reach the config."""
    monkeypatch.setenv("GENRES", "8-K")
    configs: list[PipelineConfig] = []
    _patch_prepare(monkeypatch, tmp_path, configs)

    assert cli.main(_daily(tmp_path)) == 0
    assert configs[0].genres == (GENRE_8K,)


@pytest.mark.parametrize("via_env", [False, True])
def test_an_unknown_genre_fails_before_any_stage_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via_env: bool
) -> None:
    """Argparse rejects it, flag or env, so a typo cannot silently narrow a run."""
    monkeypatch.setattr(
        run_module,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail("prepare must not run"),
    )
    if via_env:
        monkeypatch.setenv("GENRES", "10-K")
    argv = _daily(tmp_path) if via_env else _daily(tmp_path, "--genres", "10-K")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR


# --- secrets and the watchdog -------------------------------------------------


def test_placeholder_secret_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-placeholder API key exits immediately, before any stage runs."""
    monkeypatch.setenv("OPENAI_API_KEY", "PLACEHOLDER-set-via-aws-ssm-put-parameter")
    monkeypatch.setattr(
        run_module,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail(
            "stages must not run with a placeholder key"
        ),
    )

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        cli.main(_daily(tmp_path))


def test_real_secret_values_pass_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary key values do not trip the placeholder guard."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    _patch_prepare(monkeypatch, tmp_path)

    assert cli.main(_daily(tmp_path)) == 0


@pytest.mark.parametrize(
    ("extra", "expected"),
    [((), ("poll", None)), (("--max-runtime-hours", "0.5"), ("poll", 0.5))],
)
def test_every_run_arms_the_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    armed_watchdogs: list[tuple[str, object]],
    extra: tuple[str, ...],
    expected: tuple[str, object],
) -> None:
    """``cdt run`` arms the watchdog with its mode and any override."""
    _patch_tick(monkeypatch, "idle")

    assert cli.main(_poll(tmp_path, *extra)) == 0
    assert armed_watchdogs == [expected]


def test_runtime_watchdog_reaps_past_deadline() -> None:
    """The watchdog hard-exits a run that outlives its deadline."""
    exits: list[int] = []

    timer = run_module.start_runtime_watchdog("poll", 0.05 / 3600, exit_fn=exits.append)
    timer.join(timeout=5)

    assert exits == [run_module.WATCHDOG_EXIT_CODE]


def test_runtime_watchdog_defaults_per_mode() -> None:
    """Without an override, the deadline comes from the mode table."""
    exits: list[int] = []

    timer = run_module.start_runtime_watchdog("historical", exit_fn=exits.append)
    try:
        assert timer.interval == run_module.MODE_DEADLINE_HOURS["historical"] * 3600
    finally:
        timer.cancel()
    assert exits == []


# --- heartbeats and failed genres --------------------------------------------


def test_a_clean_batch_run_logs_the_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    keep_caplog: None,
) -> None:
    """No failed genre: exit 0 with the literal the daily-heartbeat alarm counts."""
    _patch_prepare(monkeypatch, tmp_path)

    with caplog.at_level("INFO"):
        assert cli.main(_daily(tmp_path)) == 0

    assert "Run complete: mode=daily" in caplog.text


def test_a_failed_genre_still_publishes_but_fails_the_run_without_a_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    keep_caplog: None,
) -> None:
    """Match and publish run over what succeeded; the daily-heartbeat alarm still fires."""
    calls: list[str] = []
    _patch_prepare(monkeypatch, tmp_path, calls=calls, failed_genres=(GENRE_6K,))

    with caplog.at_level("INFO"):
        status = cli.main(_daily(tmp_path))

    assert status == 1
    assert calls == ["prepare:daily", "match"]
    assert "Run finished with failed genres: 6-K" in caplog.text
    assert "Run complete" not in caplog.text


# --- daily / historical, live backend ----------------------------------------


@pytest.mark.parametrize("argv", [_daily, _historical])
def test_live_backend_runs_the_full_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: Callable[..., list[str]],
) -> None:
    """``--extractor-backend live`` runs the synchronous pipeline, not prepare-only."""
    calls: list[str] = []
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: (calls.append(config.mode), _result(tmp_path))[1],
    )
    monkeypatch.setattr(
        run_module,
        "run_prepare_stages",
        lambda config, **kwargs: pytest.fail("prepare-only must not run for live"),
    )

    assert cli.main(argv(tmp_path, "--extractor-backend", "live")) == 0
    assert len(calls) == 1


def test_the_backend_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXTRACTOR_BACKEND=live selects the live backend without a flag."""
    monkeypatch.setenv("EXTRACTOR_BACKEND", "live")
    calls: list[str] = []
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: (calls.append("pipeline"), _result(tmp_path))[1],
    )

    assert cli.main(_daily(tmp_path)) == 0
    assert calls == ["pipeline"]


def test_live_backend_builds_the_whole_pipeline_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stage's option reaches the live run's PipelineConfig."""
    configs: list[PipelineConfig] = []
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: (configs.append(config), _result(tmp_path))[1],
    )

    status = cli.main(
        _historical(
            tmp_path,
            "--extractor-backend",
            "live",
            "--bucket",
            "test-bucket",
            "--download",
            "--item-numbers",
            "1.01,8.01",
            "--model-dir",
            str(tmp_path / "model"),
            "--sixk-model-dir",
            str(tmp_path / "sixk-model"),
            "--concurrency",
            "2",
            "--model",
            "flag/model",
            "--extract-batch-size",
            "9",
            "--final-database-root",
            str(tmp_path / "final"),
        )
    )

    assert status == 0
    config = configs[0]
    assert config.bucket == "test-bucket"
    assert config.download is True
    assert config.item_numbers == ("1.01", "8.01")
    assert config.classifier_model_dir == tmp_path / "model"
    assert config.sixk_model_dir == tmp_path / "sixk-model"
    assert config.sixk_concurrency == 2  # noqa: PLR2004
    assert config.extractor_model == "flag/model"
    assert config.extract_batch_size == 9  # noqa: PLR2004
    assert config.final_database_root == str(tmp_path / "final")


def test_live_backend_leaves_the_model_to_the_extractor_model_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --model the config carries None, which extraction resolves."""
    models: list[object] = []
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: (
            models.append(config.extractor_model),
            _result(tmp_path),
        )[1],
    )

    assert cli.main(_daily(tmp_path, "--extractor-backend", "live")) == 0
    assert models == [None]


def test_a_partial_daily_window_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily needs both dates when either is supplied."""
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: pytest.fail("must not run"),
    )

    status = cli.main(
        _daily(tmp_path, "--extractor-backend", "live", "--start-date", "2024-01-01")
    )

    assert status == ARGPARSE_USAGE_ERROR


def test_live_backend_skips_when_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live pipeline must not interleave with a running poll tick."""
    monkeypatch.setattr(run_module, "LEASE_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: pytest.fail(
            "live run must not start while the lease is held"
        ),
    )
    held = acquire_lease(tmp_path, PIPELINE_WRITER_LEASE)
    assert held is not None

    assert cli.main(_daily(tmp_path, "--extractor-backend", "live")) == 1


def test_live_backend_renews_its_lease_and_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live run hands the pipeline a renewer, then frees the lease on exit."""
    renewals: list[str] = []

    def fake_run_pipeline(
        config: PipelineConfig, *, renew: Callable[[], None] | None = None
    ) -> PipelineRunResult:
        del config
        assert renew is not None
        renew()
        renewals.append("renewed")
        return _result(tmp_path)

    monkeypatch.setattr(run_module, "run_pipeline", fake_run_pipeline)

    assert cli.main(_daily(tmp_path, "--extractor-backend", "live")) == 0
    assert renewals == ["renewed"]
    assert acquire_lease(tmp_path, PIPELINE_WRITER_LEASE) is not None


def test_live_backend_aborts_when_its_lease_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renewal that finds the lease stolen ends the live run with exit 1."""

    def fake_run_pipeline(
        config: PipelineConfig, *, renew: Callable[[], None] | None = None
    ) -> PipelineRunResult:
        del config
        assert renew is not None
        renew()
        return _result(tmp_path)

    monkeypatch.setattr(run_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("cdt.lease.renew_lease", lambda lease: False)

    assert cli.main(_daily(tmp_path, "--extractor-backend", "live")) == 1


def test_a_live_run_with_a_failed_genre_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live backend reports a partial success as a failure too."""
    monkeypatch.setattr(
        run_module,
        "run_pipeline",
        lambda config, **kwargs: _result(tmp_path, failed_genres=(GENRE_8K,)),
    )

    assert cli.main(_daily(tmp_path, "--extractor-backend", "live")) == 1
    assert "Failed genres: 8-K" in capsys.readouterr().out
