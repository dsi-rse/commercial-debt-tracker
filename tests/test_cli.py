"""Tests for the ``cdt`` stage commands: ingest, segment, classify, extract, match, publish."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from support import _seed_classifications

from cdt import cli, settings
from cdt.datasets import GENRE_6K, GENRE_8K
from cdt.extractor.state import ExtractionRowState
from cdt.ingest.core import IngestConfig, IngestRunResult
from cdt.lease import PIPELINE_WRITER_LEASE, acquire_lease
from cdt.pipeline import ALL_TIME_START_DATE, DAILY_LOOKBACK_DAYS
from cdt.segmenter.eightk import POTENTIALLY_RELEVANT_ITEM_NUMBERS

ARGPARSE_USAGE_ERROR = 2

IngestCall = tuple[str, IngestConfig, set[str] | None]


def _ingest_result(config: IngestConfig, *, total_rows: int = 0) -> IngestRunResult:
    return IngestRunResult(
        mode=config.mode,
        start_date=config.start_date,
        end_date=config.end_date,
        ciks_count=1,
        candidates_seen=total_rows,
        skipped_existing=0,
        downloaded=total_rows,
        failures=0,
        total_rows=total_rows,
        output_root=str(config.output_root),
        documents_root=str(config.dataset_name),
        document_partitions=(),
        failure_file="failures.json",
        run_manifest="run.json",
    )


def _recording_acquirer(
    calls: list[IngestCall], label: str
) -> Callable[..., tuple[pd.DataFrame, IngestRunResult]]:
    """Fake acquire function recording the genre-narrowed config it receives."""

    def acquire(
        config: IngestConfig,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
        return_documents: bool = False,
        renew: Callable[[], None] | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del s3_client, return_documents
        calls.append((label, config, ciks))
        return pd.DataFrame(), _ingest_result(config)

    return acquire


@pytest.fixture
def ingest_calls(monkeypatch: pytest.MonkeyPatch) -> list[IngestCall]:
    """Record both genres' acquisitions instead of reaching S3."""
    calls: list[IngestCall] = []
    monkeypatch.setattr(
        "cdt.ingest.genres.acquire_eightk_documents",
        _recording_acquirer(calls, GENRE_8K),
    )
    monkeypatch.setattr(
        "cdt.ingest.genres.acquire_scraped_sixk_documents",
        _recording_acquirer(calls, GENRE_6K),
    )
    return calls


@pytest.fixture
def cik_file(tmp_path: Path) -> Path:
    """A one-CIK list."""
    path = tmp_path / "ciks.txt"
    path.write_text("320193\n", encoding="utf-8")
    return path


def _ingest(tmp_path: Path, cik_file: Path, *extra: str) -> list[str]:
    return [
        "ingest",
        "--quiet",
        "--artifact-root",
        str(tmp_path),
        "--cik-file",
        str(cik_file),
        *extra,
    ]


# --- ingest -------------------------------------------------------------------


def test_ingest_builds_its_config_from_the_flags(
    tmp_path: Path, ingest_calls: list[IngestCall]
) -> None:
    """Every ingest flag reaches the config, and the CIK list is read once."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("0000320193\n\n789019\n", encoding="utf-8")

    status = cli.main(
        _ingest(
            tmp_path,
            cik_file,
            "--genres",
            "8-K",
            "--bucket",
            "test-bucket",
            "--force",
            "--batch-size",
            "25",
            "--download",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
        )
    )

    assert status == 0
    [(genre, config, ciks)] = ingest_calls
    assert genre == GENRE_8K
    assert ciks == {"0000320193", "789019"}
    assert (
        config.mode,
        config.bucket,
        config.start_date,
        config.end_date,
        config.force,
        config.batch_size,
        config.download,
        config.aws_profile,
        config.s3_prefix,
    ) == (
        "historical",
        "test-bucket",
        date(2024, 1, 1),
        date(2024, 1, 31),
        True,
        25,
        True,
        "",
        "sec",
    )


def test_ingest_without_dates_uses_the_daily_window(
    tmp_path: Path, cik_file: Path, ingest_calls: list[IngestCall]
) -> None:
    """No dates means the daily lookback window ending yesterday."""
    assert cli.main(_ingest(tmp_path, cik_file, "--genres", "8-K")) == 0

    today = date.today()
    [(_, config, _)] = ingest_calls
    assert config.mode == "daily"
    assert (config.start_date, config.end_date) == (
        today.fromordinal(today.toordinal() - DAILY_LOOKBACK_DAYS),
        today.fromordinal(today.toordinal() - 1),
    )


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (("--start-date", "2024-01-01"), (date(2024, 1, 1), date.today())),
        (("--end-date", "2024-01-31"), (ALL_TIME_START_DATE, date(2024, 1, 31))),
    ],
)
def test_ingest_with_one_date_fills_the_other_from_the_archive_bounds(
    tmp_path: Path,
    cik_file: Path,
    ingest_calls: list[IngestCall],
    extra: tuple[str, ...],
    expected: tuple[date, date],
) -> None:
    """A single date is a historical range: open ends run to the archive's bounds."""
    assert cli.main(_ingest(tmp_path, cik_file, "--genres", "8-K", *extra)) == 0

    [(_, config, _)] = ingest_calls
    assert (config.start_date, config.end_date) == expected


def test_ingest_acquires_every_genre_by_default(
    tmp_path: Path, cik_file: Path, ingest_calls: list[IngestCall]
) -> None:
    """One CIK list, every genre, in registry order; 6-K never inlines bodies."""
    status = cli.main(
        _ingest(
            tmp_path,
            cik_file,
            "--download",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
        )
    )

    assert status == 0
    assert [
        (genre, config.form_types, config.dataset_name, config.download, ciks)
        for genre, config, ciks in ingest_calls
    ] == [
        (GENRE_8K, ("8-K",), "documents", True, {"320193"}),
        (GENRE_6K, ("6-K", "6-K/A"), "documents-sixk", False, {"320193"}),
    ]


def test_ingest_genres_narrows_the_run(
    tmp_path: Path, cik_file: Path, ingest_calls: list[IngestCall]
) -> None:
    """``--genres 6-K`` acquires 6-K filings only."""
    assert cli.main(_ingest(tmp_path, cik_file, "--genres", "6-K")) == 0
    assert [genre for genre, _, _ in ingest_calls] == [GENRE_6K]


def test_ingest_one_failing_genre_does_not_stop_the_other(
    tmp_path: Path,
    cik_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    ingest_calls: list[IngestCall],
) -> None:
    """6-K still ingests when 8-K raises, and the command exits non-zero."""

    def failing(config: IngestConfig, **kwargs: object) -> object:
        del config, kwargs
        raise RuntimeError("simulated 8-K failure")

    monkeypatch.setattr("cdt.ingest.genres.acquire_eightk_documents", failing)

    assert cli.main(_ingest(tmp_path, cik_file)) == 1
    assert [genre for genre, _, _ in ingest_calls] == [GENRE_6K]


def test_ingest_isolates_a_value_error_raised_while_acquiring(
    tmp_path: Path,
    cik_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    ingest_calls: list[IngestCall],
) -> None:
    """A corrupt manifest in one genre is that genre's failure, not bad arguments."""

    def corrupt(config: IngestConfig, **kwargs: object) -> object:
        del config, kwargs
        raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr("cdt.ingest.genres.acquire_scraped_sixk_documents", corrupt)

    assert cli.main(_ingest(tmp_path, cik_file)) == 1
    assert [genre for genre, _, _ in ingest_calls] == [GENRE_8K]


def test_ingest_rejects_an_end_before_the_start(
    tmp_path: Path, cik_file: Path, ingest_calls: list[IngestCall]
) -> None:
    """A reversed range is a usage error, and no genre is acquired."""
    status = cli.main(
        _ingest(
            tmp_path, cik_file, "--start-date", "2024-02-01", "--end-date", "2024-01-01"
        )
    )

    assert status == ARGPARSE_USAGE_ERROR
    assert ingest_calls == []


def test_ingest_without_a_cik_file_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ingest_calls: list[IngestCall]
) -> None:
    """No --cik-file and no CDT_DEFAULT_CIK_FILE fails before acquiring anything."""
    monkeypatch.delenv("CDT_DEFAULT_CIK_FILE", raising=False)

    status = cli.main(["ingest", "--quiet", "--artifact-root", str(tmp_path)])

    assert status == ARGPARSE_USAGE_ERROR
    assert ingest_calls == []


def test_ingest_reads_its_cik_file_from_the_environment(
    tmp_path: Path,
    cik_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    ingest_calls: list[IngestCall],
) -> None:
    """CDT_DEFAULT_CIK_FILE stands in for --cik-file."""
    monkeypatch.setenv("CDT_DEFAULT_CIK_FILE", str(cik_file))

    status = cli.main(
        ["ingest", "--quiet", "--artifact-root", str(tmp_path), "--genres", "8-K"]
    )

    assert status == 0
    assert ingest_calls[0][2] == {"320193"}


def test_ingest_failures_reach_the_log_file(
    tmp_path: Path, cik_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genre's failure is logged, with its traceback, to --log-file."""
    log_file = tmp_path / "ingest.log"

    def failing(config: IngestConfig, **kwargs: object) -> object:
        del config, kwargs
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("cdt.ingest.genres.acquire_eightk_documents", failing)

    status = cli.main(
        _ingest(tmp_path, cik_file, "--genres", "8-K", "--log-file", str(log_file))
    )

    assert status == 1
    logged = log_file.read_text(encoding="utf-8")
    assert "Genre ingest failed: genre=8-K" in logged
    assert "simulated failure" in logged


def test_parse_date_rejects_non_iso_date() -> None:
    """CLI dates must be provided as YYYY-MM-DD."""
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["ingest", "--start-date", "01/01/2024"])

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR


# --- segment ------------------------------------------------------------------


@pytest.fixture
def segment_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, object]]]:
    """Record each genre's segment stage instead of running it."""
    calls: list[tuple[str, dict[str, object]]] = []

    def recorder(genre: str) -> Callable[..., pd.DataFrame]:
        def stage(**kwargs: object) -> pd.DataFrame:
            calls.append((genre, kwargs))
            return pd.DataFrame([{"item_id": "row-1"}])

        return stage

    monkeypatch.setattr(
        "cdt.pipeline.segment_pending_eightk_documents", recorder(GENRE_8K)
    )
    monkeypatch.setattr(
        "cdt.pipeline.segment_pending_sixk_documents", recorder(GENRE_6K)
    )
    return calls


def test_segment_runs_every_genre_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_calls: list[tuple[str, dict[str, object]]],
) -> None:
    """Without --genres both genres segment, in registry order."""
    monkeypatch.delenv("GENRES", raising=False)

    assert cli.main(["segment", "--quiet", "--artifact-root", str(tmp_path)]) == 0
    assert [genre for genre, _ in segment_calls] == [GENRE_8K, GENRE_6K]


def test_segment_genres_8k_is_the_8k_only_itemizer(
    tmp_path: Path, segment_calls: list[tuple[str, dict[str, object]]]
) -> None:
    """``--genres 8-K`` runs only the 8-K itemizer, with every option passed through."""
    status = cli.main(
        [
            "segment",
            "--quiet",
            "--artifact-root",
            str(tmp_path),
            "--genres",
            "8-K",
            "--batch-size",
            "25",
            "--force",
        ]
    )

    assert status == 0
    [(genre, kwargs)] = segment_calls
    assert genre == GENRE_8K
    assert kwargs["artifact_root"] == str(tmp_path)
    assert kwargs["batch_size"] == 25  # noqa: PLR2004
    assert kwargs["force"] is True
    assert kwargs["item_numbers"] == POTENTIALLY_RELEVANT_ITEM_NUMBERS
    # A stage that can run long renews the lease it holds.
    assert callable(kwargs["renew"])


def test_segment_passes_custom_item_numbers(
    tmp_path: Path, segment_calls: list[tuple[str, dict[str, object]]]
) -> None:
    """The 8-K segmenter receives a custom item-number list."""
    status = cli.main(
        [
            "segment",
            "--quiet",
            "--artifact-root",
            str(tmp_path),
            "--genres",
            "8-K",
            "--item-numbers",
            "1.01,8.01",
        ]
    )

    assert status == 0
    assert segment_calls[0][1]["item_numbers"] == ("1.01", "8.01")


def test_segment_genres_come_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_calls: list[tuple[str, dict[str, object]]],
) -> None:
    """GENRES narrows a command that is not given --genres."""
    monkeypatch.setenv("GENRES", "6-K")

    assert cli.main(["segment", "--quiet", "--artifact-root", str(tmp_path)]) == 0
    assert [genre for genre, _ in segment_calls] == [GENRE_6K]


def test_a_bad_genres_environment_value_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_calls: list[tuple[str, dict[str, object]]],
) -> None:
    """A typo in GENRES is reported by argparse, not silently ignored."""
    monkeypatch.setenv("GENRES", "10-K")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["segment", "--quiet", "--artifact-root", str(tmp_path)])

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR
    assert segment_calls == []


def test_segment_one_failing_genre_does_not_stop_the_other(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_calls: list[tuple[str, dict[str, object]]],
) -> None:
    """6-K still segments when 8-K raises, and the command exits non-zero."""

    def failing(**kwargs: object) -> pd.DataFrame:
        del kwargs
        raise RuntimeError("simulated 8-K failure")

    monkeypatch.setattr("cdt.pipeline.segment_pending_eightk_documents", failing)

    assert cli.main(["segment", "--quiet", "--artifact-root", str(tmp_path)]) == 1
    assert [genre for genre, _ in segment_calls] == [GENRE_6K]


def test_the_artifact_root_comes_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_calls: list[tuple[str, dict[str, object]]],
) -> None:
    """ARTIFACT_ROOT supplies a stage command's root when --artifact-root is absent."""
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "from-env"))

    assert cli.main(["segment", "--quiet", "--genres", "8-K"]) == 0
    assert segment_calls[0][1]["artifact_root"] == str(tmp_path / "from-env")


def test_stage_commands_do_not_run_while_lease_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage commands refuse to run against a leased artifact root.

    They rewrite the same completion registries the scheduled runs do, and an
    unserialized writer loses registry updates.
    """
    monkeypatch.setattr(
        "cdt.pipeline.segment_pending_eightk_documents",
        lambda **kwargs: pytest.fail("segment must not run while the lease is held"),
    )
    held = acquire_lease(tmp_path, PIPELINE_WRITER_LEASE)
    assert held is not None

    status = cli.main(
        ["segment", "--quiet", "--artifact-root", str(tmp_path), "--genres", "8-K"]
    )

    assert status == 1


def test_parse_item_numbers_rejects_empty_list() -> None:
    """CLI item-number overrides must contain at least one value."""
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["segment", "--item-numbers", " , "])

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR


# --- classify -----------------------------------------------------------------


@pytest.fixture
def classify_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, object]]]:
    """Record each genre's classify stage instead of running it."""
    calls: list[tuple[str, dict[str, object]]] = []

    def recorder(genre: str) -> Callable[..., pd.DataFrame]:
        def stage(**kwargs: object) -> pd.DataFrame:
            calls.append((genre, kwargs))
            return pd.DataFrame([{"item_id": "row-1", "relevance": True}])

        return stage

    monkeypatch.setattr("cdt.pipeline.classify_pending_items", recorder(GENRE_8K))
    monkeypatch.setattr("cdt.pipeline.triage_pending_windows", recorder(GENRE_6K))
    return calls


def test_classify_runs_every_genre_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classify_calls: list[tuple[str, dict[str, object]]],
) -> None:
    """Without --genres both genres classify, each with its own model options."""
    monkeypatch.delenv("GENRES", raising=False)

    status = cli.main(
        [
            "classify",
            "--quiet",
            "--artifact-root",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "eightk-model"),
            "--sixk-model-dir",
            str(tmp_path / "sixk-model"),
            "--concurrency",
            "2",
        ]
    )

    assert status == 0
    by_genre = dict(classify_calls)
    assert list(by_genre) == [GENRE_8K, GENRE_6K]
    assert by_genre[GENRE_8K]["model_dir"] == tmp_path / "eightk-model"
    assert by_genre[GENRE_6K]["model_dir"] == tmp_path / "sixk-model"
    assert by_genre[GENRE_6K]["concurrency"] == 2  # noqa: PLR2004


def test_classify_genres_8k_is_the_8k_only_classifier(
    tmp_path: Path, classify_calls: list[tuple[str, dict[str, object]]]
) -> None:
    """``--genres 8-K`` runs only the 8-K item classifier, options passed through."""
    status = cli.main(
        [
            "classify",
            "--quiet",
            "--artifact-root",
            str(tmp_path),
            "--genres",
            "8-K",
            "--batch-size",
            "25",
            "--force",
        ]
    )

    assert status == 0
    [(genre, kwargs)] = classify_calls
    assert genre == GENRE_8K
    assert (kwargs["batch_size"], kwargs["force"], kwargs["model_dir"]) == (
        25,
        True,
        None,
    )
    assert callable(kwargs["renew"])


@pytest.fixture
def train_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    """Record classifier training, and point the default model dir into tmp."""
    calls: list[dict[str, object]] = []

    def fake_train(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"training_row_count": 2}

    monkeypatch.setattr(cli, "default_model_dir", lambda: tmp_path / "default-model")
    monkeypatch.setattr(cli, "train_classifier_model", fake_train)
    return calls


def test_classify_train_forwards_its_options(
    tmp_path: Path, train_calls: list[dict[str, object]]
) -> None:
    """Training gets the CSV, the default model dir and every tuning option."""
    train_csv = tmp_path / "annotations.csv"

    status = cli.main(
        [
            "classify",
            "train",
            "--train-csv",
            str(train_csv),
            "--target-recall",
            "0.99",
            "--cv-splits",
            "3",
            "--random-seed",
            "7",
            "--quiet",
        ]
    )

    assert status == 0
    assert train_calls == [
        {
            "train_csv": train_csv,
            "model_dir": tmp_path / "default-model",
            "target_recall": 0.99,
            "cv_splits": 3,
            "random_seed": 7,
        }
    ]


@pytest.mark.parametrize("before_train", [True, False])
def test_classify_train_honours_model_dir_on_either_side_of_train(
    tmp_path: Path, train_calls: list[dict[str, object]], before_train: bool
) -> None:
    """``classify --model-dir X train`` and ``classify train --model-dir X`` agree."""
    model_dir = str(tmp_path / "chosen")
    train = ["train", "--train-csv", str(tmp_path / "a.csv")]
    argv = (
        ["classify", "--model-dir", model_dir, *train]
        if before_train
        else ["classify", *train, "--model-dir", model_dir]
    )

    assert cli.main([*argv, "--quiet"]) == 0
    assert train_calls[0]["model_dir"] == Path(model_dir)


def test_classify_train_requires_train_csv() -> None:
    """Training fails argument parsing without an explicit CSV path."""
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["classify", "train"])

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR


# --- extract ------------------------------------------------------------------


def test_extract_calls_the_live_extractor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extract command forwards batch and model options."""
    calls: list[dict[str, object]] = []

    def fake_extract_pending_items(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame([{"debt_instrument_mention_id": "m-1"}])

    monkeypatch.setattr(cli, "extract_pending_items", fake_extract_pending_items)

    status = cli.main(
        [
            "extract",
            "--artifact-root",
            str(tmp_path),
            "--batch-size",
            "25",
            "--force",
            "--model",
            "anthropic/claude-sonnet-4",
            "--reasoning-effort",
            "high",
            "--max-attempts",
            "5",
            "--quiet",
        ]
    )

    assert status == 0
    assert len(calls) == 1
    # A long live run must keep the lease it took.
    assert callable(calls[0].pop("renew"))
    assert calls == [
        {
            "artifact_root": str(tmp_path),
            "batch_size": 25,
            "force": True,
            "model": "anthropic/claude-sonnet-4",
            "reasoning_effort": "high",
            "max_attempts": 5,
        }
    ]


@pytest.mark.parametrize(
    ("extra", "expected"),
    [((), "env/model"), (("--model", "flag/model"), "flag/model")],
)
def test_extract_model_defaults_to_the_extractor_model_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    expected: str,
) -> None:
    """With no --model the EXTRACTOR_MODEL setting is used; --model wins."""
    monkeypatch.setattr(settings, "EXTRACTOR_MODEL", "env/model")
    _seed_classifications(tmp_path, ["a-8-01"])
    models: list[object] = []

    async def fake_workflow(**kwargs: object) -> ExtractionRowState:
        models.append(kwargs["model"])
        row_state = ExtractionRowState(
            item_row=kwargs["item_row"], stage_name="instrument_ie"
        )
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.extractor.live.run_extraction_workflow", fake_workflow)

    status = cli.main(["extract", "--quiet", "--artifact-root", str(tmp_path), *extra])

    assert status == 0
    assert models == [expected]


def test_extract_job_show_reports_idle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``extract job show`` reports an idle root without touching anything."""
    assert cli.main(["extract", "job", "show", "--artifact-root", str(tmp_path)]) == 0
    assert "No active extract job" in capsys.readouterr().out


@pytest.mark.parametrize("root_first", [True, False])
def test_extract_job_show_honours_the_root_on_either_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_first: bool
) -> None:
    """``--artifact-root`` works before ``job show`` and after it."""
    roots: list[str] = []
    monkeypatch.setattr(
        cli,
        "describe_active_job",
        lambda root: (roots.append(root), cli.ActiveJobSummary(status="idle"))[1],
    )
    root = ["--artifact-root", str(tmp_path)]
    argv = (
        ["extract", *root, "job", "show"]
        if root_first
        else ["extract", "job", "show", *root]
    )

    assert cli.main(argv) == 0
    assert roots == [str(tmp_path)]


def test_extract_job_show_reports_corruption_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt job exits non-zero so an operator's check fails loudly."""
    monkeypatch.setattr(
        cli,
        "describe_active_job",
        lambda root: cli.ActiveJobSummary(
            status="corrupt", job_id="J", detail="missing manifest.json"
        ),
    )

    assert cli.main(["extract", "job", "show", "--artifact-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "missing manifest.json" in out
    assert "cdt extract job reset --yes" in out


def _reset(tmp_path: Path, *extra: str) -> list[str]:
    return ["extract", "job", "reset", "--artifact-root", str(tmp_path), *extra]


def test_extract_job_reset_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --yes the reset only reports what it would abandon."""
    monkeypatch.setattr(
        cli,
        "describe_active_job",
        lambda root: cli.ActiveJobSummary(
            status="active", job_id="J", in_flight_batches=2
        ),
    )
    monkeypatch.setattr(
        cli,
        "reset_active_job",
        lambda root, **kwargs: pytest.fail("must not reset without --yes"),
    )

    assert cli.main(_reset(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "2 batch(es)" in out
    assert "--yes" in out


def test_extract_job_reset_clears_under_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --yes the reset runs, pinned to the job shown, holding the lease."""
    resets: list[str] = []
    monkeypatch.setattr(
        cli,
        "describe_active_job",
        lambda root: cli.ActiveJobSummary(status="active", job_id="J"),
    )

    def fake_reset(root: str, expected_job_id: str) -> str:
        assert acquire_lease(root, PIPELINE_WRITER_LEASE) is None
        resets.append(expected_job_id)
        return "J"

    monkeypatch.setattr(cli, "reset_active_job", fake_reset)

    assert cli.main(_reset(tmp_path, "--yes")) == 0
    assert resets == ["J"]
    assert "Cleared the active extract job marker for J" in capsys.readouterr().out


def test_extract_job_reset_refuses_while_a_tick_holds_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running poll tick blocks the reset rather than racing it."""
    monkeypatch.setattr(
        cli,
        "describe_active_job",
        lambda root: cli.ActiveJobSummary(status="active", job_id="J"),
    )
    monkeypatch.setattr(
        cli,
        "reset_active_job",
        lambda root, **kwargs: pytest.fail("must not reset while locked"),
    )
    held = acquire_lease(tmp_path, PIPELINE_WRITER_LEASE)
    assert held is not None

    assert cli.main(_reset(tmp_path, "--yes")) == 1


def test_extract_job_reset_aborts_when_the_job_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reset pinned to a job that is no longer active clears nothing."""
    monkeypatch.setattr(
        cli,
        "describe_active_job",
        lambda root: cli.ActiveJobSummary(status="active", job_id="OLD"),
    )
    # reset_active_job reports the mismatch by returning None.
    monkeypatch.setattr(cli, "reset_active_job", lambda root, expected_job_id: None)

    assert cli.main(_reset(tmp_path, "--yes")) == 1
    assert "Not reset" in capsys.readouterr().out


# --- match --------------------------------------------------------------------


def test_match_matches_then_infers_lineage_both_renewing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cdt match`` forwards its options, then runs the lineage pass over every shard.

    Both rewrite whole datasets, so each is handed a renewer for the lease.
    """
    calls: list[dict[str, object]] = []

    def fake_match_pending_mentions(**kwargs: object) -> dict[str, pd.DataFrame]:
        calls.append(kwargs)
        return {
            "debt_instrument_mentions": pd.DataFrame(
                [{"debt_instrument_mention_id": "m-1"}]
            ),
            "debt_instrument": pd.DataFrame([{"debt_instrument_id": "di-1"}]),
        }

    lineage_calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "match_pending_mentions", fake_match_pending_mentions)
    monkeypatch.setattr(
        cli,
        "apply_lineage_inference_pass",
        lambda artifact_root, **kwargs: (
            lineage_calls.append({"root": str(artifact_root), **kwargs}),
            {"links": 1, "reopened": 0, "heads_before": 2, "heads_after": 1},
        )[1],
    )

    status = cli.main(
        [
            "match",
            "--quiet",
            "--artifact-root",
            str(tmp_path),
            "--batch-size",
            "25",
            "--force",
        ]
    )

    assert status == 0
    [kwargs] = calls
    assert (
        kwargs["batch_size"],
        kwargs["force"],
        kwargs["strong_match_threshold"],
        kwargs["loose_match_threshold"],
        kwargs["ambiguity_margin"],
    ) == (25, True, 0.9, 0.75, 0.05)
    assert callable(kwargs["renew"])
    [lineage] = lineage_calls
    assert lineage["root"] == str(tmp_path)
    assert callable(lineage["renew"])


# --- publish ------------------------------------------------------------------


def test_publish_requires_a_final_database_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing nowhere is a usage error, not a silent no-op."""
    monkeypatch.delenv("FINAL_DATABASE_ROOT", raising=False)
    monkeypatch.setattr(
        cli,
        "publish_final_tables",
        lambda **kwargs: pytest.fail("must not publish without a destination"),
    )

    status = cli.main(["publish", "--quiet", "--artifact-root", str(tmp_path)])

    assert status == ARGPARSE_USAGE_ERROR


@pytest.mark.parametrize("via_env", [False, True])
def test_publish_writes_the_final_tables_under_the_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    via_env: bool,
) -> None:
    """``cdt publish`` hands its destination (flag or env) and a renewer to the publisher."""
    final_root = str(tmp_path / "final")
    calls: list[dict[str, object]] = []

    def fake_publish(**kwargs: object) -> dict[str, str]:
        assert acquire_lease(tmp_path, PIPELINE_WRITER_LEASE) is None
        calls.append(kwargs)
        return {"items": f"{final_root}/items/latest.parquet"}

    monkeypatch.setattr(cli, "publish_final_tables", fake_publish)
    argv = ["publish", "--quiet", "--artifact-root", str(tmp_path), "--force"]
    if via_env:
        monkeypatch.setenv("FINAL_DATABASE_ROOT", final_root)
    else:
        argv += ["--final-database-root", final_root]

    assert cli.main(argv) == 0
    [kwargs] = calls
    assert kwargs["artifact_root"] == str(tmp_path)
    assert kwargs["final_database_root"] == final_root
    assert kwargs["force"] is True
    assert callable(kwargs["renew"])
    assert "Published items" in capsys.readouterr().out


def test_final_database_root_only_where_honored() -> None:
    """Commands that never publish reject --final-database-root instead of ignoring it."""
    parser = cli.build_parser()

    args = parser.parse_args(
        ["run", "daily", "--final-database-root", "/final", "--cik-file", "c.txt"]
    )
    assert args.final_database_root == "/final"

    with pytest.raises(SystemExit):
        parser.parse_args(["segment", "--final-database-root", "/final"])


def test_log_file_captures_every_module_s_log(tmp_path: Path) -> None:
    """--log-file reaches stage modules' loggers, not only the CLI's own."""
    log_file = tmp_path / "cdt.log"

    status = cli.main(
        [
            "segment",
            "--genres",
            "8-K",
            "--artifact-root",
            str(tmp_path / "root"),
            "--log-file",
            str(log_file),
        ]
    )

    assert status == 0
    assert "cdt.partition_stage" in log_file.read_text(encoding="utf-8")


def test_quiet_silences_stage_modules_info_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--quiet applies to stage modules' loggers too."""
    status = cli.main(
        ["segment", "--genres", "8-K", "--artifact-root", str(tmp_path), "--quiet"]
    )

    assert status == 0
    assert "partition_stage" not in capsys.readouterr().err
