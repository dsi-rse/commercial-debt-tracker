"""Tests for the cdt command-line interface."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdt import cli
from cdt.ingest import IngestRunResult
from cdt.itemizer import POTENTIALLY_RELEVANT_ITEM_NUMBERS
from cdt.pipeline import PipelineRunResult

ARGPARSE_USAGE_ERROR = 2


def test_ingest_cli_reads_cik_file_and_calls_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ingest command builds an ingest config and passes CIKs through."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("0000320193\n\n789019\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_ingest_pipeline(
        config: cli.IngestConfig,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del s3_client
        calls.append(
            {
                "mode": config.mode,
                "bucket": config.bucket,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "ciks": ciks,
                "force": config.force,
                "batch_size": config.batch_size,
                "download": config.download,
                "aws_profile": config.aws_profile,
                "s3_prefix": config.s3_prefix,
            }
        )
        return pd.DataFrame([{"accession_number": "1"}]), IngestRunResult(
            mode=config.mode,
            start_date=config.start_date,
            end_date=config.end_date,
            ciks_count=len(ciks or set()),
            candidates_seen=1,
            skipped_existing=0,
            downloaded=1,
            failures=0,
            total_rows=1,
            output_root=str(tmp_path),
            documents_root=str(tmp_path / "documents"),
            document_partitions=(
                str(tmp_path / "documents" / "date=2024-01-01" / "part-0000.parquet"),
            ),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        )

    monkeypatch.setattr(cli, "run_ingest_pipeline", fake_run_ingest_pipeline)

    status = cli.main(
        [
            "ingest",
            "--bucket",
            "test-bucket",
            "--force",
            "--batch-size",
            "25",
            "--download",
            "--quiet",
            "daily",
            str(cik_file),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
        ]
    )

    assert status == 0
    assert calls == [
        {
            "mode": "daily",
            "bucket": "test-bucket",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 1, 31),
            "ciks": {"0000320193", "789019"},
            "force": True,
            "batch_size": 25,
            "download": True,
            "aws_profile": cli.DEFAULT_AWS_PROFILE,
            "s3_prefix": "sec",
        }
    ]


def test_ingest_cli_historical_defaults_to_all_time_date_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical ingest keeps the all-time EDTGAR defaults."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_ingest_pipeline(
        config: cli.IngestConfig,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del ciks, s3_client
        calls.append(
            {
                "mode": config.mode,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "download": config.download,
            }
        )
        return pd.DataFrame(), IngestRunResult(
            mode=config.mode,
            start_date=config.start_date,
            end_date=config.end_date,
            ciks_count=1,
            candidates_seen=0,
            skipped_existing=0,
            downloaded=0,
            failures=0,
            total_rows=0,
            output_root=str(tmp_path),
            documents_root=str(tmp_path / "documents"),
            document_partitions=(),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        )

    monkeypatch.setattr(cli, "run_ingest_pipeline", fake_run_ingest_pipeline)

    status = cli.main(["ingest", "--quiet", "historical", str(cik_file)])

    assert status == 0
    assert calls == [
        {
            "mode": "historical",
            "start_date": cli.ALL_TIME_START_DATE,
            "end_date": date.today(),
            "download": False,
        }
    ]


def test_ingest_cli_daily_defaults_to_yesterday(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daily ingest defaults both dates to yesterday."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    calls: list[tuple[date, date]] = []

    def fake_run_ingest_pipeline(
        config: cli.IngestConfig,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del ciks, s3_client
        calls.append((config.start_date, config.end_date))
        return pd.DataFrame(), IngestRunResult(
            mode=config.mode,
            start_date=config.start_date,
            end_date=config.end_date,
            ciks_count=1,
            candidates_seen=0,
            skipped_existing=0,
            downloaded=0,
            failures=0,
            total_rows=0,
            output_root=str(tmp_path),
            documents_root=str(tmp_path / "documents"),
            document_partitions=(),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        )

    monkeypatch.setattr(cli, "run_ingest_pipeline", fake_run_ingest_pipeline)

    status = cli.main(["ingest", "--quiet", "daily", str(cik_file)])

    yesterday = date.today().fromordinal(date.today().toordinal() - 1)
    assert status == 0
    assert calls == [(yesterday, yesterday)]


def test_ingest_cli_daily_rejects_partial_date_range(tmp_path: Path) -> None:
    """Daily ingest requires both dates when either is supplied."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")

    status = cli.main(
        [
            "ingest",
            "--quiet",
            "daily",
            str(cik_file),
            "--start-date",
            "2024-01-01",
        ]
    )

    assert status == ARGPARSE_USAGE_ERROR


def test_ingest_cli_logs_failures_to_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected ingest failures are logged and return a failing status."""
    cik_file = tmp_path / "ciks.txt"
    log_file = tmp_path / "ingest.log"
    cik_file.write_text("320193\n", encoding="utf-8")

    def fake_run_ingest_pipeline(
        config: cli.IngestConfig,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del config, ciks, s3_client
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(cli, "run_ingest_pipeline", fake_run_ingest_pipeline)

    status = cli.main(
        [
            "ingest",
            "--quiet",
            "--log-file",
            str(log_file),
            "historical",
            str(cik_file),
        ]
    )

    assert status == 1
    assert "Ingest failed" in log_file.read_text(encoding="utf-8")
    assert "simulated failure" in log_file.read_text(encoding="utf-8")


def test_pipeline_cli_builds_pipeline_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline command builds a full PipelineConfig and runs it."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_pipeline(config: cli.PipelineConfig) -> PipelineRunResult:
        calls.append(
            {
                "mode": config.mode,
                "bucket": config.bucket,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "download": config.download,
                "item_numbers": config.item_numbers,
                "model_dir": config.classifier_model_dir,
            }
        )
        ingest_result = IngestRunResult(
            mode=config.mode,
            start_date=config.start_date or date(2024, 1, 1),
            end_date=config.end_date or date(2024, 1, 31),
            ciks_count=1,
            candidates_seen=3,
            skipped_existing=0,
            downloaded=3,
            failures=0,
            total_rows=3,
            output_root=str(tmp_path),
            documents_root=str(tmp_path / "documents"),
            document_partitions=(
                str(tmp_path / "documents" / "date=2024-01-01" / "part-0000.parquet"),
            ),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        )
        return PipelineRunResult(
            mode=config.mode,
            start_date=ingest_result.start_date,
            end_date=ingest_result.end_date,
            ingest=ingest_result,
            itemized_rows=5,
            classified_rows=5,
            extracted_rows=2,
            matched_rows=2,
            debt_instrument_rows=1,
            classifier_model_dir=config.classifier_model_dir or tmp_path / "model",
            artifact_root=str(tmp_path),
            extractor_run_path=tmp_path / "extractor_runs",
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    status = cli.main(
        [
            "pipeline",
            "--bucket",
            "test-bucket",
            "--download",
            "--item-numbers",
            "1.01,8.01",
            "--model-dir",
            str(tmp_path / "model"),
            "daily",
            str(cik_file),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
        ]
    )

    assert status == 0
    assert calls == [
        {
            "mode": "daily",
            "bucket": "test-bucket",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 1, 31),
            "download": True,
            "item_numbers": ("1.01", "8.01"),
            "model_dir": tmp_path / "model",
        }
    ]


def test_pipeline_cli_daily_rejects_partial_date_range(tmp_path: Path) -> None:
    """Daily pipeline runs require both dates when either is supplied."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")

    status = cli.main(
        [
            "pipeline",
            "--quiet",
            "daily",
            str(cik_file),
            "--start-date",
            "2024-01-01",
        ]
    )

    assert status == ARGPARSE_USAGE_ERROR


def test_itemize_cli_calls_pending_itemizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The itemize command passes batch and force options to the itemizer."""
    calls: list[dict[str, object]] = []

    def fake_itemize_pending_documents(
        *,
        artifact_root: str | Path | None = None,
        batch_size: int,
        force: bool = False,
        item_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        del artifact_root
        calls.append(
            {
                "batch_size": batch_size,
                "force": force,
                "item_numbers": item_numbers,
            }
        )
        return pd.DataFrame([{"item_id": "item-1"}])

    monkeypatch.setattr(
        cli, "itemize_pending_documents", fake_itemize_pending_documents
    )

    status = cli.main(["itemize", "--batch-size", "25", "--force", "--quiet"])

    assert status == 0
    assert calls == [
        {
            "batch_size": 25,
            "force": True,
            "item_numbers": POTENTIALLY_RELEVANT_ITEM_NUMBERS,
        }
    ]


def test_itemize_cli_passes_custom_item_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The itemize command forwards a custom item-number list."""
    calls: list[dict[str, object]] = []

    def fake_itemize_pending_documents(
        *,
        artifact_root: str | Path | None = None,
        batch_size: int,
        force: bool = False,
        item_numbers: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        del artifact_root
        calls.append(
            {
                "batch_size": batch_size,
                "force": force,
                "item_numbers": item_numbers,
            }
        )
        return pd.DataFrame([{"item_id": "item-1"}])

    monkeypatch.setattr(
        cli, "itemize_pending_documents", fake_itemize_pending_documents
    )

    status = cli.main(
        ["itemize", "--item-numbers", "1.01,8.01", "--batch-size", "10", "--quiet"]
    )

    assert status == 0
    assert calls == [
        {
            "batch_size": 10,
            "force": False,
            "item_numbers": ("1.01", "8.01"),
        }
    ]


def test_classifier_cli_calls_pending_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classifier command passes batch, force, and model_dir options through."""
    calls: list[dict[str, object]] = []

    def fake_classify_pending_items(
        *,
        artifact_root: str | Path | None = None,
        batch_size: int,
        force: bool = False,
        model_dir: Path | None = None,
    ) -> pd.DataFrame:
        del artifact_root
        calls.append(
            {
                "batch_size": batch_size,
                "force": force,
                "model_dir": model_dir,
            }
        )
        return pd.DataFrame([{"item_id": "item-1"}])

    monkeypatch.setattr(cli, "classify_pending_items", fake_classify_pending_items)

    status = cli.main(["classifier", "--batch-size", "25", "--force", "--quiet"])

    assert status == 0
    assert calls == [{"batch_size": 25, "force": True, "model_dir": None}]


def test_classifier_train_cli_calls_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classifier training command forwards resolved defaults and options."""
    calls: list[dict[str, object]] = []
    train_csv = tmp_path / "annotations.csv"
    model_dir = tmp_path / "model"

    def fake_train_classifier_model(
        *,
        train_csv: Path,
        model_dir: Path,
        target_recall: float,
        cv_splits: int,
        random_seed: int,
    ) -> dict[str, object]:
        calls.append(
            {
                "train_csv": train_csv,
                "model_dir": model_dir,
                "target_recall": target_recall,
                "cv_splits": cv_splits,
                "random_seed": random_seed,
            }
        )
        return {"training_row_count": 2}

    monkeypatch.setattr(cli, "default_model_dir", lambda: model_dir)
    monkeypatch.setattr(cli, "train_classifier_model", fake_train_classifier_model)

    status = cli.main(
        [
            "classifier",
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
    assert calls == [
        {
            "train_csv": train_csv,
            "model_dir": model_dir,
            "target_recall": 0.99,
            "cv_splits": 3,
            "random_seed": 7,
        }
    ]


def test_classifier_train_cli_requires_train_csv() -> None:
    """Training should fail argument parsing without an explicit CSV path."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["classifier", "train"])

    assert exc_info.value.code == ARGPARSE_USAGE_ERROR


def test_extractor_cli_calls_pending_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extractor command forwards batch and model options."""
    calls: list[dict[str, object]] = []

    def fake_extract_pending_items(
        *,
        artifact_root: str | Path | None = None,
        batch_size: int,
        force: bool = False,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_attempts: int,
    ) -> pd.DataFrame:
        del artifact_root
        calls.append(
            {
                "batch_size": batch_size,
                "force": force,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_attempts": max_attempts,
            }
        )
        return pd.DataFrame([{"debt_instrument_mention_id": "m-1"}])

    monkeypatch.setattr(cli, "extract_pending_items", fake_extract_pending_items)

    status = cli.main(
        [
            "extractor",
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
    assert calls == [
        {
            "batch_size": 25,
            "force": True,
            "model": "anthropic/claude-sonnet-4",
            "reasoning_effort": "high",
            "max_attempts": 5,
        }
    ]


def test_matcher_cli_calls_pending_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matcher command forwards batch and force options."""
    calls: list[dict[str, object]] = []

    def fake_match_pending_mentions(
        *,
        artifact_root: str | Path | None = None,
        batch_size: int,
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        del artifact_root
        calls.append(
            {
                "batch_size": batch_size,
                "force": force,
            }
        )
        return {
            "debt_instrument_mentions": pd.DataFrame(
                [{"debt_instrument_mention_id": "m-1"}]
            ),
            "debt_instrument": pd.DataFrame([{"debt_instrument_id": "di-1"}]),
        }

    monkeypatch.setattr(cli, "match_pending_mentions", fake_match_pending_mentions)

    status = cli.main(["matcher", "--batch-size", "25", "--force", "--quiet"])

    assert status == 0
    assert calls == [{"batch_size": 25, "force": True}]


def test_parse_date_rejects_non_iso_date() -> None:
    """CLI dates must be provided as YYYY-MM-DD."""
    parser = cli.build_parser()

    try:
        parser.parse_args(["ingest", "ciks.txt", "--start-date", "01/01/2024"])
    except SystemExit as exc:
        assert exc.code == ARGPARSE_USAGE_ERROR
    else:
        raise AssertionError("expected argparse to reject a non-ISO date")


def test_parse_item_numbers_rejects_empty_list() -> None:
    """CLI item-number overrides must contain at least one value."""
    parser = cli.build_parser()

    try:
        parser.parse_args(["itemize", "--item-numbers", " , "])
    except SystemExit as exc:
        assert exc.code == ARGPARSE_USAGE_ERROR
    else:
        raise AssertionError("expected argparse to reject an empty item-number list")
