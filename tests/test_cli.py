"""Tests for the cdt command-line interface."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdt import cli

ARGPARSE_USAGE_ERROR = 2


def test_ingest_cli_reads_cik_file_and_calls_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ingest command passes CIKs and dates to document acquisition."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("0000320193\n\n789019\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_acquire(
        bucket: str,
        start_date: date,
        end_date: date,
        ciks: set[str] | None = None,
        *,
        force: bool = False,
        batch_size: int = cli.DEFAULT_BATCH_SIZE,
        download: bool = False,
    ) -> pd.DataFrame:
        calls.append(
            {
                "bucket": bucket,
                "start_date": start_date,
                "end_date": end_date,
                "ciks": ciks,
                "force": force,
                "batch_size": batch_size,
                "download": download,
            }
        )
        return pd.DataFrame([{"accession_number": "1"}])

    monkeypatch.setattr(cli, "acquire_documents_for_date_range", fake_acquire)

    status = cli.main(
        [
            "ingest",
            str(cik_file),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
            "--bucket",
            "test-bucket",
            "--force",
            "--batch-size",
            "25",
            "--download",
            "--quiet",
        ]
    )

    assert status == 0
    assert calls == [
        {
            "bucket": "test-bucket",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 1, 31),
            "ciks": {"0000320193", "789019"},
            "force": True,
            "batch_size": 25,
            "download": True,
        }
    ]


def test_ingest_cli_defaults_to_all_time_date_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted ingest dates default to the all-time EDGAR range."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_acquire(
        bucket: str,
        start_date: date,
        end_date: date,
        ciks: set[str] | None = None,
        *,
        force: bool = False,
        batch_size: int = cli.DEFAULT_BATCH_SIZE,
        download: bool = False,
    ) -> pd.DataFrame:
        del bucket, ciks, force, batch_size
        calls.append(
            {"start_date": start_date, "end_date": end_date, "download": download}
        )
        return pd.DataFrame()

    monkeypatch.setattr(cli, "acquire_documents_for_date_range", fake_acquire)

    status = cli.main(["ingest", str(cik_file), "--quiet"])

    assert status == 0
    assert calls == [
        {
            "start_date": cli.ALL_TIME_START_DATE,
            "end_date": date.today(),
            "download": False,
        }
    ]


def test_ingest_cli_logs_failures_to_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected ingest failures are logged and return a failing status."""
    cik_file = tmp_path / "ciks.txt"
    log_file = tmp_path / "ingest.log"
    cik_file.write_text("320193\n", encoding="utf-8")

    def fake_acquire(
        bucket: str,
        start_date: date,
        end_date: date,
        ciks: set[str] | None = None,
        *,
        force: bool = False,
        batch_size: int = cli.DEFAULT_BATCH_SIZE,
        download: bool = False,
    ) -> pd.DataFrame:
        del bucket, start_date, end_date, ciks, force, batch_size, download
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(cli, "acquire_documents_for_date_range", fake_acquire)

    status = cli.main(
        [
            "ingest",
            str(cik_file),
            "--quiet",
            "--log-file",
            str(log_file),
        ]
    )

    assert status == 1
    assert "Ingest failed" in log_file.read_text(encoding="utf-8")
    assert "simulated failure" in log_file.read_text(encoding="utf-8")


def test_itemize_cli_calls_pending_itemizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The itemize command passes batch and force options to the itemizer."""
    calls: list[dict[str, object]] = []

    def fake_itemize_pending_documents(
        *,
        batch_size: int,
        force: bool = False,
    ) -> pd.DataFrame:
        calls.append({"batch_size": batch_size, "force": force})
        return pd.DataFrame([{"item_id": "item-1"}])

    monkeypatch.setattr(
        cli, "itemize_pending_documents", fake_itemize_pending_documents
    )

    status = cli.main(["itemize", "--batch-size", "25", "--force", "--quiet"])

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
