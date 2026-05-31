"""Tests for end-to-end pipeline orchestration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdt.ingest import IngestRunResult
from cdt.pipeline import (
    ALL_TIME_START_DATE,
    PipelineConfig,
    resolve_mode_dates,
    run_pipeline,
)


def test_resolve_mode_dates_daily_defaults_to_yesterday() -> None:
    """Daily mode defaults both dates to yesterday."""
    start_date, end_date = resolve_mode_dates("daily", None, None)
    yesterday = date.today().fromordinal(date.today().toordinal() - 1)
    assert start_date == yesterday
    assert end_date == yesterday


def test_resolve_mode_dates_daily_requires_both_dates() -> None:
    """Daily mode rejects partial date ranges."""
    with pytest.raises(ValueError, match="--end-date is required"):
        resolve_mode_dates("daily", date(2024, 1, 1), None)


def test_resolve_mode_dates_historical_defaults_to_all_time() -> None:
    """Historical mode uses the full default CDT range."""
    start_date, end_date = resolve_mode_dates("historical", None, None)
    assert start_date == ALL_TIME_START_DATE
    assert end_date == date.today()


def test_run_pipeline_uses_stage_backed_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end runner calls the persisted stage entrypoints in order."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    def fake_run_ingest_pipeline(
        config: object,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del s3_client
        calls.append(("ingest", ciks))
        return pd.DataFrame([{"accession_number": "1"}]), IngestRunResult(
            mode="historical",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
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

    def fake_itemize_pending_documents(**kwargs: object) -> pd.DataFrame:
        calls.append(("itemize", kwargs["batch_size"]))
        return pd.DataFrame([{"item_id": "item-1"}])

    def fake_classify_pending_items(**kwargs: object) -> pd.DataFrame:
        calls.append(("classify", kwargs["batch_size"]))
        return pd.DataFrame([{"item_id": "item-1", "relevance": True}])

    def fake_extract_pending_items(**kwargs: object) -> pd.DataFrame:
        calls.append(("extract", kwargs["batch_size"]))
        return pd.DataFrame([{"debt_instrument_mention_id": "mention-1"}])

    def fake_match_pending_mentions(**kwargs: object) -> dict[str, pd.DataFrame]:
        calls.append(("match", kwargs["batch_size"]))
        return {
            "debt_instrument_mentions": pd.DataFrame(
                [{"debt_instrument_mention_id": "mention-1"}]
            ),
            "debt_instrument": pd.DataFrame([{"debt_instrument_id": "instrument-1"}]),
        }

    monkeypatch.setattr("cdt.pipeline.run_ingest_pipeline", fake_run_ingest_pipeline)
    monkeypatch.setattr(
        "cdt.pipeline.itemize_pending_documents", fake_itemize_pending_documents
    )
    monkeypatch.setattr(
        "cdt.pipeline.classify_pending_items", fake_classify_pending_items
    )
    monkeypatch.setattr(
        "cdt.pipeline.extract_pending_items", fake_extract_pending_items
    )
    monkeypatch.setattr(
        "cdt.pipeline.match_pending_mentions", fake_match_pending_mentions
    )
    monkeypatch.setattr("cdt.pipeline.publish_config_from_env", lambda: None)

    result = run_pipeline(
        PipelineConfig(
            mode="historical",
            cik_file=str(cik_file),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            download=True,
            ingest_batch_size=10,
            itemize_batch_size=11,
            classify_batch_size=12,
            extract_batch_size=13,
            match_batch_size=14,
        )
    )

    assert result.ingest.total_rows == 1
    assert result.itemized_rows == 1
    assert result.classified_rows == 1
    assert result.extracted_rows == 1
    assert result.matched_rows == 1
    assert result.debt_instrument_rows == 1
    assert result.r2_published is False
    assert calls == [
        ("ingest", {"320193"}),
        ("itemize", 11),
        ("classify", 12),
        ("extract", 13),
        ("match", 14),
    ]


def test_run_pipeline_publishes_snapshot_when_r2_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline publishes dashboard snapshot JSON when R2 config is present."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    publish_calls: list[dict[str, object]] = []

    def fake_run_ingest_pipeline(
        config: object,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del config, ciks, s3_client
        return pd.DataFrame(), IngestRunResult(
            mode="daily",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
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

    monkeypatch.setattr("cdt.pipeline.run_ingest_pipeline", fake_run_ingest_pipeline)
    monkeypatch.setattr(
        "cdt.pipeline.itemize_pending_documents", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "cdt.pipeline.classify_pending_items", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "cdt.pipeline.extract_pending_items", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "cdt.pipeline.match_pending_mentions",
        lambda **kwargs: {
            "debt_instrument_mentions": pd.DataFrame(),
            "debt_instrument": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        "cdt.pipeline.publish_config_from_env",
        lambda: object(),
    )
    monkeypatch.setattr(
        "cdt.pipeline.publish_dashboard_snapshot",
        lambda **kwargs: publish_calls.append(kwargs),
    )

    result = run_pipeline(
        PipelineConfig(
            mode="daily",
            cik_file=str(cik_file),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )
    )

    assert result.r2_published is True
    assert len(publish_calls) == 1
    assert publish_calls[0]["artifact_root"] == result.artifact_root
    assert publish_calls[0]["data_dir"] is None
