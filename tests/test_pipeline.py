"""Tests for end-to-end pipeline orchestration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdt.classifier import core as classifier_core
from cdt.extractor.core import ExtractionRowState
from cdt.ingest import IngestRunResult
from cdt.matcher import debt_instruments_root, mention_matches_root
from cdt.pipeline import (
    ALL_TIME_START_DATE,
    PipelineConfig,
    resolve_mode_dates,
    run_pipeline,
)
from cdt.storage import read_dataset, read_table, write_partition_table


class FakeModel:
    """Classifier stub returning one relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a strong-positive score for the seeded test document."""
        del texts
        return [2.0]


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
    assert calls == [
        ("ingest", {"320193"}),
        ("itemize", 11),
        ("classify", 12),
        ("extract", 13),
        ("match", 14),
    ]


def test_run_pipeline_processes_small_seeded_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline should complete on one seeded document batch."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")

    def fake_run_ingest_pipeline(
        config: object,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del config, s3_client
        document_rows = pd.DataFrame(
            [
                {
                    "accession_number": "000114036126006577",
                    "cik": "320193",
                    "company_name": "Example Inc.",
                    "url": "https://sec.example/full.txt",
                    "text": """
ITEM INFORMATION: Other Events
<DOCUMENT>
<TYPE>8-K
<TEXT>
Item 8.01 Other Events.
This is the extracted event text.
</TEXT>
</DOCUMENT>
""",
                    "date": "2024-01-02",
                    "resource_uri": None,
                }
            ]
        )
        document_partition = write_partition_table(
            tmp_path / "documents",
            partition={"date": "2024-01-02", "shard": "0001"},
            table=document_rows,
        )
        return document_rows, IngestRunResult(
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
            document_partitions=(document_partition,),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        )

    async def fake_run_extraction_workflow(
        **kwargs: object,
    ) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {
                "debt_instrument_mention_id": "m-1",
                "item_id": item_row["item_id"],
                "accession_number": item_row["accession_number"],
                "cik": item_row["cik"],
                "company_name": item_row["company_name"],
                "date": item_row["date"],
                "raw_id": "i-1",
                "name": "Term Loan",
                "start_date": "2024-01-01",
                "end_date": None,
                "amount": "$100 million",
                "amendment_of": None,
                "retired_of": None,
                "split_of": None,
                "lenders_json": '[{"mentions": [{"text": "Acme Bank"}]}]',
                "other_interested_parties_json": "[]",
                "name_json": "{}",
                "start_date_json": "{}",
                "end_date_json": "{}",
                "amount_json": "{}",
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.pipeline.run_ingest_pipeline", fake_run_ingest_pipeline)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    monkeypatch.setattr(
        "cdt.extractor.core.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    result = run_pipeline(
        PipelineConfig(
            mode="historical",
            cik_file=str(cik_file),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            download=True,
            ingest_batch_size=1,
            itemize_batch_size=1,
            classify_batch_size=1,
            extract_batch_size=1,
            match_batch_size=1,
            artifact_root=str(tmp_path),
            final_database_root=str(tmp_path / "database" / "cdt"),
        )
    )

    written_matches = read_dataset(mention_matches_root(tmp_path))
    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    final_items = read_table(tmp_path / "database" / "cdt" / "items" / "latest.parquet")
    final_mentions = read_table(
        tmp_path / "database" / "cdt" / "debt-instrument-mentions" / "latest.parquet"
    )
    final_edges = read_table(
        tmp_path / "database" / "cdt" / "mention-cluster-edges" / "latest.parquet"
    )
    final_instruments = read_table(
        tmp_path / "database" / "cdt" / "debt-instruments" / "latest.parquet"
    )
    assert result.itemized_rows == 1
    assert result.classified_rows == 1
    assert result.extracted_rows == 1
    assert result.matched_rows == 1
    assert written_matches["edge_type"].to_list() == ["member"]
    assert written_instruments["debt_instrument_id"].to_list() == ["m-1"]
    assert final_items["item_id"].to_list() == ["000114036126006577-8-01"]
    assert final_items["company_name"].to_list() == ["Example Inc."]
    assert final_mentions["debt_instrument_mention_id"].to_list() == ["m-1"]
    assert final_mentions["company_name"].to_list() == ["Example Inc."]
    assert final_edges["debt_instrument_mention_id"].to_list() == ["m-1"]
    assert final_instruments["debt_instrument_id"].to_list() == ["m-1"]
    assert final_instruments["company_name"].to_list() == ["Example Inc."]


def _seed_final_tables(artifact_root: Path, *, rows: int = 2) -> None:
    """Write minimal rows into every dataset finalize publishes."""
    from cdt.pipeline import FINAL_OUTPUT_TABLES

    for table_name, dataset_root_fn in FINAL_OUTPUT_TABLES.items():
        write_partition_table(
            dataset_root_fn(str(artifact_root)),
            partition={"date": "2024-01-02", "shard": "0001"},
            table=pd.DataFrame(
                [{"id": f"{table_name}-{index}"} for index in range(rows)]
            ),
        )


def test_final_snapshots_publish_atomically_with_pointer(tmp_path: Path) -> None:
    """Finalize writes immutable snapshots and one atomic latest.json pointer (#91)."""
    from cdt.pipeline import write_final_output_tables
    from cdt.storage import read_json_artifact

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _seed_final_tables(artifact_root)

    written = write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )

    pointer = read_json_artifact(str(artifact_root / "final-snapshots" / "latest.json"))
    assert isinstance(pointer, dict)
    assert set(pointer["tables"]) == set(written)
    for table_name, path in written.items():
        assert f"snapshot={pointer['run_id']}" in path
        assert len(read_table(path)) == 2
        assert pointer["tables"][table_name]["rows"] == 2
        # The parquet-only contract surface under the final database root.
        assert (final_root / table_name / "latest.parquet").exists()
    assert pointer["schema_version"]
    # The database prefix stays parquet-only: pointer and snapshots live
    # under the artifact root instead.
    non_parquet = [
        p for p in final_root.rglob("*") if p.is_file() and p.suffix != ".parquet"
    ]
    assert non_parquet == []


def test_final_snapshot_guard_blocks_shrinkage_unless_forced(tmp_path: Path) -> None:
    """A snapshot that would clobber a good one with ~nothing is refused (#91)."""
    from cdt.pipeline import write_final_output_tables

    artifact_root = tmp_path / "artifacts"
    empty_root = tmp_path / "empty-artifacts"
    final_root = tmp_path / "final"
    _seed_final_tables(artifact_root)
    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )

    with pytest.raises(ValueError, match="row-count regressions"):
        write_final_output_tables(
            artifact_root=str(empty_root), final_database_root=str(final_root)
        )
    # The refused publish must not have moved the pointer.
    from cdt.storage import read_json_artifact

    pointer = read_json_artifact(str(artifact_root / "final-snapshots" / "latest.json"))
    assert pointer["tables"]["items"]["rows"] == 2

    forced = write_final_output_tables(
        artifact_root=str(empty_root),
        final_database_root=str(final_root),
        force=True,
    )
    assert forced


def test_old_final_snapshots_are_pruned(tmp_path: Path) -> None:
    """Only the current and prior snapshot generations are kept (#91)."""
    from cdt.pipeline import write_final_output_tables

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _seed_final_tables(artifact_root)

    for _ in range(3):
        write_final_output_tables(
            artifact_root=str(artifact_root), final_database_root=str(final_root)
        )

    snapshot_dirs = {
        path.parent.name
        for path in (artifact_root / "final-snapshots").glob("*/*.parquet")
    }
    assert len(snapshot_dirs) == 2
