"""Tests for end-to-end pipeline orchestration."""

from __future__ import annotations

from collections.abc import Callable
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
    DEFAULT_GENRES,
    GENRE_6K,
    GENRE_8K,
    PipelineConfig,
    normalize_genres,
    resolve_mode_dates,
    run_pipeline,
)
from cdt.storage import read_dataset, read_table, write_partition_table

EXPECTED_SIXK_SNIPPETS = 2
#: Two seeded 8-K items plus the one 6-K snippet the union brings in.
EXPECTED_UNIONED_ITEM_ROWS = 3


class FakeModel:
    """Classifier stub returning one relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a strong-positive score for the seeded test document."""
        del texts
        return [2.0]


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

    def fake_acquire_scraped_sixk_documents(
        config: object,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del s3_client
        calls.append(("ingest-sixk", ciks))
        return pd.DataFrame([{"accession_number": "2"}]), IngestRunResult(
            mode="historical",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            ciks_count=len(ciks or set()),
            candidates_seen=2,
            skipped_existing=0,
            downloaded=0,
            failures=0,
            total_rows=2,
            output_root=str(tmp_path),
            documents_root=str(tmp_path / "documents-sixk"),
            document_partitions=(),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=2.json"),
            dataset_name="documents-sixk",
        )

    def fake_triage_pending_documents(**kwargs: object) -> pd.DataFrame:
        calls.append(("sixk", kwargs["batch_size"]))
        return pd.DataFrame([{"item_id": "snippet-1"}, {"item_id": "snippet-2"}])

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
        "cdt.pipeline.acquire_scraped_sixk_documents",
        fake_acquire_scraped_sixk_documents,
    )
    monkeypatch.setattr(
        "cdt.pipeline.triage_pending_documents", fake_triage_pending_documents
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
            sixk_batch_size=15,
            genres=DEFAULT_GENRES,
        )
    )

    assert result.ingest is not None
    assert result.ingest.total_rows == 1
    assert result.itemized_rows == 1
    assert result.classified_rows == 1
    assert result.sixk_ingest is not None
    assert result.sixk_ingest.total_rows == 2
    assert result.sixk_snippet_rows == EXPECTED_SIXK_SNIPPETS
    assert result.extracted_rows == 1
    assert result.matched_rows == 1
    assert result.debt_instrument_rows == 1
    # Both genres when asked for both, and the same CIKs asked of each: the
    # caller says which issuers and which dates, not which forms those issuers
    # filed. What a scheduled run defaults to is pinned on the orchestrator, in
    # `test_scheduled_runs_prepare_both_genres_by_default`.
    assert result.genres == DEFAULT_GENRES
    assert calls == [
        ("ingest", {"320193"}),
        ("itemize", 11),
        ("classify", 12),
        ("ingest-sixk", {"320193"}),
        ("sixk", 15),
        ("extract", 13),
        ("match", 14),
    ]


def test_genres_narrow_the_run_to_one_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--genres 6-K` runs the 6-K chain and not one 8-K stage.

    The chains read different documents datasets and write different
    classification sources, so narrowing leaves the other genre's partitions
    exactly as they were — pending, for a later run — rather than half-done.
    """
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("1023514\n", encoding="utf-8")
    calls: list[str] = []

    def unexpected(name: str) -> Callable[..., object]:
        def fail(*args: object, **kwargs: object) -> object:
            del args, kwargs
            msg = f"{name} must not run when the 8-K genre is excluded"
            raise AssertionError(msg)

        return fail

    monkeypatch.setattr("cdt.pipeline.run_ingest_pipeline", unexpected("ingest"))
    monkeypatch.setattr("cdt.pipeline.itemize_pending_documents", unexpected("itemize"))
    monkeypatch.setattr("cdt.pipeline.classify_pending_items", unexpected("classify"))
    monkeypatch.setattr(
        "cdt.pipeline.acquire_scraped_sixk_documents",
        lambda config, **kwargs: (
            calls.append("ingest-sixk"),
            (pd.DataFrame(), _sixk_ingest_result(tmp_path, kwargs.get("ciks"))),
        )[1],
    )
    monkeypatch.setattr(
        "cdt.pipeline.triage_pending_documents",
        lambda **kwargs: (calls.append("sixk"), pd.DataFrame())[1],
    )
    monkeypatch.setattr(
        "cdt.pipeline.extract_pending_items",
        lambda **kwargs: (calls.append("extract"), pd.DataFrame())[1],
    )
    monkeypatch.setattr(
        "cdt.pipeline.match_pending_mentions",
        lambda **kwargs: (
            calls.append("match"),
            {
                "debt_instrument_mentions": pd.DataFrame(),
                "debt_instrument": pd.DataFrame(),
            },
        )[1],
    )

    result = run_pipeline(
        PipelineConfig(
            mode="historical",
            cik_file=str(cik_file),
            start_date=date(2026, 9, 8),
            end_date=date(2026, 9, 8),
            artifact_root=str(tmp_path),
            genres=(GENRE_6K,),
        )
    )

    assert calls == ["ingest-sixk", "sixk", "extract", "match"]
    assert result.genres == (GENRE_6K,)
    # Absent, not empty: the 8-K chain did not run, which a zero-row result
    # would not distinguish from a run that found no filings.
    assert result.ingest is None
    assert result.sixk_ingest is not None


def test_the_sixk_chain_can_take_its_own_cik_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list chosen for 8-K coverage may contain no foreign private issuers."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    sixk_cik_file = tmp_path / "fpi-ciks.txt"
    sixk_cik_file.write_text("1023514\n", encoding="utf-8")
    asked: dict[str, set[str] | None] = {}

    monkeypatch.setattr(
        "cdt.pipeline.run_ingest_pipeline",
        lambda config, **kwargs: (
            asked.__setitem__("8-K", kwargs.get("ciks")),
            (pd.DataFrame(), _sixk_ingest_result(tmp_path, kwargs.get("ciks"))),
        )[1],
    )
    monkeypatch.setattr(
        "cdt.pipeline.itemize_pending_documents", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "cdt.pipeline.classify_pending_items", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "cdt.pipeline.acquire_scraped_sixk_documents",
        lambda config, **kwargs: (
            asked.__setitem__("6-K", kwargs.get("ciks")),
            (pd.DataFrame(), _sixk_ingest_result(tmp_path, kwargs.get("ciks"))),
        )[1],
    )
    monkeypatch.setattr(
        "cdt.pipeline.triage_pending_documents", lambda **kwargs: pd.DataFrame()
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

    run_pipeline(
        PipelineConfig(
            mode="historical",
            cik_file=str(cik_file),
            sixk_cik_file=str(sixk_cik_file),
            start_date=date(2026, 9, 8),
            end_date=date(2026, 9, 8),
            artifact_root=str(tmp_path),
            genres=DEFAULT_GENRES,
        )
    )

    assert asked == {"8-K": {"320193"}, "6-K": {"1023514"}}


def test_a_config_that_names_no_genres_does_not_acquire_the_sixk_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building a config in code is not asking to scrape and pay a model.

    The 6-K chain starts with a network scrape and ends in a paid LLM call, so
    a caller that never mentions genres must not get it: the chain would run
    for real in any test or script that stubs only the 8-K stages, which is
    every caller written before 6-K existed. The scheduled run still prepares
    both -- both entry points default their flag to `DEFAULT_GENRES` and pass
    it through, pinned by `test_scheduled_runs_prepare_both_genres_by_default`.
    """
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")

    def explode(**_kwargs: object) -> None:
        raise AssertionError("the 6-K chain ran without being asked for")

    monkeypatch.setattr("cdt.pipeline.acquire_scraped_sixk_documents", explode)
    monkeypatch.setattr("cdt.pipeline.triage_pending_documents", explode)

    assert PipelineConfig(mode="historical", cik_file=str(cik_file)).genres == (
        GENRE_8K,
    )
    # And DEFAULT_GENRES stays the wider vocabulary the flags default to, so
    # narrowing this field cannot quietly narrow what `--genres 6-K` accepts.
    assert DEFAULT_GENRES == (GENRE_8K, GENRE_6K)


def test_normalize_genres_parses_validates_and_orders() -> None:
    """A genre list is parsed once, in pipeline order, or rejected."""
    assert normalize_genres("8-K,6-K") == DEFAULT_GENRES
    # Order is the pipeline's, not the caller's, and duplicates collapse.
    assert normalize_genres("6-K,8-K") == DEFAULT_GENRES
    assert normalize_genres("6-k,6-K") == (GENRE_6K,)
    assert normalize_genres([GENRE_8K]) == (GENRE_8K,)
    # A typo must not read as "prepare less than asked": a run that quietly
    # prepared nothing looks exactly like a corpus with no filings.
    with pytest.raises(ValueError, match="unknown genre"):
        normalize_genres("8-K,10-K")
    with pytest.raises(ValueError, match="no genres selected"):
        normalize_genres("")
    with pytest.raises(ValueError, match="no genres selected"):
        normalize_genres([])


def test_a_run_refuses_a_genre_selection_that_prepares_nothing(
    tmp_path: Path,
) -> None:
    """Validated in the run, not only in the CLI that usually builds the config."""
    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no genres selected"):
        run_pipeline(
            PipelineConfig(
                mode="historical",
                cik_file=str(cik_file),
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                artifact_root=str(tmp_path),
                genres=(),
            )
        )


def _sixk_ingest_result(tmp_path: Path, ciks: set[str] | None) -> IngestRunResult:
    return IngestRunResult(
        mode="historical",
        start_date=date(2026, 9, 8),
        end_date=date(2026, 9, 8),
        ciks_count=len(ciks or set()),
        candidates_seen=0,
        skipped_existing=0,
        downloaded=0,
        failures=0,
        total_rows=0,
        output_root=str(tmp_path),
        documents_root=str(tmp_path / "documents-sixk"),
        document_partitions=(),
        failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
        run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        dataset_name="documents-sixk",
    )


def test_run_pipeline_processes_small_seeded_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline should complete on one seeded document batch.

    Scoped to the 8-K genre: the seeded batch is an 8-K submission, and the
    6-K chain would go looking for filings in a bucket this test does not
    stand up. Genre narrowing is covered on its own below.
    """
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
                "retired_by_json": "[]",
                "split_of": None,
                "lenders_json": '[{"mentions": [{"text": "Acme Bank"}]}]',
                "lenders_known_incomplete": False,
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
            genres=(GENRE_8K,),
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

    for table_name, dataset_root_fns in FINAL_OUTPUT_TABLES.items():
        # The first dataset only. `items` unions a second one, and these tests
        # are about the pointer, the guard and pruning — seeding both would
        # double the row counts they assert on. The union has its own test.
        write_partition_table(
            dataset_root_fns[0](str(artifact_root)),
            partition={"date": "2024-01-02", "shard": "0001"},
            table=pd.DataFrame(
                [{"id": f"{table_name}-{index}"} for index in range(rows)]
            ),
        )


def _mention_frame(*names: str) -> pd.DataFrame:
    """Return one mention per name, all for one issuer, as the extractor publishes."""
    from cdt.extractor.core import DEBT_INSTRUMENT_MENTION_COLUMNS

    rows = []
    for index, name in enumerate(names, start=1):
        row = dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS)
        row.update(
            {
                "debt_instrument_mention_id": f"m-{index}",
                "item_id": f"item-{index}",
                "raw_id": "i-1",
                "accession_number": f"000{index}",
                "cik": "320193",
                "company_name": "Example Inc.",
                "date": f"202{index}-01-02",
                "name": name,
                "start_date": f"202{index}-01-01",
                "principal_amount": "100000000",
                "retired_by_json": "[]",
                "parties_json": "[]",
                "lender_disclosure": "complete",
                "name_json": "{}",
                "start_date_json": "{}",
                "maturity_date_json": "{}",
                "amounts_json": "[]",
                "dates_json": "[]",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=DEBT_INSTRUMENT_MENTION_COLUMNS)


def test_match_and_finalize_runs_the_lineage_pass_after_every_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass is part of the pipeline, renews the lease, and skips an empty corpus.

    Behind `cdt match --infer-lineage` it never ran in production, which is why
    537 of 542 instruments published as lineage heads (#170). It reads three
    whole datasets and rewrites every shard, so it must renew the writer lease
    like the phases around it (#89).
    """
    from cdt import pipeline as pipeline_module

    calls: list[dict[str, object]] = []
    renewals: list[int] = []

    def fake_pass(artifact_root: object, **kwargs: object) -> dict[str, int]:
        calls.append({"artifact_root": str(artifact_root), **kwargs})
        return {"links": 0, "reopened": 0, "heads_before": 0, "heads_after": 0}

    monkeypatch.setattr(pipeline_module, "apply_lineage_inference_pass", fake_pass)

    empty_root = tmp_path / "empty"
    pipeline_module.run_match_and_finalize(
        artifact_root=empty_root, renew=lambda: renewals.append(1)
    )
    assert calls == []

    root = tmp_path / "artifacts"
    write_partition_table(
        root / "mentions",
        partition={"date": "2022-01-02", "shard": "0001"},
        table=_mention_frame(
            "Credit Agreement", "Second Amended and Restated Credit Agreement"
        ),
    )
    pipeline_module.run_match_and_finalize(
        artifact_root=root, renew=lambda: renewals.append(1)
    )
    assert len(calls) == 1
    assert calls[0]["artifact_root"] == str(root)
    assert calls[0]["data_dir"] is None
    assert callable(calls[0]["renew"])
    assert renewals  # the lease was renewed around the pass


def test_published_items_union_both_genres_so_every_mention_can_join(
    tmp_path: Path,
) -> None:
    """`items/latest.parquet` carries 6-K snippets beside 8-K items (#172).

    Every consumer joins a mention to its source row by `item_id` -- the
    website reads the item text, the filing's SEC URL and its accession number
    off that row. Publishing only the 8-K units leaves a 6-K mention joining to
    nothing, and the failure is silent: the schema is intact, the join is just
    empty, so the instrument renders with no text and no link.
    """
    from cdt.itemizer.core import ITEM_COLUMNS
    from cdt.pipeline import write_final_output_tables
    from cdt.sixk.stage import SIXK_SNIPPET_COLUMNS, sixk_snippets_root

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _seed_final_tables(artifact_root)
    write_partition_table(
        sixk_snippets_root(str(artifact_root)),
        partition={"date": "2024-01-02", "shard": "0001"},
        table=pd.DataFrame(
            [
                {
                    **dict.fromkeys(SIXK_SNIPPET_COLUMNS),
                    "item_id": "000165495426008172-6K-0-0-512",
                    "item": "000165495426008172:0:0",
                    "accession_number": "000165495426008172",
                    "url": "https://sec.example/6k.txt",
                    "text": "Global Program for the issuance of Notes.",
                    "date": "2024-01-02",
                    "sixk_member_windows": "0,1",
                }
            ],
            columns=SIXK_SNIPPET_COLUMNS,
        ),
    )

    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )

    published = read_table(str(final_root / "items" / "latest.parquet"))
    # The 6-K row is there, and joinable by the id a mention carries.
    by_id = {str(row["item_id"]): row for _, row in published.iterrows()}
    assert "000165495426008172-6K-0-0-512" in by_id
    joined = by_id["000165495426008172-6K-0-0-512"]
    assert joined["url"] == "https://sec.example/6k.txt"
    assert joined["text"] == "Global Program for the issuance of Notes."
    assert joined["accession_number"] == "000165495426008172"
    # The 8-K rows the table already published are still there.
    assert len(published) == EXPECTED_UNIONED_ITEM_ROWS
    # And the table keeps the itemizer's shape plus the one column that says
    # which genre a row came from: the snippet's own span and verdict columns
    # stay in `sixk-snippets` rather than widening this one.
    assert list(published.columns) == [*ITEM_COLUMNS, "form_type"]
    # Nothing else in the row distinguishes an item section from a snippet, so
    # a consumer that needs to branch reads this rather than pattern-matching
    # the item id.
    assert joined["form_type"] == "6-K"
    eightk = [
        row
        for _, row in published.iterrows()
        if str(row["item_id"]) != "000165495426008172-6K-0-0-512"
    ]
    assert {str(row["form_type"]) for row in eightk} == {"8-K"}


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


def test_resolve_mode_dates_daily_uses_lookback_window() -> None:
    """Daily defaults to a rolling lookback ending yesterday (#90)."""
    from cdt.pipeline import DAILY_LOOKBACK_DAYS

    start, end = resolve_mode_dates("daily", None, None)

    today = date.today()
    assert end == today.fromordinal(today.toordinal() - 1)
    assert start == today.fromordinal(today.toordinal() - DAILY_LOOKBACK_DAYS)


def _stage_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    instruments: pd.DataFrame,
) -> None:
    """Stub ingest through match so a run reaches the lineage pass cheaply."""

    def fake_run_ingest_pipeline(
        config: object,
        *,
        ciks: set[str] | None = None,
        s3_client: object | None = None,
    ) -> tuple[pd.DataFrame, IngestRunResult]:
        del config, s3_client
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
            document_partitions=(),
            failure_file=str(tmp_path / "failures" / "ingest_failures.json"),
            run_manifest=str(tmp_path / "runs" / "ingest" / "run_id=1.json"),
        )

    monkeypatch.setattr("cdt.pipeline.run_ingest_pipeline", fake_run_ingest_pipeline)
    monkeypatch.setattr(
        "cdt.pipeline.itemize_pending_documents",
        lambda **_: pd.DataFrame([{"item_id": "item-1"}]),
    )
    monkeypatch.setattr(
        "cdt.pipeline.classify_pending_items",
        lambda **_: pd.DataFrame([{"item_id": "item-1", "relevance": True}]),
    )
    monkeypatch.setattr(
        "cdt.pipeline.extract_pending_items",
        lambda **_: pd.DataFrame([{"debt_instrument_mention_id": "mention-1"}]),
    )
    monkeypatch.setattr(
        "cdt.pipeline.match_pending_mentions",
        lambda **_: {
            "debt_instrument_mentions": pd.DataFrame(
                [{"debt_instrument_mention_id": "mention-1", "edge_type": "member"}]
            ),
            "debt_instrument": instruments,
        },
    )


@pytest.mark.parametrize(
    ("instruments", "expected_calls"),
    [
        (pd.DataFrame([{"debt_instrument_id": "instrument-1"}]), 1),
        (pd.DataFrame(columns=["debt_instrument_id"]), 0),
    ],
    ids=["matched-something", "matched-nothing"],
)
def test_run_pipeline_runs_the_lineage_pass_between_match_and_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instruments: pd.DataFrame,
    expected_calls: int,
) -> None:
    """`cdt pipeline` and the live backend must infer lineage before publishing.

    The pass was wired into `run_match_and_finalize` and `cdt match` only, so
    `cdt pipeline` and `cdt-orchestrator --extractor-backend live` still
    published the un-inferred lineage #170 describes, on a root the batch
    backend would have fixed. The guard matches `run_match_and_finalize`'s:
    nothing matched means three empty datasets read to write none.
    """
    from cdt import pipeline as pipeline_module

    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    _stage_stubs(monkeypatch, tmp_path, instruments=instruments)
    monkeypatch.setattr(
        pipeline_module,
        "apply_lineage_inference_pass",
        lambda artifact_root, **kwargs: (
            calls.append({"artifact_root": str(artifact_root), **kwargs}),
            {"links": 0, "reopened": 0, "heads_before": 0, "heads_after": 0},
        )[1],
    )

    renewals: list[int] = []
    run_pipeline(
        PipelineConfig(
            mode="historical",
            cik_file=str(cik_file),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            artifact_root=str(tmp_path / "artifacts"),
        ),
        renew=lambda: renewals.append(1),
    )

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls[0]["artifact_root"] == str(tmp_path / "artifacts")
        assert callable(calls[0]["renew"])
        assert renewals


def _publish_all_four(final_root: Path) -> None:
    """Stand in for a complete published generation under the database root.

    Written directly rather than by running a real publish: the gate inspects
    exactly these four objects, and seeding the *artifact* datasets with
    placeholder rows would hand the matcher mention rows it cannot parse.
    """
    from cdt.pipeline import FINAL_OUTPUT_TABLES
    from cdt.storage import write_table

    for table_name in FINAL_OUTPUT_TABLES:
        write_table(
            str(final_root / table_name / "latest.parquet"),
            pd.DataFrame([{"id": f"{table_name}-0"}]),
        )


def _record_published_generation(artifact_root: Path, final_root: Path) -> None:
    """Stand in for a complete publish: four live objects and a pointer.

    The gate compares a digest of the source partitions against the one the
    pointer recorded when the live generation was written, so "nothing has
    changed since the last publish" needs both halves seeded. Call this
    *after* the sources are in place and before changing them.
    """
    from cdt.pipeline import (
        PUBLISH_SOURCE_DIGEST_KEY,
        final_pointer_path,
        publish_source_digest,
    )
    from cdt.storage import write_json_artifact

    _publish_all_four(final_root)
    write_json_artifact(
        final_pointer_path(str(artifact_root)),
        {
            "run_id": "seeded-generation",
            PUBLISH_SOURCE_DIGEST_KEY: publish_source_digest(str(artifact_root)),
        },
    )


def test_match_and_finalize_skips_the_publish_when_no_source_partition_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-op run must not re-read the whole corpus to republish it (#110).

    Measured in production: publishing a delta of 14 documents took 25 minutes
    and 21,214 sequential GETs at ~70 ms — the cost is request count, not bytes,
    and it was paid whether or not the run produced anything. Both
    ``run_batch_backend`` and ``run_poll`` finalize, so one batch cycle paid it
    at least twice.
    """
    from cdt import pipeline as pipeline_module

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _record_published_generation(artifact_root, final_root)

    published: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "write_final_output_tables",
        lambda **kwargs: published.append(kwargs) or {},
    )

    written = pipeline_module.run_match_and_finalize(
        artifact_root=artifact_root, final_database_root=str(final_root)
    )

    assert published == []
    assert written == {}


def test_the_gate_fires_on_a_corpus_that_already_has_instruments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case #110 was actually filed about, which an emptiness gate misses.

    ``match_pending_mentions`` rewrites and returns every shard's *full*
    instrument table rather than a delta, so gating on "did match produce
    instruments" is empty only on a corpus that has never produced one. #110
    was measured on a root with 542 instruments publishing a 14-document delta;
    a gate keyed on emptiness would never have fired there. This run has
    instruments and changed nothing, and must still skip.
    """
    from cdt import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "apply_lineage_inference_pass", lambda *a, **k: {}
    )
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    write_partition_table(
        artifact_root / "mentions",
        partition={"date": "2022-01-02", "shard": "0001"},
        table=_mention_frame("Credit Agreement"),
    )
    # Match once so the corpus genuinely holds instruments, then record the
    # generation those sources produced.
    pipeline_module.run_match_and_finalize(artifact_root=artifact_root)
    matched = read_dataset(str(artifact_root / "debt-instruments"))
    assert (
        not matched.empty
    ), "the corpus must hold instruments for this to mean anything"
    _record_published_generation(artifact_root, final_root)

    published: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "write_final_output_tables",
        lambda **kwargs: published.append(kwargs) or {},
    )

    written = pipeline_module.run_match_and_finalize(
        artifact_root=artifact_root, final_database_root=str(final_root)
    )

    assert published == []
    assert written == {}


def test_new_items_publish_even_though_match_produced_no_instruments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``items`` is published too, and itemize writes it before match runs (#110).

    ``FINAL_OUTPUT_TABLES`` publishes four tables, and ``items`` comes from the
    itemizer (and, on dev, the 6-K triage) — stages upstream of match. A gate
    that asked only about instruments could not see them, so a run that
    itemized new rows and matched nothing left the published ``items`` table
    stale with nothing to warn about and nothing to repair it.
    """
    from cdt import pipeline as pipeline_module

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _record_published_generation(artifact_root, final_root)
    write_partition_table(
        artifact_root / "items",
        partition={"date": "2022-01-02", "shard": "0001"},
        table=pd.DataFrame([{"item_id": "item-1"}]),
    )

    published: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "write_final_output_tables",
        lambda **kwargs: published.append(kwargs) or {},
    )

    pipeline_module.run_match_and_finalize(
        artifact_root=artifact_root, final_database_root=str(final_root)
    )

    assert len(published) == 1


def test_a_real_publish_records_the_digest_the_next_run_gates_on(
    tmp_path: Path,
) -> None:
    """The two halves have to meet, and nothing else pins that they do (#110).

    Every other gate test seeds the pointer by calling ``publish_source_digest``
    itself. If ``write_final_output_tables`` stopped recording one, all of them
    would still pass while the gate never fired again in production — the exact
    always-false guard this replaced. So this publishes for real, then asks the
    gate, then changes a source and asks again.
    """
    from cdt.pipeline import (
        PUBLISH_SOURCE_DIGEST_KEY,
        final_pointer_path,
        publish_would_republish_nothing,
        write_final_output_tables,
    )
    from cdt.storage import read_json_artifact

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    write_partition_table(
        artifact_root / "items",
        partition={"date": "2022-01-02", "shard": "0001"},
        table=pd.DataFrame([{"item_id": "item-1"}]),
    )

    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )

    pointer = read_json_artifact(final_pointer_path(str(artifact_root)))
    assert pointer[PUBLISH_SOURCE_DIGEST_KEY]
    assert publish_would_republish_nothing(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )

    write_partition_table(
        artifact_root / "items",
        partition={"date": "2022-01-03", "shard": "0001"},
        table=pd.DataFrame([{"item_id": "item-2"}]),
    )

    assert not publish_would_republish_nothing(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )


def test_a_pointer_with_no_recorded_digest_publishes(tmp_path: Path) -> None:
    """Generations published before the gate existed record no digest (#110).

    Their sources may have moved any number of times since, so the honest
    answer is "unknown" and the publish goes ahead, which records a digest for
    next time.
    """
    from cdt.pipeline import final_pointer_path, publish_would_republish_nothing
    from cdt.storage import write_json_artifact

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _publish_all_four(final_root)
    write_json_artifact(
        final_pointer_path(str(artifact_root)),
        {"run_id": "pre-gate", "tables": {}},
    )

    assert not publish_would_republish_nothing(
        artifact_root=str(artifact_root),
        final_database_root=str(final_root),
    )


def test_force_publishes_even_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force still overrides the gate (#110).

    It is no longer the documented way to recover a run that crashed between
    writing a dataset and publishing it — the datasets moved, so the digest
    moved, so the next run republishes on its own. That matters because
    ``force`` is the pipeline-wide flag and also disables the shrinkage guard.
    """
    from cdt import pipeline as pipeline_module

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _record_published_generation(artifact_root, final_root)

    published: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "write_final_output_tables",
        lambda **kwargs: published.append(kwargs) or {},
    )

    pipeline_module.run_match_and_finalize(
        artifact_root=artifact_root,
        final_database_root=str(final_root),
        force=True,
    )

    assert len(published) == 1


def test_an_unpublished_database_root_publishes_despite_unchanged_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly pointed output root must get its first generation (#110).

    Otherwise redirecting --final-database-root at a corpus that is already
    fully matched would leave the new root empty until someone happened to pass
    --force.
    """
    from cdt import pipeline as pipeline_module

    artifact_root = tmp_path / "artifacts"

    published: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "write_final_output_tables",
        lambda **kwargs: published.append(kwargs) or {},
    )

    pipeline_module.run_match_and_finalize(
        artifact_root=artifact_root,
        final_database_root=str(tmp_path / "brand-new"),
    )

    assert len(published) == 1


def test_a_partially_published_database_root_publishes(tmp_path: Path) -> None:
    """Three of four latest.parquet objects is not a complete generation (#110).

    The digest matches here, so the missing fourth object is the only reason
    left to publish — which is what this pins.
    """
    from cdt.pipeline import (
        FINAL_OUTPUT_TABLES,
        PUBLISH_SOURCE_DIGEST_KEY,
        final_pointer_path,
        publish_source_digest,
        publish_would_republish_nothing,
    )
    from cdt.storage import write_json_artifact, write_table

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    for table_name in list(FINAL_OUTPUT_TABLES)[:-1]:
        write_table(
            str(final_root / table_name / "latest.parquet"),
            pd.DataFrame([{"id": "x"}]),
        )
    write_json_artifact(
        final_pointer_path(str(artifact_root)),
        {
            "run_id": "seeded-generation",
            PUBLISH_SOURCE_DIGEST_KEY: publish_source_digest(str(artifact_root)),
        },
    )

    assert not publish_would_republish_nothing(
        artifact_root=str(artifact_root),
        final_database_root=str(final_root),
    )


def test_no_final_database_root_skips_without_listing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default configuration has nowhere to publish to (#110).

    ``final_database_root`` defaults to None on both ``PipelineConfig`` and
    ``run_match_and_finalize``, which is what `cdt pipeline` gets without
    ``--final-database-root``. ``write_final_output_tables`` returns before it
    reads anything in that case, so the gate answers the same and should not
    pay a listing to find out.
    """
    from cdt import pipeline as pipeline_module
    from cdt.pipeline import publish_would_republish_nothing

    def _explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("no publish target means nothing needs listing")

    monkeypatch.setattr(pipeline_module, "publish_source_digest", _explode)

    assert publish_would_republish_nothing(
        artifact_root=str(tmp_path / "artifacts"),
        final_database_root=None,
    )


def test_the_publish_gate_leaves_the_live_tables_untouched(tmp_path: Path) -> None:
    """End to end, unmocked: a skipped publish does not disturb what is live (#110).

    Deliberately not mocking ``write_final_output_tables``: if the gate failed
    to fire, the real publish would read four empty artifact datasets and
    clobber these four objects (or trip the shrinkage guard). Unchanged bytes
    are the only proof that nothing ran.
    """
    from cdt import pipeline as pipeline_module
    from cdt.pipeline import FINAL_OUTPUT_TABLES

    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _record_published_generation(artifact_root, final_root)
    paths = [final_root / name / "latest.parquet" for name in FINAL_OUTPUT_TABLES]
    before = [path.read_bytes() for path in paths]

    written = pipeline_module.run_match_and_finalize(
        artifact_root=artifact_root, final_database_root=str(final_root)
    )

    assert written == {}
    assert [path.read_bytes() for path in paths] == before


def test_normalize_snapshot_text_is_not_the_publish_cost(tmp_path: Path) -> None:
    """Recorded, not optimized: it is 1.2% of read+normalize (#110).

    ``normalize_snapshot_text`` maps a Python lambda over every object column of
    every row, which looks like it should dominate. Measured on
    data/genwindow-eval-apr's 5,794-row x 16-column ``items`` dataset it is
    0.113s against an 8.9s read — 1.2% — so the 25 minutes is I/O, not this, and
    it was deliberately left alone. This test pins only that it still nulls
    placeholders, which is the behaviour the dashboard depends on.
    """
    del tmp_path
    from cdt.pipeline import normalize_snapshot_text

    normalized = normalize_snapshot_text(
        pd.DataFrame([{"a": "nan", "b": "real", "c": True}])
    )

    assert normalized["a"].to_list() == [None]
    assert normalized["b"].to_list() == ["real"]
    assert normalized["c"].to_list() == [True]


@pytest.mark.parametrize(
    ("change_a_source", "expected_publishes"),
    [(True, 1), (False, 0)],
    ids=["source-changed", "sources-unchanged"],
)
def test_run_pipeline_skips_the_publish_when_nothing_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_a_source: bool,
    expected_publishes: int,
) -> None:
    """`cdt pipeline` and the live backend pay the same publish, so same gate (#110).

    ``run_match_and_finalize`` is the batch backend's path; this is the other
    two entry points. Gating only one of them would leave the identical
    25-minute no-op publish in place for `cdt pipeline` and
    `cdt-orchestrator --extractor-backend live`, which is the shape #170 took
    when the lineage pass was wired into one path and not the others.
    """
    from cdt import pipeline as pipeline_module

    cik_file = tmp_path / "ciks.txt"
    cik_file.write_text("320193\n", encoding="utf-8")
    _stage_stubs(
        monkeypatch,
        tmp_path,
        instruments=pd.DataFrame([{"debt_instrument_id": "instrument-1"}]),
    )
    monkeypatch.setattr(
        pipeline_module, "apply_lineage_inference_pass", lambda *a, **k: {}
    )
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _record_published_generation(artifact_root, final_root)
    if change_a_source:
        write_partition_table(
            artifact_root / "items",
            partition={"date": "2022-01-02", "shard": "0001"},
            table=pd.DataFrame([{"item_id": "item-1"}]),
        )

    published: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "write_final_output_tables",
        lambda **kwargs: published.append(kwargs) or {},
    )

    run_pipeline(
        PipelineConfig(
            mode="historical",
            cik_file=str(cik_file),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            artifact_root=str(tmp_path / "artifacts"),
            final_database_root=str(final_root),
        )
    )

    assert len(published) == expected_publishes
