"""Tests for itemizing 8-K documents and classifying items over stored partitions."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from support import (
    FakeModel,
    _fact,
    build_mention_row,
    seed_document_partitions_across_months,
)

from cdt.classifier import classifications_root, classify_pending_items
from cdt.classifier import core as classifier_core
from cdt.completion import completion_registry_root, load_completed_partitions
from cdt.datasets import (
    load_row_failures,
    run_manifest_path,
)
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.state import ExtractionRowState
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import core as itemizer_core
from cdt.itemizer import itemize_pending_documents, items_root
from cdt.storage.objects import (
    artifact_exists,
    read_json_artifact,
)
from cdt.storage.tables import (
    read_dataset,
    write_partition_table,
)


def seed_document_partition(tmp_path: Path) -> str:
    """Write one canonical document partition."""
    table = pd.DataFrame(
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
        ],
        columns=DOCUMENT_COLUMNS,
    )
    return write_partition_table(
        tmp_path / "documents",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=table,
    )


def seed_document_partitions(tmp_path: Path) -> list[str]:
    """Write multiple canonical document partitions."""
    partitions: list[tuple[str, str, str, str]] = [
        (
            "000114036126006577",
            "320193",
            "2024-01-02",
            "0001",
        ),
        (
            "000078901925000010",
            "789019",
            "2024-01-03",
            "0002",
        ),
    ]
    paths: list[str] = []
    for accession_number, cik, filing_date, shard in partitions:
        table = pd.DataFrame(
            [
                {
                    "accession_number": accession_number,
                    "cik": cik,
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
                    "date": filing_date,
                    "resource_uri": None,
                }
            ],
            columns=DOCUMENT_COLUMNS,
        )
        paths.append(
            write_partition_table(
                tmp_path / "documents",
                partition={"date": filing_date, "shard": shard},
                table=table,
            )
        )
    return paths


def test_itemize_document_record_blanks_missing_company_name() -> None:
    """A missing document company name must not become the literal text 'nan'."""
    document = {
        "accession_number": "000114036126006577",
        "cik": "1821075",
        "company_name": float("nan"),
        "url": "https://sec.example/full.txt",
        "date": "2026-01-02",
        "text": """
ITEM INFORMATION: Other Events
<DOCUMENT>
<TYPE>8-K
<TEXT>
Item 8.01 Other Events.
The Company issued a promissory note.
</TEXT>
</DOCUMENT>
""".strip(),
    }

    sections = itemizer_core.itemize_document_record(document)

    assert sections
    assert {section.company_name for section in sections} == {""}


def test_itemize_pending_documents_writes_canonical_partitions(tmp_path: Path) -> None:
    """Itemization should consume document partitions and write item partitions."""
    seed_document_partition(tmp_path)

    items = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)

    written = read_dataset(items_root(tmp_path))
    assert len(items) == 1
    assert written["item_id"].to_list() == ["000114036126006577-8-01"]


def test_itemize_pending_documents_drains_all_partitions(tmp_path: Path) -> None:
    """Itemization should process all pending partitions across chunks."""
    seed_document_partitions(tmp_path)

    items = itemize_pending_documents(artifact_root=tmp_path, batch_size=1)

    written = read_dataset(items_root(tmp_path))
    assert len(items) == 2
    assert sorted(written["item_id"].to_list()) == [
        "000078901925000010-8-01",
        "000114036126006577-8-01",
    ]


def test_itemize_pending_documents_skips_empty_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty itemize results should not write parquet and should not rerun."""
    seed_document_partition(tmp_path)
    calls = 0

    def fake_itemize_documents(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        del args, kwargs
        calls += 1
        return pd.DataFrame(columns=itemizer_core.ITEM_COLUMNS)

    monkeypatch.setattr(itemizer_core, "itemize_documents", fake_itemize_documents)

    first = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    second = itemize_pending_documents(artifact_root=tmp_path, batch_size=5)

    assert first.empty
    assert second.empty
    assert calls == 1
    assert not artifact_exists(
        items_root(tmp_path) + "/date=2024-01-02/shard=0001/part-0000.parquet"
    )


def test_classify_pending_items_writes_canonical_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification should consume item partitions and write classification partitions."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    classified = classify_pending_items(artifact_root=tmp_path, batch_size=5)

    written = read_dataset(classifications_root(tmp_path))
    assert len(classified) == 1
    assert written["label"].to_list() == ["relevant"]
    assert written["relevance"].to_list() == [True]


def test_classify_pending_items_drains_all_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification should process all pending partitions across chunks."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    classified = classify_pending_items(artifact_root=tmp_path, batch_size=1)

    written = read_dataset(classifications_root(tmp_path))
    assert len(classified) == 2
    assert written["label"].to_list().count("relevant") == 2


def test_classify_pending_items_skips_empty_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty classifier results should not write parquet and should not rerun."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    calls = 0

    def fake_classify_items(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        del args, kwargs
        calls += 1
        return pd.DataFrame(columns=classifier_core.CLASSIFIED_ITEM_COLUMNS)

    monkeypatch.setattr(classifier_core, "classify_items", fake_classify_items)

    first = classify_pending_items(artifact_root=tmp_path, batch_size=5)
    second = classify_pending_items(artifact_root=tmp_path, batch_size=5)

    assert first.empty
    assert second.empty
    assert calls == 1
    assert not artifact_exists(
        classifications_root(tmp_path) + "/date=2024-01-02/shard=0001/part-0000.parquet"
    )


def test_load_training_artifacts_rejects_sklearn_minor_mismatch(
    tmp_path: Path,
) -> None:
    """A pickle trained under a different sklearn minor must fail loudly (#109).

    The classifier is the relevance gate; sklearn itself only warns, and a
    silent scoring drift looks like normal fluctuation in corpus size.
    """
    classifier_core.save_training_artifacts(
        model_dir=tmp_path,
        model=FakeModel(),
        metadata={"threshold": 0.5, "sklearn_version": "0.24.2"},
    )

    with pytest.raises(RuntimeError, match="0.24.2"):
        classifier_core.load_training_artifacts(tmp_path)


def test_load_training_artifacts_accepts_matching_sklearn_minor(
    tmp_path: Path,
) -> None:
    """Same major.minor loads fine, including across patch releases (#109)."""
    import sklearn

    patch_variant = ".".join([*sklearn.__version__.split(".")[:2], "999"])
    classifier_core.save_training_artifacts(
        model_dir=tmp_path,
        model=FakeModel(),
        metadata={"threshold": 0.5, "sklearn_version": patch_variant},
    )

    _, threshold, metadata = classifier_core.load_training_artifacts(tmp_path)

    assert threshold == 0.5
    assert metadata["sklearn_version"] == patch_variant


def test_load_training_artifacts_warns_without_recorded_version(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre-provenance metadata still loads, but the gap is logged (#109)."""
    classifier_core.save_training_artifacts(
        model_dir=tmp_path,
        model=FakeModel(),
        metadata={"threshold": 0.5},
    )

    with caplog.at_level(logging.WARNING, logger="cdt.classifier.core"):
        _, threshold, _ = classifier_core.load_training_artifacts(tmp_path)

    assert threshold == 0.5
    assert any("sklearn_version" in record.getMessage() for record in caplog.records)


def test_itemize_pending_documents_persists_progress_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash mid-stage must lose at most one batch, not the whole run (#111)."""
    seed_document_partitions(tmp_path)
    calls = 0
    real_itemize_documents = itemizer_core.itemize_documents

    def failing_itemize_documents(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "Read timed out"
            raise TimeoutError(msg)
        return real_itemize_documents(*args, **kwargs)

    monkeypatch.setattr(itemizer_core, "itemize_documents", failing_itemize_documents)

    with pytest.raises(TimeoutError):
        itemize_pending_documents(artifact_root=tmp_path, batch_size=1)

    assert len(load_completed_partitions("itemize", artifact_root=tmp_path)) == 1

    monkeypatch.setattr(itemizer_core, "itemize_documents", real_itemize_documents)
    resumed = itemize_pending_documents(artifact_root=tmp_path, batch_size=1)

    assert len(resumed) == 1
    assert len(load_completed_partitions("itemize", artifact_root=tmp_path)) == 2


def test_classify_pending_items_persists_progress_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash mid-stage must lose at most one batch, not the whole run (#111)."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    calls = 0
    real_classify_items = classifier_core.classify_items

    def failing_classify_items(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "Read timed out"
            raise TimeoutError(msg)
        return real_classify_items(*args, **kwargs)

    monkeypatch.setattr(classifier_core, "classify_items", failing_classify_items)

    with pytest.raises(TimeoutError):
        classify_pending_items(artifact_root=tmp_path, batch_size=1)

    assert len(load_completed_partitions("classify", artifact_root=tmp_path)) == 1


def test_stage_batch_boundaries_renew_the_writer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renew the lease once per batch, or a long stage is stolen mid-run (#111)."""
    seed_document_partitions(tmp_path)
    renewals: list[str] = []

    itemize_pending_documents(
        artifact_root=tmp_path,
        batch_size=1,
        renew=lambda: renewals.append("itemize"),
    )

    assert renewals.count("itemize") == 2

    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(
        artifact_root=tmp_path,
        batch_size=2,
        renew=lambda: renewals.append("classify"),
    )

    assert renewals.count("classify") == 1


def test_extract_pending_items_writes_mentions_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction consumes classifications and writes mentions plus the audit log.

    The mention carries a `prior`-marked commitment, so the mint fires and this
    covers the publish seam end to end (#211). Every publish path goes through
    `published_mention_rows`, but the seam test called the helper on a mention
    with no `prior` facts — the mint was a no-op there, so the seam was
    indistinguishable from the raw list and all four call sites could be
    swapped back to `row_state.debt_instrument_mentions` with a green suite.
    """
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

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
                "date": item_row["date"],
                "raw_id": "i-1",
                "name": "Term Loan",
                "start_date": "2024-01-01",
                "maturity_date": None,
                "amount": "$100 million",
                "amendment_of": None,
                "retired_by_json": "[]",
                "split_of": None,
                "parties_json": "[]",
                "lenders_known_incomplete": False,
                "name_json": "{}",
                "start_date_json": "{}",
                "maturity_date_json": "{}",
                "amounts_json": json.dumps(
                    [
                        _fact(
                            kind="commitment",
                            normalized_amount="300000000",
                            prior=True,
                        ),
                        _fact(
                            kind="commitment",
                            normalized_amount="250000000",
                            prior=False,
                        ),
                    ]
                ),
                "dates_json": json.dumps(
                    [
                        _fact(
                            kind="agreement",
                            normalized_date="2020-02-03",
                            prior=False,
                        )
                    ]
                ),
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    mentions = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )

    written = read_dataset(mentions_root(tmp_path))
    assert len(mentions) == 2
    assert written["debt_instrument_mention_id"].to_list()
    audit_files = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit_files) == 1

    # The minted prior state reaches the written partition, not just the
    # in-memory return value, and the successor points at it.
    minted = written[written["synthesized_by"] == "prior_state"]
    assert len(minted) == 1
    assert minted.iloc[0]["principal_amount"] == Decimal("300000000.00")
    successor = written[written["debt_instrument_mention_id"] == "m-1"].iloc[0]
    assert successor["amendment_of"] == minted.iloc[0]["debt_instrument_mention_id"]

    # and the audit record publishes the same rows the partition did
    audit = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
    audited = [
        mention
        for record in audit
        for mention in record.get("debt_instrument_mentions", [])
    ]
    assert sorted(str(mention.get("synthesized_by")) for mention in audited) == [
        "None",
        "prior_state",
    ]

    # `state.jsonl` keeps only what the model returned: the pointer the mint
    # writes onto the successor must not be persisted there.
    state_files = list((tmp_path / "extractor-runs").glob("run_id=*/state.jsonl"))
    for state_file in state_files:
        for line in state_file.read_text().splitlines():
            for mention in json.loads(line).get("debt_instrument_mentions", []):
                assert mention.get("synthesized_by") is None
                assert mention.get("amendment_of") is None


def test_extract_pending_items_drains_all_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction should process all pending partitions across chunks."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=1)

    async def fake_run_extraction_workflow(
        **kwargs: object,
    ) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {
                "debt_instrument_mention_id": f"m-{item_row['accession_number']}",
                "item_id": item_row["item_id"],
                "accession_number": item_row["accession_number"],
                "cik": item_row["cik"],
                "date": item_row["date"],
                "raw_id": "i-1",
                "name": "Term Loan",
                "start_date": "2024-01-01",
                "maturity_date": None,
                "amount": "$100 million",
                "amendment_of": None,
                "retired_by_json": "[]",
                "split_of": None,
                "parties_json": "[]",
                "lenders_known_incomplete": True,
                "name_json": "{}",
                "start_date_json": "{}",
                "maturity_date_json": "{}",
                "amounts_json": "[]",
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    mentions = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=1,
        client=None,
    )

    written = read_dataset(mentions_root(tmp_path))
    assert len(mentions) == 2
    assert sorted(written["debt_instrument_mention_id"].to_list()) == [
        "m-000078901925000010",
        "m-000114036126006577",
    ]
    audit_files = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit_files) == 1


def test_extract_pending_items_skips_empty_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty extraction results should not write parquet and should not rerun."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)
    calls = 0

    async def fake_run_extraction_workflow(
        **kwargs: object,
    ) -> ExtractionRowState:
        nonlocal calls
        item_row = kwargs["item_row"]
        calls += 1
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    first = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )
    second = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )

    assert first.empty
    assert second.empty
    assert calls == 1
    assert not artifact_exists(
        mentions_root(tmp_path) + "/date=2024-01-02/shard=0001/part-0000.parquet"
    )


def test_extract_failures_are_recorded_and_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped row lands in the failure registry; a later success clears it."""
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

    async def failing_workflow(**kwargs: object) -> ExtractionRowState:
        row_state = ExtractionRowState(
            item_row=kwargs["item_row"], stage_name="instrument_ie"
        )
        row_state.add_validation(["instrument_ie did not return valid JSON"])
        row_state.finish("ERROR")
        return row_state

    monkeypatch.setattr("cdt.extractor.live.run_extraction_workflow", failing_workflow)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # The partition is registered complete even though the row produced nothing,
    # which is exactly why the failure has to be recorded somewhere durable.
    # Read through the loader, not a fixed path: the registry is a prefix of
    # date shards since #191 and no test should re-encode its layout.
    assert load_completed_partitions("extract", artifact_root=tmp_path)
    failures = load_row_failures("extract", artifact_root=tmp_path)
    assert len(failures) == 1
    entry = next(iter(failures.values()))
    assert entry["backend"] == "live"
    assert entry["state"] == "ERROR"
    assert entry["error"] == "instrument_ie did not return valid JSON"
    assert entry["date"] and entry["shard"]

    async def succeeding_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {
                "debt_instrument_mention_id": "m-1",
                "item_id": item_row["item_id"],
                "accession_number": item_row["accession_number"],
                "cik": item_row["cik"],
                "date": item_row["date"],
                "raw_id": "i-1",
                "name": "Term Loan",
                "start_date": None,
                "maturity_date": None,
                "amount": None,
                "amendment_of": None,
                "retired_by_json": "[]",
                "split_of": None,
                "parties_json": "[]",
                "name_json": "{}",
                "start_date_json": "{}",
                "maturity_date_json": "{}",
                "amounts_json": "[]",
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow", succeeding_workflow
    )
    extract_pending_items(artifact_root=tmp_path, batch_size=5, force=True, client=None)

    assert load_row_failures("extract", artifact_root=tmp_path) == {}


def test_stage_manifest_points_at_the_registry_prefix(tmp_path: Path) -> None:
    """The manifest names where the registry actually is, shards included.

    Operators read this field to find the completion state of a run; before
    #191 it named a single object, and left unchanged it would keep naming a
    path that no longer exists.
    """
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)

    manifest = read_json_artifact(
        run_manifest_path("itemize", "latest", artifact_root=tmp_path)
    )
    prefix = completion_registry_root("itemize", artifact_root=tmp_path)
    assert manifest["completion_registry"] == prefix
    assert sorted(path.name for path in Path(prefix).iterdir()) == ["date=2024-01.json"]


def test_itemize_batch_progress_survives_a_mid_run_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#111's guarantee, re-pinned across the shard split (#191).

    #111 moved the registry save inside the chunk loop precisely so an
    interruption does not discard the run's progress -- up to 2.5 h of itemize
    on the real corpus. Sharding must not trade that away, so: interrupt a run
    partway and assert the completed prefix is persisted and skipped next time,
    including across a month boundary where the progress spans two shards.
    """
    from cdt.completion import load_completed_partitions

    days = ["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02", "2024-03-01"]
    paths = seed_document_partitions_across_months(
        tmp_path, [(day, f"{index:04d}") for index, day in enumerate(days)]
    )
    real_itemize = itemizer_core.itemize_documents
    calls: list[int] = []

    def count(*, fail_on: int | None) -> object:
        def wrapper(*args: object, **kwargs: object) -> pd.DataFrame:
            calls.append(1)
            if len(calls) == fail_on:
                raise RuntimeError("infra interruption")
            return real_itemize(*args, **kwargs)  # type: ignore[arg-type]

        return wrapper

    monkeypatch.setattr(itemizer_core, "itemize_documents", count(fail_on=4))
    with pytest.raises(RuntimeError, match="infra interruption"):
        itemize_pending_documents(artifact_root=tmp_path, batch_size=1)

    # Three partitions finished; their completion is durable, in two shards.
    completed = load_completed_partitions("itemize", artifact_root=tmp_path)
    assert completed == set(paths[:3])
    written = sorted(
        path.name for path in (tmp_path / "runs" / "itemize" / "completed").iterdir()
    )
    assert written == ["date=2024-01.json", "date=2024-02.json"]

    # And the resumed run pays only for what was left.
    monkeypatch.setattr(itemizer_core, "itemize_documents", count(fail_on=None))
    calls.clear()
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    assert len(calls) == 2
    assert load_completed_partitions("itemize", artifact_root=tmp_path) == set(paths)


def test_grown_document_partition_reitemizes_and_reclassifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Late-arriving documents merged into a partition flow downstream (#62)."""

    def _doc(accession: str) -> dict[str, object]:
        return {
            "accession_number": accession,
            "cik": "320193",
            "company_name": "Example Inc.",
            "url": "https://sec.example/full.txt",
            "text": (
                "\nITEM INFORMATION: Other Events\n<DOCUMENT>\n<TYPE>8-K\n<TEXT>\n"
                "Item 8.01 Other Events.\nThis is the extracted event text.\n"
                "</TEXT>\n</DOCUMENT>\n"
            ),
            "date": "2024-01-02",
            "resource_uri": None,
        }

    def _write_documents(accessions: list[str]) -> None:
        write_partition_table(
            str(tmp_path / "documents"),
            partition={"date": "2024-01-02", "shard": "0001"},
            table=pd.DataFrame([_doc(a) for a in accessions], columns=DOCUMENT_COLUMNS),
        )

    class SizedFakeModel:
        def decision_function(self: SizedFakeModel, texts: list[str]) -> list[float]:
            return [2.0] * len(texts)

    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (SizedFakeModel(), 0.5, {"threshold": 0.5}),
    )

    _write_documents(["000114036126006577"])
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    classify_pending_items(artifact_root=tmp_path, batch_size=5)
    assert len(read_dataset(classifications_root(tmp_path))) == 1

    # Ingest-style merge: the same partition object grows a second filing.
    _write_documents(["000114036126006577", "000114036126009999"])
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

    classified = read_dataset(classifications_root(tmp_path))
    assert sorted(classified["accession_number"].astype(str).unique()) == [
        "000114036126006577",
        "000114036126009999",
    ]


def test_classifier_loads_model_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pickled model is deserialized once, not once per partition (#76)."""
    seed_document_partitions(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=1)
    loads = 0

    def counting_load(path: object) -> tuple[FakeModel, float, dict[str, float]]:
        nonlocal loads
        del path
        loads += 1
        return (FakeModel(), 0.5, {"threshold": 0.5})

    monkeypatch.setattr(classifier_core, "load_training_artifacts", counting_load)

    classified = classify_pending_items(artifact_root=tmp_path, batch_size=1)

    assert len(classified) == 2
    assert loads == 1


def test_a_partial_row_publishes_its_mentions_and_registers_the_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#152's contract, end to end rather than on a row-state object.

    `PARTIAL` appeared only in assertions against `ExtractionRowState`, so the
    pipeline wiring was untested: counting PARTIAL as a success (dropping its
    registry entry) or removing it from `PUBLISHABLE_ROW_STATES` (dropping its
    mentions) both left the suite green.
    """
    seed_document_partition(tmp_path)
    itemize_pending_documents(artifact_root=tmp_path, batch_size=5)
    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )
    classify_pending_items(artifact_root=tmp_path, batch_size=5)

    async def salvaged_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            build_mention_row(
                mention_id="m-salvaged",
                item_id=str(item_row["item_id"]),
                accession_number=str(item_row["accession_number"]),
                cik=str(item_row["cik"]),
                date=str(item_row["date"]),
                name="Term Loan",
                start_date="2025-03-17",
                amount="500000000",
            )
        ]
        row_state.salvage_notes.append(
            "instrument_ie kept the valid entries and dropped 2 "
            "invalid ones after 3 failed attempts"
        )
        # Salvage finishes the row SUCCESS; the notes coerce it to PARTIAL.
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.extractor.live.run_extraction_workflow", salvaged_workflow)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # Half one: the salvaged mentions publish, exactly like a SUCCESS row.
    written = read_dataset(mentions_root(tmp_path))
    assert written["debt_instrument_mention_id"].to_list() == ["m-salvaged"]

    # Half two: the registry still records what was lost.
    failures = load_row_failures("extract", artifact_root=tmp_path)
    assert len(failures) == 1
    entry = next(iter(failures.values()))
    assert entry["state"] == "PARTIAL"
    assert entry["stage"] == "instrument_ie"
    assert "dropped 2" in str(entry["error"])
