"""File-native stage tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from cdt import storage as cdt_storage
from cdt.classifier import classifications_root, classify_pending_items
from cdt.classifier import core as classifier_core
from cdt.datasets import (
    completion_registry_path,
    existing_date_shard_partition_ids,
    load_completed_partitions,
    load_row_failures,
    normalize_cik,
    shard_for_accession,
    shard_for_cik,
)
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.core import (
    DEBT_INSTRUMENT_MENTION_COLUMNS,
    INSTRUMENT_RELATION_TYPES,
    CompletionResult,
    ExtractionRowState,
    InstrumentIEStage,
    InstrumentRelationStage,
    NERStage,
    canonical_amount_value,
    canonical_instrument_name,
    completion_result_from_batch_line,
    completion_result_from_response,
    currency_candidates_from_text,
    currency_from_name,
    date_plus_tenor,
    dates_agree,
    is_rate_like_amount_text,
    load_prompt,
    name_derived_principal_payload,
    normalized_amount_from_name,
    normalized_amount_from_text,
    normalized_date_from_text,
    normalized_maturity_from_text,
    normalized_month_year_from_text,
    oriented_lineage_pair,
    parse_tag_details,
    realign_tag_details,
    repair_unescaped_ampersands,
    validate_amount_is_not_rate,
    validate_dates_property,
    validate_interest_rate,
    validate_parties_property,
)
from cdt.ingest import DOCUMENT_COLUMNS
from cdt.itemizer import core as itemizer_core
from cdt.itemizer import itemize_pending_documents, items_root
from cdt.matcher import (
    debt_instruments_root,
    match_pending_mentions,
    mention_matches_root,
)
from cdt.matcher.core import (
    DEBT_INSTRUMENT_COLUMNS,
    MATCHER_SCHEMA_VERSION,
    MENTION_CLUSTER_EDGE_COLUMNS,
    apply_lineage_inference_pass,
    coerce_optional_text,
    company_names_by_cik,
    lender_signature,
    match_tables,
)
from cdt.pipeline import normalize_snapshot_text
from cdt.storage import (
    artifact_exists,
    coerce_dataset_text,
    get_object_bytes,
    read_dataset,
    read_json_artifact,
    read_table,
    write_partition_table,
)


class FakeModel:
    """Classifier stub returning a fixed relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a single strong-positive score."""
        del texts
        return [2.0]


def test_shard_for_accession_uses_eight_date_shards() -> None:
    """Date-partitioned stages should only use shards 0000 through 0007."""
    shards = {shard_for_accession(str(index)) for index in range(200)}
    assert shards == {f"{index:04d}" for index in range(8)}


def test_normalize_cik_zero_pads_digits_and_leaves_junk_visible() -> None:
    """CIKs publish as SEC's canonical 10-digit form (#153)."""
    assert normalize_cik("320193") == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"
    assert normalize_cik(320193) == "0000320193"
    assert normalize_cik(" not-a-cik ") == "not-a-cik"


def test_shard_for_cik_is_stable_across_padding() -> None:
    """Pre-#153 partitions hashed unpadded CIKs; padding must not re-shard."""
    assert shard_for_cik("0000320193") == shard_for_cik("320193")
    assert shard_for_cik("0") == shard_for_cik("0000000000")


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


def build_mention_row(
    *,
    mention_id: str,
    item_id: str,
    accession_number: str,
    cik: str,
    date: str,
    name: str,
    start_date: str,
    amount: str,
    parties_json: str = "[]",
    lender_disclosure: str = "complete",
    company_name: str | None = "Example Inc.",
    **overrides: object,
) -> dict[str, object]:
    """Return one canonical mention row for matcher tests.

    Built from `DEBT_INSTRUMENT_MENTION_COLUMNS` so every published column is
    present. Listing only the columns a test happened to need let a renamed or
    dropped column pass unnoticed: `prepare_mention` reads them all with
    `row.get`, so an absent one silently became None.
    """
    row: dict[str, object] = dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS)
    row.update(
        {
            "debt_instrument_mention_id": mention_id,
            "item_id": item_id,
            "accession_number": accession_number,
            "cik": cik,
            "company_name": company_name,
            "date": date,
            "raw_id": "i-1",
            "name": name,
            "start_date": start_date,
            "principal_amount": amount,
            "retired_by_json": "[]",
            "parties_json": parties_json,
            "lender_disclosure": lender_disclosure,
            "name_json": "{}",
            "start_date_json": "{}",
            "maturity_date_json": "{}",
            "amounts_json": "[]",
            "dates_json": "[]",
        }
    )
    unknown = set(overrides) - set(DEBT_INSTRUMENT_MENTION_COLUMNS)
    assert not unknown, f"not published mention columns: {sorted(unknown)}"
    row.update(overrides)
    return row


def test_coerce_dataset_text_treats_placeholder_values_as_missing() -> None:
    """Parquet placeholders must never survive as real text values."""
    assert coerce_dataset_text(float("nan")) is None
    assert coerce_dataset_text(None) is None
    assert coerce_dataset_text("nan") is None
    assert coerce_dataset_text("N/A") is None
    assert coerce_dataset_text("  ") is None
    assert (
        coerce_dataset_text(" Appreciate Holdings, Inc. ")
        == "Appreciate Holdings, Inc."
    )
    assert coerce_dataset_text("Nantucket Bank") == "Nantucket Bank"


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


def test_normalize_snapshot_text_nulls_placeholder_strings() -> None:
    """Dashboard-facing snapshots must not carry literal placeholder text."""
    table = pd.DataFrame(
        [
            {"company_name": "nan", "name": "convertible debentures", "amount": 1.5},
            {"company_name": "Versigent PLC", "name": "None", "amount": 2.5},
        ]
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["company_name"].to_list() == [None, "Versigent PLC"]
    assert normalized["name"].to_list() == ["convertible debentures", None]
    assert normalized["amount"].to_list() == [1.5, 2.5]


def test_normalize_snapshot_text_keeps_non_text_values_typed() -> None:
    """Booleans in an object column must not be published as text."""
    # Partitions written before lenders_known_incomplete existed leave an object
    # column holding booleans and nulls side by side.
    table = pd.DataFrame(
        {
            "lenders_known_incomplete": [True, None, False],
            "company_name": ["Acme Inc.", "nan", "Contoso Ltd."],
        }
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["lenders_known_incomplete"].to_list() == [True, None, False]
    assert normalized["company_name"].to_list() == [
        "Acme Inc.",
        None,
        "Contoso Ltd.",
    ]


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


class _TimingOutBody:
    """Body whose streaming read dies mid-stream."""

    def read(self: _TimingOutBody) -> bytes:
        raise ReadTimeoutError(endpoint_url="https://s3.test")


class _FlakyS3Client:
    """get_object succeeds; the body read times out a set number of times."""

    def __init__(self: _FlakyS3Client, payload: bytes, read_failures: int) -> None:
        self.payload = payload
        self.read_failures = read_failures
        self.get_object_calls = 0

    def get_object(self: _FlakyS3Client, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
        del Bucket, Key
        self.get_object_calls += 1
        if self.read_failures > 0:
            self.read_failures -= 1
            return {"Body": _TimingOutBody()}
        return {"Body": SimpleNamespace(read=lambda: self.payload)}


def test_get_object_bytes_retries_streaming_read_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-stream timeout must re-issue the GET, not kill the caller (#112).

    botocore's retry logic covers the get_object call, not the streaming body
    read; one such timeout previously ended a 2.5h itemize at partition
    13,121 of 18,113.
    """
    monkeypatch.setattr(cdt_storage, "sleep", lambda seconds: None)
    client = _FlakyS3Client(b"payload", read_failures=2)

    assert get_object_bytes(client, "bucket", "key") == b"payload"
    assert client.get_object_calls == 3


def test_get_object_bytes_gives_up_after_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent stream failure must surface, not retry forever (#112)."""
    monkeypatch.setattr(cdt_storage, "sleep", lambda seconds: None)
    client = _FlakyS3Client(b"payload", read_failures=99)

    with pytest.raises(ReadTimeoutError):
        get_object_bytes(client, "bucket", "key")

    assert client.get_object_calls == cdt_storage._GET_OBJECT_ATTEMPTS


def test_get_object_bytes_does_not_retry_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permanent errors (NoSuchKey, AccessDenied) must propagate immediately (#112)."""
    monkeypatch.setattr(cdt_storage, "sleep", lambda seconds: None)
    calls = 0

    class _MissingKeyClient:
        def get_object(
            self: _MissingKeyClient, Bucket: str, Key: str
        ) -> dict[str, object]:  # noqa: N803
            nonlocal calls
            del Bucket, Key
            calls += 1
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )

    with pytest.raises(ClientError):
        get_object_bytes(_MissingKeyClient(), "bucket", "key")

    assert calls == 1


def test_extract_pending_items_writes_mentions_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction should consume classifications and write mentions plus audit log."""
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
                "amounts_json": "[]",
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.core.run_extraction_workflow",
        fake_run_extraction_workflow,
    )

    mentions = extract_pending_items(
        artifact_root=tmp_path,
        batch_size=5,
        client=None,
    )

    written = read_dataset(mentions_root(tmp_path))
    assert len(mentions) == 1
    assert written["debt_instrument_mention_id"].to_list()
    audit_files = list((tmp_path / "extractor-runs").glob("run_id=*/full.jsonl"))
    assert len(audit_files) == 1


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
        "cdt.extractor.core.run_extraction_workflow",
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
        "cdt.extractor.core.run_extraction_workflow",
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


def test_instrument_ie_validate_allows_shared_evidence_and_skipped_collective_tags() -> (
    None
):
    """Shared evidence and skipped collective labels should validate successfully."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, the Company issued a
<debt_instrument id="tag-i-1">Senior Subordinated Convertible Promissory Note</debt_instrument>
(the <debt_instrument id="tag-i-2">Initial Exchange Note</debt_instrument>)
in an aggregate principal amount of <amount id="tag-a-1">$5.5 million</amount>
to <organization id="tag-o-1">EGT 11 LLC</organization>.
On <date id="tag-d-2">March 20, 2025</date>, the Company issued a
<debt_instrument id="tag-i-1b">Senior Subordinated Convertible Promissory Note</debt_instrument>
(the <debt_instrument id="tag-i-3">Subsequent Exchange Note</debt_instrument>)
in an aggregate principal amount of <amount id="tag-a-2">$269,000</amount>
to <organization id="tag-o-1b">EGT 11 LLC</organization>.
Together, the <debt_instrument id="tag-i-4">Exchange Notes</debt_instrument> were outstanding.
</body>
""".strip()
    response = """
[
  {
    "name": [
      "tag-i-1",
      "tag-i-2"
    ],
    "dates": [
      {
        "kind": "closing",
        "evidence": [
          "tag-d-1"
        ],
        "normalized_date": "2025-03-17"
      }
    ],
    "amounts": [
      {
        "evidence": [
          "tag-a-1"
        ],
        "normalized_amount": "5500000",
        "currency": "USD",
        "kind": "principal",
        "as_of_date": null
      }
    ],
    "parties": [
      {
        "tag_ids": [
          "tag-o-1"
        ],
        "role": "lender",
        "kind": "named"
      }
    ]
  },
  {
    "name": [
      "tag-i-1b",
      "tag-i-3"
    ],
    "dates": [
      {
        "kind": "closing",
        "evidence": [
          "tag-d-2"
        ],
        "normalized_date": "2025-03-20"
      }
    ],
    "amounts": [
      {
        "evidence": [
          "tag-a-2"
        ],
        "normalized_amount": "269000",
        "currency": "USD",
        "kind": "principal",
        "as_of_date": null
      }
    ],
    "parties": [
      {
        "tag_ids": [
          "tag-o-1b"
        ],
        "role": "lender",
        "kind": "named"
      }
    ]
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert failures == []


PARTY_ROLE_XML = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, <organization id="tag-o-borrower">Example Inc.</organization>
entered into a <debt_instrument id="tag-i-1">Term Loan</debt_instrument> with
<organization id="tag-o-named">JPMorgan Chase Bank, N.A.</organization> and
<organization id="tag-o-collective">the other lenders party thereto</organization>, with
<organization id="tag-o-agent">Wells Fargo Bank, National Association</organization> as administrative agent.
</body>
""".strip()


def party_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state seeded with party-role tagged XML."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    return row_state


def instrument_ie_mention(response: str) -> dict[str, object]:
    """Run instrument_ie postprocessing on one response and return its mention."""
    row_state = party_row_state()
    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert len(row_state.debt_instrument_mentions) == 1
    return row_state.debt_instrument_mentions[0]


def test_instrument_ie_validate_accepts_party_kinds_and_roles() -> None:
    """Annotated lender and other-party clusters should validate."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "parties": [
                    {"tag_ids": ["tag-o-named"], "role": "lender", "kind": "named"},
                    {
                        "tag_ids": ["tag-o-collective"],
                        "role": "lender",
                        "kind": "collective",
                    },
                    {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    {"tag_ids": ["tag-o-borrower"], "role": "borrower"},
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(party_row_state(), response) == []


def test_instrument_ie_validate_rejects_unannotated_party_clusters() -> None:
    """Bare tag-id lists and unknown annotations should fail validation."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "lenders": [["tag-o-named"]],
                "other_interested_parties": [
                    {"tag_ids": ["tag-o-agent"], "role": "servicer"}
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "'lenders' is not a property of this schema" in failure for failure in failures
    )
    assert any("'role' must be one of" in failure for failure in failures)


def test_instrument_ie_validate_rejects_non_boolean_lenders_known_incomplete() -> None:
    """The lenders_known_incomplete flag must be boolean when present."""
    response = json.dumps([{"name": ["tag-i-1"], "lenders_known_incomplete": "yes"}])

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "'lenders_known_incomplete' must be true or false" in failure
        for failure in failures
    )


def test_instrument_ie_postprocess_persists_every_party_with_role_and_kind() -> None:
    """Lenders, collective phrases, and other parties all persist labelled (#150)."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [
                        {"tag_ids": ["tag-o-named"], "kind": "named"},
                        {"tag_ids": ["tag-o-collective"], "kind": "collective"},
                    ],
                    "lenders_known_incomplete": True,
                    "other_interested_parties": [
                        {"tag_ids": ["tag-o-agent"], "role": "agent"}
                    ],
                }
            ]
        )
    )

    parties = json.loads(str(mention["parties_json"]))
    assert [
        (party["role"], party["kind"], [s["tag_id"] for s in party["spans"]])
        for party in parties
    ] == [
        ("lender", "named", ["tag-o-named"]),
        ("lender", "collective", ["tag-o-collective"]),
        ("agent", "named", ["tag-o-agent"]),
    ]
    assert all(
        set(party) == {"canonical_name", "role", "kind", "spans"} for party in parties
    )
    assert all(party["canonical_name"] for party in parties)
    assert mention["lender_disclosure"] == "collective_present"


def test_instrument_ie_postprocess_leaves_named_only_lenders_unflagged() -> None:
    """A lender list with only named clusters is not known to be incomplete."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [{"tag_ids": ["tag-o-named"], "kind": "named"}],
                }
            ]
        )
    )

    assert mention["lender_disclosure"] == "complete"
    assert len(json.loads(str(mention["parties_json"]))) == 1


def test_instrument_ie_postprocess_honors_declared_incompleteness() -> None:
    """A model-declared flag survives even when every cluster is named."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [{"tag_ids": ["tag-o-named"], "kind": "named"}],
                    "lenders_known_incomplete": True,
                }
            ]
        )
    )

    assert mention["lender_disclosure"] == "collective_present"
    assert len(json.loads(str(mention["parties_json"]))) == 1


def test_instrument_ie_postprocess_keeps_collective_lenders_and_flags() -> None:
    """A collective-only lender list persists with its surface text and flags (#150)."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "lenders": [
                        {"tag_ids": ["tag-o-collective"], "kind": "collective"}
                    ],
                }
            ]
        )
    )

    parties = json.loads(str(mention["parties_json"]))
    assert [(party["role"], party["kind"]) for party in parties] == [
        ("lender", "collective")
    ]
    assert mention["lender_disclosure"] == "collective_present"


def test_instrument_ie_postprocess_keeps_the_borrower_with_its_role() -> None:
    """The borrower persists with role "borrower" (#150).

    A subsidiary obligor under the parent filer's 8-K is exactly the identity
    worth recording.
    """
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "other_interested_parties": [
                        {"tag_ids": ["tag-o-borrower"], "role": "borrower"},
                        {"tag_ids": ["tag-o-agent"], "role": "agent"},
                    ],
                }
            ]
        )
    )

    parties = json.loads(str(mention["parties_json"]))
    assert [
        (party["role"], [s["tag_id"] for s in party["spans"]]) for party in parties
    ] == [
        ("borrower", ["tag-o-borrower"]),
        ("agent", ["tag-o-agent"]),
    ]


def test_lender_signature_prefers_the_named_party_over_an_alias() -> None:
    """A defined-term alias in the cluster must not hide the party it names."""
    payload = json.dumps([{"mentions": [{"text": "Purchasers"}, {"text": "Oaktree"}]}])

    assert lender_signature(payload) == "oaktree"


def test_lender_signature_uses_stored_lender_clusters() -> None:
    """Lender signatures come from the persisted named clusters."""
    payload = json.dumps([{"mentions": [{"text": "Acme Bank"}]}])

    assert lender_signature(payload) == "acme bank"


def test_match_pending_mentions_carries_lender_disclosure(tmp_path: Path) -> None:
    """Matcher output should carry mention-level lender incompleteness forward."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                parties_json=(
                    '[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]'
                ),
                lender_disclosure="collective_present",
            )
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows,
    )

    match_pending_mentions(artifact_root=tmp_path, batch_size=5)

    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert written_instruments["lender_disclosure"].to_list() == ["collective_present"]


def test_instrument_ie_validate_rejects_conflicting_start_dates() -> None:
    """One extracted object cannot carry two distinct start dates."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The <debt_instrument id="tag-i-1">Senior Subordinated Convertible Promissory Note</debt_instrument>
was issued on <date id="tag-d-1">March 17, 2025</date> and <date id="tag-d-2">March 20, 2025</date>.
</body>
""".strip()
    response = """
[
  {
    "name": [
      "tag-i-1"
    ],
    "dates": [
      {
        "kind": "closing",
        "evidence": [
          "tag-d-1",
          "tag-d-2"
        ],
        "normalized_date": "2025-03-17"
      }
    ]
  }
]
""".strip()

    failures = InstrumentIEStage().validate(row_state, response)

    assert any("multiple distinct normalized values" in failure for failure in failures)


MATURITY_IN_NAME_XML = """
<body>
On <date id="tag-d-1">March 17, 2025</date>, the Company issued
<debt_instrument id="tag-i-1">3.875% senior notes due 2028</debt_instrument>
in an aggregate principal amount of <amount id="tag-a-1">$500 million</amount>.
</body>
""".strip()


def maturity_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state whose instrument name carries a maturity."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = MATURITY_IN_NAME_XML
    return row_state


def maturity_mention(response: str) -> dict[str, object]:
    """Run instrument_ie postprocessing on one response and return its mention."""
    row_state = maturity_row_state()
    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert len(row_state.debt_instrument_mentions) == 1
    return row_state.debt_instrument_mentions[0]


def test_normalized_maturity_from_text_parses_due_phrases() -> None:
    """Maturity parsing should cover year-only, full-date, and ambiguous names."""
    assert normalized_maturity_from_text("3.875% senior notes due 2028") == "2028-12-31"
    assert (
        normalized_maturity_from_text("senior notes due October 1, 2028")
        == "2028-10-01"
    )
    assert normalized_maturity_from_text("notes due in 2030") == "2030-12-31"
    assert normalized_maturity_from_text("notes due 2028 and notes due 2031") is None
    assert normalized_maturity_from_text("3.875% senior notes") is None
    assert normalized_maturity_from_text("Series 2025-B Notes") is None


def test_normalized_maturity_from_text_rejects_coordinated_maturities() -> None:
    """One `due` listing two maturities identifies no single instrument (#104)."""
    assert normalized_maturity_from_text("notes due 2028 and 2030") is None
    assert normalized_maturity_from_text("notes due 2028, 2030") is None
    assert normalized_maturity_from_text("notes due 2028/2030") is None
    assert normalized_maturity_from_text("notes due October 1, 2028 and 2030") is None
    assert (
        normalized_maturity_from_text("notes due October 1, 2028 and October 1, 2030")
        is None
    )
    # A later year that is not coordinated onto the maturity is still not one.
    assert (
        normalized_maturity_from_text("notes due 2028, and 2030 obligations remain")
        == "2028-12-31"
    )


def test_instrument_ie_validate_accepts_name_span_as_end_date_evidence() -> None:
    """The instrument name span is valid end_date evidence when maturity is embedded."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "maturity",
                        "evidence": ["tag-i-1"],
                        "normalized_date": "2028-12-31",
                    }
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(maturity_row_state(), response) == []


def test_instrument_ie_validate_still_rejects_name_span_as_start_date_evidence() -> (
    None
):
    """Only end_date may cite a debt_instrument tag."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-i-1"],
                        "normalized_date": "2028-12-31",
                    }
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(maturity_row_state(), response)

    assert any("expected date" in failure for failure in failures)


def test_instrument_ie_postprocess_keeps_name_derived_end_date() -> None:
    """A maturity cited from the name span should survive normalization."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "maturity_date": {
                        "evidence": ["tag-i-1"],
                        "normalized_date": "2028-12-31",
                    },
                }
            ]
        )
    )

    assert mention["maturity_date"] == "2028-12-31"
    payload = json.loads(str(mention["maturity_date_json"]))
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-i-1"]


def test_instrument_ie_postprocess_backfills_end_date_from_name() -> None:
    """A missing end_date should fall back to the maturity in the instrument name."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "start_date": {
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2025-03-17",
                    },
                }
            ]
        )
    )

    assert mention["maturity_date"] == "2028-12-31"
    payload = json.loads(str(mention["maturity_date_json"]))
    # A maturity read from the name has no citable date tag of its own.
    assert payload["spans"] == []


def test_instrument_ie_postprocess_leaves_end_date_null_without_maturity() -> None:
    """Names without a maturity phrase should not gain an invented end date."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The Company issued <debt_instrument id="tag-i-1">3.875% senior notes</debt_instrument>
on <date id="tag-d-1">March 17, 2025</date>.
</body>
""".strip()
    row_state.stage_responses["instrument_ie"] = json.dumps([{"name": ["tag-i-1"]}])

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] is None
    assert json.loads(str(mention["maturity_date_json"]))["normalized_date"] is None


def test_instrument_ie_postprocess_drops_end_date_that_contradicts_evidence() -> None:
    """A normalized date that does not match its evidence text is still dropped."""
    mention = maturity_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "maturity_date": {
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2028-12-31",
                    },
                }
            ]
        )
    )

    payload = json.loads(str(mention["maturity_date_json"]))
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-d-1"]
    # The cited date tag says March 17, 2025, so the model value is rejected and the
    # name maturity fills the gap instead.
    assert mention["maturity_date"] == "2028-12-31"


RATE_AMOUNT_XML = """
<body>
<debt_instrument id="tag-i-1">ABR Loan</debt_instrument> borrowings bear interest at
<amount id="tag-a-rate">0.875% per annum</amount> and the facility provides for
<amount id="tag-a-principal">$500.0 million</amount> of commitments.
</body>
""".strip()


def rate_amount_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state with both a rate and a principal amount."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_AMOUNT_XML
    return row_state


def test_is_rate_like_amount_text_separates_rates_from_principal() -> None:
    """Rate detection should key on percentages without currency or scale words."""
    assert is_rate_like_amount_text("0.875% per annum") is True
    assert is_rate_like_amount_text("5.75%") is True
    assert is_rate_like_amount_text("175 basis points") is True
    assert is_rate_like_amount_text("$500.0 million") is False
    assert is_rate_like_amount_text("$500,000,000 (100% of principal)") is False
    assert is_rate_like_amount_text("100% of the outstanding 30.0 million") is False
    assert is_rate_like_amount_text(None) is False


def test_is_rate_like_amount_text_requires_every_number_to_carry_a_rate() -> None:
    """A percentage of a stated principal is not itself a rate (#103).

    The currency symbol and the scale word are not what makes these principals;
    `normalized_amount_from_text` reads the first number, so a span whose first
    number carries no rate marker is stating an amount.
    """
    assert is_rate_like_amount_text("500,000,000 (100% of principal)") is False
    assert (
        is_rate_like_amount_text("500 million U.S. dollars, or 5% of assets") is False
    )
    assert is_rate_like_amount_text("1,500,000") is False
    # A margin range is still every-number-rated.
    assert is_rate_like_amount_text("0.875% to 1.875%") is True
    assert is_rate_like_amount_text("SOFR plus 100 basis points") is True


def test_instrument_ie_validate_rejects_rate_only_amount_evidence() -> None:
    """An amount citing only a rate should fail validation and retry."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-rate"],
                    "normalized_amount": "0.875",
                    "currency": None,
                },
            }
        ]
    )

    failures = InstrumentIEStage().validate(rate_amount_row_state(), response)

    assert any(
        "describes an interest rate, margin, or fee" in failure for failure in failures
    )


def test_instrument_ie_validate_rejects_wrapped_basis_point_evidence() -> None:
    """Basis-point evidence is rejected, and the retry quotes readable text (#102).

    The evidence span wraps across a line, as filings do. Judging whitespace-free
    text hid the `basis point` marker from the predicate entirely, so the retry
    never fired and the amount was silently nulled instead.
    """
    tag_details = {"tag-a-bps": {"type": "amount", "text": "100 basis\npoints"}}
    failures = validate_amount_is_not_rate(
        index=0,
        value={"evidence": ["tag-a-bps"]},
        tag_details=tag_details,
    )

    assert len(failures) == 1
    assert "'100 basis points'" in failures[0]
    assert "describes an interest rate, margin, or fee" in failures[0]


def test_instrument_ie_validate_accepts_spelled_currency_principal() -> None:
    """A principal whose currency and scale are words, not symbols, validates (#102)."""
    tag_details = {
        "tag-a-spelled": {
            "type": "amount",
            "text": "500 million U.S. dollars, or 5% of assets",
        }
    }

    assert (
        validate_amount_is_not_rate(
            index=0,
            value={"evidence": ["tag-a-spelled"]},
            tag_details=tag_details,
        )
        == []
    )


def test_instrument_ie_validate_accepts_principal_amount_evidence() -> None:
    """A principal amount should still validate."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-principal"],
                        "normalized_amount": "500000000",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(rate_amount_row_state(), response) == []


def test_instrument_ie_postprocess_drops_rate_amount() -> None:
    """A rate that slips past validation should not be stored as an amount."""
    row_state = rate_amount_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-rate"],
                    "normalized_amount": "0.875",
                    "currency": "USD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert mention["principal_amount"] is None
    assert payload["normalized_amount"] is None
    assert payload["currency"] is None
    # Evidence is preserved so the dropped value stays auditable.
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-a-rate"]


def test_lineage_pair_is_oriented_by_relation_type() -> None:
    """Each lineage type has its own side that must hold the later date (#138, #142).

    `amendment_of` runs from the amended instrument to the predecessor, so its
    source is the later of the two; `retired_by` runs from the retired
    obligation to the instrument that retired it, so its source is the earlier.
    Both directions of both types are asserted here: flipping unconditionally
    is as wrong as never flipping, and only one of the two used to be covered
    (#180).

    Alclear's revolver is the confirmed case: a Credit Agreement dated as of
    2020-03-31, amended 2026-06-23 to cut commitments from $100,000,000 and
    extend maturity from 2026-06-28 to 2031-06-23. Every figure on the
    `$100,000,000` object is pre-amendment, so it is the predecessor, and the
    pointer must run the other way.
    """
    by_raw_id = {
        "i-1": {"raw_id": "i-1", "start_date": "2020-03-31", "amount": "100000000"},
        "i-2": {"raw_id": "i-2", "start_date": "2026-06-23", "amount": "50000000"},
    }
    # The model named the predecessor first; the pointer is flipped.
    assert oriented_lineage_pair("i-1", "i-2", "amendment_of", by_raw_id) == (
        "i-2",
        "i-1",
    )
    # Already the right way round, so it is left alone.
    assert oriented_lineage_pair("i-2", "i-1", "amendment_of", by_raw_id) == (
        "i-2",
        "i-1",
    )
    # `retired_by` runs the other way: the retired obligation is the earlier
    # side, so a pair the model named retirer-first is flipped.
    assert oriented_lineage_pair("i-2", "i-1", "retired_by", by_raw_id) == (
        "i-1",
        "i-2",
    )
    # And a `retired_by` pair already named retired-first is left alone, which
    # is what an unconditional flip would break.
    assert oriented_lineage_pair("i-1", "i-2", "retired_by", by_raw_id) == (
        "i-1",
        "i-2",
    )


def test_lineage_pair_is_left_alone_without_two_dates_to_compare() -> None:
    """Nothing to orient on means nothing is changed (#138)."""
    same = {
        "i-1": {"raw_id": "i-1", "start_date": "2025-08-01"},
        "i-2": {"raw_id": "i-2", "start_date": "2025-08-01"},
    }
    # Equal dates carry no ordering, which is exactly the state the old prompt
    # convention produced, and why the direction went unchecked for so long.
    assert oriented_lineage_pair("i-1", "i-2", "amendment_of", same) == ("i-1", "i-2")
    # Equal dates are not a contradiction for `retired_by` either: a filing
    # that issues and redeems on the same day gives nothing to orient on.
    assert oriented_lineage_pair("i-1", "i-2", "retired_by", same) == ("i-1", "i-2")
    missing = {
        "i-1": {"raw_id": "i-1", "start_date": None},
        "i-2": {"raw_id": "i-2", "start_date": "2026-06-23"},
    }
    assert oriented_lineage_pair("i-1", "i-2", "amendment_of", missing) == (
        "i-1",
        "i-2",
    )
    assert oriented_lineage_pair("i-2", "i-1", "retired_by", missing) == (
        "i-2",
        "i-1",
    )
    # `split_of` is not a before/after relation, so it is never reoriented.
    ordered = {
        "i-1": {"raw_id": "i-1", "start_date": "2020-03-31"},
        "i-2": {"raw_id": "i-2", "start_date": "2026-06-23"},
    }
    assert oriented_lineage_pair("i-1", "i-2", "split_of", ordered) == ("i-1", "i-2")


def test_relation_postprocess_flips_an_inverted_amendment() -> None:
    """The published mention carries the corrected direction (#138)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": "x"}, stage_name="instrument_relation"
    )
    row_state.debt_instrument_mentions = [
        {
            "raw_id": "i-1",
            "debt_instrument_mention_id": "dim::pred",
            "start_date": "2020-03-31",
            "amendment_of": None,
        },
        {
            "raw_id": "i-2",
            "debt_instrument_mention_id": "dim::amended",
            "start_date": "2026-06-23",
            "amendment_of": None,
        },
    ]
    row_state.stage_responses["instrument_relation"] = json.dumps(
        [{"from": "i-1", "to": "i-2", "type": "amendment_of"}]
    )

    InstrumentRelationStage().postprocess(row_state)

    by_raw = {m["raw_id"]: m for m in row_state.debt_instrument_mentions}
    assert by_raw["i-2"]["amendment_of"] == "dim::pred"
    assert by_raw["i-1"]["amendment_of"] is None


def test_completion_result_captures_a_filtered_live_response() -> None:
    """A provider abort is recorded, not just its partial text (#135).

    This is the shape observed live: `content_filter`, partial content, and a
    zeroed unbilled usage block. Without `finish_reason` the audit log cannot
    tell it from a model that chose to stop, and the two need opposite remedies.
    """
    usage = SimpleNamespace(
        completion_tokens=0, prompt_tokens=0, total_tokens=0, cost=0.0
    )
    message = SimpleNamespace(content="<body>partial", refusal=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="content_filter", message=message)],
        usage=usage,
        id="gen-1788206728-iWdiE2p9PV0",
        model="openai/gpt-5.6-terra",
    )

    result = completion_result_from_response(response)

    assert result.text == "<body>partial"
    assert result.finish_reason == "content_filter"
    assert result.response_id == "gen-1788206728-iWdiE2p9PV0"
    assert result.served_model == "openai/gpt-5.6-terra"
    assert result.usage == {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def test_completion_result_captures_a_filtered_batch_line() -> None:
    """The batch route carries the same fields on the same path (#135).

    A filtered batch response arrives as a normal 200 with a body, so it never
    reaches the infrastructure-error path and would otherwise be indistinguishable
    from ordinary bad output.
    """
    line = {
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-abc",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": "<body>partial", "refusal": None},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 0,
                    "total_tokens": 11,
                },
            },
        }
    }

    result = completion_result_from_batch_line(line)

    assert result.text == "<body>partial"
    assert result.finish_reason == "content_filter"
    assert result.response_id == "chatcmpl-abc"
    assert result.served_model == "gpt-5.4"
    assert result.usage["total_tokens"] == 11


def test_attempt_records_provider_metadata() -> None:
    """The metadata reaches the attempt, and so `full.jsonl` (#135)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": "x"}, stage_name="ner"
    )
    row_state.add_response(
        "<body>x</body>",
        CompletionResult(
            text="<body>x</body>",
            finish_reason="length",
            usage={"total_tokens": 42},
            response_id="gen-9",
            served_model="openai/gpt-5.4",
        ),
    )

    recorded = row_state.current_attempt.to_dict()
    assert recorded["finish_reason"] == "length"
    assert recorded["usage"] == {"total_tokens": 42}
    assert recorded["response_id"] == "gen-9"
    assert recorded["served_model"] == "openai/gpt-5.4"
    # An attempt recorded without provider metadata stays null rather than absent.
    other = ExtractionRowState(item_row={"item_id": "i", "text": "x"}, stage_name="ner")
    other.add_response("<body>x</body>")
    assert other.current_attempt.to_dict()["finish_reason"] is None


def test_repair_unescaped_ampersands_leaves_real_entities_alone() -> None:
    """A bare ampersand is escaped; anything already an entity is untouched (#127)."""
    assert repair_unescaped_ampersands("A&R Agreement") == "A&amp;R Agreement"
    assert repair_unescaped_ampersands("Smith & Wesson & Co") == (
        "Smith &amp; Wesson &amp; Co"
    )
    for already_valid in (
        "A&amp;R",
        "a &lt; b",
        "a &gt; b",
        "&quot;x&quot;",
        "&apos;x&apos;",
        "&#8217;s",
        "&#x2019;s",
    ):
        assert repair_unescaped_ampersands(already_valid) == already_valid


def test_ner_validate_accepts_a_response_carrying_a_bare_ampersand() -> None:
    """The exact-text and well-formed-XML requirements conflict without this (#127).

    `NERStage.preprocess` wraps the item text in `<body>` unescaped, so an item
    naming an `A&R Registration Rights Agreement` is handed to the model as
    invalid XML. Reproducing it verbatim then yields invalid XML, and 44 of 342
    relevant items in one held-out window carry a bare ampersand.
    """
    text = "The Company entered into the A&R Registration Rights Agreement."
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": text}, stage_name="ner"
    )
    response = (
        "<body>The Company entered into the <agreement>A&R Registration Rights "
        "Agreement</agreement>.</body>"
    )

    assert NERStage().validate(row_state, response) == []

    row_state.stage_responses["ner"] = response
    NERStage().postprocess(row_state)
    _, plain_text, tag_details = parse_tag_details(str(row_state.ner_tagged_xml))
    # The ampersand round-trips, so the text invariant still holds downstream.
    assert plain_text == text
    assert [d["text"] for d in tag_details.values()] == [
        "A&R Registration Rights Agreement"
    ]


def test_ner_validate_still_rejects_a_stray_angle_bracket() -> None:
    """Only `&` is repaired; a malformed tag is still a failure (#127)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": "The Company borrowed."},
        stage_name="ner",
    )
    failures = NERStage().validate(row_state, "<body>The Company <borrowed.</body>")
    assert failures and "not valid XML" in failures[0]


def test_normalized_date_from_text_reads_every_filing_spelling() -> None:
    """A format the parser cannot read discards a date the model got right (#133).

    `M/D/YYYY` alone cost 67 start dates and 67 end dates on one held-out
    window, all from tabular schedules whose rows the model had parsed
    correctly.
    """
    assert normalized_date_from_text("7/28/2026") == "2026-07-28"
    assert normalized_date_from_text("07/28/2026") == "2026-07-28"
    assert normalized_date_from_text("12/31/2030") == "2030-12-31"
    # The comma is optional, and non-US issuers write the day first.
    assert normalized_date_from_text("July 28 2026") == "2026-07-28"
    assert normalized_date_from_text("28 July 2026") == "2026-07-28"
    assert normalized_date_from_text("July 28, 2026") == "2026-07-28"
    # Zero-padding is optional on the way in.
    assert normalized_date_from_text("2026-7-8") == "2026-07-08"
    # A two-digit year needs a century guessed, so it stays unparsed.
    assert normalized_date_from_text("7/28/26") is None
    # Impossible dates and non-dates stay unparsed.
    assert normalized_date_from_text("13/28/2026") is None
    assert normalized_date_from_text("2/30/2026") is None
    assert normalized_date_from_text("Closing Date") is None
    assert normalized_date_from_text("August 2056") is None
    assert normalized_date_from_text("Section 8 2026") is None
    assert normalized_date_from_text("6.875% Senior Notes due March 2027") is None


def test_dates_agree_ignores_shape_but_not_value() -> None:
    """A model writing the same day differently keeps its date (#133)."""
    assert dates_agree("2026-7-28", "2026-07-28") is True
    assert dates_agree("2026-07-28", "2026-07-28") is True
    assert dates_agree("2026-07-29", "2026-07-28") is False
    assert dates_agree("soon", "2026-07-28") is False
    assert dates_agree(None, "2026-07-28") is False


def test_instrument_ie_postprocess_keeps_a_slash_format_date() -> None:
    """A tabular schedule row publishes its dates (#133)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>Schedule A lists <debt_instrument id="tag-i-1">Consolidated '
        'Obligation Bonds</debt_instrument> settling <date id="tag-d-1">7/28/2026'
        '</date> and maturing <date id="tag-d-2">7/28/2028</date>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "start_date": {
                    "evidence": ["tag-d-1"],
                    "normalized_date": "2026-07-28",
                },
                "maturity_date": {
                    "evidence": ["tag-d-2"],
                    "normalized_date": "2028-07-28",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["start_date"] == "2026-07-28"
    assert mention["maturity_date"] == "2028-07-28"


def test_normalized_amount_from_name_reads_an_embedded_principal() -> None:
    """A principal inside the name parses; a coupon or maturity does not (#129).

    NER tags `$183.36 million term loan` as one `debt_instrument`, so there is no
    `amount` span to cite. Requiring a currency marker is what keeps the coupon
    rate and the maturity year from being read as the principal.
    """
    assert normalized_amount_from_name("$183.36 million term loan") == "183360000"
    assert normalized_amount_from_name("C$300 million notes due 2033") == "300000000"
    assert normalized_amount_from_name("$1,299,870.00 Promissory Note") == "1299870"
    assert currency_from_name("C$300 million notes due 2033") == "CAD"
    assert currency_from_name("€600.0 million 3.625% Notes due 2032") == "EUR"
    # No currency marker means no principal is stated in the name.
    assert normalized_amount_from_name("3.875% senior notes due 2028") is None
    assert normalized_amount_from_name("revolving credit facility") is None
    # A name stating two figures names no single principal, as #104 requires for
    # maturities.
    assert (
        normalized_amount_from_name("$500 million and $750 million facilities") is None
    )


def test_instrument_ie_postprocess_recovers_a_principal_from_the_name() -> None:
    """An uncited principal inside the name still publishes (#129)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The Company prepaid its <debt_instrument id="tag-i-1">$183.36 million '
        "term loan</debt_instrument>.</body>"
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                # No `amount` span exists, so the model cites nothing.
                "amount": {
                    "evidence": [],
                    "normalized_amount": "183360000",
                    "currency": "USD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert mention["principal_amount"] == "183360000"
    assert payload["currency"] == "USD"
    # Nothing was cited, so the evidence list stays empty, as it does for a
    # name-derived maturity.
    assert payload["spans"] == []


def test_instrument_ie_validate_accepts_the_name_span_as_amount_evidence() -> None:
    """Citing the instrument's own name span for a name-embedded amount is valid (#129)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">$183.36 million term loan'
        "</debt_instrument> was prepaid.</body>"
    )
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-i-1"],
                        "normalized_amount": "183360000",
                        "currency": "USD",
                    }
                ],
            }
        ]
    )

    assert InstrumentIEStage().validate(row_state, response) == []

    row_state.stage_responses["instrument_ie"] = response
    InstrumentIEStage().postprocess(row_state)
    assert row_state.debt_instrument_mentions[0]["principal_amount"] == "183360000"


def test_normalized_amount_from_text_keeps_cents_exact() -> None:
    """A cents value parses to itself, not to a float artifact (#119).

    `float("372246148.11")` is not that number, and the old `f"{value:.12f}"`
    rendering exposed the difference, so the string never matched what the model
    reported and the amount published as null.
    """
    assert normalized_amount_from_text("$372,246,148.11") == "372246148.11"
    assert normalized_amount_from_text("$55,637.41") == "55637.41"
    assert normalized_amount_from_text("$5,529,722.96") == "5529722.96"
    # Trailing zeros and scale words still collapse to one canonical form.
    assert normalized_amount_from_text("$500,000.00") == "500000"
    assert normalized_amount_from_text("$70.0 million") == "70000000"
    assert normalized_amount_from_text("1.5 billion") == "1500000000"


def test_instrument_ie_postprocess_keeps_an_amount_with_cents() -> None:
    """A principal with cents survives the model/parser agreement gate (#119)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">construction loan facility'
        "</debt_instrument> provides for "
        '<amount id="tag-a-1">$372,246,148.11</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-1"],
                    # The model reports the value it read, without the float
                    # artifact the parser used to produce.
                    "normalized_amount": "372246148.11",
                    "currency": "USD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert mention["principal_amount"] == "372246148.11"
    assert payload["normalized_amount"] == "372246148.11"
    assert payload["currency"] == "USD"


def test_instrument_ie_postprocess_accepts_a_differently_formatted_amount() -> None:
    """`500000.00` and `500000` are the same amount, so neither is lost (#119)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">promissory note'
        '</debt_instrument> is for <amount id="tag-a-1">$500,000.00</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-1"],
                    "normalized_amount": "500000.00",
                    "currency": "USD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    # The parser's canonical string is what gets published.
    assert mention["principal_amount"] == "500000"
    # A genuinely different value is still rejected.
    assert json.loads(str(mention["amounts_json"]))[0]["normalized_amount"] == "500000"


def test_canonical_amount_value_prefers_the_span_that_parses() -> None:
    """A longer label must not beat the figure it names (#120)."""
    tag_details = {
        "tag-a-1": {"type": "amount", "text": "$2,000,000"},
        "tag-a-2": {"type": "amount", "text": "Principal Amount"},
    }

    assert canonical_amount_value(["tag-a-1", "tag-a-2"], tag_details) == "$2,000,000"
    # With nothing parseable, the longest span is still the canonical text.
    assert canonical_amount_value(["tag-a-2"], tag_details) == "Principal Amount"
    # Among parseable spans the longest still wins, as it did before.
    tag_details["tag-a-3"] = {"type": "amount", "text": "$2,000,000 in principal"}
    assert (
        canonical_amount_value(["tag-a-1", "tag-a-3"], tag_details)
        == "$2,000,000 in principal"
    )


def test_instrument_ie_postprocess_keeps_an_amount_clustered_with_its_label() -> None:
    """The figure survives being clustered with a longer label span (#120)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">Secured Convertible Promissory Note'
        '</debt_instrument> has a <amount id="tag-a-label">Principal Amount</amount> '
        'of <amount id="tag-a-figure">$2,000,000</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-figure", "tag-a-label"],
                    "normalized_amount": "2000000",
                    "currency": "USD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] == "2000000"
    # Both spans stay in the payload as provenance.
    payload = json.loads(str(mention["amounts_json"]))[0]
    assert [s["tag_id"] for s in payload["spans"]] == ["tag-a-figure", "tag-a-label"]


def test_currency_candidates_read_a_qualified_dollar_sign() -> None:
    """`C$` is Canadian, not US, dollars (#121)."""
    assert currency_candidates_from_text("C$300 million") == {"CAD"}
    assert currency_candidates_from_text("A$50,000,000") == {"AUD"}
    assert currency_candidates_from_text("NZ$10 million") == {"NZD"}
    # An unqualified dollar sign keeps its USD reading.
    assert currency_candidates_from_text("$500.0 million") == {"USD"}
    assert currency_candidates_from_text("500 million U.S. dollars") == {"USD"}
    # A span quoting both currencies offers both.
    assert currency_candidates_from_text("C$300 million (US$220 million)") == {
        "CAD",
        "USD",
    }


def test_instrument_ie_postprocess_keeps_a_canadian_dollar_currency() -> None:
    """A C$ principal publishes CAD rather than a null currency (#121)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        '<body>The <debt_instrument id="tag-i-1">4.200% Senior Notes due 2033'
        '</debt_instrument> total <amount id="tag-a-1">C$300 million</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-1"],
                    "normalized_amount": "300000000",
                    "currency": "CAD",
                },
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    payload = json.loads(str(row_state.debt_instrument_mentions[0]["amounts_json"]))[0]
    assert payload["normalized_amount"] == "300000000"
    assert payload["currency"] == "CAD"


def test_instrument_ie_postprocess_normalizes_wrapped_names() -> None:
    """A name wrapped across lines is stored collapsed, verbatim in name_json."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        "<body>The Company entered into a "
        '<debt_instrument id="tag-i-1">revolving credit\nfacility</debt_instrument> '
        "and issued "
        '<debt_instrument id="tag-i-2">4.85% Remarketable\tSenior\xa0Notes '
        "due 2032</debt_instrument>.</body>"
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"]}, {"name": ["tag-i-2"]}]
    )

    InstrumentIEStage().postprocess(row_state)

    names = [mention["name"] for mention in row_state.debt_instrument_mentions]
    assert names == [
        "revolving credit facility",
        "4.85% Remarketable Senior Notes due 2032",
    ]
    # Provenance keeps the verbatim span, since the char offsets index into it.
    verbatim = [
        json.loads(str(mention["name_json"]))["spans"][0]["text"]
        for mention in row_state.debt_instrument_mentions
    ]
    assert verbatim == [
        "revolving credit\nfacility",
        "4.85% Remarketable\tSenior\xa0Notes due 2032",
    ]


def test_instrument_ie_postprocess_drops_duplicate_identical_mentions() -> None:
    """Objects that differ in no extracted property must not duplicate a mention row."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = """
<body>
The trust issued
<debt_instrument id="tag-i-1">Class A-1 Notes, Class A-2 Notes, and Class A-3 Notes</debt_instrument>.
</body>
""".strip()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"]}, {"name": ["tag-i-1"]}, {"name": ["tag-i-1"]}]
    )

    InstrumentIEStage().postprocess(row_state)

    mentions = row_state.debt_instrument_mentions
    assert len(mentions) == 1
    assert mentions[0]["raw_id"] == "i-1"


def test_instrument_ie_prompt_requires_one_object_per_class() -> None:
    """The IE prompt must keep telling the model to split multi-class offerings."""
    prompt = load_prompt("instrument_ie")

    assert "Each class, tranche, or series of an offering" in prompt
    assert "Class A-1" in prompt


def test_instrument_relation_stage_accepts_retired_by() -> None:
    """Relation validation and postprocessing should support retired_by."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_relation",
    )
    row_state.debt_instrument_mentions = [
        {
            "debt_instrument_mention_id": "m-1",
            "raw_id": "i-1",
        },
        {
            "debt_instrument_mention_id": "m-2",
            "raw_id": "i-2",
        },
    ]
    response = '[{"from": "i-2", "to": "i-1", "type": "retired_by"}]'

    failures = InstrumentRelationStage().validate(row_state, response)
    assert failures == []

    row_state.stage_responses["instrument_relation"] = response
    InstrumentRelationStage().postprocess(row_state)

    assert row_state.debt_instrument_mentions[1]["retired_by_json"] == '["m-1"]'


def test_instrument_relation_stage_accumulates_every_retirer() -> None:
    """Several edges out of one obligation all survive on its mention (#180).

    Venture Global's two new series jointly redeem one earlier obligation, so
    the relation stage emits two `retired_by` edges sharing a `from`. A scalar
    column kept only the last one, which is the bug this guards: with one edge
    per test, an append and an overwrite look identical. The repeated edge
    covers the deduplication guard at the same time.
    """
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_relation",
    )
    row_state.debt_instrument_mentions = [
        {"debt_instrument_mention_id": "m-old", "raw_id": "i-old"},
        {"debt_instrument_mention_id": "m-2034", "raw_id": "i-2034"},
        {"debt_instrument_mention_id": "m-2036", "raw_id": "i-2036"},
    ]
    response = json.dumps(
        [
            {"from": "i-old", "to": "i-2034", "type": "retired_by"},
            {"from": "i-old", "to": "i-2036", "type": "retired_by"},
            {"from": "i-old", "to": "i-2034", "type": "retired_by"},
        ]
    )

    failures = InstrumentRelationStage().validate(row_state, response)
    assert failures == []

    row_state.stage_responses["instrument_relation"] = response
    InstrumentRelationStage().postprocess(row_state)

    assert (
        row_state.debt_instrument_mentions[0]["retired_by_json"]
        == '["m-2034", "m-2036"]'
    )


def test_instrument_relation_prompt_matches_the_accepted_relation_types() -> None:
    """The prompt's type vocabulary must track the validator's (#180).

    `validate` rejects any type outside `INSTRUMENT_RELATION_TYPES`, so a
    prompt naming a type the code no longer accepts fails every relation the
    model returns and loses all lineage silently -- with nothing else in the
    suite noticing.
    """
    prompt = load_prompt("instrument_relation")

    for relation_type in INSTRUMENT_RELATION_TYPES:
        assert f"`{relation_type}`" in prompt
    assert "retired_of" not in prompt


def test_match_pending_mentions_writes_match_datasets(tmp_path: Path) -> None:
    """Matcher should consume mention dataset and write match outputs."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                parties_json='[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]',
            )
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows,
    )

    tables = match_pending_mentions(artifact_root=tmp_path, batch_size=5)

    written_matches = read_dataset(mention_matches_root(tmp_path))
    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert len(tables["debt_instrument_mentions"]) == 1
    assert written_matches["edge_type"].to_list() == ["member"]
    assert written_instruments["debt_instrument_id"].to_list() == ["m-1"]


def test_company_names_by_cik_takes_the_newest_known_name() -> None:
    """CIK name resolution should ignore missing values and prefer newer filings."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="2078008",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                company_name=None,
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="2078008",
                date="2024-02-02",
                name="Revolver",
                start_date="2024-02-01",
                amount="$50 million",
                company_name="Versigent PLC",
            ),
            build_mention_row(
                mention_id="m-3",
                item_id="item-3",
                accession_number="0003",
                cik="320193",
                date="2024-03-02",
                name="Senior Notes",
                start_date="2024-03-01",
                amount="$1 billion",
            ),
        ]
    )

    # Keys are canonical zero-padded CIKs (#153).
    assert company_names_by_cik(mention_rows) == {
        "0002078008": "Versigent PLC",
        "0000320193": "Example Inc.",
    }


def test_match_tables_backfills_company_name_from_cik() -> None:
    """An instrument seeded by a mention without display metadata is still named."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="2078008",
                date="2024-01-02",
                name="6.125% senior unsecured notes due 2031",
                start_date="2024-01-01",
                amount="$400 million",
                company_name=None,
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="2078008",
                date="2024-02-02",
                name="Revolving Credit Facility",
                start_date="2024-02-01",
                amount="$50 million",
                company_name="Versigent PLC",
            ),
        ]
    )

    tables = match_tables(mention_rows)

    instruments = tables["debt_instrument"].set_index("debt_instrument_id")
    assert len(instruments) == 2
    assert instruments.loc["m-1", "company_name"] == "Versigent PLC"
    assert instruments.loc["m-2", "company_name"] == "Versigent PLC"


def test_coerce_optional_text_treats_nan_like_text_as_missing() -> None:
    """Literal placeholder strings must never reach a dashboard-facing column."""
    assert coerce_optional_text("nan") is None
    assert coerce_optional_text("NaN") is None
    assert coerce_optional_text("None") is None
    assert coerce_optional_text("N/A") is None
    assert coerce_optional_text("  ") is None
    assert coerce_optional_text("Nantucket Bank") == "Nantucket Bank"


def test_lineage_inference_pass_writes_pointers_and_rederives_the_rollup(
    tmp_path: Path,
) -> None:
    """The post-pass is the only production entry point, so cover it end to end.

    Before this existed, gutting the pointer write, deleting the rollup
    re-derive, or never writing partitions at all each left the suite green
    (#184).
    """
    rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2020-01-02",
                name="Credit Agreement",
                start_date="2020-01-01",
                amount="$100 million",
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2024-01-02",
                name="Second Amended and Restated Credit Agreement",
                start_date="2024-01-01",
                amount="$250 million",
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=rows,
    )
    match_pending_mentions(artifact_root=tmp_path, batch_size=5)
    stats = apply_lineage_inference_pass(str(tmp_path))

    published = {
        str(row["debt_instrument_id"]): row
        for row in read_dataset(debt_instruments_root(tmp_path)).to_dict("records")
    }
    assert stats == {"links": 1, "heads_before": 2, "heads_after": 1}
    child = published["m-2"]
    parent = published["m-1"]
    assert child["amendment_of_debt_instrument_id"] == "m-1"
    assert child["amendment_inferred_by"] == "ordinal_chain"
    assert child["is_lineage_head"]
    # the rollup must be re-derived from the new pointer, not left stale
    assert parent["is_lineage_head"] is False
    assert parent["superseded_by_debt_instrument_id"] == "m-2"
    assert parent["status"] == "closed"
    assert parent["status_subtype"] == "superseded"
    assert child["lineage_family_id"] == parent["lineage_family_id"]

    # An ordinary rematch keeps the pointer, so it must keep the provenance too:
    # a guess that reads as an extracted relation is worse than no guess (#184).
    match_pending_mentions(artifact_root=tmp_path, batch_size=5)
    after = {
        str(row["debt_instrument_id"]): row
        for row in read_dataset(debt_instruments_root(tmp_path)).to_dict("records")
    }
    assert after["m-2"]["amendment_of_debt_instrument_id"] == "m-1"
    assert after["m-2"]["amendment_inferred_by"] == "ordinal_chain"
    assert after["m-1"]["status"] == "closed"

    # --force drops both together: no pointer, no stale provenance.
    match_pending_mentions(artifact_root=tmp_path, batch_size=5, force=True)
    forced = {
        str(row["debt_instrument_id"]): row
        for row in read_dataset(debt_instruments_root(tmp_path)).to_dict("records")
    }
    assert forced["m-2"]["amendment_of_debt_instrument_id"] is None
    assert pd.isna(forced["m-2"]["amendment_inferred_by"])


def test_match_pending_mentions_drains_all_shards(tmp_path: Path) -> None:
    """Matcher should process all shard groups across chunks."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Term Loan",
                start_date="2024-01-01",
                amount="$100 million",
                parties_json='[{"mentions": [{"text": "Acme Bank"}], "tag_ids": ["tag-l-1"]}]',
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="789019",
                date="2024-01-03",
                name="Revolving Credit Facility",
                start_date="2024-01-01",
                amount="$250 million",
                parties_json='[{"mentions": [{"text": "Contoso Bank"}], "tag_ids": ["tag-l-2"]}]',
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows.iloc[[0]],
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-03", "shard": "0002"},
        table=mention_rows.iloc[[1]],
    )

    tables = match_pending_mentions(artifact_root=tmp_path, batch_size=1)

    written_matches = read_dataset(mention_matches_root(tmp_path))
    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert len(tables["debt_instrument_mentions"]) == 2
    assert sorted(written_matches["debt_instrument_mention_id"].to_list()) == [
        "m-1",
        "m-2",
    ]
    assert sorted(written_instruments["debt_instrument_id"].to_list()) == [
        "m-1",
        "m-2",
    ]


def test_match_pending_mentions_force_rebuilds_existing_memberships(
    tmp_path: Path,
) -> None:
    """Force reruns should discard stale member assignments for the shard."""
    mention_rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="Alpha Loan",
                start_date="2024-01-01",
                amount="$100 million",
            ),
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2024-01-02",
                name="Beta Facility",
                start_date="2024-01-01",
                amount="$100 million",
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-01", "shard": "0001"},
        table=mention_rows.iloc[[0]],
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=mention_rows.iloc[[1]],
    )

    first = match_pending_mentions(
        artifact_root=tmp_path,
        batch_size=5,
        strong_match_threshold=0.75,
        loose_match_threshold=0.75,
    )
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in first["debt_instrument_mentions"]
        .query("edge_type == 'member'")
        .to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-1"}

    second = match_pending_mentions(
        artifact_root=tmp_path,
        batch_size=5,
        force=True,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    written_matches = read_dataset(mention_matches_root(tmp_path)).sort_values(
        ["debt_instrument_mention_id", "edge_type", "debt_instrument_id"]
    )
    written_instruments = read_dataset(debt_instruments_root(tmp_path)).sort_values(
        "debt_instrument_id"
    )
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in second["debt_instrument_mentions"]
        .query("edge_type == 'member'")
        .to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-2"}
    assert written_matches["edge_type"].to_list() == ["member", "member", "related"]
    assert written_matches["debt_instrument_id"].to_list() == ["m-1", "m-2", "m-1"]
    assert written_instruments["debt_instrument_id"].to_list() == ["m-1", "m-2"]


def test_match_tables_supports_incremental_batches_against_existing_clusters() -> None:
    """Delta batches should match against existing clusters without full history."""
    existing_mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-1",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="Alpha Loan",
                start_date="2024-01-01",
                amount="$100 million",
                parties_json='[{"mentions": [{"text": "Acme Bank"}]}]',
            )
        ]
    )
    existing_tables = match_tables(existing_mentions)

    new_mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-2",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2024-01-02",
                name="Alpha Loan",
                start_date="2024-01-01",
                amount="$100 million",
                parties_json='[{"mentions": [{"text": "Acme Bank"}]}]',
            )
        ]
    )
    tables = match_tables(
        new_mentions,
        existing_edges=existing_tables["debt_instrument_mentions"],
        existing_instruments=existing_tables["debt_instrument"],
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-1"}
    assert tables["debt_instrument"]["debt_instrument_id"].to_list() == ["m-1"]
    assert tables["debt_instrument"]["name"].to_list() == ["Alpha Loan"]
    assert tables["debt_instrument"]["company_name"].to_list() == ["Example Inc."]


def test_match_tables_carries_forward_previously_published_retirers() -> None:
    """A delta batch adds to the published retirers instead of replacing them (#180).

    The retired obligation's column is read back out of the existing
    `debt_instrument` row and merged with whatever the new batch found, so an
    obligation retired in two tranches across two filings ends up with both.
    Dropping that read-back loses every retirer published before the delta,
    and nothing else in the suite exercises the parse of the persisted column.
    """
    first_batch = pd.DataFrame(
        [
            {
                **build_mention_row(
                    mention_id="m-old",
                    item_id="item-1",
                    accession_number="0001",
                    cik="320193",
                    date="2026-06-11",
                    name="8.125% senior secured notes due 2028",
                    start_date="2021-06-11",
                    amount="$1,250 million",
                ),
                "retired_by_json": '["m-2034"]',
            },
            build_mention_row(
                mention_id="m-2034",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2026-06-11",
                name="6.375% senior secured notes due 2034",
                start_date="2026-06-11",
                amount="$1,125 million",
            ),
        ]
    )
    existing_tables = match_tables(first_batch)
    assert (
        existing_tables["debt_instrument"]
        .set_index("debt_instrument_id")
        .loc["m-old", "retired_by_debt_instrument_ids"]
        == '["m-2034"]'
    )

    # A later filing redeems the rest of the same notes with a second series.
    second_batch = pd.DataFrame(
        [
            {
                **build_mention_row(
                    mention_id="m-old-again",
                    item_id="item-2",
                    accession_number="0002",
                    cik="320193",
                    date="2026-09-01",
                    name="8.125% senior secured notes due 2028",
                    start_date="2021-06-11",
                    amount="$1,250 million",
                ),
                "retired_by_json": '["m-2036"]',
            },
            build_mention_row(
                mention_id="m-2036",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2026-09-01",
                name="6.625% senior secured notes due 2036",
                start_date="2026-09-01",
                amount="$1,125 million",
            ),
        ]
    )

    tables = match_tables(
        second_batch,
        existing_edges=existing_tables["debt_instrument_mentions"],
        existing_instruments=existing_tables["debt_instrument"],
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    instruments = {
        row["debt_instrument_id"]: row
        for row in tables["debt_instrument"].to_dict("records")
    }
    assert instruments["m-old"]["retired_by_debt_instrument_ids"] == (
        '["m-2034", "m-2036"]'
    )


def test_coerce_optional_text_treats_pandas_nan_as_missing() -> None:
    """Matcher text coercion should drop pandas null sentinels."""
    assert coerce_optional_text(pd.NA) is None
    assert coerce_optional_text(float("nan")) is None


def test_match_tables_does_not_emit_literal_nan_company_names() -> None:
    """Matched instruments should keep missing filer names as null, not 'nan'."""
    mention = build_mention_row(
        mention_id="m-1",
        item_id="item-1",
        accession_number="0001",
        cik="320193",
        date="2024-01-01",
        name="Alpha Loan",
        start_date="2024-01-01",
        amount="$100 million",
        parties_json='[{"mentions": [{"text": "Acme Bank"}]}]',
    )
    mention["company_name"] = pd.NA

    mentions = pd.DataFrame([mention])

    tables = match_tables(mentions)

    assert tables["debt_instrument"]["company_name"].to_list() == [None]


def test_match_tables_retired_by_keeps_separate_clusters_and_ends_the_instrument() -> (
    None
):
    """The retired instrument keeps its own cluster, end date, and retirer pointer."""
    mentions = pd.DataFrame(
        [
            {
                **build_mention_row(
                    mention_id="m-1",
                    item_id="item-1",
                    accession_number="0001",
                    cik="320193",
                    date="2024-01-01",
                    name="Term Loan",
                    start_date="2024-01-01",
                    amount="$100 million",
                    parties_json='[{"mentions": [{"text": "Acme Bank"}]}]',
                ),
                "maturity_date": None,
            },
            {
                **build_mention_row(
                    mention_id="m-2",
                    item_id="item-2",
                    accession_number="0002",
                    cik="320193",
                    date="2024-03-01",
                    name="Term Loan",
                    start_date="2024-01-01",
                    amount="$100 million",
                    parties_json='[{"mentions": [{"text": "Acme Bank"}]}]',
                ),
                "maturity_date": "2024-03-01",
                "retired_by_json": '["m-1"]',
            },
        ]
    )

    tables = match_tables(
        mentions,
        strong_match_threshold=0.90,
        loose_match_threshold=0.75,
    )

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assert {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    } == {"m-1": "m-1", "m-2": "m-2"}

    instruments = {
        row["debt_instrument_id"]: row
        for row in tables["debt_instrument"].to_dict("records")
    }
    assert instruments["m-2"]["retired_by_debt_instrument_ids"] == '["m-1"]'
    # No cross-row propagation: the retirement filing's mention sits in the
    # retired instrument's own cluster, so its end date is already there.
    assert instruments["m-2"]["maturity_date"] == "2024-03-01"
    assert instruments["m-1"]["maturity_date"] is None


def test_match_tables_publishes_two_kinds_of_lineage_for_one_instrument() -> None:
    """Split and retirement lineage coexist in their own columns (#130).

    Pitney Bowes' incremental tranche A term loans split from the existing
    tranche A loans and redeemed the 2027 notes with the proceeds; a later
    refinancing then repaid the incremental loans. The incremental loans' row
    carries both a split parent and a retirer; nulling every parent column
    whenever a second kind appeared discarded both links.
    """
    mentions = pd.DataFrame(
        [
            {
                **build_mention_row(
                    mention_id="m-notes",
                    item_id="item-1",
                    accession_number="0001",
                    cik="320193",
                    date="2025-02-07",
                    name="6.875% Senior Notes due March 2027",
                    start_date="2025-02-07",
                    amount="$347 million",
                ),
                "retired_by_json": '["m-incremental"]',
            },
            build_mention_row(
                mention_id="m-tranche",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2025-02-07",
                name="tranche A term loans",
                start_date="2025-02-07",
                amount="$302 million",
            ),
            {
                **build_mention_row(
                    mention_id="m-incremental",
                    item_id="item-1",
                    accession_number="0001",
                    cik="320193",
                    date="2026-06-23",
                    name="Incremental Term Loans",
                    start_date="2026-06-23",
                    amount="$150 million",
                ),
                "split_of": "m-tranche",
                "retired_by_json": '["m-refi"]',
            },
            build_mention_row(
                mention_id="m-refi",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2027-03-01",
                name="Refinancing Term Loans",
                start_date="2027-03-01",
                amount="$150 million",
            ),
        ]
    )

    tables = match_tables(mentions)

    instruments = {
        row["debt_instrument_id"]: row
        for row in tables["debt_instrument"].to_dict("records")
    }
    row = instruments["m-incremental"]
    assert row["split_of_debt_instrument_id"] == "m-tranche"
    assert row["retired_by_debt_instrument_ids"] == '["m-refi"]'
    assert row["amendment_of_debt_instrument_id"] is None
    # The retirer is itself retired one link up the chain, and the instrument
    # at the end of the chain retires nothing, so its column is null rather
    # than an empty array (#180).
    assert (
        instruments["m-notes"]["retired_by_debt_instrument_ids"] == '["m-incremental"]'
    )
    assert instruments["m-refi"]["retired_by_debt_instrument_ids"] is None


def test_match_tables_keeps_every_joint_retirer() -> None:
    """Several instruments jointly retiring one obligation all publish.

    Venture Global's two new series ($1.125B due 2034 and due 2036) jointly
    redeem the 8.125% notes due 2028. Two retirers is a legitimate state of the
    world, not extraction ambiguity, so the retired row keeps both -- unlike the
    single-parent amendment/split columns, which clear on ambiguity (#130).
    """
    mentions = pd.DataFrame(
        [
            {
                **build_mention_row(
                    mention_id="m-old",
                    item_id="item-1",
                    accession_number="0001",
                    cik="320193",
                    date="2026-06-11",
                    name="8.125% senior secured notes due 2028",
                    start_date="2021-06-11",
                    amount="$1,250 million",
                ),
                "retired_by_json": '["m-2034", "m-2036"]',
            },
            build_mention_row(
                mention_id="m-2034",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2026-06-11",
                name="6.375% senior secured notes due 2034",
                start_date="2026-06-11",
                amount="$1,125 million",
            ),
            build_mention_row(
                mention_id="m-2036",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2026-06-11",
                name="6.625% senior secured notes due 2036",
                start_date="2026-06-11",
                amount="$1,125 million",
            ),
        ]
    )

    tables = match_tables(mentions)

    instruments = {
        row["debt_instrument_id"]: row
        for row in tables["debt_instrument"].to_dict("records")
    }
    assert instruments["m-old"]["retired_by_debt_instrument_ids"] == (
        '["m-2034", "m-2036"]'
    )


def test_match_tables_drops_only_the_ambiguous_relation_kind() -> None:
    """Two parents of one kind stay unresolvable; a different kind survives (#130)."""
    mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-a",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="Facility A",
                start_date="2024-01-01",
                amount="$100 million",
            ),
            build_mention_row(
                mention_id="m-b",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="Facility B",
                start_date="2024-02-01",
                amount="$200 million",
            ),
            build_mention_row(
                mention_id="m-notes",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-01",
                name="7.000% Senior Notes due 2030",
                start_date="2024-03-01",
                amount="$300 million",
            ),
            {
                **build_mention_row(
                    mention_id="m-1",
                    item_id="item-2",
                    accession_number="0002",
                    cik="320193",
                    date="2026-01-01",
                    name="New Facility",
                    start_date="2026-01-01",
                    amount="$400 million",
                ),
                "amendment_of": "m-a",
                "retired_by_json": '["m-notes"]',
            },
            {
                # A separate filing's mention of the same facility: same keys
                # merge cross-item (#161 only blocks same-item pairs), and it
                # brings a second amendment parent with it.
                **build_mention_row(
                    mention_id="m-2",
                    item_id="item-3",
                    accession_number="0003",
                    cik="320193",
                    date="2026-01-02",
                    name="New Facility",
                    start_date="2026-01-01",
                    amount="$400 million",
                ),
                "amendment_of": "m-b",
            },
        ]
    )

    tables = match_tables(mentions)

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assignment = {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    }
    # m-1 and m-2 share every key, so they cluster and bring two amendment
    # parents with them.
    assert assignment["m-1"] == assignment["m-2"]
    row = {
        r["debt_instrument_id"]: r for r in tables["debt_instrument"].to_dict("records")
    }[assignment["m-1"]]
    assert row["amendment_of_debt_instrument_id"] is None
    assert row["retired_by_debt_instrument_ids"] == '["m-notes"]'


def test_match_tables_keeps_same_day_siblings_apart() -> None:
    """Same start date plus a conflicting principal means two instruments (#131).

    Longevity Health issued a $1,250,000 and a $1,100,000 `10% Senior Secured
    Convertible Note` on one day. Both maturities are null and both coupons are
    `10%`, so #64's gates cannot separate them and #79's key-conflicting
    fingerprint path merged them, publishing one principal and losing the other.
    """
    mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-initial",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2026-08-13",
                name="10% Senior Secured Convertible Note",
                start_date="2026-08-13",
                amount="$1,250,000",
            ),
            build_mention_row(
                mention_id="m-additional",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2026-08-13",
                name="10% Senior Secured Convertible Note",
                start_date="2026-08-13",
                amount="$1,100,000",
            ),
        ]
    )

    tables = match_tables(mentions)

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assignment = {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    }
    assert assignment["m-initial"] != assignment["m-additional"]
    amounts = {
        row["debt_instrument_id"]: row["principal_amount"]
        for row in tables["debt_instrument"].to_dict("records")
    }
    assert amounts[assignment["m-initial"]] == "$1,250,000"
    assert amounts[assignment["m-additional"]] == "$1,100,000"


def test_match_tables_still_attaches_an_add_on_to_its_series() -> None:
    """An add-on on a later date keeps merging into the existing series (#131).

    Encompass Health sold $100 million of additional 5.875% Senior Notes due
    2034 into its existing $500 million series. The start dates differ, which is
    what distinguishes a second observation of one instrument from a same-day
    sibling, so #79's path must still fire here.
    """
    mentions = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-series",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2026-05-29",
                name="5.875% Senior Notes due 2034",
                start_date="2026-05-29",
                amount="$500 million",
            ),
            build_mention_row(
                mention_id="m-addon",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date="2026-08-13",
                name="5.875% Senior Notes due 2034",
                start_date="2026-08-13",
                amount="$100 million",
            ),
        ]
    )

    tables = match_tables(mentions)

    member_edges = tables["debt_instrument_mentions"].query("edge_type == 'member'")
    assignment = {
        row["debt_instrument_mention_id"]: row["debt_instrument_id"]
        for row in member_edges.to_dict("records")
    }
    assert assignment["m-series"] == assignment["m-addon"]
    via = {
        row["debt_instrument_mention_id"]: row["match_via"]
        for row in member_edges.to_dict("records")
    }
    assert via["m-addon"] == "member:name_fingerprint"


def test_retry_includes_prior_response_as_assistant_turn() -> None:
    """Retry conversations must include the failed output the retry references."""
    row_state = ExtractionRowState(item_row={"item_id": "item-1"}, stage_name="ner")
    row_state.add_messages([{"role": "user", "content": "tag this filing"}])
    row_state.add_response("<bad-xml>")
    row_state.add_validation(["unclosed tag"])
    row_state.retry("Your previous NER output failed validation: unclosed tag")
    assert row_state.current_attempt.messages == [
        {"role": "user", "content": "tag this filing"},
        {"role": "assistant", "content": "<bad-xml>"},
        {
            "role": "user",
            "content": "Your previous NER output failed validation: unclosed tag",
        },
    ]


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

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", failing_workflow)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # The partition is registered complete even though the row produced nothing,
    # which is exactly why the failure has to be recorded somewhere durable.
    assert (
        "extract"
        in read_json_artifact(
            completion_registry_path("extract", artifact_root=tmp_path)
        )["stage"]
    )
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
        "cdt.extractor.core.run_extraction_workflow", succeeding_workflow
    )
    extract_pending_items(artifact_root=tmp_path, batch_size=5, force=True, client=None)

    assert load_row_failures("extract", artifact_root=tmp_path) == {}


def test_existing_date_shard_partition_ids_lists_written_partitions(
    tmp_path: Path,
) -> None:
    """The one-LIST partition-id set matches exactly what was written (#83)."""
    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-03-04", "shard": "0001"},
        table=table,
    )

    ids = existing_date_shard_partition_ids("items", artifact_root=str(root))

    assert ids == {("2026-01-02", "0007"), ("2026-03-04", "0001")}
    assert (
        existing_date_shard_partition_ids("mentions", artifact_root=str(root)) == set()
    )


def test_iter_date_shard_partitions_skips_orphaned_tempfiles(tmp_path: Path) -> None:
    """A tempfile left by a crash between create and rename must not brick the scan (#68)."""
    from cdt.datasets import iter_date_shard_partitions

    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (
        root / "items" / "date=2026-01-02" / "shard=0007" / "tmpabc123.parquet"
    ).write_bytes(b"")

    paths = iter_date_shard_partitions("items", artifact_root=str(root))

    assert len(paths) == 1
    assert paths[0].endswith("date=2026-01-02/shard=0007/part-0000.parquet")


def test_iter_date_shard_partitions_raises_on_non_canonical_data(
    tmp_path: Path,
) -> None:
    """Real data laid out wrong must fail loudly, not silently empty the run.

    Skipping a pre-migration flat file would let every stage process nothing
    and exit 0 while ingest keeps counting the flat file's rows as ingested —
    those filings would be invisible to the pipeline forever.
    """
    from cdt.datasets import iter_date_shard_partitions

    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (root / "items" / "items.parquet").write_bytes(b"")

    with pytest.raises(ValueError, match="Non-canonical parquet file"):
        iter_date_shard_partitions("items", artifact_root=str(root))


def test_read_dataset_skips_orphaned_tempfiles(tmp_path: Path) -> None:
    """Every read path, not just the partition scan, must survive an orphan.

    ingest's existing-accession scan, its per-partition merge, the matcher, and
    pipeline finalize all read through read_dataset; a zero-byte tmp*.parquet
    orphan previously made each of them raise ArrowInvalid.
    """
    root = tmp_path / "artifacts" / "items"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (root / "date=2026-01-02" / "shard=0007" / "tmpabc123.parquet").write_bytes(b"")

    read = read_dataset(str(root))

    assert read["item_id"].to_list() == ["a"]


def test_pending_source_partitions_skips_orphans_and_raises_on_flat_files(
    tmp_path: Path,
) -> None:
    """Fingerprint work selection follows the same stray contract as the scan.

    Silently dropping a mis-laid-out real file here would run the stage on
    nothing while ingest keeps counting the file's rows as ingested.
    """
    from cdt.datasets import pending_source_partitions

    root = tmp_path / "artifacts"
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    write_partition_table(
        str(root / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    (
        root / "items" / "date=2026-01-02" / "shard=0007" / "tmpabc123.parquet"
    ).write_bytes(b"")

    pending, _ = pending_source_partitions(
        "classify", "items", "classifications", artifact_root=str(root)
    )

    assert len(pending) == 1
    assert pending[0][0].endswith("date=2026-01-02/shard=0007/part-0000.parquet")

    (root / "items" / "items.parquet").write_bytes(b"")

    with pytest.raises(ValueError, match="Non-canonical parquet file"):
        pending_source_partitions(
            "classify", "items", "classifications", artifact_root=str(root)
        )


def test_completion_registry_saves_merge_concurrent_updates(tmp_path: Path) -> None:
    """Overlapping writers must not lose each other's registry entries (#88).

    A lost entry silently strands a partition (or fake-completes it with empty
    item_ids), so saves overlay only the entries a run changed onto the freshest
    persisted state instead of overwriting the file with a stale snapshot.
    """
    from cdt.datasets import (
        CompletedPartition,
        load_completion_registry,
        save_completion_registry,
    )

    save_completion_registry(
        "itemize", {"P": CompletedPartition(fingerprint="f1")}, artifact_root=tmp_path
    )
    writer_a = load_completion_registry("itemize", artifact_root=tmp_path)
    writer_b = load_completion_registry("itemize", artifact_root=tmp_path)

    writer_b["P"] = CompletedPartition(fingerprint="f2")
    writer_b["Q"] = CompletedPartition(fingerprint="q1")
    save_completion_registry("itemize", writer_b, artifact_root=tmp_path)

    # A loaded P at f1 but never touched it; its save must not revert B's f2.
    writer_a["R"] = CompletedPartition(fingerprint="r1")
    save_completion_registry("itemize", writer_a, artifact_root=tmp_path)

    final = load_completion_registry("itemize", artifact_root=tmp_path)
    assert set(final) == {"P", "Q", "R"}
    assert final["P"].fingerprint == "f2"
    assert final["Q"].fingerprint == "q1"
    assert final["R"].fingerprint == "r1"


def test_pending_source_partitions_stamps_survive_concurrent_saves(
    tmp_path: Path,
) -> None:
    """Legacy-entry stamping counts as a change and survives the merge (#88)."""
    from cdt.datasets import (
        CompletedPartition,
        load_completion_registry,
        pending_source_partitions,
        save_completion_registry,
    )

    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    source_path = write_partition_table(
        str(tmp_path / "items"),
        partition={"date": "2026-01-02", "shard": "0007"},
        table=table,
    )
    # A v1-migrated entry: complete but fingerprint-less.
    save_completion_registry(
        "classify", {source_path: CompletedPartition()}, artifact_root=tmp_path
    )

    pending, registry = pending_source_partitions(
        "classify", "items", "classifications", artifact_root=str(tmp_path)
    )
    assert pending == []
    save_completion_registry("classify", registry, artifact_root=tmp_path)

    final = load_completion_registry("classify", artifact_root=tmp_path)
    assert final[source_path].fingerprint is not None


def _seed_classifications(
    tmp_path: Path, item_ids: list[str], *, date: str = "2024-01-02"
) -> None:
    """Write one classifications partition with the given relevant items."""
    from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS

    rows = []
    for item_id in item_ids:
        row: dict[str, object] = {column: None for column in CLASSIFIED_ITEM_COLUMNS}
        row.update(
            {
                "item_id": item_id,
                "accession_number": item_id.split("-")[0],
                "cik": "320193",
                "date": date,
                "item": "8.01",
                "text": f"text for {item_id}",
                "relevance": True,
            }
        )
        rows.append(row)
    write_partition_table(
        str(classifications_root(tmp_path)),
        partition={"date": date, "shard": "0001"},
        table=pd.DataFrame(rows, columns=CLASSIFIED_ITEM_COLUMNS),
    )


def _fake_success_workflow(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the extraction workflow to succeed with one mention per row."""
    calls: list[str] = []

    async def fake_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        calls.append(str(item_row["item_id"]))
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {"item_id": str(item_row["item_id"]), "name": "Term Loan"}
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", fake_workflow)
    return calls


def test_late_arriving_rows_extract_after_partition_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows merged into an already-processed partition are picked up (#62)."""
    _seed_classifications(tmp_path, ["a-8-01"])
    calls = _fake_success_workflow(monkeypatch)

    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)
    assert calls == ["a-8-01"]

    # Ingest-style in-place merge: the partition object grows a new row.
    _seed_classifications(tmp_path, ["a-8-01", "b-8-01"])
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # Only the new row is paid for, and the target holds both rows' mentions.
    assert calls == ["a-8-01", "b-8-01"]
    written = read_dataset(mentions_root(tmp_path))
    assert sorted(written["item_id"]) == ["a-8-01", "b-8-01"]


def test_infrastructure_error_aborts_and_preserves_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider failure stops the run; terminal rows are never re-paid (#49)."""
    from cdt.datasets import load_completion_registry
    from cdt.extractor.core import InfrastructureError

    _seed_classifications(tmp_path, ["a-8-01", "b-8-01"])
    calls: list[str] = []

    async def failing_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        calls.append(str(item_row["item_id"]))
        if str(item_row["item_id"]) == "b-8-01":
            raise InfrastructureError("PaymentRequiredResponseError: 402")
        row_state = ExtractionRowState(item_row=item_row, stage_name="instrument_ie")
        row_state.debt_instrument_mentions = [
            {"item_id": str(item_row["item_id"]), "name": "Term Loan"}
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", failing_workflow)
    with pytest.raises(InfrastructureError):
        extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    registry = load_completion_registry("extract", artifact_root=tmp_path)
    (entry,) = registry.values()
    assert not entry.complete
    assert entry.item_ids == frozenset({"a-8-01"})
    # The finished row's mentions survived the abort.
    assert sorted(read_dataset(mentions_root(tmp_path))["item_id"]) == ["a-8-01"]

    # Recovery: a healthy run pays only for the row that never got a verdict.
    recovery_calls = _fake_success_workflow(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)
    assert recovery_calls == ["b-8-01"]
    registry = load_completion_registry("extract", artifact_root=tmp_path)
    (entry,) = registry.values()
    assert entry.complete
    assert sorted(read_dataset(mentions_root(tmp_path))["item_id"]) == [
        "a-8-01",
        "b-8-01",
    ]


def test_infrastructure_error_classification() -> None:
    """Status- and name-shaped provider errors classify as infrastructure."""
    from cdt.extractor.core import is_infrastructure_error

    class PaymentRequiredResponseError(Exception):
        pass

    class WithStatus(Exception):
        status_code = 503

    assert is_infrastructure_error(PaymentRequiredResponseError())
    assert is_infrastructure_error(WithStatus())
    assert is_infrastructure_error(ConnectionResetError())
    assert not is_infrastructure_error(ValueError("bad xml"))


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


def test_match_pending_mentions_renews_lease_per_shard(tmp_path: Path) -> None:
    """The matcher extends the writer lease before rewriting each shard (#89)."""
    for index, (cik, date_value, shard) in enumerate(
        [("320193", "2024-01-02", "0001"), ("789019", "2024-01-03", "0002")], start=1
    ):
        write_partition_table(
            tmp_path / "mentions",
            partition={"date": date_value, "shard": shard},
            table=pd.DataFrame(
                [
                    build_mention_row(
                        mention_id=f"m-{index}",
                        item_id=f"item-{index}",
                        accession_number=f"000{index}",
                        cik=cik,
                        date=date_value,
                        name="Term Loan",
                        start_date="2024-01-01",
                        amount="$100 million",
                    )
                ]
            ),
        )
    renewals: list[int] = []

    match_pending_mentions(
        artifact_root=tmp_path, batch_size=5, renew=lambda: renewals.append(1)
    )

    assert len(renewals) == 2


def test_read_table_projects_columns_and_tolerates_missing_ones(
    tmp_path: Path,
) -> None:
    """Column projection is pushed down; absent columns reindex instead of raising (#69)."""
    from cdt.storage import write_table

    path = tmp_path / "table.parquet"
    write_table(path, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    projected = read_table(path, ["a"])
    assert list(projected.columns) == ["a"]

    tolerant = read_table(path, ["a", "missing"])
    assert list(tolerant.columns) == ["a", "missing"]
    assert tolerant["missing"].isna().all()


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


def test_realign_tag_details_maps_offsets_onto_the_original_text() -> None:
    """Evidence offsets index the item's own text, not the model's echo (#154)."""
    from cdt.extractor.core import realign_tag_details

    original = "The  $5,000,000\tTerm Loan closed."
    roundtrip = "The $5,000,000 Term Loan closed."
    tag_details = {
        "tag-1": {
            "type": "amount",
            "text": "$5,000,000",
            "char_start": roundtrip.index("$5,000,000"),
            "char_end": roundtrip.index("$5,000,000") + len("$5,000,000"),
        },
        "tag-2": {
            "type": "debt_instrument",
            "text": "Term Loan",
            "char_start": roundtrip.index("Term Loan"),
            "char_end": roundtrip.index("Term Loan") + len("Term Loan"),
        },
    }
    realigned = realign_tag_details(tag_details, roundtrip, original)
    for tag_id in tag_details:
        start = realigned[tag_id]["char_start"]
        end = realigned[tag_id]["char_end"]
        assert original[start:end] == realigned[tag_id]["text"]
    assert realigned["tag-1"]["text"] == "$5,000,000"
    assert realigned["tag-2"]["text"] == "Term Loan"


def test_realign_tag_details_is_identity_when_texts_match() -> None:
    """The common exact-echo case pays no alignment cost."""
    from cdt.extractor.core import realign_tag_details

    text = "A $10 note."
    details = {
        "tag-1": {"type": "amount", "text": "$10", "char_start": 2, "char_end": 5}
    }
    assert realign_tag_details(details, text, text) is details


def test_realign_tag_details_handles_model_deleted_whitespace() -> None:
    """collapse-equality permits dropped whitespace; spans still land right."""
    from cdt.extractor.core import realign_tag_details

    original = "Senior Notes due 2028\nwere issued."
    roundtrip = "Senior Notes due 2028 were issued."
    details = {
        "tag-1": {
            "type": "debt_instrument",
            "text": "Senior Notes due 2028",
            "char_start": 0,
            "char_end": len("Senior Notes due 2028"),
        }
    }
    realigned = realign_tag_details(details, roundtrip, original)
    start = realigned["tag-1"]["char_start"]
    end = realigned["tag-1"]["char_end"]
    assert original[start:end] == "Senior Notes due 2028"


def test_standardized_payloads_record_where_their_values_came_from() -> None:
    """Payloads carry derived_from so consumers know a value's provenance (#128)."""
    from cdt.extractor.core import (
        standardized_amount_payload,
        standardized_date_payload,
        standardized_end_date_payload,
    )

    tag_details = {
        "tag-d-1": {
            "type": "date",
            "text": "June 1, 2028",
            "char_start": 0,
            "char_end": 12,
        },
        "tag-i-1": {
            "type": "debt_instrument",
            "text": "3.875% senior notes due 2028",
            "char_start": 20,
            "char_end": 48,
        },
        "tag-a-1": {
            "type": "amount",
            "text": "$500,000,000",
            "char_start": 60,
            "char_end": 72,
        },
    }
    stated_date = standardized_date_payload(
        {"evidence": ["tag-d-1"], "normalized_date": "2028-06-01"},
        tag_details,
    )
    assert stated_date["normalized_date"] == "2028-06-01"
    assert stated_date["derived_from"] == "stated"

    name_maturity = standardized_end_date_payload(
        {"evidence": ["tag-i-1"], "normalized_date": "2028-12-31"},
        tag_details,
        name_text="3.875% senior notes due 2028",
    )
    assert name_maturity["normalized_date"] == "2028-12-31"
    assert name_maturity["derived_from"] == "name"

    fallback_maturity = standardized_end_date_payload(
        None,
        tag_details,
        name_text="3.875% senior notes due 2028",
    )
    assert fallback_maturity["normalized_date"] == "2028-12-31"
    assert fallback_maturity["derived_from"] == "name"
    assert fallback_maturity["spans"] == []

    stated_amount = standardized_amount_payload(
        {"evidence": ["tag-a-1"], "normalized_amount": "500000000"},
        tag_details,
    )
    assert stated_amount["normalized_amount"] == "500000000"
    assert stated_amount["derived_from"] == "stated"

    absent = standardized_date_payload(None, tag_details)
    assert absent["normalized_date"] is None
    assert absent["derived_from"] is None


def test_terminal_ie_failure_salvages_the_valid_entries() -> None:
    """One invalid entry no longer drops the whole item (#152)."""
    from cdt.extractor.core import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": None,
                    }
                ],
            },
            {"name": ["tag-o-named"]},
        ]
    )
    # attempt_index reaches max_attempts on this response, forcing terminal
    # handling of the validation failure from the second entry's bad name tag.
    result = handle_response(row_state, response, max_attempts=1)

    assert result is None
    assert row_state.state == "PARTIAL"
    assert len(row_state.debt_instrument_mentions) == 1
    assert row_state.debt_instrument_mentions[0]["name"] == "Term Loan"
    assert row_state.salvage_notes
    assert "dropped 1" in row_state.salvage_notes[0]


def test_terminal_relation_failure_publishes_mentions_without_lineage() -> None:
    """A relation-stage failure keeps the already-validated mentions (#152)."""
    from cdt.extractor.core import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    ie_response = json.dumps(
        [
            {"name": ["tag-i-1"]},
            {
                "name": ["tag-i-1"],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": None,
                    }
                ],
            },
        ]
    )
    next_messages = handle_response(row_state, ie_response, max_attempts=3)
    assert next_messages is not None
    assert row_state.current_attempt.stage_name == "instrument_relation"
    assert len(row_state.debt_instrument_mentions) == 2

    result = handle_response(row_state, "not json at all", max_attempts=1)

    assert result is None
    assert row_state.state == "PARTIAL"
    assert len(row_state.debt_instrument_mentions) == 2
    assert any("without lineage" in note for note in row_state.salvage_notes)


def test_terminal_ie_failure_with_nothing_valid_still_fails() -> None:
    """Salvage never invents output: no valid entry means FAILED as before."""
    from cdt.extractor.core import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    response = json.dumps([{"name": ["tag-unknown"]}])

    result = handle_response(row_state, response, max_attempts=1)

    assert result is None
    assert row_state.state == "FAILED"
    assert row_state.debt_instrument_mentions == []


def test_salvage_notes_round_trip_through_batch_state() -> None:
    """PARTIAL provenance survives the resumable batch state (#152)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.salvage_notes.append("instrument_ie dropped 1 entry")
    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())
    assert restored.salvage_notes == ["instrument_ie dropped 1 entry"]
    # State written before salvage existed lacks the key entirely.
    legacy = row_state.to_state_dict()
    del legacy["salvage_notes"]
    assert ExtractionRowState.from_state_dict(legacy).salvage_notes == []


BALANCE_ITEM_XML = """
<body>
The <debt_instrument id="tag-i-1">Revolving Credit Facility</debt_instrument> provides
<amount id="tag-a-commitment">$300 million</amount> of commitments. As of
<date id="tag-d-asof">June 9, 2026</date>, the Company had
<amount id="tag-a-balance">$270.5 million</amount> outstanding.
</body>
""".strip()


def balance_row_state() -> ExtractionRowState:
    """Return one instrument_ie row state with a commitment and a balance."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = BALANCE_ITEM_XML
    return row_state


def test_amounts_are_kind_typed_and_the_balance_never_becomes_principal() -> None:
    """A commitment and a balance publish as two entries; principal is the commitment (#140)."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "commitment",
                        "evidence": ["tag-a-commitment"],
                        "normalized_amount": "300000000",
                        "currency": "USD",
                    },
                    {
                        "kind": "outstanding_balance",
                        "evidence": ["tag-a-balance"],
                        "normalized_amount": "270500000",
                        "currency": "USD",
                        "as_of_date": "2026-06-09",
                    },
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    payloads = json.loads(str(mention["amounts_json"]))
    assert [(entry["kind"], entry["normalized_amount"]) for entry in payloads] == [
        ("commitment", "300000000"),
        ("outstanding_balance", "270500000"),
    ]
    assert payloads[1]["as_of_date"] == "2026-06-09"
    assert mention["principal_amount"] == "300000000"
    assert mention["principal_currency"] == "USD"
    assert mention["principal_amount_kind"] == "commitment"


def test_a_balance_only_mention_publishes_no_principal() -> None:
    """A balance observation alone never becomes the headline amount (#140)."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "outstanding_balance",
                        "evidence": ["tag-a-balance"],
                        "normalized_amount": "270500000",
                        "currency": "USD",
                        "as_of_date": "2026-06-09",
                    }
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] is None
    assert mention["principal_amount_kind"] is None
    payloads = json.loads(str(mention["amounts_json"]))
    assert payloads[0]["kind"] == "outstanding_balance"


def test_instrument_ie_validate_rejects_unknown_amount_kinds() -> None:
    """An amounts entry must carry a known kind and a valid as_of_date (#140)."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "headline",
                        "evidence": ["tag-a-commitment"],
                        "normalized_amount": "300000000",
                        "as_of_date": "June 9, 2026",
                    }
                ],
            }
        ]
    )
    failures = InstrumentIEStage().validate(balance_row_state(), response)
    assert any("'amounts[0].kind' must be one of" in failure for failure in failures)
    assert any(
        "'amounts[0].as_of_date' must be YYYY-MM-DD" in failure for failure in failures
    )


def test_legacy_single_amount_shape_still_replays() -> None:
    """Pre-#140 batch responses with a bare `amount` keep their value."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amount": {
                    "evidence": ["tag-a-commitment"],
                    "normalized_amount": "300000000",
                    "currency": "USD",
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] == "300000000"
    # The legacy shape carries no kind; the flat column still fills.
    assert mention["principal_amount_kind"] is None


DRAW_PERIOD_XML = """
<body>
The <debt_instrument id="tag-i-1">Delayed Draw Term Loan</debt_instrument> draw period
ends on <date id="tag-d-draw">June 30, 2027</date> and the loans mature on
<date id="tag-d-maturity">June 30, 2031</date>.
</body>
""".strip()


def test_commitment_termination_date_is_its_own_field() -> None:
    """Draw-period ends publish separately from the maturity (#158)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = DRAW_PERIOD_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "maturity_date": {
                    "evidence": ["tag-d-maturity"],
                    "normalized_date": "2031-06-30",
                },
                "commitment_termination_date": {
                    "evidence": ["tag-d-draw"],
                    "normalized_date": "2027-06-30",
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] == "2031-06-30"
    assert mention["commitment_termination_date"] == "2027-06-30"
    payload = json.loads(str(mention["commitment_termination_date_json"]))
    assert payload["derived_from"] == "stated"


def test_legacy_end_date_property_replays_into_maturity_date() -> None:
    """Pre-#158 batch responses using `end_date` keep their maturity."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = DRAW_PERIOD_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "end_date": {
                    "evidence": ["tag-d-maturity"],
                    "normalized_date": "2031-06-30",
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    assert row_state.debt_instrument_mentions[0]["maturity_date"] == "2031-06-30"


TERMINATION_ITEM_XML = """
<body>
On <date id="tag-d-term">June 2, 2026</date>, the Company terminated its
<debt_instrument id="tag-i-1">$3.5 billion five-year revolving credit facility</debt_instrument>,
dated as of <date id="tag-d-dated">October 11, 2023</date>.
</body>
""".strip()


def test_status_event_records_a_standalone_termination() -> None:
    """A 1.02 termination is recordable without a successor object (#141)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TERMINATION_ITEM_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "start_date": {
                    "evidence": ["tag-d-dated"],
                    "normalized_date": "2023-10-11",
                },
                "status_event": {
                    "status": "terminated",
                    "status_date": {
                        "evidence": ["tag-d-term"],
                        "normalized_date": "2026-06-02",
                    },
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["status"] == "terminated"
    assert mention["status_date"] == "2026-06-02"
    payload = json.loads(str(mention["status_json"]))
    assert payload["status"] == "terminated"
    assert [s["tag_id"] for s in payload["status_date"]["spans"]] == ["tag-d-term"]


def test_status_event_validation_rejects_unknown_statuses() -> None:
    """Only the seven agreed statuses validate; matured is derived, not extracted."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TERMINATION_ITEM_XML
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "status_event": {"status": "matured"},
            }
        ]
    )
    failures = InstrumentIEStage().validate(row_state, response)
    assert any(
        "'status_event.status' must be one of" in failure for failure in failures
    )


def test_missing_status_event_publishes_null_status() -> None:
    """A mention that states no event carries no status."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TERMINATION_ITEM_XML
    row_state.stage_responses["instrument_ie"] = json.dumps([{"name": ["tag-i-1"]}])
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["status"] is None
    assert mention["status_date"] is None


def test_instrument_type_persists_and_rejects_unknown_values() -> None:
    """instrument_type is one of four categories or absent (#156)."""
    row_state = balance_row_state()
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [{"name": ["tag-i-1"], "instrument_type": "revolving_credit"}]
    )
    InstrumentIEStage().postprocess(row_state)
    assert row_state.debt_instrument_mentions[0]["instrument_type"] == (
        "revolving_credit"
    )

    failures = InstrumentIEStage().validate(
        balance_row_state(),
        json.dumps([{"name": ["tag-i-1"], "instrument_type": "surety_bond"}]),
    )
    assert any("'instrument_type' must be one of" in failure for failure in failures)


RATE_TAGGED_XML = """
<body>
The <debt_instrument id="tag-i-1">3.875% senior notes due 2028</debt_instrument> were
issued, and revolver borrowings bear interest at
<interest_rate id="tag-r-1">SOFR plus 0.875% per annum</interest_rate>.
</body>
""".strip()


def test_interest_rate_is_parser_verified_from_name_or_evidence() -> None:
    """rate_pct publishes only when a cited or name-embedded rate matches (#157)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_TAGGED_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "interest_rate": {
                    "kind": "fixed",
                    "rate_pct": "3.875",
                    "evidence": ["tag-i-1"],
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["interest_rate_kind"] == "fixed"
    assert mention["interest_rate_pct"] == "3.875"
    payload = json.loads(str(mention["interest_rate_json"]))
    assert payload["derived_from"] == "name"


def test_interest_rate_mismatch_publishes_null_rate() -> None:
    """A model rate contradicted by every cited span keeps kind but no number."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_TAGGED_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "interest_rate": {
                    "kind": "floating",
                    "rate_pct": "5.5",
                    "evidence": ["tag-r-1"],
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["interest_rate_kind"] == "floating"
    assert mention["interest_rate_pct"] is None


def test_interest_rate_validation_rejects_bad_kind_and_evidence() -> None:
    """Kind must be fixed or floating; evidence must be rate or own-name tags."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = RATE_TAGGED_XML
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "interest_rate": {"kind": "variable", "evidence": ["tag-r-1"]},
            }
        ]
    )
    failures = InstrumentIEStage().validate(row_state, response)
    assert any("'interest_rate.kind' must be one of" in failure for failure in failures)


def test_canonical_fields_record_their_source_mention() -> None:
    """Each canonical value points at the mention it came from (#151)."""
    from cdt.matcher.core import build_debt_instrument_rows, prepare_mention

    older = prepare_mention(
        build_mention_row(
            mention_id="m-old",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2024-01-02",
            name="Term Loan",
            start_date="2024-01-01",
            amount="$100 million",
        )
    )
    newer = prepare_mention(
        build_mention_row(
            mention_id="m-new",
            item_id="item-2",
            accession_number="0002",
            cik="320193",
            date="2024-06-02",
            name="Term Loan",
            start_date=None,
            amount=None,
        )
    )
    mention_index = {"m-old": older, "m-new": newer}
    rows = build_debt_instrument_rows(
        {"m-old": ["m-old", "m-new"]},
        mention_index,
        {},
    )
    row = rows[0]
    # The name comes from the newest mention; the start date and amount only
    # exist on the older one, and their provenance says so.
    assert row["name_source_mention_id"] == "m-new"
    assert row["start_date"] == "2024-01-01"
    assert row["start_date_source_mention_id"] == "m-old"
    assert row["principal_source_mention_id"] == "m-old"


def test_lifecycle_rollup_marks_heads_families_and_status() -> None:
    """Amendment chains get superseded/head markers, families, and status (#155)."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    predecessor = prepare_mention(
        build_mention_row(
            mention_id="m-old",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2024-01-02",
            name="Revolver (original)",
            start_date="2020-01-01",
            amount="$100 million",
        )
    )
    amended = prepare_mention(
        build_mention_row(
            mention_id="m-new",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2024-01-02",
            name="Revolver (as amended)",
            start_date="2020-01-01",
            amount="$150 million",
        )
    )
    rows = [
        {
            "debt_instrument_id": "inst-old",
            "amendment_of_debt_instrument_id": None,
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": None,
            "maturity_date": "2026-06-28",
        },
        {
            "debt_instrument_id": "inst-new",
            "amendment_of_debt_instrument_id": "inst-old",
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": None,
            "maturity_date": "2031-06-23",
        },
    ]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-old": ["m-old"], "inst-new": ["m-new"]},
        mention_index={"m-old": predecessor, "m-new": amended},
    )
    old_row, new_row = rows
    assert old_row["superseded_by_debt_instrument_id"] == "inst-new"
    assert old_row["is_lineage_head"] is False
    assert (old_row["status"], old_row["status_subtype"]) == ("closed", "superseded")
    assert new_row["is_lineage_head"] is True
    assert new_row["status"] == "active"
    assert old_row["lineage_family_id"] == new_row["lineage_family_id"]
    assert new_row["first_seen_filing_date"] == "2024-01-02"
    assert new_row["mention_count"] == 1
    assert new_row["document_count"] == 1


def test_lifecycle_status_prefers_terminal_events_and_derives_expected_closed() -> None:
    """A terminated event closes the row; a past maturity is only expected_closed (#183)."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    terminated = prepare_mention(
        build_mention_row(
            mention_id="m-term",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-06-03",
            name="Old Facility",
            start_date="2023-10-11",
            amount=None,
        )
        | {"status": "terminated", "status_date": "2026-06-02"}
    )
    stale = prepare_mention(
        build_mention_row(
            mention_id="m-stale",
            item_id="item-2",
            accession_number="0002",
            cik="320193",
            date="2026-06-03",
            name="4.875% Senior Notes due 2024",
            start_date="2017-12-19",
            amount=None,
        )
    )
    rows = [
        {
            "debt_instrument_id": "inst-term",
            "amendment_of_debt_instrument_id": None,
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": None,
            "maturity_date": None,
        },
        {
            "debt_instrument_id": "inst-stale",
            "amendment_of_debt_instrument_id": None,
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": None,
            "maturity_date": "2024-01-15",
        },
    ]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-term": ["m-term"], "inst-stale": ["m-stale"]},
        mention_index={"m-term": terminated, "m-stale": stale},
    )
    term_row, stale_row = rows
    # An extracted terminal event is a closure with its cause in the subtype.
    assert (term_row["status"], term_row["status_subtype"]) == ("closed", "terminated")
    assert term_row["status_date"] == "2026-06-02"
    assert term_row["status_source_mention_id"] == "m-term"
    # A passed maturity is a schedule, not an observed repayment, so the row is
    # only *expected* closed and carries no cause.
    assert stale_row["status"] == "expected_closed"
    assert stale_row["status_subtype"] is None
    assert stale_row["status_date"] == "2024-01-15"


def test_canonical_maturity_prefers_stated_over_name_derived() -> None:
    """A newer `due 2030` synthetic never outranks an older stated maturity (#162)."""
    import json as _json

    from cdt.matcher.core import build_debt_instrument_rows, prepare_mention

    closing = prepare_mention(
        build_mention_row(
            mention_id="m-closing",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-06-29",
            name="6.75% PIK Notes due 2030",
            start_date="2026-06-29",
            amount="$350 million",
        )
        | {
            "maturity_date": "2030-07-01",
            "maturity_date_json": _json.dumps({"derived_from": "stated"}),
        }
    )
    later = prepare_mention(
        build_mention_row(
            mention_id="m-later",
            item_id="item-2",
            accession_number="0002",
            cik="320193",
            date="2026-07-23",
            name="6.75% PIK Notes due 2030",
            start_date=None,
            amount=None,
        )
        | {
            "maturity_date": "2030-12-31",
            "maturity_date_json": _json.dumps({"derived_from": "name"}),
        }
    )
    rows = build_debt_instrument_rows(
        {"inst": ["m-closing", "m-later"]},
        {"m-closing": closing, "m-later": later},
        {},
    )
    assert rows[0]["maturity_date"] == "2030-07-01"
    assert rows[0]["maturity_source_mention_id"] == "m-closing"

    # With no stated value anywhere, the name-derived one still publishes.
    rows = build_debt_instrument_rows(
        {"inst": ["m-later"]},
        {"m-later": later},
        {},
    )
    assert rows[0]["maturity_date"] == "2030-12-31"
    assert rows[0]["maturity_source_mention_id"] == "m-later"


def test_normalized_maturity_from_text_parses_month_year_phrases() -> None:
    """`due April 2033` normalizes to the month's last day (#164)."""
    assert normalized_maturity_from_text("notes due April 2033") == "2033-04-30"
    assert normalized_maturity_from_text("notes due in February 2028") == "2028-02-29"
    assert normalized_maturity_from_text("due September 2031") == "2031-09-30"
    # A full date still wins its own precision, and coordinated month-years
    # are two maturities.
    assert normalized_maturity_from_text("due April 7, 2033") == "2033-04-07"
    assert normalized_maturity_from_text("due April 2033 and June 2035") is None
    assert normalized_maturity_from_text("due April 2033 and 2035") is None
    assert normalized_maturity_from_text("due October 1, 2028 and April 2030") is None
    # A month-year restating the full date's own month is not a second maturity.
    assert (
        normalized_maturity_from_text("due April 7, 2033, i.e. due April 2033")
        == "2033-04-07"
    )


def test_computed_sum_amount_accepts_only_the_exact_sum_of_cited_spans() -> None:
    """An increase-by amendment's unstated total publishes as computed (#165)."""
    from cdt.extractor.core import standardized_amount_payload

    tag_details = {
        "tag-a-before": {
            "type": "amount",
            "text": "$200 million",
            "char_start": 10,
            "char_end": 22,
        },
        "tag-a-increment": {
            "type": "amount",
            "text": "$50 million",
            "char_start": 40,
            "char_end": 51,
        },
        "tag-a-rate": {
            "type": "amount",
            "text": "0.50%",
            "char_start": 60,
            "char_end": 65,
        },
    }
    computed = standardized_amount_payload(
        {
            "evidence": ["tag-a-before", "tag-a-increment"],
            "normalized_amount": "250000000",
            "currency": "USD",
        },
        tag_details,
    )
    assert computed["normalized_amount"] == "250000000"
    assert computed["derived_from"] == "computed"
    assert computed["currency"] == "USD"

    # A value that is not the exact sum stays null.
    wrong = standardized_amount_payload(
        {
            "evidence": ["tag-a-before", "tag-a-increment"],
            "normalized_amount": "300000000",
        },
        tag_details,
    )
    assert wrong["normalized_amount"] is None
    assert wrong["derived_from"] is None

    # A single-span citation is agreement, never computation.
    single = standardized_amount_payload(
        {
            "evidence": ["tag-a-before"],
            "normalized_amount": "200000000",
            "currency": "USD",
        },
        tag_details,
    )
    assert single["normalized_amount"] == "200000000"
    assert single["derived_from"] == "stated"

    # A rate-like span in the citation disables the computed path.
    with_rate = standardized_amount_payload(
        {
            "evidence": ["tag-a-before", "tag-a-rate"],
            "normalized_amount": "200000000.5",
        },
        tag_details,
    )
    assert with_rate["normalized_amount"] is None


TENOR_FACILITY_XML = """
<body>
On <date id="tag-d-close">June 24, 2026</date>, the Company entered into a
<duration id="tag-t-1">five-year</duration>
<debt_instrument id="tag-i-1">senior secured revolving credit facility</debt_instrument>
of <amount id="tag-a-1">$1.0 billion</amount>.
</body>
""".strip()


def test_computed_maturity_from_start_plus_tenor() -> None:
    """A cited closing date plus a cited tenor verifies the maturity (#166)."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TENOR_FACILITY_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "start_date": {
                    "evidence": ["tag-d-close"],
                    "normalized_date": "2026-06-24",
                },
                "maturity_date": {
                    "evidence": ["tag-t-1", "tag-d-close"],
                    "normalized_date": "2031-06-24",
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] == "2031-06-24"
    payload = json.loads(str(mention["maturity_date_json"]))
    assert payload["derived_from"] == "computed"
    assert {s["tag_id"] for s in payload["spans"]} == {"tag-t-1", "tag-d-close"}


def test_computed_maturity_rejects_arithmetic_that_misses() -> None:
    """A model date that is not start plus tenor stays null."""
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = TENOR_FACILITY_XML
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "maturity_date": {
                    "evidence": ["tag-t-1", "tag-d-close"],
                    "normalized_date": "2030-06-24",
                },
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)
    assert row_state.debt_instrument_mentions[0]["maturity_date"] is None


def test_tenor_parsing_and_date_arithmetic() -> None:
    """Tenor spans parse conservatively; month-end days clamp (#166)."""
    from cdt.extractor.core import date_plus_tenor, tenor_from_text

    assert tenor_from_text("five-year") == (5, "year")
    assert tenor_from_text("364-day") == (364, "day")
    assert tenor_from_text("18-month") == (18, "month")
    assert tenor_from_text("three year") == (3, "year")
    # Two distinct tenors anchor nothing.
    assert tenor_from_text("three-year term plus two one-year extensions") is None
    assert tenor_from_text("no tenor here") is None

    assert date_plus_tenor("2026-06-24", (5, "year")) == "2031-06-24"
    assert date_plus_tenor("2026-01-02", (364, "day")) == "2027-01-01"
    assert date_plus_tenor("2026-08-31", (18, "month")) == "2028-02-29"


def test_lifecycle_status_treats_future_dated_retirement_as_pending() -> None:
    """A `repaid` event dated after its filing is an intent; the row stays active."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    target = prepare_mention(
        build_mention_row(
            mention_id="m-target",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-03-04",
            name="5.25% Senior Notes due 2027",
            start_date=None,
            amount=None,
        )
        | {"status": "repaid", "status_date": "2026-04-03"}
    )
    new_notes = prepare_mention(
        build_mention_row(
            mention_id="m-new",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-03-04",
            name="6.00% Senior Notes due 2031",
            start_date=None,
            amount=None,
        )
        | {"status": "announced", "status_date": "2026-03-18"}
    )
    rows = [
        {
            "debt_instrument_id": "inst-target",
            "amendment_of_debt_instrument_id": None,
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": '["inst-new"]',
            "maturity_date": "2027-12-31",
        },
        {
            "debt_instrument_id": "inst-new",
            "amendment_of_debt_instrument_id": None,
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": None,
            "maturity_date": "2031-12-31",
        },
    ]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-target": ["m-target"], "inst-new": ["m-new"]},
        mention_index={"m-target": target, "m-new": new_notes},
    )
    target_row, new_row = rows
    # The redemption has not happened, and the retiring notes have not closed.
    assert target_row["status"] == "active"
    assert target_row["status_source_mention_id"] is None
    # An announcement is dated no later than the filing that announced it.
    assert new_row["status"] == "announced"
    assert new_row["status_date"] == "2026-03-04"


def test_resolve_candidates_attaches_on_name_only_tie_instead_of_seeding() -> None:
    """A mention tying two clusters on its name joins the exact-name one."""
    from cdt.matcher.core import CandidateScore, prepare_mention, resolve_candidates

    mention = prepare_mention(
        build_mention_row(
            mention_id="m-3",
            item_id="item-3",
            accession_number="0003",
            cik="923796",
            date="2022-01-06",
            name="5.875% Senior Notes due 2024",
            start_date=None,
            amount=None,
        )
    )
    candidates = [
        CandidateScore(
            debt_instrument_id="inst-generic",
            match_score=0.9,
            support_family="name",
            basis="name_fingerprint",
            exact_name=False,
            cluster_size=3,
        ),
        CandidateScore(
            debt_instrument_id="inst-series",
            match_score=0.9,
            support_family="name",
            basis="name_fingerprint",
            exact_name=True,
            cluster_size=1,
        ),
    ]
    cluster_id, edges = resolve_candidates(
        mention,
        candidates,
        strong_match_threshold=0.9,
        loose_match_threshold=0.75,
        ambiguity_margin=0.05,
        evaluated_run_id="run-1",
    )
    assert cluster_id == "inst-series"
    by_type = {edge["edge_type"]: edge for edge in edges}
    assert by_type["member"]["debt_instrument_id"] == "inst-series"
    assert by_type["member"]["match_via"] == "member:name_fingerprint"
    assert by_type["ambiguous_candidate"]["debt_instrument_id"] == "inst-generic"


def test_date_payload_verifies_against_every_cited_span() -> None:
    """A date co-cited with a defined term keeps its value (2026-09 window)."""
    from cdt.extractor.core import standardized_date_payload

    tags = {
        "tag-7": {
            "text": "March 2, 2026",
            "type": "date",
            "char_start": 0,
            "char_end": 13,
        },
        "tag-8": {
            "text": "Redemption Date",
            "type": "date",
            "char_start": 20,
            "char_end": 35,
        },
    }
    payload = standardized_date_payload(
        {"evidence": ["tag-7", "tag-8"], "normalized_date": "2026-03-02"}, tags
    )
    assert payload["normalized_date"] == "2026-03-02"
    assert payload["derived_from"] == "stated"


def test_month_year_maturity_is_read_outside_due_phrases() -> None:
    """`in March 2056` is a stated month-resolution maturity, not a start date."""
    from cdt.extractor.core import (
        normalized_date_from_text,
        normalized_month_year_from_text,
        standardized_date_payload,
    )

    assert normalized_month_year_from_text("in March 2056") == "2056-03-31"
    assert normalized_month_year_from_text("June 2016") == "2016-06-30"
    assert normalized_month_year_from_text("Series 2026A") is None
    assert normalized_month_year_from_text("April 2033 and June 2035") is None
    assert normalized_date_from_text("in March 2056") is None
    tags = {
        "tag-3": {
            "text": "in March 2056",
            "type": "date",
            "char_start": 0,
            "char_end": 13,
        }
    }
    value = {"evidence": ["tag-3"], "normalized_date": "2056-03-31"}
    maturity = standardized_date_payload(value, tags, allow_maturity_phrase=True)
    assert maturity["normalized_date"] == "2056-03-31"
    assert maturity["derived_from"] == "stated"
    start = standardized_date_payload(value, tags)
    assert start["normalized_date"] is None


def test_rate_spelled_percent_parses() -> None:
    """`6.5 percent` is a rate, for the rate payload and the amount guard alike."""
    from cdt.extractor.core import RATE_PCT_PATTERN, is_rate_like_amount_text

    assert RATE_PCT_PATTERN.findall("6.5 percent senior notes due 2028") == ["6.5"]
    assert RATE_PCT_PATTERN.findall("4.950% notes") == ["4.950"]
    assert is_rate_like_amount_text("6.5 percent") is True


def test_computed_maturity_accepts_a_cited_date_minus_a_tenor() -> None:
    """`extended six months to September 3, 2027` anchors the prior maturity (#166)."""
    from cdt.extractor.core import computed_maturity_date, date_plus_tenor

    assert date_plus_tenor("2027-09-03", (6, "month"), sign=-1) == "2027-03-03"
    tags = {
        "tag-1": {
            "text": "September 3, 2027",
            "type": "date",
            "char_start": 0,
            "char_end": 17,
        },
        "tag-2": {
            "text": "six months",
            "type": "duration",
            "char_start": 20,
            "char_end": 30,
        },
    }
    assert (
        computed_maturity_date(["tag-1", "tag-2"], tags, "2027-03-03") == "2027-03-03"
    )
    assert (
        computed_maturity_date(["tag-1", "tag-2"], tags, "2028-03-03") == "2028-03-03"
    )
    assert computed_maturity_date(["tag-1", "tag-2"], tags, "2027-04-03") is None


def test_instrument_ie_accepts_a_bare_object_as_one_entry() -> None:
    """A bare object is the one-instrument case, not a validation failure."""
    from cdt.extractor.core import instrument_entries_from_response

    assert instrument_entries_from_response('{"name": ["tag-1"]}') == [
        {"name": ["tag-1"]}
    ]
    assert instrument_entries_from_response('[{"name": ["tag-1"]}]') == [
        {"name": ["tag-1"]}
    ]
    assert instrument_entries_from_response("[]") == []


def _dates_tag_details() -> dict[str, dict[str, object]]:
    return {
        "tag-1": {
            "text": "5.000% Senior Notes due 2031",
            "type": "debt_instrument",
            "char_start": 0,
            "char_end": 28,
        },
        "tag-2": {
            "text": "March 5, 2026",
            "type": "date",
            "char_start": 40,
            "char_end": 53,
        },
        "tag-3": {
            "text": "March 12, 2026",
            "type": "date",
            "char_start": 60,
            "char_end": 74,
        },
        "tag-4": {
            "text": "June 28, 2026",
            "type": "date",
            "char_start": 80,
            "char_end": 93,
        },
        "tag-5": {
            "text": "June 23, 2031",
            "type": "date",
            "char_start": 100,
            "char_end": 113,
        },
        "tag-6": {
            "text": "in March 2056",
            "type": "date",
            "char_start": 120,
            "char_end": 133,
        },
    }


def test_dates_facts_publish_columns_from_current_closing_and_maturity() -> None:
    """dates[] replaces the single-value slots; prior and projected dates stay out of the columns."""
    from cdt.extractor.core import select_date_payload, standardized_dates_payloads

    obj = {
        "name": ["tag-1"],
        "dates": [
            {
                "kind": "announcement",
                "evidence": ["tag-2"],
                "normalized_date": "2026-03-05",
            },
            {
                "kind": "expected_closing",
                "evidence": ["tag-3"],
                "normalized_date": "2026-03-12",
            },
            {
                "kind": "maturity",
                "evidence": ["tag-4"],
                "normalized_date": "2026-06-28",
                "prior": True,
            },
            {
                "kind": "maturity",
                "evidence": ["tag-5"],
                "normalized_date": "2031-06-23",
            },
        ],
    }
    payloads = standardized_dates_payloads(
        obj, _dates_tag_details(), name_text="5.000% Senior Notes due 2031"
    )
    by_kind = {(p["kind"], p["prior"]): p for p in payloads}
    assert by_kind[("announcement", False)]["normalized_date"] == "2026-03-05"
    expected = [p for p in payloads if p["kind"] == "closing" and p["expected"]]
    assert expected and expected[0]["normalized_date"] == "2026-03-12"
    assert by_kind[("maturity", True)]["normalized_date"] == "2026-06-28"
    assert by_kind[("maturity", False)]["normalized_date"] == "2031-06-23"
    assert by_kind[("maturity", False)]["precision"] == "day"
    # No closing fact: the announced instrument publishes no start date.
    assert select_date_payload(payloads, "closing")["normalized_date"] is None
    assert select_date_payload(payloads, "maturity")["normalized_date"] == "2031-06-23"


def test_dates_facts_precision_and_legacy_shape() -> None:
    """Month and year precision are read off the text; old responses replay with implied kinds."""
    from cdt.extractor.core import standardized_dates_payloads

    tags = _dates_tag_details()
    month = standardized_dates_payloads(
        {
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-6"],
                    "normalized_date": "2056-03-31",
                }
            ]
        },
        tags,
        name_text="Class A-2 Notes",
    )
    assert (
        month[0]["normalized_date"] == "2056-03-31" and month[0]["precision"] == "month"
    )
    year = standardized_dates_payloads(
        {"name": ["tag-1"]}, tags, name_text="5.000% Senior Notes due 2031"
    )
    assert year[0]["kind"] == "maturity" and year[0]["normalized_date"] == "2031-12-31"
    assert year[0]["precision"] == "year" and year[0]["derived_from"] == "name"
    legacy = standardized_dates_payloads(
        {
            "start_date": {"evidence": ["tag-2"], "normalized_date": "2026-03-05"},
            "maturity_date": {"evidence": ["tag-5"], "normalized_date": "2031-06-23"},
        },
        tags,
        name_text=None,
    )
    assert {(p["kind"], p["normalized_date"]) for p in legacy} == {
        ("closing", "2026-03-05"),
        ("maturity", "2031-06-23"),
    }


def test_dates_property_validation_rejects_bad_kind_and_two_current_maturities() -> (
    None
):
    """Two current maturities are two instruments; a prior one is history."""
    from cdt.extractor.core import validate_dates_property

    tags = _dates_tag_details()
    bad_kind = validate_dates_property(
        index=0,
        obj={
            "dates": [{"kind": "issue", "evidence": ["tag-2"], "normalized_date": None}]
        },
        tag_details=tags,
    )
    assert any("kind" in failure for failure in bad_kind)
    two = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-4"],
                    "normalized_date": "2026-06-28",
                },
                {
                    "kind": "maturity",
                    "evidence": ["tag-5"],
                    "normalized_date": "2031-06-23",
                },
            ]
        },
        tag_details=tags,
    )
    assert any("2 current entries of kind 'maturity'" in failure for failure in two)
    ok = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-4"],
                    "normalized_date": "2026-06-28",
                    "prior": True,
                },
                {
                    "kind": "maturity",
                    "evidence": ["tag-5"],
                    "normalized_date": "2031-06-23",
                },
            ]
        },
        tag_details=tags,
    )
    assert ok == []
    name_as_closing = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {"kind": "closing", "evidence": ["tag-1"], "normalized_date": None}
            ]
        },
        tag_details=tags,
    )
    assert any("expected date" in failure for failure in name_as_closing)


def test_prior_amounts_never_supply_the_principal() -> None:
    """A `prior: true` commitment is history; the current figure supplies the principal."""
    from cdt.extractor.core import select_principal_amount

    payloads = [
        {"kind": "commitment", "normalized_amount": "25000000", "prior": True},
        {"kind": "commitment", "normalized_amount": "50000000", "prior": False},
    ]
    assert select_principal_amount(payloads)["normalized_amount"] == "50000000"
    assert select_principal_amount(payloads[:1]) == {}


def test_status_is_derived_from_event_date_facts() -> None:
    """Stage 2: the newest completed event decides status; expected events decide nothing."""
    from cdt.extractor.core import (
        derived_status_payload,
        expected_retirement_in_payloads,
    )

    facts = [
        {
            "kind": "closing",
            "normalized_date": "2023-10-11",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
        {
            "kind": "termination",
            "normalized_date": "2026-06-02",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
    ]
    status = derived_status_payload(facts)
    assert (
        status["status"] == "terminated"
        and status["status_date"]["normalized_date"] == "2026-06-02"
    )
    announced = derived_status_payload(
        [
            {
                "kind": "closing",
                "normalized_date": "2026-07-06",
                "spans": [],
                "derived_from": "stated",
                "prior": False,
                "expected": True,
            }
        ]
    )
    assert announced["status"] == "announced"
    pending = [
        {
            "kind": "retirement",
            "normalized_date": None,
            "spans": [],
            "derived_from": None,
            "prior": False,
            "expected": True,
        }
    ]
    assert derived_status_payload(pending)["status"] is None
    assert expected_retirement_in_payloads(pending) is True
    undated = [
        {
            "kind": "closing",
            "normalized_date": None,
            "spans": [],
            "derived_from": None,
            "prior": False,
            "expected": False,
        },
        {
            "kind": "retirement",
            "normalized_date": None,
            "spans": [],
            "derived_from": None,
            "prior": False,
            "expected": False,
        },
    ]
    assert derived_status_payload(undated)["status"] == "repaid"
    assert (
        derived_status_payload(
            [
                {
                    "kind": "maturity",
                    "normalized_date": "2031-12-31",
                    "spans": [],
                    "derived_from": "name",
                    "prior": False,
                    "expected": False,
                }
            ]
        )["status"]
        is None
    )


def test_parties_list_derives_lender_disclosure() -> None:
    """Stage 2: one parties list, and disclosure distinguishes its three states."""
    from cdt.extractor.core import party_payloads_and_disclosure

    tags = {
        "tag-1": {
            "text": "JPMorgan Chase Bank, N.A.",
            "type": "organization",
            "char_start": 0,
            "char_end": 25,
        },
        "tag-2": {
            "text": "the other lenders party thereto",
            "type": "organization",
            "char_start": 30,
            "char_end": 61,
        },
        "tag-3": {
            "text": "Wells Fargo Bank",
            "type": "organization",
            "char_start": 70,
            "char_end": 86,
        },
        "tag-4": {
            "text": "The Bank of New York Mellon",
            "type": "organization",
            "char_start": 90,
            "char_end": 117,
        },
    }
    parties, disclosure = party_payloads_and_disclosure(
        {
            "parties": [
                {"tag_ids": ["tag-1"], "role": "lender"},
                {"tag_ids": ["tag-2"], "role": "lender", "kind": "collective"},
            ]
        },
        tags,
    )
    assert [(p["role"], p["kind"]) for p in parties] == [
        ("lender", "named"),
        ("lender", "collective"),
    ]
    assert disclosure == "collective_present"
    _, every_lender_named = party_payloads_and_disclosure(
        {
            "parties": [
                {"tag_ids": ["tag-1"], "role": "lender"},
                {"tag_ids": ["tag-3"], "role": "lender", "kind": "named"},
            ]
        },
        tags,
    )
    assert every_lender_named == "complete"
    trustee_only, no_lender = party_payloads_and_disclosure(
        {"parties": [{"tag_ids": ["tag-4"], "role": "trustee"}]}, tags
    )
    assert trustee_only[0]["role"] == "trustee"
    # A trustee-only indenture names nobody who holds the debt. Under the
    # boolean this replaced, that read the same as a collective phrase.
    assert no_lender == "none_named"


def test_current_shape_entry_without_parties_or_dates_is_not_legacy() -> None:
    """An entry omitting both keys is current-schema, so disclosure is derived.

    The prompt tells the model to omit a property the document says nothing
    about, so `{name, instrument_type, amounts}` is an ordinary response.
    Inferring the shape from the *presence* of `parties`/`dates` sent it down
    the legacy path, which published "every lender named" next to an empty
    party list.
    """
    from cdt.extractor.core import party_payloads_and_disclosure

    parties, disclosure = party_payloads_and_disclosure(
        {"name": ["tag-i-1"], "instrument_type": "revolving_credit", "amounts": []},
        {},
    )
    assert parties == []
    assert disclosure == "none_named"


def test_legacy_declared_incompleteness_maps_onto_the_three_values() -> None:
    """A stored response's declared boolean still replays, onto the new field."""
    from cdt.extractor.core import party_payloads_and_disclosure

    tags = {
        "tag-1": {
            "text": "JPMorgan Chase Bank, N.A.",
            "type": "organization",
            "char_start": 0,
            "char_end": 25,
        },
    }
    _, declared = party_payloads_and_disclosure(
        {"lenders": [["tag-1"]], "lenders_known_incomplete": True}, tags
    )
    assert declared == "collective_present"
    _, undeclared = party_payloads_and_disclosure(
        {"lenders": [["tag-1"]], "lenders_known_incomplete": False}, tags
    )
    assert undeclared == "complete"
    _, nobody = party_payloads_and_disclosure(
        {"lenders": [], "lenders_known_incomplete": False}, tags
    )
    assert nobody == "none_named"


def test_aggregate_lender_disclosure_precedence() -> None:
    """Worst-of across an instrument's mentions, `complete` beating `none_named`."""
    from cdt.matcher.core import aggregate_lender_disclosure

    assert aggregate_lender_disclosure(["complete", "none_named"]) == "complete"
    assert (
        aggregate_lender_disclosure(["complete", "collective_present"])
        == "collective_present"
    )
    assert (
        aggregate_lender_disclosure(["none_named", "collective_present"])
        == "collective_present"
    )
    assert aggregate_lender_disclosure(["none_named"]) == "none_named"
    # A missing or unrecognized value cannot invent a complete syndicate list.
    assert aggregate_lender_disclosure([None, "junk"]) == "none_named"


def test_lifecycle_treats_expected_retirement_fact_as_pending() -> None:
    """A planned redemption recorded as an expected date fact keeps the row active."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    target = prepare_mention(
        build_mention_row(
            mention_id="m-t",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-03-04",
            name="5.25% Senior Notes due 2027",
            start_date=None,
            amount=None,
        )
        | {
            "status": None,
            "dates_json": json.dumps(
                [
                    {
                        "kind": "retirement",
                        "normalized_date": None,
                        "expected": True,
                        "prior": False,
                        "spans": [],
                    }
                ]
            ),
        }
    )
    rows = [
        {
            "debt_instrument_id": "inst-t",
            "amendment_of_debt_instrument_id": None,
            "split_of_debt_instrument_id": None,
            "retired_by_debt_instrument_ids": '["inst-new"]',
            "maturity_date": "2027-12-31",
        }
    ]
    apply_lifecycle_rollup(
        rows, member_groups={"inst-t": ["m-t"]}, mention_index={"m-t": target}
    )
    assert rows[0]["status"] == "active"


def test_post_filing_closing_is_expected_and_agreement_supplies_start() -> None:
    """Pilot fixes: a closing dated after the filing is planned; a lone agreement date is the start."""
    from cdt.extractor.core import (
        derived_status_payload,
        mark_post_filing_events_expected,
        normalized_date_from_text,
        select_date_payload,
    )

    facts = [
        {
            "kind": "closing",
            "normalized_date": "2022-04-06",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
        {
            "kind": "announcement",
            "normalized_date": "2022-03-25",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        },
    ]
    mark_post_filing_events_expected(facts, "2022-03-25")
    assert facts[0]["expected"] is True and facts[1]["expected"] is False
    assert select_date_payload(facts, "closing")["normalized_date"] is None
    assert derived_status_payload(facts)["status"] == "announced"
    agreement_only = [
        {
            "kind": "agreement",
            "normalized_date": "2015-02-27",
            "spans": [],
            "derived_from": "stated",
            "prior": False,
            "expected": False,
        }
    ]
    assert (
        select_date_payload(agreement_only, "agreement")["normalized_date"]
        == "2015-02-27"
    )
    status = derived_status_payload(agreement_only)
    assert (
        status["status"] == "entered_into"
        and status["status_date"]["normalized_date"] == "2015-02-27"
    )
    assert normalized_date_from_text("March 5 , 2026") == "2026-03-05"


def _semantic_tags() -> dict[str, dict[str, object]]:
    return {
        "tag-1": {
            "text": "5% Senior Notes due 2031",
            "type": "debt_instrument",
            "char_start": 0,
            "char_end": 24,
        },
        "tag-2": {
            "text": "March 5, 2026",
            "type": "date",
            "char_start": 30,
            "char_end": 43,
        },
        "tag-3": {
            "text": "$100 million",
            "type": "amount",
            "char_start": 50,
            "char_end": 62,
        },
    }


def test_semantic_validators_reject_misplaced_flags_and_kinds() -> None:
    """Stage 2: `expected` only on events, `prior` only on terms, kinds fit the type, repayments pair."""
    from cdt.extractor.core import (
        validate_cross_field_semantics,
        validate_dates_property,
    )

    tags = _semantic_tags()
    expected_maturity = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "maturity",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                    "expected": True,
                }
            ]
        },
        tag_details=tags,
    )
    assert any("cannot be `expected`" in f for f in expected_maturity)
    prior_event = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "retirement",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                    "prior": True,
                }
            ]
        },
        tag_details=tags,
    )
    assert any("cannot be `prior`" in f for f in prior_event)
    ok = validate_dates_property(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                    "expected": True,
                },
                {
                    "kind": "maturity",
                    "evidence": ["tag-1"],
                    "normalized_date": "2031-12-31",
                    "prior": True,
                },
            ]
        },
        tag_details=tags,
    )
    assert ok == []
    prior_balance = validate_cross_field_semantics(
        index=0,
        obj={
            "amounts": [
                {
                    "kind": "outstanding_balance",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                    "prior": True,
                }
            ]
        },
    )
    assert any("cannot be `prior`" in f for f in prior_balance)
    unpaired_date = validate_cross_field_semantics(
        index=0,
        obj={
            "dates": [{"kind": "repayment", "evidence": [], "normalized_date": None}],
            "amounts": [],
        },
    )
    assert any("needs the repaid figure" in f for f in unpaired_date)
    unpaired_amount = validate_cross_field_semantics(
        index=0,
        obj={
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-2"],
                    "normalized_date": "2026-03-05",
                }
            ],
            "amounts": [
                {
                    "kind": "repayment",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                }
            ],
        },
    )
    assert any("is an event" in f for f in unpaired_amount)
    paired = validate_cross_field_semantics(
        index=0,
        obj={
            "dates": [{"kind": "repayment", "evidence": [], "normalized_date": None}],
            "amounts": [
                {
                    "kind": "repayment",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                }
            ],
        },
    )
    assert paired == []
    mismatch = validate_cross_field_semantics(
        index=0,
        obj={
            "instrument_type": "note_bond",
            "amounts": [
                {
                    "kind": "commitment",
                    "evidence": ["tag-3"],
                    "normalized_amount": "100000000",
                }
            ],
        },
    )
    assert any("does not fit instrument_type" in f for f in mismatch)
    assert (
        validate_cross_field_semantics(
            index=0,
            obj={
                "instrument_type": "revolving_credit",
                "amounts": [
                    {
                        "kind": "commitment",
                        "evidence": ["tag-3"],
                        "normalized_amount": "100000000",
                    }
                ],
            },
        )
        == []
    )


def test_live_validation_rejects_legacy_properties_but_replay_accepts_them() -> None:
    """A fresh response reverting to status_event/lenders fails; stored responses still post-process."""
    from cdt.extractor.core import validate_no_legacy_properties

    legacy = {
        "name": ["tag-1"],
        "status_event": {"status": "repaid"},
        "lenders": [],
        "start_date": {"evidence": [], "normalized_date": None},
    }
    failures = validate_no_legacy_properties(0, legacy)
    assert len(failures) == 3 and all(
        "is not a property of this schema" in f for f in failures
    )
    assert (
        validate_no_legacy_properties(
            0, {"name": ["tag-1"], "dates": [], "parties": []}
        )
        == []
    )


def test_relation_manifest_marks_expected_retirement() -> None:
    """The relation stage sees a planned retirement it cannot read off the body tags."""
    from cdt.extractor.core import ExtractionRowState, relation_instrument_manifest

    state = ExtractionRowState(
        item_row={"item_id": "item-1"}, stage_name="instrument_relation"
    )
    state.debt_instrument_mentions = [
        {
            "raw_id": "i-1",
            "name": "New Notes",
            "principal_amount": "600000000",
            "start_date": None,
            "maturity_date": "2036-12-31",
            "status": "announced",
            "dates_json": json.dumps([{"kind": "closing", "expected": True}]),
        },
        {
            "raw_id": "i-2",
            "name": "5.25% Senior Notes due 2027",
            "principal_amount": None,
            "start_date": None,
            "maturity_date": "2027-12-31",
            "status": None,
            "dates_json": json.dumps(
                [{"kind": "retirement", "expected": True, "normalized_date": None}]
            ),
        },
    ]
    manifest = relation_instrument_manifest(state)
    assert 'id="i-2"' in manifest and 'expected_retirement="true"' in manifest
    assert manifest.count('expected_retirement="true"') == 1
    assert 'status="announced"' in manifest


def test_repayment_amount_is_dated_by_a_terminal_event() -> None:
    """A repayment figure beside a retirement needs no separate repayment date."""
    from cdt.extractor.core import validate_cross_field_semantics

    obj = {
        "dates": [
            {
                "kind": "retirement",
                "evidence": ["tag-2"],
                "normalized_date": "2026-03-05",
            }
        ],
        "amounts": [
            {
                "kind": "repayment",
                "evidence": ["tag-3"],
                "normalized_amount": "100000000",
            }
        ],
    }
    assert validate_cross_field_semantics(index=0, obj=obj) == []


def test_table_cells_publish_coupon_and_document_currency() -> None:
    """FHLB schedules: a bare `4.125` under COUPON PCT is the rate; `($)` in the header is the currency."""
    from cdt.extractor.core import (
        currency_candidates_from_text,
        standardized_amount_payload,
        standardized_interest_rate_payload,
    )

    tags = {
        "tag-51": {
            "text": "4.125",
            "type": "interest_rate",
            "char_start": 0,
            "char_end": 5,
        },
        "tag-52": {
            "text": "35,000,000",
            "type": "amount",
            "char_start": 10,
            "char_end": 20,
        },
        "tag-53": {
            "text": "4.125% per annum",
            "type": "interest_rate",
            "char_start": 30,
            "char_end": 46,
        },
    }
    rate = standardized_interest_rate_payload(
        {"kind": "fixed", "rate_pct": "4.125", "evidence": ["tag-51"]},
        tags,
        name_text=None,
    )
    assert rate["rate_pct"] == "4.125" and rate["derived_from"] == "stated"
    assert (
        standardized_interest_rate_payload(
            {"kind": "fixed", "rate_pct": "4.125", "evidence": ["tag-53"]},
            tags,
            name_text=None,
        )["rate_pct"]
        == "4.125"
    )
    doc = frozenset(currency_candidates_from_text("BANK PAR ($)\n35,000,000"))
    assert doc == {"USD"}
    usd = standardized_amount_payload(
        {
            "kind": "principal",
            "evidence": ["tag-52"],
            "normalized_amount": "35000000",
            "currency": "USD",
        },
        tags,
        document_currencies=doc,
    )
    assert usd["currency"] == "USD"
    # No document-level evidence, or two currencies in the document: the model's code is still rejected.
    assert (
        standardized_amount_payload(
            {
                "kind": "principal",
                "evidence": ["tag-52"],
                "normalized_amount": "35000000",
                "currency": "USD",
            },
            tags,
            document_currencies=frozenset(),
        )["currency"]
        is None
    )
    assert (
        standardized_amount_payload(
            {
                "kind": "principal",
                "evidence": ["tag-52"],
                "normalized_amount": "35000000",
                "currency": "USD",
            },
            tags,
            document_currencies=frozenset({"USD", "CAD"}),
        )["currency"]
        is None
    )


def test_fractional_coupons_publish_as_decimal_rates() -> None:
    """`6 1/2%` and `5 7/8% Senior Notes due 2026` are 6.5 and 5.875, not missing rates."""
    from cdt.extractor.core import rate_tokens, standardized_interest_rate_payload

    assert rate_tokens("6 1/2%") == ["6.5"]
    assert rate_tokens("5 7/8 % senior unsecured notes due 2030") == ["5.875"]
    assert rate_tokens("4.125% per annum") == ["4.125"]
    tags = {
        "tag-1": {
            "text": "5 7/8% Senior Notes due 2026",
            "type": "debt_instrument",
            "char_start": 0,
            "char_end": 28,
        }
    }
    payload = standardized_interest_rate_payload(
        {"kind": "fixed", "rate_pct": "5.875", "evidence": ["tag-1"]},
        tags,
        name_text="5 7/8% Senior Notes due 2026",
    )
    assert payload["rate_pct"] == "5.875" and payload["derived_from"] == "name"


# --- Published schema contract -------------------------------------------------
# Nothing referenced these lists, so dropping a column from either silently
# dropped the data: `match_tables` publishes via
# `pd.DataFrame(rows, columns=DEBT_INSTRUMENT_COLUMNS)`.


def test_published_mention_columns_are_pinned() -> None:
    """The mention schema is a contract; a rename must fail here first."""
    assert DEBT_INSTRUMENT_MENTION_COLUMNS == [
        "debt_instrument_mention_id",
        "item_id",
        "accession_number",
        "cik",
        "company_name",
        "date",
        "raw_id",
        "name",
        "instrument_type",
        "start_date",
        "maturity_date",
        "commitment_termination_date",
        "principal_amount",
        "principal_currency",
        "principal_amount_kind",
        "status",
        "status_date",
        "interest_rate_kind",
        "interest_rate_pct",
        "amendment_of",
        "retired_by_json",
        "split_of",
        "parties_json",
        "name_json",
        "start_date_json",
        "maturity_date_json",
        "commitment_termination_date_json",
        "amounts_json",
        "status_json",
        "interest_rate_json",
        "dates_json",
        "lender_disclosure",
    ]


def test_published_instrument_columns_are_pinned() -> None:
    """The instrument schema is what the dashboard publisher reads (#151, #155)."""
    assert DEBT_INSTRUMENT_COLUMNS == [
        "debt_instrument_id",
        "cik",
        "company_name",
        "seed_debt_instrument_mention_id",
        "amendment_of_debt_instrument_id",
        "retired_by_debt_instrument_ids",
        "split_of_debt_instrument_id",
        "superseded_by_debt_instrument_id",
        "lineage_family_id",
        "is_lineage_head",
        "status",
        "status_subtype",
        "status_date",
        "status_source_mention_id",
        "first_seen_filing_date",
        "last_seen_filing_date",
        "mention_count",
        "document_count",
        "name",
        "name_source_mention_id",
        "instrument_type",
        "instrument_type_source_mention_id",
        "start_date",
        "start_date_source_mention_id",
        "maturity_date",
        "maturity_source_mention_id",
        "commitment_termination_date",
        "commitment_termination_source_mention_id",
        "principal_amount",
        "principal_currency",
        "principal_amount_kind",
        "principal_source_mention_id",
        "outstanding_balance",
        "outstanding_balance_currency",
        "outstanding_balance_as_of",
        "outstanding_balance_source_mention_id",
        "interest_rate_kind",
        "interest_rate_pct",
        "interest_rate_source_mention_id",
        "parties_json",
        "lender_disclosure",
        "amendment_inferred_by",
    ]


def test_matcher_schema_version_is_pinned() -> None:
    """The version is how a downstream reader learns a rebuild is required."""
    assert MATCHER_SCHEMA_VERSION == 4


def test_match_tables_publishes_exactly_the_declared_columns() -> None:
    """A column removed from the list would otherwise vanish without a failure."""
    tables = match_tables(
        pd.DataFrame(
            [
                build_mention_row(
                    mention_id="m-1",
                    item_id="item-1",
                    accession_number="0001",
                    cik="0000320193",
                    date="2024-01-02",
                    name="7% Senior Notes due 2030",
                    start_date="2024-01-01",
                    amount="500000000",
                )
            ]
        )
    )
    assert list(tables["debt_instrument"].columns) == DEBT_INSTRUMENT_COLUMNS
    assert (
        list(tables["debt_instrument_mentions"].columns) == MENTION_CLUSTER_EDGE_COLUMNS
    )


def test_published_evidence_spans_index_the_item_text_exactly() -> None:
    """#154's contract, asserted on a published payload rather than the helper.

    Every other span assertion in this suite projects to `tag_id` or `text`, so
    stripping the offsets out of `cluster_payload` entirely went unnoticed.
    """
    item_text = (
        "On March 5, 2026 the Company entered into a $500,000,000 term loan "
        "under the Credit Agreement."
    )
    tagged = (
        '<document>On <date id="tag-d-1">March 5, 2026</date> the Company '
        'entered into a <amount id="tag-a-1">$500,000,000</amount> '
        '<debt_instrument id="tag-i-1">term loan</debt_instrument> under the '
        "Credit Agreement.</document>"
    )
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "text": item_text, "date": "2026-03-10"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = tagged
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "instrument_type": "term_loan",
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "500000000",
                        "currency": "USD",
                    }
                ],
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2026-03-05",
                    }
                ],
                "parties": [],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)
    mention = row_state.debt_instrument_mentions[0]

    checked = 0
    for column in ("name_json", "amounts_json", "dates_json", "start_date_json"):
        payload = json.loads(str(mention[column]))
        for entry in payload if isinstance(payload, list) else [payload]:
            for span in entry.get("spans", []):
                assert (
                    item_text[span["char_start"] : span["char_end"]] == span["text"]
                ), f"{column} span does not index the item text"
                checked += 1
    assert checked >= 3


def test_instrument_rollup_publishes_balance_and_rate_columns() -> None:
    """The seven #140/#157 instrument columns had no test at all."""
    from cdt.matcher.core import build_debt_instrument_rows, prepare_mention

    older = prepare_mention(
        build_mention_row(
            mention_id="m-old",
            item_id="item-1",
            accession_number="0001",
            cik="0000320193",
            date="2026-01-01",
            name="Revolving Credit Facility",
            start_date="2024-01-01",
            amount="300000000",
            interest_rate_kind="fixed",
            interest_rate_pct="7.000",
            amounts_json=json.dumps(
                [
                    {
                        "kind": "outstanding_balance",
                        "normalized_amount": "270500000",
                        "currency": "USD",
                        "as_of_date": None,
                    }
                ]
            ),
        )
    )
    rows = build_debt_instrument_rows(
        {"inst-1": ["m-old"]},
        {"m-old": older},
        {},
        existing_instruments=pd.DataFrame(),
        company_names={},
    )
    row = rows[0]
    assert row["outstanding_balance"] == "270500000"
    assert row["outstanding_balance_currency"] == "USD"
    # An undated balance is bounded by the filing that observed it.
    assert row["outstanding_balance_as_of"] == "2026-01-01"
    assert row["outstanding_balance_source_mention_id"] == "m-old"
    assert row["interest_rate_kind"] == "fixed"
    assert row["interest_rate_pct"] == "7.000"
    assert row["interest_rate_source_mention_id"] == "m-old"
    # A balance is never the headline amount (#140).
    assert row["principal_amount"] == "300000000"


# --- Lifecycle rollup: the branches the hand-built fixtures never reached ------


def _rollup_row(row_id: str, **overrides: object) -> dict[str, object]:
    """Return one bare instrument row for `apply_lifecycle_rollup`."""
    row: dict[str, object] = {
        "debt_instrument_id": row_id,
        "amendment_of_debt_instrument_id": None,
        "split_of_debt_instrument_id": None,
        "retired_by_debt_instrument_ids": None,
        "maturity_date": None,
    }
    row.update(overrides)
    return row


def test_lineage_family_id_is_the_lowest_member_id_not_merely_shared() -> None:
    """Asserting only that a family is *shared* let `min` become `max`."""
    from cdt.matcher.core import apply_lifecycle_rollup

    rows = [
        _rollup_row("zzz-parent"),
        _rollup_row("mmm-child", amendment_of_debt_instrument_id="zzz-parent"),
        _rollup_row("aaa-grandchild", amendment_of_debt_instrument_id="mmm-child"),
    ]
    apply_lifecycle_rollup(rows, member_groups={}, mention_index={})

    assert {row["lineage_family_id"] for row in rows} == {"aaa-grandchild"}


def test_lineage_families_span_retirement_and_split_pointers() -> None:
    """Only amendment edges were exercised, so dropping the other two passed."""
    from cdt.matcher.core import apply_lifecycle_rollup

    retired = _rollup_row(
        "b-retired", retired_by_debt_instrument_ids=json.dumps(["a-retirer"])
    )
    retirer = _rollup_row("a-retirer")
    split_child = _rollup_row("d-split", split_of_debt_instrument_id="c-parent")
    split_parent = _rollup_row("c-parent")
    rows = [retired, retirer, split_child, split_parent]
    apply_lifecycle_rollup(rows, member_groups={}, mention_index={})

    assert retired["lineage_family_id"] == retirer["lineage_family_id"] == "a-retirer"
    assert split_child["lineage_family_id"] == split_parent["lineage_family_id"]
    assert split_child["lineage_family_id"] == "c-parent"
    # A retirement is not an amendment, so neither row is superseded by it.
    assert retired["superseded_by_debt_instrument_id"] is None
    assert retired["is_lineage_head"] is True


def test_two_amendment_children_publish_no_superseded_pointer() -> None:
    """An ambiguous inverse publishes nothing, as the parent pointers do."""
    from cdt.matcher.core import apply_lifecycle_rollup

    parent = _rollup_row("p")
    rows = [
        parent,
        _rollup_row("c1", amendment_of_debt_instrument_id="p"),
        _rollup_row("c2", amendment_of_debt_instrument_id="p"),
    ]
    apply_lifecycle_rollup(rows, member_groups={}, mention_index={})

    assert parent["superseded_by_debt_instrument_id"] is None
    # The rollup still knows the row was replaced, so it is not a live head.
    assert parent["is_lineage_head"] is False


# --- The announced / expected_active split (#183) -----------------------------


def _expected_closing_dates_json(normalized_date: str | None) -> str:
    """Return a dates_json holding one planned closing and nothing else."""
    return json.dumps(
        [
            {
                "kind": "closing",
                "normalized_date": normalized_date,
                "expected": True,
                "prior": False,
                "spans": [],
            }
        ]
    )


def test_expected_dates_reads_planned_starts_and_retirements_apart() -> None:
    """The two kinds of expectation answer different legs, so they cannot merge."""
    from cdt.matcher.core import expected_dates_from_dates_json

    expected = expected_dates_from_dates_json(
        json.dumps(
            [
                # The planned start.
                {"kind": "closing", "normalized_date": "2026-04-01", "expected": True},
                # A prior term is what the instrument used to say, not a plan.
                {
                    "kind": "closing",
                    "normalized_date": "2019-01-01",
                    "expected": True,
                    "prior": True,
                },
                # An occurred closing is not a plan either.
                {"kind": "closing", "normalized_date": "2020-01-01", "expected": False},
                {
                    "kind": "retirement",
                    "normalized_date": "2026-09-30",
                    "expected": True,
                },
                # An event the filing states without a date is still a plan.
                {"kind": "termination", "normalized_date": None, "expected": True},
            ]
        )
    )
    assert expected.start_date == "2026-04-01"
    assert expected.retirement_dates == ("2026-09-30",)
    assert expected.undated_retirement is True
    # Malformed and absent payloads say nothing rather than raising.
    for value in (None, "", "{not json", "[]"):
        empty = expected_dates_from_dates_json(value)
        assert (empty.start_date, empty.retirement_dates, empty.undated_retirement) == (
            None,
            (),
            False,
        )


def test_an_announcement_whose_planned_start_has_not_arrived_is_announced() -> None:
    """Rule 1: announced holds while the planned start is still ahead (#183)."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    announced = prepare_mention(
        build_mention_row(
            mention_id="m-a",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-03-04",
            name="6.0% Senior Notes due 2034",
            start_date=None,
            amount="500000000",
        )
        | {
            "status": "announced",
            "status_date": "2026-03-04",
            "dates_json": _expected_closing_dates_json("2026-03-20"),
        }
    )
    rows = [_rollup_row("inst-a")]
    apply_lifecycle_rollup(
        rows, member_groups={"inst-a": ["m-a"]}, mention_index={"m-a": announced}
    )

    assert rows[0]["status"] == "announced"
    assert rows[0]["status_subtype"] is None
    assert rows[0]["status_date"] == "2026-03-04"
    assert rows[0]["status_source_mention_id"] == "m-a"


def test_an_announcement_with_no_planned_start_stays_announced() -> None:
    """Rule 1's other half: nothing to measure against means nothing to infer."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    announced = prepare_mention(
        build_mention_row(
            mention_id="m-a",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2020-03-04",
            name="Commitment Letter Facility",
            start_date=None,
            amount="500000000",
        )
        | {"status": "announced", "status_date": "2020-03-04"}
    )
    # A corpus that has moved six years on still cannot say this one started.
    other = prepare_mention(
        build_mention_row(
            mention_id="m-o",
            item_id="item-2",
            accession_number="0002",
            cik="320193",
            date="2026-06-01",
            name="Unrelated Revolver",
            start_date="2026-06-01",
            amount="100000000",
        )
    )
    rows = [_rollup_row("inst-a"), _rollup_row("inst-o")]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-a": ["m-a"], "inst-o": ["m-o"]},
        mention_index={"m-a": announced, "m-o": other},
    )

    assert rows[0]["status"] == "announced"


def test_a_planned_start_the_corpus_has_passed_is_expected_active() -> None:
    """Rule 3: past the planned start with no confirming filing (#183).

    Under #155 this row published `announced` forever, which reads as "has not
    happened yet" about an instrument that by its own schedule closed months
    ago.
    """
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    announced = prepare_mention(
        build_mention_row(
            mention_id="m-a",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-01-05",
            name="6.0% Senior Notes due 2034",
            start_date=None,
            amount="500000000",
        )
        | {
            "status": "announced",
            "status_date": "2026-01-05",
            "dates_json": _expected_closing_dates_json("2026-02-01"),
        }
    )
    other = prepare_mention(
        build_mention_row(
            mention_id="m-o",
            item_id="item-2",
            accession_number="0002",
            cik="320193",
            date="2026-06-01",
            name="Unrelated Revolver",
            start_date="2026-06-01",
            amount="100000000",
        )
    )
    rows = [_rollup_row("inst-a"), _rollup_row("inst-o")]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-a": ["m-a"], "inst-o": ["m-o"]},
        mention_index={"m-a": announced, "m-o": other},
    )

    assert rows[0]["status"] == "expected_active"
    assert rows[0]["status_date"] == "2026-02-01"
    # An inferred state cites no mention, because no mention states it.
    assert rows[0]["status_source_mention_id"] is None


def test_an_observed_start_outranks_a_later_announcement() -> None:
    """#169: an announcement must not revert an instrument that already closed.

    The newest decisive event is the announcement, but the instrument has a
    recorded start date behind the corpus, so it is `active`.
    """
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    def mention(mention_id: str, accession: str, date: str, **extra: object) -> object:
        return prepare_mention(
            build_mention_row(
                mention_id=mention_id,
                item_id=f"item-{accession}",
                accession_number=accession,
                cik="320193",
                date=date,
                name="6.0% Senior Notes due 2034",
                start_date="2026-01-20",
                amount="500000000",
            )
            | extra
        )

    closed = mention("m-close", "0001", "2026-01-22", status="entered_into")
    reannounced = mention("m-again", "0002", "2026-03-01", status="announced")
    rows = [_rollup_row("inst-a", start_date="2026-01-20")]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-a": ["m-close", "m-again"]},
        mention_index={"m-close": closed, "m-again": reannounced},
    )

    assert rows[0]["status"] == "active"
    assert rows[0]["status_date"] == "2026-01-20"


def test_an_unstarted_announcement_still_blocks_a_retirement_it_funds() -> None:
    """The announced-retirer guard tracks the *derived* state, not the raw event.

    `announced_instrument_ids` exists so a use-of-proceeds retirement is not
    asserted before the financing closes. Once the financing's own planned
    start is behind the corpus it reads `expected_active`, so it must stop
    blocking — otherwise a row could be held open by an instrument that no
    longer reads `announced` itself.
    """
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    def announcement(date: str, expected_closing: str) -> object:
        return prepare_mention(
            build_mention_row(
                mention_id="m-new",
                item_id="item-2",
                accession_number="0002",
                cik="320193",
                date=date,
                name="New Notes",
                start_date=None,
                amount="500000000",
            )
            | {
                "status": "announced",
                "status_date": date,
                "dates_json": _expected_closing_dates_json(expected_closing),
            }
        )

    old = prepare_mention(
        build_mention_row(
            mention_id="m-old",
            item_id="item-1",
            accession_number="0001",
            cik="320193",
            date="2026-06-01",
            name="5.25% Senior Notes due 2027",
            start_date="2017-01-01",
            amount="500000000",
        )
    )

    def rollup(expected_closing: str) -> str:
        rows = [
            _rollup_row(
                "inst-old",
                retired_by_debt_instrument_ids=json.dumps(["inst-new"]),
                start_date="2017-01-01",
            ),
            _rollup_row("inst-new"),
        ]
        apply_lifecycle_rollup(
            rows,
            member_groups={"inst-old": ["m-old"], "inst-new": ["m-new"]},
            mention_index={
                "m-old": old,
                "m-new": announcement("2026-06-01", expected_closing),
            },
        )
        return str(rows[0]["status"])

    # The financing has not closed: the old notes are still outstanding.
    assert rollup("2026-07-15") == "active"
    # Its planned close is behind the corpus, so the repayment it funds stands.
    assert rollup("2026-05-15") == "closed"


def test_every_derived_status_is_in_the_published_vocabulary() -> None:
    """A typo in one leg would otherwise publish a value no consumer can filter."""
    import inspect
    import re

    from cdt.matcher.core import (
        CLOSED_STATUS_SUBTYPES,
        INSTRUMENT_STATUS_VALUES,
        derive_instrument_status,
    )

    source = inspect.getsource(derive_instrument_status)
    returned = re.findall(r'return "([a-z_]+)", ("[a-z_]+"|None)', source)
    assert returned, "the legs stopped returning literal statuses; update this test"
    for status, subtype in returned:
        assert status in INSTRUMENT_STATUS_VALUES
        if subtype == "None":
            continue
        assert status == "closed"
        assert subtype.strip('"') in CLOSED_STATUS_SUBTYPES
    # Only `closed` carries a cause, and `superseded` is the one non-event cause.
    assert CLOSED_STATUS_SUBTYPES == {
        "repaid",
        "terminated",
        "exchanged",
        "defaulted",
        "superseded",
    }


def test_two_amendment_children_leave_no_unreachable_parent() -> None:
    """A row replaced by two amendments must not read as a live obligation.

    The ambiguous inverse publishes no `superseded_by` pointer, so a status leg
    reading only that pointer left the parent `active` while every
    `is_lineage_head` view excluded it — unreachable and alive at once (#183).
    """
    from cdt.matcher.core import apply_lifecycle_rollup

    parent = _rollup_row("p")
    rows = [
        parent,
        _rollup_row("c1", amendment_of_debt_instrument_id="p"),
        _rollup_row("c2", amendment_of_debt_instrument_id="p"),
    ]
    apply_lifecycle_rollup(rows, member_groups={}, mention_index={})

    assert parent["superseded_by_debt_instrument_id"] is None
    assert (parent["status"], parent["status_subtype"]) == ("closed", "superseded")


def test_first_and_last_seen_span_distinct_filing_dates() -> None:
    """Both fixture mentions shared a date, so a swap was invisible."""
    from cdt.matcher.core import apply_lifecycle_rollup, prepare_mention

    def mention(mention_id: str, accession: str, date: str) -> object:
        return prepare_mention(
            build_mention_row(
                mention_id=mention_id,
                item_id=f"item-{accession}",
                accession_number=accession,
                cik="0000320193",
                date=date,
                name="7% Senior Notes due 2030",
                start_date="2024-01-01",
                amount="500000000",
            )
        )

    rows = [_rollup_row("inst-1")]
    apply_lifecycle_rollup(
        rows,
        member_groups={"inst-1": ["m-a", "m-b", "m-c"]},
        mention_index={
            "m-a": mention("m-a", "0002", "2024-03-01"),
            "m-b": mention("m-b", "0001", "2024-01-15"),
            "m-c": mention("m-c", "0002", "2024-06-30"),
        },
    )
    row = rows[0]
    assert row["first_seen_filing_date"] == "2024-01-15"
    assert row["last_seen_filing_date"] == "2024-06-30"
    assert row["mention_count"] == 3
    # Three mentions, two filings.
    assert row["document_count"] == 2


def test_superseded_wins_over_a_retirement_pointer() -> None:
    """Leg order: no fixture had both, so reordering them passed."""
    from cdt.matcher.core import derive_instrument_status

    row = {
        "debt_instrument_id": "p",
        "superseded_by_debt_instrument_id": "c",
        "retired_by_debt_instrument_ids": json.dumps(["r"]),
        "maturity_date": None,
    }
    status, subtype, _, _ = derive_instrument_status(
        row,
        reference_date="2026-01-01",
        event_result=None,
        announced_instrument_ids=set(),
    )
    assert (status, subtype) == ("closed", "superseded")


def test_a_retirement_by_an_announced_instrument_is_not_yet_repaid() -> None:
    """The announced-retirer guard, reached without `retirement_pending` masking it.

    The existing test's row was already pending, so `retired_by and not
    retirement_pending` short-circuited and this guard never ran.
    """
    from cdt.matcher.core import derive_instrument_status

    row = {
        "debt_instrument_id": "old",
        "superseded_by_debt_instrument_id": None,
        "retired_by_debt_instrument_ids": json.dumps(["new"]),
        "maturity_date": None,
    }
    kwargs = {"reference_date": "2026-01-01", "event_result": None}
    unclosed, _, _, _ = derive_instrument_status(
        row, announced_instrument_ids={"new"}, **kwargs
    )
    assert unclosed == "active"
    closed, subtype, _, _ = derive_instrument_status(
        row, announced_instrument_ids=set(), **kwargs
    )
    assert (closed, subtype) == ("closed", "repaid")


def test_an_undated_planned_retirement_blocks_the_expected_closed_leg() -> None:
    """A plan that can never be shown to have come due keeps the row alive (#183)."""
    from cdt.matcher.core import RetirementExpectation, derive_instrument_status

    row = {
        "debt_instrument_id": "x",
        "superseded_by_debt_instrument_id": None,
        "retired_by_debt_instrument_ids": None,
        "maturity_date": "2020-01-01",
    }
    kwargs = {
        "reference_date": "2026-01-01",
        "event_result": None,
        "announced_instrument_ids": set(),
    }
    assert (
        derive_instrument_status(row, expectation=RetirementExpectation(), **kwargs)[0]
        == "expected_closed"
    )
    assert (
        derive_instrument_status(
            row, expectation=RetirementExpectation(undated=True), **kwargs
        )[0]
        == "active"
    )


def test_a_planned_retirement_stops_blocking_once_the_corpus_passes_it() -> None:
    """The whole point of `expected_closed`: a notice that came due is not `active`.

    Under #155 a dated redemption notice blocked the terminal legs forever, so
    an instrument noticed for redemption in 2024 still published `active` in
    2026. The date has to be re-checked against the reference date, not merely
    latched as a pending flag.
    """
    from cdt.matcher.core import RetirementExpectation, derive_instrument_status

    row = {
        "debt_instrument_id": "x",
        "superseded_by_debt_instrument_id": None,
        "retired_by_debt_instrument_ids": None,
        "maturity_date": None,
    }
    kwargs = {"reference_date": "2026-01-01", "event_result": None}
    ahead = derive_instrument_status(
        row, expectation=RetirementExpectation(dates=("2027-05-01",)), **kwargs
    )
    assert ahead[0] == "active"
    behind = derive_instrument_status(
        row, expectation=RetirementExpectation(dates=("2024-05-01",)), **kwargs
    )
    assert behind[0] == "expected_closed"
    assert behind[2] == "2024-05-01"


def test_the_latest_end_date_governs_expected_closed() -> None:
    """A lapsed commitment does not close a facility still owed to a later maturity."""
    from cdt.matcher.core import RetirementExpectation, derive_instrument_status

    row = {
        "debt_instrument_id": "x",
        "superseded_by_debt_instrument_id": None,
        "retired_by_debt_instrument_ids": None,
        "commitment_termination_date": "2024-06-30",
        "maturity_date": "2029-06-30",
    }
    kwargs = {
        "reference_date": "2026-01-01",
        "event_result": None,
        "expectation": RetirementExpectation(),
    }
    assert derive_instrument_status(row, **kwargs)[0] == "active"
    # With no later maturity, the lapsed commitment is the end date we are past.
    lapsed = dict(row) | {"maturity_date": None}
    status, _, status_date, _ = derive_instrument_status(lapsed, **kwargs)
    assert (status, status_date) == ("expected_closed", "2024-06-30")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known: `event_status_for_instrument` breaks the scan at the newest "
        "entered_into/amended status, so an older mention's expected "
        "retirement never reaches the pending flag and the rollup asserts "
        "`repaid`. Fix is in matcher/core.py, outside this commit's scope."
    ),
)
def test_a_planned_retirement_survives_a_newer_amendment_mention() -> None:
    """The status scan's `break` must not also discard the pending flag.

    A newer `amended` mention stopped the scan before an older mention's
    `expected` retirement was seen, so the rollup asserted `repaid` for a
    retirement the filings say is only planned.
    """
    from cdt.matcher.core import event_status_for_instrument, prepare_mention

    planned = prepare_mention(
        build_mention_row(
            mention_id="m-planned",
            item_id="item-1",
            accession_number="0001",
            cik="0000320193",
            date="2026-06-01",
            name="7% Senior Notes due 2030",
            start_date="2024-01-01",
            amount="500000000",
            dates_json=json.dumps(
                [
                    {
                        "kind": "retirement",
                        "expected": True,
                        "normalized_date": "2026-09-01",
                    }
                ]
            ),
        )
    )
    amended = prepare_mention(
        build_mention_row(
            mention_id="m-amended",
            item_id="item-2",
            accession_number="0002",
            cik="0000320193",
            date="2026-08-01",
            name="7% Senior Notes due 2030",
            start_date="2024-01-01",
            amount="500000000",
            status="amended",
        )
    )
    index = {"m-planned": planned, "m-amended": amended}

    _, pending_alone = event_status_for_instrument(["m-planned"], index)
    assert pending_alone is True
    _, pending_with_amendment = event_status_for_instrument(
        ["m-planned", "m-amended"], index
    )
    assert pending_with_amendment is True


# --- Fixes made in this commit -------------------------------------------------


def test_two_current_closing_dates_are_rejected() -> None:
    """`closing` is singular too, though it is an event kind.

    The cap counted only kinds outside `EVENT_DATE_KINDS`, so `closing` escaped
    it while `maturity`, `agreement` and `commitment_termination` did not —
    contradicting the prompt's own "(validated)" claim. One of the two dates
    then published and the other was silently dropped.
    """
    tags = _dates_tag_details()
    date_ids = [tag_id for tag_id, detail in tags.items() if detail["type"] == "date"][
        :2
    ]
    assert len(date_ids) == 2

    def two_of(kind: str) -> list[str]:
        return validate_dates_property(
            index=0,
            obj={
                "dates": [
                    {"kind": kind, "evidence": [date_ids[0]]},
                    {"kind": kind, "evidence": [date_ids[1]]},
                ]
            },
            tag_details=tags,
        )

    for kind in ("closing", "maturity", "agreement", "commitment_termination"):
        failures = two_of(kind)
        assert any(
            f"'dates' has 2 current entries of kind '{kind}'" in failure
            for failure in failures
        ), f"{kind} should be capped at one current entry"
    # Events genuinely may repeat: two amendments are two amendments.
    assert not any(
        "current entries of kind" in failure for failure in two_of("amendment")
    )
    # A planned closing beside a real one is one *current* closing, not two:
    # `select_date_payload` skips `expected` entries, so counting them rejected
    # Costamare's 6-K, which states both for one facility.
    assert not any(
        "current entries of kind" in failure
        for failure in validate_dates_property(
            index=0,
            obj={
                "dates": [
                    {"kind": "closing", "evidence": [date_ids[0]]},
                    {"kind": "closing", "evidence": [date_ids[1]], "expected": True},
                ]
            },
            tag_details=tags,
        )
    )
    # A `prior` term is likewise not current.
    assert not any(
        "current entries of kind" in failure
        for failure in validate_dates_property(
            index=0,
            obj={
                "dates": [
                    {"kind": "maturity", "evidence": [date_ids[0]]},
                    {"kind": "maturity", "evidence": [date_ids[1]], "prior": True},
                ]
            },
            tag_details=tags,
        )
    )


def test_interest_rate_without_an_evidence_key_is_accepted() -> None:
    """The prompt's own `3.875% senior notes due 2028` example omits `evidence`.

    Rejecting it cost a full retry cycle on a shape postprocess already handles
    by verifying the rate against the instrument's name.
    """
    assert (
        validate_interest_rate(
            index=0,
            obj={"interest_rate": {"kind": "fixed", "rate_pct": "3.875"}},
            tag_details={},
        )
        == []
    )
    # A present-but-wrong evidence value is still a failure.
    assert validate_interest_rate(
        index=0,
        obj={"interest_rate": {"kind": "fixed", "rate_pct": "3.875", "evidence": "x"}},
        tag_details={},
    )


def test_name_derived_principal_is_synthesized_when_no_amount_supplies_one() -> None:
    """#129's fallback was unreachable: its payload had no model value to agree with."""
    assert name_derived_principal_payload("$183.36 million term loan") == {
        "spans": [],
        "normalized_amount": "183360000",
        "currency": "USD",
        "derived_from": "name",
        "kind": "principal",
        "as_of_date": None,
        "prior": False,
    }
    assert (
        name_derived_principal_payload("C$300 million notes due 2033")["currency"]
        == "CAD"
    )
    # A name with no embedded principal synthesizes nothing.
    assert name_derived_principal_payload("Revolving Credit Facility") is None


def test_a_repayment_figure_does_not_suppress_the_name_derived_principal() -> None:
    """The old gate asked "any amount at all", so an unrelated figure hid the name."""
    from cdt.extractor.core import standardized_amounts_payloads

    tags = {
        "tag-i-1": {
            "type": "debt_instrument",
            "text": "$183.36 million term loan",
            "char_start": 0,
            "char_end": 25,
        }
    }
    payloads = standardized_amounts_payloads(
        {
            "name": ["tag-i-1"],
            "amounts": [
                {"kind": "repayment", "evidence": [], "normalized_amount": None}
            ],
        },
        tags,
        name_text="$183.36 million term loan",
    )
    principal = [p for p in payloads if p["kind"] == "principal"]
    assert principal and principal[0]["normalized_amount"] == "183360000"


def test_out_of_range_date_arithmetic_returns_none_instead_of_raising() -> None:
    """These raised out of postprocess, past the driver, killing the whole run.

    `extract_pending_items` catches only `InfrastructureError`, so the exception
    unwound past the failure registry, the mentions write and the audit write.
    """
    assert date_plus_tenor("9999-01-01", (999, "year")) is None
    assert date_plus_tenor("9999-12-31", (999, "day")) is None
    assert normalized_month_year_from_text("notes due January 0000") is None
    # The ordinary cases still work.
    assert date_plus_tenor("2026-05-15", (364, "day")) == "2027-05-14"
    assert normalized_month_year_from_text("matures in March 2056") == "2056-03-31"


def test_canonical_instrument_name_keeps_the_obligation_over_the_agreement() -> None:
    """`ner.md` rule 11: the descriptive phrase must survive the agreement title.

    Longest-span selection did the opposite on 9 of the 476 multi-span names in
    the 2026-09 window, and the published name feeds the matcher's fingerprint.
    """

    def tags(*texts: str) -> dict[str, dict[str, object]]:
        return {
            f"tag-{index}": {
                "type": "debt_instrument",
                "text": text,
                "char_start": 0,
                "char_end": len(text),
            }
            for index, text in enumerate(texts)
        }

    pair = tags("Second Amended and Restated Credit Agreement", "term loan B facility")
    assert canonical_instrument_name(list(pair), pair) == "term loan B facility"

    dip = tags("Super-Priority Senior Secured Priming Credit Agreement", "DIP Facility")
    assert canonical_instrument_name(list(dip), dip) == "DIP Facility"

    # A non-agreement span that names no obligation is not an improvement:
    # preferring it published `Local Currency Addendums` over `Credit Agreement
    # (2025 364-Day Facility)` and `RFA` over `receivables financing agreement`.
    vacuous = tags(
        "Credit Agreement (2025 364-Day Facility)", "Local Currency Addendums"
    )
    assert (
        canonical_instrument_name(list(vacuous), vacuous)
        == "Credit Agreement (2025 364-Day Facility)"
    )
    abbreviation = tags("receivables financing agreement", "RFA")
    assert (
        canonical_instrument_name(list(abbreviation), abbreviation)
        == "receivables financing agreement"
    )

    # An instrument the filing only ever names by its agreement keeps that name.
    only_agreement = tags("Credit Agreement", "Amended Credit Agreement")
    assert (
        canonical_instrument_name(list(only_agreement), only_agreement)
        == "Amended Credit Agreement"
    )
    # With no agreement title in play, the longest span still wins.
    notes = tags("the Notes", "5.875% Senior Notes due 2034")
    assert (
        canonical_instrument_name(list(notes), notes) == "5.875% Senior Notes due 2034"
    )


def test_realign_tag_details_leaves_unalignable_text_untouched() -> None:
    """The documented degradation path: degrade to old offsets, never guess.

    Proceeding with a partial alignment map would emit realigned-but-wrong
    offsets for some tags, which is worse than the stale ones.
    """
    details = {
        "tag-1": {
            "type": "date",
            "text": "March 5, 2026",
            "char_start": 3,
            "char_end": 16,
        }
    }
    # Differs by more than whitespace, so no alignment exists.
    assert (
        realign_tag_details(details, "on March 5, 2026", "on April 5, 2026") is details
    )
    # A span whose recorded offsets include surrounding whitespace still snaps
    # onto the non-whitespace run.
    padded = {
        "tag-1": {
            "type": "date",
            "text": " March 5, 2026 ",
            "char_start": 2,
            "char_end": 17,
        }
    }
    realigned = realign_tag_details(padded, "on  March 5, 2026 .", "on March 5, 2026.")
    span = realigned["tag-1"]
    assert "on March 5, 2026."[span["char_start"] : span["char_end"]] == span["text"]
    assert span["text"] == "March 5, 2026"


def test_validate_parties_property_rejects_every_bad_shape() -> None:
    """The new validator was indistinguishable from absent: `return []` passed.

    The test that looked like its coverage feeds the legacy `lenders` key, so
    its message assertion was satisfied by the older validator instead.
    """
    tags = {
        "tag-p-1": {
            "type": "organization",
            "text": "Acme Bank",
            "char_start": 0,
            "char_end": 9,
        },
        "tag-d-1": {
            "type": "date",
            "text": "March 5, 2026",
            "char_start": 10,
            "char_end": 23,
        },
    }

    def failures(parties: object) -> list[str]:
        return validate_parties_property(
            index=0, obj={"parties": parties}, tag_details=tags
        )

    assert any("must be a list of cluster objects" in f for f in failures({}))
    assert any("must be an object with" in f for f in failures([["tag-p-1"]]))
    assert any(
        "'role' must be one of" in f
        for f in failures([{"tag_ids": ["tag-p-1"], "role": "financier"}])
    )
    assert any(
        "'kind' must be named or collective" in f
        for f in failures(
            [{"tag_ids": ["tag-p-1"], "role": "lender", "kind": "anonymous"}]
        )
    )
    assert any(
        "'tag_ids' must be a list" in f
        for f in failures([{"tag_ids": "tag-p-1", "role": "lender"}])
    )
    assert any(
        "string tag IDs only" in f
        for f in failures([{"tag_ids": [7], "role": "lender"}])
    )
    assert any(
        "unknown tag ID" in f
        for f in failures([{"tag_ids": ["tag-missing"], "role": "lender"}])
    )
    assert any(
        "must be person or organization" in f
        for f in failures([{"tag_ids": ["tag-d-1"], "role": "lender"}])
    )
    # The shape the prompt describes passes.
    assert failures([{"tag_ids": ["tag-p-1"], "role": "lender", "kind": "named"}]) == []


def test_a_salvaged_row_registers_the_salvage_note_not_the_last_stage() -> None:
    """A PARTIAL row's registry entry must say what salvage dropped (#152).

    `_failure_record` read `current_attempt`, so a row salvaged at
    `instrument_ie` that then completed `instrument_relation` published
    `stage: instrument_relation` and the invented error "Unexpected response at
    stage instrument_relation" — naming a stage that succeeded and describing a
    failure that never happened. The one PARTIAL row in the 364-unit 2026-09 run
    published exactly that, so the row here keeps two mentions in order to
    advance past the stage it was salvaged at.
    """
    from cdt.extractor.core import _failure_record, failed_stage_name, handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "accession_number": "0001", "cik": "0000320193"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    # Two valid entries plus one whose name tag is an organization: salvage keeps
    # the pair, which is enough mentions to run the relation stage. The two differ
    # on `instrument_type` so they hash to distinct mention ids rather than
    # collapsing into one.
    advanced = handle_response(
        row_state,
        json.dumps(
            [
                {"name": ["tag-i-1"], "instrument_type": "term_loan"},
                {"name": ["tag-i-1"], "instrument_type": "revolving_credit"},
                {"name": ["tag-o-named"]},
            ]
        ),
        max_attempts=1,
    )
    assert advanced is not None, "salvage should advance to instrument_relation"
    assert len(row_state.debt_instrument_mentions) == 2

    # The relation stage then succeeds, so the row's last attempt is clean.
    assert handle_response(row_state, json.dumps([]), max_attempts=1) is None
    assert row_state.state == "PARTIAL"
    assert row_state.current_attempt.stage_name == "instrument_relation"
    assert not row_state.current_attempt.validation_errors

    record = _failure_record(
        row_state,
        partition_date="2026-09-04",
        shard="0025",
        run_id="run-1",
        backend="batch",
    )
    assert record["state"] == "PARTIAL"
    # The stage salvage fired in, not the last stage the row ran.
    assert record["stage"] == "instrument_ie"
    assert failed_stage_name(row_state) == "instrument_ie"
    assert "Unexpected response" not in str(record["error"])
    assert record["error"] == "; ".join(row_state.salvage_notes)
    assert "dropped 1" in str(record["error"])


def test_a_whole_response_rejection_that_drops_nothing_says_so() -> None:
    """`dropped 0` is a shape rejection, not a loss; the note must not claim one."""
    from cdt.extractor.core import handle_response

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = PARTY_ROLE_XML
    # A bare object, plus a legacy property that only the whole-response check
    # rejects: every entry validates on its own, so nothing is dropped.
    handle_response(
        row_state,
        json.dumps({"name": ["tag-i-1"], "lenders_known_incomplete": True}),
        max_attempts=1,
    )

    assert row_state.state == "PARTIAL"
    assert row_state.salvage_notes
    assert "dropped 0" not in row_state.salvage_notes[0]
    assert "published every entry" in row_state.salvage_notes[0]


def test_normalize_snapshot_text_pads_the_cik_column() -> None:
    """#153's snapshot padding had no test, so disabling it passed."""
    table = pd.DataFrame(
        [
            {"cik": "320193", "company_name": "Example Inc."},
            {"cik": "0000707605", "company_name": "Already Padded Co"},
            {"cik": None, "company_name": "No CIK Co"},
        ]
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["cik"].to_list() == ["0000320193", "0000707605", None]


def test_computed_sum_needs_two_addends_even_when_no_span_matches() -> None:
    """The two guards masked each other: either alone rejected the only case.

    The existing test cites one span whose value equals the model's, so both
    "at least two addends" and "no single span equals the value" reject it. This
    case isolates the first: two spans, neither equal to the model's figure, but
    only one of them parseable.
    """
    from cdt.extractor.core import computed_sum_amount

    tags = {
        "tag-a-1": {
            "type": "amount",
            "text": "$200 million",
            "char_start": 0,
            "char_end": 12,
        },
        "tag-a-2": {
            "type": "amount",
            "text": "an undisclosed amount",
            "char_start": 13,
            "char_end": 34,
        },
        "tag-a-3": {
            "type": "amount",
            "text": "$50 million",
            "char_start": 35,
            "char_end": 46,
        },
    }
    # One parseable addend is not a sum, however many spans are cited.
    assert computed_sum_amount(["tag-a-1", "tag-a-2"], tags, "250000000") is None
    # Two parseable addends that do sum to the model's figure are accepted.
    assert computed_sum_amount(["tag-a-1", "tag-a-3"], tags, "250000000") == "250000000"
    # And the second guard, isolated: two addends, but one already equals the
    # model's value, so this is agreement rather than arithmetic.
    equal_tags = dict(tags)
    equal_tags["tag-a-4"] = {
        "type": "amount",
        "text": "$250 million",
        "char_start": 50,
        "char_end": 62,
    }
    assert computed_sum_amount(["tag-a-1", "tag-a-4"], equal_tags, "250000000") is None


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

    monkeypatch.setattr("cdt.extractor.core.run_extraction_workflow", salvaged_workflow)
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


def test_abbreviated_magnitudes_parse_to_full_amounts() -> None:
    """#182: `mil.`, `mm`, `bn` and `trillion` were unknown to the multiplier table.

    On a cited span the wrong parse merely published null, because `amounts_agree`
    rejected the mismatch. On the name-derived path (#129) there is no model value
    to disagree with, so `Citibank $382.5 mil. Revolving Credit Facility` published
    a principal of 382.5 — six orders of magnitude out.
    """
    assert (
        normalized_amount_from_name("Citibank $382.5 mil. Revolving Credit Facility")
        == "382500000"
    )
    assert normalized_amount_from_name("Syndicated $850.0 mil. Facility") == "850000000"
    assert normalized_amount_from_name("$500mm notes") == "500000000"
    assert normalized_amount_from_name("$1.2 bn facility") == "1200000000"
    # `trillion` matched the name pattern's scale group but multiplied nowhere.
    assert normalized_amount_from_name("$1.5 trillion facility") == "1500000000000"
    # The spelled-out forms and the no-magnitude case are unchanged.
    assert normalized_amount_from_name("$183.36 million term loan") == "183360000"
    assert normalized_amount_from_name("C$300 million notes due 2033") == "300000000"
    assert normalized_amount_from_text("$472,934,000") == "472934000"
    # A magnitude abbreviation cannot match inside a longer word.
    assert normalized_amount_from_text("$5 millions") == "5000000"
