"""File-native stage tests."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from cdt import datasets as cdt_datasets
from cdt import storage as cdt_storage
from cdt.classifier import classifications_root, classify_pending_items
from cdt.classifier import core as classifier_core
from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import (
    completion_registry_root,
    existing_date_shard_partition_ids,
    load_completed_partitions,
    load_row_failures,
    normalize_cik,
    run_manifest_path,
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
from cdt.itemizer.core import ITEM_COLUMNS
from cdt.matcher import (
    debt_instruments_root,
    match_pending_mentions,
    mention_cluster_edges_root,
)
from cdt.matcher.core import (
    DEBT_INSTRUMENT_COLUMNS,
    MATCHER_SCHEMA_VERSION,
    MENTION_CLUSTER_EDGE_COLUMNS,
    _stale_schema_forces_rematch,
    apply_lineage_inference_pass,
    coerce_optional_text,
    company_names_by_cik,
    lender_signature,
    match_tables,
)
from cdt.pipeline import normalize_snapshot_text
from cdt.storage import (
    apply_declared_column_types,
    artifact_exists,
    coerce_dataset_text,
    decimal_column_values,
    get_object_bytes,
    read_dataset,
    read_json_artifact,
    read_table,
    write_json_artifact,
    write_partition_table,
    write_table,
)


class FakeModel:
    """Classifier stub returning a fixed relevant score."""

    def decision_function(self: FakeModel, texts: list[str]) -> list[float]:
        """Return a single strong-positive score."""
        del texts
        return [2.0]


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
            # Production writes the parsed, canonical figure here, never the
            # display text a filing used, so fixtures normalize the same way:
            # `principal_amount` publishes as an exact decimal (#185).
            "principal_amount": normalized_amount_from_text(amount) or amount,
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
        "cdt.extractor.core.run_extraction_workflow",
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


def test_published_mention_rows_is_the_single_publish_seam() -> None:
    """Every publish path reads through one helper, which hands out a copy.

    The live loop, batch finalize, `extract_tables` and the audit record all
    call `published_mention_rows`; a derivation attached there (#203) reaches
    every backend at once. The helper returns a fresh list so a caller that
    extends its result cannot mutate the state persisted to `state.jsonl`.
    """
    from cdt.extractor.core import ExtractionRowState, published_mention_rows

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"}, stage_name="instrument_ie"
    )
    row_state.debt_instrument_mentions = [{"debt_instrument_mention_id": "m-1"}]

    published = published_mention_rows(row_state)
    assert published == [{"debt_instrument_mention_id": "m-1"}]
    published.append({"debt_instrument_mention_id": "m-2"})
    assert row_state.debt_instrument_mentions == [{"debt_instrument_mention_id": "m-1"}]
    assert row_state.to_audit_dict()["debt_instrument_mentions"] == published[:1]


# --- Synthesized prior states (#203) ------------------------------------------


def _fact(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {"spans": [], "derived_from": "stated"}
    base.update(fields)
    return base


def amended_row(
    mention_id: str = "m-amend",
    *,
    item_id: str = "item-1",
    amounts: list[dict[str, object]] | None = None,
    dates: list[dict[str, object]] | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Return one amended instrument the way the IE stage publishes it.

    Default: `reduced commitments from $300,000,000 to $250,000,000` under a
    `Credit Agreement dated as of 2020-02-03`, amended 2024-06-01, maturing
    2029-02-03, 5.25% fixed, borrower and lender named.
    """
    amounts = (
        amounts
        if amounts is not None
        else [
            _fact(
                kind="commitment",
                normalized_amount="300000000",
                currency="USD",
                prior=True,
            ),
            _fact(
                kind="commitment",
                normalized_amount="250000000",
                currency="USD",
                prior=False,
            ),
            _fact(
                kind="outstanding_balance", normalized_amount="100000000", prior=False
            ),
        ]
    )
    dates = (
        dates
        if dates is not None
        else [
            _fact(
                kind="agreement",
                normalized_date="2020-02-03",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2029-02-03",
                prior=False,
                expected=False,
            ),
        ]
    )
    principal = next(
        (
            a
            for a in amounts
            if not a.get("prior") and a.get("kind") in ("commitment", "principal")
        ),
        {},
    )
    row = build_mention_row(
        mention_id=mention_id,
        item_id=item_id,
        accession_number="0002",
        cik="0000320193",
        date="2024-06-01",
        name="Credit Agreement",
        start_date="2020-02-03",
        amount=str(principal.get("normalized_amount") or ""),
        parties_json=json.dumps(
            [
                {"role": "borrower", "canonical_name": "Example Inc.", "spans": []},
                {
                    "role": "lender",
                    "canonical_name": "Bank of America, N.A.",
                    "spans": [],
                },
            ]
        ),
        raw_id="i-1",
        instrument_type="revolving_credit",
        maturity_date="2029-02-03",
        interest_rate_kind="fixed",
        interest_rate_pct="5.25",
        interest_rate_json=json.dumps(_fact(kind="fixed", rate_pct="5.25")),
        amounts_json=json.dumps(amounts, sort_keys=True),
        dates_json=json.dumps(dates, sort_keys=True),
        name_json=json.dumps({"spans": [{"text": "Credit Agreement"}]}),
    )
    if not principal:
        row["principal_amount"] = None
    row.update(overrides)
    return row


def test_mint_builds_the_prior_state_from_the_prior_marked_terms() -> None:
    """The amended object's `prior` terms become the predecessor's current ones.

    The predecessor keeps the agreement's dated-as-of (the same agreement), its
    unchanged maturity and rate marked `inherited`, the borrower and nothing
    the joinder may have changed; the successor is untouched apart from the
    pointer, and its id — which never hashed `amendment_of` — is unchanged.
    """
    from cdt.extractor.core import mint_prior_state_rows

    successor = amended_row()
    counters: dict[str, int] = {}
    published = mint_prior_state_rows([dict(successor)], counters)

    assert counters == {"minted": 1}
    assert len(published) == 2
    after, minted = published
    assert (
        after["debt_instrument_mention_id"] == successor["debt_instrument_mention_id"]
    )
    assert after["amendment_of"] == minted["debt_instrument_mention_id"]
    assert json.loads(str(after["amounts_json"]))[0]["prior"] is True  # no aliasing

    assert minted["synthesized_by"] == "prior_state"
    assert (
        minted["synthesized_from_mention_id"] == successor["debt_instrument_mention_id"]
    )
    assert minted["item_id"] == "item-1"
    assert minted["raw_id"] == "i-1-prior"
    assert minted["name"] == "Credit Agreement"
    assert minted["instrument_type"] == "revolving_credit"
    assert minted["principal_amount"] == "300000000"
    assert minted["principal_amount_kind"] == "commitment"
    assert minted["start_date"] == "2020-02-03"
    assert minted["maturity_date"] == "2029-02-03"
    assert minted["status"] == "entered_into"
    assert minted["amendment_of"] is None
    assert minted["lender_disclosure"] == "none_named"
    assert [p["role"] for p in json.loads(str(minted["parties_json"]))] == ["borrower"]

    amounts = json.loads(str(minted["amounts_json"]))
    assert [
        (a["kind"], a["normalized_amount"], a["prior"], a["derived_from"])
        for a in amounts
    ] == [
        ("commitment", "300000000", False, "stated")
    ]  # the new $250M is gone; the balance observation stays with the successor
    dates = {d["kind"]: d for d in json.loads(str(minted["dates_json"]))}
    assert set(dates) == {"maturity", "agreement"}  # no event kinds
    assert dates["maturity"]["derived_from"] == "inherited"
    assert dates["agreement"]["derived_from"] == "stated"
    assert json.loads(str(minted["interest_rate_json"]))["derived_from"] == "inherited"


def test_mint_refusals_are_counted_and_leave_the_rows_alone() -> None:
    """Each way the trigger can fail is named, and nothing is minted."""
    from cdt.extractor.core import mint_prior_state_rows

    def run(
        rows: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], dict[str, int]]:
        counters: dict[str, int] = {}
        return mint_prior_state_rows([dict(r) for r in rows], counters), counters

    # no prior term at all: not an amendment with a before-figure
    plain, counts = run(
        [
            amended_row(
                amounts=[
                    _fact(kind="commitment", normalized_amount="250000000", prior=False)
                ]
            )
        ]
    )
    assert len(plain) == 1 and counts == {}

    # amendment date only: no origin to place the predecessor at
    rows, counts = run(
        [
            amended_row(
                dates=[
                    _fact(kind="amendment", normalized_date="2024-06-01", prior=False)
                ]
            )
        ]
    )
    assert len(rows) == 1 and counts == {"skipped_no_origin": 1}

    # the relation stage already paired it with a model-emitted predecessor
    rows, counts = run([amended_row(amendment_of="m-model-predecessor")])
    assert len(rows) == 1 and counts == {"skipped_model_paired": 1}

    # a sibling in the item already *is* the predecessor
    sibling = build_mention_row(
        mention_id="m-sibling",
        item_id="item-1",
        accession_number="0002",
        cik="0000320193",
        date="2024-06-01",
        name="Credit Agreement",
        start_date="2020-02-03",
        amount="300000000",
    )
    rows, counts = run([amended_row(), sibling])
    assert len(rows) == 2 and counts == {"skipped_sibling_is_predecessor": 1}

    # two before-values of one kind
    rows, counts = run(
        [
            amended_row(
                amounts=[
                    _fact(kind="commitment", normalized_amount="300000000", prior=True),
                    _fact(kind="commitment", normalized_amount="200000000", prior=True),
                    _fact(
                        kind="commitment", normalized_amount="250000000", prior=False
                    ),
                ]
            )
        ]
    )
    assert len(rows) == 1 and counts == {"skipped_ambiguous_prior": 1}


def test_mint_places_the_predecessor_at_the_right_origin() -> None:
    """A prior agreement is the predecessor's own date and wins outright.

    An origin equal to the amendment date is the restatement's own dated-as-of
    (MPLX: agreement 2019-07-31 == amendment 2019-07-31), so the predecessor
    is minted with no start date rather than a date the filing did not state
    for that state of the facility.
    """
    from cdt.extractor.core import mint_prior_state_rows

    restated = amended_row(
        dates=[
            _fact(
                kind="agreement",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="closing",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="agreement",
                normalized_date="2020-02-03",
                prior=True,
                expected=False,
            ),
        ]
    )
    counters: dict[str, int] = {}
    _, minted = mint_prior_state_rows([restated], counters)
    assert counters == {"minted": 1}
    assert minted["start_date"] == "2020-02-03"
    kinds = [d["kind"] for d in json.loads(str(minted["dates_json"]))]
    assert kinds == ["agreement"]  # the restatement's closing/agreement were not copied

    mplx = amended_row(
        dates=[
            _fact(
                kind="agreement",
                normalized_date="2019-07-31",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2019-07-31",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2024-07-31",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2020-12-04",
                prior=True,
                expected=False,
            ),
        ]
    )
    counters = {}
    _, minted = mint_prior_state_rows([mplx], counters)
    assert counters == {"minted": 1, "minted_no_origin": 1}
    assert minted["start_date"] is None
    assert minted["maturity_date"] == "2020-12-04"
    assert minted["principal_amount"] == "300000000"


def test_minted_no_origin_is_not_counted_when_the_mint_is_then_refused() -> None:
    """The tag names a subset of `minted`, so it cannot outlive a refusal.

    Bumped where the origin was resolved, `minted_no_origin` fired ahead of the
    two guards that still stand between that point and the append. A successor
    whose only origin candidate is its own amendment date, sitting beside the
    sibling that *is* its predecessor, then reported `minted_no_origin: 1` with
    no synthesized row anywhere — a mint that never happened, inside the
    counters this docstring calls the pre-registered yield (#211).
    """
    from cdt.extractor.core import mint_prior_state_rows

    successor = amended_row(
        "m-successor",
        dates=[
            # The only origin candidate is the restatement's own dated-as-of,
            # which is what empties `origin_payloads`.
            _fact(kind="agreement", normalized_date="2024-06-01", prior=False),
            _fact(kind="amendment", normalized_date="2024-06-01", prior=False),
        ],
    )
    # The model returned the predecessor as its own object: it carries the
    # successor's prior commitment and states no start date of its own.
    predecessor = amended_row(
        "m-predecessor",
        amounts=[],
        dates=[],
        principal_amount="300000000",
        start_date=None,
    )

    counters: dict[str, int] = {}
    published = mint_prior_state_rows([successor, predecessor], counters)

    assert counters == {"skipped_sibling_is_predecessor": 1}
    assert not [row for row in published if row.get("synthesized_by") == "prior_state"]


def test_mint_from_a_prior_maturity_alone_inherits_the_amount() -> None:
    """`extended the maturity from 2029 to 2031`: the commitment is unchanged."""
    from cdt.extractor.core import mint_prior_state_rows

    extended = amended_row(
        amounts=[
            _fact(
                kind="commitment",
                normalized_amount="250000000",
                currency="USD",
                prior=False,
            )
        ],
        dates=[
            _fact(
                kind="agreement",
                normalized_date="2020-02-03",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="amendment",
                normalized_date="2024-06-01",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2031-02-03",
                prior=False,
                expected=False,
            ),
            _fact(
                kind="maturity",
                normalized_date="2029-02-03",
                prior=True,
                expected=False,
            ),
        ],
    )
    _, minted = mint_prior_state_rows([extended])
    assert minted["maturity_date"] == "2029-02-03"
    assert minted["principal_amount"] == "250000000"
    amounts = json.loads(str(minted["amounts_json"]))
    assert amounts[0]["derived_from"] == "inherited"


def test_two_successors_sharing_one_prior_state_point_at_one_mint() -> None:
    """Byte-identical mints collapse to one row; both pointers still land."""
    from cdt.extractor.core import mint_prior_state_rows

    first = amended_row("m-a", raw_id="i-1")
    second = amended_row("m-b", raw_id="i-1", interest_rate_pct="5.25")
    counters: dict[str, int] = {}
    published = mint_prior_state_rows([first, second], counters)
    assert counters == {"minted": 1, "minted_shared": 1}
    assert len(published) == 3
    assert (
        published[0]["amendment_of"]
        == published[1]["amendment_of"]
        == published[2]["debt_instrument_mention_id"]
    )


def test_mint_is_id_stable_and_idempotent() -> None:
    """Model-emitted ids never change, and minting its own output adds nothing."""
    from cdt.extractor.core import mint_prior_state_rows

    rows = [
        amended_row(),
        amended_row(
            "m-plain",
            amounts=[_fact(kind="commitment", normalized_amount="1", prior=False)],
        ),
    ]
    before = {r["debt_instrument_mention_id"] for r in rows}
    published = mint_prior_state_rows([dict(r) for r in rows])
    real_after = {
        r["debt_instrument_mention_id"]
        for r in published
        if r.get("synthesized_by") is None
    }
    assert real_after == before

    again = mint_prior_state_rows([dict(r) for r in published])
    assert again == published


def test_backfill_mints_over_existing_partitions_and_is_a_no_op_twice(
    tmp_path: Path,
) -> None:
    """A partition written before #203 gains its prior states, once."""
    from cdt.extractor.core import (
        ExtractionRowState,
        backfill_mentions,
        published_mention_rows,
    )

    successor = amended_row()
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-06-01", "shard": "0001"},
        table=pd.DataFrame([successor], columns=DEBT_INSTRUMENT_MENTION_COLUMNS),
    )

    dry = backfill_mentions(tmp_path, dry_run=True)
    assert dry == {"partitions": 1, "partitions_rewritten": 0, "minted": 1}
    assert len(read_dataset(tmp_path / "mentions")) == 1

    first = backfill_mentions(tmp_path)
    assert first == {"partitions": 1, "partitions_rewritten": 1, "minted": 1}
    published = read_dataset(tmp_path / "mentions").sort_values(
        "debt_instrument_mention_id"
    )
    assert len(published) == 2
    assert published["synthesized_by"].notna().sum() == 1

    second = backfill_mentions(tmp_path)
    assert second["minted"] == 1
    again = read_dataset(tmp_path / "mentions").sort_values(
        "debt_instrument_mention_id"
    )
    pd.testing.assert_frame_equal(
        published.reset_index(drop=True), again.reset_index(drop=True)
    )

    # A mint built at write time and one built from the parquet round trip
    # must be the same row: parquet reads None back as NaN, and every copied
    # field is coerced so the hash does not notice.
    row_state = ExtractionRowState(
        item_row={"item_id": "item-1"}, stage_name="instrument_ie"
    )
    row_state.debt_instrument_mentions = [successor]
    write_time = {
        r["debt_instrument_mention_id"] for r in published_mention_rows(row_state)
    }
    assert write_time == set(published["debt_instrument_mention_id"])


def test_backfill_renews_the_writer_lease_once_per_rewritten_partition(
    tmp_path: Path,
) -> None:
    """A whole-dataset rewrite must keep renewing, or it outlives its lease.

    The CLI hands `backfill_mentions` a renewal callback, and a test pins that
    it does. Nothing pinned that the function ever calls it: deleting the
    `renew()` block left the whole suite green, so the #89 guard could be
    removed without a single failure. Same shape as the seams this branch
    exists to close — both halves pinned, the connection not (#211).
    """
    from cdt.extractor.core import backfill_mentions

    for date, mention_id in (("2024-06-01", "m-june"), ("2024-07-01", "m-july")):
        write_partition_table(
            tmp_path / "mentions",
            partition={"date": date, "shard": "0001"},
            table=pd.DataFrame(
                [amended_row(mention_id)], columns=DEBT_INSTRUMENT_MENTION_COLUMNS
            ),
        )

    # A dry run writes nothing, so it takes no lease and must not renew one.
    renewals: list[int] = []
    dry = backfill_mentions(tmp_path, dry_run=True, renew=lambda: renewals.append(1))
    assert dry["partitions_rewritten"] == 0
    assert renewals == []

    counts = backfill_mentions(tmp_path, renew=lambda: renewals.append(1))

    assert counts["partitions_rewritten"] == 2
    assert len(renewals) == counts["partitions_rewritten"]


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
    """Bare tag-id lists and unknown roles should fail validation."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "parties": [
                    ["tag-o-named"],
                    {"tag_ids": ["tag-o-agent"], "role": "servicer"},
                ],
            }
        ]
    )

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "must be an object with 'tag_ids' and 'role'" in failure for failure in failures
    )
    assert any("'role' must be one of" in failure for failure in failures)


def test_instrument_ie_validate_rejects_non_boolean_lenders_known_incomplete() -> None:
    """The retired lenders_known_incomplete flag is rejected outright."""
    response = json.dumps([{"name": ["tag-i-1"], "lenders_known_incomplete": "yes"}])

    failures = InstrumentIEStage().validate(party_row_state(), response)

    assert any(
        "'lenders_known_incomplete' is not a property of this schema" in failure
        for failure in failures
    )


def test_instrument_ie_postprocess_persists_every_party_with_role_and_kind() -> None:
    """Lenders, collective phrases, and other parties all persist labelled (#150)."""
    mention = instrument_ie_mention(
        json.dumps(
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
                    "parties": [
                        {"tag_ids": ["tag-o-named"], "role": "lender", "kind": "named"}
                    ],
                }
            ]
        )
    )

    assert mention["lender_disclosure"] == "complete"
    assert len(json.loads(str(mention["parties_json"]))) == 1


def test_instrument_ie_postprocess_keeps_collective_lenders_and_flags() -> None:
    """A collective-only lender list persists with its surface text and flags (#150)."""
    mention = instrument_ie_mention(
        json.dumps(
            [
                {
                    "name": ["tag-i-1"],
                    "parties": [
                        {
                            "tag_ids": ["tag-o-collective"],
                            "role": "lender",
                            "kind": "collective",
                        }
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
                    "parties": [
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
    payload = json.dumps(
        [{"role": "lender", "spans": [{"text": "Purchasers"}, {"text": "Oaktree"}]}]
    )

    assert lender_signature(payload) == "oaktree"


def test_lender_signature_uses_stored_lender_clusters() -> None:
    """Lender signatures come from the persisted named clusters."""
    payload = json.dumps([{"role": "lender", "spans": [{"text": "Acme Bank"}]}])

    assert lender_signature(payload) == "acme bank"


def test_party_dedupe_trusts_the_extractors_canonical_name() -> None:
    """Two clusters the extractor named alike are one party, whatever they span.

    Re-deriving the key from the spans normalized `EQT Corporation` down to
    `eqt` before choosing the longest text, so its own `Buyer Parent` alias won
    and the two clusters below stayed apart as two lenders (#203).
    """
    from cdt.matcher.core import dedupe_party_clusters

    named_with_alias = {
        "role": "lender",
        "canonical_name": "EQT Corporation",
        "spans": [{"text": "EQT Corporation"}, {"text": "Buyer Parent"}],
    }
    named_alone = {
        "role": "lender",
        "canonical_name": "EQT Corporation",
        "spans": [{"text": "EQT Corporation"}],
    }
    deduped = dedupe_party_clusters(
        [json.dumps([named_with_alias]), json.dumps([named_alone])]
    )
    assert len(deduped) == 1


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
                    "dates": [
                        {
                            "kind": "maturity",
                            "evidence": ["tag-i-1"],
                            "normalized_date": "2028-12-31",
                        },
                    ],
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
                    "dates": [
                        {
                            "kind": "maturity",
                            "evidence": ["tag-d-1"],
                            "normalized_date": "2028-12-31",
                        },
                    ],
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


def test_a_basis_point_margin_is_rate_like_however_the_filing_abbreviates_it() -> None:
    """`bps` is the spelling filings use, and it was not in the marker set (#228).

    `RATE_SUFFIX_PATTERN` carried the spelled-out `basis points` but not the
    abbreviation, so `50 bps` read as a money amount and
    `validate_amount_is_not_rate` passed a basis-point margin through as an
    `amount`. `amounts_agree` does not catch it downstream either: it only
    rejects a model figure that disagrees with the number in the cited span,
    and the model reports the basis-point figure itself, so the two agree and
    the margin publishes as a principal of 50. A wrong published value, not the
    silent null of #182.

    Distinct from #75/#102, which was the whitespace that hid the multi-word
    marker from the predicate. That one is fixed, and the spelled-out assertions
    above still pin it; this is the vocabulary rather than the normalization, so
    it would have been present even with #102 perfect.
    """
    for spelling in ("50 bps", "50 bp", "50 BPS", "50 basis points"):
        assert is_rate_like_amount_text(spelling) is True, spelling
    # The shapes a margin is actually written in.
    assert is_rate_like_amount_text("L+250 bps") is True
    assert is_rate_like_amount_text("a margin of 275 bps") is True

    # ... and the validator that depends on it now refuses the amount.
    failures = validate_amount_is_not_rate(
        index=0,
        value={"evidence": ["tag-1"]},
        tag_details={"tag-1": {"text": "50 bps"}},
    )
    assert len(failures) == 1
    assert "'50 bps'" in failures[0]

    # `bps?\b` must not swallow a real figure: the word boundary keeps it off
    # longer words, and a principal still reads as a principal.
    assert is_rate_like_amount_text("500,000 bpd of crude") is False
    assert is_rate_like_amount_text("$500.0 million") is False
    assert is_rate_like_amount_text("1,500,000") is False


def test_instrument_ie_validate_rejects_rate_only_amount_evidence() -> None:
    """An amount citing only a rate should fail validation and retry."""
    response = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-rate"],
                        "normalized_amount": "0.875",
                        "currency": None,
                    }
                ],
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
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-rate"],
                        "normalized_amount": "0.875",
                        "currency": "USD",
                    }
                ],
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


# --------------------------------------------------------------------------- #
# #176: a NER retry must not pass by returning the input untagged
# --------------------------------------------------------------------------- #

# MPLX item 2.03 (`000119312519257376-2-03`) in miniature: one note series the
# model tags, in an item whose verbatim reproduction it then gets wrong.
MPLX_TEXT = "The Company issued 6.250% Senior Notes due 2022."
MPLX_TAGGED_BUT_UNFAITHFUL = (
    "<body>The Company issued <debt_instrument>6.250% Senior Notes due "
    "2022</debt_instrument>!</body>"
)
MPLX_IE = json.dumps([{"name": ["tag-1"]}])
# The same tagging with the text left exactly as it was given.
MPLX_TAGGED = (
    "<body>The Company issued <debt_instrument>6.250% Senior Notes due "
    "2022</debt_instrument>.</body>"
)


def _ner_row(text: str) -> ExtractionRowState:
    return ExtractionRowState(
        item_row={"item_id": "item-1", "text": text}, stage_name="ner"
    )


def test_ner_validate_rejects_a_zero_tag_retry_after_an_earlier_attempt_tagged() -> (
    None
):
    """The high-water mark: dropping every tag on retry is a failure (#176)."""
    from cdt.extractor.core import AttemptRecord

    row_state = _ner_row(MPLX_TEXT)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=1,
            response=MPLX_TAGGED_BUT_UNFAITHFUL,
            status="FAILED",
        )
    )

    # Valid, faithful, and carrying no `debt_instrument` -- it passes all six
    # structural checks. Tagged with a `date` so it is not also a byte echo,
    # which isolates the high-water check from the echo check.
    failures = NERStage().validate(
        row_state,
        "<body>The Company issued 6.250% Senior Notes due <date>2022</date>.</body>",
    )

    assert failures
    assert any("no <debt_instrument> tags" in failure for failure in failures)
    # The message quotes the count so the retry turn names what was lost.
    assert any("tagged 1" in failure for failure in failures)


def test_ner_validate_high_water_never_misfires_on_a_debt_free_item() -> None:
    """An item with no debt found none on attempt 1 either, so nothing regresses (#176).

    This is the check that makes the high-water mark safe to apply
    unconditionally: it is a comparison against the row's own history, not a
    floor on how many tags a response must carry.
    """
    from cdt.extractor.core import AttemptRecord

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=1,
            response="not xml at all",
            status="FAILED",
        )
    )
    response = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )

    assert NERStage().validate(row_state, response) == []


def test_ner_high_water_counts_tags_in_attempts_that_never_parsed() -> None:
    """A truncated prior attempt still proves the model was tagging (#176, #127).

    11 of 27 NER failures on the PR #57 window were `Response is not valid
    XML`, so a high-water mark built on `parse_tag_details` would read zero for
    exactly the attempts that matter most.
    """
    from cdt.extractor.core import (
        AttemptRecord,
        count_debt_instrument_tags,
        prior_debt_instrument_high_water,
    )

    truncated = (
        "<body>The Company issued <debt_instrument>6.250% Senior Notes"
        "</debt_instrument> and <debt_instrument>5.250% Notes"
    )
    assert count_debt_instrument_tags(truncated) == 2

    row_state = _ner_row(MPLX_TEXT)
    row_state.all_attempts.append(
        AttemptRecord(stage_name="ner", response=truncated, status="FAILED")
    )
    # Another stage's attempts are not this stage's history.
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="instrument_ie",
            response="<debt_instrument><debt_instrument><debt_instrument>",
            status="FAILED",
        )
    )

    assert prior_debt_instrument_high_water(row_state, "ner") == 2


def test_ner_validate_rejects_a_byte_identical_echo_after_the_model_tagged() -> None:
    """Returning the input verbatim is a regression against the model's own work (#176).

    MPLX attempt 3 was byte-identical to the model's own input, `<body>`
    wrapper included, and passed. Driven through `handle_response` rather than
    hand-setting `attempt_index`: what makes the echo a give-up is that an
    earlier attempt on this row tagged something, and only a driven row builds
    that history the way production does.
    """
    from cdt.extractor.core import handle_response, ner_input_body

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)

    echo = ner_input_body(row_state)
    assert echo == f"<body>{MPLX_TEXT}</body>"

    failures = NERStage().validate(row_state, echo)

    assert failures
    assert any("drops every one of them" in failure for failure in failures)


def test_ner_validate_rejects_an_echo_that_drops_non_debt_tags() -> None:
    """Giving up is giving up even when the item has no debt (#176).

    The high-water mark asks the narrower question that drives `early_stop`:
    did an earlier attempt find a `debt_instrument`? A response that tagged an
    organization and a date and then regressed to a bare echo reads zero there,
    but the model has still discarded everything it found.
    """
    from cdt.extractor.core import handle_response

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    # Tags an organization and a date, and fails only copy fidelity.
    unfaithful = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>!</body>"
    )
    assert handle_response(row_state, unfaithful, max_attempts=3)

    failures = NERStage().validate(row_state, f"<body>{text}</body>")

    assert failures
    assert any("drops every one of them" in failure for failure in failures)


def test_ner_validate_accepts_an_untagged_first_attempt() -> None:
    """On attempt 1 an untagged echo is the honest answer for a debt-free item (#176).

    The echo check is gated on earlier tagging for this reason, and was never
    made unconditional for it. Measured over the three
    stored corpora carrying attempt logs (`genwindow-run-branch`,
    `genwindow-run-dev`, `genwindow-sol-retried`): of 761 attempt-1 NER
    responses, zero were byte-identical echoes, and the 63 with no
    `debt_instrument` tag all carried some other tag. The only echo in the
    corpus is MPLX's attempt 3.

    Driven through `handle_response` rather than calling `validate` directly,
    because the boundary is exactly where `add_response` leaves
    `attempt_index`: 1 on a first attempt, not 0. A direct `validate` call on a
    freshly built row state sees 0, so it would pass an off-by-one guard that
    rejects every genuine first attempt.
    """
    from cdt.extractor.core import handle_response

    text = "This is the extracted event text."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    assert handle_response(row_state, f"<body>{text}</body>", max_attempts=3) is None

    assert row_state.all_attempts[0].attempt_index == 1
    assert row_state.all_attempts[0].validation_errors == []
    # An item with nothing to tag is a clean zero, not a give-up.
    assert row_state.state == "SUCCESS"
    assert row_state.debt_instrument_mentions == []


def test_an_untagged_echo_is_accepted_after_a_failure_that_found_nothing() -> None:
    """The regression the old `attempt_index > 1` gate caused (#176).

    A debt-free item whose first attempt failed for a reason unrelated to
    tagging -- malformed XML here -- answers honestly with a bare echo on its
    second. The old gate rejected that answer on every remaining attempt and
    the row died FAILED after three whole-item calls -- the stage's whole
    budget -- losing the item. Nothing
    about the first attempt suggests the model can find anything here, so
    there is no earlier work for the echo to regress against.

    Nothing on this row is evidence of a give-up, so it finishes SUCCESS. The
    zero used to be filed as a possible loss instead; see
    `_advance_after_stage` for why that was dropped.
    """
    from cdt.extractor.core import (
        PUBLISHABLE_ROW_STATES,
        handle_response,
        published_mention_rows,
    )

    text = "This is the extracted event text."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    assert handle_response(row_state, "not xml at all", max_attempts=3)
    assert handle_response(row_state, f"<body>{text}</body>", max_attempts=3) is None

    # Two calls, not three, and the echo itself was accepted.
    assert len([a for a in row_state.all_attempts if a.response is not None]) == 2
    assert row_state.all_attempts[0].status == "FAILED"
    assert row_state.all_attempts[-1].validation_errors == []
    assert row_state.all_attempts[-1].status == "SUCCESS"
    assert row_state.state == "SUCCESS"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    assert published_mention_rows(row_state) == []
    assert row_state.salvage_notes == []


def test_ner_high_water_is_the_most_any_attempt_found_not_the_least() -> None:
    """The mark is the *maximum* across attempts, which is what makes it a guard (#176).

    MPLX is the shape that needs it: attempt 1 tagged 91 spans, attempt 2 came
    back malformed and tagged none, and attempt 3 was the untagged echo. Taking
    the minimum -- or just the last -- would read zero off attempt 2, switch the
    guard off, and let attempt 3 publish as a clean zero, which is the defect
    #176 describes, verbatim.
    """
    from cdt.extractor.core import AttemptRecord, prior_debt_instrument_high_water

    row_state = _ner_row(MPLX_TEXT)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=1,
            response=MPLX_TAGGED_BUT_UNFAITHFUL,
            status="FAILED",
        )
    )
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner",
            attempt_index=2,
            response="not xml at all",
            status="FAILED",
        )
    )

    # Tag counts across this row's history are 1 then 0: the max is 1, while
    # the min and the most recent are both 0.
    assert prior_debt_instrument_high_water(row_state, "ner") == 1


def test_ner_give_up_check_is_not_escaped_by_whitespace() -> None:
    """Perturbing the whitespace must not buy a give-up a pass (#176).

    The check was first written as "byte-identical to the input", and every
    one of these clears that comparison: a trailing newline, a space after
    `<body>`, newlines inside it, a space in the opening tag, doubled
    inter-word spacing. None of them clears copy fidelity either way, because
    that check runs on `collapse_whitespace`, so each one used to reach
    `early_stop` as an accepted zero-tag response. Asking for a tag count
    instead makes the whole family unreachable rather than enumerable.
    """
    from cdt.extractor.core import AttemptRecord

    for response in (
        f"<body>{MPLX_TEXT}</body>",
        f"<body>{MPLX_TEXT}</body>\n",
        f"<body> {MPLX_TEXT}</body>",
        f"<body>\n{MPLX_TEXT}\n</body>",
        f"<body >{MPLX_TEXT}</body>",
        "<body>" + MPLX_TEXT.replace(" ", "  ") + "</body>",
    ):
        row_state = _ner_row(MPLX_TEXT)
        row_state.all_attempts.append(
            AttemptRecord(
                stage_name="ner",
                attempt_index=1,
                response=MPLX_TAGGED_BUT_UNFAITHFUL,
                status="FAILED",
            )
        )

        failures = NERStage().validate(row_state, response)

        assert any(
            "drops every one of them" in failure for failure in failures
        ), response


def test_ner_high_water_accepts_a_reduced_but_nonzero_tag_count() -> None:
    """The high-water check is exact-zero by design, and the boundary is deliberate.

    A model that merges two adjacent spans into one has not given up, and no
    false-positive rate has been measured for any ratio threshold. Pinned so
    that tightening this to "fewer tags than before" is a visible decision
    rather than a quiet one (#176).
    """
    from cdt.extractor.core import AttemptRecord

    text = "The Company issued 6.250% Notes due 2022 and 5.250% Notes due 2025."
    two_tags = (
        "<body>The Company issued <debt_instrument>6.250% Notes due "
        "2022</debt_instrument> and <debt_instrument>5.250% Notes due "
        "2025</debt_instrument>.</body>"
    )
    one_tag = (
        "<body>The Company issued <debt_instrument>6.250% Notes due 2022 and "
        "5.250% Notes due 2025</debt_instrument>.</body>"
    )
    row_state = _ner_row(text)
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner", attempt_index=1, response=two_tags, status="FAILED"
        )
    )

    assert NERStage().validate(row_state, one_tag) == []


def test_prior_attempt_tagged_reads_the_row_the_way_the_echo_guard_needs() -> None:
    """The three properties the echo gate rests on (#176).

    Mirrors `test_ner_high_water_counts_tags_in_attempts_that_never_parsed`
    for the wider any-tag question the echo check asks.
    """
    from cdt.extractor.core import (
        AttemptRecord,
        count_ner_entity_tags,
        prior_attempt_tagged,
    )

    # 1. `<body>` is the wrapper this stage supplies, not something the model
    #    found, so an untagged echo must not count as having tagged anything.
    assert count_ner_entity_tags("<body>plain text</body>") == 0
    # 2. A truncated attempt that never parsed still proves the model was
    #    tagging, which is why this counts by regex rather than by parse.
    assert count_ner_entity_tags("<body>x <debt_instrument>Term Loa") == 1
    assert count_ner_entity_tags("not xml at all") == 0

    row_state = _ner_row(MPLX_TEXT)
    # An earlier attempt that wrapped the input and tagged nothing.
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="ner", response="<body>wrong text</body>", status="FAILED"
        )
    )
    assert prior_attempt_tagged(row_state, "ner") is False

    # 3. Another stage's attempts are not this stage's history.
    row_state.all_attempts.append(
        AttemptRecord(
            stage_name="instrument_ie",
            response="<organization>A</organization>",
            status="FAILED",
        )
    )
    assert prior_attempt_tagged(row_state, "ner") is False

    row_state.all_attempts.append(
        AttemptRecord(stage_name="ner", response="<body>x <date>2022", status="FAILED")
    )
    assert prior_attempt_tagged(row_state, "ner") is True


def test_an_echo_is_accepted_when_the_earlier_attempt_also_tagged_nothing() -> None:
    """A wrapper is not a tag: the `<body>` the stage supplies proves nothing (#176).

    Attempt 1 wraps the input and tags nothing, failing only copy fidelity. It
    gives no evidence the model can find anything in this item, so attempt 2's
    honest echo is still the honest answer.
    """
    from cdt.extractor.core import handle_response

    text = "This is the extracted event text."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    assert handle_response(
        row_state, "<body>wrong text entirely</body>", max_attempts=3
    )
    assert handle_response(row_state, f"<body>{text}</body>", max_attempts=3) is None

    assert row_state.all_attempts[-1].validation_errors == []
    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_a_give_up_is_retried_and_the_row_recovers_if_a_later_answer_passes() -> None:
    """A give-up spends an attempt, it does not end the row (#176).

    The give-up check is an ordinary validation failure, so the model is told
    what was wrong and asked again, and a row that answers properly on the
    next attempt finishes SUCCESS with its mentions -- the same as any other
    row that needed a retry. Only running out of attempts ends it.
    """
    row_state, client = _run_live(
        MPLX_TEXT,
        [
            # Tagged, but it rewrote the text: fails copy fidelity.
            _stopped(MPLX_TAGGED_BUT_UNFAITHFUL),
            # Gives up -- drops every tag it had found.
            _stopped(f"<body>{MPLX_TEXT}</body>"),
            # Then answers properly, and the row carries on to instrument_ie.
            _stopped(MPLX_TAGGED),
            _stopped(MPLX_IE),
        ],
    )

    assert len(client.requests) == 4
    assert row_state.state == "SUCCESS"
    assert len(row_state.debt_instrument_mentions) == 1
    assert row_state.salvage_notes == []
    # The give-up was scored as a failed attempt, not as a terminal verdict.
    statuses = [a.status for a in row_state.all_attempts if a.stage_name == "ner"]
    assert statuses == ["FAILED", "FAILED", "SUCCESS"]


def test_an_echo_after_a_truncated_tagged_attempt_is_still_rejected() -> None:
    """A response cut mid-tag still proves the model was tagging (#176, #127)."""
    from cdt.extractor.core import handle_response

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    # Truncated mid-name, so it never parses -- a parse-based count reads zero.
    truncated = "<body>The Company issued <debt_instrument>6.250% Senior No"
    assert handle_response(row_state, truncated, max_attempts=3)

    failures = NERStage().validate(row_state, f"<body>{MPLX_TEXT}</body>")

    assert failures
    assert any("drops every one of them" in failure for failure in failures)


def test_ner_retry_message_tells_the_model_to_keep_its_tags() -> None:
    """The retry turn must ask for a repair, not invite a redo (#176)."""
    message = NERStage().build_retry_message(["stripped text mismatch"])

    assert "Keep every tag from your previous output" in message
    assert "untagged is not a valid fix" in message
    # The preserve clause leads, ahead of the copy-fidelity bullets that the
    # model previously satisfied by discarding its work.
    assert message.index("Keep every tag") < message.index(
        "Return the original input text exactly"
    )


def test_mplx_untagged_echo_no_longer_publishes_as_a_clean_success() -> None:
    """End to end on #176's headline case: a counted loss, not a silent zero.

    Before this, attempts 1 and 2 failed the copy-fidelity check with 91
    `debt_instrument` spans each, attempt 3 returned the input byte for byte,
    validated clean, and the row published `SUCCESS` with 0 mentions against
    six note series and a term loan -- writing a completion record that makes a
    re-run skip the item.
    """
    from cdt.extractor.core import handle_response, summarize_failure

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)

    # Attempts 1 and 2 tag densely and fail only copy fidelity, as MPLX's did.
    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)
    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)

    # Attempt 3 is the give-up that used to pass, and so is every attempt after
    # it: the row exhausts NER's budget being rejected rather than early-stopping
    # SUCCESS on the first one.
    echo = f"<body>{MPLX_TEXT}</body>"
    calls = 2
    result: list[dict[str, str]] | None = [{}]
    while result is not None and calls < 20:
        result = handle_response(row_state, echo, max_attempts=3)
        calls += 1

    assert result is None
    # One budget for every stage (#127).
    assert calls == 3
    assert row_state.state == "FAILED"
    # FAILED exactly: a give-up the row can evidence retries to the stage's
    # budget and then terminates like any other exhausted stage.
    assert row_state.state == "FAILED"
    assert row_state.debt_instrument_mentions == []
    assert "drops every one of them" in summarize_failure(row_state)
    # No attempt on this row was ever accepted.
    assert [a.status for a in row_state.all_attempts] == ["FAILED"] * 3


def test_a_zero_tag_row_with_no_earlier_tagging_is_a_clean_zero() -> None:
    """With no earlier tags to compare against, a zero is a finding, not a loss.

    Attempt 1 failed for a reason unrelated to tagging and itself found no
    `debt_instrument`, so neither the high-water mark nor the give-up check has
    anything to fire on -- and nothing else on the row suggests the model can
    find debt here. The row finishes SUCCESS with no failure record.

    This is the case that used to finish PARTIAL as a "possible loss". It was
    dropped because PARTIAL is terminal like any other state, so it bought a
    registry entry and no re-extraction while asserting a loss nothing had
    evidence for. A give-up the row *can* evidence is a validation failure, so
    it retries and then fails for real -- see
    `test_mplx_untagged_echo_no_longer_publishes_as_a_clean_success`.
    """
    from cdt.extractor.core import (
        PUBLISHABLE_ROW_STATES,
        handle_response,
        published_mention_rows,
    )

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    tagged_no_debt = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )

    assert handle_response(row_state, "not xml at all", max_attempts=3)
    assert handle_response(row_state, tagged_no_debt, max_attempts=3) is None

    assert row_state.state == "SUCCESS"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    assert published_mention_rows(row_state) == []
    assert row_state.salvage_notes == []


def test_a_genuinely_debt_free_item_still_early_stops_success() -> None:
    """The no-retry path is untouched: a clean zero stays a clean SUCCESS (#176)."""
    from cdt.extractor.core import handle_response

    text = "Acme Corp filed this report on January 1, 2024."
    row_state = _ner_row(text)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    tagged_no_debt = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )

    assert handle_response(row_state, tagged_no_debt, max_attempts=3) is None

    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_ner_high_water_survives_the_resumable_batch_state() -> None:
    """Batch resumability needed no schema change: `all_attempts` already round-trips.

    A batch row can cross a process exit between its failed attempt and its
    retry, so the guard has to be rebuildable from `state.jsonl` alone (#176).
    """
    from cdt.extractor.core import prior_debt_instrument_high_water

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    from cdt.extractor.core import handle_response

    assert handle_response(row_state, MPLX_TAGGED_BUT_UNFAITHFUL, max_attempts=3)
    assert prior_debt_instrument_high_water(row_state, "ner") == 1

    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())

    assert prior_debt_instrument_high_water(restored, "ner") == 1
    # And the restored row rejects the give-up just as the live one would. A
    # zero-tag response that is not also a byte echo, so this exercises the
    # high-water mark rather than the echo check, which returns first.
    failures = NERStage().validate(
        restored,
        "<body>The Company issued 6.250% Senior Notes due <date>2022</date>.</body>",
    )
    assert failures and any("no <debt_instrument> tags" in f for f in failures)


# --------------------------------------------------------------------------- #
# #127: a NER-specific attempt budget, and unbilled content_filter aborts
# --------------------------------------------------------------------------- #

# The debt-free item and its honest answer: an untagged echo is correct here,
# which is what makes it the right probe for the echo guard's rule that an
# echo is only a give-up once the model has tagged something on this row.
NODEBT_TEXT = "This is the extracted event text."
NODEBT_NER = f"<body>{NODEBT_TEXT}</body>"

# A two-instrument item, so the relation stage actually runs on it.
MULTI_TEXT = "Company entered into a Term Loan and a Revolver on January 1, 2024."
MULTI_NER = (
    "<body>Company entered into a <debt_instrument>Term Loan</debt_instrument> "
    "and a <debt_instrument>Revolver</debt_instrument> on "
    "<date>January 1, 2024</date>.</body>"
)
# One entry that validates and one that cites a tag id the NER output never
# produced, so `instrument_ie` rejects the response as a whole while
# `salvage_instrument_ie_entries` keeps the first entry (#152).
MULTI_IE_ONE_BAD = json.dumps(
    [
        {
            "name": ["tag-1"],
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-3"],
                    "normalized_date": "2024-01-01",
                }
            ],
        },
        {"name": ["tag-99"]},
    ]
)
MULTI_IE = json.dumps(
    [
        {
            "name": ["tag-1"],
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-3"],
                    "normalized_date": "2024-01-01",
                }
            ],
        },
        {
            "name": ["tag-2"],
            "dates": [
                {
                    "kind": "closing",
                    "evidence": ["tag-3"],
                    "normalized_date": "2024-01-01",
                }
            ],
        },
    ]
)

# As observed live: aborted upstream, nothing generated, nothing billed.
# A real `content_filter` body, trimmed, from item 000114036126024567-8-01 in
# `ie_review/runs/followups/full.jsonl` of the models repo. An abort arrives as
# a normal 200 carrying whatever the provider had emitted before it cut, so it
# is partially-tagged XML ending mid-tag -- not an empty string. All 58
# `content_filter` responses in the stored corpora carry 9-107 entity tags and
# at least one `debt_instrument` tag, and none is empty, so a `text=""` fixture
# exercises a shape that has never occurred. It mattered: with an empty body
# the cross-attempt guards in #176 could read an aborted call as the model's
# own tagged work and no test objected.
CONTENT_FILTER_PARTIAL = "<body>Item 8.01\nOther Events.\nOn <date>June 9, 2026</date>, the <organization>Company</organization> commenced an offering of <amount>$500.0 million</amount> in aggregate principal amount of its <debt_instrument>senior secured notes due 2031</debt_instrument> (the “<debt_instrument>Notes"
CONTENT_FILTERED = CompletionResult(
    text=CONTENT_FILTER_PARTIAL,
    finish_reason="content_filter",
    usage={"completion_tokens": 0, "prompt_tokens": 0, "cost": 0.0},
)
# The degenerate shape, kept so both are covered: some aborts may carry nothing.
CONTENT_FILTERED_EMPTY = CompletionResult(
    text="",
    finish_reason="content_filter",
    usage={"completion_tokens": 0, "prompt_tokens": 0, "cost": 0.0},
)


def _stopped(text: str) -> CompletionResult:
    return CompletionResult(text=text, finish_reason="stop")


class _ScriptedCompletionClient:
    """Fake chat client replaying scripted `CompletionResult`s.

    Unlike the text-only scripted client in `test_extractor_batch.py`, this one
    scripts the whole completion, so a `content_filter` abort can be injected
    where it really occurs -- at the provider boundary, above
    `handle_response` -- rather than simulated by hand-feeding the scorer
    (#127, #135).
    """

    def __init__(
        self: _ScriptedCompletionClient, completions: list[CompletionResult]
    ) -> None:
        """Store the completions to hand back in order."""
        self.completions = list(completions)
        self.requests: list[list[dict[str, str]]] = []

    async def complete(
        self: _ScriptedCompletionClient,
        *,
        messages: list[dict[str, str]],
        model: str,
        reasoning_effort: str,
    ) -> CompletionResult:
        """Record the request and return the next scripted completion."""
        del model, reasoning_effort
        self.requests.append([dict(message) for message in messages])
        return self.completions[len(self.requests) - 1]


def _run_live(
    text: str, completions: list[CompletionResult], max_attempts: int = 3
) -> tuple[ExtractionRowState, _ScriptedCompletionClient]:
    """Drive the real live loop, returning (row_state, client)."""
    import asyncio

    from cdt.extractor.core import run_extraction_workflow

    client = _ScriptedCompletionClient(completions)
    row_state = asyncio.run(
        run_extraction_workflow(
            item_row={"item_id": "item-1", "text": text},
            model="m",
            reasoning_effort="none",
            max_attempts=max_attempts,
            client=client,
        )
    )
    return row_state, client


def test_every_stage_gets_the_same_attempt_budget() -> None:
    """`--max-attempts` means the same thing for every stage (#127).

    #127 proposed a larger NER budget; the stored corpora do not support the
    number (only two of 761 rows ever reached the cap, and one of those is the
    give-up #176 now rejects), so NER is back in line with the rest. Driven
    end to end rather than asserted against a helper, because the budget is
    only real if the loop stops there.
    """
    from cdt.extractor.core import DEFAULT_MAX_ATTEMPTS

    # Pinned as a literal for the same reason MAX_CONTENT_FILTER_RESENDS is:
    # the budget *is* the number, so restating the constant asserts nothing
    # about its value.
    assert DEFAULT_MAX_ATTEMPTS == 3

    # NER: three scored attempts, then the row terminates.
    row_state, client = _run_live(MPLX_TEXT, [_stopped("not xml")] * 10)
    assert len(client.requests) == 3
    assert row_state.state == "FAILED"

    # instrument_ie: the same three, after a NER pass.
    row_state, client = _run_live(
        MULTI_TEXT, [_stopped(MULTI_NER)] + [_stopped("not json")] * 10
    )
    assert len(client.requests) == 1 + 3
    assert row_state.state == "FAILED"

    # And the operator's knob still moves it.
    row_state, client = _run_live(MPLX_TEXT, [_stopped("not xml")] * 10, max_attempts=5)
    assert len(client.requests) == 5


def test_a_content_filtered_attempt_is_resent_as_the_original_request() -> None:
    """An upstream abort has nothing for the model to correct (#127, #135).

    The ordinary retry path would append the aborted response as an assistant
    turn plus a validation complaint about it, which every later call in the
    row then pays prompt tokens for. A `content_filter` abort is not the
    model answering badly, so the same request goes back out unchanged.
    """
    row_state, client = _run_live(NODEBT_TEXT, [CONTENT_FILTERED, _stopped(NODEBT_NER)])

    assert len(client.requests) == 2
    # Byte-identical resend: no assistant turn, no complaint, same request.
    assert client.requests[1] == client.requests[0]
    assert all(message["role"] != "assistant" for message in client.requests[1])
    assert row_state.state == "SUCCESS"


def test_a_provider_abort_is_recorded_but_not_scored() -> None:
    """The audit log keeps the call; the scorer never sees it (#127, #135).

    `attempt_index` stays a true count of *scored* attempts, so the model's
    first real answer is attempt 1 even when the provider aborted first -- which
    is what keeps #176's cross-attempt checks from reading an abort as a retry.
    """
    row_state, _ = _run_live(NODEBT_TEXT, [CONTENT_FILTERED, _stopped(NODEBT_NER)])

    statuses = [
        (a.stage_name, a.attempt_index, a.status) for a in row_state.all_attempts
    ]
    assert statuses == [("ner", 0, "ABORTED"), ("ner", 1, "SUCCESS")]
    aborted = row_state.all_attempts[0]
    assert aborted.finish_reason == "content_filter"
    assert aborted.usage == {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0.0}
    # The request is identical to the scored attempt's, so it is not stored twice.
    assert aborted.messages == []


def test_content_filter_aborts_do_not_consume_the_stage_budget() -> None:
    """Unscored calls must not spend the row's attempts (#127, #135).

    Eleven of thirteen live NER calls came back `content_filter` with
    `completion_tokens=0` and `cost=0.0`, and the cut point is nondeterministic
    (1,163 / 280 / 1,375 characters on three repeats of one item), so resending
    is both correct and free.
    """
    # Three aborts, then the stage's full budget of three real attempts, all
    # rejected. The aborts cost the row nothing: it still gets all three.
    completions = [CONTENT_FILTERED] * 3 + [_stopped("not xml")] * 3
    row_state, client = _run_live(MPLX_TEXT, completions)

    assert len(client.requests) == 6
    assert row_state.state == "FAILED"
    scored = [a for a in row_state.all_attempts if a.status != "ABORTED"]
    assert [a.attempt_index for a in scored] == [1, 2, 3]


def test_persistent_content_filtering_terminates_at_the_resend_cap() -> None:
    """Free retries still have to stop, and stopping must not blame the model (#127).

    Past the cap the abort used to fall through to `handle_response` and be
    scored as an ordinary bad answer: the row paid the stage's whole budget
    again in whole-item calls,
    each one growing the retry conversation with a junk assistant turn, and
    the resulting `FAILED` attempt made #176's checks read a provider abort as
    a model failure. The row now terminates instead.
    """
    from cdt.extractor.core import MAX_CONTENT_FILTER_RESENDS, summarize_failure

    # Pinned as a literal: the cap bounds spend on a filtered row, so restating
    # the constant here would assert nothing about its value.
    assert MAX_CONTENT_FILTER_RESENDS == 6

    row_state, client = _run_live(MPLX_TEXT, [CONTENT_FILTERED] * 40)

    # Six re-sends plus the call that trips the cap, and no more.
    assert len(client.requests) == 7
    assert row_state.state == "FAILED"
    # Not one call was scored, so the model is never blamed for the abort.
    # The trailing "incomplete" is the unused outstanding attempt that
    # `finish` always appends, not a call that was made.
    assert [a.status for a in row_state.all_attempts] == ["ABORTED"] * 7 + [
        "incomplete"
    ]
    assert row_state.current_attempt.attempt_index == 0
    # Every request was the pristine one: no conversation growth past the cap.
    assert all(request == client.requests[0] for request in client.requests)
    summary = summarize_failure(row_state)
    assert "aborted by the provider on 7 of its calls" in summary
    # Nothing was scored on this row, so saying so is accurate here.
    assert "no attempt was scored" in summary


def test_the_abort_note_reports_the_model_failure_that_happened_too() -> None:
    """When both the provider and the model failed, the registry says both (#127).

    The abort count is a per-stage lifetime count with no reset, so aborts need
    not be consecutive: this row is aborted six times, answers badly once, and
    is aborted again past the cap. The note used to call all seven calls
    "consecutive" and add "no attempt was scored", and because
    `summarize_failure` prefers salvage notes over `validation_errors`, the
    model's own error never reached the registry. An operator was told to go
    and talk to the provider while the extraction defect stayed invisible.
    """
    from cdt.extractor.core import summarize_failure

    row_state, client = _run_live(
        MPLX_TEXT,
        [CONTENT_FILTERED] * 6 + [_stopped("not xml at all")] + [CONTENT_FILTERED] * 5,
    )

    assert len(client.requests) == 6 + 1 + 1
    assert row_state.state == "FAILED"
    statuses = [a.status for a in row_state.all_attempts if a.stage_name == "ner"]
    assert statuses == ["ABORTED"] * 6 + ["FAILED", "ABORTED", "incomplete"]

    summary = summarize_failure(row_state)
    # The count is right, and no longer claims the calls were back to back.
    assert "aborted by the provider on 7 of its calls" in summary
    assert "consecutive" not in summary
    # One call *was* scored, so the opposite claim is gone...
    assert "no attempt was scored" not in summary
    assert "1 scored attempt also failed" in summary
    # ...and the model's own error reaches the registry alongside the abort.
    assert "not valid XML" in summary


def test_the_abort_note_only_reports_failures_from_the_stage_that_aborted() -> None:
    """An earlier stage's failure is not this stage's story (#127).

    NER fails once and then recovers, so the row reaches `instrument_ie` with a
    scored failure already in its history. `instrument_ie` is then aborted to
    its cap without ever being answered, and its note must say so -- citing
    NER's error here would send an operator to the wrong stage.
    """
    from cdt.extractor.core import failed_stage_name, summarize_failure

    row_state, client = _run_live(
        MPLX_TEXT,
        [_stopped("not xml at all"), _stopped(MPLX_TAGGED)] + [CONTENT_FILTERED] * 10,
    )

    assert len(client.requests) == 2 + 7
    assert row_state.state == "FAILED"
    assert failed_stage_name(row_state) == "instrument_ie"

    summary = summarize_failure(row_state)
    assert "instrument_ie aborted by the provider on 7 of its calls" in summary
    # instrument_ie itself was never answered, so that claim is true of it...
    assert "no attempt was scored" in summary
    # ...and NER's failure, which belongs to a stage that went on to succeed,
    # stays out of it.
    assert "not valid XML" not in summary


def test_the_abort_note_reports_the_most_recent_scored_failure() -> None:
    """Of several scored failures, the latest is the one worth reporting (#127).

    The model is shown its error and asked again, so the last rejection is the
    state the row actually died in; an earlier one has already been superseded.
    """
    from cdt.extractor.core import summarize_failure

    row_state, client = _run_live(
        MPLX_TEXT,
        [
            _stopped("not xml at all"),
            _stopped("<body>wrong text entirely</body>"),
        ]
        + [CONTENT_FILTERED] * 10,
    )

    assert len(client.requests) == 2 + 7
    assert row_state.state == "FAILED"

    summary = summarize_failure(row_state)
    assert "2 scored attempts also failed" in summary
    assert "must match the input text exactly" in summary
    assert "not valid XML" not in summary


def test_a_filtered_row_is_registered_against_the_stage_that_was_aborted() -> None:
    """An operator retrying the row needs the stage, not a generic failure (#127)."""
    from cdt.extractor.core import failed_stage_name

    row_state, _ = _run_live(MPLX_TEXT, [CONTENT_FILTERED] * 40)

    assert failed_stage_name(row_state) == "ner"


def test_aborts_at_the_relation_stage_still_publish_the_items_mentions() -> None:
    """#152's rule holds: a stage the provider will not run costs lineage, not instruments.

    The relation stage is the last one, and its output is only lineage. A row
    whose instruments already validated must not lose them because the
    provider refused to run the final call.
    """
    from cdt.extractor.core import PUBLISHABLE_ROW_STATES, summarize_failure

    row_state, client = _run_live(
        MULTI_TEXT,
        [_stopped(MULTI_NER), _stopped(MULTI_IE)] + [CONTENT_FILTERED] * 40,
    )

    # Two scored calls for ner and instrument_ie, then the relation stage is
    # aborted to its cap.
    assert len(client.requests) == 2 + 7
    assert row_state.state == "PARTIAL"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    assert len(row_state.debt_instrument_mentions) == 2
    assert "mentions published without lineage relations" in summarize_failure(
        row_state
    )


def test_aborts_at_the_ie_stage_still_publish_the_entries_that_validated() -> None:
    """The abort cap applies #152 the same way a scored failure does (#127, #152).

    An `instrument_ie` response rejected as a whole can still hold valid
    entries, and when the provider then aborts the stage to its cap those
    entries are already sitting in `stage_responses` -- the aborts came after
    the answer, not instead of it. Terminating FAILED there threw them away,
    while the same row reaching the same dead end through three *scored*
    failures published them. Same loss, same remedy, so the two terminal paths
    now agree.
    """
    from cdt.extractor.core import PUBLISHABLE_ROW_STATES, summarize_failure

    row_state, client = _run_live(
        MULTI_TEXT,
        [_stopped(MULTI_NER), _stopped(MULTI_IE_ONE_BAD)] + [CONTENT_FILTERED] * 40,
    )

    # One NER call, one scored instrument_ie answer, then the stage is aborted
    # to its cap.
    assert len(client.requests) == 1 + 1 + 7
    assert row_state.state == "PARTIAL"
    assert row_state.state in PUBLISHABLE_ROW_STATES
    # The entry that validated publishes; the one citing a missing tag does not.
    assert len(row_state.debt_instrument_mentions) == 1
    assert "published without lineage relations" in summarize_failure(row_state)
    # And the invalid entry is gone from the stored response, not merely
    # ignored downstream: `salvage_instrument_ie_entries` rewrites what the
    # audit log keeps, so the kept set is recorded rather than inferred.
    # `postprocess` would have skipped the bad entry either way, so the mention
    # count alone cannot tell a salvage from an unfiltered publish.
    assert "tag-99" not in row_state.stage_responses["instrument_ie"]


def test_content_filter_resend_cap_survives_the_resumable_batch_state() -> None:
    """A batch row can cross a process exit between an abort and its resend (#127).

    The cap is read off `all_attempts` rather than a counter held by the
    caller, precisely because the batch backend folds one response per tick.
    """
    from cdt.extractor.core import count_content_filter_aborts

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    for _ in range(2):
        row_state.record_unbilled_abort("", CONTENT_FILTERED)

    assert count_content_filter_aborts(row_state, "ner") == 2

    restored = ExtractionRowState.from_state_dict(row_state.to_state_dict())

    assert count_content_filter_aborts(restored, "ner") == 2
    # And the outstanding request is unchanged, so the resend is byte-identical.
    assert restored.current_attempt.messages == NERStage().preprocess(row_state)
    assert restored.current_attempt.attempt_index == 0


# --------------------------------------------------------------------------- #
# #176 and #127 together: an abort must not look like a retry
# --------------------------------------------------------------------------- #


def test_an_abort_does_not_make_an_honest_zero_tag_row_partial() -> None:
    """A provider abort is not the model failing, so the row is a clean zero (#176, #127).

    An aborted call carries its own status rather than `FAILED`, so nothing
    downstream treats the row as one the model answered badly: a debt-free item
    whose first call was aborted publishes its honest zero with no failure
    record attached.
    """
    tagged_no_debt = (
        "<body><organization>Acme Corp</organization> filed this report on "
        "<date>January 1, 2024</date>.</body>"
    )
    row_state, _ = _run_live(
        "Acme Corp filed this report on January 1, 2024.",
        [CONTENT_FILTERED, _stopped(tagged_no_debt)],
    )

    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_the_ner_guards_do_not_read_an_aborted_calls_partial_output() -> None:
    """An abort is not the model's work, so it is not history to regress against.

    `record_unbilled_abort` stores whatever the provider emitted before it cut,
    and that text is tagged: all 58 `content_filter` responses in the stored
    corpora carry between 9 and 107 entity tags and at least one
    `debt_instrument` tag, and none of them is empty. Counted as the model's
    own earlier work it makes the echo guard and the high-water mark fire on an
    item the model has never successfully tagged, and the row then dies FAILED
    on a complaint it cannot act on -- the abort adds no assistant turn, so
    "keep every tag you found" names work that is not in its context
    (#176, #127).
    """
    from cdt.extractor.core import (
        count_debt_instrument_tags,
        count_ner_entity_tags,
        prior_attempt_tagged,
        prior_debt_instrument_high_water,
    )

    # The fixture really is tagged, so the assertions below are not vacuous --
    # this is exactly what a `text=""` fixture failed to exercise.
    assert count_ner_entity_tags(CONTENT_FILTER_PARTIAL) == 5
    assert count_debt_instrument_tags(CONTENT_FILTER_PARTIAL) == 2

    row_state = _ner_row(MPLX_TEXT)
    row_state.current_attempt.messages = NERStage().preprocess(row_state)
    row_state.record_unbilled_abort(CONTENT_FILTER_PARTIAL, CONTENT_FILTERED)

    assert prior_attempt_tagged(row_state, "ner") is False
    assert prior_debt_instrument_high_water(row_state, "ner") == 0

    # And the same row driven end to end: the honest echo publishes in two
    # calls instead of looping to FAILED on an unanswerable complaint.
    row_state, client = _run_live(
        NODEBT_TEXT, [CONTENT_FILTERED, _stopped(f"<body>{NODEBT_TEXT}</body>")]
    )

    assert len(client.requests) == 2
    assert row_state.state == "SUCCESS"
    assert row_state.salvage_notes == []


def test_an_empty_aborted_body_behaves_the_same_as_a_truncated_one() -> None:
    """Both abort shapes are unscored calls, so neither changes the verdict (#127)."""
    for aborted in (CONTENT_FILTERED, CONTENT_FILTERED_EMPTY):
        row_state, client = _run_live(
            NODEBT_TEXT, [aborted, _stopped(f"<body>{NODEBT_TEXT}</body>")]
        )

        assert len(client.requests) == 2
        assert row_state.state == "SUCCESS"
        assert row_state.salvage_notes == []


def test_abort_counts_are_kept_per_stage_not_per_row() -> None:
    """Each stage gets its own resend budget (#127).

    Filtering is nondeterministic per call, so one row can be aborted on an
    early stage, recover, and be aborted again later. Counted per row, the
    earlier stage's aborts would eat the later stage's budget and terminate a
    row that still had resends coming to it.
    """
    from cdt.extractor.core import count_content_filter_aborts

    row_state, client = _run_live(
        MULTI_TEXT,
        [CONTENT_FILTERED] * 3
        + [_stopped(MULTI_NER), _stopped(MULTI_IE)]
        + [CONTENT_FILTERED] * 40,
    )

    # Three NER aborts, a NER pass, an instrument_ie pass, and then the
    # relation stage still gets its own full seven before terminating.
    assert len(client.requests) == 3 + 2 + 7
    assert count_content_filter_aborts(row_state, "ner") == 3
    assert count_content_filter_aborts(row_state, "instrument_relation") == 7
    # #152 still applies: the mentions it already earned publish.
    assert row_state.state == "PARTIAL"
    assert len(row_state.debt_instrument_mentions) == 2


def test_after_a_resend_the_next_answer_is_still_a_first_attempt() -> None:
    """The echo guard must not fire on an answer the model was never corrected on.

    The model has been shown no error after a provider abort -- the pristine
    request went back out -- so an untagged echo is still the honest answer for
    an item with nothing to tag (#176, #127).
    """
    echo = f"<body>{NODEBT_TEXT}</body>"
    row_state, client = _run_live(NODEBT_TEXT, [CONTENT_FILTERED, _stopped(echo)])

    assert len(client.requests) == 2
    scored = [a for a in row_state.all_attempts if a.status != "ABORTED"]
    assert [a.validation_errors for a in scored] == [[]]
    assert row_state.state == "SUCCESS"
    assert row_state.debt_instrument_mentions == []


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
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2026-07-28",
                    },
                    {
                        "kind": "maturity",
                        "evidence": ["tag-d-2"],
                        "normalized_date": "2028-07-28",
                    },
                ],
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


def test_an_only_prior_amount_publishes_no_current_principal() -> None:
    """The figure in the name is the prior figure; it must not come back as current.

    `Amendment No. 2 to the $100 million Credit Agreement ... from $100,000,000`
    states only the before-figure. With every commitment `prior`, the head's
    current capacity is unstated, and reading `$100 million` back off the name
    published the pre-amendment figure as current — #165's stale head by a
    second route (#206). The honest answer is null; the minted prior state is
    where that figure belongs.
    """
    from cdt.extractor.core import mint_prior_state_rows

    row_state = ExtractionRowState(
        item_row={"item_id": "item-1", "date": "2024-06-01"},
        stage_name="instrument_ie",
    )
    row_state.ner_tagged_xml = (
        "<body>Amendment No. 2 to the "
        '<debt_instrument id="tag-i-1">$100 million Credit Agreement</debt_instrument>, '
        'dated as of <date id="tag-d-1">February 3, 2020</date>, reduced the '
        'commitments from <amount id="tag-a-1">$100,000,000</amount>.</body>'
    )
    row_state.stage_responses["instrument_ie"] = json.dumps(
        [
            {
                "name": ["tag-i-1"],
                "amounts": [
                    {
                        "kind": "commitment",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "100000000",
                        "currency": "USD",
                        "prior": True,
                    }
                ],
                "dates": [
                    {
                        "kind": "agreement",
                        "evidence": ["tag-d-1"],
                        "normalized_date": "2020-02-03",
                    }
                ],
            }
        ]
    )

    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["principal_amount"] is None
    amounts = json.loads(str(mention["amounts_json"]))
    assert [(a["normalized_amount"], a["prior"]) for a in amounts] == [
        ("100000000", True)
    ]

    published = mint_prior_state_rows([dict(mention)])
    assert len(published) == 2
    assert published[1]["principal_amount"] == "100000000"
    assert published[1]["start_date"] == "2020-02-03"
    assert published[0]["amendment_of"] == published[1]["debt_instrument_mention_id"]


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
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        # The model reports the value it read, without the float
                        # artifact the parser used to produce.
                        "normalized_amount": "372246148.11",
                        "currency": "USD",
                    }
                ],
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
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "500000.00",
                        "currency": "USD",
                    }
                ],
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
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-figure", "tag-a-label"],
                        "normalized_amount": "2000000",
                        "currency": "USD",
                    }
                ],
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
                "amounts": [
                    {
                        "kind": "principal",
                        "evidence": ["tag-a-1"],
                        "normalized_amount": "300000000",
                        "currency": "CAD",
                    }
                ],
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


def test_match_reads_a_partition_written_before_a_json_column_existed(
    tmp_path: Path,
) -> None:
    """A mentions partition older than a `_json` column must still match (#193).

    Nothing in this suite wrote a partition *missing* a later-added column, read
    it back through `read_dataset`, and matched it -- every other old-shape test
    writes the full current column list and leaves a value null, which is a
    different thing. The gap mattered: `read_table` answers a projected read for
    a column the file does not have by falling back to a full read plus
    `reindex`, which fills it with NaN, and NaN is truthy, so
    `str(row.get(col) or "[]")` yields the literal text `nan`.

    `retired_by_json` fed `json.loads` directly, so the whole match pass died
    with `JSONDecodeError: Expecting value`. Reproduced on the real corpus
    before the fix: `cdt match` over `data/genwindow-run-dev` (245 mentions
    partitions written before `retired_by_json`, `amounts_json` and
    `parties_json` existed) crashed in `prepare_mention`; after it, the same
    root completes with 679 edge rows and 572 instruments.
    """
    row = build_mention_row(
        mention_id="m-1",
        item_id="item-1",
        accession_number="0001",
        cik="320193",
        date="2024-01-02",
        name="Term Loan",
        start_date="2024-01-01",
        amount="$100 million",
    )
    # The partition as it was actually written, not the current shape with
    # nulls: these three columns are simply not in the file's schema.
    absent = ["retired_by_json", "amounts_json", "parties_json"]
    legacy = pd.DataFrame([{k: v for k, v in row.items() if k not in absent}])
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=legacy,
    )
    # Pin the premise: if a future writer starts backfilling these columns this
    # test would still pass while testing nothing.
    stored = read_table(
        tmp_path / "mentions" / "date=2024-01-02" / "shard=0001" / "part-0000.parquet"
    )
    assert not set(absent) & set(stored.columns)

    tables = match_pending_mentions(artifact_root=tmp_path, batch_size=5)

    assert tables["debt_instrument"]["debt_instrument_id"].to_list() == ["m-1"]
    # The literal text `nan` must not have reached a published payload.
    instruments = read_dataset(debt_instruments_root(tmp_path))
    assert instruments["parties_json"].to_list() == ["[]"]


def test_prepare_mention_reads_an_absent_json_column_as_an_empty_payload() -> None:
    """Every `_json` read must survive NaN, not just the one that raised (#193).

    Six sites shared the `str(row.get(col) or "[]")` idiom. Only
    `retired_by_json` raised; the rest handed `parse_cluster_list` the text
    `nan`, which it discards as a `JSONDecodeError` and returns `[]` for -- so
    they degraded to the same empty payload the guard produces rather than
    erroring. That makes them invisible, which is why they are pinned here
    rather than left to the integration test above.
    """
    from cdt.matcher.core import prepare_mention

    row = build_mention_row(
        mention_id="m-1",
        item_id="item-1",
        accession_number="0001",
        cik="320193",
        date="2024-01-02",
        name="Term Loan",
        start_date="2024-01-01",
        amount="$100 million",
    )
    # What `read_table`'s reindex fallback actually hands the matcher.
    for column in ("retired_by_json", "amounts_json", "parties_json"):
        row[column] = float("nan")

    mention = prepare_mention(row)

    assert mention.retired_by == ()
    assert mention.amounts_json == "[]"
    assert mention.parties_json == "[]"
    assert mention.lender_signature == ""


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

    written_matches = read_dataset(mention_cluster_edges_root(tmp_path))
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
                cik="0002078008",
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
                cik="0002078008",
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
                cik="0000320193",
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


def _ordinal_chain_root(tmp_path: Path, **mention_overrides: object) -> Path:
    """Write two mentions the pass links by ordinal, and match them."""
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
                **mention_overrides,
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
                **mention_overrides,
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=rows,
    )
    match_pending_mentions(artifact_root=tmp_path, batch_size=5)
    return tmp_path


def _published_instruments(root: Path) -> dict[str, dict[str, object]]:
    return {
        str(row["debt_instrument_id"]): row
        for row in read_dataset(debt_instruments_root(root)).to_dict("records")
    }


def test_lineage_pass_reopens_a_pointer_the_rules_would_now_refuse(
    tmp_path: Path,
) -> None:
    """A link an earlier run inferred does not outlive the evidence against it.

    `infer_amendment_parents` only considers rows whose pointer is null and the
    matcher carries an existing pointer forward, so the EQT/EQM cross-borrower
    link #197's guard was written to remove survived on every already-matched
    root and was republished by the next plain match (#204). Here the first
    pass links the chain; the published rows then acquire disagreeing
    borrowers, as a tightened rule or a later filing would give them, and a
    plain second pass must take the link back.
    """
    root = _ordinal_chain_root(tmp_path)
    first = apply_lineage_inference_pass(root)
    assert first["links"] == 1
    assert (
        _published_instruments(root)["m-2"]["amendment_inferred_by"] == "ordinal_chain"
    )

    from cdt.datasets import shard_for_cik

    published = read_dataset(debt_instruments_root(root))
    borrowers = {
        "m-1": [{"role": "borrower", "canonical_name": "EQT Corporation"}],
        "m-2": [{"role": "borrower", "canonical_name": "EQM Midstream Partners, LP"}],
    }
    published["parties_json"] = published["debt_instrument_id"].map(
        lambda value: json.dumps(borrowers[str(value)])
    )
    write_partition_table(
        debt_instruments_root(root),
        partition={"cik_shard": shard_for_cik("320193")},
        table=published,
    )

    second = apply_lineage_inference_pass(root)
    after = _published_instruments(root)
    assert second["reopened"] == 1
    assert second["links"] == 0
    assert after["m-2"]["amendment_of_debt_instrument_id"] is None
    assert after["m-2"]["amendment_inferred_by"] is None
    # and the rollup follows the pointer back out
    assert after["m-1"]["is_lineage_head"] is True
    assert after["m-1"]["superseded_by_debt_instrument_id"] is None


def test_lineage_pass_never_reopens_an_extracted_pointer(tmp_path: Path) -> None:
    """An extracted relation is a cited fact; only inferred pointers re-derive."""
    rows = pd.DataFrame(
        [
            build_mention_row(
                mention_id="m-old",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Credit Agreement",
                start_date="2020-01-01",
                amount="$100 million",
            ),
            build_mention_row(
                mention_id="m-new",
                item_id="item-1",
                accession_number="0001",
                cik="320193",
                date="2024-01-02",
                name="Credit Agreement",
                start_date="2024-01-01",
                amount="$250 million",
                amendment_of="m-old",
            ),
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-01-02", "shard": "0001"},
        table=rows,
    )
    match_pending_mentions(artifact_root=tmp_path, batch_size=5)
    before = _published_instruments(tmp_path)
    assert before["m-new"]["amendment_of_debt_instrument_id"] == "m-old"
    assert before["m-new"]["amendment_inferred_by"] is None

    stats = apply_lineage_inference_pass(tmp_path)
    after = _published_instruments(tmp_path)
    assert stats["reopened"] == 0
    assert after["m-new"]["amendment_of_debt_instrument_id"] == "m-old"
    assert after["m-new"]["amendment_inferred_by"] is None


def test_lineage_pass_keeps_a_null_cik_row_in_one_shard(tmp_path: Path) -> None:
    """The rewrite must shard a null cik the way the matcher did, not twice.

    `match_pending_mentions` maps `cik.fillna("")`; the pass mapped `str(cik)`,
    so a null cik went to `shard_for_cik("nan")` while its original copy stayed
    under `shard_for_cik("")` — a permanently duplicated instrument (#204).
    """
    from cdt.datasets import shard_for_cik

    root = _ordinal_chain_root(tmp_path)
    published = read_dataset(debt_instruments_root(root))
    orphan = published.iloc[[0]].copy()
    orphan["debt_instrument_id"] = "orphan"
    orphan["cik"] = None
    write_partition_table(
        debt_instruments_root(root),
        partition={"cik_shard": shard_for_cik("")},
        table=orphan,
    )

    apply_lineage_inference_pass(root)

    after = read_dataset(debt_instruments_root(root))
    assert (after["debt_instrument_id"] == "orphan").sum() == 1
    orphan_files = [
        path
        for path in (tmp_path / "debt-instruments").rglob("*.parquet")
        if "orphan" in set(pd.read_parquet(path)["debt_instrument_id"])
    ]
    assert [path.parent.name for path in orphan_files] == [
        f"cik_shard={shard_for_cik('')}"
    ]


def test_lineage_pass_renews_the_writer_lease(tmp_path: Path) -> None:
    """A full-corpus read and rewrite must keep renewing, or it outlives its lease."""
    root = _ordinal_chain_root(tmp_path)
    renewals: list[int] = []

    apply_lineage_inference_pass(root, renew=lambda: renewals.append(1))

    # once after the reads, once per shard written
    assert len(renewals) >= 2


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
    assert stats == {"links": 1, "reopened": 0, "heads_before": 2, "heads_after": 1}
    child = published["m-2"]
    parent = published["m-1"]
    assert child["amendment_of_debt_instrument_id"] == "m-1"
    assert child["amendment_inferred_by"] == "ordinal_chain"
    assert child["is_lineage_head"]
    # the rollup must be re-derived from the new pointer, not left stale
    assert parent["is_lineage_head"] is False
    assert parent["superseded_by_debt_instrument_id"] == "m-2"
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
    assert after["m-1"]["is_lineage_head"] is False

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

    written_matches = read_dataset(mention_cluster_edges_root(tmp_path))
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

    written_matches = read_dataset(mention_cluster_edges_root(tmp_path)).sort_values(
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
    assert amounts[assignment["m-initial"]] == "1250000"
    assert amounts[assignment["m-additional"]] == "1100000"


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

    pending, _ = pending_source_partitions("classify", "items", artifact_root=str(root))

    assert len(pending) == 1
    assert pending[0][0].endswith("date=2026-01-02/shard=0007/part-0000.parquet")

    (root / "items" / "items.parquet").write_bytes(b"")

    with pytest.raises(ValueError, match="Non-canonical parquet file"):
        pending_source_partitions("classify", "items", artifact_root=str(root))


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


def seed_document_partitions_across_months(
    tmp_path: Path, partitions: list[tuple[str, str]]
) -> list[str]:
    """Write one document partition per ``(date, shard)``, dates free-form.

    seed_document_partitions' fixed pair is same-month; the registry shards by
    year-month, so anything about sharding needs dates that straddle months.
    """
    paths: list[str] = []
    for index, (filing_date, shard) in enumerate(partitions):
        table = pd.DataFrame(
            [
                {
                    "accession_number": f"00011403612600{index:04d}",
                    "cik": "320193",
                    "company_name": "Example Inc.",
                    "url": "https://sec.example/full.txt",
                    "text": (
                        "ITEM INFORMATION: Other Events\n"
                        "<DOCUMENT>\n<TYPE>8-K\n<TEXT>\n"
                        "Item 8.01 Other Events.\n"
                        "This is the extracted event text.\n"
                        "</TEXT>\n</DOCUMENT>\n"
                    ),
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


def test_completion_registry_shards_entries_by_date_prefix(tmp_path: Path) -> None:
    """One object per year-month, each holding only that month's entries (#191).

    The single object it replaces is 56.8 MB at full corpus scale and was read
    and rewritten whole every 100 partitions, 4,400 times per itemize pass.
    """
    from cdt.datasets import (
        CompletedPartition,
        completion_registry_shard_path,
        date_shard_partition_path,
        load_completion_registry,
        save_completion_registry,
    )

    keys = [
        date_shard_partition_path(
            "documents", partition_date=day, shard="0001", artifact_root=tmp_path
        )
        for day in ("2024-01-02", "2024-01-31", "2024-02-01", "2024-03-15")
    ]
    save_completion_registry(
        "itemize",
        {
            key: CompletedPartition(fingerprint=f"f{index}")
            for index, key in enumerate(keys)
        },
        artifact_root=tmp_path,
    )

    shards = sorted(
        path.name for path in (tmp_path / "runs" / "itemize" / "completed").iterdir()
    )
    assert shards == ["date=2024-01.json", "date=2024-02.json", "date=2024-03.json"]
    january = json.loads(
        Path(
            completion_registry_shard_path("itemize", "2024-01", artifact_root=tmp_path)
        ).read_text()
    )
    assert sorted(january["partitions"]) == [
        "documents/date=2024-01-02/shard=0001/part-0000.parquet",
        "documents/date=2024-01-31/shard=0001/part-0000.parquet",
    ]
    assert january["date_prefix"] == "2024-01"

    # And the loader reassembles every shard into one registry.
    loaded = load_completion_registry("itemize", artifact_root=tmp_path)
    assert set(loaded) == set(keys)
    assert loaded[keys[3]].fingerprint == "f3"


def test_completion_registry_save_rewrites_only_the_months_it_touched(
    tmp_path: Path,
) -> None:
    """The #191 fix itself: a batch save must not rewrite untouched months.

    Before sharding, persisting one more partition read and rewrote every
    entry the corpus had -- 113.5 MB of transfer per batch boundary at full
    scale. Asserted on the objects actually touched, because an assertion that
    the final state is correct passes just as well on the quadratic version.
    """
    from cdt.datasets import (
        CompletedPartition,
        completion_registry_shard_path,
        date_shard_partition_path,
        load_completion_registry,
        save_completion_registry,
    )

    def key(day: str) -> str:
        return date_shard_partition_path(
            "documents", partition_date=day, shard="0001", artifact_root=tmp_path
        )

    old_months = [f"2024-{month:02d}-15" for month in range(1, 10)]
    save_completion_registry(
        "itemize",
        {key(day): CompletedPartition(fingerprint="old") for day in old_months},
        artifact_root=tmp_path,
    )
    january = Path(
        completion_registry_shard_path("itemize", "2024-01", artifact_root=tmp_path)
    )
    january_before = january.read_bytes()

    registry = load_completion_registry("itemize", artifact_root=tmp_path)
    assert len(registry) == len(old_months)
    registry[key("2024-10-15")] = CompletedPartition(fingerprint="new")

    moved: list[str] = []

    def record(name: str, function: object) -> object:
        def wrapper(path: object, *args: object, **kwargs: object) -> object:
            moved.append(f"{name}:{Path(str(path)).name}")
            return function(path, *args, **kwargs)  # type: ignore[operator]

        return wrapper

    with pytest.MonkeyPatch.context() as patch:
        for name, attribute in (
            ("read", "read_json_artifact_versioned"),
            ("write", "replace_json_artifact_if_match"),
            ("create", "write_json_artifact_if_absent"),
        ):
            patch.setattr(
                cdt_datasets,
                attribute,
                record(name, getattr(cdt_datasets, attribute)),
            )
        save_completion_registry("itemize", registry, artifact_root=tmp_path)

    # Exactly one object created, nothing else read or rewritten.
    assert moved == ["create:date=2024-10.json"]
    assert january.read_bytes() == january_before


def test_completion_registry_batch_saves_do_not_resend_earlier_batches(
    tmp_path: Path,
) -> None:
    """A later batch boundary must not rewrite the months earlier ones wrote.

    Stages save repeatedly against one registry object, at every batch
    boundary, so an interruption does not discard the run's progress (#111).
    The dirty set is what a save sends, so unless a committed key leaves it,
    batch k re-sends every key from batches 1..k-1 and rewrites every shard
    they span -- cost quadratic in the run's length, which is the thing #191
    removes. The single-save test above cannot see this: it never saves the
    same registry object twice, and no stage ever saves one only once.
    """
    from cdt.datasets import (
        CompletedPartition,
        CompletionRegistry,
        date_shard_partition_path,
        save_completion_registry,
    )

    def key(day: str) -> str:
        return date_shard_partition_path(
            "documents", partition_date=day, shard="0001", artifact_root=tmp_path
        )

    touched: list[str] = []

    def record(function: object) -> object:
        def wrapper(path: object, *args: object, **kwargs: object) -> object:
            touched.append(Path(str(path)).name)
            return function(path, *args, **kwargs)  # type: ignore[operator]

        return wrapper

    registry = CompletionRegistry()
    with pytest.MonkeyPatch.context() as patch:
        for attribute in (
            "read_json_artifact_versioned",
            "replace_json_artifact_if_match",
            "write_json_artifact_if_absent",
        ):
            patch.setattr(
                cdt_datasets, attribute, record(getattr(cdt_datasets, attribute))
            )
        for day in ("2024-01-15", "2024-02-15", "2024-03-15"):
            registry[key(day)] = CompletedPartition(fingerprint="new")
            touched.clear()
            save_completion_registry("itemize", registry, artifact_root=tmp_path)
            assert touched == [f"date={day[:7]}.json"], f"batch {day} touched {touched}"
    # Every key is persisted, so nothing is owed to the next save.
    assert not registry.dirty


def test_completion_registry_load_reads_its_shards_concurrently(
    tmp_path: Path,
) -> None:
    """A load must not serialize one round trip per occupied month (#227, #110).

    Sharding turned one GET into one per month, and the merged #220 read them
    in a plain loop: at full-corpus shape that is one HeadObject, one LIST and
    393 serial GETs, or 27.5 s per load at the 70 ms round trip #110 measured
    on this stack, five times per pipeline run. Asserted on observed overlap
    rather than on wall clock, because a timing assertion passes on the serial
    version whenever the machine is fast enough.
    """
    import threading

    from cdt.datasets import (
        CompletedPartition,
        date_shard_partition_path,
        load_completion_registry,
        save_completion_registry,
    )

    for month in range(1, 7):
        save_completion_registry(
            "itemize",
            {
                date_shard_partition_path(
                    "documents",
                    partition_date=f"2024-{month:02d}-15",
                    shard="0001",
                    artifact_root=tmp_path,
                ): CompletedPartition(fingerprint=f"f{month}")
            },
            artifact_root=tmp_path,
        )

    real_read = cdt_datasets.read_json_artifact
    lock = threading.Lock()
    in_flight = 0
    peak = 0
    barrier = threading.Barrier(2, timeout=10)

    def overlapping_read(path: object) -> object:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            # Two reads must be in flight at once for this to return; on the
            # serial loop it raises BrokenBarrierError at the timeout.
            barrier.wait()
        except threading.BrokenBarrierError:  # pragma: no cover - serial only
            pass
        try:
            return real_read(path)  # type: ignore[operator]
        finally:
            with lock:
                in_flight -= 1

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cdt_datasets, "read_json_artifact", overlapping_read)
        loaded = load_completion_registry("itemize", artifact_root=tmp_path)

    assert peak > 1, "shard reads were issued serially"
    # And every entry still arrives, from all six shards.
    assert len(loaded) == 6


def test_completion_registry_load_reads_only_date_shards() -> None:
    """A sibling object the S3 prefix also matches is not read as a shard.

    On S3 the shard prefix ``runs/<stage>/completed`` is a raw ``Prefix=``, so
    it also lists ``runs/<stage>/completed-partitions.json``, the pre-#191
    single object (#227). Local listing globs inside the directory and never
    sees it, so the listing is stubbed to return what S3 would.
    """
    root = "s3://bucket/root"
    shard = f"{root}/runs/itemize/completed/date=2024-01.json"
    sibling = f"{root}/runs/itemize/completed-partitions.json"
    key = "documents/date=2024-01-02/shard=0001/part-0000.parquet"
    reads: list[str] = []

    def read(path: object) -> object:
        reads.append(str(path))
        fingerprint = "shard" if path == shard else "sibling"
        return {
            "stage": "itemize",
            "version": 3,
            "partitions": {
                key: {"fingerprint": fingerprint},
                f"{key}.only-in-{fingerprint}": {"fingerprint": fingerprint},
            },
        }

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cdt_datasets, "list_artifacts", lambda *a, **k: sorted([shard, sibling])
        )
        patch.setattr(cdt_datasets, "read_json_artifact", read)
        loaded = cdt_datasets.load_completion_registry("itemize", artifact_root=root)

    assert reads == [shard]
    assert {entry.fingerprint for entry in loaded.values()} == {"shard"}


def test_completion_registry_load_merges_shards_in_sorted_order(
    tmp_path: Path,
) -> None:
    """Concurrent reads must still overlay in path order, not completion order.

    The overlay sequence is load-bearing (a later shard's entry wins), so a
    parallel read that merged in whichever order finished first would make the
    winner depend on thread scheduling. Pinned by holding the earlier shard's
    read until the later one has returned, so the reads finish in reverse path
    order, and asserting the merge still follows the sorted paths.
    """
    import threading
    import time

    from cdt.datasets import load_completion_registry

    shard_root = Path(
        cdt_datasets.completion_registry_root("itemize", artifact_root=tmp_path)
    )
    shard_root.mkdir(parents=True, exist_ok=True)
    key = "documents/date=2024-01-02/shard=0001/part-0000.parquet"
    # Two shards both claiming the same key. date=2024-02 sorts last, so its
    # value is the one a sorted overlay must leave in place.
    for label, fingerprint in (("2024-01", "first"), ("2024-02", "last")):
        (shard_root / f"date={label}.json").write_text(
            json.dumps(
                {
                    "stage": "itemize",
                    "version": 3,
                    "date_prefix": label,
                    "partitions": {key: {"fingerprint": fingerprint}},
                }
            )
        )

    real_read = cdt_datasets.read_json_artifact
    later_returned = threading.Event()
    finished: list[str] = []

    def reversing_read(path: object) -> object:
        # Finish the later shard first, so a completion-ordered merge would
        # leave "first" as the winner. The earlier read waits for the later one
        # to return, then pauses so the later read's result is fully handed
        # back to the pool before the earlier one is.
        name = Path(str(path)).name
        if name == "date=2024-01.json":
            assert later_returned.wait(timeout=10), "later shard never read"
            time.sleep(0.05)
        payload = real_read(path)  # type: ignore[operator]
        finished.append(name)
        if name == "date=2024-02.json":
            later_returned.set()
        return payload

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cdt_datasets, "read_json_artifact", reversing_read)
        loaded = load_completion_registry("itemize", artifact_root=tmp_path)

    # The premise: the reads really did finish out of path order.
    assert finished == ["date=2024-02.json", "date=2024-01.json"]
    absolute = cdt_datasets.join_artifact_path(str(tmp_path), key)
    assert loaded[absolute].fingerprint == "last"


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


def test_completion_registry_keeps_undated_keys(tmp_path: Path) -> None:
    """A key with no partition date round-trips instead of being dropped.

    Dropping it would lose completion state silently.
    """
    from cdt.datasets import (
        CompletedPartition,
        load_completion_registry,
        save_completion_registry,
    )

    # Including one that lives *under* the artifact root: the root is stripped
    # from canonical partition keys only, because the reader reattaches it to
    # canonical keys only. Strip it here too and the key changes identity on
    # the way back, stranding whatever it names.
    under_root = str(tmp_path / "documents" / "legacy-flat-file.parquet")
    save_completion_registry(
        "itemize",
        {
            "bookkeeping-key": CompletedPartition(fingerprint="f"),
            under_root: CompletedPartition(fingerprint="g"),
        },
        artifact_root=tmp_path,
    )

    assert (tmp_path / "runs" / "itemize" / "completed" / "date=unknown.json").exists()
    loaded = load_completion_registry("itemize", artifact_root=tmp_path)
    assert loaded["bookkeeping-key"].fingerprint == "f"
    assert loaded[under_root].fingerprint == "g"


def test_completion_registry_shard_saves_merge_a_real_race(tmp_path: Path) -> None:
    """Two writers racing on one shard must not lose each other's entries (#88).

    The race is forced: writer A's compare-and-swap is interposed so writer B
    lands *between* A's read and A's write, so A genuinely loses the swap and
    has to re-read. A test where the two writers merely run in sequence proves
    nothing about the retry loop.
    """
    from cdt.datasets import (
        CompletedPartition,
        date_shard_partition_path,
        load_completion_registry,
        save_completion_registry,
    )

    def key(shard: str) -> str:
        return date_shard_partition_path(
            "documents",
            partition_date="2024-01-02",
            shard=shard,
            artifact_root=tmp_path,
        )

    save_completion_registry(
        "itemize",
        {key("0000"): CompletedPartition(fingerprint="seed")},
        artifact_root=tmp_path,
    )
    writer_a = load_completion_registry("itemize", artifact_root=tmp_path)
    writer_a[key("0001")] = CompletedPartition(fingerprint="a")

    real_replace = cdt_datasets.replace_json_artifact_if_match
    interposed: list[str] = []

    def replace_after_b_writes(path: object, payload: object, *, version: str) -> bool:
        if not interposed:
            interposed.append("b")
            # B commits its own entry into the same shard, invalidating A's
            # version token; A's swap below must therefore fail and re-read.
            writer_b = load_completion_registry("itemize", artifact_root=tmp_path)
            writer_b[key("0002")] = CompletedPartition(fingerprint="b")
            save_completion_registry("itemize", writer_b, artifact_root=tmp_path)
        return real_replace(path, payload, version=version)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cdt_datasets, "replace_json_artifact_if_match", replace_after_b_writes
        )
        save_completion_registry("itemize", writer_a, artifact_root=tmp_path)

    assert interposed == ["b"]
    final = load_completion_registry("itemize", artifact_root=tmp_path)
    assert set(final) == {key("0000"), key("0001"), key("0002")}
    assert final[key("0001")].fingerprint == "a"
    assert final[key("0002")].fingerprint == "b"


def test_completion_registry_shard_gives_up_after_losing_every_race(
    tmp_path: Path,
) -> None:
    """Endless swap losses fail loudly rather than dropping the entries."""
    from cdt.datasets import CompletedPartition, save_completion_registry

    save_completion_registry(
        "itemize", {"k": CompletedPartition(fingerprint="seed")}, artifact_root=tmp_path
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cdt_datasets,
            "replace_json_artifact_if_match",
            lambda *args, **kwargs: False,
        )
        with pytest.raises(RuntimeError, match="compare-and-swap races"):
            save_completion_registry(
                "itemize",
                {"k": CompletedPartition(fingerprint="new")},
                artifact_root=tmp_path,
            )


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
    from cdt.datasets import load_completed_partitions

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


def document_keys(tmp_path: Path, days: list[str]) -> list[str]:
    """Canonical document partition keys for the given days."""
    from cdt.datasets import date_shard_partition_path

    return [
        date_shard_partition_path(
            "documents", partition_date=day, shard="0001", artifact_root=tmp_path
        )
        for day in days
    ]


def test_registry_keys_persist_without_the_artifact_root(tmp_path: Path) -> None:
    """v3 stores the dataset-relative key, not the whole path (#191).

    The root was repeated in all 440,000 entries: 129 B per entry with it,
    105 B without, measured over a full-corpus-shaped registry. The in-memory
    key is still the whole path, so none of the five call sites change.
    """
    from cdt.datasets import (
        CompletedPartition,
        completion_registry_shard_path,
        load_completion_registry,
        save_completion_registry,
    )

    key = document_keys(tmp_path, ["2024-01-02"])[0]
    save_completion_registry(
        "itemize", {key: CompletedPartition(fingerprint="f")}, artifact_root=tmp_path
    )

    payload = json.loads(
        Path(
            completion_registry_shard_path("itemize", "2024-01", artifact_root=tmp_path)
        ).read_text()
    )
    assert payload["version"] == 3
    assert list(payload["partitions"]) == [
        "documents/date=2024-01-02/shard=0001/part-0000.parquet"
    ]
    assert load_completion_registry("itemize", artifact_root=tmp_path)[key].fingerprint


def test_registry_size_no_longer_depends_on_how_deep_the_root_is(
    tmp_path: Path,
) -> None:
    """The saving is measurable, so measure it rather than asserting it.

    Two roots, one 60-odd characters deeper than the other, holding the same
    partitions: the persisted shards must come out byte-identical in size. With
    the root in the keys the deeper root paid its whole length 20 times over.
    """
    from cdt.datasets import (
        CompletedPartition,
        completion_registry_shard_path,
        save_completion_registry,
    )

    days = [f"2024-01-{day:02d}" for day in range(1, 21)]

    def shard_bytes(root: Path) -> int:
        root.mkdir(parents=True, exist_ok=True)
        save_completion_registry(
            "itemize",
            {
                key: CompletedPartition(fingerprint="1111043-1788971943321871218")
                for key in document_keys(root, days)
            },
            artifact_root=root,
        )
        return len(
            Path(
                completion_registry_shard_path("itemize", "2024-01", artifact_root=root)
            ).read_bytes()
        )

    shallow = shard_bytes(tmp_path / "a")
    deep = shard_bytes(tmp_path / ("b" * 40) / ("c" * 40))
    assert shallow == deep
    assert shallow / len(days) < 130


def test_registry_follows_a_copied_artifact_root(tmp_path: Path) -> None:
    """Dropping the root makes a registry portable, which it was not (#191).

    Before this, copying an artifact root left every key prefixed with the
    source root. Nothing in the copy matched, so the copy's corpus read as
    entirely unprocessed -- #107's failure arriving by way of `cp -r`, and the
    reason a scratch copy of an eval root could never be used to check
    completion behaviour.
    """
    import shutil

    from cdt.datasets import load_completed_partitions, pending_source_partitions

    source = tmp_path / "source"
    seed_document_partitions_across_months(
        source, [("2024-01-02", "0000"), ("2024-02-05", "0001")]
    )
    itemize_pending_documents(artifact_root=source, batch_size=5)
    assert len(load_completed_partitions("itemize", artifact_root=source)) == 2

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)

    completed = load_completed_partitions("itemize", artifact_root=copy)
    assert completed == {
        cdt_datasets.date_shard_partition_path(
            "documents", partition_date=day, shard=shard, artifact_root=copy
        )
        for day, shard in (("2024-01-02", "0000"), ("2024-02-05", "0001"))
    }
    pending, _ = pending_source_partitions("itemize", "documents", artifact_root=copy)
    assert pending == []


def test_a_v2_shard_is_normalized_on_its_next_write(tmp_path: Path) -> None:
    """A shard at the old key convention gains no duplicate spelling of a key.

    Both spellings surviving in one object would double-count the entry and,
    worse, let the stale copy win a later merge.
    """
    from cdt.datasets import (
        CompletedPartition,
        completion_registry_shard_path,
        load_completion_registry,
        save_completion_registry,
    )

    keys = document_keys(tmp_path, ["2024-01-02", "2024-01-03"])
    shard = Path(
        completion_registry_shard_path("itemize", "2024-01", artifact_root=tmp_path)
    )
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        json.dumps(
            {
                "stage": "itemize",
                "version": 2,
                "date_prefix": "2024-01",
                "partitions": {key: {"fingerprint": "old"} for key in keys},
            }
        )
    )

    save_completion_registry(
        "itemize",
        {keys[0]: CompletedPartition(fingerprint="new")},
        artifact_root=tmp_path,
    )

    payload = json.loads(shard.read_text())
    assert payload["version"] == 3
    assert sorted(payload["partitions"]) == [
        "documents/date=2024-01-02/shard=0001/part-0000.parquet",
        "documents/date=2024-01-03/shard=0001/part-0000.parquet",
    ]
    loaded = load_completion_registry("itemize", artifact_root=tmp_path)
    assert loaded[keys[0]].fingerprint == "new"
    assert loaded[keys[1]].fingerprint == "old"


def test_registry_key_prefixes_match_the_per_key_spelling(tmp_path: Path) -> None:
    """The hoisted prefixes must agree with `join_artifact_path` on every root.

    The per-key work was hoisted out of the load and save comprehensions for
    cost (#227): `join_artifact_path` built a `pathlib.Path` per key and the
    strip prefix was rebuilt per key. A hoisted prefix is only safe if it is
    byte-identical to what the per-key call produced, and the roots where it
    could differ are exactly the ones `pathlib` treats specially -- `.` and
    `""` collapse the join instead of prefixing it, so an f-string prefix would
    silently produce `./documents/...` where the real join produces
    `documents/...`, inventing a key that names no partition.
    """
    bare = "documents/date=2024-01-02/shard=0001/part-0000.parquet"
    for root in (
        str(tmp_path),
        f"{str(tmp_path)}/",
        ".",
        "",
        "/",
        "relative/root",
        "s3://bucket/prefix",
        "s3://bucket/prefix/",
        "s3://bucket",
        str(tmp_path / "a" / "very" / "deeply" / "nested" / "artifact" / "root"),
    ):
        hoisted = cdt_datasets._prepend_registry_root(  # noqa: SLF001
            bare,
            cdt_datasets._registry_join_prefix(root),  # noqa: SLF001
        )
        assert hoisted == cdt_datasets.join_artifact_path(root, bare), root
        # And the pair still inverts through the hoisted strip prefix.
        stripped = cdt_datasets._strip_registry_root(  # noqa: SLF001
            hoisted,
            cdt_datasets._registry_strip_prefix(root),  # noqa: SLF001
        )
        assert (
            cdt_datasets._prepend_registry_root(  # noqa: SLF001
                stripped,
                cdt_datasets._registry_join_prefix(root),  # noqa: SLF001
            )
            == hoisted
        ), root


def test_registry_payload_sorts_on_the_key_without_comparing_entries() -> None:
    """Two keys that relativize alike must not crash the payload build.

    The sort was over `(key, entry)` tuples, which falls through to comparing
    two `CompletedPartition` dataclasses when the keys tie -- and they are
    unordered, so it raised TypeError. Not reachable through
    `save_completion_registry` today, since `_registry_entries` absolutizes
    every stored key first, but crashing a save on a key collision is a bad
    trade for a sort key that costs nothing.

    The collision still collapses to one persisted entry, because the payload's
    `partitions` is a dict keyed on the relativized key -- that is inherent to
    the format and not what the sort key changes. What it changes is crashing
    versus a deterministic survivor: stable sort plus insertion-ordered dicts
    means the later of the tied keys wins, every time.
    """
    payload = cdt_datasets._registry_payload(  # noqa: SLF001
        "itemize",
        "2024-01",
        {
            "documents/date=2024-01-02/shard=0001/part-0000.parquet": (
                cdt_datasets.CompletedPartition(fingerprint="a")
            ),
            # Relativizes to the same bare key under the root below.
            "./documents/date=2024-01-02/shard=0001/part-0000.parquet": (
                cdt_datasets.CompletedPartition(fingerprint="b")
            ),
        },
        artifact_root=".",
    )
    assert payload["partitions"] == {
        "documents/date=2024-01-02/shard=0001/part-0000.parquet": {"fingerprint": "b"}
    }


def test_completion_registry_root_and_its_deprecated_alias_agree(
    tmp_path: Path,
) -> None:
    """The alias must keep delegating, and the shard must sit under the root.

    `completion_registry_path` returned a prefix while keeping the `_path`
    name, which this module otherwise reserves for single objects (#227). The
    alias stays only until the four stage-module call sites move. Pinned so it
    cannot silently diverge from the name it forwards to while both exist.
    """
    root = cdt_datasets.completion_registry_root("itemize", artifact_root=tmp_path)
    assert (
        cdt_datasets.completion_registry_path("itemize", artifact_root=tmp_path) == root
    )
    shard = cdt_datasets.completion_registry_shard_path(
        "itemize", "2024-01", artifact_root=tmp_path
    )
    assert shard == str(Path(root, "date=2024-01.json"))


def test_registry_key_relativizing_is_invertible(tmp_path: Path) -> None:
    """Every in-memory key shape round-trips through the persisted form unchanged.

    The pair is only safe if it is inverse: strip a root from a key the reader
    would not reattach one to and the key silently changes identity, which
    strands the partition it names. The shapes an in-memory registry can hold
    are whole paths under the root (what the dataset listings produce), whole
    paths that are not under it, S3 URIs, and non-partition bookkeeping keys.
    """
    root = str(tmp_path)
    outside = str(tmp_path.parent / "elsewhere" / "documents")
    for key in (
        cdt_datasets.date_shard_partition_path(
            "documents", partition_date="2024-01-02", shard="0001", artifact_root=root
        ),
        "s3://bucket/prefix/documents/date=2024-01-02/shard=0001/part-0000.parquet",
        f"{outside}/date=2024-01-02/shard=0001/part-0000.parquet",
        # Under the root but not a date/shard partition: a cik-sharded
        # dataset, and a pre-migration flat file. Relativizing these would
        # strip a prefix the reader will not put back, so the key changes
        # identity and the partition it names is stranded.
        cdt_datasets.cik_shard_partition_path(
            "debt-instruments", cik_shard="0001", artifact_root=root
        ),
        str(tmp_path / "documents" / "legacy-flat-file.parquet"),
        "bookkeeping-key",
        "P",
    ):
        stored = cdt_datasets._relative_registry_key(key, root)  # noqa: SLF001
        assert cdt_datasets._absolute_registry_key(stored, root) == key  # noqa: SLF001

    # A bare relative canonical key is the *persisted* spelling, so it reads
    # back as that partition under the current root -- which is exactly the
    # portability the v3 keys buy.
    bare = "documents/date=2024-01-02/shard=0001/part-0000.parquet"
    assert cdt_datasets._absolute_registry_key(bare, root) == str(  # noqa: SLF001
        tmp_path / bare
    )


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
                "dates": [
                    {
                        "kind": "maturity",
                        "evidence": ["tag-d-maturity"],
                        "normalized_date": "2031-06-30",
                    },
                    {
                        "kind": "commitment_termination",
                        "evidence": ["tag-d-draw"],
                        "normalized_date": "2027-06-30",
                    },
                ],
            }
        ]
    )
    InstrumentIEStage().postprocess(row_state)

    mention = row_state.debt_instrument_mentions[0]
    assert mention["maturity_date"] == "2031-06-30"
    assert mention["commitment_termination_date"] == "2027-06-30"
    payload = json.loads(str(mention["commitment_termination_date_json"]))
    assert payload["derived_from"] == "stated"


TERMINATION_ITEM_XML = """
<body>
On <date id="tag-d-term">June 2, 2026</date>, the Company terminated its
<debt_instrument id="tag-i-1">$3.5 billion five-year revolving credit facility</debt_instrument>,
dated as of <date id="tag-d-dated">October 11, 2023</date>.
</body>
""".strip()


def test_no_event_publishes_null_status() -> None:
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


def test_lifecycle_rollup_marks_heads_and_families() -> None:
    """Amendment chains get superseded/head markers and families (#155)."""
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
    assert new_row["is_lineage_head"] is True
    assert old_row["lineage_family_id"] == new_row["lineage_family_id"]
    assert new_row["first_seen_filing_date"] == "2024-01-02"
    assert new_row["mention_count"] == 1
    assert new_row["document_count"] == 1


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
                "dates": [
                    {
                        "kind": "closing",
                        "evidence": ["tag-d-close"],
                        "normalized_date": "2026-06-24",
                    },
                    {
                        "kind": "maturity",
                        "evidence": ["tag-t-1", "tag-d-close"],
                        "normalized_date": "2031-06-24",
                    },
                ],
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
                "kind": "closing",
                "expected": True,
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


def test_dates_facts_precision() -> None:
    """Month and year precision are read off the text."""
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


def test_entry_without_parties_names_no_lender() -> None:
    """An entry omitting `parties` discloses no lender, rather than every one.

    The prompt tells the model to omit a property the document says nothing
    about, so `{name, instrument_type, amounts}` is an ordinary response.
    """
    from cdt.extractor.core import party_payloads_and_disclosure

    parties, disclosure = party_payloads_and_disclosure(
        {"name": ["tag-i-1"], "instrument_type": "revolving_credit", "amounts": []},
        {},
    )
    assert parties == []
    assert disclosure == "none_named"


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


def test_validation_rejects_legacy_properties() -> None:
    """A response reverting to status_event/lenders/start_date fails validation."""
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
    # The published rate is canonical, not the model's spelling: verification is
    # numeric, so `5`, `5.00` and `5.000` all verified and all persisted
    # verbatim, splitting one rate across three distinct published strings.
    for spelling in ("4.1250", "4.12500"):
        assert (
            standardized_interest_rate_payload(
                {"kind": "fixed", "rate_pct": spelling, "evidence": ["tag-51"]},
                tags,
                name_text=None,
            )["rate_pct"]
            == "4.125"
        )
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
        "synthesized_by",
        "synthesized_from_mention_id",
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
        "outstanding_balance_as_of_is_filing_date",
        "outstanding_balance_source_mention_id",
        "interest_rate_kind",
        "interest_rate_pct",
        "interest_rate_source_mention_id",
        "parties_json",
        "lender_disclosure",
        "amendment_inferred_by",
        "synthesized_only",
    ]


def realistic_frame(columns: list[str]) -> pd.DataFrame:
    """Return one row of a dataset the way its real producer writes it.

    Comparing the empty frame against an all-`None` row proved nothing: both
    infer `null` for every column and the declared-type layer rescues both to
    text, so a column that really carries booleans or counts agreed with itself
    whether or not it was declared. A row from the actual writer carries those
    values as native types — an undeclared bool infers `bool` here and `string`
    in the empty frame, and the schema comparison goes red. The instrument and
    edge rows come out of `match_tables` for that reason: it populates every
    column it owns, so a flag added to the matcher cannot slip past this test.
    """
    mention = build_mention_row(
        mention_id="dim::realistic",
        item_id="item-1",
        accession_number="0001",
        cik="0000320193",
        date="2026-06-01",
        name="7% Senior Notes due 2030",
        start_date="2024-01-01",
        amount="500000000",
        parties_json=json.dumps(
            [{"role": "borrower", "canonical_name": "Example Inc.", "spans": []}]
        ),
    )
    if columns is DEBT_INSTRUMENT_MENTION_COLUMNS:
        return pd.DataFrame([mention], columns=columns)
    if columns is MENTION_CLUSTER_EDGE_COLUMNS:
        tables = match_tables(pd.DataFrame([mention]))
        return tables["debt_instrument_mentions"].reindex(columns=columns)
    if columns is DEBT_INSTRUMENT_COLUMNS:
        tables = match_tables(pd.DataFrame([mention]))
        return tables["debt_instrument"].reindex(columns=columns)
    item: dict[str, object] = dict.fromkeys(columns, "x")
    item.update({"start_line": 1, "end_line": 2, "section_char_count": 3})
    if columns is CLASSIFIED_ITEM_COLUMNS:
        item.update({"relevance": True, "classification_score": 0.9})
    return pd.DataFrame([item], columns=columns)


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(DOCUMENT_COLUMNS, id="documents"),
        pytest.param(ITEM_COLUMNS, id="items"),
        pytest.param(CLASSIFIED_ITEM_COLUMNS, id="classifications"),
        pytest.param(DEBT_INSTRUMENT_MENTION_COLUMNS, id="mentions"),
        pytest.param(MENTION_CLUSTER_EDGE_COLUMNS, id="mention-cluster-edges"),
        pytest.param(DEBT_INSTRUMENT_COLUMNS, id="debt-instruments"),
    ],
)
def test_a_columns_physical_type_does_not_depend_on_the_data(
    columns: list[str],
) -> None:
    """One column publishes one type, whatever a given partition happens to hold.

    Arrow infers an object column's type from its values, so a column with no
    value in this partition serialised as `null` and as `string` in the next.
    That made 23 of the 42 `debt-instruments` columns then published vary, and
    every standard reader — `pyarrow.dataset`, `pq.read_table`,
    `ParquetDataset`, `pandas.read_parquet` — failed on the directory with
    "Unsupported cast from string to null" (#187). An empty frame was worse: it
    inferred `null` for every column, counts and flags included.
    """
    empty = apply_declared_column_types(pd.DataFrame(columns=columns))
    populated = apply_declared_column_types(realistic_frame(columns))
    assert empty.schema == populated.schema
    assert not [
        field.name for field in empty.schema if pa.types.is_null(field.type)
    ], "a null-typed column has no stable physical type"


def test_a_multi_partition_dataset_reads_with_a_standard_reader(
    tmp_path: Path,
) -> None:
    """The consumer promise: point any parquet reader at the directory.

    The first partition read must be the one with no value: Arrow takes the
    unified type from the first fragment, and casting `null` data up to `string`
    succeeds while casting `string` data down to `null` is what fails. A test
    with the partitions the other way round passes even when the fix is removed.
    """
    root = tmp_path / "debt-instruments"
    # `split_of_debt_instrument_id` is a published nullable text column: null in
    # the first partition, a value in the second — the exact shape that failed.
    for shard, split_of in (("0001", None), ("0002", "d-0001")):
        frame = pd.DataFrame(
            [
                dict.fromkeys(DEBT_INSTRUMENT_COLUMNS)
                | {
                    "debt_instrument_id": f"d-{shard}",
                    "cik": "320193",
                    "split_of_debt_instrument_id": split_of,
                    "mention_count": 1,
                    "document_count": 1,
                    "is_lineage_head": True,
                }
            ]
        )
        write_partition_table(root, partition={"cik_shard": shard}, table=frame)

    table = pyarrow.dataset.dataset(
        root, format="parquet", partitioning="hive"
    ).to_table()
    assert table.num_rows == 2
    assert len(pd.read_parquet(root)) == 2
    assert len(read_dataset(root)) == 2


def test_a_rewrite_may_mix_read_back_decimals_with_fresh_text(tmp_path: Path) -> None:
    """A partition rewrite holds `Decimal` and text in one money column.

    Rows read back from parquet carry `Decimal`; a row built in memory carries
    the parser's text. `Table.from_pandas` refused the mixed object column
    before the declared-type layer could quantize either, which stopped the
    first backfill that appended a minted row to an existing partition (#203).
    """
    from decimal import Decimal

    frame = pd.DataFrame(
        [
            dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS)
            | {
                "debt_instrument_mention_id": "m-read-back",
                "item_id": "item-1",
                "principal_amount": Decimal("300000000.00"),
            },
            dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS)
            | {
                "debt_instrument_mention_id": "m-fresh",
                "item_id": "item-1",
                "principal_amount": "250000000",
            },
        ]
    )
    write_partition_table(
        tmp_path / "mentions",
        partition={"date": "2024-06-01", "shard": "0001"},
        table=frame,
    )
    published = read_dataset(tmp_path / "mentions").set_index(
        "debt_instrument_mention_id"
    )
    assert published.loc["m-read-back", "principal_amount"] == Decimal("300000000.00")
    assert published.loc["m-fresh", "principal_amount"] == Decimal("250000000.00")


def test_declared_decimal_columns_publish_as_exact_decimals(tmp_path: Path) -> None:
    """Money and rates publish as `decimal128`, pinned at the single write path.

    Text sorted `962500000` before `2000000000`, and float cannot hold
    `372246148.11` — the failure behind #119 (#185).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = pd.DataFrame(
        {
            "principal_amount": ["2000000000", "14881621.34", None],
            "outstanding_balance": [None, "402131.51", None],
            "interest_rate_pct": ["5", "4.375", None],
            "name": ["a", "b", "c"],
        }
    )
    written = Path(write_table(tmp_path / "t.parquet", frame))
    schema = pq.read_schema(written)
    assert schema.field("principal_amount").type == pa.decimal128(38, 2)
    assert schema.field("outstanding_balance").type == pa.decimal128(38, 2)
    assert schema.field("interest_rate_pct").type == pa.decimal128(9, 4)

    # cents survive, and the pipeline's own reader hands back one spelling
    # rather than the scale-padded form a decimal column round-trips as
    back = read_table(written)
    assert [coerce_dataset_text(v) for v in back["principal_amount"]] == [
        "2000000000",
        "14881621.34",
        None,
    ]
    assert [coerce_dataset_text(v) for v in back["interest_rate_pct"]] == [
        "5",
        "4.375",
        None,
    ]


def test_all_null_decimal_partition_keeps_its_declared_type(tmp_path: Path) -> None:
    """Pin the type for an all-null partition too.

    Otherwise the column's physical type varies from partition to partition and
    a strict reader breaks on the union.
    """
    frame = pd.DataFrame({"principal_amount": [None, None], "name": ["x", "y"]})
    import pyarrow as pa
    import pyarrow.parquet as pq

    written = Path(write_table(tmp_path / "empty.parquet", frame))
    assert pq.read_schema(written).field("principal_amount").type == pa.decimal128(
        38, 2
    )


def test_decimal_coercion_quantizes_legacy_float_error_but_refuses_junk() -> None:
    """Quantize legacy float error, but refuse text that is not a number.

    A pre-#119 partition carries float error in its text and a replay of it must
    not fail: the extra digits are the error, not the value.
    """
    import pyarrow as pa

    money = pa.decimal128(38, 2)
    assert decimal_column_values(
        ["372246148.110000014305"], money, column="principal_amount"
    ) == [Decimal("372246148.11")]
    # placeholders become null rather than an Arrow error
    assert decimal_column_values(
        ["", "nan", None], money, column="principal_amount"
    ) == [None, None, None]
    # but text that is not a number at all is an upstream bug, not drift
    with pytest.raises(ValueError, match="principal_amount is not a number"):
        decimal_column_values(["$100 million"], money, column="principal_amount")


def test_matcher_schema_version_is_pinned() -> None:
    """The version is how a downstream reader learns a rebuild is required."""
    assert MATCHER_SCHEMA_VERSION == 7


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
    # An undated balance is bounded by the filing that observed it — and says so,
    # so the substituted date is never mistaken for a stated one (#203).
    assert row["outstanding_balance_as_of"] == "2026-01-01"
    assert row["outstanding_balance_as_of_is_filing_date"] is True
    assert row["outstanding_balance_source_mention_id"] == "m-old"
    assert row["interest_rate_kind"] == "fixed"
    assert row["interest_rate_pct"] == "7.000"
    assert row["interest_rate_source_mention_id"] == "m-old"
    # A balance is never the headline amount (#140).
    assert row["principal_amount"] == "300000000"


def test_outstanding_balance_as_of_flag_tells_stated_from_substituted() -> None:
    """A stated as-of reads false; a substituted one carried forward stays true.

    The flag is what lets a consumer trust `outstanding_balance_as_of` (#203):
    without it a filing date the matcher filled in looked exactly like a date
    the filing stated. It has to survive an incremental rematch too, or the
    first run to see no new balance would silently drop it.
    """
    from cdt.matcher.core import build_debt_instrument_rows, prepare_mention

    def balance_mention(mention_id: str, as_of_date: str | None) -> object:
        return prepare_mention(
            build_mention_row(
                mention_id=mention_id,
                item_id=f"item-{mention_id}",
                accession_number="0001",
                cik="0000320193",
                date="2026-01-01",
                name="Revolving Credit Facility",
                start_date="2024-01-01",
                amount="300000000",
                amounts_json=json.dumps(
                    [
                        {
                            "kind": "outstanding_balance",
                            "normalized_amount": "270500000",
                            "currency": "USD",
                            "as_of_date": as_of_date,
                        }
                    ]
                ),
            )
        )

    stated = build_debt_instrument_rows(
        {"inst-1": ["m-stated"]},
        {"m-stated": balance_mention("m-stated", "2025-12-31")},
        {},
        existing_instruments=pd.DataFrame(),
        company_names={},
    )[0]
    assert stated["outstanding_balance_as_of"] == "2025-12-31"
    assert stated["outstanding_balance_as_of_is_filing_date"] is False

    # No member carries a balance, so every balance field comes off the row
    # written last time — including the flag, as a bool, not the text "True".
    no_balance = prepare_mention(
        build_mention_row(
            mention_id="m-later",
            item_id="item-later",
            accession_number="0002",
            cik="0000320193",
            date="2026-03-01",
            name="Revolving Credit Facility",
            start_date="2024-01-01",
            amount="300000000",
        )
    )
    existing = pd.DataFrame(
        [
            dict.fromkeys(DEBT_INSTRUMENT_COLUMNS)
            | {
                "debt_instrument_id": "inst-1",
                "seed_debt_instrument_mention_id": "m-stated",
                "cik": "0000320193",
                "outstanding_balance": "270500000",
                "outstanding_balance_as_of": "2026-01-01",
                "outstanding_balance_as_of_is_filing_date": True,
                "outstanding_balance_source_mention_id": "m-stated",
            }
        ]
    )
    carried = build_debt_instrument_rows(
        {"inst-1": ["m-later"]},
        {"m-later": no_balance},
        {},
        existing_instruments=existing,
        company_names={},
    )[0]
    assert carried["outstanding_balance_as_of"] == "2026-01-01"
    assert carried["outstanding_balance_as_of_is_filing_date"] is True


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
    """Each malformed `parties` cluster gets its own failure message."""
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


def test_a_root_matched_under_an_older_schema_forces_a_full_rematch(
    tmp_path: Path,
) -> None:
    """An incremental match over an older root publishes wrong rows, so promote it.

    `MATCHER_SCHEMA_VERSION` was written into the match manifest and read by
    nothing. Mention ids are content hashes, so a schema change that alters the
    hashed payload changes every id, and the surviving clusters are then keyed
    on ids the mentions dataset no longer holds.

    It also degraded #203 in a way that looked like success. Minting an amended
    instrument's prior state adds mentions, so on a root at an older version
    the pre-existing clusters keep the slots the mints would take on a clean
    build, and a cluster can end up holding two members that name two different
    amendment parents — which `derive_parent_links` then correctly refuses.
    Measured on `data/lineage-verify` (recorded at 4, code at 7): backfill plus
    one plain match gave 19 amendment pointers and 539 heads against a forced
    match's 22 and 536, and further plain matches never recovered it.
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
    apply_lineage_inference_pass(str(tmp_path))

    manifest_path = run_manifest_path("match", "latest", artifact_root=str(tmp_path))
    assert read_json_artifact(manifest_path)["schema_version"] == (
        MATCHER_SCHEMA_VERSION
    )
    # A root at the current version is left alone: forcing every run would turn
    # an incremental match into a full-corpus rewrite on every tick.
    assert _stale_schema_forces_rematch(str(tmp_path)) is False
    # A fresh root has no manifest and must not be forced either.
    assert _stale_schema_forces_rematch(str(tmp_path / "unwritten")) is False

    stale = read_json_artifact(manifest_path)
    stale["schema_version"] = MATCHER_SCHEMA_VERSION - 1
    write_json_artifact(manifest_path, stale)
    assert _stale_schema_forces_rematch(str(tmp_path)) is True

    # The plain call now behaves as `--force` does: the guessed pointer and its
    # provenance are dropped together rather than surviving into a corpus whose
    # identity has moved underneath them.
    match_pending_mentions(artifact_root=tmp_path, batch_size=5)
    published = {
        str(row["debt_instrument_id"]): row
        for row in read_dataset(debt_instruments_root(tmp_path)).to_dict("records")
    }
    assert published["m-2"]["amendment_of_debt_instrument_id"] is None
    assert pd.isna(published["m-2"]["amendment_inferred_by"])
    # and the manifest now records the current version, so the next plain run
    # is an ordinary incremental match again
    assert read_json_artifact(manifest_path)["schema_version"] == (
        MATCHER_SCHEMA_VERSION
    )
    assert _stale_schema_forces_rematch(str(tmp_path)) is False


def test_an_unparsed_prior_term_suppresses_inheritance_rather_than_licensing_it() -> (
    None
):
    """A stated before-figure the parser could not resolve still says this changed.

    The kind sets that answer "did this term change?" were built from the
    *parsed* prior entries, so a `prior: true` term whose value did not resolve
    was invisible to them, and the current value of that kind was copied onto
    the predecessor marked `derived_from: "inherited"` — asserting the
    post-amendment figure as the prior state's own term, the one thing the rule
    must never do. The claims decide the kinds; only the values come from what
    parsed (#211).

    Incidence of this shape on the reference corpus is 0, so no published row
    was ever wrong because of it.
    """
    from cdt.extractor.core import mint_prior_state_rows

    counters: dict[str, int] = {}
    minted = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    amounts=[
                        _fact(
                            kind="commitment",
                            normalized_amount="250000000",
                            prior=False,
                        ),
                        # stated, but the parser could not resolve it
                        _fact(kind="commitment", normalized_amount=None, prior=True),
                    ],
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2021-03-01", prior=True
                        ),
                        _fact(
                            kind="maturity", normalized_date="2029-05-01", prior=False
                        ),
                        _fact(kind="maturity", normalized_date=None, prior=True),
                    ],
                )
            ],
            counters,
        )
        if row.get("synthesized_by") == "prior_state"
    ]

    assert counters == {"minted": 1}
    assert len(minted) == 1
    # neither post-amendment value is laundered onto the predecessor
    assert minted[0]["principal_amount"] is None
    assert minted[0]["maturity_date"] is None
    assert json.loads(minted[0]["amounts_json"]) == []
    inherited = [
        entry
        for entry in json.loads(minted[0]["dates_json"])
        if entry.get("derived_from") == "inherited"
    ]
    assert inherited == []


def test_a_prior_claim_that_never_parsed_is_counted_not_silently_dropped() -> None:
    """The counters are the pre-registered yield, so a refusal cannot be silent.

    An object whose *only* prior term failed to parse fell through the combined
    "no prior amounts and no prior dates" guard with no counter at all, which
    is why the window's 22 objects carrying a prior term summed to 21 across
    the counters. On `data/genwindow-run-branch` the swallowed object is
    `dim::5542bb4c…`, a Loan and Security Agreement whose prior commitment has
    a null amount; the counters now sum to 22 (#211).
    """
    from cdt.extractor.core import mint_prior_state_rows

    counters: dict[str, int] = {}
    rows = mint_prior_state_rows(
        [
            amended_row(
                amounts=[
                    _fact(
                        kind="commitment", normalized_amount="250000000", prior=False
                    ),
                    _fact(kind="commitment", normalized_amount=None, prior=True),
                ],
                dates=[
                    _fact(kind="agreement", normalized_date="2020-02-03", prior=False),
                ],
            )
        ],
        counters,
    )

    assert len(rows) == 1
    assert counters == {"skipped_unparsed_prior": 1}


def test_two_before_figures_are_ambiguous_even_when_one_did_not_parse() -> None:
    """Ambiguity is judged on the claims: two stated before-values are two states."""
    from cdt.extractor.core import mint_prior_state_rows

    counters: dict[str, int] = {}
    rows = mint_prior_state_rows(
        [
            amended_row(
                amounts=[
                    _fact(kind="commitment", normalized_amount="300000000", prior=True),
                    _fact(kind="commitment", normalized_amount=None, prior=True),
                    _fact(
                        kind="commitment", normalized_amount="250000000", prior=False
                    ),
                ]
            )
        ],
        counters,
    )

    assert len(rows) == 1
    assert counters == {"skipped_ambiguous_prior": 1}


def test_mint_does_not_write_the_pointer_onto_the_rows_it_was_handed() -> None:
    """`amendment_of` belongs on the published row, never on the persisted state.

    Both callers happened to be safe — `published_mention_rows` copies and
    `backfill_mentions` owns its records — so this was latent rather than live.
    A future caller passing `row_state.debt_instrument_mentions` straight in
    would have persisted a minted pointer into `state.jsonl` (#211).
    """
    from cdt.extractor.core import mint_prior_state_rows

    caller_rows = [amended_row()]

    published = mint_prior_state_rows(caller_rows)

    assert caller_rows[0]["amendment_of"] is None
    successor = next(
        row
        for row in published
        if row["debt_instrument_mention_id"]
        == caller_rows[0]["debt_instrument_mention_id"]
    )
    assert successor["amendment_of"] is not None
    assert successor is not caller_rows[0]


def test_lineage_pass_does_not_infer_against_a_column_it_then_overwrites(
    tmp_path: Path,
) -> None:
    """Two passes over one unchanged corpus must agree (#211).

    `infer_amendment_parents` reads `first_seen_filing_date` as both the
    predecessor-ordering guard and the chain sort key, and the rollup rewrites
    that column from the member edges *after* the inference. So when the
    recomputed value differed from what was on disk, pass N+1 inferred against
    a different corpus than pass N.

    A row whose member edges point at mentions that no longer exist is the
    reachable case, and it arises on its own: mention ids are content hashes,
    so re-extracting an item mints a new id, the old member edge is never
    deleted, and the old instrument survives with `mention_count` 0. Its stored
    `first_seen_filing_date` is then a date no surviving mention supports.
    """
    root = _ordinal_chain_root(tmp_path)
    apply_lineage_inference_pass(root)

    # Strand a third instrument on a member edge whose mention is gone, while
    # leaving a stored observation date the surviving members cannot support.
    instruments = read_dataset(debt_instruments_root(root))
    stranded = dict(instruments.iloc[0])
    stranded.update(
        {
            "debt_instrument_id": "m-stranded",
            "seed_debt_instrument_mention_id": "m-gone",
            "name": "Amended and Restated Credit Agreement",
            "amendment_of_debt_instrument_id": None,
            "amendment_inferred_by": None,
            "superseded_by_debt_instrument_id": None,
            # Later than the successor's, so the predecessor-ordering guard
            # refuses the link while this value is believed. The rollup nulls
            # it, because no surviving mention supports it.
            "first_seen_filing_date": "2099-01-01",
            "last_seen_filing_date": "2099-01-01",
            "mention_count": 1,
        }
    )
    write_partition_table(
        debt_instruments_root(root),
        partition={"cik_shard": shard_for_cik("320193")},
        table=pd.DataFrame(
            [*instruments.to_dict("records"), stranded],
            columns=DEBT_INSTRUMENT_COLUMNS,
        ),
    )
    edges = read_dataset(mention_cluster_edges_root(root))
    stranded_edge = dict(edges.iloc[0])
    stranded_edge.update(
        {
            "debt_instrument_id": "m-stranded",
            "debt_instrument_mention_id": "m-gone",
            "edge_type": "member",
        }
    )
    write_partition_table(
        mention_cluster_edges_root(root),
        partition={"cik_shard": shard_for_cik("320193")},
        table=pd.DataFrame(
            [*edges.to_dict("records"), stranded_edge],
            columns=MENTION_CLUSTER_EDGE_COLUMNS,
        ),
    )

    first = apply_lineage_inference_pass(root)
    after_first = _published_instruments(root)
    second = apply_lineage_inference_pass(root)
    after_second = _published_instruments(root)

    pointers_first = {
        instrument_id: row["amendment_of_debt_instrument_id"]
        for instrument_id, row in after_first.items()
    }
    pointers_second = {
        instrument_id: row["amendment_of_debt_instrument_id"]
        for instrument_id, row in after_second.items()
    }
    assert first["links"] == second["links"]
    assert pointers_first == pointers_second
    # Not vacuous: the rank-1 stranded row is the parent the chain lands on,
    # and it is reachable only once its unsupported date has been recomputed.
    assert pointers_first["m-2"] == "m-stranded"
    assert after_first["m-stranded"]["first_seen_filing_date"] is None
    assert after_first["m-stranded"]["mention_count"] == 0


def test_an_unhashable_date_value_does_not_kill_the_whole_mint_pass() -> None:
    """One malformed partition row must not abort an extract or a backfill (#211).

    `{"kind": "amendment", "normalized_date": ["2020-01-01"]}` raised
    `TypeError: cannot use 'list' as a set element` out of the amendment-date
    set, taking down every remaining item in the run. No model output can reach
    it — `standardized_date_payload` overwrites `normalized_date` with this
    repo's own parser output, always `str | None` — so this is hardening for a
    tampered or hand-edited partition, and it is the failure class the
    `_borrowers` guard was written for.
    """
    from cdt.extractor.core import mint_prior_state_rows

    counters: dict[str, int] = {}
    rows = mint_prior_state_rows(
        [
            amended_row(
                dates=[
                    _fact(kind="agreement", normalized_date="2020-02-03", prior=True),
                    _fact(kind="amendment", normalized_date=["2024-06-01"]),
                ]
            )
        ],
        counters,
    )

    assert counters == {"minted": 1}
    assert len(rows) == 2


def _row_state_with_a_prior_term(item_id: str = "item-1") -> ExtractionRowState:
    """Return a terminal row state whose single mention triggers the mint."""
    row_state = ExtractionRowState(
        item_row={
            "item_id": item_id,
            "accession_number": "0001",
            "cik": "0000320193",
            "company_name": "Example Inc.",
            "date": "2024-06-01",
            "text": "amended",
        },
        stage_name="instrument_ie",
    )
    row_state.debt_instrument_mentions = [amended_row(item_id=item_id)]
    row_state.finish("SUCCESS")
    return row_state


def test_the_batch_finalize_publishes_through_the_mint_seam(tmp_path: Path) -> None:
    """`finalize_extract_outputs` must mint, not just the live loop (#211).

    The seam exists so one derivation reaches every backend at once, and the
    batch backend is the deployed default. Swapping this call site back to
    `row_state.debt_instrument_mentions` left the whole suite green, because no
    test drove this function with a mention carrying a `prior` term.
    """
    from cdt.extractor.core import finalize_extract_outputs

    finalize_extract_outputs(
        [(_row_state_with_a_prior_term(), "2024-06-01", "0001")],
        claimed={},
        run_id="20240601T000000000000Z",
        model="test-model",
        reasoning_effort="none",
        max_attempts=3,
        artifact_root=tmp_path,
    )

    written = read_dataset(mentions_root(tmp_path))
    minted = written[written["synthesized_by"] == "prior_state"]
    assert len(minted) == 1
    successor = written[written["synthesized_by"].isna()].iloc[0]
    assert successor["amendment_of"] == minted.iloc[0]["debt_instrument_mention_id"]


def test_the_batch_finalize_purges_an_item_re_extracted_to_zero_mentions(
    tmp_path: Path,
) -> None:
    """Re-extraction withdrawing every mention has to withdraw the rows (#209).

    The guard here tested `retired` alone. `retired` is built from the claimed
    sources' `prior_item_ids` minus what is still relevant, so for the target
    case -- item still relevant, re-extracted, model now returns nothing -- it
    is empty while `replaced` holds the id. `mentions.empty and not retired` was
    therefore true, the `continue` skipped the merge, and the previous pass's
    rows stayed published as facts this pass had just withdrawn.

    The suite's other pruning tests all drive the retired path instead
    (`test_extractor_sources.py`), or assert the opposite: a visited-but-empty
    partition must not be written when there is genuinely nothing to purge,
    which is the branch that survives here because `replaced` is empty too.
    """
    from cdt.extractor.core import finalize_extract_outputs

    def row_state(mentions: list[dict[str, object]]) -> ExtractionRowState:
        state = ExtractionRowState(
            item_row={
                "item_id": "item-1",
                "accession_number": "0001",
                "cik": "0000320193",
                "company_name": "Example Inc.",
                "date": "2024-06-01",
                "text": "a credit agreement",
            },
            stage_name="instrument_ie",
        )
        state.debt_instrument_mentions = mentions
        state.finish("SUCCESS")
        return state

    common = {
        "claimed": {},
        "run_id": "20240601T000000000000Z",
        "model": "test-model",
        "reasoning_effort": "none",
        "max_attempts": 3,
        "artifact_root": tmp_path,
    }
    # A plain mention, with no `prior` term: this test is about the purge, and a
    # minted predecessor would put a second row in the partition.
    mention = build_mention_row(
        mention_id="m-1",
        item_id="item-1",
        accession_number="0001",
        cik="0000320193",
        date="2024-06-01",
        name="Term Loan",
        start_date="2024-06-01",
        amount="$100 million",
    )
    finalize_extract_outputs(
        [(row_state([mention]), "2024-06-01", "0001")],
        **common,  # type: ignore[arg-type]
    )
    assert read_dataset(mentions_root(tmp_path))["item_id"].astype(str).to_list() == [
        "item-1"
    ]

    # The same still-relevant item, re-extracted to nothing.
    finalize_extract_outputs(
        [(row_state([]), "2024-06-01", "0001")],
        **common,  # type: ignore[arg-type]
    )

    written = read_dataset(mentions_root(tmp_path))
    assert written.empty, f"stale mentions survived: {written['item_id'].to_list()}"


def test_the_batch_finalize_purges_an_item_that_stopped_being_relevant(
    tmp_path: Path,
) -> None:
    """The batch guard's other input has to work too (#209).

    `_mentions_partition_needs_write` takes two reasons to purge, and the test
    above only drives one of them. `replaced` is "this item was re-extracted";
    `retired` is "this item is gone from the source" -- built in this backend
    from each claim's `prior_item_ids` minus whatever the claimed classification
    partition still marks relevant. Nothing else prunes those rows, and a
    mention whose item no longer exists still publishes, inflating the
    instrument's counts and asserting facts from text the pipeline has stopped
    sending.

    That half predates #230 and was carried through the rewrite unchanged, which
    is exactly why it needed pinning: passing `retired_item_ids=set()` at the
    batch call site left the whole suite green. The live path's equivalent is
    covered by `test_a_snippet_that_stops_being_relevant_loses_its_mentions` in
    `test_extractor_sources.py`; this backend had nothing.

    An item stops being relevant when the classifier is re-run and changes its
    verdict, or when its id ceases to exist -- the 6-K path merging several
    windows into one snippet (#172). Here the classification partition is
    rewritten with `relevance` False, which is the first case.
    """
    from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
    from cdt.extractor.core import finalize_extract_outputs

    def row_state(mentions: list[dict[str, object]]) -> ExtractionRowState:
        state = ExtractionRowState(
            item_row={
                "item_id": "item-1",
                "accession_number": "0001",
                "cik": "0000320193",
                "company_name": "Example Inc.",
                "date": "2024-06-01",
                "text": "a credit agreement",
            },
            stage_name="instrument_ie",
        )
        state.debt_instrument_mentions = mentions
        state.finish("SUCCESS")
        return state

    common = {
        "run_id": "20240601T000000000000Z",
        "model": "test-model",
        "reasoning_effort": "none",
        "max_attempts": 3,
        "artifact_root": tmp_path,
    }
    mention = build_mention_row(
        mention_id="m-1",
        item_id="item-1",
        accession_number="0001",
        cik="0000320193",
        date="2024-06-01",
        name="Term Loan",
        start_date="2024-06-01",
        amount="$100 million",
    )
    finalize_extract_outputs(
        [(row_state([mention]), "2024-06-01", "0001")],
        claimed={},
        **common,  # type: ignore[arg-type]
    )
    assert read_dataset(mentions_root(tmp_path))["item_id"].astype(str).to_list() == [
        "item-1"
    ]

    # The classifier has since changed its verdict: the item the previous pass
    # extracted is no longer relevant, so the source no longer offers it.
    classification_path = write_partition_table(
        classifications_root(tmp_path),
        partition={"date": "2024-06-01", "shard": "0001"},
        table=pd.DataFrame(
            [{"item_id": "item-1", "relevance": False}],
            columns=CLASSIFIED_ITEM_COLUMNS,
        ),
    )

    # No row entries at all: this pass claimed the partition, found nothing
    # relevant left in it, and so has only ids to withdraw.
    finalize_extract_outputs(
        [],
        claimed={classification_path: {"prior_item_ids": ["item-1"]}},
        **common,  # type: ignore[arg-type]
    )

    written = read_dataset(mentions_root(tmp_path))
    assert written.empty, f"stale mentions survived: {written['item_id'].to_list()}"


def test_extract_tables_publishes_through_the_mint_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-memory path mints too, so a notebook sees what the pipeline writes."""
    from cdt.extractor.core import extract_tables

    async def fake_run_extraction_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        return _row_state_with_a_prior_term(str(item_row["item_id"]))

    monkeypatch.setattr(
        "cdt.extractor.core.run_extraction_workflow", fake_run_extraction_workflow
    )

    tables = extract_tables(
        pd.DataFrame(
            [
                {
                    "item_id": "item-1",
                    "accession_number": "0001",
                    "cik": "0000320193",
                    "company_name": "Example Inc.",
                    "date": "2024-06-01",
                    "text": "amended",
                    "relevance": True,
                }
            ]
        ),
        artifact_root=tmp_path,
        client=None,
    )

    rows = tables["debt_instrument_mentions"]
    assert (rows["synthesized_by"] == "prior_state").sum() == 1


def test_a_prior_commitment_termination_mints_and_is_not_inherited_over() -> None:
    """`commitment_termination` is in both date frozensets, and both halves matter.

    Every mint test used `maturity` and `agreement` only, so dropping
    `commitment_termination` from `PRIOR_TERM_DATE_KINDS` or from
    `INHERITED_DATE_KINDS` left the suite green (#211). The two sets answer
    different questions: the first is which prior dates can trigger and carry
    onto the predecessor, the second is which current dates are carried forward
    as unchanged when the filing states no before-value for them.
    """
    from cdt.extractor.core import mint_prior_state_rows

    counters: dict[str, int] = {}
    minted = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2020-02-03", prior=False
                        ),
                        _fact(
                            kind="commitment_termination",
                            normalized_date="2026-02-03",
                            prior=True,
                        ),
                        _fact(
                            kind="commitment_termination",
                            normalized_date="2029-02-03",
                            prior=False,
                        ),
                        _fact(
                            kind="maturity", normalized_date="2030-02-03", prior=False
                        ),
                    ]
                )
            ],
            counters,
        )
        if row.get("synthesized_by") == "prior_state"
    ]

    assert counters == {"minted": 1}
    dates = {entry["kind"]: entry for entry in json.loads(minted[0]["dates_json"])}
    # the prior value triggers and lands on the predecessor as its own, stated
    assert dates["commitment_termination"]["normalized_date"] == "2026-02-03"
    assert dates["commitment_termination"]["derived_from"] == "stated"
    # the current maturity has no prior sibling, so it carries forward marked
    assert dates["maturity"]["normalized_date"] == "2030-02-03"
    assert dates["maturity"]["derived_from"] == "inherited"

    # And the mirror case, which is the other frozenset: with the prior value on
    # `maturity` instead, the current `commitment_termination` has no prior
    # sibling and is the one carried forward. Without `commitment_termination`
    # in `INHERITED_DATE_KINDS` the predecessor simply loses that term.
    mirrored = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2020-02-03", prior=False
                        ),
                        _fact(
                            kind="maturity", normalized_date="2027-02-03", prior=True
                        ),
                        _fact(
                            kind="maturity", normalized_date="2030-02-03", prior=False
                        ),
                        _fact(
                            kind="commitment_termination",
                            normalized_date="2029-02-03",
                            prior=False,
                        ),
                    ]
                )
            ]
        )
        if row.get("synthesized_by") == "prior_state"
    ]
    mirrored_dates = {
        entry["kind"]: entry for entry in json.loads(mirrored[0]["dates_json"])
    }
    assert mirrored_dates["maturity"]["normalized_date"] == "2027-02-03"
    assert mirrored_dates["maturity"]["derived_from"] == "stated"
    assert mirrored_dates["commitment_termination"]["normalized_date"] == "2029-02-03"
    assert mirrored_dates["commitment_termination"]["derived_from"] == "inherited"


def test_an_expected_date_is_never_inherited_onto_the_predecessor() -> None:
    """A date the filing only projects cannot be a term the earlier state had.

    No fixture carried `expected: True`, so deleting the
    `and not entry.get("expected")` guard left the suite green (#211).
    """
    from cdt.extractor.core import mint_prior_state_rows

    minted = [
        row
        for row in mint_prior_state_rows(
            [
                amended_row(
                    dates=[
                        _fact(
                            kind="agreement", normalized_date="2020-02-03", prior=True
                        ),
                        _fact(
                            kind="maturity",
                            normalized_date="2030-02-03",
                            prior=False,
                            expected=True,
                        ),
                    ]
                )
            ]
        )
        if row.get("synthesized_by") == "prior_state"
    ]

    kinds = {entry["kind"] for entry in json.loads(minted[0]["dates_json"])}
    assert "maturity" not in kinds
    assert minted[0]["maturity_date"] is None


def test_the_lineage_pass_records_a_run_manifest(tmp_path: Path) -> None:
    """The pass rewrites every published shard, so it must say so (#211).

    Every writing stage in this repo records a run manifest, and
    `docs/architecture.md` names stage manifests as a design property. The match
    manifest lists its own `partitions_written`, and then this pass rewrites
    every one of them — so without a manifest of its own, the last record of the
    `debt-instruments` dataset described a state something else had changed
    afterwards. Tolerable while the pass was opt-in behind `--infer-lineage`;
    #203 made it the unconditional default.
    """
    root = _ordinal_chain_root(tmp_path)
    stats = apply_lineage_inference_pass(root)

    manifest = read_json_artifact(
        run_manifest_path("infer-lineage", "latest", artifact_root=str(root))
    )
    assert isinstance(manifest, dict)
    assert manifest["stage"] == "infer-lineage"
    assert manifest["schema_version"] == MATCHER_SCHEMA_VERSION
    assert manifest["links"] == stats["links"]
    assert manifest["heads_after"] == stats["heads_after"]
    # the partitions it names are the ones it actually rewrote
    assert manifest["partitions_written"]
    for path in manifest["partitions_written"]:
        assert artifact_exists(path)
        assert "debt-instruments" in path


def test_magnitude_in_amount_text_is_the_magnitude_the_parser_applies() -> None:
    """The helper and the parser must not drift about what a magnitude is (#213).

    `scaled_amount_from_sibling` asks two questions the parser answers only
    implicitly, so the loop was factored out rather than copied. This pins the
    invariant that makes that safe: for every magnitude word in the table, the
    helper reports exactly the factor the parser multiplied by.
    """
    from cdt.extractor.core import (
        AMOUNT_MULTIPLIERS,
        magnitude_in_amount_text,
        normalized_amount_from_text,
    )

    for word, factor in AMOUNT_MULTIPLIERS.items():
        assert magnitude_in_amount_text(f"$1 {word}") == factor
        assert normalized_amount_from_text(f"$1 {word}") == str(factor)
    # A bare figure carries no magnitude of its own, which is the condition the
    # rescue's fourth guard tests.
    assert magnitude_in_amount_text("$400.0") is None
    assert magnitude_in_amount_text("1,050,000") is None
    assert magnitude_in_amount_text(None) is None
    # `(?<![a-z])` on the left, so a magnitude cannot match inside a word.
    assert magnitude_in_amount_text("$500mm") == 1_000_000
    assert magnitude_in_amount_text("$7 per bnillion") is None


def _shared_magnitude_tags() -> dict[str, dict[str, object]]:
    """Crescent Capital BDC's two spans, at their real offsets (#213).

    `000119312526241887-1-01`: "increased the facility size from $400.0 to
    $500.0 million".
    """
    return {
        "tag-15": {
            "type": "amount",
            "text": "$400.0",
            "char_start": 654,
            "char_end": 660,
        },
        "tag-16": {
            "type": "amount",
            "text": "$500.0 million",
            "char_start": 664,
            "char_end": 678,
        },
    }


def test_scaled_amount_from_sibling_refuses_everything_but_the_exact_product() -> None:
    """Each of the rescue's seven refusals, isolated (#213).

    Mirrors `computed_sum_amount`'s guards one for one, and the two are
    mutually exclusive: a sum needs `MINIMUM_COMPUTED_SUM_SPANS` parsed spans
    and this fires on one.
    """
    from cdt.extractor.core import scaled_amount_from_sibling

    tags = _shared_magnitude_tags()
    own, sibling = ["tag-15"], ["tag-16"]

    # The case the issue was filed for: $400.0 scaled by the sibling's shared
    # `million` is exactly the 400000000 the model returned.
    assert scaled_amount_from_sibling(own, sibling, tags, "400000000") == "400000000"

    # 1. A non-string model amount is not a value to confirm.
    assert scaled_amount_from_sibling(own, sibling, tags, 400000000) is None
    assert scaled_amount_from_sibling(own, sibling, tags, None) is None

    # 2. A model amount that is not a number cannot be compared to a product.
    assert (
        scaled_amount_from_sibling(own, sibling, tags, "four hundred million") is None
    )

    # 3. A rate is not a principal, whether it is this fact's span or a
    #    sibling's. Both halves have to bite: 0.50 x 1,000,000 is 500000, and
    #    a rate span would serve as a borrowed magnitude just as well.
    rate_tags = dict(tags) | {
        "tag-rate": {
            "type": "amount",
            "text": "0.50%",
            "char_start": 700,
            "char_end": 705,
        }
    }
    assert (
        scaled_amount_from_sibling(["tag-rate"], sibling, rate_tags, "500000") is None
    )
    assert (
        scaled_amount_from_sibling(own, ["tag-16", "tag-rate"], rate_tags, "400000000")
        is None
    )

    # 4. A span already carrying a magnitude is never rescaled. `$500.0
    #    million` means what it says; multiplying it again by the sibling's
    #    `million` would publish a figure six orders of magnitude out.
    assert scaled_amount_from_sibling(sibling, own, tags, "500000000") is None
    both_scaled = dict(tags) | {
        "tag-17": {
            "type": "amount",
            "text": "$400.0 million",
            "char_start": 654,
            "char_end": 668,
        }
    }
    assert (
        scaled_amount_from_sibling(["tag-17"], sibling, both_scaled, "400000000000000")
        is None
    )

    # 5. Exactly one distinct magnitude among the siblings, or refuse. None at
    #    all is #214's untagged `($ in thousands)` header: there is nothing
    #    cited to borrow.
    bare_tags = dict(tags) | {
        "tag-bare": {
            "type": "amount",
            "text": "$500.0",
            "char_start": 664,
            "char_end": 670,
        }
    }
    assert scaled_amount_from_sibling(own, ["tag-bare"], bare_tags, "400000000") is None
    #    And two disagreeing magnitudes: guessing between `million` and
    #    `billion` is a three-orders-of-magnitude error.
    two_tags = dict(tags) | {
        "tag-18": {
            "type": "amount",
            "text": "$2.0 billion",
            "char_start": 700,
            "char_end": 712,
        }
    }
    assert (
        scaled_amount_from_sibling(own, ["tag-16", "tag-18"], two_tags, "400000000")
        is None
    )

    # 6. The product must equal the model's value exactly -- no rounding, no
    #    tolerance, in either direction.
    assert scaled_amount_from_sibling(own, sibling, tags, "400000001") is None
    assert scaled_amount_from_sibling(own, sibling, tags, "399999999") is None
    assert scaled_amount_from_sibling(own, sibling, tags, "400000000.5") is None

    # 7. The return is the model's own value re-normalized, so the rescue only
    #    ever confirms a figure and never originates one.
    assert scaled_amount_from_sibling(own, sibling, tags, "400,000,000") == "400000000"


def test_a_shared_magnitude_word_publishes_the_prior_commitment() -> None:
    """A magnitude written once for two figures no longer drops one (#213).

    Crescent Capital BDC's amendment says the facility size went "from $400.0
    to $500.0 million". NER tags the two figures separately, the model reads
    the shared `million` correctly and returns 400000000 for the `prior`
    commitment -- and `amounts_agree` compared that against the parser's
    reading of `$400.0` alone, which is 400, so the correct value published as
    null with `validation_errors: []`.

    Re-derived from the stored responses on `data/genwindow-run-branch`, this
    is the one fact of 587 the rescue reaches, and it moves
    `skipped_unparsed_prior` 1 -> 0 and `minted` 15 -> 16 because
    `mint_prior_state_rows` builds a predecessor only out of `prior` facts that
    carry a value.
    """
    from cdt.extractor.core import standardized_amounts_payloads

    payloads = standardized_amounts_payloads(
        {
            "amounts": [
                {
                    "kind": "commitment",
                    "evidence": ["tag-15"],
                    "normalized_amount": "400000000",
                    "currency": "USD",
                    "prior": True,
                },
                {
                    "kind": "commitment",
                    "evidence": ["tag-16"],
                    "normalized_amount": "500000000",
                    "currency": "USD",
                },
            ]
        },
        _shared_magnitude_tags(),
        name_text=None,
    )
    prior, current = payloads
    assert prior["normalized_amount"] == "400000000"
    # A new marker rather than `"computed"`, which means arithmetic over
    # addends; nothing in `src/` consumes `derived_from` on the amount side.
    assert prior["derived_from"] == "scaled"
    # The currency came from this fact's own cited span and is kept as it was.
    assert prior["currency"] == "USD"
    assert prior["prior"] is True
    # The sibling that carries the magnitude is read as stated, not rescaled.
    assert current["normalized_amount"] == "500000000"
    assert current["derived_from"] == "stated"


def test_the_scale_rescue_leaves_an_untagged_unit_header_alone() -> None:
    """#214's table stays null: no fact cites the scale, so none can borrow it.

    Blue Owl Technology Income Corp (`000186945326000042-8-01`) reports its
    debt-capacity table under a bare `($ in thousands)` header. NER tags 83
    `amount` spans on that item and nothing covering the unit -- the tag
    vocabulary has no category it falls under -- so there is no cited sibling
    magnitude and this rescue cannot reach it. That is #214, deferred to
    post-Beta because it needs a NER category plus `AMOUNT_EVIDENCE_TAG_TYPES`
    plus an `instrument_ie` rule.

    Pinned as a test rather than left to the guard: re-deriving
    `data/genwindow-sol-retried` from its stored responses leaves all 18 of
    this filing's amounts null, and a change that silently started rescaling
    table cells would be out of scope and unreviewed.
    """
    from cdt.extractor.core import standardized_amounts_payloads

    payloads = standardized_amounts_payloads(
        {
            "amounts": [
                {
                    "kind": "commitment",
                    "evidence": ["tag-cell-1"],
                    "normalized_amount": "1050000000",
                    "currency": "USD",
                },
                {
                    "kind": "outstanding_balance",
                    "evidence": ["tag-cell-2"],
                    "normalized_amount": "435230000",
                    "currency": "USD",
                },
            ]
        },
        {
            "tag-cell-1": {
                "type": "amount",
                "text": "1,050,000",
                "char_start": 200,
                "char_end": 209,
            },
            "tag-cell-2": {
                "type": "amount",
                "text": "435,230",
                "char_start": 213,
                "char_end": 220,
            },
        },
        name_text=None,
    )
    assert [payload["normalized_amount"] for payload in payloads] == [None, None]
    assert [payload["derived_from"] for payload in payloads] == [None, None]


def test_the_scale_rescue_refuses_a_magnitude_on_any_own_span_not_just_canonical() -> (
    None
):
    """A fact citing its own magnitude is never rescaled, whichever span holds it.

    The bare-figure guard reads every span the fact cites. It used to ask only
    `canonical_amount_value`, which is the *longest parseable* span, so a
    magnitude sitting on any other cited span was invisible to it and the
    rescue scaled the fact anyway -- while the fact held `$500.0 million` in
    its own evidence. Multiplying a span that already carries `million` by a
    sibling's `million` is the six-orders-out figure the guard exists to
    refuse.

    Reachable from model output rather than hypothetical: unlike the single
    `amount` property, `validate_amounts_property` never calls
    `validate_standardized_single_value_cardinality`, so one `amounts[*]` entry
    may cite several spans with distinct parsed values and still validate. Of
    762 amount facts in the stored corpus, 4 cite two or more spans and 1
    carries a magnitude on a non-canonical span.
    """
    from cdt.extractor.core import canonical_amount_value, scaled_amount_from_sibling

    tags = {
        # The longer span is the bare figure, so it wins canonical selection
        # and the magnitude-bearing span is the one the old guard could not see.
        "own-bare": {
            "type": "amount",
            "text": "aggregate principal amount of $400.0",
            "char_start": 100,
            "char_end": 136,
        },
        "own-scaled": {
            "type": "amount",
            "text": "$500.0 million",
            "char_start": 140,
            "char_end": 154,
        },
        "sibling": {
            "type": "amount",
            "text": "$2.0 million",
            "char_start": 200,
            "char_end": 212,
        },
    }
    own = ["own-bare", "own-scaled"]
    # The premise: the canonical span really is the bare one, so this case
    # reaches the guard rather than being refused earlier for some other reason.
    assert canonical_amount_value(own, tags) == "aggregate principal amount of $400.0"
    # 400 x 1,000,000 is exactly the model's value, so every other refusal --
    # the rate guard, the one-distinct-magnitude guard, the exact product --
    # passes. Only reading the magnitude off `$500.0 million` refuses it.
    assert scaled_amount_from_sibling(own, ["sibling"], tags, "400000000") is None

    # The same guard covers what filtering the siblings against the fact's own
    # span texts used to catch: a sibling citing the very span that carries
    # this fact's magnitude. The filter was redundant once the guard reads
    # every own span, so it is gone and this pins the behaviour it provided.
    assert scaled_amount_from_sibling(own, ["own-scaled"], tags, "400000000") is None

    # A fact whose every cited span is a bare figure is still rescued, so the
    # widened guard did not simply switch the rescue off: the label carries no
    # magnitude, and the reading comes from the span that parses.
    labelled = {
        "own-figure": {
            "type": "amount",
            "text": "$400.0",
            "char_start": 654,
            "char_end": 660,
        },
        "own-label": {
            "type": "amount",
            "text": "Aggregate Commitment",
            "char_start": 600,
            "char_end": 620,
        },
        "sibling": {
            "type": "amount",
            "text": "$500.0 million",
            "char_start": 664,
            "char_end": 678,
        },
    }
    assert (
        scaled_amount_from_sibling(
            ["own-figure", "own-label"], ["sibling"], labelled, "400000000"
        )
        == "400000000"
    )
