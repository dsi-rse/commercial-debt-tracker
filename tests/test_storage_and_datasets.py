"""Tests for partition paths and listings, object reads, and declared column types."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from support import build_mention_row

from cdt.classifier.core import CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import (
    existing_date_shard_partition_ids,
    normalize_cik,
    shard_for_cik,
)
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.ingest.core import DOCUMENT_COLUMNS
from cdt.itemizer.core import ITEM_COLUMNS
from cdt.matcher.schema import (
    DEBT_INSTRUMENT_COLUMNS,
    MENTION_CLUSTER_EDGE_COLUMNS,
)
from cdt.matcher.stage import match_tables
from cdt.storage import objects as storage_objects
from cdt.storage.columns import (
    apply_declared_column_types,
    coerce_dataset_text,
    decimal_column_values,
)
from cdt.storage.objects import get_object_bytes
from cdt.storage.tables import (
    read_dataset,
    read_table,
    write_partition_table,
    write_table,
)


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
    monkeypatch.setattr(storage_objects, "sleep", lambda seconds: None)
    client = _FlakyS3Client(b"payload", read_failures=2)

    assert get_object_bytes(client, "bucket", "key") == b"payload"
    assert client.get_object_calls == 3


def test_get_object_bytes_gives_up_after_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent stream failure must surface, not retry forever (#112)."""
    monkeypatch.setattr(storage_objects, "sleep", lambda seconds: None)
    client = _FlakyS3Client(b"payload", read_failures=99)

    with pytest.raises(ReadTimeoutError):
        get_object_bytes(client, "bucket", "key")

    assert client.get_object_calls == storage_objects._GET_OBJECT_ATTEMPTS


def test_get_object_bytes_does_not_retry_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permanent errors (NoSuchKey, AccessDenied) must propagate immediately (#112)."""
    monkeypatch.setattr(storage_objects, "sleep", lambda seconds: None)
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


def test_read_table_projects_columns_and_tolerates_missing_ones(
    tmp_path: Path,
) -> None:
    """Column projection is pushed down; absent columns reindex instead of raising (#69)."""
    from cdt.storage.tables import write_table

    path = tmp_path / "table.parquet"
    write_table(path, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    projected = read_table(path, ["a"])
    assert list(projected.columns) == ["a"]

    tolerant = read_table(path, ["a", "missing"])
    assert list(tolerant.columns) == ["a", "missing"]
    assert tolerant["missing"].isna().all()


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
