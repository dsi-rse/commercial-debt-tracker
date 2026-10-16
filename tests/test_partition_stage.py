"""Tests for the stage loop's month writer: replace by source, defer completion, sort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import cdt.partition_stage as partition_stage
from cdt.completion import load_completion_registry
from cdt.datasets import dataset_root, document_shard
from cdt.partition_stage import PartitionOutput, run_partition_stage
from cdt.storage.objects import list_artifacts
from cdt.storage.tables import iter_partition_paths, read_dataset, write_partition_table

STAGE = "test-segment"
SOURCE = "test-documents"
OUTPUT = "test-items"
COLUMNS = ["item_id", "accession_number", "cik", "date"]


def _write_source(tmp_path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write ``(accession, cik, date)`` rows at the date/shard ingest would use."""
    table = pd.DataFrame(rows, columns=["accession_number", "cik", "date"])
    for (filing_date, shard), group in table.groupby(
        [table["date"], table["accession_number"].map(document_shard)]
    ):
        write_partition_table(
            dataset_root(SOURCE, artifact_root=str(tmp_path)),
            partition={"date": str(filing_date), "shard": str(shard)},
            table=group,
        )


def _one_item_per_document(
    source_path: str,
    partition: dict[str, str],
    done: frozenset[str],
    renew: object = None,
) -> PartitionOutput:
    del partition, done
    documents = pd.read_parquet(source_path)
    rows = documents.assign(item_id=documents["accession_number"] + "-1")
    return PartitionOutput(rows=rows[COLUMNS], source_rows=len(documents))


def _run(tmp_path: Path, process: object = _one_item_per_document) -> object:
    return run_partition_stage(
        STAGE,
        source_dataset=SOURCE,
        output_dataset=OUTPUT,
        output_columns=COLUMNS,
        process=process,  # type: ignore[arg-type]
        artifact_root=str(tmp_path),
    )


def _items(tmp_path: Path) -> pd.DataFrame:
    return read_dataset(dataset_root(OUTPUT, artifact_root=str(tmp_path)))


def test_sources_of_one_month_write_one_sorted_month_file(tmp_path: Path) -> None:
    """Many date/shard sources land in one month file, rows sorted by CIK then date."""
    _write_source(
        tmp_path,
        [
            ("acc-a", "0000000900", "2024-01-02"),
            ("acc-b", "0000000100", "2024-01-03"),
            ("acc-c", "0000000100", "2024-01-02"),
            ("acc-d", "0000000500", "2024-02-01"),
        ],
    )

    result = _run(tmp_path)

    written = list_artifacts(dataset_root(OUTPUT, artifact_root=str(tmp_path)))
    assert [Path(path).parent.name for path in written] == [
        "month=2024-01",
        "month=2024-02",
    ]
    assert len(set(result.partitions_written)) == 2
    january = pd.read_parquet(written[0])
    assert january["accession_number"].to_list() == ["acc-c", "acc-b", "acc-a"]


def test_a_late_filing_rewrites_only_its_own_sources_rows(tmp_path: Path) -> None:
    """Reprocessing one source replaces its rows and keeps the month's others."""
    _write_source(
        tmp_path,
        [("acc-a", "100", "2024-01-02"), ("acc-b", "200", "2024-01-03")],
    )
    _run(tmp_path)
    calls: list[str] = []

    def counting(
        source_path: str,
        partition: dict[str, str],
        done: frozenset[str],
        renew: object = None,
    ) -> PartitionOutput:
        calls.append(source_path)
        return _one_item_per_document(source_path, partition, done)

    # A late filing for 2024-01-02 that shares acc-a's shard, so that one file
    # changes and the other source of the month does not.
    late = next(
        f"acc-late-{index}"
        for index in range(1000)
        if document_shard(f"acc-late-{index}") == document_shard("acc-a")
    )
    _write_source(
        tmp_path, [("acc-a", "100", "2024-01-02"), (late, "100", "2024-01-02")]
    )

    _run(tmp_path, counting)

    assert len(calls) == 1
    assert sorted(_items(tmp_path)["accession_number"]) == sorted(
        ["acc-a", "acc-b", late]
    )


def test_an_emptied_source_deletes_its_rows_and_keeps_the_month_file(
    tmp_path: Path,
) -> None:
    """A source that now yields nothing takes its rows out; an empty month is written."""
    _write_source(tmp_path, [("acc-a", "100", "2024-01-02")])
    _run(tmp_path)

    def nothing(
        source_path: str,
        partition: dict[str, str],
        done: frozenset[str],
        renew: object = None,
    ) -> PartitionOutput:
        del partition, done
        return PartitionOutput(
            rows=pd.DataFrame(columns=COLUMNS),
            source_rows=len(pd.read_parquet(source_path)),
        )

    run_partition_stage(
        STAGE,
        source_dataset=SOURCE,
        output_dataset=OUTPUT,
        output_columns=COLUMNS,
        process=nothing,
        artifact_root=str(tmp_path),
        force=True,
    )

    assert _items(tmp_path).empty
    assert len(list_artifacts(dataset_root(OUTPUT, artifact_root=str(tmp_path)))) == 1


def test_completion_waits_for_the_month_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A month write that fails records no completion for its sources."""
    _write_source(tmp_path, [("acc-a", "100", "2024-01-02")])

    def failing_write(*args: object, **kwargs: object) -> str:
        raise RuntimeError("write failed")

    monkeypatch.setattr(partition_stage, "write_partition_table", failing_write)
    with pytest.raises(RuntimeError, match="write failed"):
        _run(tmp_path)

    assert not load_completion_registry(STAGE, artifact_root=str(tmp_path))


def test_a_held_output_with_a_done_set_writes_and_stays_pending(
    tmp_path: Path,
) -> None:
    """Incomplete work with done keys is written, recorded incomplete, and redone next run."""
    _write_source(tmp_path, [("acc-a", "100", "2024-01-02")])
    seen_done: list[frozenset[str]] = []

    def partial(
        source_path: str,
        partition: dict[str, str],
        done: frozenset[str],
        renew: object = None,
    ) -> PartitionOutput:
        seen_done.append(done)
        output = _one_item_per_document(source_path, partition, done)
        return PartitionOutput(
            rows=output.rows,
            source_rows=output.source_rows,
            complete=bool(done),
            done_keys=frozenset({"acc-a"}),
        )

    _run(tmp_path, partial)
    _run(tmp_path, partial)

    assert seen_done == [frozenset(), frozenset({"acc-a"})]
    assert _items(tmp_path)["accession_number"].to_list() == ["acc-a"]
    entries = load_completion_registry(STAGE, artifact_root=str(tmp_path))
    assert [entry.complete for entry in entries.values()] == [True]


def test_a_held_output_that_writes_nothing_keeps_the_earlier_done_set(
    tmp_path: Path,
) -> None:
    """Holding a source with nothing to write does not throw its done set away (#292)."""
    _write_source(tmp_path, [("acc-a", "100", "2024-01-02")])
    seen_done: list[frozenset[str]] = []
    outputs = [
        PartitionOutput(
            rows=pd.DataFrame(columns=COLUMNS),
            source_rows=1,
            done_keys=frozenset({"acc-a"}),
        ),
        PartitionOutput(
            rows=pd.DataFrame(columns=COLUMNS), source_rows=1, complete=False
        ),
        PartitionOutput(rows=pd.DataFrame(columns=COLUMNS), source_rows=1),
    ]

    def scripted(
        source_path: str, partition: dict[str, str], done: frozenset[str], renew: object
    ) -> PartitionOutput:
        seen_done.append(done)
        return outputs[len(seen_done) - 1]

    for _ in outputs:
        run_partition_stage(
            STAGE,
            source_dataset=SOURCE,
            output_dataset=OUTPUT,
            output_columns=COLUMNS,
            process=scripted,
            artifact_root=str(tmp_path),
        )
        # A changed source is pending again.
        _write_source(tmp_path, [("acc-a", str(len(seen_done)), "2024-01-02")])

    assert seen_done == [frozenset(), frozenset({"acc-a"}), frozenset({"acc-a"})]


def test_a_partition_filter_matches_whole_path_segments(tmp_path: Path) -> None:
    """``shard=0001`` is not ``cik_shard=0001``, and a month is not its prefix."""
    for partition in (
        {"cik_shard": "0001"},
        {"date": "2024-01-02", "shard": "0001"},
        {"month": "2026-01"},
        {"month": "2026-10"},
    ):
        write_partition_table(
            tmp_path, partition=partition, table=pd.DataFrame({"x": [1]})
        )

    shard_paths = list(
        iter_partition_paths(tmp_path, partition_filter={"shard": "0001"})
    )
    month_paths = list(
        iter_partition_paths(tmp_path, partition_filter={"month": "2026-1"})
    )

    assert [Path(path).parent.name for path in shard_paths] == ["shard=0001"]
    assert month_paths == []


def test_sorting_ignores_the_frame_index() -> None:
    """A frame sliced out of another keeps its labels; the sort must not use them."""
    frame = pd.DataFrame(
        {
            "cik": ["3", "1", "2"],
            "date": ["d", "d", "d"],
            "accession_number": list("abc"),
        },
        index=[10, 20, 30],
    )

    assert partition_stage.sort_partition_rows(frame)["cik"].to_list() == [
        "1",
        "2",
        "3",
    ]
