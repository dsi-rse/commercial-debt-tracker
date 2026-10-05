"""The publish gate's digest: what moves it, and when it is recorded (#110)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cdt import pipeline as pipeline_module
from cdt.pipeline import (
    FINAL_OUTPUT_TABLES,
    PUBLISH_SOURCE_DIGEST_KEY,
    final_pointer_path,
    publish_source_digest,
    publish_would_republish_nothing,
    write_final_output_tables,
)
from cdt.storage import read_json_artifact, read_table, write_partition_table

_PARTITION = {"date": "2022-01-02", "shard": "0001"}


def _items(artifact_root: Path, item_id: str = "item-1") -> None:
    write_partition_table(
        artifact_root / "items",
        partition=_PARTITION,
        table=pd.DataFrame([{"item_id": item_id}]),
    )


def _skips(artifact_root: Path, final_root: Path) -> bool:
    return publish_would_republish_nothing(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )


def _every_source_root(artifact_root: Path) -> list[str]:
    """Every dataset root the publish reads, straight from FINAL_OUTPUT_TABLES.

    Derived independently of ``_publish_source_roots`` on purpose: a test that
    asked that function which roots to check would agree with it however many
    it dropped.
    """
    roots: list[str] = []
    for entry in FINAL_OUTPUT_TABLES.values():
        for dataset_root_fn in entry if isinstance(entry, tuple) else (entry,):
            root = dataset_root_fn(str(artifact_root))
            if root not in roots:
                roots.append(root)
    return roots


def test_a_publish_interrupted_after_the_pointer_publishes_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash refreshing ``latest.parquet`` must not read as published (#222).

    The pointer flips before the four database objects are rewritten. If it
    already carried the new digest, a crash in that loop left a matching digest
    beside the *previous* generation's tables, and every later run skipped until
    some source happened to move. The digest is recorded only once all four are
    written, so the interrupted publish leaves none and the next run publishes.
    """
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _items(artifact_root)
    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )
    assert _skips(artifact_root, final_root)

    _items(artifact_root, "item-2")
    real_write_table = pipeline_module.write_table

    def dies_on_the_database_root(path: object, table: pd.DataFrame) -> str:
        if str(path).startswith(str(final_root)):
            msg = "simulated crash refreshing the database root"
            raise RuntimeError(msg)
        return real_write_table(path, table)

    monkeypatch.setattr(pipeline_module, "write_table", dies_on_the_database_root)
    with pytest.raises(RuntimeError, match="simulated crash"):
        write_final_output_tables(
            artifact_root=str(artifact_root), final_database_root=str(final_root)
        )
    monkeypatch.setattr(pipeline_module, "write_table", real_write_table)

    pointer = read_json_artifact(final_pointer_path(str(artifact_root)))
    assert not pointer.get(PUBLISH_SOURCE_DIGEST_KEY)
    assert not _skips(artifact_root, final_root)

    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )
    live = read_table(str(final_root / "items" / "latest.parquet"))
    assert live["item_id"].to_list() == ["item-2"]
    assert _skips(artifact_root, final_root)


def test_a_source_written_while_the_publish_reads_is_not_recorded_as_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest describes the bytes before the reads, never after (#222).

    Taken after them, a partition written mid-publish would be recorded as
    already published while the tables were built without it, and the next
    run would skip it. Taken before, it reads as moved: one redundant publish.
    """
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _items(artifact_root)
    real_read_dataset = pipeline_module.read_dataset
    writes: list[str] = []

    def a_writer_lands_mid_read(*args: object, **kwargs: object) -> pd.DataFrame:
        if not writes:
            writes.append("late")
            write_partition_table(
                artifact_root / "items",
                partition={"date": "2022-01-03", "shard": "0001"},
                table=pd.DataFrame([{"item_id": "item-late"}]),
            )
        return real_read_dataset(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "read_dataset", a_writer_lands_mid_read)
    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )

    assert writes == ["late"]
    assert not _skips(artifact_root, final_root)


@pytest.mark.parametrize(
    "constant", ["PUBLISH_FORMAT_VERSION", "MATCHER_SCHEMA_VERSION"]
)
def test_a_publisher_change_moves_the_digest_without_any_source_moving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    """The same bytes published differently are a different snapshot (#223).

    The source digest cannot see code, so a deploy that reshapes a published
    table would otherwise skip until some partition happened to move, leaving
    the old shape live.
    """
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _items(artifact_root)
    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )
    assert _skips(artifact_root, final_root)

    monkeypatch.setattr(
        pipeline_module, constant, getattr(pipeline_module, constant) + 1
    )

    assert not _skips(artifact_root, final_root)


def test_rerouting_a_table_to_another_root_moves_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which roots feed which table is part of the publish, not only the bytes (#223)."""
    artifact_root = tmp_path / "artifacts"
    before = publish_source_digest(str(artifact_root))
    rerouted = dict(FINAL_OUTPUT_TABLES)
    rerouted["items"], rerouted["debt-instruments"] = (
        rerouted["debt-instruments"],
        rerouted["items"],
    )
    monkeypatch.setattr(pipeline_module, "FINAL_OUTPUT_TABLES", rerouted)

    assert publish_source_digest(str(artifact_root)) != before


def test_every_source_root_moves_the_digest(tmp_path: Path) -> None:
    """A root silently dropped from the digest would skip publishes it should not.

    Lineage rewrites only ``debt-instruments``; if that root fell out, the
    publish would skip and leave un-inferred lineage live — #170 again, with
    nothing failing. So each root the publish reads is changed on its own.
    """
    artifact_root = tmp_path / "artifacts"
    roots = _every_source_root(artifact_root)
    assert len(roots) == len(set(roots)) >= len(FINAL_OUTPUT_TABLES)

    for index, root in enumerate(roots):
        before = publish_source_digest(str(artifact_root))
        write_partition_table(
            root,
            partition={"date": "2022-01-02", "shard": f"{index:04d}"},
            table=pd.DataFrame([{"id": f"row-{index}"}]),
        )
        assert publish_source_digest(str(artifact_root)) != before, root


def test_rewriting_a_partition_in_place_moves_the_digest(tmp_path: Path) -> None:
    """The common change is a rewrite at the same path, not a new partition.

    Ingest merges into existing date partitions (#62) and match rewrites its
    shards where they are. Every other gate test adds a partition, which a
    digest of paths alone, or of sizes, would also notice; this one keeps the
    path and the byte count and changes only the content.
    """
    artifact_root = tmp_path / "artifacts"
    final_root = tmp_path / "final"
    _items(artifact_root, "item-1")
    write_final_output_tables(
        artifact_root=str(artifact_root), final_database_root=str(final_root)
    )
    assert _skips(artifact_root, final_root)
    (partition,) = (artifact_root / "items").rglob("*.parquet")
    size = partition.stat().st_size

    _items(artifact_root, "item-2")

    assert list((artifact_root / "items").rglob("*.parquet")) == [partition]
    assert partition.stat().st_size == size
    assert not _skips(artifact_root, final_root)
