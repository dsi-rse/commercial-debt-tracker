"""Tests for the completion registry: sharding, keys, concurrent saves and pending selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from support import (
    _fake_success_workflow,
    _seed_classifications,
    seed_document_partitions_across_months,
)

from cdt import completion as cdt_completion
from cdt import datasets as cdt_datasets
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.state import ExtractionRowState
from cdt.segmenter.eightk import segment_pending_eightk_documents
from cdt.storage.tables import (
    read_dataset,
    write_partition_table,
)


def test_pending_source_partitions_skips_orphans_and_raises_on_flat_files(
    tmp_path: Path,
) -> None:
    """Fingerprint work selection follows the same stray contract as the scan.

    Silently dropping a mis-laid-out real file here would run the stage on
    nothing while ingest keeps counting the file's rows as ingested.
    """
    from cdt.completion import pending_source_partitions

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


def test_pending_source_partitions_reprocesses_outputs_without_a_registry_entry(
    tmp_path: Path,
) -> None:
    """A target partition at the same coordinates does not mark its source done."""
    from cdt.completion import pending_source_partitions

    root = tmp_path / "artifacts"
    partition = {"date": "2026-01-02", "shard": "0007"}
    table = pd.DataFrame({"item_id": ["a"], "text": ["x"]})
    source_path = write_partition_table(
        str(root / "items"), partition=partition, table=table
    )
    write_partition_table(
        str(root / "classifications"), partition=partition, table=table
    )

    pending, registry = pending_source_partitions(
        "classify", "items", artifact_root=str(root)
    )

    assert [path for path, _fingerprint in pending] == [source_path]
    assert source_path not in registry


def test_completion_registry_saves_merge_concurrent_updates(tmp_path: Path) -> None:
    """Overlapping writers must not lose each other's registry entries (#88).

    A lost entry silently strands a partition (or fake-completes it with empty
    item_ids), so saves overlay only the entries a run changed onto the freshest
    persisted state instead of overwriting the file with a stale snapshot.
    """
    from cdt.completion import (
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


def test_completion_registry_shards_entries_by_date_prefix(tmp_path: Path) -> None:
    """One object per year-month, each holding only that month's entries (#191).

    The single object it replaces is 56.8 MB at full corpus scale and was read
    and rewritten whole every 100 partitions, 4,400 times per itemize pass.
    """
    from cdt.completion import (
        CompletedPartition,
        completion_registry_shard_path,
        load_completion_registry,
        save_completion_registry,
    )
    from cdt.datasets import date_shard_partition_path

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
    from cdt.completion import (
        CompletedPartition,
        completion_registry_shard_path,
        load_completion_registry,
        save_completion_registry,
    )
    from cdt.datasets import date_shard_partition_path

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
                cdt_completion,
                attribute,
                record(name, getattr(cdt_completion, attribute)),
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
    from cdt.completion import (
        CompletedPartition,
        CompletionRegistry,
        save_completion_registry,
    )
    from cdt.datasets import date_shard_partition_path

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
                cdt_completion, attribute, record(getattr(cdt_completion, attribute))
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

    from cdt.completion import (
        CompletedPartition,
        load_completion_registry,
        save_completion_registry,
    )
    from cdt.datasets import date_shard_partition_path

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

    real_read = cdt_completion.read_json_artifact
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
        patch.setattr(cdt_completion, "read_json_artifact", overlapping_read)
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
            cdt_completion, "list_artifacts", lambda *a, **k: sorted([shard, sibling])
        )
        patch.setattr(cdt_completion, "read_json_artifact", read)
        loaded = cdt_completion.load_completion_registry("itemize", artifact_root=root)

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

    from cdt.completion import load_completion_registry

    shard_root = Path(
        cdt_completion.completion_registry_root("itemize", artifact_root=tmp_path)
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

    real_read = cdt_completion.read_json_artifact
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
        patch.setattr(cdt_completion, "read_json_artifact", reversing_read)
        loaded = load_completion_registry("itemize", artifact_root=tmp_path)

    # The premise: the reads really did finish out of path order.
    assert finished == ["date=2024-02.json", "date=2024-01.json"]
    absolute = cdt_completion.join_artifact_path(str(tmp_path), key)
    assert loaded[absolute].fingerprint == "last"


def test_completion_registry_keeps_undated_keys(tmp_path: Path) -> None:
    """A key with no partition date round-trips instead of being dropped.

    Dropping it would lose completion state silently.
    """
    from cdt.completion import (
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
    from cdt.completion import (
        CompletedPartition,
        load_completion_registry,
        save_completion_registry,
    )
    from cdt.datasets import date_shard_partition_path

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

    real_replace = cdt_completion.replace_json_artifact_if_match
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
            cdt_completion, "replace_json_artifact_if_match", replace_after_b_writes
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
    from cdt.completion import CompletedPartition, save_completion_registry

    save_completion_registry(
        "itemize", {"k": CompletedPartition(fingerprint="seed")}, artifact_root=tmp_path
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cdt_completion,
            "replace_json_artifact_if_match",
            lambda *args, **kwargs: False,
        )
        with pytest.raises(RuntimeError, match="compare-and-swap races"):
            save_completion_registry(
                "itemize",
                {"k": CompletedPartition(fingerprint="new")},
                artifact_root=tmp_path,
            )


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
    from cdt.completion import (
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
    from cdt.completion import (
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

    from cdt.completion import load_completed_partitions, pending_source_partitions

    source = tmp_path / "source"
    seed_document_partitions_across_months(
        source, [("2024-01-02", "0000"), ("2024-02-05", "0001")]
    )
    segment_pending_eightk_documents(artifact_root=source, batch_size=5)
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
    from cdt.completion import (
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
        hoisted = cdt_completion._prepend_registry_root(  # noqa: SLF001
            bare,
            cdt_completion._registry_join_prefix(root),  # noqa: SLF001
        )
        assert hoisted == cdt_completion.join_artifact_path(root, bare), root
        # And the pair still inverts through the hoisted strip prefix.
        stripped = cdt_completion._strip_registry_root(  # noqa: SLF001
            hoisted,
            cdt_completion._registry_strip_prefix(root),  # noqa: SLF001
        )
        assert (
            cdt_completion._prepend_registry_root(  # noqa: SLF001
                stripped,
                cdt_completion._registry_join_prefix(root),  # noqa: SLF001
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
    payload = cdt_completion._registry_payload(  # noqa: SLF001
        "itemize",
        "2024-01",
        {
            "documents/date=2024-01-02/shard=0001/part-0000.parquet": (
                cdt_completion.CompletedPartition(fingerprint="a")
            ),
            # Relativizes to the same bare key under the root below.
            "./documents/date=2024-01-02/shard=0001/part-0000.parquet": (
                cdt_completion.CompletedPartition(fingerprint="b")
            ),
        },
        artifact_root=".",
    )
    assert payload["partitions"] == {
        "documents/date=2024-01-02/shard=0001/part-0000.parquet": {"fingerprint": "b"}
    }


def test_completion_registry_shards_sit_under_the_registry_root(
    tmp_path: Path,
) -> None:
    """A registry shard is one object under its stage's registry prefix."""
    root = cdt_completion.completion_registry_root("itemize", artifact_root=tmp_path)
    shard = cdt_completion.completion_registry_shard_path(
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
        stored = cdt_completion._relative_registry_key(key, root)  # noqa: SLF001
        assert cdt_completion._absolute_registry_key(stored, root) == key  # noqa: SLF001

    # A bare relative canonical key is the *persisted* spelling, so it reads
    # back as that partition under the current root -- which is exactly the
    # portability the v3 keys buy.
    bare = "documents/date=2024-01-02/shard=0001/part-0000.parquet"
    assert cdt_completion._absolute_registry_key(bare, root) == str(  # noqa: SLF001
        tmp_path / bare
    )


def test_infrastructure_error_aborts_and_preserves_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider failure stops the run; terminal rows are never re-paid (#49)."""
    from cdt.completion import load_completion_registry
    from cdt.extractor.state import InfrastructureError

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

    monkeypatch.setattr("cdt.extractor.live.run_extraction_workflow", failing_workflow)
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
