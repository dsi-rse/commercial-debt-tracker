"""Tests for the extractor claiming work from both genres' sources.

The 8-K classifier and the 6-K triage stage write different datasets in the same
shape, and the extractor unions them. Both write mentions into one dataset keyed
by ``(date, shard)``, which is where the sharp edges are.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cdt.classifier.core import CLASSIFICATION_DATASET_NAME, CLASSIFIED_ITEM_COLUMNS
from cdt.datasets import (
    SIXK_SNIPPET_DATASET_NAME,
    CompletedPartition,
    load_completion_registry,
    save_completion_registry,
)
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.live import collect_pending_extract_items
from cdt.extractor.outputs import CLASSIFICATION_SOURCES, pending_extract_partitions
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.state import ExtractionRowState
from cdt.sixk.stage import SIXK_SNIPPET_COLUMNS, item_id_for
from cdt.storage import read_dataset, write_partition_table

PARTITION = {"date": "2026-09-08", "shard": "0001"}
EIGHTK_ACCESSION = "000114036126006577"
SIXK_ACCESSION = "000165495426008172"
EIGHTK_ITEM_ID = f"{EIGHTK_ACCESSION}-1-01"
SIXK_ITEM_ID = item_id_for(SIXK_ACCESSION, 0, 0, 512)


def _classified_row(item_id: str, accession_number: str) -> dict[str, object]:
    """One relevant 8-K classification row."""
    return {
        "item_id": item_id,
        "item": "1.01",
        "accession_number": accession_number,
        "cik": "320193",
        "company_name": "Example Inc.",
        "url": "https://sec.example/full.txt",
        "text": "The Company entered into a credit agreement.",
        "date": PARTITION["date"],
        "resource_uri": None,
        "item_information": "Entry into a Material Definitive Agreement",
        "extraction_status": "extracted",
        "duplicate_resolution": None,
        "section_heading": "Item 1.01",
        "start_line": 1,
        "end_line": 9,
        "section_char_count": 44,
        "label": "relevant",
        "relevance": True,
        "classification_score": 0.9,
    }


def _snippet_row(item_id: str, accession_number: str) -> dict[str, object]:
    """One relevant 6-K snippet row."""
    return {
        **_classified_row(item_id, accession_number),
        "item": f"{accession_number}:0:0",
        "cik": "904851",
        "company_name": "Example PLC",
        "item_information": None,
        "extraction_status": None,
        "section_heading": "EX-99.1",
        "start_line": None,
        "end_line": None,
        "sixk_window_start": 0,
        "sixk_window_end": 44,
        "sixk_token_count": 12,
        "sixk_verdict": "kept",
        "sixk_duplicate_of": None,
    }


def _write_classifications(tmp_path: Path) -> str:
    return write_partition_table(
        str(tmp_path / CLASSIFICATION_DATASET_NAME),
        partition=PARTITION,
        table=pd.DataFrame(
            [_classified_row(EIGHTK_ITEM_ID, EIGHTK_ACCESSION)],
            columns=CLASSIFIED_ITEM_COLUMNS,
        ),
    )


def _write_snippets(tmp_path: Path) -> str:
    return write_partition_table(
        str(tmp_path / SIXK_SNIPPET_DATASET_NAME),
        partition=PARTITION,
        table=pd.DataFrame(
            [_snippet_row(SIXK_ITEM_ID, SIXK_ACCESSION)],
            columns=SIXK_SNIPPET_COLUMNS,
        ),
    )


def _write_mentions(tmp_path: Path, item_id: str) -> str:
    """Seed a mentions partition, as an earlier extract run would have."""
    return write_partition_table(
        str(tmp_path / "mentions"),
        partition=PARTITION,
        table=pd.DataFrame(
            [
                {
                    **dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS),
                    "debt_instrument_mention_id": f"m-{item_id}",
                    "item_id": item_id,
                    "accession_number": EIGHTK_ACCESSION,
                    "cik": "320193",
                    "date": PARTITION["date"],
                    "name": "Term Loan",
                }
            ],
            columns=DEBT_INSTRUMENT_MENTION_COLUMNS,
        ),
    )


def test_both_genres_are_extraction_sources() -> None:
    """The 8-K source stays first, so its claim order is unchanged."""
    assert CLASSIFICATION_SOURCES == (
        CLASSIFICATION_DATASET_NAME,
        SIXK_SNIPPET_DATASET_NAME,
    )


def test_pending_partitions_span_both_sources(tmp_path: Path) -> None:
    """Work is claimed from either dataset, with no per-genre knowledge."""
    eightk_path = _write_classifications(tmp_path)
    sixk_path = _write_snippets(tmp_path)

    pending, _registry = pending_extract_partitions(artifact_root=tmp_path)

    assert sorted(entry.classification_path for entry in pending) == sorted(
        [eightk_path, sixk_path]
    )


def test_a_sixk_partition_is_pending_even_where_eightk_mentions_exist(
    tmp_path: Path,
) -> None:
    """Mentions at a 6-K partition's (date, shard) do not make it complete."""
    _write_classifications(tmp_path)
    sixk_path = _write_snippets(tmp_path)
    _write_mentions(tmp_path, EIGHTK_ITEM_ID)

    pending, _registry = pending_extract_partitions(artifact_root=tmp_path)

    assert sixk_path in [entry.classification_path for entry in pending]


def test_existing_mentions_without_a_registry_entry_are_reprocessed(
    tmp_path: Path,
) -> None:
    """Only the completion registry marks a partition done, not its outputs."""
    eightk_path = _write_classifications(tmp_path)
    _write_mentions(tmp_path, EIGHTK_ITEM_ID)

    pending, registry = pending_extract_partitions(artifact_root=tmp_path)

    assert [entry.classification_path for entry in pending] == [eightk_path]
    assert eightk_path not in registry


def test_forcing_reclaims_partitions_from_both_sources(tmp_path: Path) -> None:
    """--force ignores the registry for either genre."""
    eightk_path = _write_classifications(tmp_path)
    sixk_path = _write_snippets(tmp_path)
    registry = load_completion_registry("extract", artifact_root=tmp_path)
    for path in (eightk_path, sixk_path):
        registry[path] = CompletedPartition(fingerprint="stale", complete=True)
    save_completion_registry("extract", registry, artifact_root=tmp_path)

    pending, _registry = pending_extract_partitions(artifact_root=tmp_path, force=True)

    assert sorted(entry.classification_path for entry in pending) == sorted(
        [eightk_path, sixk_path]
    )


def test_collect_claims_relevant_rows_from_both_sources(tmp_path: Path) -> None:
    """The batch backend's claim covers both genres, row by row."""
    _write_classifications(tmp_path)
    _write_snippets(tmp_path)

    entries, claimed = collect_pending_extract_items(artifact_root=tmp_path)

    assert sorted(str(row["item_id"]) for row, _date, _shard in entries) == sorted(
        [EIGHTK_ITEM_ID, SIXK_ITEM_ID]
    )
    # Registry keys are full paths, so two sources sharing one (date, shard)
    # cannot collide.
    assert {Path(path).parts[-4] for path in claimed} == {
        CLASSIFICATION_DATASET_NAME,
        SIXK_SNIPPET_DATASET_NAME,
    }
    assert {
        Path(path).parts[-3:-1] == ("date=2026-09-08", "shard=0001") for path in claimed
    } == {True}


def _stub_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make extraction produce exactly one mention per claimed row."""

    async def fake_run_extraction_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        row_state = ExtractionRowState(
            item_row=item_row,  # type: ignore[arg-type]
            stage_name="instrument_ie",
        )
        row_state.debt_instrument_mentions = [
            {
                **dict.fromkeys(DEBT_INSTRUMENT_MENTION_COLUMNS),
                "debt_instrument_mention_id": f"m-{item_row['item_id']}",  # type: ignore[index]
                "item_id": item_row["item_id"],  # type: ignore[index]
                "accession_number": item_row["accession_number"],  # type: ignore[index]
                "cik": item_row["cik"],  # type: ignore[index]
                "date": item_row["date"],  # type: ignore[index]
                "name": "Term Loan",
            }
        ]
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow", fake_run_extraction_workflow
    )


def test_extract_writes_both_genres_mentions_into_one_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sources, one mentions partition, and neither genre's rows lost.

    The genres share the (date, shard) space by construction — both shard on
    crc32 of the accession — so this is the common case, not an edge one.
    """
    _write_classifications(tmp_path)
    _write_snippets(tmp_path)
    _stub_workflow(monkeypatch)

    mentions = extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    written = read_dataset(mentions_root(tmp_path))
    assert sorted(written["item_id"].astype(str)) == sorted(
        [EIGHTK_ITEM_ID, SIXK_ITEM_ID]
    )
    assert len(mentions) == 2
    # One partition file, holding both.
    assert len(list((tmp_path / "mentions").glob("**/*.parquet"))) == 1


def test_a_regrouped_snippet_is_re_extracted_and_its_retired_mentions_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merging retires item ids, and both halves of that have to be handled (#172).

    A re-triage can merge two admitted windows into one snippet. The row that
    results covers a different span, so it is a different item id, and the two
    ids it replaces no longer exist in the source. Two things then have to be
    true at once: the merged row must be extracted rather than skipped as work
    already done, and the mentions belonging to the ids that went away must be
    removed. Neither happens on its own -- extract keys completion on the item
    id, and the mentions merge only replaces ids it just extracted -- so a
    published instrument would otherwise keep counting mentions from text the
    pipeline has stopped sending.
    """
    _write_classifications(tmp_path)
    first = item_id_for(SIXK_ACCESSION, 0, 0, 512)
    second = item_id_for(SIXK_ACCESSION, 0, 512, 1024)
    write_partition_table(
        str(tmp_path / SIXK_SNIPPET_DATASET_NAME),
        partition=PARTITION,
        table=pd.DataFrame(
            [
                _snippet_row(first, SIXK_ACCESSION),
                _snippet_row(second, SIXK_ACCESSION),
            ],
            columns=SIXK_SNIPPET_COLUMNS,
        ),
    )
    _stub_workflow(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    before = read_dataset(mentions_root(tmp_path))
    assert sorted(before["item_id"].astype(str)) == sorted(
        [EIGHTK_ITEM_ID, first, second]
    )

    # The stage re-runs and the two windows merge: one row, spanning both.
    merged = item_id_for(SIXK_ACCESSION, 0, 0, 1024)
    write_partition_table(
        str(tmp_path / SIXK_SNIPPET_DATASET_NAME),
        partition=PARTITION,
        table=pd.DataFrame(
            [_snippet_row(merged, SIXK_ACCESSION)],
            columns=SIXK_SNIPPET_COLUMNS,
        ),
    )

    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    written = read_dataset(mentions_root(tmp_path))
    item_ids = sorted(written["item_id"].astype(str))
    # The merged row was extracted, not skipped.
    assert merged in item_ids
    # The ids it replaced are gone, not orphaned beside it.
    assert first not in item_ids
    assert second not in item_ids
    # The other genre is untouched.
    assert item_ids == sorted([EIGHTK_ITEM_ID, merged])


def test_a_snippet_that_stops_being_relevant_loses_its_mentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same pruning, for the simpler shape it also covers.

    Stage 2 is a non-deterministic LLM, so a re-triage can drop a snippet it
    previously kept. The row stays in the dataset marked irrelevant, so extract
    never claims it again and nothing would otherwise remove what it already
    produced.
    """
    _write_classifications(tmp_path)
    _write_snippets(tmp_path)
    _stub_workflow(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)
    assert SIXK_ITEM_ID in set(read_dataset(mentions_root(tmp_path))["item_id"])

    dropped = _snippet_row(SIXK_ITEM_ID, SIXK_ACCESSION)
    dropped["relevance"] = False
    dropped["label"] = "irrelevant"
    dropped["sixk_verdict"] = "dropped_no_details"
    write_partition_table(
        str(tmp_path / SIXK_SNIPPET_DATASET_NAME),
        partition=PARTITION,
        table=pd.DataFrame([dropped], columns=SIXK_SNIPPET_COLUMNS),
    )

    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    written = read_dataset(mentions_root(tmp_path))
    assert sorted(written["item_id"].astype(str)) == [EIGHTK_ITEM_ID]


def test_new_snippets_merge_without_dropping_the_other_genres_mentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-triaged 6-K partition adds mentions beside the 8-K ones, not over.

    Mentions are merged by replacing only the item ids just extracted, so rows
    from the genre that did not change have to survive. The two genres share the
    partition by construction — both shard on crc32 of the accession — so this
    is how a normal day behaves once ingest merges a late filing.
    """
    _write_classifications(tmp_path)
    _write_snippets(tmp_path)
    _stub_workflow(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    # Ingest merged another 6-K filing into the source documents partition, so
    # the triage stage rewrote this snippets partition with an extra row.
    second_accession = "000129281426002379"
    second_item_id = item_id_for(second_accession, 0, 0, 512)
    write_partition_table(
        str(tmp_path / SIXK_SNIPPET_DATASET_NAME),
        partition=PARTITION,
        table=pd.DataFrame(
            [
                _snippet_row(SIXK_ITEM_ID, SIXK_ACCESSION),
                _snippet_row(second_item_id, second_accession),
            ],
            columns=SIXK_SNIPPET_COLUMNS,
        ),
    )

    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)

    written = read_dataset(mentions_root(tmp_path))
    assert sorted(written["item_id"].astype(str)) == sorted(
        [EIGHTK_ITEM_ID, SIXK_ITEM_ID, second_item_id]
    )
    assert len(list((tmp_path / "mentions").glob("**/*.parquet"))) == 1


def _stub_workflow_with_no_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make extraction succeed and produce no mentions at all."""

    async def fake_run_extraction_workflow(**kwargs: object) -> ExtractionRowState:
        row_state = ExtractionRowState(
            item_row=kwargs["item_row"],  # type: ignore[arg-type]
            stage_name="instrument_ie",
        )
        row_state.debt_instrument_mentions = []
        row_state.finish("SUCCESS")
        return row_state

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow", fake_run_extraction_workflow
    )


def test_a_still_relevant_item_re_extracted_to_zero_mentions_is_purged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-extraction withdrawing every mention has to withdraw the rows (#209).

    Every other pruning test in this suite drives the *retired* path: the item
    stopped being relevant, or its id ceased to exist, so the source no longer
    carries it. This is the case none of them reached -- the item is still
    relevant and still claimed, it is re-extracted, and this time the model
    returns nothing. `retired_item_ids` is therefore empty (it is
    `done_item_ids - relevant_item_ids`, and the item is relevant) while
    `replaced_item_ids` holds the id.

    Under `--force` the merge was gated on
    `replaced_item_ids & pending.done_item_ids`, and a forced partition carries
    `done_item_ids=frozenset()` by construction -- `pending_extract_partitions`
    forgets what was done so the rows get re-extracted -- so that intersection
    was always empty and the partition fell through to the branch that counted
    it as empty and never merged. The previous pass's mentions stayed published
    as facts the new pass had just withdrawn.
    """
    _write_classifications(tmp_path)
    _stub_workflow(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None)
    assert read_dataset(mentions_root(tmp_path))["item_id"].astype(str).to_list() == [
        EIGHTK_ITEM_ID
    ]

    # The same still-relevant item, re-extracted to nothing. `force` is what
    # makes extract claim an item it has already completed, so the leak is only
    # reachable this way -- and `force` is also what empties `done_item_ids`.
    _stub_workflow_with_no_mentions(monkeypatch)
    extract_pending_items(artifact_root=tmp_path, batch_size=5, client=None, force=True)

    written = read_dataset(mentions_root(tmp_path))
    assert written.empty, f"stale mentions survived: {written['item_id'].to_list()}"
