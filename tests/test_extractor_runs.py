"""Tests for extraction runs end to end: backfill, partition growth and finalize."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from support import (
    _fake_success_workflow,
    _seed_classifications,
    amended_row,
    build_mention_row,
)

from cdt.classifier.core import classifications_root
from cdt.extractor import extract_pending_items, mentions_root
from cdt.extractor.schema import DEBT_INSTRUMENT_MENTION_COLUMNS
from cdt.extractor.state import ExtractionRowState
from cdt.storage.tables import (
    read_dataset,
    write_partition_table,
)


def test_backfill_mints_over_existing_partitions_and_is_a_no_op_twice(
    tmp_path: Path,
) -> None:
    """A partition written before #203 gains its prior states, once."""
    from cdt.extractor.outputs import backfill_mentions
    from cdt.extractor.prior_state import published_mention_rows
    from cdt.extractor.state import ExtractionRowState

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
    from cdt.extractor.outputs import backfill_mentions

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
    from cdt.extractor.outputs import finalize_extract_outputs

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
    from cdt.extractor.outputs import finalize_extract_outputs

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
    from cdt.extractor.outputs import finalize_extract_outputs

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
    from cdt.extractor.live import extract_tables

    async def fake_run_extraction_workflow(**kwargs: object) -> ExtractionRowState:
        item_row = kwargs["item_row"]
        return _row_state_with_a_prior_term(str(item_row["item_id"]))

    monkeypatch.setattr(
        "cdt.extractor.live.run_extraction_workflow", fake_run_extraction_workflow
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
