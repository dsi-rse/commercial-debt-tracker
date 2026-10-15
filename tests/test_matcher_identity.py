"""Tests for matching as a pure function of a CIK's mentions, and for instrument ids."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from support import build_mention_row

from cdt.matcher import debt_instruments_root, match_pending_mentions
from cdt.matcher.identity import (
    INSTRUMENT_ID_PREFIX,
    PreviousClusters,
    assign_instrument_ids,
    mint_instrument_id,
    updated_redirects,
)
from cdt.matcher.normalize import prepare_mention
from cdt.matcher.schema import (
    debt_instrument_redirects_root,
    mention_cluster_edges_root,
)
from cdt.storage.tables import read_dataset, write_partition_table


def _corpus() -> pd.DataFrame:
    """Mentions of two issuers over four months: merges, lineage and singletons."""
    rows = []

    def add(
        index: int,
        cik: str,
        date: str,
        name: str,
        start: str,
        amount: str,
        **kw: object,
    ) -> None:
        rows.append(
            build_mention_row(
                mention_id=f"m-{index:02d}",
                item_id=f"item-{index:02d}",
                accession_number=f"acc-{index:02d}",
                cik=cik,
                date=date,
                name=name,
                start_date=start,
                amount=amount,
                **kw,
            )
        )

    add(
        1,
        "320193",
        "2024-01-05",
        "5.25% Senior Notes due 2030",
        "2024-01-05",
        "$500 million",
    )
    add(
        2,
        "320193",
        "2024-02-07",
        "5.25% Senior Notes due 2030",
        "2024-01-05",
        "$500 million",
    )
    add(
        3,
        "320193",
        "2024-03-09",
        "5.25% senior notes due 2030",
        "2024-01-08",
        "$550 million",
    )
    add(4, "320193", "2024-01-15", "Credit Agreement", "2020-01-01", "$1 billion")
    add(
        5,
        "320193",
        "2024-04-02",
        "Second Amended and Restated Credit Agreement",
        "2024-04-01",
        "$1.5 billion",
    )
    add(6, "320193", "2024-02-20", "Term Loan", "2024-02-01", "$200 million")
    add(7, "320193", "2024-03-20", "Term Loan", "2024-02-01", "$200 million")
    add(8, "320193", "2024-04-20", "Term Loan", "2024-04-01", "$300 million")
    add(
        9,
        "789019",
        "2024-01-10",
        "Revolving Credit Facility",
        "2023-12-01",
        "$2 billion",
    )
    add(
        10,
        "789019",
        "2024-03-10",
        "Revolving Credit Facility",
        "2023-12-01",
        "$2 billion",
    )
    add(
        11,
        "0000789019",
        "2024-04-10",
        "6% Notes due 2034",
        "2024-04-10",
        "$750 million",
    )
    add(12, "789019", "2024-02-10", "6% Notes due 2034", "2024-04-10", "$750 million")
    return pd.DataFrame(rows)


def _write(root: Path, mentions: pd.DataFrame) -> None:
    """Replace the root's mentions with ``mentions``, one file per month."""
    shutil.rmtree(root / "mentions", ignore_errors=True)
    for month, rows in mentions.groupby(mentions["date"].str[:7]):
        write_partition_table(root / "mentions", partition={"month": month}, table=rows)


def _clusters(root: Path) -> set[frozenset[str]]:
    """Return the root's clusters as sets of member mention ids."""
    edges = read_dataset(mention_cluster_edges_root(root))
    members = edges.loc[edges["edge_type"] == "member"]
    return {
        frozenset(group["debt_instrument_mention_id"])
        for _, group in members.groupby("debt_instrument_id")
    }


def _instruments(root: Path) -> pd.DataFrame:
    table = read_dataset(debt_instruments_root(root))
    return table.sort_values("debt_instrument_id").reset_index(drop=True)


def _arrive_in(root: Path, batches: list[pd.DataFrame]) -> None:
    """Match after each batch of mentions arrives, the batches accumulating."""
    seen = pd.DataFrame()
    for batch in batches:
        seen = pd.concat([seen, batch], ignore_index=True)
        _write(root, seen)
        match_pending_mentions(artifact_root=root)


def _orders() -> dict[str, list[pd.DataFrame]]:
    corpus = _corpus()
    months = [rows for _, rows in corpus.groupby(corpus["date"].str[:7])]
    halves = [
        corpus.loc[corpus["date"] < "2024-03"],
        corpus.loc[corpus["date"] >= "2024-03"],
    ]
    return {
        "all at once": [corpus],
        "forward by month": months,
        "reverse by month": months[::-1],
        "two halves": halves,
        "backfill, newer half first": halves[::-1],
    }


@pytest.mark.parametrize("order", list(_orders()))
def test_clusters_do_not_depend_on_the_order_mentions_arrive_in(
    tmp_path: Path, order: str
) -> None:
    """Every arrival order gives the clusters a single full match gives (#258)."""
    reference = tmp_path / "reference"
    _arrive_in(reference, [_corpus()])
    root = tmp_path / "root"

    _arrive_in(root, _orders()[order])

    assert _clusters(root) == _clusters(reference)
    lineage_columns = ["mention_count", "is_lineage_head", "amendment_inferred_by"]
    assert sorted(
        map(tuple, _instruments(root)[lineage_columns].astype(str).values)
    ) == sorted(map(tuple, _instruments(reference)[lineage_columns].astype(str).values))


@pytest.mark.parametrize("order", list(_orders()))
def test_a_remint_makes_ids_independent_of_arrival_order(
    tmp_path: Path, order: str
) -> None:
    """Carried ids depend on history; after `--remint` they are content ids (#259)."""
    reference = tmp_path / "reference"
    _arrive_in(reference, [_corpus()])
    match_pending_mentions(artifact_root=reference, remint=True)
    root = tmp_path / "root"
    _arrive_in(root, _orders()[order])

    match_pending_mentions(artifact_root=root, remint=True)

    pd.testing.assert_frame_equal(_instruments(root), _instruments(reference))


def test_ids_are_minted_from_content_and_carried_through_a_backfill(
    tmp_path: Path,
) -> None:
    """A backfilled older filing joins a cluster without changing its id."""
    corpus = _corpus()
    newer = corpus.loc[corpus["date"] >= "2024-02"]
    _arrive_in(tmp_path, [newer])
    before = _instruments(tmp_path)
    assert before["debt_instrument_id"].str.startswith(INSTRUMENT_ID_PREFIX).all()
    notes_id = before.loc[
        before["seed_debt_instrument_mention_id"] == "m-02", "debt_instrument_id"
    ].item()

    _arrive_in(tmp_path, [corpus])

    after = _instruments(tmp_path)
    notes = after.loc[after["debt_instrument_id"] == notes_id].iloc[0]
    # m-01 is older than m-02, so it now seeds the cluster; the id stays.
    assert notes["seed_debt_instrument_mention_id"] == "m-01"
    assert notes["mention_count"] == 3


def test_a_split_cluster_keeps_its_id_on_one_part(tmp_path: Path) -> None:
    """A cluster that splits keeps its id on one part; the other part gets a new id."""
    first = [
        build_mention_row(
            mention_id="m-a",
            item_id="item-a",
            accession_number="0001",
            cik="320193",
            date="2024-01-01",
            name="Alpha Loan",
            start_date="2024-01-01",
            amount="$100 million",
        ),
        build_mention_row(
            mention_id="m-b",
            item_id="item-b",
            accession_number="0002",
            cik="320193",
            date="2024-01-02",
            name="Beta Facility",
            start_date="2024-01-01",
            amount="$100 million",
        ),
    ]
    _write(tmp_path, pd.DataFrame(first))
    match_pending_mentions(
        artifact_root=tmp_path, strong_match_threshold=0.75, loose_match_threshold=0.75
    )
    [merged_id] = _instruments(tmp_path)["debt_instrument_id"]

    match_pending_mentions(artifact_root=tmp_path, force=True)

    after = _instruments(tmp_path).set_index("seed_debt_instrument_mention_id")
    assert after.loc["m-a", "debt_instrument_id"] == merged_id
    assert after.loc["m-b", "debt_instrument_id"] != merged_id
    assert read_dataset(debt_instrument_redirects_root(tmp_path)).empty


def _prepared(mention_id: str, item_id: str, date: str = "2024-01-01") -> object:
    return prepare_mention(
        build_mention_row(
            mention_id=mention_id,
            item_id=item_id,
            accession_number=f"acc-{item_id}",
            cik="0000320193",
            date=date,
            name="Term Loan",
            start_date="2024-01-01",
            amount="$1 million",
        )
    )


def test_ids_carry_by_mention_overlap_then_by_item_overlap() -> None:
    """A re-extracted item has new mention ids but the same item id, so it keeps its cluster."""
    index = {
        "m-1": _prepared("m-1", "item-1"),
        "m-2-re": _prepared("m-2-re", "item-2", "2024-02-01"),
        "m-3": _prepared("m-3", "item-3", "2024-03-01"),
    }
    previous = PreviousClusters(
        mention_ids={
            "di::kept": frozenset({"m-1"}),
            "di::reextracted": frozenset({"m-2"}),
        },
        item_ids={
            "di::kept": frozenset({"item-1"}),
            "di::reextracted": frozenset({"item-2"}),
        },
    )

    final, redirects = assign_instrument_ids(
        {"p-1": ["m-1"], "p-2": ["m-2-re"], "p-3": ["m-3"]},
        cik="0000320193",
        mention_index=index,
        previous=previous,
    )

    assert final["p-1"] == "di::kept"
    assert final["p-2"] == "di::reextracted"
    assert final["p-3"] == mint_instrument_id(
        "0000320193", ("2024-03-01", "acc-item-3", "item-3", 1, "m-3")
    )
    assert redirects == {}


def test_a_merge_keeps_the_larger_previous_id_and_redirects_the_other() -> None:
    """Two clusters that now merge: the larger keeps its id, the smaller redirects to it."""
    index = {key: _prepared(key, f"item-{key}") for key in ("m-1", "m-2", "m-3")}
    previous = PreviousClusters(
        mention_ids={
            "di::big": frozenset({"m-1", "m-2"}),
            "di::small": frozenset({"m-3"}),
        }
    )

    final, redirects = assign_instrument_ids(
        {"p": ["m-1", "m-2", "m-3"]},
        cik="0000320193",
        mention_index=index,
        previous=previous,
    )

    assert final == {"p": "di::big"}
    assert redirects == {"di::small": "di::big"}


def test_stored_redirects_follow_a_retired_target_and_drop_a_revived_id() -> None:
    """Redirect chains collapse, so every target is a live id."""
    resolved = updated_redirects(
        stored={"di::a": "di::b", "di::c": "di::d"},
        retired={"di::b": "di::e"},
        live_ids={"di::e", "di::c"},
    )

    assert resolved == {"di::a": "di::e", "di::b": "di::e"}


def test_one_large_ciks_name_classes_are_counted_in_one_pass() -> None:
    """Thousands of mentions with a few names cost a few comparisons each (#269)."""
    from time import perf_counter

    from cdt.matcher.compat import name_class_sizes

    names = [f"{rate}% Senior Notes due {2030 + rate}" for rate in range(40)]
    index = {
        f"m-{i}": prepare_mention(
            build_mention_row(
                mention_id=f"m-{i}",
                item_id=f"item-{i}",
                accession_number=f"acc-{i}",
                cik="0000320193",
                date="2024-01-01",
                name=names[i % len(names)],
                start_date="2024-01-01",
                amount="$1 million",
            )
        )
        for i in range(4000)
    }

    started = perf_counter()
    sizes = name_class_sizes(index)

    assert perf_counter() - started < 2.0
    assert sizes["m-0"] == 100
