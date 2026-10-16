"""Tests for matching as a pure function of a CIK's mentions, and for instrument ids."""

from __future__ import annotations

import shutil
from dataclasses import replace
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
    # Apart in a full match, but 14 and 15 merge when they arrive before 13:
    # 13 is the competing candidate that makes 15's match ambiguous.
    add(13, "1018724", "2024-02-15", "Credit Agreement", "2024-01-01", "$100 million")
    add(
        14,
        "1018724",
        "2024-03-15",
        "Amended and Restated Credit Agreement",
        "",
        "$150 million",
    )
    add(15, "1018724", "2024-04-15", "Credit Agreement", "2024-01-01", "$150 million")
    return pd.DataFrame(rows)


def _write(root: Path, mentions: pd.DataFrame) -> None:
    """Replace the root's mentions with ``mentions``, one shuffled file per month."""
    shutil.rmtree(root / "mentions", ignore_errors=True)
    shuffled = mentions.sample(frac=1, random_state=len(mentions))
    for month, rows in shuffled.groupby(shuffled["date"].str[:7]):
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
        content={"di::reextracted": {"instrument_type": "term_loan"}},
    )
    index["m-2-re"] = replace(index["m-2-re"], instrument_type="term_loan")

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


def test_an_item_overlap_with_no_shared_content_carries_nothing() -> None:
    """Sharing an item is not enough: the cluster must state some of the old content."""
    index = {"m-2-re": _prepared("m-2-re", "item-2")}
    previous = PreviousClusters(
        mention_ids={"di::old": frozenset({"m-2"})},
        item_ids={"di::old": frozenset({"item-2"})},
        content={"di::old": {"instrument_type": "revolver"}},
    )

    final, redirects = assign_instrument_ids(
        {"p": ["m-2-re"]},
        cik="0000320193",
        mention_index=index,
        previous=previous,
    )

    assert final["p"] != "di::old"
    assert redirects == {}


def _two_instrument_item(
    revolver_id: str = "m-revolver", term_loan_id: str = "m-term-loan"
) -> list[dict[str, object]]:
    """One 8-K item describing a revolver and a term loan B on the same day."""
    common = {
        "item_id": "item-1",
        "accession_number": "0001",
        "cik": "320193",
        "date": "2024-01-10",
        "start_date": "2024-01-10",
    }
    return [
        build_mention_row(
            mention_id=revolver_id,
            name="Revolving Credit Facility",
            amount="$500 million",
            instrument_type="revolver",
            maturity_date="2029-01-10",
            **common,
        ),
        build_mention_row(
            mention_id=term_loan_id,
            name="Term Loan B",
            amount="$750 million",
            instrument_type="term_loan",
            maturity_date="2031-01-10",
            **common,
        ),
    ]


def _ids_by_name(root: Path) -> dict[str, str]:
    instruments = _instruments(root)
    return dict(
        zip(instruments["name"], instruments["debt_instrument_id"], strict=True)
    )


@pytest.mark.parametrize(
    "new_ids", [("m-a-revolver", "m-z-term-loan"), ("m-z-revolver", "m-a-term-loan")]
)
def test_re_extracting_an_item_keeps_each_instruments_id(
    tmp_path: Path, new_ids: tuple[str, str]
) -> None:
    """Two instruments of one item keep their own ids when its mention ids change (#289)."""
    _write(tmp_path, pd.DataFrame(_two_instrument_item()))
    match_pending_mentions(artifact_root=tmp_path)
    before = _ids_by_name(tmp_path)

    _write(tmp_path, pd.DataFrame(_two_instrument_item(*new_ids)))
    match_pending_mentions(artifact_root=tmp_path)

    assert _ids_by_name(tmp_path) == before
    assert read_dataset(debt_instrument_redirects_root(tmp_path)).empty


def _two_tranche_item(
    first_id: str = "m-large", second_id: str = "m-small"
) -> list[dict[str, object]]:
    """One 8-K item describing two same-named note tranches of different sizes."""
    common = {
        "item_id": "item-1",
        "accession_number": "0001",
        "cik": "320193",
        "date": "2024-01-10",
        "start_date": "2024-01-10",
        "name": "Convertible Senior Notes",
        "instrument_type": "notes",
    }
    return [
        build_mention_row(mention_id=first_id, amount="$1.6 billion", **common),
        build_mention_row(mention_id=second_id, amount="$690 million", **common),
    ]


@pytest.mark.parametrize(
    "new_ids", [("m-a-large", "m-z-small"), ("m-z-large", "m-a-small")]
)
def test_re_extracted_tranches_that_differ_only_by_amount_keep_their_ids(
    tmp_path: Path, new_ids: tuple[str, str]
) -> None:
    """Principal amount tells same-named tranches of one item apart (#302)."""
    _write(tmp_path, pd.DataFrame(_two_tranche_item()))
    match_pending_mentions(artifact_root=tmp_path)
    before = _instruments(tmp_path).set_index("principal_amount")["debt_instrument_id"]
    assert len(before) == 2

    _write(tmp_path, pd.DataFrame(_two_tranche_item(*new_ids)))
    match_pending_mentions(artifact_root=tmp_path)

    after = _instruments(tmp_path).set_index("principal_amount")["debt_instrument_id"]
    assert after.to_dict() == before.to_dict()
    assert read_dataset(debt_instrument_redirects_root(tmp_path)).empty


def test_indistinguishable_instruments_of_one_item_get_new_ids() -> None:
    """When content cannot tell two re-extracted instruments apart, neither carries."""
    index = {
        "m-a": _prepared("m-a", "item-1"),
        "m-b": _prepared("m-b", "item-1"),
    }
    previous = PreviousClusters(
        mention_ids={"di::x": frozenset({"m-x"}), "di::y": frozenset({"m-y"})},
        item_ids={"di::x": frozenset({"item-1"}), "di::y": frozenset({"item-1"})},
        content={
            "di::x": {"start_date": "2024-01-01"},
            "di::y": {"start_date": "2024-01-01"},
        },
    )

    final, redirects = assign_instrument_ids(
        {"p-a": ["m-a"], "p-b": ["m-b"]},
        cik="0000320193",
        mention_index=index,
        previous=previous,
    )

    assert not set(final.values()) & {"di::x", "di::y"}
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


def test_one_large_ciks_name_classes_are_counted_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thousands of mentions with a few names compare each name pair once (#269)."""
    from cdt.matcher import compat

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
    compared: list[tuple[str | None, str | None]] = []
    compatible = compat.name_fingerprints_are_compatible

    def counting(left: str | None, right: str | None) -> bool:
        compared.append((left, right))
        return compatible(left, right)

    monkeypatch.setattr(compat, "name_fingerprints_are_compatible", counting)

    sizes = compat.name_class_sizes(index)

    assert len(compared) <= len(names) ** 2
    assert sizes["m-0"] == 100


def _loan(mention_id: str, cik: str, name: str, **overrides: object) -> dict:
    return build_mention_row(
        mention_id=mention_id,
        item_id=f"item-{mention_id}",
        accession_number=f"acc-{mention_id}",
        cik=cik,
        date="2024-01-01",
        name=name,
        start_date="2024-01-01",
        amount="$100 million",
        **overrides,
    )


def _alpha_and_beta() -> pd.DataFrame:
    """Two loans that cluster apart at the default thresholds and together at 0.75."""
    return pd.DataFrame(
        [_loan("m-a", "320193", "Alpha Loan"), _loan("m-b", "320193", "Beta Facility")]
    )


def test_a_change_to_a_column_the_id_does_not_hash_re_matches_the_cik(
    tmp_path: Path,
) -> None:
    """The CIK digest covers every column the matcher reads, not just the ids (#290)."""
    mentions = pd.DataFrame([_loan("m-a", "320193", "Alpha Loan")])
    _write(tmp_path, mentions)
    match_pending_mentions(artifact_root=tmp_path)

    _write(tmp_path, mentions.assign(company_name="Renamed Inc."))
    result = match_pending_mentions(artifact_root=tmp_path)

    assert result.ciks_matched == 1
    assert _instruments(tmp_path)["company_name"].to_list() == ["Renamed Inc."]
    assert match_pending_mentions(artifact_root=tmp_path).ciks_matched == 0


def test_a_rematch_keeps_the_rows_of_other_ciks_in_the_same_shard(
    tmp_path: Path,
) -> None:
    """320193 and 100032 share ``cik_shard=0044``; re-matching one leaves the other."""
    other = _loan("m-other", "100032", "Gamma Notes")
    _write(tmp_path, pd.DataFrame([_loan("m-a", "320193", "Alpha Loan"), other]))
    match_pending_mentions(artifact_root=tmp_path)
    before = _instruments(tmp_path).set_index("cik").loc["0000100032"]

    _write(
        tmp_path,
        pd.DataFrame([_loan("m-a2", "320193", "Alpha Loan Two"), other]),
    )
    result = match_pending_mentions(artifact_root=tmp_path)

    assert result.ciks_matched == 1
    after = _instruments(tmp_path).set_index("cik")
    pd.testing.assert_series_equal(after.loc["0000100032"], before)
    assert set(after.index) == {"0000320193", "0000100032"}


def _merge_alpha_into_beta(root: Path) -> tuple[dict[str, str], str]:
    """Match Alpha and Beta apart, then force a match that merges them."""
    _write(root, _alpha_and_beta())
    match_pending_mentions(artifact_root=root)
    before = _instruments(root).set_index("seed_debt_instrument_mention_id")
    return before["debt_instrument_id"].to_dict(), str(root)


def test_a_merge_publishes_a_redirect_from_the_retired_id(tmp_path: Path) -> None:
    """The retired id redirects to the survivor, kept by later re-matches and published."""
    from cdt.publish import publish_final_tables, published_table_path
    from cdt.storage.tables import read_table

    ids, _ = _merge_alpha_into_beta(tmp_path)
    match_pending_mentions(
        artifact_root=tmp_path,
        force=True,
        strong_match_threshold=0.75,
        loose_match_threshold=0.75,
    )
    [survivor] = _instruments(tmp_path)["debt_instrument_id"]
    [retired] = set(ids.values()) - {survivor}
    _write(
        tmp_path,
        pd.concat(
            [_alpha_and_beta(), pd.DataFrame([_loan("m-c", "320193", "Zeta Notes")])]
        ),
    )
    assert (
        match_pending_mentions(
            artifact_root=tmp_path,
            strong_match_threshold=0.75,
            loose_match_threshold=0.75,
        ).ciks_matched
        == 1
    )

    publish_final_tables(
        artifact_root=str(tmp_path), final_database_root=str(tmp_path / "final")
    )

    published = read_table(
        published_table_path(tmp_path / "final", "debt-instrument-redirects")
    )
    pairs = published[["old_debt_instrument_id", "debt_instrument_id"]]
    assert pairs.to_numpy().tolist() == [[retired, survivor]]


def test_redirects_survive_a_crash_between_their_write_and_the_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that stops between the redirects and the edges keeps its redirects (#291).

    The crash is on the instruments write, which comes after the redirects and
    before the edges; with the edges written first, the rerun would see no
    retired ids and the redirects would be lost.
    """
    from cdt.matcher import stage

    ids, _ = _merge_alpha_into_beta(tmp_path)
    write = stage.write_partition_table

    def crash_on_instruments(root: object, **kwargs: object) -> object:
        if Path(str(root)).name == "debt-instruments":
            raise RuntimeError("simulated crash")
        return write(root, **kwargs)

    merge = {"strong_match_threshold": 0.75, "loose_match_threshold": 0.75}
    monkeypatch.setattr(stage, "write_partition_table", crash_on_instruments)
    with pytest.raises(RuntimeError):
        match_pending_mentions(artifact_root=tmp_path, force=True, **merge)
    monkeypatch.undo()

    match_pending_mentions(artifact_root=tmp_path, **merge)

    [survivor] = _instruments(tmp_path)["debt_instrument_id"]
    redirects = read_dataset(debt_instrument_redirects_root(tmp_path))
    assert dict(
        zip(
            redirects["old_debt_instrument_id"],
            redirects["debt_instrument_id"],
            strict=True,
        )
    ) == {old: survivor for old in set(ids.values()) - {survivor}}


def test_an_interrupted_remint_is_finished_by_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--remint` that stops part-way is repeated, remint included, by a plain run (#294)."""
    from cdt.matcher import stage

    corpus = _corpus()
    reference = tmp_path / "reference"
    _write(reference, corpus)
    match_pending_mentions(artifact_root=reference, remint=True)
    root = tmp_path / "root"
    _arrive_in(root, _orders()["backfill, newer half first"])
    write = stage.write_partition_table
    shards_written: list[str] = []

    def crash_on_second_shard(root: object, **kwargs: object) -> object:
        shard = kwargs["partition"]["cik_shard"]
        if shard not in shards_written:
            shards_written.append(shard)
        if len(shards_written) > 1:
            raise RuntimeError("simulated crash")
        return write(root, **kwargs)

    monkeypatch.setattr(stage, "write_partition_table", crash_on_second_shard)
    with pytest.raises(RuntimeError):
        match_pending_mentions(artifact_root=root, remint=True)
    monkeypatch.undo()

    match_pending_mentions(artifact_root=root)

    pd.testing.assert_frame_equal(_instruments(root), _instruments(reference))
    assert not (root / "runs" / "match" / "rebuild-pending.json").exists()
