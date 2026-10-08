"""Tests for matching mentions into instruments, the rollup and lineage over stored tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from support import build_mention_row

from cdt.datasets import (
    run_manifest_path,
    shard_for_cik,
)
from cdt.matcher import (
    debt_instruments_root,
    match_pending_mentions,
    mention_cluster_edges_root,
)
from cdt.matcher.instruments import company_names_by_cik
from cdt.matcher.lineage_inference import apply_lineage_inference_pass
from cdt.matcher.normalize import coerce_optional_text, lender_signature
from cdt.matcher.schema import (
    DEBT_INSTRUMENT_COLUMNS,
    MATCHER_SCHEMA_VERSION,
    MENTION_CLUSTER_EDGE_COLUMNS,
)
from cdt.matcher.stage import _stale_schema_forces_rematch, match_tables
from cdt.storage.objects import (
    artifact_exists,
    read_json_artifact,
    write_json_artifact,
)
from cdt.storage.tables import (
    read_dataset,
    read_table,
    write_partition_table,
)


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
    from cdt.matcher.normalize import dedupe_party_clusters

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

    match_pending_mentions(artifact_root=tmp_path)

    written_instruments = read_dataset(debt_instruments_root(tmp_path))
    assert written_instruments["lender_disclosure"].to_list() == ["collective_present"]


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

    tables = match_pending_mentions(artifact_root=tmp_path)

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
    from cdt.matcher.normalize import prepare_mention

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

    tables = match_pending_mentions(artifact_root=tmp_path)

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
    match_pending_mentions(artifact_root=tmp_path)
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
    match_pending_mentions(artifact_root=tmp_path)
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
    match_pending_mentions(artifact_root=tmp_path)
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
    match_pending_mentions(artifact_root=tmp_path)
    after = {
        str(row["debt_instrument_id"]): row
        for row in read_dataset(debt_instruments_root(tmp_path)).to_dict("records")
    }
    assert after["m-2"]["amendment_of_debt_instrument_id"] == "m-1"
    assert after["m-2"]["amendment_inferred_by"] == "ordinal_chain"
    assert after["m-1"]["is_lineage_head"] is False

    # --force drops both together: no pointer, no stale provenance.
    match_pending_mentions(artifact_root=tmp_path, force=True)
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

    tables = match_pending_mentions(artifact_root=tmp_path)

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

    match_pending_mentions(artifact_root=tmp_path, renew=lambda: renewals.append(1))

    assert len(renewals) == 2


def test_canonical_fields_record_their_source_mention() -> None:
    """Each canonical value points at the mention it came from (#151)."""
    from cdt.matcher.instruments import build_debt_instrument_rows
    from cdt.matcher.normalize import prepare_mention

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
    from cdt.matcher.instruments import apply_lifecycle_rollup
    from cdt.matcher.normalize import prepare_mention

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

    from cdt.matcher.instruments import build_debt_instrument_rows
    from cdt.matcher.normalize import prepare_mention

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


def test_resolve_candidates_attaches_on_name_only_tie_instead_of_seeding() -> None:
    """A mention tying two clusters on its name joins the exact-name one."""
    from cdt.matcher.normalize import prepare_mention
    from cdt.matcher.schema import CandidateScore
    from cdt.matcher.scoring import resolve_candidates

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


def test_aggregate_lender_disclosure_precedence() -> None:
    """Worst-of across an instrument's mentions, `complete` beating `none_named`."""
    from cdt.matcher.normalize import aggregate_lender_disclosure

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


def test_instrument_rollup_publishes_balance_and_rate_columns() -> None:
    """The seven #140/#157 instrument columns had no test at all."""
    from cdt.matcher.instruments import build_debt_instrument_rows
    from cdt.matcher.normalize import prepare_mention

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
    from cdt.matcher.instruments import build_debt_instrument_rows
    from cdt.matcher.normalize import prepare_mention

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
    from cdt.matcher.instruments import apply_lifecycle_rollup

    rows = [
        _rollup_row("zzz-parent"),
        _rollup_row("mmm-child", amendment_of_debt_instrument_id="zzz-parent"),
        _rollup_row("aaa-grandchild", amendment_of_debt_instrument_id="mmm-child"),
    ]
    apply_lifecycle_rollup(rows, member_groups={}, mention_index={})

    assert {row["lineage_family_id"] for row in rows} == {"aaa-grandchild"}


def test_lineage_families_span_retirement_and_split_pointers() -> None:
    """Only amendment edges were exercised, so dropping the other two passed."""
    from cdt.matcher.instruments import apply_lifecycle_rollup

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
    from cdt.matcher.instruments import apply_lifecycle_rollup

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
    from cdt.matcher.instruments import apply_lifecycle_rollup
    from cdt.matcher.normalize import prepare_mention

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
    match_pending_mentions(artifact_root=tmp_path)
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
    match_pending_mentions(artifact_root=tmp_path)
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
