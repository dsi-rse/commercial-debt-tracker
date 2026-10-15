"""Tests for what every genre's classify stage shares: failure lists and the retry."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pandas as pd

from cdt.classifier.core import (
    LABEL_FAILED,
    LABEL_RELEVANT,
    VERDICT_FAILED,
    VERDICT_KEPT,
    retry_failed_classifications,
)
from cdt.completion import load_completion_registry
from cdt.datasets import dataset_root, load_row_failures
from cdt.partition_stage import PartitionOutput, run_partition_stage
from cdt.storage.tables import read_dataset, write_partition_table

STAGE = "test-classify"
SOURCE = "test-segments"
OUTPUT = "test-classified"
COLUMNS = ["item_id", "accession_number", "label", "relevance", "verdict"]
PARTITION = {"month": "2026-01"}
FAILING = "acc-fails"
FINE = "acc-fine"


def _seed_source(tmp_path: Path) -> str:
    return write_partition_table(
        dataset_root(SOURCE, artifact_root=str(tmp_path)),
        partition=PARTITION,
        table=pd.DataFrame(
            {
                "item_id": [f"{FINE}-1", f"{FAILING}-1"],
                "accession_number": [FINE, FAILING],
            }
        ),
    )


def _row(accession: str, *, failed: bool) -> dict[str, object]:
    return {
        "item_id": f"{accession}-1",
        "accession_number": accession,
        "label": LABEL_FAILED if failed else LABEL_RELEVANT,
        "relevance": not failed,
        "verdict": VERDICT_FAILED if failed else VERDICT_KEPT,
    }


class FakeClassifier:
    """Classifies every filing, failing those in ``failing``; records each call."""

    def __init__(self: Self, failing: frozenset[str] = frozenset()) -> None:
        """Remember which filings fail."""
        self.failing = failing
        self.calls: list[frozenset[str] | None] = []

    def __call__(
        self: Self,
        source_path: str,
        partition: dict[str, str],
        accessions: frozenset[str] | None = None,
    ) -> PartitionOutput:
        """Return one row per filing of the source partition, or of ``accessions``."""
        del partition
        self.calls.append(accessions)
        source = read_dataset(str(Path(source_path).parent.parent))
        wanted = [
            accession
            for accession in source["accession_number"]
            if accessions is None or accession in accessions
        ]
        return PartitionOutput(
            rows=pd.DataFrame(
                [_row(a, failed=a in self.failing) for a in wanted], columns=COLUMNS
            ),
            source_rows=len(wanted),
            failures={a: "stage 2 gave up" for a in wanted if a in self.failing},
        )


def _run(tmp_path: Path, classifier: FakeClassifier) -> None:
    run_partition_stage(
        STAGE,
        source_dataset=SOURCE,
        output_dataset=OUTPUT,
        output_columns=COLUMNS,
        process=lambda path, partition, _done: classifier(path, partition, None),
        artifact_root=str(tmp_path),
        force=True,
    )


def _retry(tmp_path: Path, classifier: FakeClassifier) -> object:
    return retry_failed_classifications(
        STAGE,
        source_dataset=SOURCE,
        output_dataset=OUTPUT,
        output_columns=COLUMNS,
        classify_partition=classifier,
        artifact_root=str(tmp_path),
    )


def test_a_stage_records_failed_filings_with_their_source_partition(
    tmp_path: Path,
) -> None:
    """The failure list names each failed filing's source partition and error."""
    source_path = _seed_source(tmp_path)

    _run(tmp_path, FakeClassifier(failing=frozenset({FAILING})))

    assert load_row_failures(STAGE, artifact_root=str(tmp_path)) == {
        FAILING: {"source_partition": source_path, "error": "stage 2 gave up"}
    }


def test_a_rerun_of_the_partition_replaces_its_failure_entries(
    tmp_path: Path,
) -> None:
    """A filing that succeeds when its partition runs again leaves the list."""
    _seed_source(tmp_path)
    _run(tmp_path, FakeClassifier(failing=frozenset({FAILING})))

    _run(tmp_path, FakeClassifier())

    assert load_row_failures(STAGE, artifact_root=str(tmp_path)) == {}


def test_retry_reclassifies_only_failed_filings_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    """Only the failed filing is classified again; the other's row is untouched."""
    _seed_source(tmp_path)
    _run(tmp_path, FakeClassifier(failing=frozenset({FAILING})))
    registry_before = dict(load_completion_registry(STAGE, artifact_root=str(tmp_path)))
    retrier = FakeClassifier()

    result = _retry(tmp_path, retrier)

    assert retrier.calls == [frozenset({FAILING})]
    assert (result.retried, result.recovered, result.still_failed) == (1, 1, 0)
    written = read_dataset(dataset_root(OUTPUT, artifact_root=str(tmp_path)))
    assert dict(zip(written["accession_number"], written["verdict"], strict=True)) == {
        FINE: VERDICT_KEPT,
        FAILING: VERDICT_KEPT,
    }
    assert load_row_failures(STAGE, artifact_root=str(tmp_path)) == {}
    assert dict(load_completion_registry(STAGE, artifact_root=str(tmp_path))) == (
        registry_before
    )


def test_retry_with_no_failures_does_nothing(tmp_path: Path) -> None:
    """A stage that never failed, such as 8-K today, retries nothing."""
    _seed_source(tmp_path)
    _run(tmp_path, FakeClassifier())
    retrier = FakeClassifier()

    result = _retry(tmp_path, retrier)

    assert retrier.calls == []
    assert result.retried == 0
