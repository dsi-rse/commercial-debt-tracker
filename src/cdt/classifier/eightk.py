"""Classify 8-K item rows for relevance and write the classifications dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cdt.classifier.core import (
    CLASSIFIED_ITEM_COLUMNS,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    VERDICT_DROPPED_BELOW_THRESHOLD,
    VERDICT_KEPT,
    ClassifyPartition,
    RetryResult,
    default_model_dir,
    load_training_artifacts,
    normalize_text,
    retry_failed_classifications,
    score_model,
)
from cdt.datasets import CLASSIFICATION_DATASET_NAME, ITEM_DATASET_NAME
from cdt.partition_stage import PartitionOutput, run_partition_stage
from cdt.segmenter.core import ITEM_COLUMNS
from cdt.storage.tables import read_table

LOGGER = logging.getLogger(__name__)

#: Completion-registry and run-manifest name of the 8-K classify stage.
STAGE_NAME = "classify"


def classify_items(
    items: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    force: bool = False,
    artifacts: tuple[object, float] | None = None,
) -> pd.DataFrame:
    """Classify in-memory item rows using a saved binary model.

    Adds ``label``, ``relevance``, ``classification_score`` and ``verdict``.
    ``artifacts`` is a pre-loaded ``(model, threshold)`` pair; None loads them
    from ``model_dir`` (default ``default_model_dir()``). ``force`` is ignored.
    """
    del force
    if items.empty:
        return pd.DataFrame(columns=CLASSIFIED_ITEM_COLUMNS)

    if artifacts is not None:
        model, threshold = artifacts
    else:
        resolved_model_dir = model_dir or default_model_dir()
        model, threshold, _ = load_training_artifacts(resolved_model_dir)
    classified = items.copy()
    texts = [normalize_text(str(value)) for value in classified["text"].fillna("")]
    scores = score_model(model, texts)
    above = scores >= threshold
    classified["classification_score"] = scores
    classified["label"] = np.where(above, LABEL_RELEVANT, LABEL_IRRELEVANT)
    classified["relevance"] = classified["label"].eq(LABEL_RELEVANT)
    classified["verdict"] = np.where(
        above, VERDICT_KEPT, VERDICT_DROPPED_BELOW_THRESHOLD
    )

    output_columns = list(
        dict.fromkeys([*classified.columns, *CLASSIFIED_ITEM_COLUMNS])
    )
    return classified.reindex(columns=output_columns)


def item_partition_classifier(*, model_dir: Path | None = None) -> ClassifyPartition:
    """Return the function that classifies one ``items`` partition.

    Given accessions, it classifies only those filings' items. The model is
    unpickled on the first call, once per returned function.
    """
    artifacts: tuple[object, float] | None = None

    def classify_partition(
        source_path: str,
        partition: dict[str, str],
        accessions: frozenset[str] | None = None,
    ) -> PartitionOutput:
        nonlocal artifacts
        del partition
        if artifacts is None:
            model, threshold, _ = load_training_artifacts(
                model_dir or default_model_dir()
            )
            artifacts = (model, threshold)
        items = read_table(source_path, ITEM_COLUMNS).reindex(columns=ITEM_COLUMNS)
        if accessions is not None:
            items = items.loc[items["accession_number"].astype(str).isin(accessions)]
        return PartitionOutput(
            rows=classify_items(items, artifacts=artifacts), source_rows=len(items)
        )

    return classify_partition


def classify_pending_items(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    force: bool = False,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Classify pending item partitions into classification partitions.

    A partition is pending when its fingerprint differs from the completion
    registry's (or always, with ``force``) and is recomputed whole. Completion
    is checkpointed and ``renew`` called as :func:`run_partition_stage` does.

    Returns:
        The classified rows written this run, in CLASSIFIED_ITEM_COLUMNS order.
    """
    classify_partition = item_partition_classifier(model_dir=model_dir)
    return run_partition_stage(
        STAGE_NAME,
        source_dataset=ITEM_DATASET_NAME,
        output_dataset=CLASSIFICATION_DATASET_NAME,
        output_columns=CLASSIFIED_ITEM_COLUMNS,
        process=lambda source_path, partition: classify_partition(
            source_path, partition, None
        ),
        artifact_root=artifact_root,
        data_dir=data_dir,
        force=force,
        renew=renew,
    ).rows


def retry_failed_items(
    *,
    artifact_root: str | Path | None = None,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    renew: Callable[[], None] | None = None,
) -> RetryResult:
    """Classify again the 8-K filings in this stage's failure list.

    See :func:`cdt.classifier.core.retry_failed_classifications`.
    """
    return retry_failed_classifications(
        STAGE_NAME,
        source_dataset=ITEM_DATASET_NAME,
        output_dataset=CLASSIFICATION_DATASET_NAME,
        output_columns=CLASSIFIED_ITEM_COLUMNS,
        classify_partition=item_partition_classifier(model_dir=model_dir),
        artifact_root=artifact_root,
        data_dir=data_dir,
        renew=renew,
    )
