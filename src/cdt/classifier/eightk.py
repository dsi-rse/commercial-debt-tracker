"""Classify 8-K item rows for relevance and write the classifications dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cdt.classifier.core import (
    CLASSIFIED_ITEM_COLUMNS,
    default_model_dir,
    load_training_artifacts,
    normalize_text,
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

    Adds ``label``, ``relevance`` and ``classification_score``. ``artifacts`` is
    a pre-loaded ``(model, threshold)`` pair; None loads them from
    ``model_dir`` (default ``default_model_dir()``). ``force`` is ignored.
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
    classified["classification_score"] = scores
    classified["label"] = np.where(scores >= threshold, "relevant", "irrelevant")
    classified["relevance"] = classified["label"].eq("relevant")

    output_columns = list(
        dict.fromkeys(
            [*classified.columns, "label", "relevance", "classification_score"]
        )
    )
    return classified.reindex(columns=output_columns)


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
    # Unpickled on the first pending partition, once per run.
    artifacts: tuple[object, float] | None = None

    def process(source_path: str, partition: dict[str, str]) -> PartitionOutput:
        nonlocal artifacts
        del partition
        if artifacts is None:
            model, threshold, _ = load_training_artifacts(
                model_dir or default_model_dir()
            )
            artifacts = (model, threshold)
        items = read_table(source_path, ITEM_COLUMNS).reindex(columns=ITEM_COLUMNS)
        return PartitionOutput(
            rows=classify_items(items, artifacts=artifacts), source_rows=len(items)
        )

    return run_partition_stage(
        STAGE_NAME,
        source_dataset=ITEM_DATASET_NAME,
        output_dataset=CLASSIFICATION_DATASET_NAME,
        output_columns=CLASSIFIED_ITEM_COLUMNS,
        process=process,
        artifact_root=artifact_root,
        data_dir=data_dir,
        force=force,
        renew=renew,
    ).rows
