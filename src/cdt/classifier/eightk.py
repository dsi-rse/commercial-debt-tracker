"""Classify 8-K item rows for relevance and write the classifications dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cdt.classifier.core import (
    CLASSIFICATION_DATASET_NAME,
    CLASSIFIED_ITEM_COLUMNS,
    classifications_root,
    default_model_dir,
    load_training_artifacts,
    normalize_text,
    score_model,
)
from cdt.completion import (
    CompletedPartition,
    completion_registry_path,
    pending_source_partitions,
    save_completion_registry,
)
from cdt.datasets import (
    date_shard_partition_path,
    parse_date_shard_partition,
    resolve_artifact_root,
    run_manifest_path,
)
from cdt.segmenter.core import ITEM_COLUMNS, ITEM_DATASET_NAME
from cdt.storage.objects import write_json_artifact
from cdt.storage.tables import read_table, write_partition_table

LOGGER = logging.getLogger(__name__)


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
    ``model_dir`` (default ``default_model_dir(data_dir)``). ``force`` is ignored.
    """
    del force
    if items.empty:
        return pd.DataFrame(columns=CLASSIFIED_ITEM_COLUMNS)

    if artifacts is not None:
        model, threshold = artifacts
    else:
        resolved_model_dir = model_dir or default_model_dir(data_dir)
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
    batch_size: int = 100,
    force: bool = False,
    renew: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Classify pending item partitions into classification partitions.

    A partition is pending when its fingerprint differs from the completion
    registry's (or always, with ``force``) and is recomputed whole. Completion
    is saved and ``renew`` called after every ``batch_size`` partitions.

    Returns:
        The classified rows written this run, in CLASSIFIED_ITEM_COLUMNS order.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    resolved_root = resolve_artifact_root(artifact_root, data_dir=data_dir)
    processed_frames: list[pd.DataFrame] = []
    partitions_written: list[str] = []
    visited_item_paths: set[str] = set()
    empty_partitions = 0
    pending_with_fingerprints, registry = pending_source_partitions(
        "classify",
        ITEM_DATASET_NAME,
        artifact_root=resolved_root,
        data_dir=data_dir,
        force=force,
    )
    pending_item_paths = [path for path, _ in pending_with_fingerprints]
    source_fingerprints = dict(pending_with_fingerprints)

    # Unpickle the model once per run, not per partition.
    artifacts: tuple[object, float] | None = None
    if pending_item_paths:
        resolved_model_dir = model_dir or default_model_dir(data_dir)
        model, threshold, _ = load_training_artifacts(resolved_model_dir)
        artifacts = (model, threshold)

    total_partitions = len(pending_item_paths)
    for chunk_start in range(0, total_partitions, batch_size):
        chunk_paths = pending_item_paths[chunk_start : chunk_start + batch_size]
        for partition_index, item_path in enumerate(chunk_paths, start=chunk_start + 1):
            partition = parse_date_shard_partition(item_path)
            partition_label = f"date={partition['date']} shard={partition['shard']}"
            partition_start = perf_counter()
            visited_item_paths.add(item_path)
            batch_items = read_table(item_path, ITEM_COLUMNS).reindex(
                columns=ITEM_COLUMNS
            )
            classified = classify_items(
                batch_items,
                data_dir=data_dir,
                model_dir=model_dir,
                artifacts=artifacts,
            )
            if classified.empty:
                empty_partitions += 1
            else:
                write_partition_table(
                    classifications_root(resolved_root, data_dir=data_dir),
                    partition={"date": partition["date"], "shard": partition["shard"]},
                    table=classified.reindex(columns=CLASSIFIED_ITEM_COLUMNS),
                )
                processed_frames.append(classified)
                partitions_written.append(
                    date_shard_partition_path(
                        CLASSIFICATION_DATASET_NAME,
                        partition_date=partition["date"],
                        shard=partition["shard"],
                        artifact_root=resolved_root,
                        data_dir=data_dir,
                    )
                )
            relevant_count = int(classified["relevance"].fillna(False).sum())
            LOGGER.info(
                "Classification partition complete: %s progress=%s/%s items=%s relevant=%s wrote_output=%s elapsed=%.1fs",
                partition_label,
                partition_index,
                total_partitions,
                len(batch_items),
                relevant_count,
                not classified.empty,
                perf_counter() - partition_start,
            )

        # Save completion and renew the lease at every batch boundary, so an
        # interruption keeps finished batches and a long stage is not stolen.
        for item_path in chunk_paths:
            registry[item_path] = CompletedPartition(
                fingerprint=source_fingerprints.get(item_path)
            )
        save_completion_registry(
            "classify",
            registry,
            artifact_root=resolved_root,
            data_dir=data_dir,
        )
        if renew is not None:
            renew()

    # Save once more: pending_source_partitions can dirty entries even when
    # nothing was pending.
    save_completion_registry(
        "classify",
        registry,
        artifact_root=resolved_root,
        data_dir=data_dir,
    )

    write_json_artifact(
        run_manifest_path(
            "classify",
            "latest",
            artifact_root=resolved_root,
            data_dir=data_dir,
        ),
        {
            "artifact_root": resolved_root,
            "stage": "classify",
            "batch_size": batch_size,
            "force": force,
            "partitions_visited": sorted(visited_item_paths),
            "partitions_written": partitions_written,
            "empty_partitions_skipped_from_write": empty_partitions,
            "completion_registry": completion_registry_path(
                "classify", artifact_root=resolved_root, data_dir=data_dir
            ),
        },
    )
    LOGGER.info("Classifier complete: total_partitions=%s", len(partitions_written))
    if not processed_frames:
        return pd.DataFrame(columns=CLASSIFIED_ITEM_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=CLASSIFIED_ITEM_COLUMNS
    )
