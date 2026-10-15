"""What every genre's classify stage shares, and the relevance classifier.

Shared: the classified-row columns and their ``label`` and ``verdict`` values,
and the retry of filings a classify stage could not classify. The classifier:
train, score and threshold a TF-IDF linear SVC, and load or save its artifacts.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import pandas as pd

from cdt import settings
from cdt.completion import pending_source_partitions
from cdt.datasets import (
    CLASSIFICATION_DATASET_NAME,
    dataset_root,
    date_shard_partition_path,
    load_row_failures,
    parse_date_shard_partition,
    resolve_artifact_root,
    save_row_failures,
)
from cdt.lease import throttled
from cdt.partition_stage import PartitionOutput, replace_partition_failures
from cdt.segmenter.core import ITEM_COLUMNS
from cdt.storage.tables import read_table, write_partition_table

LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_RECALL = 0.99
DEFAULT_RANDOM_SEED = 42
DEFAULT_CV_SPLITS = 5
MIN_CLASS_COUNT = 2
MIN_CV_SPLITS = 2
MODEL_NAME = "tfidf_linear_svc"
MODEL_FILENAME = "model.pkl"
METADATA_FILENAME = "metadata.json"
#: ``label`` values. A failed row was not classified, so it is neither
#: relevant nor irrelevant; ``relevance`` is ``label == LABEL_RELEVANT``.
LABEL_RELEVANT = "relevant"
LABEL_IRRELEVANT = "irrelevant"
LABEL_FAILED = "failed"
#: ``verdict`` values: why a row has its label.
VERDICT_KEPT = "kept"
#: The relevance score fell below the classifier's threshold.
VERDICT_DROPPED_BELOW_THRESHOLD = "dropped_below_threshold"
#: 6-K stage 2: the snippet names debt but gives no details.
VERDICT_DROPPED_NO_DETAILS = "dropped_no_details"
#: 6-K stage 2: another snippet of the filing covers the same debt.
VERDICT_DROPPED_DUPLICATE = "dropped_duplicate"
#: The filing could not be classified; it has an entry in the stage's failure
#: list, and ``cdt classify --retry-failures`` classifies it again.
VERDICT_FAILED = "failed"
CLASSIFIED_ITEM_COLUMNS = [
    *ITEM_COLUMNS,
    "label",
    "relevance",
    "classification_score",
    "verdict",
]

#: Classify one source partition: ``(source_path, {date, shard}, accessions)``.
#: ``accessions`` None classifies every filing in it; a set classifies only
#: those filings.
ClassifyPartition = Callable[
    [str, dict[str, str], frozenset[str] | None], PartitionOutput
]


@dataclass(frozen=True)
class RetryResult:
    """What one ``retry_failed_classifications`` run did, counted in filings."""

    retried: int = 0
    recovered: int = 0
    still_failed: int = 0
    #: In a source partition that changed since it was classified; the next
    #: plain run classifies the whole partition.
    skipped: int = 0


class SupportsDecisionFunction(Protocol):
    """Protocol for fitted classifiers used by this module."""

    def fit(
        self: Self, texts: Sequence[str], labels: np.ndarray
    ) -> SupportsDecisionFunction:
        """Fit the estimator."""

    def decision_function(self: Self, texts: Sequence[str]) -> np.ndarray:
        """Return class margins for the provided texts."""


class SupportsFit(Protocol):
    """Protocol for unfitted estimators that can be cloned and trained."""

    def fit(
        self: Self, texts: Sequence[str], labels: np.ndarray
    ) -> SupportsDecisionFunction:
        """Fit the estimator."""


def default_model_dir() -> Path:
    """Return the committed 8-K classifier artifact directory."""
    return settings.MODELS_DIR / "classifier" / "tfidf-linear-svc"


def classifications_root(
    artifact_root: str | Path | None = None,
) -> str:
    """Return the canonical classifications dataset root."""
    return dataset_root(
        CLASSIFICATION_DATASET_NAME,
        artifact_root=artifact_root,
    )


def train_classifier_model(
    *,
    train_csv: Path,
    model_dir: Path,
    target_recall: float = DEFAULT_TARGET_RECALL,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    """Train the binary classifier and persist its artifacts."""
    texts, labels = load_training_examples(train_csv)
    model = build_linear_svc_pipeline(random_seed)
    cv_scores = cross_validated_scores(
        model,
        texts,
        labels,
        cv_splits=cv_splits,
        random_seed=random_seed,
    )
    threshold_metrics = select_threshold(
        cv_scores,
        labels,
        target_recall=target_recall,
    )
    pr_auc = compute_pr_auc(cv_scores, labels)
    fitted_model = clone_model(model)
    fitted_model.fit(texts, labels)

    import sklearn

    metadata = {
        "model_name": MODEL_NAME,
        "sklearn_version": sklearn.__version__,
        "train_csv": str(train_csv),
        "training_row_count": int(len(texts)),
        "target_recall": float(target_recall),
        "cv_splits": int(cv_splits),
        "random_seed": int(random_seed),
        "threshold": float(threshold_metrics["threshold"]),
        "precision": float(threshold_metrics["precision"]),
        "recall": float(threshold_metrics["recall"]),
        "pr_auc": float(pr_auc),
    }
    save_training_artifacts(model_dir=model_dir, model=fitted_model, metadata=metadata)
    LOGGER.info(
        "Classifier training complete: rows=%s precision=%.4f recall=%.4f "
        "pr_auc=%.4f threshold=%.4f model_dir=%s",
        metadata["training_row_count"],
        metadata["precision"],
        metadata["recall"],
        metadata["pr_auc"],
        metadata["threshold"],
        model_dir,
    )
    return metadata


def _minor_version(version: str) -> tuple[str, ...]:
    """Return the major.minor prefix sklearn keeps pickles compatible within."""
    return tuple(version.split(".")[:2])


def load_training_artifacts(
    model_dir: Path,
) -> tuple[SupportsDecisionFunction, float, dict[str, object]]:
    """Load a fitted model, its threshold and its metadata from ``model_dir``.

    Logs a warning when the metadata records no ``sklearn_version``.

    Raises:
        RuntimeError: If the recorded scikit-learn major.minor differs from
            the running one.
    """
    model_path = model_dir / MODEL_FILENAME
    metadata_path = model_dir / METADATA_FILENAME
    with metadata_path.open(encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    import sklearn

    recorded_version = metadata.get("sklearn_version")
    if recorded_version is None:
        LOGGER.warning(
            "Model metadata at %s records no sklearn_version; cannot verify "
            "the pickle matches the running scikit-learn %s (#109)",
            metadata_path,
            sklearn.__version__,
        )
    elif _minor_version(str(recorded_version)) != _minor_version(sklearn.__version__):
        msg = (
            f"Model at {model_path} was trained under scikit-learn "
            f"{recorded_version} but the runtime has {sklearn.__version__}. "
            "Retrain the model under the running version (cdt classify train) "
            "or pin scikit-learn back to the training minor (#109)."
        )
        raise RuntimeError(msg)

    with model_path.open("rb") as file_obj:
        model = pickle.load(file_obj)  # noqa: S301
    return model, float(metadata["threshold"]), metadata


def load_training_examples(train_csv: Path) -> tuple[list[str], np.ndarray]:
    """Load normalized training texts and binary labels from a CSV."""
    table = pd.read_csv(train_csv)
    missing_columns = {"text", "label"}.difference(table.columns)
    if missing_columns:
        msg = f"Training CSV is missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    texts: list[str] = []
    labels: list[int] = []
    for row in table[["text", "label"]].to_dict("records"):
        text = normalize_text(str(row["text"]) if pd.notna(row["text"]) else "")
        label_text = str(row["label"]).strip() if pd.notna(row["label"]) else ""
        if not text or not label_text:
            continue
        texts.append(text)
        labels.append(parse_label(label_text))
    if not texts:
        msg = f"No usable labeled rows found in {train_csv}"
        raise ValueError(msg)
    return texts, np.asarray(labels, dtype=int)


def save_training_artifacts(
    *,
    model_dir: Path,
    model: SupportsDecisionFunction,
    metadata: dict[str, object],
) -> None:
    """Persist the fitted model and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / MODEL_FILENAME).open("wb") as file_obj:
        pickle.dump(model, file_obj)
    with (model_dir / METADATA_FILENAME).open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def build_linear_svc_pipeline(random_seed: int) -> SupportsFit:
    """Create the TF-IDF + LinearSVC pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def clone_model(model: SupportsFit) -> SupportsFit:
    """Clone an unfitted scikit-learn model."""
    from sklearn.base import clone

    return clone(model)


def cross_validated_scores(
    model: SupportsFit,
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    cv_splits: int,
    random_seed: int,
) -> np.ndarray:
    """Compute out-of-fold scores for threshold selection."""
    from sklearn.model_selection import StratifiedKFold

    label_counts = np.bincount(labels, minlength=MIN_CLASS_COUNT)
    if int((label_counts > 0).sum()) < MIN_CLASS_COUNT:
        raise ValueError("Need examples for both relevant and irrelevant classes")

    effective_cv_splits = min(cv_splits, int(label_counts[label_counts > 0].min()))
    if effective_cv_splits < MIN_CV_SPLITS:
        raise ValueError("Need at least two examples per class for cross-validation")

    splitter = StratifiedKFold(
        n_splits=effective_cv_splits,
        shuffle=True,
        random_state=random_seed,
    )
    scores = np.zeros(len(texts), dtype=float)
    texts_array = np.asarray(texts, dtype=object)
    for train_indices, test_indices in splitter.split(texts_array, labels):
        fold_model = clone_model(model)
        fold_model.fit(texts_array[train_indices].tolist(), labels[train_indices])
        scores[test_indices] = score_model(
            fold_model,
            texts_array[test_indices].tolist(),
        )
    return scores


def score_model(model: SupportsDecisionFunction, texts: Sequence[str]) -> np.ndarray:
    """Return logistic-transformed decision scores for a fitted model."""
    margins = np.asarray(model.decision_function(texts), dtype=float)
    clipped = np.clip(margins, -50, 50)
    return 1.0 / (1.0 + np.exp(-clipped))


def select_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_recall: float,
) -> dict[str, float]:
    """Pick the highest threshold that still meets the recall target."""
    best: dict[str, float] | None = None
    for threshold in sorted({float(value) for value in scores}, reverse=True):
        metrics = evaluate_threshold(scores, labels, threshold)
        if metrics["recall"] >= target_recall:
            best = metrics
            break
    if best is None:
        msg = f"Could not find threshold meeting target recall {target_recall:.3f}"
        raise ValueError(msg)
    return best


def evaluate_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, float]:
    """Compute precision and recall for a binary threshold."""
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    predicted = scores_array >= threshold
    true_positive_count = int(np.logical_and(predicted, labels_array == 1).sum())
    false_positive_count = int(np.logical_and(predicted, labels_array == 0).sum())
    false_negative_count = int(np.logical_and(~predicted, labels_array == 1).sum())
    precision = (
        true_positive_count / (true_positive_count + false_positive_count)
        if true_positive_count + false_positive_count
        else 0.0
    )
    recall = (
        true_positive_count / (true_positive_count + false_negative_count)
        if true_positive_count + false_negative_count
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute area under the precision-recall curve."""
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def parse_label(value: str) -> int:
    """Parse a supported binary training label."""
    normalized = value.strip().upper()
    if normalized in {"T", "TRUE", "RELEVANT", "1"}:
        return 1
    if normalized in {"F", "FALSE", "IRRELEVANT", "0"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace in free text."""
    return " ".join(text.split())


def retry_failed_classifications(
    stage_name: str,
    *,
    source_dataset: str,
    output_dataset: str,
    output_columns: list[str],
    classify_partition: ClassifyPartition,
    artifact_root: str | Path | None = None,
    renew: Callable[[], None] | None = None,
) -> RetryResult:
    """Classify again the filings in a classify stage's failure list.

    The failure list is ``failures/<stage_name>/failures.json``
    (:func:`cdt.datasets.failure_registry_path`), one entry per filing the
    stage could not classify, keyed by accession number and naming the source
    partition it was read from. For each source partition, ``classify_partition``
    classifies only the source rows of those filings. Their rows in the output
    partition (same date and shard of ``output_dataset``) are replaced by the
    new ones, and every other row is written back unchanged. Filings that now
    succeed leave the list; filings that fail again keep an entry with the new
    error. A partition ``classify_partition`` holds (``complete`` False) writes
    nothing and keeps its entries.

    A source partition that is pending (changed since it was classified, or
    held) is skipped: the next plain run classifies it whole. The completion
    registry is not written, since no source partition changed. ``renew`` is
    called before each write, throttled (:func:`cdt.lease.throttled`).
    """
    resolved_root = resolve_artifact_root(artifact_root)
    failures = load_row_failures(stage_name, artifact_root=resolved_root)
    if not failures:
        return RetryResult()
    by_partition: dict[str, set[str]] = {}
    for accession, entry in failures.items():
        by_partition.setdefault(str(entry.get("source_partition")), set()).add(
            accession
        )
    pending, _ = pending_source_partitions(
        stage_name, source_dataset, artifact_root=resolved_root
    )
    pending_paths = {path for path, _ in pending}
    output_root = dataset_root(output_dataset, artifact_root=resolved_root)
    keep_lease = throttled(renew) if renew is not None else None
    retried = recovered = still_failed = skipped = 0
    for source_path, accession_set in sorted(by_partition.items()):
        accessions = frozenset(accession_set)
        if source_path in pending_paths:
            LOGGER.info(
                "%s: not retrying %s filing(s) of %s, which is pending; the next "
                "plain run classifies it whole.",
                stage_name,
                len(accessions),
                source_path,
            )
            skipped += len(accessions)
            continue
        partition = parse_date_shard_partition(source_path)
        output = classify_partition(source_path, partition, accessions)
        retried += len(accessions)
        if not output.complete:
            still_failed += len(accessions)
            continue
        if keep_lease is not None:
            keep_lease()
        output_path = date_shard_partition_path(
            output_dataset,
            partition_date=partition["date"],
            shard=partition["shard"],
            artifact_root=resolved_root,
        )
        existing = read_table(output_path, output_columns).reindex(
            columns=output_columns
        )
        unchanged = existing.loc[
            ~existing["accession_number"].astype(str).isin(accessions)
        ]
        write_partition_table(
            output_root,
            partition={"date": partition["date"], "shard": partition["shard"]},
            table=pd.concat(
                [unchanged, output.rows.reindex(columns=output_columns)],
                ignore_index=True,
            ),
        )
        replace_partition_failures(failures, source_path, output.failures)
        save_row_failures(stage_name, failures, artifact_root=resolved_root)
        failed_again = len(accessions & output.failures.keys())
        still_failed += failed_again
        recovered += len(accessions) - failed_again
    result = RetryResult(
        retried=retried,
        recovered=recovered,
        still_failed=still_failed,
        skipped=skipped,
    )
    LOGGER.info(
        "%s retry: retried=%s recovered=%s still_failed=%s skipped=%s",
        stage_name,
        result.retried,
        result.recovered,
        result.still_failed,
        result.skipped,
    )
    return result
