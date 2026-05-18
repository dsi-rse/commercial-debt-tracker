"""Binary relevance classifier for SEC 8-K item rows."""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

from cdt import settings
from cdt.database import (
    ITEM_STATUSES,
    cdt_db_path,
    connect_cdt_db,
    mark_items_classified,
    read_items,
)
from cdt.itemizer.core import ITEM_COLUMNS
from cdt.storage import read_table

LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_RECALL = 0.99
DEFAULT_RANDOM_SEED = 42
DEFAULT_CV_SPLITS = 5
MIN_CLASS_COUNT = 2
MIN_CV_SPLITS = 2
MODEL_NAME = "tfidf_linear_svc"
MODEL_FILENAME = "model.pkl"
METADATA_FILENAME = "metadata.json"
CLASSIFIED_ITEM_COLUMNS = [*ITEM_COLUMNS, "label", "relevance", "classification_score"]


@runtime_checkable
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


def default_model_dir(data_dir: Path | None = None) -> Path:
    """Return the default model artifact directory."""
    return (
        (data_dir or settings.DATA_DIR) / "models" / "classifier" / "tfidf-linear-svc"
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

    metadata = {
        "model_name": MODEL_NAME,
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


def classify_items(
    items: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Classify in-memory item rows using a saved binary model."""
    del force
    if items.empty:
        return pd.DataFrame(columns=CLASSIFIED_ITEM_COLUMNS)

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
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    batch_size: int = 100,
    force: bool = False,
) -> pd.DataFrame:
    """Classify pending item rows indexed in SQLite and update them in place."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    conn = connect_cdt_db(cdt_db_path(data_dir))
    processed_item_ids: set[str] = set()
    processed_frames: list[pd.DataFrame] = []
    total_rows = 0
    try:
        while True:
            index_rows = read_items(
                conn,
                statuses=ITEM_STATUSES if force else ("itemized",),
                exclude_item_ids=processed_item_ids,
                limit=batch_size,
            )
            if not index_rows:
                break
            batch_items = load_pending_item_batches(index_rows, data_dir=data_dir)
            classified = classify_items(
                batch_items,
                data_dir=data_dir,
                model_dir=model_dir,
            )
            mark_items_classified(
                conn,
                classified[
                    ["item_id", "label", "relevance", "classification_score"]
                ].to_dict("records"),
            )
            processed_frames.append(classified)
            processed_item_ids.update(str(item_id) for item_id in classified["item_id"])
            total_rows += len(classified)
    finally:
        conn.close()

    LOGGER.info("Classifier complete: %s item rows updated", total_rows)
    if not processed_frames:
        return pd.DataFrame(columns=CLASSIFIED_ITEM_COLUMNS)
    return pd.concat(processed_frames, ignore_index=True).reindex(
        columns=CLASSIFIED_ITEM_COLUMNS
    )


def load_training_artifacts(
    model_dir: Path,
) -> tuple[SupportsDecisionFunction, float, dict[str, object]]:
    """Load a fitted model and metadata from disk."""
    model_path = model_dir / MODEL_FILENAME
    metadata_path = model_dir / METADATA_FILENAME
    with model_path.open("rb") as file_obj:
        model = pickle.load(file_obj)  # noqa: S301
    with metadata_path.open(encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
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


def load_pending_item_batches(
    index_rows: Sequence[dict[str, object]],
    *,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load item parquet rows referenced by SQLite index rows."""
    if not index_rows:
        return pd.DataFrame(columns=ITEM_COLUMNS)

    item_ids_by_batch_path: dict[str, list[str]] = defaultdict(list)
    item_id_order: list[str] = []
    for row in index_rows:
        batch_path = str(row["batch_path"])
        item_id = str(row["item_id"])
        item_ids_by_batch_path[batch_path].append(item_id)
        item_id_order.append(item_id)

    frames: list[pd.DataFrame] = []
    for batch_path, item_ids in item_ids_by_batch_path.items():
        path = Path(batch_path)
        if not path.is_absolute():
            path = (data_dir or settings.DATA_DIR) / path
        batch = read_table(path, ITEM_COLUMNS).reindex(columns=ITEM_COLUMNS)
        selected = batch.loc[batch["item_id"].isin(item_ids)]
        missing_item_ids = set(item_ids).difference(selected["item_id"])
        if missing_item_ids:
            msg = (
                f"Item batch {path} is missing expected item IDs: "
                f"{sorted(missing_item_ids)}"
            )
            raise ValueError(msg)
        frames.append(selected)

    combined = pd.concat(frames, ignore_index=True)
    ordered = combined.set_index("item_id").loc[item_id_order].reset_index()
    return ordered.reindex(columns=ITEM_COLUMNS)
