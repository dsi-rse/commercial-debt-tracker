"""Tests for classifier training, migration, and SQLite-backed inference."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cdt.classifier import core as classifier_core
from cdt.database import cdt_db_path, connect_cdt_db, read_items, upsert_items
from cdt.itemizer.core import ITEM_COLUMNS


class FakeModel:
    """Small deterministic model used for classifier tests."""

    def fit(self: FakeModel, texts: list[str], labels: np.ndarray) -> FakeModel:
        """Record the training payload and return self."""
        self.fit_texts = list(texts)
        self.fit_labels = labels.tolist()
        return self

    def decision_function(self: FakeModel, texts: list[str]) -> np.ndarray:
        """Return positive margins for texts mentioning debt terms."""
        return np.asarray(
            [
                1.5 if "loan" in text.lower() or "credit" in text.lower() else -1.5
                for text in texts
            ],
            dtype=float,
        )


def test_load_training_examples_parses_binary_labels(tmp_path: Path) -> None:
    """Training examples should normalize text and parse supported labels."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "text,label\n"
        '"  Loan   agreement  ",relevant\n'
        '"",irrelevant\n'
        '"Press release",FALSE\n',
        encoding="utf-8",
    )

    texts, labels = classifier_core.load_training_examples(train_csv)

    assert texts == ["Loan agreement", "Press release"]
    assert labels.tolist() == [1, 0]


def test_train_classifier_model_saves_artifacts_and_logs_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Training should persist artifacts and emit metrics in the log."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "text,label\nloan agreement,relevant\npress release,irrelevant\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    fake_scores = np.asarray([0.95, 0.10], dtype=float)

    monkeypatch.setattr(
        classifier_core,
        "build_linear_svc_pipeline",
        lambda random_seed: FakeModel(),
    )
    monkeypatch.setattr(
        classifier_core,
        "clone_model",
        lambda model: FakeModel(),
    )
    monkeypatch.setattr(
        classifier_core,
        "cross_validated_scores",
        lambda model, texts, labels, cv_splits, random_seed: fake_scores,
    )
    monkeypatch.setattr(
        classifier_core,
        "compute_pr_auc",
        lambda scores, labels: 1.0,
    )
    caplog.set_level(logging.INFO)

    metadata = classifier_core.train_classifier_model(
        train_csv=train_csv,
        model_dir=model_dir,
        target_recall=0.99,
        cv_splits=2,
        random_seed=7,
    )

    assert metadata["training_row_count"] == 2
    assert metadata["threshold"] == pytest.approx(0.95)
    assert metadata["precision"] == pytest.approx(1.0)
    assert metadata["recall"] == pytest.approx(1.0)
    assert metadata["pr_auc"] == pytest.approx(1.0)
    assert (model_dir / classifier_core.MODEL_FILENAME).exists()
    assert (model_dir / classifier_core.METADATA_FILENAME).exists()

    loaded_model, threshold, loaded_metadata = classifier_core.load_training_artifacts(
        model_dir
    )
    assert isinstance(loaded_model, FakeModel)
    assert threshold == pytest.approx(0.95)
    assert loaded_metadata["pr_auc"] == pytest.approx(1.0)
    assert "precision=" in caplog.text
    assert "recall=" in caplog.text
    assert "pr_auc=" in caplog.text


def test_connect_cdt_db_migrates_item_columns(tmp_path: Path) -> None:
    """Connecting should add classifier columns to an older items table."""
    db_path = cdt_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE items (
                item_id TEXT PRIMARY KEY,
                accession_number TEXT NOT NULL,
                item TEXT NOT NULL,
                batch_path TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    conn = connect_cdt_db(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    finally:
        conn.close()

    assert {
        "label",
        "relevance",
        "classification_score",
        "classified_at",
        "extracted_at",
        "extractor_model",
        "extractor_reasoning",
        "extractor_run_path",
        "extractor_error",
    } <= columns


def test_classify_pending_items_updates_sqlite_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending item rows should be classified once unless force is requested."""
    batch_path = tmp_path / "items" / "item-batch-000001.parquet"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    item_rows = pd.DataFrame(
        [
            {
                "item_id": "000114036126006577-8-01",
                "item": "8.01",
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/item",
                "text": "Loan agreement announced.",
                "date": "2024-01-02",
                "item_information": "Other Events",
                "extraction_status": "ok",
                "duplicate_resolution": "",
                "section_heading": "Item 8.01 Other Events.",
                "start_line": 1,
                "end_line": 3,
                "section_char_count": 25,
            }
        ],
        columns=ITEM_COLUMNS,
    )
    item_rows.to_parquet(batch_path, index=False)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        upsert_items(
            conn,
            item_rows[["item_id", "accession_number", "item"]].to_dict("records"),
            batch_path=str(batch_path),
            status="itemized",
        )
    finally:
        conn.close()

    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    first = classifier_core.classify_pending_items(data_dir=tmp_path, batch_size=1)
    second = classifier_core.classify_pending_items(data_dir=tmp_path, batch_size=1)
    forced = classifier_core.classify_pending_items(
        data_dir=tmp_path,
        batch_size=1,
        force=True,
    )

    assert first["label"].to_list() == ["relevant"]
    assert first["relevance"].to_list() == [True]
    assert second.empty
    assert forced["label"].to_list() == ["relevant"]

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        rows = read_items(conn)
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["item_id"] == "000114036126006577-8-01"
    assert rows[0]["status"] == "classified"
    assert rows[0]["label"] == "relevant"
    assert rows[0]["relevance"] == 1
    assert rows[0]["classification_score"] > 0.5
    assert rows[0]["classified_at"]


def test_classify_pending_items_logs_relevant_and_irrelevant_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Classifier logs end-of-run relevant and irrelevant totals."""
    batch_path = tmp_path / "items" / "item-batch-000001.parquet"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    item_rows = pd.DataFrame(
        [
            {
                "item_id": "000114036126006577-8-01",
                "item": "8.01",
                "accession_number": "000114036126006577",
                "cik": "320193",
                "url": "https://sec.example/item-1",
                "text": "Loan agreement announced.",
                "date": "2024-01-02",
                "item_information": "Other Events",
                "extraction_status": "ok",
                "duplicate_resolution": "",
                "section_heading": "Item 8.01 Other Events.",
                "start_line": 1,
                "end_line": 3,
                "section_char_count": 25,
            },
            {
                "item_id": "000114036126006578-8-01",
                "item": "8.01",
                "accession_number": "000114036126006578",
                "cik": "320193",
                "url": "https://sec.example/item-2",
                "text": "General corporate update.",
                "date": "2024-01-03",
                "item_information": "Other Events",
                "extraction_status": "ok",
                "duplicate_resolution": "",
                "section_heading": "Item 8.01 Other Events.",
                "start_line": 1,
                "end_line": 3,
                "section_char_count": 25,
            },
        ],
        columns=ITEM_COLUMNS,
    )
    item_rows.to_parquet(batch_path, index=False)

    conn = connect_cdt_db(cdt_db_path(tmp_path))
    try:
        upsert_items(
            conn,
            item_rows[["item_id", "accession_number", "item"]].to_dict("records"),
            batch_path=str(batch_path),
            status="itemized",
        )
    finally:
        conn.close()

    monkeypatch.setattr(
        classifier_core,
        "load_training_artifacts",
        lambda path: (FakeModel(), 0.5, {"threshold": 0.5}),
    )

    with caplog.at_level(logging.INFO, logger="cdt.classifier.core"):
        classifier_core.classify_pending_items(data_dir=tmp_path, batch_size=10)

    assert "total_classified=2" in caplog.text
    assert "relevant=1" in caplog.text
    assert "irrelevant=1" in caplog.text
