"""Stub extractor stage for relevant SEC 8-K items."""

import json
import logging
from pathlib import Path

import pandas as pd

from cdt import settings

LOGGER = logging.getLogger(__name__)


def extracted_tables_path(data_dir: Path | None = None) -> Path:
    """Return the canonical extracted tables directory."""
    return (data_dir or settings.DATA_DIR) / "extracted_tables"


def extract_tables(
    classified_items: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Stub extractor that records completion and returns no extracted tables."""
    del force
    output_dir = extracted_tables_path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relevant_count = (
        int(classified_items["relevance"].sum())
        if "relevance" in classified_items
        else 0
    )
    metadata = {
        "stage": "extractor",
        "status": "stub",
        "relevant_items_seen": relevant_count,
    }
    metadata_path = output_dir / "_SUCCESS.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info(
        "Extractor complete: %s relevant items seen at %s",
        relevant_count,
        metadata_path,
    )
    return {}
