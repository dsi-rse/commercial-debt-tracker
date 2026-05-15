"""Pipeline orchestration for SEC 8-K document processing."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cdt.classifier import classify_items
from cdt.extractor import extract_tables
from cdt.ingest import acquire_documents
from cdt.itemizer import itemize_documents


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single 8-K pipeline invocation."""

    bucket: str
    year: int
    ciks: set[str] | None = None
    data_dir: Path | None = None
    force: bool = False


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Run acquisition, itemization, classification, and extraction."""
    documents = acquire_documents(
        config.bucket,
        config.year,
        config.ciks,
        data_dir=config.data_dir,
        force=config.force,
    )
    items = itemize_documents(documents, data_dir=config.data_dir, force=config.force)
    classified_items = classify_items(
        items, data_dir=config.data_dir, force=config.force
    )
    return extract_tables(
        classified_items, data_dir=config.data_dir, force=config.force
    )
