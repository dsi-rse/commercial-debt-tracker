"""Pipeline orchestration for SEC 8-K document processing."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cdt.classifier import classify_items
from cdt.extractor import extract_tables
from cdt.ingest import acquire_documents
from cdt.itemizer import itemize_documents
from cdt.matcher import match_tables


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single 8-K pipeline invocation."""

    bucket: str
    year: int
    ciks: set[str] | None = None
    data_dir: Path | None = None
    force: bool = False


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Run acquisition, itemization, classification, extraction, and matching."""
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
    extracted = extract_tables(
        classified_items, data_dir=config.data_dir, force=config.force
    )
    mentions = extracted["instrument_mentions"]
    if mentions.empty:
        return {
            **extracted,
            **match_tables(pd.DataFrame()),
        }
    mention_context = classified_items[
        ["item_id", "accession_number", "cik", "date"]
    ].drop_duplicates()
    matcher_input = mentions.merge(mention_context, on="item_id", how="left")
    return {
        **extracted,
        **match_tables(matcher_input),
    }
