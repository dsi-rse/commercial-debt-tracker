"""Stub classifier stage for SEC 8-K items."""

import logging
from pathlib import Path

import pandas as pd

from cdt import settings
from cdt.itemizer.core import ITEM_COLUMNS
from cdt.storage import append_new_rows, read_table

LOGGER = logging.getLogger(__name__)
CLASSIFIED_ITEM_COLUMNS = [*ITEM_COLUMNS, "relevance"]


def classified_items_path(data_dir: Path | None = None) -> Path:
    """Return the canonical classified items table path."""
    return (
        (data_dir or settings.DATA_DIR)
        / "classified_items"
        / "classified_items.parquet"
    )


def classify_items(
    items: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Stub classifier that marks every item as not relevant."""
    path = classified_items_path(data_dir)
    existing = read_table(path, CLASSIFIED_ITEM_COLUMNS)
    requested_item_ids = set(items["item_id"]) if not items.empty else set()
    existing_item_ids = (
        set(existing["item_id"]) if not existing.empty and not force else set()
    )
    items_to_process = items.loc[~items["item_id"].isin(existing_item_ids)]

    LOGGER.info(
        "Starting classifier: %s items requested, %s items to process",
        len(items),
        len(items_to_process),
    )
    classified = items_to_process.copy()
    classified["relevance"] = False
    updated = append_new_rows(
        path,
        classified.reindex(columns=CLASSIFIED_ITEM_COLUMNS),
        ["item_id"],
        CLASSIFIED_ITEM_COLUMNS,
        replace_keys=requested_item_ids if force else None,
        replace_key_column="item_id" if force else None,
    )
    LOGGER.info("Classifier complete: %s rows at %s", len(updated), path)
    return updated
