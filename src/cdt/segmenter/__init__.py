"""Segmenter stage: cut documents into the item rows the classifier reads.

The code says *segment*; the data keeps its original names (``items``,
``item_id``), which the published tables and the completion registry depend on.
"""

from cdt.segmenter.core import items_root
from cdt.segmenter.eightk import (
    POTENTIALLY_RELEVANT_ITEM_NUMBERS,
    ItemSection,
    extract_items_from_document,
    item_id_for,
    itemize_document_record,
    itemize_documents,
    itemize_pending_documents,
)
from cdt.segmenter.text import DocumentText

__all__ = [
    "DocumentText",
    "ItemSection",
    "POTENTIALLY_RELEVANT_ITEM_NUMBERS",
    "extract_items_from_document",
    "item_id_for",
    "itemize_document_record",
    "itemize_documents",
    "itemize_pending_documents",
    "items_root",
]
