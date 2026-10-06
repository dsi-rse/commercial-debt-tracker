"""Tests for the genre registry and the lists derived from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from cdt.datasets import GENRE_6K, GENRE_8K, GENRES, dataset_root
from cdt.extractor.outputs import CLASSIFICATION_SOURCES
from cdt.pipeline import DEFAULT_GENRES, normalize_genres
from cdt.publish import FINAL_OUTPUT_TABLE_FORM_TYPES, FINAL_OUTPUT_TABLES


def test_the_registry_lists_both_genres_in_prepare_order() -> None:
    """8-K prepares before 6-K, and each record is keyed by its own name."""
    assert list(GENRES) == [GENRE_8K, GENRE_6K]
    assert all(key == genre.name for key, genre in GENRES.items())


def test_each_genre_names_its_forms_and_datasets() -> None:
    """The records carry the values the stages used as separate constants."""
    eightk, sixk = GENRES[GENRE_8K], GENRES[GENRE_6K]
    assert (
        eightk.form_types,
        eightk.document_dataset,
        eightk.item_dataset,
        eightk.classified_dataset,
    ) == (("8-K",), "documents", "items", "classifications")
    assert (
        sixk.form_types,
        sixk.document_dataset,
        sixk.item_dataset,
        sixk.classified_dataset,
    ) == (("6-K", "6-K/A"), "documents-sixk", "sixk-snippets", "sixk-snippets")


def test_derived_lists_equal_the_literals_they_replace() -> None:
    """The extractor's sources, the CLI default and the items form types."""
    assert CLASSIFICATION_SOURCES == ("classifications", "sixk-snippets")
    assert DEFAULT_GENRES == ("8-K", "6-K")
    assert FINAL_OUTPUT_TABLE_FORM_TYPES == {"items": ("8-K", "6-K")}


def test_the_items_union_reads_each_genres_item_dataset_in_genre_order(
    tmp_path: Path,
) -> None:
    """Root ``i`` of the published items table is genre ``i``'s item dataset.

    FINAL_OUTPUT_TABLE_FORM_TYPES stamps ``form_type`` by position, so the two
    must line up.
    """
    roots = [root_fn(str(tmp_path)) for root_fn in FINAL_OUTPUT_TABLES["items"]]
    assert roots == [
        dataset_root(genre.item_dataset, artifact_root=str(tmp_path))
        for genre in GENRES.values()
    ]


def test_the_items_root_functions_keep_the_names_the_publish_digest_hashes() -> None:
    """Deriving these from the registry would rename them and force a republish."""
    assert [fn.__name__ for fn in FINAL_OUTPUT_TABLES["items"]] == [
        "items_root",
        "sixk_snippets_root",
    ]


def test_normalize_genres_orders_by_the_registry_and_rejects_unknowns() -> None:
    """Selection order follows the registry; an unknown genre is an error."""
    assert normalize_genres("6-k, 8-K") == ("8-K", "6-K")
    with pytest.raises(ValueError, match="unknown genre"):
        normalize_genres("8-K,10-Q")
    with pytest.raises(ValueError, match="no genres selected"):
        normalize_genres(" , ")
