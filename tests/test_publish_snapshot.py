"""Tests for normalizing snapshot text on the way into a published table."""

from __future__ import annotations

import pandas as pd

from cdt.publish import normalize_snapshot_text


def test_normalize_snapshot_text_nulls_placeholder_strings() -> None:
    """Dashboard-facing snapshots must not carry literal placeholder text."""
    table = pd.DataFrame(
        [
            {"company_name": "nan", "name": "convertible debentures", "amount": 1.5},
            {"company_name": "Versigent PLC", "name": "None", "amount": 2.5},
        ]
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["company_name"].to_list() == [None, "Versigent PLC"]
    assert normalized["name"].to_list() == ["convertible debentures", None]
    assert normalized["amount"].to_list() == [1.5, 2.5]


def test_normalize_snapshot_text_keeps_non_text_values_typed() -> None:
    """Booleans in an object column must not be published as text."""
    # Partitions written before lenders_known_incomplete existed leave an object
    # column holding booleans and nulls side by side.
    table = pd.DataFrame(
        {
            "lenders_known_incomplete": [True, None, False],
            "company_name": ["Acme Inc.", "nan", "Contoso Ltd."],
        }
    )

    normalized = normalize_snapshot_text(table)

    assert normalized["lenders_known_incomplete"].to_list() == [True, None, False]
    assert normalized["company_name"].to_list() == [
        "Acme Inc.",
        None,
        "Contoso Ltd.",
    ]
