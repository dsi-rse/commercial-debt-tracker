"""Tests for nulling placeholder text on the way into a published table."""

from __future__ import annotations

import pyarrow as pa

from cdt.publish import null_placeholder_text


def test_placeholder_text_is_published_as_null() -> None:
    """Dashboard-facing snapshots must not carry literal placeholder text."""
    table = pa.table(
        {
            "company_name": ["nan", "Versigent PLC", "  ", None],
            "name": ["convertible debentures", "None", " N/A ", "Nantucket Bank "],
            "amount": [1.5, 2.5, 3.5, 4.5],
        }
    )

    normalized = null_placeholder_text(table)

    assert normalized["company_name"].to_pylist() == [None, "Versigent PLC", None, None]
    assert normalized["name"].to_pylist() == [
        "convertible debentures",
        None,
        None,
        "Nantucket Bank",
    ]
    assert normalized["amount"].to_pylist() == [1.5, 2.5, 3.5, 4.5]


def test_non_text_columns_keep_their_types() -> None:
    """Booleans and numbers are never rewritten as text."""
    table = pa.table({"flag": [True, None, False], "count": [1, 2, None]})

    normalized = null_placeholder_text(table)

    assert normalized.schema == table.schema
    assert normalized["flag"].to_pylist() == [True, None, False]
