"""Column types and stored values: declared physical types, decimals, text and JSON coercion."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet

MISSING_TEXT_VALUES = frozenset({"nan", "none", "null", "<na>", "n/a"})


# Every published column's physical type, declared rather than inferred from
# values, so a column has the same type in every partition. Keyed by name because
# a column name means one thing across datasets. Money and rates are exact
# decimals, never float. Rationale: docs/decisions/storage-and-completion.md.
DECLARED_COLUMN_TYPES: dict[str, pa.DataType] = {
    "principal_amount": pa.decimal128(38, 2),
    "outstanding_balance": pa.decimal128(38, 2),
    # Four places carries basis points with room to spare; the corpus uses at
    # most three.
    "interest_rate_pct": pa.decimal128(9, 4),
    "mention_count": pa.int64(),
    "document_count": pa.int64(),
    "candidate_rank": pa.int64(),
    "start_line": pa.int64(),
    "end_line": pa.int64(),
    "section_char_count": pa.int64(),
    "is_lineage_head": pa.bool_(),
    "relevance": pa.bool_(),
    "synthesized_only": pa.bool_(),
    "outstanding_balance_as_of_is_filing_date": pa.bool_(),
    # Model scores, not measured quantities, so a float is the honest type.
    "classification_score": pa.float64(),
    "match_score": pa.float64(),
}


# Everything not declared above is nullable text, which is the contract
# `docs/schema.md` states.
DEFAULT_COLUMN_TYPE = pa.string()


def canonical_numeric_text(value: Decimal) -> str:
    """Return one deterministic numeric string for a decimal value.

    Trailing zeros are dropped (``Decimal("5.0000")`` -> ``"5"``) so a value has
    one spelling: decimal columns read back scale-padded, and the pipeline
    compares amounts as text.
    """
    quantized = value.normalize()
    if quantized == quantized.to_integral_value():
        quantized = quantized.to_integral_value()
    return f"{quantized:f}"


def decimal_column_values(
    values: Iterable[object], dtype: pa.Decimal128Type, *, column: str
) -> list[Decimal | None]:
    """Return one column's values as decimals quantized half-up to ``dtype``'s scale.

    Digits beyond the scale are rounded away rather than rejected, and
    placeholder text (``nan``, empty) becomes None. Raises ValueError for text
    that is not a number: these columns are always written from a parsed
    amount, so that is an upstream bug.
    """
    exponent = Decimal(1).scaleb(-dtype.scale)
    coerced: list[Decimal | None] = []
    for value in values:
        if isinstance(value, Decimal):
            coerced.append(value.quantize(exponent, rounding=ROUND_HALF_UP))
            continue
        text = coerce_dataset_text(value)
        if text is None:
            coerced.append(None)
            continue
        try:
            coerced.append(Decimal(text).quantize(exponent, rounding=ROUND_HALF_UP))
        except InvalidOperation as error:
            message = f"{column} is not a number: {text!r}"
            raise ValueError(message) from error
    return coerced


def declared_column_type(name: str, inferred: pa.DataType) -> pa.DataType:
    """Return the physical type one column publishes as.

    A declared type wins. Otherwise an inferred ``null`` (an object column with
    no value in this frame) becomes ``DEFAULT_COLUMN_TYPE``, and any other
    inferred type, which came from a real pandas dtype, is kept.
    """
    declared = DECLARED_COLUMN_TYPES.get(name)
    if declared is not None:
        return declared
    if pa.types.is_null(inferred):
        return DEFAULT_COLUMN_TYPE
    return inferred


def apply_declared_column_types(table: pd.DataFrame) -> pa.Table:
    """Return ``table`` as an Arrow table with the declared physical types applied.

    Declared decimal columns are canonicalised to text first: rewriting an
    existing partition mixes parquet-read ``Decimal`` values with in-memory
    text in one object column, which ``Table.from_pandas`` rejects.
    """
    prepared = table.copy()
    for name, dtype in DECLARED_COLUMN_TYPES.items():
        if name in prepared.columns and pa.types.is_decimal(dtype):
            prepared[name] = pd.Series(
                [coerce_dataset_text(value) for value in prepared[name]],
                index=prepared.index,
                dtype=object,
            )
    arrow = pa.Table.from_pandas(prepared, preserve_index=False)
    for index, field in enumerate(arrow.schema):
        dtype = declared_column_type(field.name, field.type)
        if dtype == field.type:
            continue
        if pa.types.is_decimal(dtype):
            values = decimal_column_values(
                arrow.column(field.name).to_pylist(), dtype, column=field.name
            )
            column = pa.array(values, type=dtype)
        else:
            column = arrow.column(field.name).cast(dtype)
        arrow = arrow.set_column(index, pa.field(field.name, dtype), column)
    return arrow


def coerce_dataset_text(value: object) -> str | None:
    """Return one trimmed dataset text value, or None when it carries no name.

    Parquet round-trips a missing value as NaN, and ``str(float("nan"))`` is the
    literal text ``nan``, so placeholder strings are treated as missing too.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        # Not ``str()``: that gives the scale-padded ``2000000000.00`` where the
        # pipeline wrote ``2000000000``, and textual comparisons would miss.
        return canonical_numeric_text(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in MISSING_TEXT_VALUES:
        return None
    return text or None


def json_column(row: Mapping[str, object], column: str) -> object | None:
    """Parse one JSON text column; None when it is absent, missing or not JSON."""
    text = coerce_dataset_text(row.get(column))
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
