"""Helpers for reading and updating pipeline artifacts."""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


def read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a Parquet table or return an empty table when it does not exist."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Atomically write a Parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, suffix=".parquet", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        table.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def next_batch_path(directory: Path, prefix: str) -> Path:
    """Return the next sequential Parquet batch path in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.parquet$")
    highest = 0
    for path in directory.glob(f"{prefix}-*.parquet"):
        match = pattern.match(path.name)
        if match is None:
            continue
        highest = max(highest, int(match.group(1)))
    return directory / f"{prefix}-{highest + 1:06d}.parquet"


def write_parquet_batch(directory: Path, prefix: str, table: pd.DataFrame) -> Path:
    """Write a sequentially numbered Parquet batch and return its path."""
    path = next_batch_path(directory, prefix)
    write_table(path, table)
    return path


def append_new_rows(
    path: Path,
    rows: pd.DataFrame,
    key_columns: Sequence[str],
    columns: Sequence[str],
    *,
    replace_keys: Iterable[object] | None = None,
    replace_key_column: str | None = None,
) -> pd.DataFrame:
    """Append rows to a keyed Parquet table and return the full updated table."""
    existing = read_table(path, columns)
    existing = existing.reindex(columns=columns)
    rows = rows.reindex(columns=columns)
    if existing.empty and rows.empty:
        write_table(path, existing)
        return existing

    if (
        replace_keys is not None
        and replace_key_column is not None
        and not existing.empty
    ):
        existing = existing.loc[~existing[replace_key_column].isin(set(replace_keys))]

    if rows.empty:
        updated = existing
    elif existing.empty:
        updated = rows
    else:
        combined = pd.concat([existing, rows], ignore_index=True)
        updated = combined.drop_duplicates(subset=list(key_columns), keep="last")

    write_table(path, updated)
    return updated


def missing_keys(
    existing: pd.DataFrame, candidates: Iterable[object], key_column: str
) -> set[object]:
    """Return candidate keys that are absent from an existing table."""
    candidate_set = set(candidates)
    if existing.empty or key_column not in existing:
        return candidate_set
    return candidate_set.difference(set(existing[key_column]))
