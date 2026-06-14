"""Parquet IO + schema validation (AGENT.md §5.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ingest.instance import InstanceFrame, parse_meta
from schema import ALL_COLUMNS, SIGNAL_COLUMNS


class SchemaError(ValueError):
    """Raised when an instance does not match the authoritative 29-column schema."""


def _validate_columns(df: pd.DataFrame, path: str) -> None:
    # AGENT.md §8 forbids inventing or dropping columns: the §3 schema is authoritative.
    # Strict both ways — an unexpected column likely means a parsing/source bug, and a
    # missing one would silently break a downstream feature that assumes the channel.
    cols = set(df.columns)
    expected = set(ALL_COLUMNS)
    unknown = cols - expected
    if unknown:
        raise SchemaError(f"{path}: unknown columns {sorted(unknown)} (schema is authoritative)")
    missing = expected - cols
    if missing:
        raise SchemaError(f"{path}: missing required columns {sorted(missing)}")


def load_instance(path: str | Path) -> InstanceFrame:
    """Read a parquet instance, coerce dtypes, parse the timestamp index, validate.

    Raises :class:`SchemaError` on unknown/missing columns (AGENT.md §5.1).
    """
    path = str(path)
    df = pd.read_parquet(path, engine="pyarrow")  # pyarrow/brotli is the 3W storage format (§3)

    # 3W stores the timestamp as the index; tolerate it as a column too.
    if df.index.name != "timestamp":
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise SchemaError(f"{path}: no 'timestamp' index or column")
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    # Sort chronologically so resampling and forward-fill see time in order.
    df = df.sort_index()

    _validate_columns(df, path)

    # Sensor signals -> float64 for arithmetic features. Labels -> pandas nullable Int64:
    # 'class'/'state' are categorical codes (§3) and may be NaN at recording edges, so a
    # plain int can't hold them and a float would misrepresent discrete class codes.
    df[SIGNAL_COLUMNS] = df[SIGNAL_COLUMNS].astype("float64")
    df["class"] = df["class"].astype("Int64")
    df["state"] = df["state"].astype("Int64")

    # Reindex to canonical column order so every InstanceFrame is layout-identical.
    return InstanceFrame(df=df[ALL_COLUMNS], meta=parse_meta(path))
