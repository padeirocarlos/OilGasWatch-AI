"""Regular-grid resampling + quality channels (AGENT.md §5.1, §2).

Continuous signals are forward-filled across gaps up to ``max_gap_s``; beyond that
a per-channel ``<chan>__is_missing`` indicator fires and the value is set to 0 — we
never silently impute and forget (AGENT.md §2). Valve states are *never*
interpolated: they are ordinal and only ever carry-forward. Frozen (zero-variance)
sensors get explicit ``<chan>__is_frozen`` channels.

The output is gap-free: after resampling no signal column contains NaN, so every
feature function downstream can assume clean inputs and read the quality channels
to know what was synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DataConfig
from ingest.instance import InstanceFrame
from schema import CONTINUOUS_COLUMNS, SIGNAL_COLUMNS, VALVE_COLUMNS


# Quality channels are named "<signal>__is_missing" / "<signal>__is_frozen" so a model
# can attend to data-trustworthiness per channel, not just the (synthetic) values.
def _missing_suffix(col: str) -> str:
    return f"{col}__is_missing"


def _frozen_suffix(col: str) -> str:
    return f"{col}__is_frozen"


def resample(frame: InstanceFrame, cfg: DataConfig) -> InstanceFrame:
    """Resample onto a regular ``cfg.resample_rate_s`` grid and add quality channels."""
    df = frame.df
    rule = pd.Timedelta(seconds=cfg.resample_rate_s)

    # Resample onto one fixed cadence so all instances share a common time grid (raw 3W
    # sampling is irregular). ".last()" = the value as-of each tick; empty bins yield NaN.
    grid = df.resample(rule).last()
    # Longest fillable gap, expressed in grid ticks; clamp to >=1 so a tick can fill itself.
    max_gap_ticks = max(1, int(round(cfg.max_gap_s / cfg.resample_rate_s)))

    quality_cols: list[str] = []

    # --- Continuous channels: limited fill, flag the rest, then zero-fill. ----
    # Pressures/temps/flows drift smoothly, so short gaps are safely bridged by holding
    # the last value; bfill also covers a leading gap before the first observation.
    for col in CONTINUOUS_COLUMNS:
        raw_na = grid[col].isna()
        filled = grid[col].ffill(limit=max_gap_ticks).bfill(limit=max_gap_ticks)
        # Missing = bin had no observation AND no real value within max_gap to fill it.
        # (raw_na alone would also flag bins that fill legitimately bridged.)
        is_missing = (raw_na & filled.isna()).astype("int8")
        # Zero-fill the still-NaN cells so the output is gap-free; the is_missing flag is
        # what carries the "this 0.0 is synthetic, don't trust it" signal downstream.
        grid[col] = filled.fillna(0.0)
        grid[_missing_suffix(col)] = is_missing.to_numpy()
        quality_cols.append(_missing_suffix(col))

    # --- Valves: pure carry-forward (ordinal), flag absent, then zero-fill. ---
    # Valve states are ordinal {0, 0.5, 1} (§3); ffill/bfill only — interpolating would
    # invent impossible fractional openings, and there is no max_gap limit because a valve
    # holds its last commanded position until physically actuated again.
    for col in VALVE_COLUMNS:
        raw_na = grid[col].isna()
        filled = grid[col].ffill().bfill()
        is_missing = (raw_na & filled.isna()).astype("int8")
        grid[col] = filled.fillna(0.0)
        grid[_missing_suffix(col)] = is_missing.to_numpy()
        quality_cols.append(_missing_suffix(col))

    # --- Frozen indicators (continuous channels only). -----------------------
    # A "frozen" sensor reads a near-constant value because it has stuck/failed, not
    # because the process is steady; flag it so the model can discount the channel.
    win = max(2, int(round(cfg.frozen_window_s / cfg.resample_rate_s)))
    for col in CONTINUOUS_COLUMNS:
        # Rolling variance ~ 0 over the window => the trace is flat. Valves are excluded
        # because a held-open valve is *legitimately* constant and would false-positive.
        var = grid[col].rolling(win, min_periods=win).var()
        # Early rows lack a full window (var=NaN); treat as +inf so they're never "frozen".
        frozen = (var.fillna(np.inf) <= cfg.frozen_eps).astype("int8")
        grid[_frozen_suffix(col)] = frozen.to_numpy()
        quality_cols.append(_frozen_suffix(col))

    # --- Labels carry forward (an event label persists across resample bins). -
    # Labels are sparse annotations, not per-sample sensor reads, so ffill propagates the
    # active class/state across the bins between annotation points; no zero-fill (keep NaN).
    grid["class"] = grid["class"].ffill().astype("Int64")
    grid["state"] = grid["state"].ffill().astype("Int64")

    # Reorder: signals, labels, then quality channels.
    ordered = SIGNAL_COLUMNS + ["class", "state"] + quality_cols
    grid = grid[ordered]
    grid.index.name = "timestamp"
    return InstanceFrame(df=grid, meta=frame.meta, quality_columns=quality_cols)
