"""Sliding-window construction and per-window labelling (AGENT.md §5.3).

Windows carry the event ``class``, ``state``, and the transient flag. For the GBT
baseline we flatten each window into summary statistics; the deep model consumes the
raw windowed sequence (see :func:`sequence_windows`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import WindowConfig
from schema import TRANSIENT_OFFSET

#: Summary statistics used to flatten a window for the GBT baseline.
AGG_FUNCS = ("mean", "std", "min", "max", "last")


@dataclass
class WindowLabels:
    event: np.ndarray  # (n_win,) decoded event code in 0..9
    transient: np.ndarray  # (n_win,) 1 if window is in the onset/transient window
    state: np.ndarray  # (n_win,) operational state (or -1 if unknown)
    valid: np.ndarray  # (n_win,) 1 if the window had any labelled timestep


def window_bounds(n: int, cfg: WindowConfig, rate_s: float) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index pairs for full-size windows only."""
    # Config is in seconds (no magic numbers, AGENT.md §2); convert to sample counts via
    # the resample rate. window = window length, stride = hop between consecutive windows.
    # A smaller stride => more overlap => more windows (denser coverage, more correlation).
    win = max(1, int(round(cfg.window_s / rate_s)))
    stride = max(1, int(round(cfg.stride_s / rate_s)))
    min_win = max(1, int(round(cfg.min_window_s / rate_s)))
    bounds = []
    start = 0
    # Emit only full-size windows so every window has identical length — required for
    # a fixed-shape feature matrix (GBT) and a fixed-length tensor (deep model).
    while start + win <= n:
        bounds.append((start, start + win))
        start += stride
    # Fallback: an instance shorter than one full window still yields a single window,
    # provided it clears the configured minimum length, so short instances aren't dropped.
    if not bounds and n >= min_win:
        bounds.append((0, n))
    return bounds


def aggregate_features(
    feat_df: pd.DataFrame, bounds: list[tuple[int, int]], columns: list[str]
) -> np.ndarray:
    """Flatten each window into ``len(columns) * len(AGG_FUNCS)`` summary features."""
    # GBT path: collapse the time axis into per-channel summary stats so a tree model
    # (which has no notion of sequence) sees one fixed-width row per window.
    arr = feat_df[columns].to_numpy(dtype=np.float64)
    out = np.empty((len(bounds), len(columns) * len(AGG_FUNCS)), dtype=np.float64)
    for i, (s, e) in enumerate(bounds):
        seg = arr[s:e]
        # Order here must match AGG_FUNCS / aggregate_feature_names so column names line up.
        # `last` preserves the window's endpoint value — informative for monotone trends.
        stats = [
            seg.mean(axis=0),
            seg.std(axis=0),
            seg.min(axis=0),
            seg.max(axis=0),
            seg[-1],
        ]
        out[i] = np.concatenate(stats)
    # Trees can't handle NaN/inf; replace them so a single bad cell doesn't poison a window.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def aggregate_feature_names(columns: list[str]) -> list[str]:
    # Names mirror the concatenation order in aggregate_features: all channels for the
    # first stat, then all channels for the next — so X[:, j] maps to name[j].
    return [f"{c}__{fn}" for fn in AGG_FUNCS for c in columns]


def label_windows(
    class_series: pd.Series, state_series: pd.Series, bounds: list[tuple[int, int]]
) -> WindowLabels:
    """Assign one event/transient/state label per window.

    Event = majority decoded code over labelled timesteps. Transient = 1 when at
    least half the labelled timesteps fall in the onset window (raw code ≥ 100).
    """
    raw = class_series.to_numpy()  # object array with possible pd.NA
    state = state_series.to_numpy()
    n = len(bounds)
    event = np.zeros(n, dtype=np.int64)
    transient = np.zeros(n, dtype=np.int8)
    st = np.full(n, -1, dtype=np.int64)  # -1 = unknown state
    valid = np.zeros(n, dtype=np.int8)

    for i, (s, e) in enumerate(bounds):
        seg = raw[s:e]
        # Only labelled timesteps vote; unlabelled (pd.NA) ones are ignored entirely.
        mask = np.array([not pd.isna(v) for v in seg])
        if not mask.any():
            continue  # a window with no label at all is left invalid (valid[i] stays 0)
        valid[i] = 1
        vals = np.array([int(v) for v in seg[mask]], dtype=np.int64)
        # Onset (transient) timesteps carry code+TRANSIENT_OFFSET (e.g. 1 -> 101); strip
        # the offset back to the base class so the event label is 0..9 regardless of phase.
        is_trans = vals >= TRANSIENT_OFFSET
        codes = np.where(is_trans, vals - TRANSIENT_OFFSET, vals)
        # Majority vote: the window's event is whichever base code dominates its timesteps.
        event[i] = np.bincount(codes).argmax()
        # Label the *window* transient only if onset spans at least half of it — this is
        # the early-detection target (AGENT.md §2): catch the onset, not just the settled fault.
        transient[i] = 1 if is_trans.mean() >= 0.5 else 0
        # Operational state is an auxiliary label; majority-vote it the same way.
        seg_state = state[s:e][mask]
        seg_state = np.array([int(v) for v in seg_state if not pd.isna(v)], dtype=np.int64)
        if len(seg_state):
            st[i] = np.bincount(seg_state).argmax()
    return WindowLabels(event=event, transient=transient, state=st, valid=valid)


def sequence_windows(
    signal_df: pd.DataFrame, bounds: list[tuple[int, int]], columns: list[str]
) -> np.ndarray:
    """Stack raw windowed sequences into ``(n_win, window_len, n_channels)`` for the deep model."""
    # Deep path: unlike the GBT path, keep the full time axis so the network can learn
    # temporal dynamics directly from the raw windowed signal.
    arr = signal_df[columns].to_numpy(dtype=np.float32)
    if not bounds:
        return np.empty((0, 0, len(columns)), dtype=np.float32)
    # All full-size windows share this length; the fallback partial window is zero-padded
    # up to it below so the output tensor stays rectangular.
    win_len = bounds[0][1] - bounds[0][0]
    out = np.zeros((len(bounds), win_len, len(columns)), dtype=np.float32)
    for i, (s, e) in enumerate(bounds):
        seg = arr[s:e]
        out[i, : len(seg)] = seg  # left-align; trailing rows stay zero if seg is short
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
