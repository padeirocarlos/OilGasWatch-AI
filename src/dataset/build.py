"""Materialise a windowed dataset from instance metadata (AGENT.md §5.3).

Ties together ingest → features → windowing for a list of instances, producing the
flat arrays the baseline and eval harness consume. Each window keeps its originating
well id and source so evaluation can stay well-disjoint and stratify by provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DataConfig, FeaturesConfig, ModelConfig
from dataset.windows import (
    aggregate_feature_names,
    aggregate_features,
    label_windows,
    sequence_windows,
    window_bounds,
)
from features.pipeline import build_features
from ingest import load_and_resample
from ingest.instance import InstanceMeta
from schema import SIGNAL_COLUMNS


@dataclass
class WindowDataset:
    """Flattened windows ready for the GBT baseline / eval."""

    X: np.ndarray  # (n_win, n_agg_features)
    event: np.ndarray  # (n_win,) event code 0..9
    transient: np.ndarray  # (n_win,) 0/1
    state: np.ndarray  # (n_win,)
    # Provenance carried per window so eval can group by well (well-disjoint scoring),
    # stratify/exclude by source (real vs simulated vs hand-drawn, AGENT.md §3), and trace
    # a window back to its instance. `order` is the start index = chronological position
    # within the instance, needed for time-to-detection (latency from onset).
    well_id: np.ndarray  # (n_win,) object
    source: np.ndarray  # (n_win,) object
    instance_id: np.ndarray  # (n_win,) object
    order: np.ndarray  # (n_win,) window start sample index (chronological within instance)
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


def build_window_dataset(
    metas: Sequence[InstanceMeta],
    dcfg: DataConfig,
    fcfg: FeaturesConfig,
    mcfg: ModelConfig,
    feature_columns: list[str] | None = None,
    progress: bool = True,
) -> WindowDataset:
    # Accumulate per-instance arrays, then concatenate once at the end (cheaper than
    # growing one big array instance-by-instance).
    Xs, ev, tr, stt, wid, src, iid, ordr = [], [], [], [], [], [], [], []
    cols: list[str] | None = feature_columns
    names: list[str] = []

    iterator = tqdm(metas, desc="windowing", disable=not progress)
    for meta in iterator:
        # Per instance: ingest+resample, then build engineered features (ingest->features).
        inst = load_and_resample(meta.path, dcfg)
        feats = build_features(inst, fcfg)
        # Lock the feature column set from the first instance so every instance flattens to
        # the same columns in the same order. Excluded as model inputs: the label columns,
        # and any "__"-prefixed *diagnostic* column. The latter matters for correctness:
        # `__norm_fallback` is 1 exactly for the steady-state classes' fault-only files (no
        # NORMAL segment), so it leaks the label — feeding it to the GBT let it "cheat" on
        # classes 3/4. Per-sensor quality channels ("<chan>__is_missing/__is_frozen") start
        # with a channel name, not "__", so they are kept (AGENT.md §2: quality is a feature).
        if cols is None:
            cols = [
                c for c in feats.columns if c not in ("class", "state") and not c.startswith("__")
            ]
            names = aggregate_feature_names(cols)

        bounds = window_bounds(len(feats), mcfg.window, dcfg.resample_rate_s)
        if not bounds:
            continue  # instance too short to yield any window
        labels = label_windows(feats["class"], feats["state"], bounds)
        # Drop windows that had no labelled timestep — they have no target to learn from.
        keep = labels.valid.astype(bool)
        if not keep.any():
            continue

        X = aggregate_features(feats, bounds, cols)
        # Apply the same `keep` mask to features, labels, and every provenance column so
        # all arrays stay row-aligned across the concatenation.
        Xs.append(X[keep])
        ev.append(labels.event[keep])
        tr.append(labels.transient[keep])
        stt.append(labels.state[keep])
        n = int(keep.sum())
        wid.append(np.full(n, meta.well_id, dtype=object))
        src.append(np.full(n, meta.source.name, dtype=object))
        iid.append(np.full(n, meta.instance_id, dtype=object))
        starts = np.array([s for s, _ in bounds], dtype=np.int64)
        ordr.append(starts[keep])  # window start index = chronological order tag

    if not Xs:
        raise ValueError("no valid windows produced from the given instances")

    return WindowDataset(
        X=np.concatenate(Xs),
        event=np.concatenate(ev),
        transient=np.concatenate(tr),
        state=np.concatenate(stt),
        well_id=np.concatenate(wid),
        source=np.concatenate(src),
        instance_id=np.concatenate(iid),
        order=np.concatenate(ordr),
        feature_names=names,
    )


@dataclass
class SequenceDataset:
    """Raw windowed signal sequences for the ST-MoE network."""

    X: np.ndarray  # (n_win, window_len, 27) float32, in SIGNAL_COLUMNS order
    event: np.ndarray
    transient: np.ndarray
    state: np.ndarray
    well_id: np.ndarray  # kept for well-disjoint eval grouping (see WindowDataset)
    # instance_id + order mirror WindowDataset so the deep model's transient head can be
    # scored for time-to-detection (latency from onset) on the same footing as the baseline.
    instance_id: np.ndarray  # (n_win,) object
    order: np.ndarray  # (n_win,) window start sample index (chronological within instance)

    def __len__(self) -> int:
        return len(self.X)


def _prepare_instance_windows(
    meta: InstanceMeta, dcfg: DataConfig, mcfg: ModelConfig, normalise_signals: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Ingest one instance and produce its kept windows + per-window labels.

    Returns ``(seq, event, transient, state, order)`` for the valid (labelled) windows,
    or ``None`` when the instance yields no usable window. Shared by the in-RAM and the
    memmap builders so both apply identical scaling, windowing and labelling.
    """
    inst = load_and_resample(meta.path, dcfg)
    sig = inst.df[SIGNAL_COLUMNS].copy()
    if normalise_signals:
        # Within-instance robust scaling (AGENT.md §2): centre/scale each channel using
        # only this instance's NORMAL (class 0) timesteps, so the model sees relative
        # deviations from this well's own baseline — not absolute pressures, which don't
        # generalise across wells. Robust (median/IQR) resists fault-driven outliers.
        ref = sig[(inst.df["class"] == 0).fillna(False).to_numpy(dtype=bool)]
        ref = ref if len(ref) else sig  # fall back to all rows if no NORMAL segment exists
        med = ref.median()
        # Floor the IQR (not just the exact-zero case): a channel that is *nearly* flat
        # during NORMAL has a tiny IQR, and dividing by it would blow the deep model's
        # inputs up to ~1e10 and produce NaN losses. Floor relative to each channel's own
        # scale so the z-score stays well-conditioned.
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        floor = med.abs() * 1e-3 + 1e-6
        iqr = iqr.where(iqr > floor, floor)
        # Clip robust z-scores to ±10 IQRs: anything past that is an extreme outlier, and
        # clipping keeps the TCN/BatchNorm numerically stable without losing the signal.
        sig = ((sig - med) / iqr).clip(-10.0, 10.0)

    bounds = window_bounds(len(sig), mcfg.window, dcfg.resample_rate_s)
    if not bounds:
        return None
    # Label from the raw instance frame; identical windowing logic as the GBT path so
    # the two datasets stay label-consistent.
    labels = label_windows(inst.df["class"], inst.df["state"], bounds)
    keep = labels.valid.astype(bool)
    if not keep.any():
        return None
    seq = sequence_windows(sig, bounds, SIGNAL_COLUMNS)[keep]
    starts = np.array([s for s, _ in bounds], dtype=np.int64)[keep]
    return seq, labels.event[keep], labels.transient[keep], labels.state[keep], starts


def build_sequence_dataset(
    metas: Sequence[InstanceMeta],
    dcfg: DataConfig,
    fcfg: FeaturesConfig,
    mcfg: ModelConfig,
    normalise_signals: bool = True,
    progress: bool = True,
) -> SequenceDataset:
    """Build raw sequence windows in RAM. Use :func:`build_sequence_memmap` instead when
    the full dataset would not fit in memory."""
    Xs, ev, tr, stt, wid, iid, ordr = [], [], [], [], [], [], []
    for meta in tqdm(metas, desc="seq-windowing", disable=not progress):
        res = _prepare_instance_windows(meta, dcfg, mcfg, normalise_signals)
        if res is None:
            continue
        seq, e, t, s, starts = res
        Xs.append(seq)
        ev.append(e)
        tr.append(t)
        stt.append(s)
        ordr.append(starts)
        n = len(seq)
        wid.append(np.full(n, meta.well_id, dtype=object))
        iid.append(np.full(n, meta.instance_id, dtype=object))

    if not Xs:
        raise ValueError("no valid sequence windows produced")
    return SequenceDataset(
        # float16 halves the RAM footprint of the (n, 300, 27) tensor — essential for
        # full-data runs — and is cast back to float32 per batch at train/eval time.
        X=np.concatenate(Xs).astype(np.float16),
        event=np.concatenate(ev),
        transient=np.concatenate(tr),
        state=np.concatenate(stt),
        well_id=np.concatenate(wid),
        instance_id=np.concatenate(iid),
        order=np.concatenate(ordr),
    )


def build_sequence_memmap(
    metas: Sequence[InstanceMeta],
    dcfg: DataConfig,
    fcfg: FeaturesConfig,
    mcfg: ModelConfig,
    cache_path: str | Path,
    normalise_signals: bool = True,
    progress: bool = True,
) -> SequenceDataset:
    """Build the sequence windows on disk and return a memmap-backed dataset.

    Unlike :func:`build_sequence_dataset`, this never holds all windows in RAM: each
    instance's windows are streamed (appended) to a flat float16 file on disk, so peak
    memory is one instance's windows plus the small label arrays. The returned
    ``SequenceDataset.X`` is a read-only ``np.memmap``, so training/eval page windows in
    on demand (OS-cached, reclaimable) — this is what unblocks full-data deep training.
    """
    win_len = max(1, int(round(mcfg.window.window_s / dcfg.resample_rate_s)))
    n_channels = len(SIGNAL_COLUMNS)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ev, tr, stt, wid, iid, ordr = [], [], [], [], [], []
    total = 0
    with open(cache_path, "wb") as fp:
        for meta in tqdm(metas, desc="seq-memmap", disable=not progress):
            res = _prepare_instance_windows(meta, dcfg, mcfg, normalise_signals)
            if res is None:
                continue
            seq, e, t, s, starts = res
            if seq.shape[1] != win_len:
                # All windows must share win_len to live in a fixed-shape memmap. With
                # min_window_s == window_s this never triggers; skip defensively if it does.
                continue
            # Append this instance's windows as raw float16 bytes (C-contiguous so the
            # on-disk layout matches the (total, win_len, C) memmap we open afterwards).
            fp.write(np.ascontiguousarray(seq, dtype=np.float16).tobytes())
            total += len(seq)
            ev.append(e)
            tr.append(t)
            stt.append(s)
            ordr.append(starts)
            n = len(seq)
            wid.append(np.full(n, meta.well_id, dtype=object))
            iid.append(np.full(n, meta.instance_id, dtype=object))

    if total == 0:
        cache_path.unlink(missing_ok=True)
        raise ValueError("no valid sequence windows produced")

    X = np.memmap(cache_path, dtype=np.float16, mode="r", shape=(total, win_len, n_channels))
    return SequenceDataset(
        X=X,
        event=np.concatenate(ev),
        transient=np.concatenate(tr),
        state=np.concatenate(stt),
        well_id=np.concatenate(wid),
        instance_id=np.concatenate(iid),
        order=np.concatenate(ordr),
    )
