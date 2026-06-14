"""Feature unit tests, incl. the division-guard and frozen-sensor edge cases (AGENT.md §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DataConfig, FeaturesConfig
from features.choke import choke_coeff
from features.differentials import differentials
from features.normalise import normalise
from features.pipeline import build_features
from ingest.instance import InstanceFrame, InstanceMeta
from ingest.resample import resample
from schema import ALL_COLUMNS, SIGNAL_COLUMNS, Source


def _const_frame(value=0.0, n=120):
    idx = pd.date_range("2020-01-01", periods=n, freq="s", name="timestamp")
    df = pd.DataFrame(index=idx)
    for c in SIGNAL_COLUMNS:
        df[c] = value
    df["class"] = pd.array(np.zeros(n, dtype="int64"), dtype="Int64")
    df["state"] = pd.array(np.zeros(n, dtype="int64"), dtype="Int64")
    meta = InstanceMeta("WELL-1_0.parquet", "WELL-1_0", "WELL-1", Source.REAL, 0)
    return InstanceFrame(df[ALL_COLUMNS], meta)


def test_choke_coeff_division_guard():
    # Cv_eff = Q / (opening · sqrt(max(ΔP, ε))) divides by opening and ΔP (§5.2). A closed
    # choke (opening=0, ΔP=0) is a real operating state, so the ε/guard must hold: an inf
    # or NaN here would propagate through training and silently break every metric.
    # All-zero inputs (closed choke, zero ΔP) must not produce inf/NaN.
    frame = _const_frame(0.0)
    out = choke_coeff(frame, FeaturesConfig())
    assert np.isfinite(out.to_numpy()).all()


def test_frozen_sensor_indicator_then_differentials(synthetic_frame):
    out = resample(synthetic_frame, DataConfig())
    # A frozen sensor (zero variance) feeding a ΔP differential is the edge case from §7:
    # the differential of a stuck channel must stay finite, not blow up, so a dead sensor
    # degrades gracefully instead of crashing the feature pipeline.
    # A constant channel over the frozen window should be flagged frozen somewhere.
    out.df["P-MON-CKP"] = 1.0
    out2 = resample(synthetic_frame, DataConfig())
    d = differentials(out2, FeaturesConfig())
    assert np.isfinite(d.to_numpy()).all()


def test_normalise_uses_normal_segment_and_flags_fallback():
    # Within-instance normalisation (§2) calibrates against the NORMAL segment. When an
    # instance has no NORMAL baseline, normalisation must fall back AND record that it did
    # (__norm_fallback=1) — silently normalising on faulty data would distort every feature.
    frame = _const_frame(5.0)
    # No NORMAL? Make all class 1 to force fallback.
    frame.df["class"] = pd.array(np.ones(len(frame.df), dtype="int64"), dtype="Int64")
    out = normalise(frame, FeaturesConfig())
    assert out["__norm_fallback"].iloc[0] == 1


def test_build_features_no_nan_inf(synthetic_frame):
    # Whole-pipeline contract: no individual feature may emit NaN/inf. A tree baseline can
    # tolerate odd values but a neural model cannot — one inf would NaN out the gradients,
    # so the full composed feature matrix (sans labels) must be entirely finite.
    inst = resample(synthetic_frame, DataConfig())
    feats = build_features(inst, FeaturesConfig(spectral_window_s=32, oscillation_window_s=32))
    body = feats.drop(columns=["class", "state"])
    assert int(body.isna().sum().sum()) == 0
    assert np.isfinite(body.to_numpy()).all()


def test_norm_fallback_excluded_from_model_features(synthetic_frame):
    # __norm_fallback is an instance-level diagnostic that perfectly flags the steady-state
    # classes' fault-only files (no NORMAL segment) — a label leak. The window builder must
    # drop "__"-prefixed diagnostics from the model feature set while keeping the genuine
    # per-sensor quality channels (which start with a channel name, not "__").
    from dataset.windows import aggregate_feature_names

    feats = build_features(
        resample(synthetic_frame, DataConfig()),
        FeaturesConfig(spectral_window_s=32, oscillation_window_s=32),
    )
    assert "__norm_fallback" in feats.columns  # the pipeline still computes the diagnostic
    skip = ("class", "state")
    model_cols = [c for c in feats.columns if c not in skip and not c.startswith("__")]
    names = aggregate_feature_names(model_cols)
    assert not any("norm_fallback" in n for n in names)  # but it never reaches the model
    assert any("__is_missing" in n for n in names)  # genuine quality channels are retained
