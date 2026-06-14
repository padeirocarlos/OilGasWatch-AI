"""Ingest: schema validation, resampling, quality channels (AGENT.md §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DataConfig
from ingest.instance import parse_meta
from ingest.io import SchemaError, load_instance
from ingest.resample import resample
from schema import SIGNAL_COLUMNS, Source


def test_parse_meta_sources():
    # Source (real / simulated / hand-drawn) must be inferred from the filename so eval
    # can later exclude or stratify by provenance (§3) instead of trusting raw labels.
    assert parse_meta("dataset/1/WELL-00001_20140124083303.parquet").source is Source.REAL
    assert parse_meta("dataset/1/WELL-00001_20140124083303.parquet").well_id == "WELL-00001"
    assert parse_meta("dataset/1/SIMULATED_00014.parquet").source is Source.SIMULATED
    assert parse_meta("dataset/1/DRAWN_00001.parquet").source is Source.HAND_DRAWN
    # Synthetic instances get a namespaced well id (SIM::) so they can NEVER be mistaken
    # for a real well — otherwise the well-disjoint splitter could leak across sources.
    assert parse_meta("dataset/1/SIMULATED_00014.parquet").well_id.startswith("SIM::")


def test_resample_is_gap_free_and_adds_quality(synthetic_frame):
    out = resample(synthetic_frame, DataConfig())
    # Downstream feature math assumes a dense grid: any leftover NaN would silently
    # poison differentials/spectra, so resampling must leave zero gaps in signals.
    assert int(out.df[SIGNAL_COLUMNS].isna().sum().sum()) == 0
    # Quality is a feature, not a nuisance (§2): imputation/freezing must be RECORDED as
    # explicit indicator channels rather than silently filled, so the model can react.
    assert any(c.endswith("__is_missing") for c in out.quality_columns)
    assert any(c.endswith("__is_frozen") for c in out.quality_columns)


def test_resample_flags_long_gap():
    # A gap longer than max_gap_s must NOT be forward-filled and forgotten: beyond the
    # threshold the data is genuinely missing and is_missing must mark it as such.
    # Build a tiny instance with a 100s hole in a continuous channel.
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="s", name="timestamp")
    df = pd.DataFrame(index=idx)
    from schema import ALL_COLUMNS

    for c in SIGNAL_COLUMNS:
        df[c] = 1.0
    df.loc[df.index[50:150], "P-PDG"] = np.nan  # 100s gap > max_gap_s=30
    df["class"] = pd.array(np.zeros(n, dtype="int64"), dtype="Int64")
    df["state"] = pd.array(np.zeros(n, dtype="int64"), dtype="Int64")
    from ingest.instance import InstanceFrame, InstanceMeta

    frame = InstanceFrame(
        df[ALL_COLUMNS], InstanceMeta("WELL-1_0.parquet", "WELL-1_0", "WELL-1", Source.REAL, 0)
    )
    out = resample(frame, DataConfig(max_gap_s=30.0))
    # The 100s hole exceeds the 30s max_gap, so it stays flagged rather than imputed.
    assert out.df["P-PDG__is_missing"].sum() > 0


def test_load_instance_rejects_unknown_columns(tmp_path):
    # The 29+2 schema is authoritative (§8): loading must FAIL LOUDLY on an unknown
    # column rather than ingest a mislabelled/corrupt instance and corrupt the model.
    bad = tmp_path / "WELL-1_0.parquet"
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2020", periods=3, freq="s"), "BOGUS": [1, 2, 3]}
    ).set_index("timestamp")
    df.to_parquet(bad)
    with pytest.raises(SchemaError):
        load_instance(bad)
