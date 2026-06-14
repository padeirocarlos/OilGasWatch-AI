"""Shared fixtures: a synthetic instance and a real-data availability guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, FeaturesConfig, ModelConfig
from ingest.instance import InstanceFrame, InstanceMeta
from schema import ALL_COLUMNS, SIGNAL_COLUMNS, Source

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "dataset"


def has_real_data() -> bool:
    # The 3W parquet dataset is large and not committed; detect it by looking for
    # any real WELL-* instance so data-dependent tests can opt in.
    return DATASET.exists() and any(DATASET.glob("*/WELL-*.parquet"))


# Gate for tests that need the real 3W corpus: they SKIP (not fail) when it is
# absent, so CI without the dataset stays green while local runs still exercise them.
requires_data = pytest.mark.skipif(not has_real_data(), reason="3W dataset not present")


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(dataset_dir=str(DATASET))


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    # Small windows so synthetic fixtures exercise the spectral/oscillation paths.
    return FeaturesConfig(spectral_window_s=32.0, oscillation_window_s=32.0)


@pytest.fixture
def model_cfg() -> ModelConfig:
    cfg = ModelConfig()
    cfg.window.window_s = 64.0
    cfg.window.stride_s = 32.0
    cfg.window.min_window_s = 64.0
    return cfg


@pytest.fixture
def synthetic_frame() -> InstanceFrame:
    """A 400-sample instance: 200 NORMAL then 200 with a transient onset (class 1).

    A deterministic (seeded) stand-in for a real well so unit tests run without the
    dataset. It deliberately contains both a NORMAL segment (for normalisation
    baselines) and a regime shift carrying the +100 transient-onset label, so the
    feature/window/label paths can be exercised end-to-end offline.
    """
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="s", name="timestamp")
    # Fixed seed: the fixture must be byte-identical across runs (determinism, §2).
    rng = np.random.default_rng(0)
    df = pd.DataFrame(index=idx)
    for col in SIGNAL_COLUMNS:
        base = rng.normal(1e6 if col.startswith("P") else 50.0, 1.0, n)
        df[col] = base
    # A clear regime shift halfway through: shifts the dP_grad = P-PDG − P-TPT
    # differential so detectors have a real signal to find at the onset.
    df.loc[df.index[200:], "P-PDG"] += 5e5
    df.loc[df.index[200:], "P-TPT"] -= 2e5
    # Valve states are ordinal {0, 0.5, 1}; hold them open so they are valid, not noise.
    for col in [c for c in SIGNAL_COLUMNS if c.startswith("ESTADO-")]:
        df[col] = 1.0
    cls = np.zeros(n, dtype="int64")
    # 101 = class 1 + TRANSIENT_OFFSET(100): the onset window the transient head trains on.
    cls[200:300] = 101  # transient (onset) window
    cls[300:] = 1  # settled fault
    df["class"] = pd.array(cls, dtype="Int64")
    df["state"] = pd.array(np.zeros(n, dtype="int64"), dtype="Int64")
    df = df[ALL_COLUMNS]
    # A distinctive well id keeps this fixture from colliding with any real well.
    meta = InstanceMeta(
        "synthetic/WELL-09999_0.parquet", "WELL-09999_0", "WELL-09999", Source.REAL, 1
    )
    return InstanceFrame(df=df, meta=meta)
