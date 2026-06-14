"""Golden-file test on one known real instance (AGENT.md §7).

Rather than committing a large artifact, we pin the deterministic *shape* of the
ingest+feature output for a fixed instance and assert byte-stable reproducibility
across two runs. If the schema or feature set changes intentionally, update EXPECTED.
"""

from __future__ import annotations

import numpy as np

from config import DataConfig, FeaturesConfig
from features.pipeline import build_features
from ingest import load_and_resample
from schema import SIGNAL_COLUMNS
from tests.conftest import DATASET, requires_data

# Pinned feature count from a known real run: this is the "golden" value. A change to the
# feature set must be a DELIBERATE edit of this number, never an accidental schema drift.
# 27 signals + 27 is_missing + 18 is_frozen + 27 norm + ... — pinned below from a real run.
EXPECTED_N_FEATURES = 140


@requires_data
def test_golden_instance_is_deterministic():
    # sorted()[0] pins ONE fixed real instance so the golden assertions are stable.
    path = sorted(DATASET.glob("1/WELL-*.parquet"))[0]
    dcfg, fcfg = DataConfig(dataset_dir=str(DATASET)), FeaturesConfig()

    # Run the full ingest+feature pipeline twice on identical input + config.
    a = build_features(load_and_resample(path, dcfg), fcfg)
    b = build_features(load_and_resample(path, dcfg), fcfg)

    body_a = a.drop(columns=["class", "state"])
    body_b = b.drop(columns=["class", "state"])
    assert body_a.shape[1] == EXPECTED_N_FEATURES  # feature set unchanged (golden shape)
    assert int(body_a.isna().sum().sum()) == 0  # no NaN escaped the pipeline
    assert np.isfinite(body_a.to_numpy()).all()  # and no inf either
    # Same input + config -> identical features (determinism, AGENT.md §2).
    np.testing.assert_array_equal(body_a.to_numpy(), body_b.to_numpy())
    # Labels (nullable ints) compare equal including NA positions.
    assert a[["class", "state"]].equals(b[["class", "state"]])


@requires_data
def test_golden_resample_grid_is_regular():
    # Spectral/oscillation features assume a fixed sample rate: an irregular time grid would
    # make every frequency-domain feature meaningless. Assert the post-resample index is
    # evenly spaced and gap-free on a real instance, not just on the synthetic fixture.
    path = sorted(DATASET.glob("1/WELL-*.parquet"))[0]
    inst = load_and_resample(path, DataConfig(dataset_dir=str(DATASET)))
    deltas = np.diff(inst.df.index.view("int64")) / 1e9
    assert np.allclose(deltas, deltas[0])  # uniform spacing
    assert int(inst.df[SIGNAL_COLUMNS].isna().sum().sum()) == 0
