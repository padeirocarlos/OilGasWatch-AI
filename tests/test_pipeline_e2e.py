"""End-to-end baseline run on real data, well-disjoint and reproducible (AGENT.md §7)."""

from __future__ import annotations

from config import DataConfig, FeaturesConfig, ModelConfig
from pipeline import run_baseline
from tests.conftest import DATASET, requires_data


def _small_cfgs(tmp_path):
    # Shrunken config so the full real-data pipeline runs in test time: wider windows,
    # fewer trees. The point is exercising the END-TO-END wiring, not chasing accuracy.
    dcfg = DataConfig(dataset_dir=str(DATASET), runs_dir=str(tmp_path / "runs"))
    fcfg = FeaturesConfig()
    mcfg = ModelConfig()
    mcfg.window.window_s = 300.0
    mcfg.window.stride_s = 150.0
    mcfg.baseline.n_estimators = 60  # keep the test fast
    return dcfg, fcfg, mcfg


@requires_data
def test_baseline_runs_and_logs(tmp_path):
    dcfg, fcfg, mcfg = _small_cfgs(tmp_path)
    run = run_baseline(
        dcfg, fcfg, mcfg, classes=[0, 1, 2], test_frac=0.34, limit_per_class=4, progress=False
    )
    # macro_f1 must be a valid score: an out-of-range value signals a broken metric or
    # a label/prediction mix-up upstream, even if no exception was raised.
    assert 0.0 <= run.macro_f1 <= 1.0
    # Every run must be logged to runs/ (§7): an unlogged run is not reproducible evidence.
    assert (tmp_path / "runs" / run.run_id / "metrics.json").exists()
    # The split actually held out wells: n_test_wells>=1 proves the well-disjoint test set
    # was non-empty, so the reported score is a genuine held-out measurement.
    assert run.metrics["n_test_wells"] >= 1


@requires_data
def test_baseline_is_reproducible(tmp_path):
    dcfg, fcfg, mcfg = _small_cfgs(tmp_path)
    r1 = run_baseline(
        dcfg, fcfg, mcfg, classes=[0, 1], test_frac=0.34, limit_per_class=4, progress=False
    )
    r2 = run_baseline(
        dcfg, fcfg, mcfg, classes=[0, 1], test_frac=0.34, limit_per_class=4, progress=False
    )
    # End-to-end determinism (§2): same data + config + seed must give the EXACT same
    # macro-F1. Any drift here means a hidden source of randomness (unseeded split, sampler,
    # or estimator) that would make published baseline numbers unreproducible.
    assert r1.macro_f1 == r2.macro_f1
