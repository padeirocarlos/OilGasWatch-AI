"""The disk-backed memmap builder must match the in-RAM builder exactly (full-data path)."""

from __future__ import annotations

import numpy as np

from config import DataConfig, FeaturesConfig, ModelConfig
from dataset.build import build_sequence_dataset, build_sequence_memmap
from ingest import discover_instances
from tests.conftest import DATASET, requires_data


def _cfgs():
    dcfg = DataConfig(dataset_dir=str(DATASET))
    mcfg = ModelConfig()
    mcfg.window.window_s = 300.0
    mcfg.window.stride_s = 300.0  # non-overlap keeps the test quick
    mcfg.window.min_window_s = 300.0
    return dcfg, FeaturesConfig(), mcfg


@requires_data
def test_memmap_matches_in_ram(tmp_path):
    dcfg, fcfg, mcfg = _cfgs()
    metas = discover_instances(dcfg, classes=[1])[:3]

    ram = build_sequence_dataset(metas, dcfg, fcfg, mcfg, progress=False)
    mm = build_sequence_memmap(metas, dcfg, fcfg, mcfg, tmp_path / "x.f16", progress=False)

    # Same number and shape of windows, stored on disk as a memmap.
    assert isinstance(mm.X, np.memmap)
    assert mm.X.shape == ram.X.shape
    # Identical window contents and labels (same scaling/windowing/labelling path).
    np.testing.assert_array_equal(np.asarray(mm.X), np.asarray(ram.X))
    np.testing.assert_array_equal(mm.event, ram.event)
    np.testing.assert_array_equal(mm.transient, ram.transient)
    np.testing.assert_array_equal(mm.order, ram.order)
    np.testing.assert_array_equal(mm.instance_id, ram.instance_id)


@requires_data
def test_memmap_keeps_ram_low(tmp_path):
    # The memmap file holds the windows on disk; the returned X must not be a plain
    # in-RAM ndarray (which is the whole point — full data without OOM).
    dcfg, fcfg, mcfg = _cfgs()
    metas = discover_instances(dcfg, classes=[1])[:2]
    mm = build_sequence_memmap(metas, dcfg, fcfg, mcfg, tmp_path / "x.f16", progress=False)
    assert (tmp_path / "x.f16").exists()
    assert mm.X.dtype == np.float16
