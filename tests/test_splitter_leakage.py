"""Well-disjoint guarantee — the leakage test must stay green (AGENT.md §2, §7)."""

from __future__ import annotations

import pytest

from dataset.splitter import LeakageError, Split, WellDisjointSplitter
from ingest.instance import InstanceMeta
from schema import Source


def _metas(n_wells=10, per_well=3, cls=1):
    out = []
    for w in range(n_wells):
        for i in range(per_well):
            wid = f"WELL-{w:05d}"
            out.append(InstanceMeta(f"d/{wid}_{i}.parquet", f"{wid}_{i}", wid, Source.REAL, cls))
    return out


def test_train_test_split_is_well_disjoint():
    # The core invariant of the whole project (§2): a single well appearing in both train
    # and test leaks per-well idiosyncrasies across the boundary, silently inflating metrics
    # and destroying cross-well generalisation. This is enforced in code, asserted here.
    metas = _metas()
    split = WellDisjointSplitter(seed=42).train_test_split(metas, test_frac=0.3)
    train_wells = {m.well_id for m in split.train}
    test_wells = {m.well_id for m in split.test}
    assert train_wells.isdisjoint(test_wells)
    split.assert_disjoint()  # must not raise


def test_kfold_each_fold_well_disjoint():
    # The disjointness guarantee must hold for EVERY cross-validation fold, not just a
    # single split — one leaky fold is enough to make the averaged CV score a lie.
    metas = _metas(n_wells=12)
    for split in WellDisjointSplitter(seed=1).k_folds(metas, n_folds=4):
        assert {m.well_id for m in split.train}.isdisjoint({m.well_id for m in split.test})


def test_assert_disjoint_detects_leak():
    # The guard itself must work: feed it a deliberately leaky split (same wells on both
    # sides) and confirm it RAISES. A silent guard would let real leakage slip through.
    metas = _metas(n_wells=2, per_well=1)
    leaky = Split(train=metas, test=metas)  # same wells on both sides
    with pytest.raises(LeakageError):
        leaky.assert_disjoint()


def test_stratified_split_is_disjoint_and_covers_rare_class():
    # A common class with many wells plus a RARE class with only 4 wells. A plain random
    # split can drop all rare-class wells into train (zero test support); the stratified
    # split must give the rare class test wells *and* stay well-disjoint.
    common = _metas(n_wells=40, per_well=2, cls=0)
    rare = []
    for w in range(4):
        wid = f"WELL-9{w:04d}"
        rare.append(InstanceMeta(f"d/{wid}_0.parquet", f"{wid}_0", wid, Source.REAL, 2))
    split = WellDisjointSplitter(seed=7).stratified_train_test_split(common + rare, test_frac=0.25)
    split.assert_disjoint()  # disjointness still guaranteed
    test_dirs = {m.event_class_dir for m in split.test}
    train_dirs = {m.event_class_dir for m in split.train}
    assert 2 in test_dirs  # the rare class secured a test foothold
    assert 2 in train_dirs and 0 in test_dirs  # both classes appear on both sides
