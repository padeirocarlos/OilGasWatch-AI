"""Well-disjoint splitting (AGENT.md §2, §5.3).

Train and test wells must never overlap — a split that leaks a well is a bug, not a
config choice. The splitter groups by well id and guarantees zero overlap; the
guarantee is asserted here and re-checked in tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ingest.instance import InstanceMeta


# Subclass AssertionError so this reads as a violated invariant, not a routine
# exception: a leaking split is a *bug* (AGENT.md §2), and tests can assert against it.
class LeakageError(AssertionError):
    """Raised when a split shares a well id across train and test."""


@dataclass(frozen=True)
class Split:
    train: list[InstanceMeta]
    test: list[InstanceMeta]

    def assert_disjoint(self) -> None:
        # Leakage = the same well (and thus correlated sensor behaviour) appearing in
        # both train and test, which inflates metrics by letting the model memorise a
        # well rather than learn the fault. Comparing well-id *sets* catches any overlap
        # regardless of how many instances each well contributed.
        tr = {m.well_id for m in self.train}
        te = {m.well_id for m in self.test}
        overlap = tr & te
        if overlap:
            raise LeakageError(f"wells leak across split: {sorted(overlap)}")


class WellDisjointSplitter:
    """Group-by-well splitter producing well-disjoint train/test partitions."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _well_groups(self, metas: Sequence[InstanceMeta]) -> dict[str, list[InstanceMeta]]:
        # Bucket instances by well id. Splitting then operates on whole wells, which is
        # what structurally guarantees disjointness — a well's instances never split apart.
        groups: dict[str, list[InstanceMeta]] = {}
        for m in metas:
            groups.setdefault(m.well_id, []).append(m)
        return groups

    def train_test_split(self, metas: Sequence[InstanceMeta], test_frac: float = 0.25) -> Split:
        """Hold out whole wells for the test set."""
        groups = self._well_groups(metas)
        # Sort before shuffling so the seed alone determines the partition (determinism,
        # AGENT.md §2): dict iteration order must not leak into the split.
        wells = sorted(groups)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(wells)
        # Hold out at least one whole well even when test_frac rounds down to zero.
        n_test = max(1, int(round(len(wells) * test_frac)))
        test_wells = set(wells[:n_test])

        # Partition by well, then flatten back to instances. Because each well lands
        # entirely on one side, the resulting sides cannot share a well.
        train = [m for w in wells if w not in test_wells for m in groups[w]]
        test = [m for w in test_wells for m in groups[w]]
        split = Split(train=train, test=test)
        # Belt-and-braces: the construction is disjoint by design, but we assert it in
        # code so a future refactor can never silently reintroduce leakage.
        split.assert_disjoint()
        return split

    def stratified_train_test_split(
        self, metas: Sequence[InstanceMeta], test_frac: float = 0.25
    ) -> Split:
        """Well-disjoint split that also stratifies by event class.

        A plain random well split can place every well of a rare class (e.g. class 2,
        with only a handful of wells) entirely in train, leaving that class with zero
        test support. Here we assign wells to test *per class*, processing the rarest
        classes first so their scarce wells secure a test foothold before a common
        class can claim them. Wells are still assigned globally (each to exactly one
        side), so disjointness is preserved; stratification is best-effort under that
        hard constraint. A well that spans several classes counts toward each.
        """
        groups = self._well_groups(metas)
        class_wells: dict[int, set[str]] = {}
        for m in metas:
            class_wells.setdefault(m.event_class_dir, set()).add(m.well_id)

        rng = np.random.default_rng(self.seed)
        test_wells: set[str] = set()
        assigned: set[str] = set()
        # Rarest classes first: they have the fewest wells, so reserve their test quota
        # before common classes consume the shared well pool.
        for cls in sorted(class_wells, key=lambda c: len(class_wells[c])):
            wells = sorted(class_wells[cls])  # deterministic order before the seeded shuffle
            rng.shuffle(wells)
            n = len(wells)
            # Keep at least one well on each side when the class has >= 2 wells; a
            # single-well class goes to train so the model can at least learn it.
            target_test = min(max(1, round(n * test_frac)), n - 1) if n >= 2 else 0
            cur_test = sum(1 for w in wells if w in test_wells)
            for w in wells:
                if w in assigned:
                    continue
                if cur_test < target_test:
                    test_wells.add(w)
                    cur_test += 1
                assigned.add(w)

        train = [m for w in groups if w not in test_wells for m in groups[w]]
        test = [m for w in test_wells for m in groups[w]]
        split = Split(train=train, test=test)
        split.assert_disjoint()
        return split

    def k_folds(self, metas: Sequence[InstanceMeta], n_folds: int = 5) -> Iterator[Split]:
        """Yield ``n_folds`` well-disjoint folds (each well appears in exactly one test fold)."""
        groups = self._well_groups(metas)
        wells = sorted(groups)  # deterministic order before the seeded shuffle
        rng = np.random.default_rng(self.seed)
        rng.shuffle(wells)
        # Round-robin each well into one fold; assigning at the *well* level (not the
        # instance level) is what keeps every fold well-disjoint.
        fold_assign = {w: i % n_folds for i, w in enumerate(wells)}
        for k in range(n_folds):
            # Fold k is the test set; all other wells train. One well is in exactly one
            # test fold, so across folds every well is tested without ever leaking.
            test = [m for w in wells if fold_assign[w] == k for m in groups[w]]
            train = [m for w in wells if fold_assign[w] != k for m in groups[w]]
            # Skip degenerate folds (e.g. n_folds > n_wells) rather than yield an empty side.
            if not test or not train:
                continue
            split = Split(train=train, test=test)
            split.assert_disjoint()  # re-check the invariant per fold
            yield split
