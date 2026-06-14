"""Layer 3 — windowing, well-disjoint splitting, samplers (AGENT.md §5.3).

This package turns ingested+featurised instances into model-ready windows. The three
concerns it owns map onto the three submodules: ``splitter`` keeps train/test wells
disjoint (no leakage), ``windows`` slices each instance into labelled windows, and
``sampler`` corrects the heavy class imbalance. ``build`` orchestrates all three.
"""

# Re-export the public surface so callers can ``from dataset import ...`` without
# knowing which submodule a symbol lives in.
from dataset.build import (
    SequenceDataset,
    WindowDataset,
    build_sequence_dataset,
    build_sequence_memmap,
    build_window_dataset,
)
from dataset.sampler import effective_number_weights, sample_weights
from dataset.splitter import LeakageError, Split, WellDisjointSplitter
from dataset.windows import (
    AGG_FUNCS,
    WindowLabels,
    aggregate_feature_names,
    aggregate_features,
    label_windows,
    sequence_windows,
    window_bounds,
)

__all__ = [
    "WindowDataset",
    "SequenceDataset",
    "build_window_dataset",
    "build_sequence_dataset",
    "build_sequence_memmap",
    "WellDisjointSplitter",
    "Split",
    "LeakageError",
    "sample_weights",
    "effective_number_weights",
    "AGG_FUNCS",
    "WindowLabels",
    "window_bounds",
    "aggregate_features",
    "aggregate_feature_names",
    "label_windows",
    "sequence_windows",
]
