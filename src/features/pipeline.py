"""Compose the feature functions into one engineered DataFrame (AGENT.md §5.2).

Each function is pure ``f(InstanceFrame, FeaturesConfig) -> DataFrame`` indexed by the
instance timestamp, so they compose by column-concatenation and test in isolation.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from config import FeaturesConfig
from features.choke import choke_coeff
from features.differentials import differentials
from features.hydrate import hydrate_margin
from features.normalise import normalise
from features.oscillation import oscillation
from features.spectral import spectral
from features.valve import valve_transitions
from ingest.instance import InstanceFrame

FeatureFn = Callable[[InstanceFrame, FeaturesConfig], pd.DataFrame]

#: Registry of feature builders, in a stable order.
FEATURE_FUNCTIONS: dict[str, FeatureFn] = {
    "differentials": differentials,
    "choke": choke_coeff,
    "hydrate": hydrate_margin,
    "spectral": spectral,
    "oscillation": oscillation,
    "valve": valve_transitions,
    "normalise": normalise,
}


def build_features(
    frame: InstanceFrame,
    cfg: FeaturesConfig,
    include: list[str] | None = None,
) -> pd.DataFrame:
    """Run the selected feature functions and concatenate them column-wise.

    Also carries the instance's quality channels through, so windowing can use them.
    The ``class`` and ``state`` labels are attached for downstream labelling.
    """
    names = include or list(FEATURE_FUNCTIONS)
    # Each builder returns a DataFrame on the SAME timestamp index, so composing the
    # whole feature set is just a column-wise concat of these independent pure functions.
    parts = [FEATURE_FUNCTIONS[name](frame, cfg) for name in names]

    # Quality channels (is_missing / is_frozen from ingest) ride along as ordinary
    # feature columns: the model learns to distrust a frozen/imputed sensor (§2).
    if frame.quality_columns:
        parts.append(frame.df[frame.quality_columns])

    # Labels travel with the features so downstream windowing can attach class/state.
    labels = frame.df[["class", "state"]].copy()
    feats = pd.concat([*parts, labels], axis=1)
    feats.index.name = "timestamp"
    return feats
