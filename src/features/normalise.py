"""Within-instance robust normalisation (AGENT.md §2, §5.2).

Raw absolute pressures destroy cross-well generalisation, so every continuous
channel is robust-scaled ``(x − median) / IQR`` using statistics from the instance's
NORMAL segment (``class == 0``) only. If an instance has no NORMAL samples we fall
back to whole-instance stats and record it via the ``__norm_fallback`` flag rather
than silently changing behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame
from schema import CONTINUOUS_COLUMNS


def normalise(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    out = pd.DataFrame(index=df.index)

    # Reference stats come ONLY from this instance's NORMAL samples (class == 0): that
    # is the well's healthy baseline, so scaling against it puts every well on a common
    # "deviation from its own normal" axis — the key to cross-well generalisation (§2).
    cls = df["class"]
    normal_mask = (cls == 0).fillna(False).to_numpy(dtype=bool)
    # If the instance has no NORMAL window we must use whole-instance stats instead, and
    # we flag it (no silent change of behaviour — AGENT.md "quality is a feature").
    ref = df.loc[normal_mask, CONTINUOUS_COLUMNS] if normal_mask.any() else df[CONTINUOUS_COLUMNS]
    fallback = 0 if normal_mask.any() else 1

    # Median + IQR (not mean/std): robust to the very outliers a fault produces, so the
    # baseline is not skewed by the event we are trying to detect.
    med = ref.median()
    q1 = ref.quantile(0.25)
    q3 = ref.quantile(0.75)
    iqr = (q3 - q1).replace(0.0, np.nan)  # zero IQR (frozen channel) → handled below

    for col in CONTINUOUS_COLUMNS:
        scale = iqr[col] if not np.isnan(iqr[col]) else 1.0  # degenerate spread → no rescale
        out[f"{col}__norm"] = (df[col] - med[col]) / scale

    out["__norm_fallback"] = fallback
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
