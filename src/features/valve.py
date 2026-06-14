"""Valve transition features (AGENT.md §5.2, class 2).

Spurious DHSV closure and other valve events show as state changes. We emit, per
valve, a change flag (Δstate ≠ 0), the signed transition, and ``time_since_change``
(seconds since the last transition). Valve states are ordinal in {0, 0.5, 1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame
from schema import VALVE_COLUMNS


def valve_transitions(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    out = pd.DataFrame(index=df.index)

    for col in VALVE_COLUMNS:
        s = df[col]
        # Δstate: a spurious DHSV (downhole safety valve) closure shows as an abrupt
        # 1→0 step; the signed delta keeps closing(−) vs opening(+) distinguishable.
        delta = s.diff().fillna(0.0)
        changed = (delta != 0).astype("int8")
        out[f"{col}__delta"] = delta
        out[f"{col}__changed"] = changed

        # time_since_change lets the model relate a downstream transient to a recent
        # valve event (cause→effect timing) — e.g. a pressure swing right after a closure.
        # Seconds since the most recent change (resets to 0 on each transition).
        change_idx = np.flatnonzero(changed.to_numpy())
        tsc = np.arange(len(s), dtype=float)
        if len(change_idx):
            # For each position, subtract the index of the last change at or before it.
            last_change = np.zeros(len(s), dtype=float)
            ptr = 0
            cur = 0
            for i in range(len(s)):
                if ptr < len(change_idx) and change_idx[ptr] == i:
                    cur = i
                    ptr += 1
                last_change[i] = cur
            tsc = (np.arange(len(s)) - last_change) * 1.0
        out[f"{col}__time_since_change"] = tsc

    return out.fillna(0.0)
