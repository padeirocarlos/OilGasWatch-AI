"""Oscillation amplitude and period (AGENT.md §5.2, classes 3 & 4).

Severe slugging and flow instability are steady-state *oscillatory* regimes with no
clean onset; they are characterised by the amplitude and period of the detrended
signal. We measure rolling peak-to-peak amplitude and the dominant period via
zero-crossing spacing of the detrended series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame

_OSC_CHANNELS = ["P-PDG", "P-TPT", "PT-P", "QGL", "P-MON-CKP"]


def _zero_cross_period(seg: np.ndarray) -> float:
    # Period estimate via mean-crossings: a periodic signal crosses its mean twice per
    # cycle, so the average sample-spacing between crossings is ~half a period.
    centred = seg - seg.mean()
    signs = np.sign(centred)
    signs[signs == 0] = 1  # treat exact-zero samples as positive to avoid phantom crossings
    crossings = np.flatnonzero(np.diff(signs) != 0)
    if len(crossings) < 2:  # fewer than 2 crossings ⇒ no resolvable oscillation
        return 0.0
    # Mean spacing between zero crossings ≈ half-period; double it.
    return float(2.0 * np.mean(np.diff(crossings)))


def oscillation(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    win = max(4, int(round(cfg.oscillation_window_s)))
    out = pd.DataFrame(index=df.index)

    for chan in _OSC_CHANNELS:
        if chan not in df.columns:
            continue
        s = df[chan]
        roll = s.rolling(win, min_periods=win // 2)
        # Detrend by subtracting the rolling mean so slow drift (changing operating point)
        # is removed and only the oscillation about the local baseline remains — that
        # oscillation's size and rhythm are what distinguish slugging/flow-instability.
        detr = s - roll.mean()
        out[f"{chan}__osc_amp"] = (
            detr.rolling(win, min_periods=win // 2).max()
            - detr.rolling(win, min_periods=win // 2).min()
        )
        out[f"{chan}__osc_std"] = detr.rolling(win, min_periods=win // 2).std()
        # Period: zero-crossing estimate on the detrended signal, on a hop grid.
        x = s.to_numpy(dtype=float)
        n = len(x)
        period = np.zeros(n)
        hop = max(1, win // 4)
        for end in range(win, n + 1, hop):
            p = _zero_cross_period(x[end - win : end])
            period[end - 1 : min(end + hop, n)] = p
        out[f"{chan}__osc_period"] = period

    return out.fillna(0.0)
