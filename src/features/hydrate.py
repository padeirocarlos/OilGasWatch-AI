"""Hydrate-formation margin (AGENT.md §5.2, classes 8 & 9).

Distance from the operating point ``(T, P)`` to a parameterised hydrate-equilibrium
curve ``P_eq(T) = exp(a + b/(T + c))``. A small/negative margin means the well is in
the hydrate-stable region. Curve coefficients come from config, never hardcoded
(AGENT.md §8). # TODO: validate coefficients against PVT data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame


def _p_eq(temp_c: pd.Series, cfg: FeaturesConfig) -> pd.Series:
    # Hydrate-equilibrium pressure at temperature T: above this P (for a given T) gas
    # hydrates (ice-like plugs) are thermodynamically stable. The Antoine-like form
    # exp(a + b/(T+c)) approximates that boundary; a/b/c come from config (AGENT.md §8).
    hc = cfg.hydrate_curve
    # Clip exponent to avoid overflow for extreme/garbage temperatures.
    expo = (hc.a + hc.b / (temp_c + hc.c)).clip(-50, 50)
    return np.exp(expo)


def hydrate_margin(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    out = pd.DataFrame(index=df.index)

    # Production line (class 8): operating point is (T-TPT, P-TPT) at the flow tree.
    # margin = actual P − equilibrium P: small or negative ⇒ inside the hydrate-stable
    # region ⇒ plug-formation risk. The ratio is a scale-free version of the same idea.
    p_eq_prod = _p_eq(df["T-TPT"], cfg)
    out["hydrate_margin_prod"] = df["P-TPT"] - p_eq_prod
    out["hydrate_ratio_prod"] = df["P-TPT"] / p_eq_prod.clip(lower=cfg.differential_eps)

    # Service line (class 9): downstream of the service pump; T-JUS-CKP is the nearest
    # available temperature, P-JUS-BS the line pressure for the same margin computation.
    p_eq_svc = _p_eq(df["T-JUS-CKP"], cfg)
    out["hydrate_margin_svc"] = df["P-JUS-BS"] - p_eq_svc

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
