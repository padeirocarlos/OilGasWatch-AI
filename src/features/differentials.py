"""Cross-component differentials — the core signal (AGENT.md §2, §5.2).

Differentials over absolutes: ΔP/ΔT between physically adjacent components and the
downhole-to-tree pressure gradient. These generalise across wells where raw absolute
pressures do not.
"""

from __future__ import annotations

import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame


def differentials(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    out = pd.DataFrame(index=df.index)

    # A choke (PCK = production choke, GLCK = gas-lift choke) is a restriction valve;
    # MON = upstream/montante, JUS = downstream/jusante. The ΔP across it depends on the
    # flow regime, not the well's absolute reservoir pressure, so it transfers between
    # wells where raw pressures (well-specific operating points) would not (AGENT.md §2).
    out["dP_CKP"] = df["P-MON-CKP"] - df["P-JUS-CKP"]
    out["dP_CKGL"] = df["P-MON-CKGL"] - df["P-JUS-CKGL"]
    # Joule-Thomson cooling across the production choke; a temperature drop signature.
    out["dT_CKP"] = df["T-MON-CKP"] - df["T-JUS-CKP"]
    # Downhole(PDG = permanent downhole gauge) → tree(TPT) gradient along the tubing:
    # the hydrostatic + friction pressure/temperature signature of the producing column.
    out["dP_grad"] = df["P-PDG"] - df["P-TPT"]
    out["dT_grad"] = df["T-PDG"] - df["T-TPT"]
    # Tree(TPT) → flowline(PT-P) step across the production wing valve.
    out["dP_tree"] = df["P-TPT"] - df["PT-P"]

    win = max(2, int(round(cfg.rolling_diff_window_s)))
    # Rate-of-change of the gradients — a developing fault perturbs the *slope* of the
    # flow-path gradient before it shifts the absolute level, so onset shows here first.
    out["dP_grad_roc"] = out["dP_grad"].diff().rolling(win, min_periods=1).mean()
    out["dP_CKP_roc"] = out["dP_CKP"].diff().rolling(win, min_periods=1).mean()
    return out.fillna(0.0)
