"""Choke flow coefficients (AGENT.md §5.2).

``Cv_eff = Q / (opening · sqrt(max(ΔP, ε)))`` — an effective valve coefficient that
should be roughly stationary under normal operation; restrictions/scaling shift it.
The division is guarded: the opening (%) and ΔP are both floored at ``ε`` so a closed
choke or zero pressure drop cannot produce inf/NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame


def choke_coeff(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    # eps floors every denominator: a closed choke (opening→0) or zero ΔP would
    # otherwise make Cv blow up to inf/NaN — these are the guarded divisions of §5.2.
    eps = cfg.differential_eps
    out = pd.DataFrame(index=df.index)

    # Orifice flow gives Q ≈ Cv · opening · sqrt(ΔP), so inverting yields an effective
    # valve coefficient Cv ≈ Q / (opening · sqrt(ΔP)) that is ~stationary when the valve
    # is healthy; scaling/erosion/restriction make it drift, which is the signal we want.
    # Gas-lift choke (GLCK) has a real flow meter QGL, so we can form a true Cv here.
    open_gl = (df["ABER-CKGL"] / 100.0).clip(lower=eps)  # ABER = abertura/opening, %→fraction
    dp_gl = (df["P-MON-CKGL"] - df["P-JUS-CKGL"]).clip(lower=eps)
    out["Cv_CKGL"] = df["QGL"] / (open_gl * np.sqrt(dp_gl))

    # Production choke (PCK) has NO direct flow meter, so a true Cv is impossible.
    # Use a conductance proxy opening / sqrt(ΔP): for fixed flow it tracks how "open"
    # the path effectively is; a downward drift flags a restriction (AGENT.md class 6)
    # without depending on the well's absolute pressure level.
    open_p = (df["ABER-CKP"] / 100.0).clip(lower=eps)
    dp_p = (df["P-MON-CKP"] - df["P-JUS-CKP"]).clip(lower=eps)
    out["cond_CKP"] = open_p / np.sqrt(dp_p)

    # Service line: QBS (service-pump flow) against its downstream pressure P-JUS-BS;
    # no separate opening signal, so this is a pressure-normalised flow coefficient.
    out["Cv_BS"] = df["QBS"] / np.sqrt((df["P-JUS-BS"]).clip(lower=eps))

    # Belt-and-braces: any residual inf/NaN (e.g. missing sensor) collapses to 0.
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
