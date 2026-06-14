"""Rolling spectral features (AGENT.md §5.2).

Dominant frequency, spectral entropy, and band-power for the configured channels
(default ``P-PDG, P-TPT, PT-P, QGL``). A true per-sample rolling FFT is wasteful, so
we compute the FFT on a hop grid (¼ window) and forward-fill between hops — the
window/band parameters all come from ``features.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FeaturesConfig
from ingest.instance import InstanceFrame


def _window_spectrum(seg: np.ndarray, fs: float) -> tuple[float, float, np.ndarray]:
    """Return (dominant_freq, spectral_entropy, power_spectrum) for one segment."""
    # Remove the mean so the DC (zero-frequency) component does not dominate the spectrum.
    seg = seg - seg.mean()
    n = len(seg)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(seg)) ** 2
    total = power.sum()
    if total <= 0:  # flat/silent segment: no spectral content to report
        return 0.0, 0.0, power
    # Dominant frequency = the strongest oscillation; skip bin 0 (residual DC offset).
    dom = float(freqs[1 + int(np.argmax(power[1:]))]) if len(power) > 1 else 0.0
    # Spectral entropy of the power distribution: low ⇒ one sharp tone (e.g. a clean
    # slug cycle), high ⇒ broadband/noisy. Normalised to [0,1] by log(#bins).
    p = power / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(power)))  # normalised [0, 1]
    return dom, entropy, power


def spectral(frame: InstanceFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    df = frame.df
    fs = 1.0  # resample grid is 1 sample / resample_rate_s; bands are in those Hz units
    win = max(8, int(round(cfg.spectral_window_s)))
    # Compute the FFT only every `hop` samples (¼ window) and forward-fill in between:
    # a true per-sample rolling FFT would recompute almost-identical spectra O(n·win·log win)
    # times for negligible extra information, so the hop grid trades that cost for speed.
    hop = max(1, win // 4)
    bands = cfg.spectral_bands
    out = pd.DataFrame(index=df.index)

    for chan in cfg.spectral_channels:
        if chan not in df.columns:
            continue
        x = df[chan].to_numpy(dtype=float)
        n = len(x)
        dom = np.zeros(n)
        ent = np.zeros(n)
        bandp = np.zeros((n, len(bands)))
        freqs = np.fft.rfftfreq(win, d=1.0 / fs)

        last = (0.0, 0.0, np.zeros(len(bands)))
        for end in range(win, n + 1, hop):
            seg = x[end - win : end]  # trailing window ending at `end`
            d, e, power = _window_spectrum(seg, fs)
            total = power.sum() or 1.0
            # Band-power = fraction of energy in each configured [lo, hi) frequency band;
            # e.g. a slow-slugging band lighting up flags a low-frequency oscillation.
            bp = np.array([power[(freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in bands])
            last = (d, e, bp)
            # Forward-fill this hop's result across the samples until the next hop.
            nxt = min(end + hop, n)
            dom[end - 1 : nxt] = d
            ent[end - 1 : nxt] = e
            bandp[end - 1 : nxt] = bp
        # Leading edge (before first full window) holds the first computed value.
        dom[:win] = last[0] if n < win else dom[win - 1]
        out[f"{chan}__dom_freq"] = dom
        out[f"{chan}__spec_entropy"] = ent
        for i, (lo, hi) in enumerate(bands):
            out[f"{chan}__band_{lo}_{hi}"] = bandp[:, i]

    return out.fillna(0.0)
